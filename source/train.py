import os, copy, sys, uuid, warnings, json, re
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import sys
import random

from tqdm import tqdm
from fused_ssim import fused_ssim
from rasterizer import imlsplatting
from scene import Scene, GaussianModel
from utils.loss_utils import (
    l1_loss,
    normal_loss,
    pearson_depth_loss,
    pearson_disparity_loss,
)
from utils.depth_interactor import OnlineDepthEstimator
from utils.surface_regularization import (
    implicit_residual_regularization,
    imls_bidirectional_consistency_loss,
    surface_tangent_loss,
)
from utils.general_utils import safe_state
from utils.refnerf_alpha import composite_reference
from utils.mesh_domain import aligned_mesh_domain
from utils.visible_contour import visible_contour_loss
from utils.opaque_boundary import boundary_terms, prepare_target
from utils.export_views import ordered_export_cameras, write_view_manifest
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import inverse_sigmoid
# from refer import rasterize_set, eval_cd_dtu, eval_psnr_nerf, eval_cd_nerf
from PIL import Image
import numpy as np
from lpipsPyTorch.modules.lpips import LPIPS


torch.set_num_threads(10)
warnings.filterwarnings("ignore")

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

torch.set_num_threads(10)
warnings.filterwarnings("ignore")

def prepare_output_and_logger(args, expname):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
            unique_str = unique_str[0:10]
        else:
            unique_str = str(uuid.uuid4())
            unique_str = unique_str[0:10]
        if expname != '':
            unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
        
    tb_writer = None
    if SummaryWriter:
        tb_writer = SummaryWriter(args.model_path)
        print("Tensorboard Logdir: {}".format(args.model_path))
    return tb_writer


def _map_to_uint8(value, valid_mask=None):
    value = value.detach().float().cpu().squeeze().numpy()
    if valid_mask is not None:
        mask = valid_mask.detach().float().cpu().squeeze().numpy() > 0.5
    else:
        mask = np.isfinite(value)
    mask = mask & np.isfinite(value)
    if not np.any(mask):
        return None
    low, high = np.percentile(value[mask], [1.0, 99.0])
    normalized = np.clip((value - low) / (high - low + 1e-8), 0.0, 1.0)
    normalized[~mask] = 0.0
    return (normalized * 255.0).round().astype(np.uint8)


def _save_depth_comparison(path, predicted_depth, reference_depth, valid_mask, reference_mask=None):
    # Depth Anything is an inverse-depth estimator (nearer surfaces are
    # brighter), whereas the renderer stores metric ray distance.  Convert
    # the latter to inverse depth before composing a visual comparison.
    predicted_depth = predicted_depth.detach()
    positive = torch.isfinite(predicted_depth) & (predicted_depth > 1e-6)
    predicted_mask = positive.to(predicted_depth.dtype)
    if valid_mask is not None:
        predicted_mask = predicted_mask * (valid_mask > 0.5).to(predicted_depth.dtype)
    predicted_inverse = torch.where(
        positive,
        torch.reciprocal(predicted_depth.clamp_min(1e-6)),
        torch.zeros_like(predicted_depth),
    )
    predicted = _map_to_uint8(predicted_inverse, predicted_mask)
    reference = _map_to_uint8(reference_depth, reference_mask if reference_mask is not None else valid_mask)
    if predicted is None or reference is None:
        return
    comparison = np.concatenate((predicted, reference), axis=1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(comparison, mode="L").save(path)


def _save_normal_comparison(
    path, predicted_normal, gt_normal, predicted_alpha=None, gt_alpha=None
):
    predicted = predicted_normal.detach().float().cpu().clamp(0, 1)
    gt = gt_normal.detach().float().cpu().permute(1, 2, 0).clamp(0, 1)
    if predicted_alpha is not None:
        predicted = predicted * (predicted_alpha.detach().float().cpu() > 0.5)
    if gt_alpha is not None:
        gt_mask = gt_alpha.detach().float().cpu().permute(1, 2, 0) > 0.5
        gt = gt * gt_mask
    predicted = predicted.numpy()
    gt = gt.numpy()
    if predicted.shape != gt.shape:
        return
    comparison = np.concatenate((predicted, gt), axis=1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(
        (comparison * 255.0).round().astype(np.uint8),
        mode="RGB",
    ).save(path)


def _save_geometry_raw(path, meshdict, view):
    """Save geometry without PNG quantization or alpha premultiplication."""
    payload = {
        "image_name": np.asarray(view.image_name),
        "source_id": np.asarray(view.colmap_id),
        "pred_normal_world": meshdict["normal_world"].detach().float().cpu().numpy(),
        "pred_alpha": meshdict["alpha"].detach().float().cpu().numpy(),
        "pred_camera_z": meshdict["camera_z_depth"].detach().float().cpu().numpy(),
    }
    if getattr(view, "original_normal", None) is not None:
        payload["gt_normal_encoded"] = (
            view.original_normal.detach().float().cpu().permute(1, 2, 0).numpy()
        )
    if view.gt_alpha_mask is not None:
        payload["gt_alpha"] = (
            view.gt_alpha_mask.detach().float().cpu().permute(1, 2, 0).numpy()
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **payload)

def training(dataset, opt, saving_iterations, checkpoint_ply, expname, tb_writer=None):
    first_iter       = 0
    prepare_output_and_logger(dataset, expname)
    gaussians = GaussianModel(opt)
    scene     = Scene(dataset, gaussians, checkpoint_ply=checkpoint_ply)
    if opt.auto_mesh_domain:
        # Fit the fixed extraction box to the checkpoint, preserving world-space
        # voxel spacing. Never optimize this bound from image gradients.
        with torch.no_grad():
            old_scale = float(opt.meshscale)
            spacing = old_scale / (opt.grid_resolution - 1)
            radius = (gaussians.get_xyz - gaussians.axis_center).abs().max().item()
            margin = max(3.0 * gaussians.get_scaling.max().item(), 2.0 * spacing)
            required_scale = max(old_scale, 2.0 * (radius + margin))
            # Preserve centered-grid parity: otherwise every old IMLS sample
            # shifts by half a voxel when the extraction cube is enlarged.
            new_resolution, opt.meshscale = aligned_mesh_domain(
                opt.grid_resolution, old_scale, required_scale
            )
            opt.grid_resolution = new_resolution
            print(f"[Domain] {old_scale:.6f} -> {opt.meshscale:.6f}; "
                  f"grid={new_resolution}; voxel={spacing:.7f}; margin={margin:.6f}")
    gaussians.training_setup(opt)
    reference_xyz = gaussians.get_xyz.detach().clone()
    reference_normal = gaussians.get_normal.detach().clone()

    perceptual_metric = None
    if opt.lambda_lpips > 0:
        perceptual_metric = LPIPS('vgg', '0.1').cuda().eval()
        for parameter in perceptual_metric.parameters():
            parameter.requires_grad_(False)
        print(
            f"[LPIPS] Enabled (lambda={opt.lambda_lpips}, "
            f"interval={opt.lpips_interval}, resolution={opt.lpips_resolution})."
        )
    
    # ================== 【新增：在线生成深度图】 ==================
    # 逻辑：在训练循环开始前，跑一遍 Depth Anything，把深度图挂载到相机对象上
    # 仅当我们需要深度监督时运行
    
    print(
        "\n[Depth] Starting online pseudo-GT generation..."
        if opt.lambda_depth > 0
        else "\n[Depth] Disabled (lambda_depth=0)."
    )
    try:
        if hasattr(sys.stdout, 'isatty') is False:
             sys.stdout.isatty = lambda: False
                
        # 1. 初始化估计器
        depth_estimator = (
            OnlineDepthEstimator(checkpoint=opt.depth_model_path)
            if opt.lambda_depth > 0
            else None
        )
        
        # 2. 获取所有训练相机
        train_cams = scene.getTrainCameras() if depth_estimator is not None else []
        
        # 3. 批量推理
        # 使用 tqdm 显示进度
        for cam in tqdm(train_cams, desc="Generating Depth Maps"):
        
#            pseudo_depth = depth_estimator.infer_depth(cam.original_image)
#            cam.original_depth = pseudo_depth.detach()
            
            pseudo_depth = depth_estimator.infer_depth(cam.original_image)
            cam.original_depth = pseudo_depth.detach().to(
                device="cpu",
                dtype=torch.float16
            )
            
        # 4. 立即释放显存！非常重要！
        if depth_estimator is not None:
            depth_estimator.clear_memory()
        
    except Exception as e:
        print(f"[Warning] Online depth generation failed: {e}")
        print("Training will continue without depth supervision.")
    # ============================================================

    opt.model_path = scene.model_path
    with open(scene.model_path + '/opt.json', 'w') as json_file:
        json.dump(opt.__dict__, json_file, indent=4)
    imlsplat = imlsplatting(opt.__dict__)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end   = torch.cuda.Event(enable_timing = True)
    bg         = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    # first_iter       = 0
    viewpoint_stack  = None
    boundary_targets = {}
    holdout_stride = getattr(opt, 'opaque_boundary_holdout_stride', 0)
    boundary_region_weight = getattr(opt, 'lambda_opaque_boundary_region', 0.)
    boundary_edge_weight = getattr(opt, 'lambda_opaque_boundary_edges', 0.)
    boundary_displacement = getattr(opt, 'opaque_boundary_max_displacement_voxels', 0.)
    if boundary_displacement > 0 and (opt.resample_until > 0 or opt.upsample_until > 0):
        raise ValueError('Boundary displacement control requires a fixed point set and grid')
    if holdout_stride == 1 or holdout_stride < 0:
        raise ValueError('Boundary holdout stride must be zero or at least two')
    if holdout_stride:
        print(f'[OpaqueBoundary] Holding out training source IDs divisible by {holdout_stride}', flush=True)
    ema_loss_for_log = 0.0
    progress_bar     = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter      += 1

    with torch.no_grad():
        meshdict = imlsplat(gaussians, scene.getTrainCameras()[0], bg, record='record/'+str(0), cameras=scene.getTrainCameras(), step=0)
    if getattr(opt, "support_diagnostics_only", False):
        print("[IMLS support] Diagnostic extraction complete; skipping training and final rendering.")
        return

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            if holdout_stride:
                viewpoint_stack = [v for v in viewpoint_stack if v.colmap_id % holdout_stride != 0]
                if not viewpoint_stack:
                    raise ValueError('Boundary holdout excluded all training cameras')
        viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1))

        train_bgcolor = [1, 1, 1] if random.randint(0, 1)==1 else [0, 0, 0]
        train_bg      = torch.tensor(train_bgcolor, dtype=torch.float32, device="cuda")

        meshdict      = imlsplat(gaussians, viewpoint_cam, train_bg, step=iteration)

        gt_image = composite_reference(
            viewpoint_cam.original_image.cuda(), viewpoint_cam.gt_alpha_mask,
            train_bg, viewpoint_cam.image_background,
        ).permute(1, 2, 0)

        loss_L1   = l1_loss(meshdict['image'][..., 0:3].permute(2,0,1), gt_image.permute(2,0,1))
        loss_ssim = fused_ssim(meshdict['image'][..., 0:3].permute(2,0,1).unsqueeze(0), gt_image.permute(2,0,1).unsqueeze(0))
        loss      = (1.0 - opt.lambda_image) * loss_L1 + opt.lambda_image * (1.0 - loss_ssim)
        if opt.geometry_only:
            # Geometry is fitted from training normals and silhouettes, without
            # confusing reflected image edges with surface discontinuities.
            loss = loss * 0.0

        if opt.lambda_visible_contour > 0 and viewpoint_cam.gt_alpha_mask is not None:
            contour_loss, contour_pred_count, contour_gt_count = visible_contour_loss(
                meshdict['verts_clip'], meshdict['mesh_faces'], meshdict['rast'],
                viewpoint_cam.gt_alpha_mask.to(device='cuda'),
                max_samples=opt.visible_contour_samples,
            )
            loss = loss + opt.lambda_visible_contour * contour_loss
            if iteration == 1 or iteration % 50 == 0:
                print(f'[Visible contour] iter={iteration} loss={float(contour_loss.detach()):.6f} '
                      f'pred={contour_pred_count} target={contour_gt_count}')

        loss_lpips = torch.zeros((), device="cuda")
        if (
            perceptual_metric is not None
            and opt.lpips_interval > 0
            and iteration % opt.lpips_interval == 0
        ):
            predicted_for_lpips = meshdict['image'][..., 0:3].permute(2, 0, 1).unsqueeze(0)
            target_for_lpips = gt_image.permute(2, 0, 1).unsqueeze(0)
            target_size = (opt.lpips_resolution, opt.lpips_resolution)
            predicted_for_lpips = torch.nn.functional.interpolate(
                predicted_for_lpips,
                size=target_size,
                mode='bilinear',
                align_corners=False,
            )
            target_for_lpips = torch.nn.functional.interpolate(
                target_for_lpips,
                size=target_size,
                mode='bilinear',
                align_corners=False,
            )
            if opt.lpips_standard_input:
                predicted_for_lpips = predicted_for_lpips * 2.0 - 1.0
                target_for_lpips = target_for_lpips * 2.0 - 1.0
            loss_lpips = perceptual_metric(
                predicted_for_lpips,
                target_for_lpips,
            ).mean()
            loss = loss + opt.lambda_lpips * loss_lpips

        if opt.lambda_reflection_tv > 0:
            reflection_texture = gaussians.rendernet.reflection_texture
            longitude_tv = (
                reflection_texture - torch.roll(reflection_texture, 1, dims=-1)
            ).abs().mean()
            latitude_tv = (
                reflection_texture[..., 1:, :]
                - reflection_texture[..., :-1, :]
            ).abs().mean()
            reflection_tv = longitude_tv + latitude_tv
            if hasattr(gaussians.rendernet, "reflection_detail_texture"):
                detail_texture = gaussians.rendernet.reflection_detail_texture
                detail_longitude_tv = (
                    detail_texture - torch.roll(detail_texture, 1, dims=-1)
                ).abs().mean()
                detail_latitude_tv = (
                    detail_texture[..., 1:, :]
                    - detail_texture[..., :-1, :]
                ).abs().mean()
                reflection_tv = reflection_tv + 0.5 * (
                    detail_longitude_tv + detail_latitude_tv
                )
            if (
                getattr(opt, "factorized_reflection", False)
                and hasattr(gaussians.rendernet, "reflection_factor_texture")
            ):
                factor_texture = gaussians.rendernet.reflection_factor_texture
                factor_longitude_tv = (
                    factor_texture
                    - torch.roll(factor_texture, 1, dims=-1)
                ).abs().mean()
                factor_latitude_tv = (
                    factor_texture[..., 1:, :]
                    - factor_texture[..., :-1, :]
                ).abs().mean()
                reflection_tv = reflection_tv + 0.25 * (
                    factor_longitude_tv + factor_latitude_tv
                )
            loss = loss + opt.lambda_reflection_tv * reflection_tv

        # V14 keeps the reflection-only micro-normal close to the actual
        # reconstructed normal. It can correct small specular misregistration,
        # but cannot replace the geometry normal or affect exported normal GT
        # comparisons.
        if (
            opt.lambda_reflection_normal > 0
            and "reflection_normal_offset_map" in meshdict
        ):
            reflection_normal_offset = meshdict["reflection_normal_offset_map"]
            reflection_normal_reg = reflection_normal_offset.square().sum(-1)
            if "alpha" in meshdict:
                reflection_normal_reg = (
                    reflection_normal_reg * meshdict["alpha"].squeeze(-1)
                ).sum() / meshdict["alpha"].sum().clamp_min(1.0)
            else:
                reflection_normal_reg = reflection_normal_reg.mean()
            loss = loss + opt.lambda_reflection_normal * reflection_normal_reg

        reflection_normal_tv_loss = torch.zeros((), device="cuda")
        if (
            opt.lambda_reflection_normal_tv > 0
            and "reflection_normal_offset_map" in meshdict
            and viewpoint_cam.original_normal is not None
        ):
            offset_map = meshdict["reflection_normal_offset_map"]
            gt_normal_tv = torch.nn.functional.normalize(
                viewpoint_cam.original_normal.to(
                    offset_map.device, offset_map.dtype
                ).permute(1, 2, 0) * 2.0 - 1.0,
                dim=-1,
                eps=1e-6,
            )
            valid_tv = meshdict["alpha"][..., 0].detach() > 0.5
            if viewpoint_cam.gt_alpha_mask is not None:
                gt_alpha_tv = viewpoint_cam.gt_alpha_mask.to(
                    offset_map.device, offset_map.dtype
                )[None]
                erode = max(1, int(opt.reflection_normal_tv_erode))
                if erode % 2 == 0:
                    erode += 1
                interior_tv = -torch.nn.functional.max_pool2d(
                    -gt_alpha_tv,
                    kernel_size=erode,
                    stride=1,
                    padding=erode // 2,
                )[0, 0] > 0.99
                valid_tv = valid_tv & interior_tv

            tv_total = offset_map.new_zeros(())
            tv_terms = 0
            for dim in (0, 1):
                for offset in (1, 4):
                    length = offset_map.shape[dim] - offset
                    if length <= 0:
                        continue
                    offset_a = offset_map.narrow(dim, offset, length)
                    offset_b = offset_map.narrow(dim, 0, length)
                    normal_a = gt_normal_tv.narrow(dim, offset, length)
                    normal_b = gt_normal_tv.narrow(dim, 0, length)
                    pair_valid = (
                        valid_tv.narrow(dim, offset, length)
                        & valid_tv.narrow(dim, 0, length)
                        & (
                            (normal_a * normal_b).sum(-1)
                            > float(opt.reflection_normal_tv_cosine)
                        )
                    )
                    if bool(pair_valid.any()):
                        tv_total = tv_total + torch.nn.functional.smooth_l1_loss(
                            (offset_a - offset_b)[pair_valid],
                            torch.zeros_like((offset_a - offset_b)[pair_valid]),
                            beta=0.01,
                        )
                        tv_terms += 1
            if tv_terms:
                reflection_normal_tv_loss = tv_total / tv_terms
                loss = loss + (
                    opt.lambda_reflection_normal_tv
                    * reflection_normal_tv_loss
                )

        loss_mask = torch.zeros((), device="cuda")
        if viewpoint_cam.gt_alpha_mask is not None and "alpha" in meshdict:
            predicted_alpha = meshdict["alpha"].permute(2, 0, 1)
            target_alpha = viewpoint_cam.gt_alpha_mask.to(
                device=predicted_alpha.device,
                dtype=predicted_alpha.dtype,
            )
            loss_mask = l1_loss(predicted_alpha, target_alpha)
            loss = loss + opt.lambda_mask * loss_mask
            if opt.lambda_silhouette_iou > 0:
                intersection = (predicted_alpha * target_alpha).sum()
                union = (
                    predicted_alpha.sum() + target_alpha.sum() - intersection
                )
                silhouette_iou_loss = 1.0 - (
                    intersection + 1.0
                ) / (union + 1.0)
                loss = loss + opt.lambda_silhouette_iou * silhouette_iou_loss
            if opt.lambda_silhouette_recall > 0:
                # Opaque reflective objects must remain closed silhouettes.
                # This asymmetric term targets missing foreground instead of
                # letting the much larger correct background dominate L1/IoU.
                missing_foreground_loss = (
                    target_alpha * (1.0 - predicted_alpha)
                ).sum() / target_alpha.sum().clamp_min(1.0)
                loss = loss + (
                    opt.lambda_silhouette_recall * missing_foreground_loss
                )
            if opt.lambda_silhouette_boundary > 0:
                # Supervise a narrow GT contour band. Reflection-image edges
                # are deliberately not used because they are appearance, not
                # evidence of a geometric discontinuity.
                target_4d = target_alpha[None]
                dilated = torch.nn.functional.max_pool2d(
                    target_4d, kernel_size=7, stride=1, padding=3
                )
                eroded = -torch.nn.functional.max_pool2d(
                    -target_4d, kernel_size=7, stride=1, padding=3
                )
                boundary = (dilated - eroded).clamp(0.0, 1.0)[0]
                boundary_loss = (
                    (predicted_alpha - target_alpha).abs() * boundary
                ).sum() / boundary.sum().clamp_min(1.0)
                loss = loss + (
                    opt.lambda_silhouette_boundary * boundary_loss
                )
            if opt.lambda_silhouette_pyramid > 0:
                pyramid = predicted_alpha.new_zeros(())
                for factor in (1, 4, 16):
                    pa = torch.nn.functional.avg_pool2d(predicted_alpha[None], factor)
                    ta = torch.nn.functional.avg_pool2d(target_alpha[None], factor)
                    pyramid = pyramid + (pa - ta).abs().mean() / ta.mean().clamp_min(0.05)
                loss = loss + opt.lambda_silhouette_pyramid * pyramid / 3

        if opt.lambda_surface_smooth > 0:
            # Local regularity in normal space; no RGB edge weighting, since
            # mirror-image contours need not be geometric edges.
            n = torch.nn.functional.normalize(2 * meshdict['image_normals'] - 1, dim=-1)
            valid = meshdict['alpha'][..., 0].detach() > 0.99
            surface_loss = n.new_zeros(())
            for dim in (0, 1):
                a, b = n.narrow(dim, 1, n.shape[dim]-1), n.narrow(dim, 0, n.shape[dim]-1)
                va = valid.narrow(dim, 1, valid.shape[dim]-1)
                vb = valid.narrow(dim, 0, valid.shape[dim]-1)
                cos = (a*b).sum(-1).clamp(-1, 1)
                w = (va & vb & (cos.detach() > 0.9)).to(cos.dtype)
                surface_loss = surface_loss + ((1-cos)*w).sum()/w.sum().clamp_min(1)
            loss = loss + opt.lambda_surface_smooth * surface_loss

        imls_surface_loss = torch.zeros((), device="cuda")
        imls_sign_loss = torch.zeros((), device="cuda")
        imls_tangent_loss = torch.zeros((), device="cuda")
        imls_valid_fraction = torch.zeros((), device="cuda")
        consistency_enabled = (
            opt.lambda_imls_surface > 0
            or opt.lambda_imls_sign > 0
            or opt.lambda_imls_tangent > 0
        )
        consistency_interval = max(1, int(opt.imls_consistency_interval))
        if consistency_enabled and iteration % consistency_interval == 0:
            (
                imls_surface_loss,
                imls_sign_loss,
                imls_tangent_loss,
                imls_valid_fraction,
            ) = imls_bidirectional_consistency_loss(
                meshdict["surface_sdfs"],
                meshdict["out_sums"],
                meshdict["means3D"],
                meshdict["normals"],
                offset_voxels=opt.imls_consistency_offset_voxels,
                support_floor=opt.imls_consistency_support_floor,
                max_samples=opt.imls_consistency_samples,
            )
            # Correct for interval sampling so each configured coefficient is
            # an expected per-iteration weight rather than silently shrinking.
            interval_scale = float(consistency_interval)
            loss = loss + interval_scale * (
                opt.lambda_imls_surface * imls_surface_loss
                + opt.lambda_imls_sign * imls_sign_loss
                + opt.lambda_imls_tangent * imls_tangent_loss
            )

        implicit_magnitude_loss = torch.zeros((), device="cuda")
        implicit_tv_loss = torch.zeros((), device="cuda")
        if (
            gaussians.implicit_sdf_grid is not None
            and (
                opt.lambda_implicit_sdf_magnitude > 0
                or opt.lambda_implicit_sdf_tv > 0
            )
        ):
            implicit_magnitude_loss, implicit_tv_loss = (
                implicit_residual_regularization(gaussians.implicit_sdf_grid)
            )
            loss = loss + (
                opt.lambda_implicit_sdf_magnitude * implicit_magnitude_loss
                + opt.lambda_implicit_sdf_tv * implicit_tv_loss
            )
        
        # ================== 【新增：法线损失】 ==================
        # 初始化用于显示的变量
        
        loss_normal = torch.tensor(0.0, device="cuda")

        # 1. 获取 GT 法线 (我们在 Camera 类中处理过，通常是 [3, H, W])
        gt_normal = viewpoint_cam.original_normal
        if gt_normal is not None:
            gt_normal = gt_normal.to(meshdict['image_normals'])
        if gt_normal is not None:
            gt_normal = gt_normal.to(meshdict['image_normals'])

        if opt.lambda_surface_tangent > 0 and gt_normal is not None:
            valid_surface = meshdict['alpha'][..., 0].detach() > 0.99
            if viewpoint_cam.gt_alpha_mask is not None:
                valid_surface = valid_surface & (viewpoint_cam.gt_alpha_mask.squeeze(0).to(valid_surface.device) > 0.99)
            loss = loss + opt.lambda_surface_tangent * surface_tangent_loss(
                meshdict['surface_position'], gt_normal, valid_surface, opt.meshscale
            )

        if gt_normal is not None:
            # 2. 获取预测法线
            # meshdict['image_normals'] 形状通常是 [H, W, 3]
            pred_normal = meshdict['image_normals']
            
            # 3. [关键] 维度对齐
            # 必须把预测值从 [H, W, 3] 变为 [3, H, W] 才能和 gt_normal 对比
            pred_normal_permuted = pred_normal.permute(2, 0, 1)

            # 4. 计算 Loss (调用 loss_utils 中的函数)
            normal_mask = viewpoint_cam.gt_alpha_mask
            if normal_mask is not None:
                normal_mask = normal_mask.to(
                    device=pred_normal_permuted.device,
                    dtype=pred_normal_permuted.dtype,
                ) * (
                    meshdict['alpha'][..., 0].detach() > 0.99
                ).to(pred_normal_permuted.dtype).unsqueeze(0)
            loss_normal = normal_loss(
                pred_normal_permuted,
                gt_normal,
                normal_mask,
            )
            
            if torch.isnan(loss_normal):
                print(f"[WARNING] Normal loss is NaN at iter {iteration}!")
                loss_normal = torch.tensor(0.0, device="cuda")

            # 5. 加权并添加到总 Loss
            # Ref-NeRF 推荐权重，可以根据效果调整
            lambda_normal = opt.lambda_normal

            loss += lambda_normal * loss_normal

            if opt.lambda_normal_cvar > 0:
                # Mirror-image boundaries can create a small number of severe
                # geometry/normal outliers.  Select them only where GT normals
                # and the opaque silhouette say that the true surface is
                # locally continuous; real rims and object boundaries are
                # excluded before the top-tail (CVaR) reduction.
                pred_vec_cvar = torch.nn.functional.normalize(
                    2.0 * pred_normal - 1.0, dim=-1, eps=1e-6
                )
                gt_vec_cvar = torch.nn.functional.normalize(
                    2.0 * gt_normal.permute(1, 2, 0) - 1.0,
                    dim=-1,
                    eps=1e-6,
                )
                if viewpoint_cam.gt_alpha_mask is not None:
                    gt_alpha_cvar = viewpoint_cam.gt_alpha_mask.to(
                        pred_vec_cvar.device, pred_vec_cvar.dtype
                    )[None]
                else:
                    gt_alpha_cvar = torch.ones_like(
                        pred_vec_cvar[..., :1].permute(2, 0, 1)[None]
                    )
                erode_kernel = max(1, int(opt.normal_cvar_erode))
                if erode_kernel % 2 == 0:
                    erode_kernel += 1
                interior = -torch.nn.functional.max_pool2d(
                    -gt_alpha_cvar,
                    kernel_size=erode_kernel,
                    stride=1,
                    padding=erode_kernel // 2,
                )[0, 0] > 0.99
                gt_chw = gt_vec_cvar.permute(2, 0, 1)[None]
                gt_local = torch.nn.functional.avg_pool2d(
                    gt_chw, kernel_size=3, stride=1, padding=1
                )[0].permute(1, 2, 0)
                gt_agreement = (
                    gt_vec_cvar * torch.nn.functional.normalize(
                        gt_local, dim=-1, eps=1e-6
                    )
                ).sum(-1)
                valid_cvar = (
                    interior
                    & (meshdict['alpha'][..., 0].detach() > 0.5)
                    & (gt_agreement > float(opt.normal_cvar_cosine))
                )
                cvar_errors = (
                    1.0 - (pred_vec_cvar * gt_vec_cvar).sum(-1).clamp(-1.0, 1.0)
                )[valid_cvar]
                if cvar_errors.numel() > 0:
                    fraction = min(1.0, max(1e-4, float(opt.normal_cvar_fraction)))
                    tail_count = max(1, int(cvar_errors.numel() * fraction))
                    normal_cvar = torch.topk(
                        cvar_errors, tail_count, sorted=False
                    ).values.mean()
                    loss = loss + opt.lambda_normal_cvar * normal_cvar

            if opt.lambda_gt_normal_smooth > 0:
                pred_vec = torch.nn.functional.normalize(
                    2.0 * pred_normal - 1.0, dim=-1, eps=1e-6
                )
                gt_vec = torch.nn.functional.normalize(
                    2.0 * gt_normal.permute(1, 2, 0) - 1.0,
                    dim=-1,
                    eps=1e-6,
                )
                valid = meshdict['alpha'][..., 0].detach() > 0.5
                if viewpoint_cam.gt_alpha_mask is not None:
                    valid = valid & (
                        viewpoint_cam.gt_alpha_mask.squeeze(0).to(valid.device) > 0.5
                    )
                smooth_loss = pred_vec.new_zeros(())
                smooth_terms = 0
                for dim in (0, 1):
                    for offset in (1, 4):
                        length = pred_vec.shape[dim] - offset
                        pred_a = pred_vec.narrow(dim, offset, length)
                        pred_b = pred_vec.narrow(dim, 0, length)
                        gt_a = gt_vec.narrow(dim, offset, length)
                        gt_b = gt_vec.narrow(dim, 0, length)
                        valid_a = valid.narrow(dim, offset, length)
                        valid_b = valid.narrow(dim, 0, length)
                        gt_agreement = (gt_a * gt_b).sum(-1)
                        pair_valid = valid_a & valid_b & (gt_agreement > 0.995)
                        if pair_valid.any():
                            pred_agreement = (pred_a * pred_b).sum(-1).clamp(-1, 1)
                            smooth_loss = smooth_loss + (
                                1.0 - pred_agreement[pair_valid]
                            ).mean()
                            smooth_terms += 1
                if smooth_terms:
                    loss = loss + opt.lambda_gt_normal_smooth * (
                        smooth_loss / smooth_terms
                    )
            if opt.lambda_gt_normal_gradient > 0:
                pred_vec = torch.nn.functional.normalize(
                    2.0 * pred_normal - 1.0, dim=-1, eps=1e-6
                )
                gt_vec = torch.nn.functional.normalize(
                    2.0 * gt_normal.permute(1, 2, 0) - 1.0,
                    dim=-1,
                    eps=1e-6,
                )
                valid = meshdict['alpha'][..., 0].detach() > 0.5
                if viewpoint_cam.gt_alpha_mask is not None:
                    valid = valid & (
                        viewpoint_cam.gt_alpha_mask.squeeze(0).to(valid.device) > 0.5
                    )
                gradient_loss = pred_vec.new_zeros(())
                gradient_terms = 0
                for dim in (0, 1):
                    for offset in (1, 4):
                        length = pred_vec.shape[dim] - offset
                        pred_delta = (
                            pred_vec.narrow(dim, offset, length)
                            - pred_vec.narrow(dim, 0, length)
                        )
                        gt_delta = (
                            gt_vec.narrow(dim, offset, length)
                            - gt_vec.narrow(dim, 0, length)
                        )
                        pair_valid = (
                            valid.narrow(dim, offset, length)
                            & valid.narrow(dim, 0, length)
                        )
                        if pair_valid.any():
                            gradient_loss = gradient_loss + torch.nn.functional.smooth_l1_loss(
                                pred_delta[pair_valid],
                                gt_delta[pair_valid],
                                beta=0.05,
                            )
                            gradient_terms += 1
                if gradient_terms:
                    loss = loss + opt.lambda_gt_normal_gradient * (
                        gradient_loss / gradient_terms
                    )
        # =======================================================

        if opt.lambda_point_anchor > 0 or opt.lambda_point_tangent > 0:
            displacement = (gaussians.get_xyz - reference_xyz) / opt.meshscale
            if opt.lambda_point_anchor > 0:
                loss = loss + opt.lambda_point_anchor * torch.nn.functional.smooth_l1_loss(
                    displacement,
                    torch.zeros_like(displacement),
                    beta=0.002,
                )
            if opt.lambda_point_tangent > 0:
                normal_offset = (
                    displacement * reference_normal
                ).sum(-1, keepdim=True) * reference_normal
                tangent_offset = displacement - normal_offset
                loss = loss + opt.lambda_point_tangent * torch.nn.functional.smooth_l1_loss(
                    tangent_offset,
                    torch.zeros_like(tangent_offset),
                    beta=0.001,
                )
        
        # ================== 【修改后：深度 Loss 计算】 =================        
        loss_depth = torch.zeros((), device="cuda")
        pred_depth = None
        gt_depth_for_vis = None
        
        if (
           getattr(viewpoint_cam, "original_depth", None) is not None
           and "depth" in meshdict
        ):
           raw_gt_depth = viewpoint_cam.original_depth.to(
               device="cuda",
               dtype=meshdict["depth"].dtype,
               non_blocking=True
           )

           pred_depth = meshdict["depth"].permute(2, 0, 1)

           loss_depth, gt_depth_for_vis = pearson_depth_loss(
               pred_depth,
               raw_gt_depth,
               viewpoint_cam.gt_alpha_mask
           )
           
           lambda_depth = opt.lambda_depth

           loss = loss + lambda_depth * loss_depth

        if (
            opt.lambda_disparity > 0
            and getattr(viewpoint_cam, "original_disparity", None) is not None
            and "depth" in meshdict
        ):
            predicted_camera_depth = meshdict.get(
                "camera_z_depth", meshdict["depth"]
            ).permute(2, 0, 1)
            gt_disparity = viewpoint_cam.original_disparity.to(
                device=predicted_camera_depth.device,
                dtype=predicted_camera_depth.dtype,
            )
            loss_disparity = pearson_disparity_loss(
                predicted_camera_depth,
                gt_disparity,
                viewpoint_cam.gt_alpha_mask,
            )
            loss = loss + opt.lambda_disparity * loss_disparity
           

        if boundary_region_weight > 0 or boundary_edge_weight > 0:
            key = (viewpoint_cam.colmap_id, viewpoint_cam.image_name)
            if key not in boundary_targets:
                boundary_targets[key] = prepare_target(viewpoint_cam.gt_alpha_mask)
            target = boundary_targets[key]
            if target is None:
                raise ValueError(f'Opaque boundary refinement requires reliable foreground coverage: {key}')
            region, edges, count = boundary_terms(
                meshdict, target, use_region=boundary_region_weight > 0,
                use_edges=boundary_edge_weight > 0, max_samples=opt.opaque_boundary_samples,
            )
            boundary_loss = boundary_region_weight * region + boundary_edge_weight * edges
            loss = loss + boundary_loss
            if iteration == 1 and opt.opaque_boundary_audit_gradient:
                gradients = torch.autograd.grad(boundary_loss, (gaussians._xyz, gaussians._normal),
                                                retain_graph=True, allow_unused=True)
                if any(g is not None and not torch.isfinite(g).all() for g in gradients):
                    raise RuntimeError('Nonfinite boundary gradient reached IMLS points')
                print('[OpaqueBoundary] isolated xyz/normal gradient L2',
                      [float(g.norm()) if g is not None else None for g in gradients], flush=True)
            if iteration == 1 or iteration % 50 == 0:
                print(f'[OpaqueBoundary] iter={iteration} region={float(region.detach()):.6f} '
                      f'edges={float(edges.detach()):.6f} visible_subsegments={count}', flush=True)

        loss.backward()                
#                # ================== 【调试：保存深度图】 ==================
#                if iteration % 500 == 0:
#
#                    debug_dir = os.path.join(scene.model_path, "debug_depth")
#                    os.makedirs(debug_dir, exist_ok=True)
#                    
#                    # 1. 处理预测深度
#                    p_d = pred_depth.detach().cpu().squeeze().numpy()
#                    p_d = (p_d - p_d.min()) / (p_d.max() - p_d.min() + 1e-8)
#                    
#                    # 2. 处理 GT 深度 (使用函数返回的一致性 GT)
#                    if iteration % 500 == 0 and pred_depth is not None:
#                        g_d = gt_depth_for_vis.detach().cpu().squeeze().numpy()
#                        # 直接归一化即可，逻辑已经在函数里对齐了
#                        g_d = (g_d - g_d.min()) / (g_d.max() - g_d.min() + 1e-8)
#                    else:
#                        g_d = np.zeros_like(p_d)
#                    
#                    # 3. 保存
#                    p_img = (p_d * 255).astype(np.uint8)
#                    g_img = (g_d * 255).astype(np.uint8)
#                    concat_img = np.concatenate([p_img, g_img], axis=1)
#                    Image.fromarray(concat_img).save(os.path.join(debug_dir, f"iter_{iteration:05d}.png"))
#                # ============================================================

        iter_end.record()
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "Norm": f"{loss_normal.item():.{5}f}",
                                          "Depth": f"{loss_depth.item():.5f}",
                                          "Mask": f"{loss_mask.item():.5f}"
                                         }) 
                progress_bar.update(10)
                
            if iteration % 100 == 0:
                    if tb_writer:
                        tb_writer.add_scalar('train_loss_patches/l1_loss', loss_L1.item(), iteration)
                        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
                        tb_writer.add_scalar('train_loss_patches/normal_loss', loss_normal.item(), iteration) # <--- 新增监控
                        if consistency_enabled:
                            tb_writer.add_scalar('train_loss_patches/imls_surface', imls_surface_loss.item(), iteration)
                            tb_writer.add_scalar('train_loss_patches/imls_sign', imls_sign_loss.item(), iteration)
                            tb_writer.add_scalar('train_loss_patches/imls_tangent', imls_tangent_loss.item(), iteration)
                            tb_writer.add_scalar('train_loss_patches/imls_valid_fraction', imls_valid_fraction.item(), iteration)
                        if gaussians.implicit_sdf_grid is not None:
                            tb_writer.add_scalar('train_loss_patches/implicit_sdf_magnitude', implicit_magnitude_loss.item(), iteration)
                            tb_writer.add_scalar('train_loss_patches/implicit_sdf_tv', implicit_tv_loss.item(), iteration)
                        if opt.lambda_reflection_normal_tv > 0:
                            tb_writer.add_scalar('train_loss_patches/reflection_normal_tv', reflection_normal_tv_loss.item(), iteration)
            
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # upsample
            if (iteration <= opt.upsample_until) and (iteration > opt.upsample_from) and ((iteration - opt.upsample_from) % opt.upsample_interval == 0):
                opt.grid_resolution = opt.grid_resolution + 64
                imlsplat = imlsplatting(opt.__dict__)
                print('upsample to ', str(opt.grid_resolution))

                gaussians.scale_range = True
                max_scale  = (opt.meshscale / (opt.grid_resolution - 1) * 1.5)
                init_scale = (opt.meshscale / (opt.grid_resolution - 1) * 1.0)
                gaussians.scale_max_val = torch.ones([gaussians.get_xyz.shape[0],1]).cuda() * max_scale

                new_scaling = inverse_sigmoid(init_scale / gaussians.scale_max_val)
                new_xyz     = gaussians._xyz
                new_normal  = gaussians._normal
                new_feature = gaussians._feature
                
                gaussians.prune_points(torch.ones(gaussians.get_xyz.shape[0]).bool().cuda())
                gaussians.densify_points({"xyz": new_xyz, "normal": new_normal, "scaling": new_scaling, "feature": new_feature})
                upsample = True
            else:
                upsample = False

            # resample
            if (not upsample) and (iteration <= opt.resample_until) and (iteration > opt.resample_from) and (iteration % opt.resample_interval == 0):
                opt_mesh = imlsplat(gaussians, scene.getTrainCameras()[0], bg, resample=True, cameras=scene.getTrainCameras(), step=iteration)
                
            # record
            if (iteration % 500 == 0):
                meshdict = imlsplat(gaussians, scene.getTrainCameras()[0], bg, record='record/'+str(iteration), cameras=scene.getTrainCameras(), step=iteration)

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                if boundary_displacement > 0:
                    limit = boundary_displacement * opt.meshscale / (opt.grid_resolution - 1)
                    shift = gaussians.get_xyz - reference_xyz
                    ratio = (limit / torch.linalg.vector_norm(shift, dim=1, keepdim=True).clamp_min(1e-12)).clamp_max(1)
                    gaussians._xyz.copy_(reference_xyz + shift * ratio)
                gaussians.optimizer.zero_grad(set_to_none = True)
                
                
            if iteration == opt.iterations and not opt.skip_final_render:
                print(f"\n[Final] Reached iteration {iteration}. Rendering ALL views...")
                
                # 1. 临时清空缓存
                torch.cuda.empty_cache()
                
                # 2. 定义保存路径
                final_render_path = os.path.join(scene.model_path, f"renders_iter_{iteration}")

                final_depth_estimator = None
                if opt.save_final_geometry_maps and opt.save_final_depth_maps:
                    try:
                        final_depth_estimator = OnlineDepthEstimator(
                            checkpoint=opt.depth_model_path
                        )
                    except Exception as error:
                        print(
                            "[Warning] Final Depth Anything export disabled:",
                            error
                        )

                with torch.no_grad():
                    # 分别处理 train 和 test
                    # render_tasks = {
                    #     "train": scene.getTrainCameras(),
                    #     "test":  scene.getTestCameras()
                    # }
                   
                    render_tasks = {
                        "train": ordered_export_cameras(scene.getTrainCameras()),
                        "test": ordered_export_cameras(scene.getTestCameras())
                    }
                    if not opt.final_render_train:
                        render_tasks.pop("train", None)
                    if opt.final_render_test_limit > 0:
                        render_tasks['test'] = render_tasks['test'][:opt.final_render_test_limit]

                    for split_name, cameras in render_tasks.items():
                        if not cameras: continue
                        
                        # 创建子文件夹
                        render_dir = os.path.join(final_render_path, split_name, "renders")
                        gt_dir     = os.path.join(final_render_path, split_name, "gt")
                        os.makedirs(render_dir, exist_ok=True)
                        os.makedirs(gt_dir, exist_ok=True)
                        write_view_manifest(os.path.join(final_render_path, split_name, 'views.json'), cameras)
                        depth_dir = os.path.join(
                            final_render_path, split_name, "depth_comparison"
                        )
                        normal_dir = os.path.join(
                            final_render_path, split_name, "normal_comparison"
                        )
                        raw_geometry_dir = os.path.join(
                            final_render_path, split_name, "geometry_raw"
                        )
                        
                        # 使用 tqdm 显示进度
                        for idx, view in enumerate(tqdm(cameras, desc=f"Rendering {split_name}", leave=False)):
                            
                            # --- 核心渲染逻辑 ---
                            # 使用当前的 imlsplat (包含正确的分辨率) 和 gaussians (包含正确的 scale_range/max_val)
                            meshdict = imlsplat(gaussians, view, bg, step=iteration)
                            
                            # --- 图像转换 ---
                            render_tensor = meshdict['image'][..., :3]
                            render_tensor = torch.clamp(render_tensor, 0, 1)
                            render_image = (render_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
                            
                            # --- GT 图像转换 ---
                            gt_tensor = composite_reference(
                                view.original_image[:3].cuda(), view.gt_alpha_mask,
                                bg, view.image_background,
                            ).permute(1, 2, 0)
                            gt_image = (torch.clamp(gt_tensor, 0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
                            
                            # --- 保存 ---
                            # fname = view.image_name if hasattr(view, 'image_name') else '{0:05d}'.format(idx)
                            fname = '{0:05d}'.format(idx)
                            
                            # Image.fromarray(render_image).save(os.path.join(save_dir, f"{fname}.png"))
                            # Image.fromarray(gt_image).save(os.path.join(save_dir, f"gt_{fname}.png"))
                            
                            Image.fromarray(render_image).save(os.path.join(render_dir, f"{fname}.png"))
                            Image.fromarray(gt_image).save(os.path.join(gt_dir, f"{fname}.png"))

                            if (
                                opt.save_final_geometry_maps
                                and "normal_world" in meshdict
                                and "camera_z_depth" in meshdict
                            ):
                                _save_geometry_raw(
                                    os.path.join(raw_geometry_dir, f"{fname}.npz"),
                                    meshdict,
                                    view,
                                )

                            if final_depth_estimator is not None and "depth" in meshdict:
                                try:
                                    depth_anything = final_depth_estimator.infer_depth(
                                        view.original_image
                                    )
                                    _save_depth_comparison(
                                        os.path.join(depth_dir, f"{fname}.png"),
                                        meshdict.get("camera_z_depth", meshdict["depth"]),
                                        depth_anything,
                                        meshdict.get("alpha"),
                                        view.gt_alpha_mask,
                                    )
                                except Exception as error:
                                    print(
                                        f"[Warning] Skip depth {split_name}/{fname}:",
                                        error
                                    )

                            if (
                                opt.save_final_geometry_maps
                                and "image_normals" in meshdict
                                and getattr(view, "original_normal", None) is not None
                            ):
                                _save_normal_comparison(
                                    os.path.join(normal_dir, f"{fname}.png"),
                                    meshdict["image_normals"],
                                    view.original_normal,
                                    meshdict.get("alpha"),
                                    view.gt_alpha_mask,
                                )
                
                if final_depth_estimator is not None:
                    final_depth_estimator.clear_memory()
                torch.cuda.empty_cache()
        
   

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint_ply", type=str, default = None)
    parser.add_argument("--expname", type=str, default='')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)
    
    # Start GUI server, configure and run training
    training(lp.extract(args), op.extract(args), args.save_iterations, args.start_checkpoint_ply, args.expname)

    # All done
    print("\nTraining complete.")
