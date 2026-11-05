import os, time
import trimesh
import torch
import numpy  as np
import open3d as o3d
import torch.nn.functional as F
import nvdiffrast.torch    as dr
import diffsplatting

from PIL import Image
from os import makedirs
from diso import DiffMC
from utils.general_utils import inverse_sigmoid
from utils.surface_regularization import apply_bounded_implicit_residual
from utils.implicit_pixel_normals import blend_implicit_pixel_normals

os.environ["PYOPENGL_PLATFORM"] = "egl"

def compute_grid_gradients(sdf_grid):
    # 使用 padding 保持尺寸一致，避免边缘丢失
    pad_sdf = F.pad(sdf_grid, (1, 1, 1, 1, 1, 1), mode='replicate')

    # 中心差分: f(x+1) - f(x-1)
    # 注意：out_sdfs 的维度顺序通常是 [D, H, W] (Z, Y, X)
    # grid_sample 期望的 coordinate 是 (x, y, z)
    # 因此我们需要对应的梯度通道顺序为 (gx, gy, gz) 对应 (W, H, D)

    # Gradient along D (Z axis)
    g_z = (pad_sdf[..., 2:, 1:-1, 1:-1] - pad_sdf[..., :-2, 1:-1, 1:-1]) * 0.5
    # Gradient along H (Y axis)
    g_y = (pad_sdf[..., 1:-1, 2:, 1:-1] - pad_sdf[..., 1:-1, :-2, 1:-1]) * 0.5
    # Gradient along W (X axis)
    g_x = (pad_sdf[..., 1:-1, 1:-1, 2:] - pad_sdf[..., 1:-1, 1:-1, :-2]) * 0.5

    # 堆叠顺序必须适配 grid_sample 的 xyz 输入顺序 -> (g_x, g_y, g_z)
    gradients = torch.cat([g_x, g_y, g_z], dim=1)

    # 归一化梯度作为法线 (防止除零)
    gradients = F.normalize(gradients, dim=1, eps=1e-6)
    return gradients

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def auto_normals(meshdict):
    i0 = meshdict['mesh_faces'][:, 0].to(torch.long)
    i1 = meshdict['mesh_faces'][:, 1].to(torch.long)
    i2 = meshdict['mesh_faces'][:, 2].to(torch.long)

    v0 = meshdict['mesh_verts'][i0, :]
    v1 = meshdict['mesh_verts'][i1, :]
    v2 = meshdict['mesh_verts'][i2, :]

    face_normals = torch.cross(v1 - v0, v2 - v0)

    # Splat face normals to vertices
    v_nrm = torch.zeros_like(meshdict['mesh_verts'])
    v_nrm.scatter_add_(0, i0[:, None].repeat(1,3), face_normals)
    v_nrm.scatter_add_(0, i1[:, None].repeat(1,3), face_normals)
    v_nrm.scatter_add_(0, i2[:, None].repeat(1,3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    v_nrm = torch.where(dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
    v_nrm = safe_normalize(v_nrm)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_nrm))
    
    meshdict['mesh_v_nrm'] = v_nrm
    meshdict['mesh_t_nrm_idx'] = meshdict['mesh_faces']
    meshdict['mesh_face_nrm'] = face_normals
    return meshdict


class splatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means3D, normals, scales, features, resolution):
        args = (means3D, normals, scales, features, resolution)
        
        num_computed, num_phases, out_sums, out_sdfs, out_feat, geoBuffer, binBuffer, voxBuffer, spaBuffer = diffsplatting.splatterforward(*args)
        
        ctx.resolution   = resolution
        ctx.num_computed = num_computed
        ctx.num_phases   = num_phases
        ctx.save_for_backward(means3D, normals, scales, features, geoBuffer, binBuffer, voxBuffer, spaBuffer, out_sums, out_sdfs, out_feat)

        return out_sums, out_sdfs, out_feat

    @staticmethod
    def backward(ctx, dL_dout_sums, dL_dout_sdfs, dL_dout_feat):
        resolution   = ctx.resolution
        num_computed = ctx.num_computed
        num_phases   = ctx.num_phases
        means3D, normals, scales, features, geoBuffer, binBuffer, voxBuffer, spaBuffer, out_sums, out_sdfs, out_feat = ctx.saved_tensors

        args = (num_computed, num_phases, resolution,
                means3D, normals, scales, features,
                geoBuffer, binBuffer, voxBuffer, spaBuffer,
                out_sums, out_sdfs, out_feat, 
                dL_dout_sums, dL_dout_sdfs, dL_dout_feat)

        dL_dmeans3D, dL_dnormals, dL_dscales, dL_dfeatures = diffsplatting.splatterbackward(*args)
        grads = (dL_dmeans3D, dL_dnormals, dL_dscales, dL_dfeatures, None)
        
        return grads

class imlsplatting():
    def __init__(self, opt):
        self.opt        = opt
        self.rasterizer = dr.RasterizeCudaContext()
        self.mc_cuda    = DiffMC(dtype=torch.float32).cuda()
        self._component_face_mask = None
        self._component_signature = None

    def repair_sdf(self, surface_sdf, coverage=None):
        """Remove sub-voxel dents without changing the legacy path by default."""
        volume = surface_sdf.unsqueeze(0).unsqueeze(0)

        close_weight = min(1.0, max(0.0, float(self.opt.get('sdf_close_weight', 0.0))))
        if close_weight > 0.0:
            # Negative SDF denotes the object interior.  A min/max closing
            # fills isolated positive cavities while retaining the open mouth
            # of cups and other structures wider than one voxel.
            dilated_inside = -F.max_pool3d(-volume, kernel_size=3, stride=1, padding=1)
            closed = F.max_pool3d(dilated_inside, kernel_size=3, stride=1, padding=1)
            volume = torch.lerp(volume, closed, close_weight)

        smooth_weight = min(1.0, max(0.0, float(self.opt.get('sdf_smooth_weight', 0.0))))
        for _ in range(max(0, int(self.opt.get('sdf_smooth_iterations', 1))) if smooth_weight > 0.0 else 0):
            smoothed = F.avg_pool3d(
                F.pad(volume, (1, 1, 1, 1, 1, 1), mode='replicate'),
                kernel_size=3,
                stride=1,
            )
            volume = torch.lerp(volume, smoothed, smooth_weight)

        taubin_weight = min(1.0, max(0.0, float(
            self.opt.get('sdf_taubin_weight', 0.0)
        )))
        taubin_iterations = max(0, int(
            self.opt.get('sdf_taubin_iterations', 0)
        ))
        if taubin_weight > 0.0 and taubin_iterations > 0:
            if coverage is None:
                covered = torch.ones_like(volume, dtype=torch.bool)
            else:
                covered = coverage.to(device=volume.device, dtype=torch.bool)[None, None]
            voxel = 1.0 / max(int(surface_sdf.shape[-1]) - 1, 1)
            band = max(0.5, float(
                self.opt.get('sdf_taubin_band_voxels', 2.0)
            )) * voxel
            cosine = float(self.opt.get('sdf_taubin_cosine', 0.95))
            mu = float(self.opt.get('sdf_taubin_mu', -0.53))

            def diffuse_step(field, coefficient):
                # Edge gates are deliberately detached: they classify local
                # implicit-surface continuity without creating an incentive
                # to rotate gradients merely to disable regularization.
                grid_normals = compute_grid_gradients(field).detach()
                field_pad = F.pad(field, (1, 1, 1, 1, 1, 1), mode='replicate')
                normal_pad = F.pad(
                    grid_normals, (1, 1, 1, 1, 1, 1), mode='replicate'
                )
                covered_pad = F.pad(
                    covered.to(field.dtype), (1, 1, 1, 1, 1, 1),
                    mode='constant', value=0.0
                ) > 0.5
                neighbours = (
                    field_pad[..., 1:-1, 1:-1, :-2],
                    field_pad[..., 1:-1, 1:-1, 2:],
                    field_pad[..., 1:-1, :-2, 1:-1],
                    field_pad[..., 1:-1, 2:, 1:-1],
                    field_pad[..., :-2, 1:-1, 1:-1],
                    field_pad[..., 2:, 1:-1, 1:-1],
                )
                neighbour_normals = (
                    normal_pad[..., 1:-1, 1:-1, :-2],
                    normal_pad[..., 1:-1, 1:-1, 2:],
                    normal_pad[..., 1:-1, :-2, 1:-1],
                    normal_pad[..., 1:-1, 2:, 1:-1],
                    normal_pad[..., :-2, 1:-1, 1:-1],
                    normal_pad[..., 2:, 1:-1, 1:-1],
                )
                neighbour_coverage = (
                    covered_pad[..., 1:-1, 1:-1, :-2],
                    covered_pad[..., 1:-1, 1:-1, 2:],
                    covered_pad[..., 1:-1, :-2, 1:-1],
                    covered_pad[..., 1:-1, 2:, 1:-1],
                    covered_pad[..., :-2, 1:-1, 1:-1],
                    covered_pad[..., 2:, 1:-1, 1:-1],
                )
                neighbour_sum = torch.zeros_like(field)
                neighbour_count = torch.zeros_like(field)
                for neighbour, neighbour_normal, neighbour_valid in zip(
                    neighbours, neighbour_normals, neighbour_coverage
                ):
                    agreement = (
                        grid_normals * neighbour_normal
                    ).sum(dim=1, keepdim=True)
                    valid = (
                        covered & neighbour_valid & (agreement >= cosine)
                    ).to(field.dtype)
                    neighbour_sum = neighbour_sum + neighbour * valid
                    neighbour_count = neighbour_count + valid
                mean = neighbour_sum / neighbour_count.clamp_min(1.0)
                active = covered & (field.abs() <= band) & (neighbour_count > 0)
                return torch.where(
                    active,
                    field + coefficient * (mean - field),
                    field,
                )

            for _ in range(taubin_iterations):
                volume = diffuse_step(volume, taubin_weight)
                if mu != 0.0:
                    volume = diffuse_step(volume, mu * taubin_weight)

        return volume[0, 0]

    def regularize_low_confidence_sdf(self, surface_sdf, out_sums):
        """Smooth only weakly supported IMLS voxels with reliable neighbours.

        ``out_sums`` is the accumulated Gaussian-kernel support before IMLS
        normalization.  Reflection highlights can pull sparse boundary points
        into locally inconsistent sheets; treating every covered voxel as
        equally reliable then preserves those sheets.  This optional filter
        leaves locally well-supported voxels unchanged and blends only weak
        near-surface samples toward a support-weighted 6-neighbour estimate.
        The confidence mask is detached so optimization cannot reduce support
        merely to switch the regularizer off.  A zero weight is the exact
        historical path.
        """
        strength = min(1.0, max(0.0, float(
            self.opt.get('imls_confidence_smooth_weight', 0.0)
        )))
        iterations = max(0, int(
            self.opt.get('imls_confidence_smooth_iterations', 0)
        ))
        if strength <= 0.0 or iterations <= 0:
            return surface_sdf

        field = surface_sdf[None, None]
        support = out_sums.detach().clamp_min(0.0)[None, None]
        covered = support >= 1e-4
        voxel = 1.0 / max(int(surface_sdf.shape[-1]) - 1, 1)
        band = max(0.5, float(
            self.opt.get('imls_confidence_band_voxels', 2.0)
        )) * voxel
        floor = min(0.999, max(0.0, float(
            self.opt.get('imls_confidence_floor', 0.35)
        )))

        for _ in range(iterations):
            support_pad = F.pad(support, (1, 1, 1, 1, 1, 1))
            field_pad = F.pad(field, (1, 1, 1, 1, 1, 1), mode='replicate')
            neighbour_support = (
                support_pad[..., 1:-1, 1:-1, :-2],
                support_pad[..., 1:-1, 1:-1, 2:],
                support_pad[..., 1:-1, :-2, 1:-1],
                support_pad[..., 1:-1, 2:, 1:-1],
                support_pad[..., :-2, 1:-1, 1:-1],
                support_pad[..., 2:, 1:-1, 1:-1],
            )
            neighbour_field = (
                field_pad[..., 1:-1, 1:-1, :-2],
                field_pad[..., 1:-1, 1:-1, 2:],
                field_pad[..., 1:-1, :-2, 1:-1],
                field_pad[..., 1:-1, 2:, 1:-1],
                field_pad[..., :-2, 1:-1, 1:-1],
                field_pad[..., 2:, 1:-1, 1:-1],
            )
            local_max = support
            weighted_sum = torch.zeros_like(field)
            weight_sum = torch.zeros_like(field)
            for neighbour_s, neighbour_f in zip(
                neighbour_support, neighbour_field
            ):
                local_max = torch.maximum(local_max, neighbour_s)
                valid_weight = neighbour_s * (neighbour_s >= 1e-4)
                weighted_sum = weighted_sum + neighbour_f * valid_weight
                weight_sum = weight_sum + valid_weight

            relative_support = support / local_max.clamp_min(1e-6)
            weak = ((floor - relative_support) / max(floor, 1e-6)).clamp(0.0, 1.0)
            blend = strength * weak
            active = covered & (field.abs() <= band) & (weight_sum > 0)
            neighbour_mean = weighted_sum / weight_sum.clamp_min(1e-6)
            field = torch.where(
                active,
                torch.lerp(field, neighbour_mean, blend),
                field,
            )

        return field[0, 0]

    def fill_isolated_coverage_holes(self, meshdict, resolution):
        """Interpolate only empty voxels enclosed by reliable IMLS samples."""
        min_neighbours = int(self.opt.get('coverage_hole_fill_min_neighbors', 0))
        iterations = max(1, int(self.opt.get('coverage_hole_fill_iterations', 1)))
        out_sdfs = meshdict['out_sdfs']
        out_feat = meshdict['out_feat']
        covered = meshdict['out_sums'] > 0
        voxel_offset = 1.0 / (resolution - 1)

        if min_neighbours <= 0:
            return out_sdfs + (~covered).to(out_sdfs.dtype) * voxel_offset

        min_neighbours = min(26, max(1, min_neighbours))
        working_sdf = out_sdfs
        working_feat = out_feat
        filled_total = 0
        kernel = torch.ones(1, 1, 3, 3, 3, device=out_sdfs.device, dtype=out_sdfs.dtype)

        for _ in range(iterations):
            covered_f = covered.to(out_sdfs.dtype)[None, None]
            neighbour_count = F.conv3d(covered_f, kernel, padding=1)[0, 0]
            fill_mask = (~covered) & (neighbour_count >= min_neighbours)
            if not bool(fill_mask.any()):
                break

            sdf_sum = F.conv3d(
                (working_sdf * covered.to(out_sdfs.dtype))[None, None],
                kernel,
                padding=1,
            )[0, 0]
            interpolated_sdf = sdf_sum / neighbour_count.clamp_min(1.0)
            working_sdf = torch.where(fill_mask, interpolated_sdf, working_sdf)

            feature_volume = working_feat.permute(3, 0, 1, 2).unsqueeze(0)
            feature_sum = F.conv3d(
                feature_volume * covered_f,
                kernel.expand(feature_volume.shape[1], 1, 3, 3, 3),
                padding=1,
                groups=feature_volume.shape[1],
            )
            interpolated_feat = (
                feature_sum / neighbour_count.clamp_min(1.0)[None, None]
            )[0].permute(1, 2, 3, 0)
            working_feat = torch.where(fill_mask[..., None], interpolated_feat, working_feat)
            covered = covered | fill_mask
            filled_total += int(fill_mask.sum().item())

        meshdict['out_feat'] = working_feat
        if not hasattr(self, '_coverage_repair_reported'):
            print(
                f"[Coverage repair] interpolated {filled_total} enclosed voxels "
                f"(minimum neighbours {min_neighbours}, iterations {iterations})."
            )
            self._coverage_repair_reported = True
        return working_sdf + (~covered).to(out_sdfs.dtype) * voxel_offset

    def filter_small_mesh_components(self, meshdict):
        """Drop disconnected MC fragments, caching topology for frozen geometry."""
        min_faces = int(self.opt.get('mesh_component_min_faces', 0))
        min_fraction = max(0.0, float(self.opt.get('mesh_component_min_fraction', 0.0)))
        if min_faces <= 0 and min_fraction <= 0.0:
            return meshdict

        faces = meshdict['mesh_faces']
        signature = (int(meshdict['mesh_verts'].shape[0]), int(faces.shape[0]))
        if self._component_signature != signature or self._component_face_mask is None:
            try:
                from scipy.sparse import coo_matrix
                from scipy.sparse.csgraph import connected_components

                face_np = faces.detach().to('cpu', torch.long).numpy()
                vertex_count = signature[0]
                src = np.concatenate((face_np[:, 0], face_np[:, 1], face_np[:, 2]))
                dst = np.concatenate((face_np[:, 1], face_np[:, 2], face_np[:, 0]))
                graph = coo_matrix(
                    (np.ones(src.shape[0], dtype=np.uint8), (src, dst)),
                    shape=(vertex_count, vertex_count),
                )
                _, labels = connected_components(graph, directed=False)
                face_labels = labels[face_np[:, 0]]
                counts = np.bincount(face_labels)
                largest = int(counts.max()) if counts.size else 0
                threshold = max(min_faces, int(np.ceil(largest * min_fraction)))
                keep_np = counts[face_labels] >= threshold
                self._component_face_mask = torch.from_numpy(keep_np).to(faces.device)
                self._component_signature = signature
                removed = int((~keep_np).sum())
                print(
                    f"[Mesh repair] retained {int(keep_np.sum())}/{len(keep_np)} faces "
                    f"(removed {removed}, threshold {threshold})."
                )
            except Exception as exc:
                print(f"[Mesh repair] component filtering disabled: {exc}")
                self._component_face_mask = torch.ones(
                    faces.shape[0], dtype=torch.bool, device=faces.device
                )
                self._component_signature = signature

        meshdict['mesh_faces'] = faces[self._component_face_mask].contiguous()
        return meshdict

    def smooth_mesh_normals(self, meshdict):
        weight = min(1.0, max(0.0, float(self.opt.get('normal_smooth_weight', 0.0))))
        iterations = max(0, int(self.opt.get('normal_smooth_iterations', 0)))
        if weight <= 0.0 or iterations == 0:
            return meshdict

        cosine = float(self.opt.get('normal_smooth_cosine', 0.5))
        faces = meshdict['mesh_faces'].to(torch.long)
        src = torch.cat((faces[:, 0], faces[:, 1], faces[:, 2],
                         faces[:, 1], faces[:, 2], faces[:, 0]))
        dst = torch.cat((faces[:, 1], faces[:, 2], faces[:, 0],
                         faces[:, 0], faces[:, 1], faces[:, 2]))
        normals = meshdict['mesh_v_nrm']
        for _ in range(iterations):
            agreement = torch.sum(normals[src] * normals[dst], dim=-1)
            edge_weight = (agreement >= cosine).to(normals.dtype)
            accumulated = torch.zeros_like(normals)
            degree = torch.zeros(normals.shape[0], 1, device=normals.device, dtype=normals.dtype)
            accumulated.scatter_add_(0, src[:, None].expand(-1, 3), normals[dst] * edge_weight[:, None])
            degree.scatter_add_(0, src[:, None], edge_weight[:, None])
            neighbour_mean = accumulated / degree.clamp_min(1.0)
            has_neighbours = degree > 0
            blended = F.normalize(torch.lerp(normals, neighbour_mean, weight), dim=-1)
            normals = torch.where(has_neighbours, blended, normals)
        meshdict['mesh_v_nrm'] = normals
        return meshdict

    def smooth_mesh_vertices(self, meshdict):
        """Edge-aware Taubin smoothing of the extracted reflective surface.

        Unlike shading-normal smoothing, this changes the actual zero-level
        mesh. Normal-consistent adjacency avoids crossing genuine sharp edges;
        a negative second pass limits Laplacian shrinkage and silhouette loss.
        """
        weight = min(1.0, max(0.0, float(
            self.opt.get('mesh_vertex_smooth_weight', 0.0)
        )))
        iterations = max(0, int(self.opt.get('mesh_vertex_smooth_iterations', 0)))
        if weight <= 0.0 or iterations == 0:
            return meshdict

        faces = meshdict['mesh_faces'].to(torch.long)
        undirected = torch.cat((
            faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]
        ), dim=0).sort(dim=1).values
        edges, counts = torch.unique(undirected, dim=0, return_counts=True)
        src = torch.cat((edges[:, 0], edges[:, 1]))
        dst = torch.cat((edges[:, 1], edges[:, 0]))
        normals = F.normalize(meshdict['mesh_v_nrm'], dim=-1)
        cosine = float(self.opt.get('mesh_vertex_smooth_cosine', 0.9))
        edge_valid = (
            (normals[edges[:, 0]] * normals[edges[:, 1]]).sum(-1) >= cosine
        )
        directed_weight = torch.cat((edge_valid, edge_valid)).to(normals.dtype)

        boundary = torch.zeros(
            normals.shape[0], dtype=torch.bool, device=normals.device
        )
        boundary_edges = edges[counts == 1]
        if boundary_edges.numel():
            boundary[boundary_edges.reshape(-1)] = True
        boundary_weight = min(1.0, max(0.0, float(
            self.opt.get('mesh_vertex_boundary_weight', 0.1)
        )))
        vertex_weight = torch.where(
            boundary,
            normals.new_full((normals.shape[0],), boundary_weight),
            normals.new_ones(normals.shape[0]),
        )

        def laplacian_step(vertices, coefficient):
            accumulated = torch.zeros_like(vertices)
            degree = torch.zeros(
                vertices.shape[0], 1, device=vertices.device, dtype=vertices.dtype
            )
            accumulated.index_add_(0, src, vertices[dst] * directed_weight[:, None])
            degree.index_add_(0, src, directed_weight[:, None])
            neighbour_mean = accumulated / degree.clamp_min(1.0)
            displacement = (neighbour_mean - vertices) * vertex_weight[:, None]
            return torch.where(
                degree > 0,
                vertices + coefficient * displacement,
                vertices,
            )

        vertices = meshdict['mesh_verts']
        taubin = float(self.opt.get('mesh_vertex_smooth_taubin', -0.53))
        for _ in range(iterations):
            vertices = laplacian_step(vertices, weight)
            if taubin != 0.0:
                vertices = laplacian_step(vertices, taubin * weight)
        meshdict['mesh_verts'] = vertices.contiguous()
        meshdict = auto_normals(meshdict)
        return meshdict
        
    def normalize(self, means3D, scales, axis_center):
        means3D = means3D - axis_center
        means3D = means3D / self.opt['meshscale'] # -0.5 ~ 0.5
        means3D = means3D + 0.5 # 0 ~ 1
        scales  = scales  / self.opt['meshscale']
        return means3D, scales
    
    def denormlize(self, mesh_verts, axis_center):
        mesh_verts = (mesh_verts - 0.5) * self.opt['meshscale'] + axis_center
        return mesh_verts

    def proj2clip(self, meshdict, camera):
        verts = meshdict['mesh_verts']
        num_verts, _ = verts.shape
        projmatrix = camera.full_proj_transform
        ones = torch.ones(num_verts, 1, dtype=verts.dtype, device=verts.device)
        verts_hom  = torch.cat([verts, ones], dim=1)
        verts_clip = torch.matmul(verts_hom, projmatrix.unsqueeze(0))
        meshdict['verts_clip'] = verts_clip
        return meshdict
        
    
    
    def marchingcube(self, meshdict, isovalue=0):
      out_sums = meshdict['out_sums']
      out_feat = meshdict['out_feat']

      # 使用 __call__ 中已经统一修正过的 SDF
      out_sdfs = meshdict.get(
          'surface_sdfs',
          meshdict['out_sdfs']
      )

      verts, feats, tris = self.mc_cuda(
          out_sdfs,
          out_feat,
          isovalue=isovalue,
          normalize=True
      )

      verts = verts[:, [2, 1, 0]]
      tris = tris[:, [2, 1, 0]]

      meshdict['mesh_verts'] = verts.contiguous()
      meshdict['mesh_feats'] = feats.contiguous()
      meshdict['mesh_faces'] = tris.contiguous()

      return meshdict
#    def marchingcube(self, meshdict, isovalue=0):
#        out_sums = meshdict['out_sums']
#        out_sdfs = meshdict['out_sdfs']
#        out_feat = meshdict['out_feat']
#
#        grid_resolution = out_sums.shape[0]
#        out_sdfs = out_sdfs + (out_sums == 0) * (1 / (grid_resolution - 1))
#
#        verts, feats, tris = self.mc_cuda(out_sdfs, out_feat, isovalue=isovalue, normalize=True)
#        verts = verts[:,[2,1,0]]
#        tris  = tris[:,[2,1,0]]
#
#        meshdict['mesh_verts'] = verts.contiguous()
#        meshdict['mesh_feats'] = feats.contiguous()
#        meshdict['mesh_faces'] = tris.contiguous()
#
#        return meshdict

    def rasterize(self, meshdict, camera, bg_color, refer):
        h, w = camera.original_image.shape[1:]
        if self.opt['SSAA'] > 1:
            h*=self.opt['SSAA']
            w*=self.opt['SSAA']

        rast , _ = dr.rasterize(self.rasterizer, meshdict['verts_clip'], meshdict['mesh_faces'], resolution=[h, w])
        attri, _ = dr.interpolate(meshdict['mesh_attri'], rast, meshdict['mesh_faces'])
        meshdict['attri'] = attri
        meshdict['rast']  = rast
        return meshdict
    
    def shading(self, pc, meshdict, camera, bg_color, refer):
        feature = meshdict['attri'][...,: pc.dim_feature]
        gb_pos  = meshdict['attri'][...,  pc.dim_feature : pc.dim_feature + 3]
        normals = F.normalize(meshdict['attri'][...,pc.dim_feature + 3 : pc.dim_feature + 6], dim=-1)
        if float(self.opt.get('implicit_pixel_normal_weight', 0.0)) > 0.0:
            normals = blend_implicit_pixel_normals(
                meshdict.pop('pixel_gradient_grid'), gb_pos, normals,
                pc.axis_center, self.opt['meshscale'],
                (meshdict['rast'][..., -1:] > 0).to(normals.dtype),
                self.opt['implicit_pixel_normal_weight'],
            )
        viewdir = F.normalize(gb_pos - camera.camera_center[None,None,:], dim=-1)

        map_dict = pc.rendernet(viewdir, normals, feature, (meshdict['rast'][..., -1:] > 0).float(), gb_pos, camera.camera_center)
        rgbs     = map_dict['rgb_map']
        
#        # gb_pos 是世界坐标系中的逐像素表面位置，形状为 [1, H, W, 3]
#        ones = torch.ones_like(gb_pos[..., :1])
#        gb_pos_h = torch.cat([gb_pos, ones], dim=-1)  # [1, H, W, 4]
#
#        # 世界坐标转换到相机坐标
#        gb_pos_camera = torch.matmul(
#            gb_pos_h,
#            camera.world_view_transform
#        )
#        # 取相机坐标系的 Z 分量作为轴向深度
#        pred_depth = gb_pos_camera[..., 2:3]
#        
#        # 前景掩码
#        alpha = (meshdict['rast'][..., -1:] > 0).float()
#        
#        if not hasattr(self, "_depth_sign_checked"):
#            valid_mask = alpha > 0.5
#
#            if valid_mask.any():
#                valid_depth = pred_depth[valid_mask].detach()
#
#                print(
#                    "[Depth Check] z-depth:",
#                    "min =", valid_depth.min().item(),
#                    "max =", valid_depth.max().item(),
#                    "mean =", valid_depth.mean().item(),
#                    "positive ratio =",
#                    (valid_depth > 0).float().mean().item()
#                )
#
#            self._depth_sign_checked = True
#    
#        # 背景深度置零
#        pred_depth = pred_depth * alpha
        
#
        cam_pos = camera.camera_center[None, None, None, :] # [1, 1, 1, 3]
        
        pred_depth = torch.norm(gb_pos - cam_pos, dim=-1, keepdim=True) # [1, H, W, 1]

        # Ref-NeRF disparity is inverse camera-axis depth, not inverse
        # Euclidean ray length. Keep both quantities because generic datasets
        # and Ref-NeRF supervision require different definitions.
        gb_pos_h = torch.cat((gb_pos, torch.ones_like(gb_pos[..., :1])), dim=-1)
        camera_z_depth = torch.matmul(gb_pos_h, camera.world_view_transform)[..., 2:3]

        # 加上 Alpha Mask，背景深度设为 0 或远平面
        alpha = (meshdict['rast'][..., -1:] > 0).float()
        pred_depth = pred_depth * alpha 
        camera_z_depth = camera_z_depth * alpha


        background = torch.ones_like(rgbs) * bg_color[0]
        background = torch.cat((background, torch.zeros_like(background[..., 0:1])), dim=-1)
        I = background

        alpha = (meshdict['rast'][..., -1:] > 0).float()
        I = torch.lerp(I, torch.cat((rgbs, torch.ones_like(alpha)), dim=-1), alpha)
        I = dr.antialias(I.contiguous(), meshdict['rast'], meshdict['verts_clip'], meshdict['mesh_faces'])
        
        if self.opt['SSAA'] > 1:
            ori_h, ori_w = I.shape[1] // self.opt['SSAA'], I.shape[2] // self.opt['SSAA']
            I       = F.interpolate(I.permute(0, 3, 1, 2),   size=(ori_h, ori_w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            gb_pos  = F.interpolate(gb_pos.permute(0, 3, 1, 2),  size=(ori_h, ori_w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            normals = F.interpolate(normals.permute(0, 3, 1, 2), size=(ori_h, ori_w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            alpha   = F.interpolate(alpha.permute(0, 3, 1, 2),   size=(ori_h, ori_w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            pred_depth = F.interpolate(pred_depth.permute(0, 3, 1, 2), size=(ori_h, ori_w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            camera_z_depth = F.interpolate(camera_z_depth.permute(0, 3, 1, 2), size=(ori_h, ori_w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        # Interpolation changes vector length.  Keep the signed world-space
        # normal independent from coverage and normalize it again after SSAA.
        normals = F.normalize(normals, dim=-1, eps=1e-6)

        # Use the differentiable antialiased coverage for every screen-space
        # geometry buffer. The previous hard coverage blocked normal/position
        # supervision exactly where an incomplete silhouette must expand.
        soft_alpha = I[..., 3:4].clamp(0.0, 1.0)
        meshdict['image']         = I[0]
        meshdict['alpha']         = soft_alpha[0]
        meshdict['normal_world']   = normals[0]
        meshdict['image_normals'] = ((normals + 1) * 0.5)[0]
        meshdict['image_gb_pos']  = (((gb_pos  / self.opt['meshscale']) + 0.5) * soft_alpha)[0]
        meshdict['depth']         = pred_depth[0]
        meshdict['camera_z_depth'] = camera_z_depth[0]
        meshdict['surface_position'] = gb_pos[0]
        for key in ('diffuse_map', 'specular_map', 'roughness_map', 'fresnel_map',
                    'reflection_normal_offset_map', 'factorized_reflection_map'):
            if key not in map_dict:
                continue
            material_map = map_dict[key]
            if self.opt['SSAA'] > 1:
                material_map = F.interpolate(
                    material_map.permute(0, 3, 1, 2),
                    size=(ori_h, ori_w),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            meshdict[key] = material_map[0]
        
        return meshdict

    def resample(self, pc, meshdict, cameras, bg_color):
        verts    = meshdict['mesh_verts'].detach().cpu().numpy()
        faces    = meshdict['mesh_faces'].detach().cpu().numpy()
        vnormals = meshdict['mesh_v_nrm']

        sign_verts        = torch.zeros(vnormals.shape[0]).to('cuda')
        visiable_faces    = torch.zeros(faces.shape[0]).to('cuda')
        train_background  = torch.tensor([1,1,1], dtype=torch.float32, device="cuda")
        
        for camera in cameras:
            meshdict = self.proj2clip(meshdict, camera)
            meshdict = self.rasterize(meshdict, camera, bg_color, False)
            
            # rasterized tris
            tris_idxs = meshdict['rast'][0,:,:,3].reshape(-1)
            tris_idxs = (tris_idxs[tris_idxs > 0] - 1).to(torch.int64)
            tris_idxs = torch.unique(tris_idxs)
            vert_idxs = torch.unique(meshdict['mesh_faces'][tris_idxs, :].reshape(-1)).to(torch.int64)
            visiable_faces[tris_idxs] = 1

            # vert normal flip
            vert_mask = torch.zeros(vnormals.shape[0]).to('cuda')
            vert_mask[vert_idxs] = 1
            vert_mask = vert_mask.to(torch.bool)
            viewdir = F.normalize(camera.camera_center.unsqueeze(0) - meshdict['mesh_verts'], dim=-1)
            sign_vert = (torch.sum(vnormals * viewdir, dim=-1) > 0) * 2 - 1
            sign_verts[vert_mask] += sign_vert[vert_mask]

        # point normal flip and prune non-visiable points
        vnormals = (vnormals * ((sign_verts >= 0)*2 - 1).unsqueeze(-1))
        
        remain_faces = faces[visiable_faces.bool().detach().cpu().numpy()]
        num_resample = remain_faces.shape[0]

        face_verts = verts[remain_faces]
        v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

        face_normals = vnormals[remain_faces]
        n0, n1, n2 = face_normals[:, 0], face_normals[:, 1], face_normals[:, 2]

        face_feats = meshdict['mesh_feats'][remain_faces]
        f0, f1, f2 = face_feats[:, 0], face_feats[:, 1], face_feats[:, 2]

        points_center = (v0 + v1 + v2) / 3

        points_normal = (n0 + n1 + n2) / 3
        points_normal = np.asarray(F.normalize(points_normal, dim=-1).detach().cpu())

        points_feature = (f0 + f1 + f2) / 3
        points_feature = np.asarray(points_feature.detach().cpu())

        num_resample      = points_center.shape[0]
        resample_points   = torch.from_numpy(points_center).float().cuda()
        resample_normals  = torch.from_numpy(points_normal).float().cuda()
        resample_feature  = torch.from_numpy(points_feature).float().cuda()

        max_scale  = (self.opt['meshscale'] / (self.opt['grid_resolution'] - 1) * 1.5)
        init_scale = (self.opt['meshscale'] / (self.opt['grid_resolution'] - 1) * 1.0)
        min_scale  = 0.00001

        pc.scale_min_val = torch.ones([num_resample,1]).cuda() * min_scale
        pc.scale_max_val = torch.ones([num_resample,1]).cuda() * max_scale
        pc.scale_range = True
        resample_scaling = inverse_sigmoid(init_scale / pc.scale_max_val)

        pc.prune_points(torch.ones(pc.get_xyz.shape[0]).bool().cuda())
        pc.densify_points({"xyz": resample_points, "normal": resample_normals, "scaling": resample_scaling, "feature": resample_feature})

        print('visiable mesh point number  : ', pc.get_xyz.shape[0])
    
    def visiable_mesh(self, pc, meshdict, cameras, bg_color):
        verts    = meshdict['mesh_verts'].detach().cpu().numpy()
        faces    = meshdict['mesh_faces'].detach().cpu().numpy()
        vnormals = meshdict['mesh_v_nrm']
        visiable_faces = torch.zeros(faces.shape[0]).to('cuda')

        for camera in cameras:
            meshdict = self.proj2clip(meshdict, camera)
            meshdict = self.rasterize(meshdict, camera, bg_color, False)

            # rasterized tris
            tris_idxs = meshdict['rast'][0,:,:,3].reshape(-1)
            tris_idxs = (tris_idxs[tris_idxs > 0] - 1).to(torch.int64)
            tris_idxs = torch.unique(tris_idxs)
            vert_idxs = torch.unique(meshdict['mesh_faces'][tris_idxs, :].reshape(-1)).to(torch.int64)
            visiable_faces[tris_idxs] = 1

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        invisiable_faces = (1 - visiable_faces).detach().cpu().numpy().astype(np.bool)
        mesh.remove_triangles_by_mask(invisiable_faces)
        mesh.remove_unreferenced_vertices()

        visiable_mesh = trimesh.Trimesh(vertices = np.asarray(mesh.vertices), faces = np.asarray(mesh.triangles))     
        meshdict['visiable_mesh'] = visiable_mesh
        return meshdict

    def record(self, pc, meshdict, camera , bg_color, record, refer=False):
        with torch.no_grad():
            save_path = os.path.join(self.opt['model_path'], os.path.dirname(record))
            save_name = record.split('/')[-1]
            makedirs(save_path, exist_ok=True)

            gt_image  = camera.original_image.cuda().permute(1,2,0)
            if camera.gt_alpha_mask is not None:
                gt_alpha_mask = camera.gt_alpha_mask.cuda().permute(1,2,0)
                gt_image = gt_image * gt_alpha_mask + bg_color * (1 - gt_alpha_mask)
            gt_image       = (gt_image.cpu().numpy() * 255).astype(np.uint8)
            render_image   = (torch.clamp(meshdict['image'][...,:3], 0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
            image_normals  = (meshdict['image_normals'][...,:3].detach().cpu().numpy()   * 255).astype(np.uint8)
            image_gb_pos   = (meshdict['image_gb_pos'][...,:3].detach().cpu().numpy()    * 255).astype(np.uint8)
            all_image      = np.concatenate((gt_image, render_image, image_normals, image_gb_pos), axis=1)
            Image.fromarray(all_image).save(os.path.join(save_path, save_name+'_all.png'))

            if not refer:
                point_scales = pc.get_scaling
                scale_mean   = point_scales.mean().item()
                scale_max_thred = point_scales.max().item()
                point_scales = point_scales / scale_max_thred

                pcd = o3d.geometry.PointCloud()
                pcd.points  = o3d.utility.Vector3dVector(self.denormlize(meshdict['means3D'], pc.axis_center).detach().cpu().numpy())
                pcd.normals = o3d.utility.Vector3dVector(meshdict['normals'].detach().cpu().numpy())

                pcd.colors = o3d.utility.Vector3dVector(point_scales.repeat(1,3).detach().cpu().numpy())
                o3d.io.write_point_cloud(os.path.join(save_path, save_name + "_pc_scale_" + str(round(scale_max_thred, 3)) + '_' + str(round(scale_mean, 3)) + ".ply"), pcd)

                mesh_verts = meshdict['mesh_verts'].detach().cpu().numpy()
                mesh_faces = meshdict['mesh_faces'].detach().cpu().numpy()

                mesh_normal = (F.normalize(meshdict['mesh_v_nrm'], dim=-1) + 1) * 0.5
                mesh_normal = (mesh_normal * 255).detach().cpu().numpy()
        
                mesh = trimesh.Trimesh(vertices = mesh_verts, faces = mesh_faces, vertex_colors= mesh_normal)          
                mesh.export(os.path.join(save_path, save_name + '_mesh_nums.ply'))

    def __call__(self, pc, camera, bg_color, record=None, cameras=None, resample=False, refer=False, get_visiable=False, to_grid=False, step=0):
        
        means3D  = pc.get_xyz
        normals  = pc.get_normal
        scales   = pc.get_scaling
        features = pc.get_feature
        means3D, scales = self.normalize(means3D, scales, pc.axis_center)
#        meshdict = {'means3D': means3D, 'normals': normals, 'scales': scales, 'features': features}
        
#        out_sums, out_sdfs, out_feat = splatter.apply(means3D, normals, scales, features, self.opt['grid_resolution'])
#        
#        meshdict['out_sums'] = out_sums[0]
#        meshdict['out_sdfs'] = out_sdfs[0]
#        meshdict['out_feat'] = out_feat.permute(1, 2, 3, 0)
#        
#        grid_resolution = meshdict['out_sums'].shape[0]
#        
#        surface_sdf = (
#            meshdict['out_sdfs']
#            + (meshdict['out_sums'] == 0).to(meshdict['out_sdfs'].dtype)
#            * (1.0 / (grid_resolution - 1))
#        )
#        
#        meshdict['surface_sdfs'] = surface_sdf
#        
#        current_sdf = surface_sdf
#        
#        if current_sdf.shape[-1] == 1:
#            current_sdf = current_sdf.squeeze(-1)
#            
#        current_sdf = current_sdf.unsqueeze(0).unsqueeze(0)
#        
#        gradient_grid = compute_grid_gradients(current_sdf)
#        
#        meshdict = self.marchingcube(meshdict, isovalue=0)

        resolution = int(self.opt["grid_resolution"])

        if resolution <= 1:
            raise ValueError(
                f"grid_resolution must be larger than 1, got {resolution}"
            )

        # 自定义 CUDA 扩展使用连续张量
        means3D = means3D.contiguous()
        normals = normals.contiguous()
        scales = scales.contiguous()
        features = features.contiguous()

        # 注意：meshdict 应在 contiguous 之后重新构建
        meshdict = {
            "means3D": means3D,
            "normals": normals,
            "scales": scales,
            "features": features,
        }

        confidence = pc.get_imls_confidence.reshape(-1).to(
            device=means3D.device, dtype=means3D.dtype
        ).clamp(0.0, 1.0)
        auxiliary = confidence < 0.999
        if bool(auxiliary.any()) and bool((~auxiliary).any()):
            # V20: evaluate original and auxiliary support independently, then
            # combine their unnormalized IMLS moments.  The CUDA kernel stays
            # unchanged and the all-one legacy path above remains bit-exact.
            primary_out = splatter.apply(
                means3D[~auxiliary], normals[~auxiliary], scales[~auxiliary],
                features[~auxiliary], resolution
            )
            auxiliary_out = splatter.apply(
                means3D[auxiliary], normals[auxiliary], scales[auxiliary],
                features[auxiliary], resolution
            )
            primary_sums, primary_sdfs, primary_feat = primary_out
            auxiliary_sums, auxiliary_sdfs, auxiliary_feat = auxiliary_out
            auxiliary_weight = confidence[auxiliary].mean()
            primary_floor = max(0.0, float(
                self.opt.get('imls_auxiliary_primary_support_floor', 0.0)
            ))
            if primary_floor > 0.0:
                # Auxiliary points may refine an existing weak IMLS
                # neighbourhood, but cannot independently create a zero-level
                # surface where the original point set has no support.  The
                # gate is a fixed confidence classifier, not an optimization
                # target for the original geometry.
                primary_gate = (
                    primary_sums / (primary_sums + primary_floor)
                ).detach()
            else:
                primary_gate = 1.0
            weighted_auxiliary_sums = (
                auxiliary_sums * auxiliary_weight * primary_gate
            )
            out_sums = primary_sums + weighted_auxiliary_sums
            denominator = out_sums.clamp_min(1e-8)
            out_sdfs = (
                primary_sdfs * primary_sums
                + auxiliary_sdfs * weighted_auxiliary_sums
            ) / denominator
            out_feat = (
                primary_feat * primary_sums
                + auxiliary_feat * weighted_auxiliary_sums
            ) / denominator
            if not hasattr(self, '_auxiliary_support_reported'):
                auxiliary_covered = auxiliary_sums >= 1e-4
                primary_overlap = auxiliary_covered & (primary_sums >= 1e-4)
                covered_count = int(auxiliary_covered.sum())
                overlap_count = int(primary_overlap.sum())
                if overlap_count > 0:
                    primary_values = primary_sums[primary_overlap]
                    auxiliary_values = auxiliary_sums[primary_overlap]
                    quantiles = torch.tensor(
                        [0.1, 0.5, 0.9], device=primary_values.device
                    )
                    primary_q = torch.quantile(primary_values, quantiles).tolist()
                    auxiliary_q = torch.quantile(auxiliary_values, quantiles).tolist()
                else:
                    primary_q = auxiliary_q = [0.0, 0.0, 0.0]
                print(
                    f"[IMLS support] {int(auxiliary.sum())} auxiliary points "
                    f"at weight {float(auxiliary_weight):.4f}; "
                    f"primary floor {primary_floor:.6f}; "
                    f"voxel overlap {overlap_count}/{covered_count} "
                    f"({overlap_count / max(covered_count, 1):.4%}); "
                    f"primary q10/50/90={primary_q}; "
                    f"auxiliary q10/50/90={auxiliary_q}."
                )
                self._auxiliary_support_reported = True
        else:
            out_sums, out_sdfs, out_feat = splatter.apply(
                means3D, normals, scales, features, resolution
            )

        meshdict["out_sums"] = out_sums[0]
        meshdict["out_sdfs"] = out_sdfs[0]
        meshdict["out_feat"] = out_feat.permute(1, 2, 3, 0)

        # 对未被高斯覆盖的空体素赋予正的 SDF 偏移
        surface_sdf = self.fill_isolated_coverage_holes(meshdict, resolution)

        surface_sdf = self.regularize_low_confidence_sdf(
            surface_sdf, meshdict["out_sums"]
        )
        surface_sdf = self.repair_sdf(
            surface_sdf, meshdict["out_sums"] > 0
        )
        surface_sdf, implicit_residual = apply_bounded_implicit_residual(
            surface_sdf,
            getattr(pc, "implicit_sdf_grid", None),
            self.opt.get("implicit_sdf_amplitude_voxels", 0.0),
        )
        if implicit_residual is not None:
            meshdict["implicit_sdf_residual"] = implicit_residual
        meshdict["surface_sdfs"] = surface_sdf

        # 使用与 Marching Cubes 相同的 surface_sdf 计算梯度
        current_sdf = surface_sdf.unsqueeze(0).unsqueeze(0)
        gradient_grid = compute_grid_gradients(current_sdf)
        if float(self.opt.get('implicit_pixel_normal_weight', 0.0)) > 0.0:
            meshdict['pixel_gradient_grid'] = gradient_grid

        meshdict = self.marchingcube(
            meshdict,
            isovalue=0
        )
        meshdict = self.filter_small_mesh_components(meshdict)

#        meshdict['out_sums']  = out_sums[0]
#        meshdict['out_sdfs']  = out_sdfs[0]
#        meshdict['out_feat']  = out_feat.permute(1,2,3,0)
#        
#        # current_sdf = meshdict['out_sdfs'].squeeze(-1).unsqueeze(0).unsqueeze(0)
#        # gradient_grid = compute_grid_gradients(current_sdf)  # [1, 3, D, H, W]
#        
#        current_sdf = meshdict['out_sdfs']
#        if current_sdf.shape[-1] == 1:
#            current_sdf = current_sdf.squeeze(-1)
#        current_sdf = current_sdf.unsqueeze(0).unsqueeze(0)
#        gradient_grid = compute_grid_gradients(current_sdf)
#
#        meshdict = self.marchingcube(meshdict, isovalue=0)
        

        meshdict = auto_normals(meshdict)
        meshdict = self.smooth_mesh_vertices(meshdict)
        
        # mesh_verts 现在是 [N, 3]，范围 [0, 1]，顺序为 (x, y, z) (因为 marchingcube 内部做了 [2,1,0] 交换)
        # grid_sample 需要范围 [-1, 1]
        sample_pos = meshdict['mesh_verts'] * 2.0 - 1.0
        sample_pos = sample_pos.view(1, 1, 1, -1, 3)  # [1, 1, N, 3]

        # 从梯度场采样高质量法线
        # gradient_grid 是 (gx, gy, gz)，sample_pos 是 (x, y, z)
        # 结果形状: [1, 3, 1, N]
        sampled_sdf_normals = F.grid_sample(gradient_grid, sample_pos, align_corners=True, padding_mode='border')

        # 调整形状为 [N, 3]
        sdf_nrm = sampled_sdf_normals.view(3, -1).permute(1, 0)

        # 归一化
        sdf_nrm = F.normalize(sdf_nrm, dim=-1)

        start_decay_step = 5000
        end_decay_step = 10000
        target_alpha = 0.1  # 最终保留 10% Mesh 法线约束，90% SDF 光滑法线

        configured_normal_weight = float(self.opt.get('mesh_normal_weight', -1.0))
        if configured_normal_weight >= 0.0:
            alpha = min(1.0, configured_normal_weight)
        elif step <= start_decay_step:
            alpha = 1.0
        elif step >= end_decay_step:
            alpha = target_alpha
        else:
            # 计算进度 0.0 -> 1.0
            progress = (step - start_decay_step) / (end_decay_step - start_decay_step)
            # 线性插值：从 1.0 降到 target_alpha
            alpha = 1.0 - (1.0 - target_alpha) * progress

        # 强制 SDF 法线方向与几何法线一致 (解决梯度指向内部的问题)
#        sdf_nrm = sdf_nrm * torch.sign(orientation_dot)
        orientation_dot = torch.sum(
            meshdict['mesh_v_nrm'] * sdf_nrm,
            dim=-1,
            keepdim=True
        )
        
        orientation_sign = torch.where(
            orientation_dot >= 0,
            torch.ones_like(orientation_dot),
            -torch.ones_like(orientation_dot)
        )
        sdf_nrm = sdf_nrm * orientation_sign

        # 混合并覆盖
        hybrid_nrm = F.normalize(alpha * meshdict['mesh_v_nrm'] + (1 - alpha) * sdf_nrm, dim=-1)
        meshdict['mesh_v_nrm'] = hybrid_nrm
        meshdict = self.smooth_mesh_normals(meshdict)

        meshdict['mesh_verts'] = self.denormlize(meshdict['mesh_verts'], pc.axis_center)
        meshdict['mesh_attri'] = torch.cat([meshdict['mesh_feats'], meshdict['mesh_verts'], meshdict['mesh_v_nrm']], dim=-1).contiguous()

        meshdict = self.proj2clip(meshdict, camera)
        meshdict = self.rasterize(meshdict, camera, bg_color, refer)
        meshdict = self.shading(pc, meshdict, camera, bg_color, refer)

        if resample:
            self.resample(pc, meshdict, cameras, bg_color)

        if record is not None:
            self.record(pc, meshdict, camera , bg_color, record, refer)

        if get_visiable:
            meshdict = self.visiable_mesh(pc, meshdict, cameras, bg_color)

        return meshdict

