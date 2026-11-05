"""Measure silhouettes and raw normal error on source-identified Ref-NeRF views."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt


def outline(mask):
    edge = mask & ~binary_erosion(mask, structure=np.ones((3,3), dtype=bool))
    edge[[0,-1],:] = False
    edge[:,[0,-1]] = False
    return edge


def boundary_metrics(prediction, truth):
    p, g = outline(prediction), outline(truth)
    if not g.any():
        raise ValueError('No valid GT silhouette boundary')
    if p.any():
        d1 = distance_transform_edt(~g)[p]
        d2 = distance_transform_edt(~p)[g]
        assd = .5 * (d1.mean() + d2.mean())
        hd95 = max(np.percentile(d1,95),np.percentile(d2,95))
    else:
        assd=hd95=np.hypot(*truth.shape)
        d1=d2=np.asarray([hd95])
    result={}
    for threshold in (1,2,4):
        precision,recall=(d1<=threshold).mean(),(d2<=threshold).mean()
        result[f'boundary_f{threshold}']=float(2*precision*recall/max(precision+recall,1e-12))
    result.update(assd_px=float(assd),hd95_px=float(hd95),
                  missing=float((truth & ~prediction).sum()/truth.sum()),
                  extra=float((~truth & prediction).sum()/truth.sum()),
                  silhouette_iou=float((truth & prediction).sum()/(truth | prediction).sum()))
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for key in ('code','model','data','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--iteration',type=int,required=True)
    p.add_argument('--split',choices=['heldout','test'],default='heldout')
    p.add_argument('--stride',type=int,default=8)
    p.add_argument('--reference',type=Path,help='Directory of paired reference support maps')
    p.add_argument('--save-support',action='store_true')
    p.add_argument('--save-images',action='store_true')
    p.add_argument('--aligned-domain-audit',action='store_true',
                   help='Heldout zero-update diagnostic: grow grid, keeping its spacing and parity')
    p.add_argument('--limit',type=int,default=0,help='Short diagnostic only; never a full split')
    a=p.parse_args()
    # Retain small heldout maps so all candidates can be compared on exactly
    # the same visible pixels after the paired probe, independent of order.
    if a.split=='heldout': a.save_support=True
    if a.aligned_domain_audit and (a.split!='heldout' or a.iteration!=1 or a.limit):
        p.error('Aligned domain diagnostic is restricted to the complete heldout zero-update split')
    sys.path.insert(0,str(a.code))
    import torch
    import torch.nn.functional as F
    from scene import Scene,GaussianModel
    from rasterizer import imlsplatting
    from utils.general_utils import safe_state
    from utils.loss_utils import ssim
    from utils.mesh_domain import aligned_mesh_domain
    from utils.refnerf_alpha import composite_reference
    from lpipsPyTorch.modules.lpips import LPIPS
    safe_state(False)
    opt=json.loads((a.model/'opt.json').read_text())
    opt.update(SSAA=2, save_final_geometry_maps=0, save_final_depth_maps=0)
    a.output.mkdir(parents=True,exist_ok=False)
    ds=SimpleNamespace(source_path=str(a.data), model_path=str(a.output), images='images', resolution=1,
                       white_background=True, eval=True, data_device='cpu')
    checkpoint=a.model/'point_cloud'/f'iteration_{a.iteration}'/'point_cloud.ply'
    g=GaussianModel(SimpleNamespace(**opt))
    scene=Scene(ds,g,shuffle=False,checkpoint_ply=str(checkpoint))
    grid, scale = int(opt['grid_resolution']), float(opt['meshscale'])
    voxel = scale / (grid - 1)
    radius = (g.get_xyz - g.axis_center).abs().amax(dim=1)
    outside = int((radius > scale*.5).sum())
    if a.aligned_domain_audit:
        opt['grid_resolution'], opt['meshscale'] = aligned_mesh_domain(grid, scale, 2*(float(radius.max())+5*voxel))
    domain = dict(original_grid=grid, original_scale=scale, voxel=voxel, outside_points=outside,
                  grid=opt['grid_resolution'], scale=opt['meshscale'])
    renderer=imlsplatting(opt)
    bg=torch.ones(3,device='cuda')
    model=LPIPS(net_type='vgg').cuda().eval()
    split='test' if a.split=='test' else 'train'
    frames=json.loads((a.data/f'transforms_{split}.json').read_text())['frames']
    views=sorted(scene.getTestCameras() if split=='test' else scene.getTrainCameras(),key=lambda c:c.colmap_id)
    assert len(views)==len(frames),(len(views),len(frames))
    if a.split=='heldout': views=[v for v in views if v.colmap_id % a.stride==0]
    if a.limit: views=views[:a.limit]
    if a.save_support: (a.output/'support').mkdir()
    if a.save_images:
        for directory in ('renders','gt','normal_comparison','coverage'):
            (a.output/directory).mkdir()
    rows=[]
    with torch.no_grad():
        for view in views:
            frame=frames[view.colmap_id]
            assert Path(view.image_name).stem==Path(frame['file_path']).stem,(view.image_name,frame['file_path'])
            assert np.allclose(view.camera_center.cpu().numpy(),np.asarray(frame['transform_matrix'])[:3,3],atol=1e-4)
            assert view.gt_alpha_mask is not None
            mesh=renderer(g,view,bg,step=a.iteration)
            m=mesh['alpha'].squeeze(-1).cpu().numpy()>=.5
            target=view.gt_alpha_mask.detach().cpu().squeeze().numpy()
            gt=target>=.5
            item=boundary_metrics(m,gt)
            y=composite_reference(view.original_image.cuda(),view.gt_alpha_mask,bg,view.image_background).clamp(0,1)
            x=mesh['image'][...,:3].permute(2,0,1).clamp(0,1)
            xb,yb=(x*255).to(torch.uint8),(y*255).to(torch.uint8)
            x,y=xb[None].float()/255,yb[None].float()/255
            sq=(x-y).square().mean(1)[0]
            band=torch.from_numpy(distance_transform_edt(~outline(gt))<=4).to(sq.device)
            item.update(psnr=float(-10*torch.log10(sq.mean().clamp_min(1e-12))),ssim=float(ssim(x,y)),
                        lpips=float(model(x*2-1,y*2-1)), boundary_rgb_rmse=float(sq[band].mean().sqrt()),
                        source_id=int(view.colmap_id),source_frame=frame['file_path'],
                        gt_sha256=hashlib.sha256(yb.cpu().numpy().tobytes()).hexdigest())
            fname=f'{view.colmap_id:05d}'
            normal_errors=np.full(gt.shape,np.nan,dtype=np.float32)
            if view.original_normal is not None:
                n=mesh['normal_world']
                true_normal=F.normalize(view.original_normal.cuda().permute(1,2,0)*2-1,dim=-1)
                err=torch.rad2deg(torch.acos((n*true_normal).sum(-1).clamp(-1,1))).cpu().numpy()
                valid=(target>=.99) & m & np.isfinite(err)
                normal_errors[valid]=err[valid]
                if valid.any():
                    item.update(normal_mae=float(err[valid].mean()),normal_p95=float(np.percentile(err[valid],95)),normal_valid=int(valid.sum()))
            if a.reference:
                with np.load(a.reference/'support'/f'{fname}.npz') as ref:
                    assert str(ref['gt_sha256'])==item['gt_sha256'] and str(ref['source_frame'])==frame['file_path']
                    e0=ref['normal_error']; shared=np.isfinite(e0)&np.isfinite(normal_errors)
                    new=np.isfinite(normal_errors)&~ref['coverage']
                    if shared.any():
                        item.update(paired_normal_mae=float(normal_errors[shared].mean()),paired_reference_normal_mae=float(e0[shared].mean()),paired_normal_count=int(shared.sum()))
                    if new.any(): item.update(recovered_normal_mae=float(normal_errors[new].mean()),recovered_normal_count=int(new.sum()))
            if a.save_support:
                np.savez_compressed(a.output/'support'/f'{fname}.npz',coverage=m,normal_error=normal_errors,
                                    gt_sha256=item['gt_sha256'],source_frame=frame['file_path'])
            if a.save_images:
                Image.fromarray(xb.permute(1,2,0).cpu().numpy()).save(a.output/'renders'/f'{fname}.png')
                Image.fromarray(yb.permute(1,2,0).cpu().numpy()).save(a.output/'gt'/f'{fname}.png')
                if view.original_normal is not None:
                    n0=((mesh['normal_world']+1)*127.5).clamp(0,255).byte().cpu().numpy()
                    n1=(view.original_normal.permute(1,2,0)*255).clamp(0,255).byte().cpu().numpy()
                    Image.fromarray(np.concatenate((n0,n1),axis=1)).save(a.output/'normal_comparison'/f'{fname}.png')
                Image.fromarray(np.concatenate((m,gt),axis=1).astype(np.uint8)*255).save(a.output/'coverage'/f'{fname}.png')
            if not rows: print('MESH',len(mesh['mesh_verts']),len(mesh['mesh_faces']),flush=True)
            rows.append(item)
            print('EVAL',a.split,view.colmap_id,item['boundary_f2'],flush=True)
            del mesh
    keys=set().union(*(r.keys() for r in rows))-{'source_id','source_frame','gt_sha256'}
    result=dict(split=a.split,limited=bool(a.limit),count=len(rows),checkpoint_sha256=sha(checkpoint),domain=domain,
        paired_reference=str(a.reference) if a.reference else None,
        metrics={k:float(np.mean([r[k] for r in rows if k in r])) for k in sorted(keys)},views=rows,
        note='SSAA2; PSNR/SSIM and standard LPIPS-VGG [-1,1]; raw world normal errors; empty predicted boundary assigned image diagonal')
    (a.output/'evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__': main()
