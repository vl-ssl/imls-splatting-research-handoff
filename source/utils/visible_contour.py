"""Long-range silhouette gradients on visible IMLS mesh vertices.

Visibility is selected from raster triangle IDs, while distances differentiate
through clip-space vertex coordinates to the extracted surface. RGB edges are
not used: reflected contours do not define opaque object boundaries.
"""
import torch
import torch.nn.functional as F


def symmetric_contour_distance(projected, target, beta=0.01):
    if projected.numel() == 0 or target.numel() == 0:
        return projected.sum() * 0
    distances = torch.cdist(projected[None], target[None])[0]
    pred_distance = distances.min(dim=1).values
    gt_distance = distances.min(dim=0).values
    return 0.5 * (
        F.smooth_l1_loss(pred_distance, torch.zeros_like(pred_distance), beta=beta)
        + F.smooth_l1_loss(gt_distance, torch.zeros_like(gt_distance), beta=beta)
    )


def visible_contour_loss(verts_clip, faces, raster, target_alpha, max_samples=2048):
    """Return loss and sample counts; selection uses predicted visibility only."""
    clip = verts_clip.reshape(-1, 4)
    zero = clip.sum() * 0
    with torch.no_grad():
        ids = raster[0, ..., 3].long()
        mask = (ids > 0).float()[None, None]
        eroded = 1 - F.max_pool2d(1 - F.pad(mask, (1, 1, 1, 1)), 3, stride=1)
        boundary = (mask[0, 0] > 0) & (eroded[0, 0] < .5)
        triangle_ids = torch.unique(ids[boundary]) - 1
        if triangle_ids.numel() == 0:
            return zero, 0, 0
        indices = torch.unique(faces.long()[triangle_ids].reshape(-1))
        indices = indices[clip.detach()[indices, 3] > 1e-6]
        if indices.numel() > max_samples:
            take = torch.linspace(0, indices.numel()-1, max_samples, device=clip.device).long()
            indices = indices[take]
        gt = (target_alpha.detach().reshape(1, 1, *target_alpha.shape[-2:]) > .5).float()
        gt_eroded = 1 - F.max_pool2d(1 - F.pad(gt, (1, 1, 1, 1)), 3, stride=1)
        yx = torch.nonzero((gt[0, 0] > 0) & (gt_eroded[0, 0] < .5))
        if yx.shape[0] > max_samples:
            take = torch.linspace(0, yx.shape[0]-1, max_samples, device=clip.device).long()
            yx = yx[take]
        h, w = gt.shape[-2:]
        target = (yx[:, [1, 0]].to(clip.dtype) + .5) / clip.new_tensor([w, h]) * 2 - 1
        # Measure in common image-height units for non-square cameras.
        aspect = clip.new_tensor([w / h, 1.])
        target = target * aspect
    projected = clip[indices, :2] / clip[indices, 3:4].clamp_min(1e-6)
    loss = symmetric_contour_distance(projected * aspect, target)
    return loss, int(indices.numel()), int(target.shape[0])
