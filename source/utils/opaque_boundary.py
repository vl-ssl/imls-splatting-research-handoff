"""Geometry-only foreground boundary supervision for opaque reflective objects."""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class BoundaryTarget:
    mask: torch.Tensor
    reliable: torch.Tensor
    distance: torch.Tensor
    crossings: torch.Tensor


def prepare_target(alpha):
    """Construct a signed pixel distance and subpixel 0.5-alpha crossings."""
    if alpha is None:
        return None
    from scipy.ndimage import distance_transform_edt

    a = alpha.detach().float().cpu().squeeze().numpy()
    if a.ndim != 2 or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError('Invalid opaque foreground coverage target')
    mask = a >= .5
    if not mask.any() or mask.all():
        return None
    distance = np.where(mask, .5 - distance_transform_edt(mask), distance_transform_edt(~mask) - .5)
    y, x = np.where(mask[:, 1:] != mask[:, :-1])
    fraction = (.5 - a[y, x]) / (a[y, x+1] - a[y, x])
    horizontal = np.stack([x + fraction, y], axis=1)
    y, x = np.where(mask[1:] != mask[:-1])
    fraction = (.5 - a[y, x]) / (a[y+1, x] - a[y, x])
    vertical = np.stack([x, y + fraction], axis=1)
    crossings = np.concatenate([horizontal, vertical])
    h, w = mask.shape
    valid = ((crossings > 1).all(axis=1) & (crossings[:, 0] < w-2) & (crossings[:, 1] < h-2))
    return BoundaryTarget(torch.from_numpy(mask), torch.from_numpy((a <= .05) | (a >= .95)),
                          torch.from_numpy(distance.astype(np.float32)),
                          torch.from_numpy(crossings[valid].astype(np.float32)))


def pixel_coordinates(ndc, height, width):
    return (ndc + 1) * ndc.new_tensor([width, height]) * .5 - .5


def visible_segments(verts_clip, faces, rast, max_segments=4096):
    """Select visible subsegments of true triangle edges, not all boundary-face vertices.

    Selection is discrete. A segment must have its own first-hit triangle on
    the inside and background on the outside, at three locations along it.
    This rejects interior triangle edges and foreground-on-foreground occlusion.
    """
    vertices = verts_clip.reshape(-1, 4)
    empty = vertices.new_empty((0, 2, 2))
    with torch.no_grad():
        ids = rast[0, ..., 3].long()
        h, w = ids.shape
        hit = ids > 0
        near_background = F.max_pool2d((~hit)[None, None].float(), 3, 1, 1)[0, 0] > 0
        boundary = hit & near_background
        boundary[[0, -1], :] = False
        boundary[:, [0, -1]] = False
        face_ids = torch.unique(ids[boundary]) - 1
        if not face_ids.numel():
            return empty
        if len(face_ids) > max_segments:
            face_ids = face_ids[torch.linspace(0, len(face_ids)-1, max_segments, device=ids.device).long()]
        triangles = faces.long()[face_ids]
        edge_vertices = triangles[:, [[0, 1, 2], [1, 2, 0], [2, 0, 1]]].reshape(-1, 3)
        owners = (face_ids + 1).repeat_interleave(3)
        clips = vertices.detach()[edge_vertices]
        front = (clips[..., 3] > 1e-6).all(dim=1)
        edge_vertices, owners, clips = edge_vertices[front], owners[front], clips[front]
        if not len(clips):
            return empty
        points = pixel_coordinates(clips[..., :2] / clips[..., 3:4], h, w)
        a, b, third = points.unbind(dim=1)
        direction = b - a
        length = torch.linalg.vector_norm(direction, dim=-1)
        perp = torch.stack([-direction[:, 1], direction[:, 0]], dim=1) / length.clamp_min(1e-6)[:, None]
        signed_side = ((third - a) * perp).sum(dim=-1)
        inward = perp * signed_side.sign()[:, None]
        count = (length / 3).ceil().long().clamp(1, 48)
        edge_id = torch.arange(len(count), device=ids.device).repeat_interleave(count)
        index = torch.arange(len(edge_id), device=ids.device) - (count.cumsum(0) - count)[edge_id]
        t0 = index / count[edge_id]
        dt = 1 / count[edge_id]
        t = t0[:, None] + dt[:, None] * t0.new_tensor([.18, .5, .82])
        q = a[edge_id, None] + direction[edge_id, None] * t[..., None]

        def lookup(p):
            pix = p.round().long()
            x, y = pix.unbind(dim=-1)
            inside = (x > 0) & (x < w-1) & (y > 0) & (y < h-1)
            found = ids[y.clamp(0, h-1), x.clamp(0, w-1)]
            return torch.where(inside, found, -torch.ones_like(found))

        normal = inward[edge_id, None]
        valid = torch.zeros_like(t, dtype=torch.bool)
        for offset in (.65, 1.1):
            valid |= ((lookup(q + offset * normal) == owners[edge_id, None]) &
                      (lookup(q - offset * normal) == 0))
        keep = torch.where(valid.all(dim=1) & (length[edge_id] > .2) & (signed_side[edge_id].abs() > .15))[0]
        if len(keep) > max_segments:
            keep = keep[torch.linspace(0, len(keep)-1, max_segments, device=ids.device).long()]
        if not len(keep):
            return empty
        pair = edge_vertices[edge_id[keep], :2]
        t = t[keep][:, [0, 2]].to(vertices.dtype)
    endpoints = vertices[pair]
    projected = endpoints[..., :2] / endpoints[..., 3:4]
    return torch.lerp(projected[:, :1], projected[:, 1:], t[..., None])


def point_segment_distance(points, segments):
    """Each point to its paired segment. No sqrt singularity at zero distance."""
    start, end = segments.unbind(dim=-2)
    direction = end - start
    offset = points - start
    t = ((offset * direction).sum(dim=-1) / direction.square().sum(dim=-1).clamp_min(1e-12)).clamp(0, 1)
    return torch.linalg.vector_norm(offset - t[..., None] * direction, dim=-1)


def boundary_terms(mesh, target, use_region=True, use_edges=True, max_samples=2048):
    vertices = mesh['verts_clip']
    zero = vertices.sum() * 0
    if target is None:
        return zero, zero, 0
    alpha = mesh['alpha'].squeeze(-1)
    h, w = alpha.shape
    if target.mask.shape != alpha.shape:
        raise ValueError('GT and rendered boundary dimensions differ')
    radius = max(1., 16. * min(h, w) / 800.)
    distance = target.distance.to(alpha)
    region = zero
    if use_region:
        mask, reliable = target.mask.to(alpha.device), target.reliable.to(alpha.device)
        loss = (alpha - mask.to(alpha.dtype)) * (distance / radius).clamp(-1, 1)
        inside, outside = reliable & mask, reliable & ~mask
        region = .5 * ((loss * inside).sum() / inside.sum().clamp_min(1) +
                       (loss * outside).sum() / outside.sum().clamp_min(1))
    if not use_edges or not len(target.crossings):
        return region, zero, 0
    segments = visible_segments(vertices, mesh['mesh_faces'], mesh['rast'], max_samples)
    if not len(segments):
        return region, zero, 0
    midpoints = segments.mean(dim=1)
    d = F.grid_sample(distance[None, None], midpoints.reshape(1, 1, -1, 2),
                      align_corners=False, padding_mode='border').reshape(-1)
    pixel_segments = pixel_coordinates(segments, h, w)
    weight = torch.linalg.vector_norm(pixel_segments[:, 1] - pixel_segments[:, 0], dim=1).detach()
    pred_to_gt = (F.smooth_l1_loss(d / radius, torch.zeros_like(d), reduction='none', beta=1/radius) * weight).sum() / weight.sum().clamp_min(1e-6)
    target_points = target.crossings.to(alpha)
    if len(target_points) > max_samples:
        target_points = target_points[torch.linspace(0, len(target_points)-1, max_samples, device=alpha.device).long()]
    with torch.no_grad():
        nearest = torch.cat([point_segment_distance(p[:, None], pixel_segments[None]).argmin(dim=1)
                             for p in target_points.split(128)])
    d = point_segment_distance(target_points, pixel_segments[nearest]) / radius
    gt_to_pred = F.smooth_l1_loss(d, torch.zeros_like(d), beta=1/radius)
    return region, .5 * (pred_to_gt + gt_to_pred), len(segments)
