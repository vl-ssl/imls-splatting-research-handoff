"""Keep a centered IMLS extraction lattice aligned when extending its bounds."""
import math


def aligned_mesh_domain(resolution, meshscale, required_scale):
    """Return a larger cube with unchanged voxel spacing and grid parity.

    With a centered grid, changing parity shifts every old sample by half a
    voxel even if spacing is fixed. The returned lattice contains the old
    lattice at an integer index offset.
    """
    resolution = int(resolution)
    meshscale = float(meshscale)
    required_scale = float(required_scale)
    if resolution <= 1 or meshscale <= 0 or not math.isfinite(required_scale):
        raise ValueError("Invalid extraction domain")
    spacing = meshscale / (resolution - 1)
    new_resolution = max(resolution, math.ceil(required_scale / spacing) + 1)
    if (new_resolution - resolution) % 2:
        new_resolution += 1
    return new_resolution, (new_resolution - 1) * spacing
