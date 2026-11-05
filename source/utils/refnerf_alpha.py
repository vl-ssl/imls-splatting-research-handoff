"""Explicit masks for RGB Ref-NeRF exports; never infer object masks from RGB."""
from pathlib import Path
import numpy as np
from PIL import Image


def read_refnerf_image(path):
    """Return RGBA, optional sidecar mask, and known precomposited background.

    Native alpha takes precedence. RGB + *_alpha.png is supported only when
    its fully transparent pixels confirm a white precomposited background.
    Preserve RGB bytes: attaching alpha and compositing again would darken edges.
    """
    path = Path(path)
    with Image.open(path) as source:
        native_alpha = "A" in source.getbands() or "transparency" in source.info
        rgba = source.convert("RGBA")
    sidecar = path.with_name(path.stem + "_alpha.png")
    if native_alpha or not sidecar.is_file():
        return rgba, None, None
    with Image.open(sidecar) as source:
        mask = source.convert("L")
    if mask.size != rgba.size:
        raise ValueError(f"Alpha sidecar size mismatch: {sidecar}")
    outside = np.asarray(mask) == 0
    rgb = np.asarray(rgba)[..., :3]
    # Allow one quantization level where independently rounded alpha is zero.
    if not outside.any() or not np.all(rgb[outside] >= 254):
        raise ValueError(f"Cannot verify RGB sidecar white background: {path}")
    return rgba, mask, (1.0, 1.0, 1.0)


def composite_reference(image, alpha, background, original_background=None):
    """CHW tensors; retain precomposited white GT exactly on white evaluation."""
    if alpha is None:
        return image
    bg = background.to(device=image.device, dtype=image.dtype).reshape(3, 1, 1)
    alpha = alpha.to(device=image.device, dtype=image.dtype)
    if original_background is None:
        return image * alpha + bg * (1 - alpha)
    old_bg = image.new_tensor(original_background).reshape(3, 1, 1)
    return (image + (bg - old_bg) * (1 - alpha)).clamp(0, 1)
