"""Data-only helpers for the Canny completion experiment (no model or training)."""
import hashlib

import numpy as np
from PIL import Image, ImageOps


def sample_seed(seed, sample_id):
    value = f'edge-completion-mask-v1|{seed}|{sample_id}'
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little')


def patch_masks(size, patch_size, ratios, seed, *, rounding='nearest'):
    """True means missing. A common permutation makes evaluation masks nested."""
    if size % patch_size:
        raise ValueError('Image size must be divisible by patch size')
    side = size // patch_size
    count = side * side
    order = np.random.default_rng(seed).permutation(count)
    masks = []
    for ratio in ratios:
        if rounding not in {'nearest', 'floor'}:
            raise ValueError('rounding must be nearest or floor')
        missing = int(np.floor(count * ratio + 1e-12)) if rounding == 'floor' else int(round(count * ratio))
        if not 0 < missing < count:
            raise ValueError('Mask must contain both visible and missing patches')
        mask = np.zeros(count, dtype=bool)
        mask[order[:missing]] = True
        masks.append(mask.reshape(side, side))
    return np.stack(masks)


def pixel_mask(patch_mask, patch_size):
    return np.repeat(np.repeat(patch_mask, patch_size, axis=0), patch_size, axis=1)


def prepare_rgb(path, size):
    """EXIF orientation, RGB conversion, central square crop, bicubic resize."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        width, height = image.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        box = (left, top, left + side, top + side)
        rgb = np.asarray(image.crop(box).resize((size, size), Image.Resampling.BICUBIC)).copy()
    return rgb, [width, height], list(box)
