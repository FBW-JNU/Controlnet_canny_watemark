"""ROBIN-style FFT watermark utilities.

This keeps the transferable part of ROBIN: annular FFT mask generation,
ring-pattern construction, complex-domain injection, and L1/cosine detection.
It intentionally does not include adversarial prompt-embedding optimization.
"""

from dataclasses import dataclass
import copy
import random
from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ROBINWatermarkConfig:
    """Configuration for a ROBIN-style FFT latent watermark."""

    seed: int = 999999
    channel: int = 0
    low_radius: int = 5
    up_radius: int = 30
    mask_shape: Literal["circle", "square", "none"] = "circle"
    pattern: Literal["ring", "rand", "zeros", "const", "seed_rand", "seed_zeros", "seed_ring", "checkerboard"] = "ring"
    injection: Literal["complex", "seed"] = "complex"
    measurement: Literal["l1_complex", "l1_seed", "cosim_complex", "cosim_seed"] = "l1_complex"
    pattern_const: float = 0.0
    pattern_normalization: Literal["none", "mask_rms"] = "none"
    pattern_rms: float = 0.0
    enforce_hermitian: bool = False
    strength: float = 1.0
    spatial_window: Literal["none", "hann", "gaussian"] = "none"
    spatial_window_floor: float = 0.0


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + 1)
        torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    random.seed(seed + 4)


def circle_mask(
    size: int,
    r_max: int,
    r_min: int = 0,
    x_offset: int = 0,
    y_offset: int = 0,
    fft_centered: bool = False,
) -> np.ndarray:
    """Return a centered annular mask in FFT-shifted coordinates."""

    x0 = size // 2 + x_offset
    y0 = size // 2 + y_offset
    y, x = np.ogrid[:size, :size]
    if not fft_centered:
        # Historical ROBIN behavior. V8-H opts into the correctly centered
        # FFT mask without changing reproducibility of existing experiments.
        y = y[::-1]
    radius_sq = (x - x0) ** 2 + (y - y0) ** 2
    return (radius_sq <= r_max**2) & (radius_sq > r_min**2)


def _as_complex_input(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_complex():
        return tensor.to(torch.complex64)
    return tensor.float()


def project_hermitian_spectrum(spectrum: torch.Tensor) -> torch.Tensor:
    """Orthogonally project an FFT-shifted spectrum onto real-signal spectra."""

    spatial = torch.fft.ifft2(
        torch.fft.ifftshift(spectrum, dim=(-1, -2)), dim=(-1, -2)
    ).real
    return torch.fft.fftshift(
        torch.fft.fft2(spatial, dim=(-1, -2)), dim=(-1, -2)
    )


def build_watermark_mask(latents: torch.Tensor, config: ROBINWatermarkConfig) -> torch.BoolTensor:
    """Build a boolean mask with the same shape as `latents`."""

    mask = torch.zeros(latents.shape, dtype=torch.bool, device=latents.device)
    latent_size = latents.shape[-1]

    if config.mask_shape == "circle":
        np_mask = circle_mask(
            latent_size,
            r_max=config.up_radius,
            r_min=config.low_radius,
            fft_centered=config.enforce_hermitian,
        )
        spatial_mask = torch.as_tensor(np_mask, dtype=torch.bool, device=latents.device)
        if config.channel == -1:
            mask[:, :] = spatial_mask
        else:
            mask[:, config.channel] = spatial_mask
    elif config.mask_shape == "square":
        anchor = latent_size // 2
        r = config.up_radius
        if config.channel == -1:
            mask[:, :, anchor - r : anchor + r, anchor - r : anchor + r] = True
        else:
            mask[:, config.channel, anchor - r : anchor + r, anchor - r : anchor + r] = True
    elif config.mask_shape == "none":
        pass
    else:
        raise ValueError(f"Unsupported mask_shape: {config.mask_shape}")

    return mask


def build_watermark_pattern(
    shape: Tuple[int, ...],
    config: ROBINWatermarkConfig,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """Create a ROBIN/tree-ring watermark pattern.

    For complex injection modes, the returned tensor is in FFT-shifted frequency
    space. For seed injection modes, it is in latent space.
    """

    device = device or "cpu"
    set_random_seed(config.seed)
    gt_init = torch.randn(*shape, device=device, dtype=torch.float32)

    if config.pattern == "seed_zeros":
        return gt_init * 0
    if config.pattern == "seed_rand":
        return gt_init
    if config.pattern == "seed_ring":
        gt_patch = gt_init.clone()
        gt_patch_tmp = gt_patch.clone()
        annulus = torch.as_tensor(
            circle_mask(shape[-1], r_max=config.up_radius, r_min=config.low_radius),
            dtype=torch.bool,
            device=device,
        )
        for radius in range(config.up_radius, config.low_radius, -1):
            for channel in range(gt_patch.shape[1]):
                gt_patch[:, channel, annulus] = gt_patch_tmp[0, channel, 0, radius].item()
        return gt_patch

    gt_fft = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

    if config.pattern == "rand":
        gt_fft[:] = gt_fft[0]
    elif config.pattern == "zeros":
        gt_fft = gt_fft * 0
    elif config.pattern == "const":
        gt_fft = gt_fft * 0 + config.pattern_const
    elif config.pattern == "ring":
        gt_patch_tmp = copy.deepcopy(gt_fft)
        for radius in range(config.up_radius, config.low_radius, -1):
            ring = torch.as_tensor(
                circle_mask(
                    shape[-1],
                    r_max=radius,
                    r_min=config.low_radius,
                    fft_centered=config.enforce_hermitian,
                ),
                dtype=torch.bool,
                device=device,
            )
            for channel in range(gt_fft.shape[1]):
                gt_fft[:, channel, ring] = gt_patch_tmp[0, channel, 0, radius].item()
    elif config.pattern == "checkerboard":
        # Alternating +/-1 grid directly in FFT-shifted frequency space.
        # Same mask (annular ring) and same injection as ring/rand; only the
        # target's spatial structure changes (periodic grid vs concentric vs random).
        B, C, H, W = shape
        yy = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1)
        xx = torch.arange(W, device=device, dtype=torch.float32).view(1, -1)
        grid = (((yy + xx) % 2).to(torch.float32) * 2.0 - 1.0)   # +/-1, shape (H, W)
        gt_fft = grid.unsqueeze(0).unsqueeze(0).expand(B, C, H, W).contiguous().clone()
    else:
        raise ValueError(f"Unsupported pattern: {config.pattern}")

    # V8-H first makes the target physically realizable, then normalizes the
    # effective target energy. Reversing this order changes the post-projection RMS.
    if config.enforce_hermitian:
        gt_fft = project_hermitian_spectrum(gt_fft)

    if config.pattern_normalization == "mask_rms":
        # FFT coefficients of an HxW unit-variance latent have expected RMS
        # sqrt(H*W). Equalizing every key to that scale preserves the
        # historical injection magnitude while removing seed-specific energy.
        mask = build_watermark_mask(
            torch.empty(shape, device=device), config
        )
        values = gt_fft[mask]
        if values.numel() == 0:
            raise ValueError("mask_rms normalization requires a non-empty mask")
        observed_rms = values.abs().square().mean().sqrt().clamp_min(1e-12)
        target_rms = (
            float(config.pattern_rms)
            if config.pattern_rms > 0
            else float((shape[-2] * shape[-1]) ** 0.5)
        )
        gt_fft = gt_fft * (target_rms / observed_rms)
    elif config.pattern_normalization != "none":
        raise ValueError(
            f"Unsupported pattern_normalization: {config.pattern_normalization}"
        )

    return gt_fft


def build_spatial_window(latents: torch.Tensor, window_type: str, floor: float = 0.0) -> torch.Tensor:
    """Build a center-weighted spatial window for watermark deltas."""

    if window_type == "none":
        return torch.ones((1, 1, latents.shape[-2], latents.shape[-1]), device=latents.device, dtype=latents.dtype)
    if window_type == "hann":
        h = torch.hann_window(latents.shape[-2], periodic=False, device=latents.device, dtype=latents.dtype)
        w = torch.hann_window(latents.shape[-1], periodic=False, device=latents.device, dtype=latents.dtype)
        window = torch.outer(h, w)
        window = window / window.max().clamp_min(1e-8)
        window = window.view(1, 1, latents.shape[-2], latents.shape[-1])
        return floor + (1.0 - floor) * window
    if window_type == "gaussian":
        y = torch.linspace(-1.0, 1.0, latents.shape[-2], device=latents.device, dtype=latents.dtype)
        x = torch.linspace(-1.0, 1.0, latents.shape[-1], device=latents.device, dtype=latents.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        window = torch.exp(-2.0 * (xx.pow(2) + yy.pow(2)))
        window = window / window.max().clamp_min(1e-8)
        window = window.view(1, 1, latents.shape[-2], latents.shape[-1])
        return floor + (1.0 - floor) * window
    raise ValueError(f"Unsupported spatial_window: {window_type}")


def inject_watermark(
    latents: torch.Tensor,
    mask: torch.BoolTensor,
    pattern: torch.Tensor,
    config: ROBINWatermarkConfig,
) -> torch.Tensor:
    """Inject a watermark into latent or latent FFT coefficients."""

    if config.injection == "seed":
        out = latents.clone()
        target = pattern.to(out.dtype)
        out[mask] = (1.0 - config.strength) * out[mask] + config.strength * target[mask].clone()
        if config.spatial_window != "none":
            window = build_spatial_window(out, config.spatial_window, config.spatial_window_floor)
            out = latents + (out - latents) * window
        return out

    if config.injection != "complex":
        raise ValueError(f"Unsupported injection: {config.injection}")

    orig_dtype = latents.dtype
    latents_fft = torch.fft.fftshift(torch.fft.fft2(latents.float()), dim=(-1, -2))
    target = pattern.to(latents_fft.dtype)
    latents_fft[mask] = (1.0 - config.strength) * latents_fft[mask] + config.strength * target[mask].clone()
    watermarked = torch.fft.ifft2(torch.fft.ifftshift(latents_fft, dim=(-1, -2))).real
    if config.spatial_window != "none":
        window = build_spatial_window(watermarked, config.spatial_window, config.spatial_window_floor)
        watermarked = latents.float() + (watermarked - latents.float()) * window
    return watermarked.to(orig_dtype)


def watermark_distance(
    latents: torch.Tensor,
    mask: torch.BoolTensor,
    pattern: torch.Tensor,
    config: ROBINWatermarkConfig,
    mask_for_pattern: Optional[torch.BoolTensor] = None,
) -> float:
    """Measure distance/similarity between a latent and the watermark pattern."""

    mask_for_pattern = mask if mask_for_pattern is None else mask_for_pattern
    if "complex" in config.measurement:
        observed = torch.fft.fftshift(torch.fft.fft2(_as_complex_input(latents)), dim=(-1, -2))
        target = pattern
    elif "seed" in config.measurement:
        observed = latents
        target = pattern
    else:
        raise ValueError(f"Unsupported measurement: {config.measurement}")

    observed_values = observed[mask]
    target_values = target.to(observed_values.dtype)[mask_for_pattern]

    if "l1" in config.measurement:
        return torch.abs(observed_values - target_values).mean().item()
    if "cosim" in config.measurement:
        return F.cosine_similarity(observed_values.real, target_values.real, dim=0).mean().item()

    raise ValueError(f"Unsupported measurement: {config.measurement}")


def compare_watermark(
    latent_no_wm: torch.Tensor,
    latent_wm: torch.Tensor,
    mask: torch.BoolTensor,
    pattern: torch.Tensor,
    config: ROBINWatermarkConfig,
) -> Tuple[float, float]:
    """Return `(no_watermark_metric, watermark_metric)` for matched latents."""

    return (
        watermark_distance(latent_no_wm, mask, pattern, config),
        watermark_distance(latent_wm, mask, pattern, config),
    )
