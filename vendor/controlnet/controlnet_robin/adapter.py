"""Trainable UNet middle/up residual adapters for ROBIN-style watermarks."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import timestep_embedding


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class FeatureResidualAdapter(nn.Module):
    """Small residual block injected into an existing UNet feature tensor."""

    def __init__(self, channels: int, bottleneck: int = 64, alpha_init: float = 0.01):
        super().__init__()
        hidden = min(max(16, bottleneck), channels)
        self.net = nn.Sequential(
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.net(x)

    def regularization(self) -> torch.Tensor:
        return self.alpha.pow(2)


class WatermarkAdapter(nn.Module):
    """Adapters for UNet middle feature and selected output/up blocks."""

    def __init__(
        self,
        output_blocks: Optional[List[int]] = None,
        bottleneck: int = 64,
        alpha_init: float = 0.01,
    ):
        super().__init__()
        self.output_blocks = output_blocks if output_blocks is not None else list(range(12))
        self.bottleneck = bottleneck
        self.alpha_init = alpha_init
        self.middle = nn.ModuleDict()
        self.up = nn.ModuleDict()

    def ensure_middle(self, channels: int) -> None:
        key = str(channels)
        if key not in self.middle:
            self.middle[key] = FeatureResidualAdapter(channels, self.bottleneck, self.alpha_init)

    def ensure_up(self, block_idx: int, channels: int) -> None:
        if block_idx not in self.output_blocks:
            return
        key = f"{block_idx}:{channels}"
        if key not in self.up:
            self.up[key] = FeatureResidualAdapter(channels, self.bottleneck, self.alpha_init)

    def apply_middle(self, h: torch.Tensor) -> torch.Tensor:
        self.ensure_middle(h.shape[1])
        return self.middle[str(h.shape[1])](h)

    def apply_up(self, block_idx: int, h: torch.Tensor) -> torch.Tensor:
        if block_idx not in self.output_blocks:
            return h
        self.ensure_up(block_idx, h.shape[1])
        return self.up[f"{block_idx}:{h.shape[1]}"](h)

    def regularization(self, device: torch.device) -> torch.Tensor:
        reg = torch.zeros((), device=device)
        for module in list(self.middle.values()) + list(self.up.values()):
            reg = reg + module.regularization()
        return reg


def build_adapter_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    output_blocks: Optional[List[int]] = None,
    bottleneck: int = 64,
    alpha_init: float = 0.01,
) -> WatermarkAdapter:
    """Rebuild adapter submodules from a saved state dict."""

    adapter = WatermarkAdapter(output_blocks=output_blocks, bottleneck=bottleneck, alpha_init=alpha_init)
    for key in state_dict:
        parts = key.split(".")
        if len(parts) < 3:
            continue
        if parts[0] == "middle":
            adapter.ensure_middle(int(parts[1]))
        elif parts[0] == "up":
            block_idx, channels = parts[1].split(":")
            adapter.ensure_up(int(block_idx), int(channels))
    adapter.load_state_dict(state_dict)
    return adapter


def _unet_feature_shapes(
    diffusion_model,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    control: Optional[List[torch.Tensor]],
    only_mid_control: bool,
) -> Tuple[int, Dict[int, int]]:
    """Run a no-grad dry pass and return adapter channel sizes."""

    up_shapes: Dict[int, int] = {}
    hs = []
    with torch.no_grad():
        t_emb = timestep_embedding(timesteps, diffusion_model.model_channels, repeat_only=False)
        emb = diffusion_model.time_embed(t_emb)
        h = x.type(diffusion_model.dtype)
        for module in diffusion_model.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = diffusion_model.middle_block(h, emb, context)
        middle_channels = h.shape[1]

        if control is not None:
            ctrl_idx = len(control) - 1
            h = h + control[ctrl_idx]
            ctrl_idx -= 1

        for block_idx, module in enumerate(diffusion_model.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control[ctrl_idx]], dim=1)
                ctrl_idx -= 1
            h = module(h, emb, context)
            up_shapes[block_idx] = h.shape[1]

    return middle_channels, up_shapes


def initialize_adapter(
    adapter: WatermarkAdapter,
    diffusion_model,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    control: Optional[List[torch.Tensor]],
    only_mid_control: bool,
) -> None:
    """Materialize adapter submodules before constructing the optimizer."""

    middle_channels, up_shapes = _unet_feature_shapes(
        diffusion_model, x, timesteps, context, control, only_mid_control
    )
    adapter.ensure_middle(middle_channels)
    for block_idx, channels in up_shapes.items():
        adapter.ensure_up(block_idx, channels)
    adapter.to(x.device)


def apply_unet_with_adapter(
    diffusion_model,
    adapter: WatermarkAdapter,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    control: Optional[List[torch.Tensor]],
    only_mid_control: bool,
) -> torch.Tensor:
    """Mirror `ControlledUnetModel.forward` and inject adapter residuals.

    The original input path remains no-grad, matching ControlNet's locked SD
    behavior. Gradients flow through the output/up path and adapter parameters.
    """

    hs = []
    with torch.no_grad():
        t_emb = timestep_embedding(timesteps, diffusion_model.model_channels, repeat_only=False)
        emb = diffusion_model.time_embed(t_emb)
        h = x.type(diffusion_model.dtype)
        for module in diffusion_model.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = diffusion_model.middle_block(h, emb, context)

    h = adapter.apply_middle(h)

    if control is not None:
        ctrl_idx = len(control) - 1
        h = h + control[ctrl_idx]
        ctrl_idx -= 1

    for block_idx, module in enumerate(diffusion_model.output_blocks):
        if only_mid_control or control is None:
            h = torch.cat([h, hs.pop()], dim=1)
        else:
            h = torch.cat([h, hs.pop() + control[ctrl_idx]], dim=1)
            ctrl_idx -= 1
        h = module(h, emb, context)
        h = adapter.apply_up(block_idx, h)

    h = h.type(x.dtype)
    return diffusion_model.out(h)


def fft_ring_loss(latents: torch.Tensor, mask: torch.BoolTensor, pattern: torch.Tensor) -> torch.Tensor:
    latents_fft = torch.fft.fftshift(torch.fft.fft2(latents.float()), dim=(-1, -2))
    target = pattern.to(latents_fft.dtype)
    return torch.abs(latents_fft[mask] - target[mask]).mean()


def progressive_fft_ring_loss(
    latents: torch.Tensor,
    clean_latents: torch.Tensor,
    mask: torch.BoolTensor,
    pattern: torch.Tensor,
    progress: float,
) -> torch.Tensor:
    """Match a partial ring target between clean FFT and the full watermark."""

    latents_fft = torch.fft.fftshift(torch.fft.fft2(latents.float()), dim=(-1, -2))
    with torch.no_grad():
        clean_fft = torch.fft.fftshift(torch.fft.fft2(clean_latents.float()), dim=(-1, -2))
        target = clean_fft.clone()
        pattern = pattern.to(target.dtype)
        target[mask] = (1.0 - progress) * clean_fft[mask] + progress * pattern[mask]
    return torch.abs(latents_fft[mask] - target[mask]).mean()


def ddim_prev_from_eps(ddim, x: torch.Tensor, eps: torch.Tensor, alpha_index: int) -> torch.Tensor:
    b = x.shape[0]
    device = x.device
    a_t = torch.full((b, 1, 1, 1), ddim.ddim_alphas[alpha_index], device=device)
    a_prev = torch.full((b, 1, 1, 1), ddim.ddim_alphas_prev[alpha_index], device=device)
    sigma_t = torch.full((b, 1, 1, 1), ddim.ddim_sigmas[alpha_index], device=device)
    sqrt_one_minus_at = torch.full((b, 1, 1, 1), ddim.ddim_sqrt_one_minus_alphas[alpha_index], device=device)

    pred_x0 = (x - sqrt_one_minus_at * eps) / a_t.sqrt()
    dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * eps
    return a_prev.sqrt() * pred_x0 + dir_xt


def latent_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(a.float(), b.float(), reduction="mean")
