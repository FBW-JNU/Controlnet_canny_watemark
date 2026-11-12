"""
STDM watermark injection for ControlNet feature maps.

This module implements the "middle + down blocks only" injection path:

    f_block = ControlNet_block(x, conditioning)
    f_wm = STDM_embed(f_block, watermark_bits, key)
    output = zero_conv(f_wm)

The default target outputs are the three downsample transition blocks plus the
middle block in the original ControlNet/SD1.5 layout:

    3  -> down1 transition
    6  -> down2 transition
    9  -> down3 transition
    12 -> middle block

The implementation is intentionally standalone. Existing configs can keep using
cldm.cldm.ControlNet, while STDM experiments can point control_stage_config to
cldm.stdm_controlnet.STDMControlNet.
"""

from dataclasses import dataclass
import hashlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import timestep_embedding
from cldm.cldm import ControlNet


DEFAULT_STDM_TARGET_OUTPUTS: Tuple[int, ...] = (3, 6, 9, 12)


@dataclass
class STDMConfig:
    """Configuration for spread-transform dither modulation.

    Args:
        num_bits: Number of message bits embedded in each selected feature map.
        delta: Quantization step size in projection space.
        strength: Multiplier for the STDM correction before it is added back.
        use_ste: Use a straight-through estimator so gradients through the
            feature map behave like identity during ControlNet fine-tuning.
        max_cache_entries: Maximum number of carrier/dither tensors cached per
            embedder instance.
    """

    num_bits: int = 16
    delta: float = 0.08
    strength: float = 1.0
    use_ste: bool = True
    max_cache_entries: int = 32


def _stable_seed(key: Union[str, int], block_id: int, shape: Sequence[int], salt: str) -> int:
    payload = f"{key}|{block_id}|{tuple(shape)}|{salt}".encode("utf-8")
    digest = hashlib.blake2s(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2 ** 31)


def _as_bits(
    watermark_bits: torch.Tensor,
    batch_size: int,
    num_bits: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    bits = watermark_bits.to(device=device, dtype=dtype)
    if bits.dim() == 1:
        bits = bits.unsqueeze(0).expand(batch_size, -1)
    if bits.dim() != 2:
        raise ValueError(f"watermark_bits must be [K] or [B, K], got {tuple(bits.shape)}")
    if bits.shape[0] != batch_size:
        raise ValueError(f"watermark_bits batch mismatch: expected {batch_size}, got {bits.shape[0]}")
    if bits.shape[1] != num_bits:
        raise ValueError(f"watermark_bits width mismatch: expected {num_bits}, got {bits.shape[1]}")
    return (bits > 0.5).to(dtype=dtype)


class STDMEmbedder(nn.Module):
    """Spread-transform dither modulation for dense feature maps.

    Each bit gets a deterministic pseudo-random unit carrier vector. The feature
    map is projected onto these carriers, each scalar projection is quantized to
    the bit-specific coset, and the correction is projected back to feature
    space.
    """

    def __init__(self, config: Optional[STDMConfig] = None):
        super().__init__()
        self.config = config if config is not None else STDMConfig()
        self._cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def num_bits(self) -> int:
        return self.config.num_bits

    def clear_cache(self) -> None:
        self._cache.clear()

    def _carrier_cache_key(
        self,
        feature: torch.Tensor,
        key: Union[str, int],
        block_id: int,
    ) -> Tuple:
        return (
            str(key),
            int(block_id),
            tuple(feature.shape[1:]),
            str(feature.device),
            str(feature.dtype),
            self.config.num_bits,
            float(self.config.delta),
        )

    def _get_carriers_and_dither(
        self,
        feature: torch.Tensor,
        key: Union[str, int],
        block_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = self._carrier_cache_key(feature, key, block_id)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        shape = feature.shape[1:]
        flat_dim = int(torch.tensor(shape).prod().item())

        carrier_seed = _stable_seed(key, block_id, shape, "carrier")
        carrier_gen = torch.Generator(device="cpu")
        carrier_gen.manual_seed(carrier_seed)
        carriers = torch.randn(self.config.num_bits, flat_dim, generator=carrier_gen)
        carriers = F.normalize(carriers, p=2, dim=1, eps=1e-12)
        carriers = carriers.to(device=feature.device, dtype=feature.dtype)

        dither_seed = _stable_seed(key, block_id, shape, "dither")
        dither_gen = torch.Generator(device="cpu")
        dither_gen.manual_seed(dither_seed)
        dither = torch.rand(self.config.num_bits, generator=dither_gen)
        dither = (dither - 0.5) * float(self.config.delta)
        dither = dither.to(device=feature.device, dtype=feature.dtype)

        if len(self._cache) >= self.config.max_cache_entries:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = (carriers, dither)
        return carriers, dither

    def quantize_projection(self, projection: torch.Tensor, bits: torch.Tensor, dither: torch.Tensor) -> torch.Tensor:
        delta = projection.new_tensor(float(self.config.delta))
        bit_offset = bits * (delta * 0.5)
        dither = dither.unsqueeze(0)
        return torch.round((projection - dither - bit_offset) / delta) * delta + dither + bit_offset

    def forward(
        self,
        feature: torch.Tensor,
        watermark_bits: Optional[torch.Tensor],
        key: Union[str, int] = 0,
        block_id: int = 0,
    ) -> torch.Tensor:
        if watermark_bits is None:
            return feature

        batch_size = feature.shape[0]
        bits = _as_bits(watermark_bits, batch_size, self.config.num_bits, feature.device, feature.dtype)
        carriers, dither = self._get_carriers_and_dither(feature, key, block_id)

        flat = feature.reshape(batch_size, -1).float()
        carriers_f = carriers.float()
        projection = flat.matmul(carriers_f.t()).to(dtype=feature.dtype)
        quantized = self.quantize_projection(projection, bits, dither)
        projection_delta = (quantized - projection).float()
        gram = carriers_f.matmul(carriers_f.t())
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + eye * 1e-6
        correction_coeff = torch.linalg.solve(gram, projection_delta.t()).t()
        correction = correction_coeff.matmul(carriers_f)
        correction = correction.reshape_as(feature).to(dtype=feature.dtype)

        embedded = feature + float(self.config.strength) * correction
        if self.config.use_ste:
            return feature + (embedded - feature).detach()
        return embedded

    @torch.no_grad()
    def decode_from_feature(
        self,
        feature: torch.Tensor,
        key: Union[str, int] = 0,
        block_id: int = 0,
    ) -> torch.Tensor:
        """Estimate bits directly from a selected ControlNet feature map.

        This is mainly a diagnostic helper. The primary extraction path should
        still use DDIM inversion -> latent -> tuned VAE decoder/extractor.
        """
        carriers, dither = self._get_carriers_and_dither(feature, key, block_id)
        flat = feature.reshape(feature.shape[0], -1).float()
        projection = flat.matmul(carriers.float().t()).to(dtype=feature.dtype)
        delta = projection.new_tensor(float(self.config.delta))
        phase = torch.remainder(projection - dither.unsqueeze(0), delta)
        dist_0 = torch.minimum(phase, delta - phase)
        dist_1 = torch.abs(phase - delta * 0.5)
        return (dist_1 < dist_0).float()


class STDMControlNet(ControlNet):
    """ControlNet with STDM applied before selected zero-conv outputs."""

    def __init__(
        self,
        *args,
        stdm_num_bits: int = 16,
        stdm_delta: float = 0.08,
        stdm_strength: float = 1.0,
        stdm_use_ste: bool = True,
        stdm_target_outputs: Optional[Iterable[int]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.stdm_target_outputs = tuple(stdm_target_outputs or DEFAULT_STDM_TARGET_OUTPUTS)
        self.stdm_embedder = STDMEmbedder(
            STDMConfig(
                num_bits=stdm_num_bits,
                delta=stdm_delta,
                strength=stdm_strength,
                use_ste=stdm_use_ste,
            )
        )

    def _maybe_embed(
        self,
        feature: torch.Tensor,
        output_index: int,
        watermark_bits: Optional[torch.Tensor],
        stdm_key: Union[str, int],
    ) -> torch.Tensor:
        if output_index not in self.stdm_target_outputs:
            return feature
        return self.stdm_embedder(feature, watermark_bits, key=stdm_key, block_id=output_index)

    def forward(
        self,
        x,
        hint,
        timesteps,
        context,
        watermark_bits: Optional[torch.Tensor] = None,
        stdm_key: Union[str, int] = 0,
        return_prezero_features: bool = False,
        **kwargs,
    ):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs: List[torch.Tensor] = []
        prezero_features: Dict[int, torch.Tensor] = {}

        h = x.type(self.dtype)
        for output_index, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
            if guided_hint is not None:
                h = module(h, emb, context)
                h = h + guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)

            h_for_zero = self._maybe_embed(h, output_index, watermark_bits, stdm_key)
            if return_prezero_features and output_index in self.stdm_target_outputs:
                prezero_features[output_index] = h_for_zero
            outs.append(zero_conv(h_for_zero, emb, context))

        h = self.middle_block(h, emb, context)
        middle_index = len(self.zero_convs)
        h_for_zero = self._maybe_embed(h, middle_index, watermark_bits, stdm_key)
        if return_prezero_features and middle_index in self.stdm_target_outputs:
            prezero_features[middle_index] = h_for_zero
        outs.append(self.middle_block_out(h_for_zero, emb, context))

        if return_prezero_features:
            return outs, prezero_features
        return outs


class STDMControlOutputWrapper(nn.Module):
    """Compatibility wrapper for an already-created ControlNet.

    This wrapper embeds after the original zero-conv outputs because it does not
    have access to pre-zero-conv features. Use STDMControlNet for the exact
    proposed architecture.
    """

    def __init__(
        self,
        control_model: nn.Module,
        stdm_num_bits: int = 16,
        stdm_delta: float = 0.08,
        stdm_strength: float = 1.0,
        stdm_target_outputs: Optional[Iterable[int]] = None,
    ):
        super().__init__()
        self.control_model = control_model
        self.stdm_target_outputs = tuple(stdm_target_outputs or DEFAULT_STDM_TARGET_OUTPUTS)
        self.stdm_embedder = STDMEmbedder(
            STDMConfig(num_bits=stdm_num_bits, delta=stdm_delta, strength=stdm_strength)
        )

    def forward(
        self,
        *args,
        watermark_bits: Optional[torch.Tensor] = None,
        stdm_key: Union[str, int] = 0,
        **kwargs,
    ) -> List[torch.Tensor]:
        outs = self.control_model(*args, **kwargs)
        if watermark_bits is None:
            return outs
        embedded = list(outs)
        for output_index in self.stdm_target_outputs:
            embedded[output_index] = self.stdm_embedder(
                embedded[output_index],
                watermark_bits,
                key=stdm_key,
                block_id=output_index,
            )
        return embedded


class MultiScaleImageWatermarkExtractor(nn.Module):
    """Lightweight auxiliary extractor for pixel-space verification.

    This is not the primary extraction path. It is intended for the optional
    image-space verifier described in the scheme.
    """

    def __init__(self, num_bits: int = 16, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )
        self.down1 = self._block(base_channels, base_channels * 2)
        self.down2 = self._block(base_channels * 2, base_channels * 4)
        self.down3 = self._block(base_channels * 4, base_channels * 4)
        fused_channels = base_channels + base_channels * 2 + base_channels * 4 + base_channels * 4
        self.head = nn.Sequential(
            nn.Conv2d(fused_channels, base_channels * 4, 1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 4, num_bits),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        f0 = self.stem(image)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        size = f0.shape[-2:]
        fused = torch.cat(
            [
                f0,
                F.interpolate(f1, size=size, mode="bilinear", align_corners=False),
                F.interpolate(f2, size=size, mode="bilinear", align_corners=False),
                F.interpolate(f3, size=size, mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        return torch.sigmoid(self.head(fused))


class DifferentiableAttackLayer(nn.Module):
    """Small differentiable attack layer for robustness training.

    JPEG is intentionally omitted here because the common OpenCV/PIL JPEG path is
    non-differentiable. Add a differentiable JPEG module in the training script
    if that dependency is available.
    """

    def __init__(
        self,
        noise_sigma: float = 0.01,
        resize_min: float = 0.85,
        resize_max: float = 1.15,
        crop_min: float = 0.85,
    ):
        super().__init__()
        self.noise_sigma = noise_sigma
        self.resize_min = resize_min
        self.resize_max = resize_max
        self.crop_min = crop_min

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        attacked = image
        if self.training and self.noise_sigma > 0 and torch.rand((), device=image.device) < 0.5:
            attacked = attacked + torch.randn_like(attacked) * self.noise_sigma

        if self.training and torch.rand((), device=image.device) < 0.5:
            h, w = attacked.shape[-2:]
            scale = torch.empty((), device=image.device).uniform_(self.resize_min, self.resize_max).item()
            new_h = max(8, int(h * scale))
            new_w = max(8, int(w * scale))
            attacked = F.interpolate(attacked, size=(new_h, new_w), mode="bilinear", align_corners=False)
            attacked = F.interpolate(attacked, size=(h, w), mode="bilinear", align_corners=False)

        if self.training and self.crop_min < 1.0 and torch.rand((), device=image.device) < 0.5:
            h, w = attacked.shape[-2:]
            crop_scale = torch.empty((), device=image.device).uniform_(self.crop_min, 1.0).item()
            crop_h = max(8, int(h * crop_scale))
            crop_w = max(8, int(w * crop_scale))
            top = torch.randint(0, h - crop_h + 1, (), device=image.device).item()
            left = torch.randint(0, w - crop_w + 1, (), device=image.device).item()
            attacked = attacked[..., top : top + crop_h, left : left + crop_w]
            attacked = F.interpolate(attacked, size=(h, w), mode="bilinear", align_corners=False)

        return attacked.clamp(-1.0, 1.0)


def watermark_bce_loss(bit_probs: torch.Tensor, watermark_bits: torch.Tensor) -> torch.Tensor:
    bits = watermark_bits.to(device=bit_probs.device, dtype=bit_probs.dtype)
    return F.binary_cross_entropy(bit_probs, bits)
