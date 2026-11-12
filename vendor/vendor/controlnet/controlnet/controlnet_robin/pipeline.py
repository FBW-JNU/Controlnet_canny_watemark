"""ControlNet pipeline wrapper for ROBIN-style latent watermarking.

This module uses the original ControlNet/CompVis stack and injects the
watermark at an intermediate DDIM latent. It deliberately omits ROBIN's
adversarial prompt optimization (`opt_acond`) and continues denoising with
normal two-branch CFG after injection.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import einops
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from cldm.ddim_hacked import DDIMSampler
from cldm.hack import hack_everything
from cldm.model import create_model, load_state_dict

from .fft_watermark import (
    ROBINWatermarkConfig,
    build_watermark_mask,
    build_watermark_pattern,
    inject_watermark,
    watermark_distance,
)
from .adapter import apply_unet_with_adapter, build_adapter_from_state_dict


@dataclass
class WatermarkArtifacts:
    mask: torch.BoolTensor
    pattern: torch.Tensor
    injected_latent: torch.Tensor


class ControlNetROBINPipeline:
    """Minimal ControlNet wrapper for middle-latent ROBIN-style watermarking."""

    def __init__(self, config_path: str, ckpt_path: str, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        hack_everything()
        self.model = create_model(config_path).cpu()
        self.model.load_state_dict(load_state_dict(ckpt_path, location=device))
        self.model = self.model.to(device)
        self.model.eval()
        self.ddim = DDIMSampler(self.model)
        self.adapter = None

    def load_adapter(self, ckpt_path: str) -> None:
        state = torch.load(ckpt_path, map_location=self.device)
        adapter_state = state["adapter"] if isinstance(state, dict) and "adapter" in state else state
        args = state.get("args", {}) if isinstance(state, dict) else {}
        output_blocks = state.get("output_blocks", None) if isinstance(state, dict) else None
        bottleneck = int(args.get("adapter_bottleneck", 64))
        alpha_init = float(args.get("adapter_alpha_init", 0.01))
        self.adapter = build_adapter_from_state_dict(adapter_state, output_blocks, bottleneck, alpha_init)
        self.adapter = self.adapter.to(self.device)
        self.adapter.eval()

    def make_schedule(self, ddim_steps: int, eta: float = 0.0) -> None:
        self.ddim.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=eta, verbose=False)
        self.time_range = np.flip(self.ddim.ddim_timesteps)
        self.total_steps = len(self.time_range)

    @torch.no_grad()
    def get_conditioning(
        self,
        prompt: str,
        hint_image: Optional[Union[Image.Image, np.ndarray]],
        negative_prompt: str = "",
        image_length: int = 512,
        guess_mode: bool = False,
    ) -> Tuple[dict, dict]:
        cond_prompt = self.model.get_learned_conditioning([prompt])
        uncond_prompt = self.model.get_learned_conditioning([negative_prompt])

        if isinstance(hint_image, Image.Image):
            hint_image = hint_image.resize((image_length, image_length), Image.LANCZOS)
            hint_np = np.array(hint_image.convert("RGB")).astype(np.float32) / 255.0
        elif isinstance(hint_image, np.ndarray):
            hint_np = hint_image.astype(np.float32)
            if hint_np.max() > 1.0:
                hint_np = hint_np / 255.0
        elif hint_image is None:
            hint_np = np.zeros((image_length, image_length, 3), dtype=np.float32)
        else:
            raise TypeError("hint_image must be a PIL image, numpy array, or None")

        hint_tensor = torch.from_numpy(hint_np).unsqueeze(0).to(self.device)
        hint_tensor = einops.rearrange(hint_tensor, "b h w c -> b c h w").contiguous().float()

        cond = {"c_concat": [hint_tensor], "c_crossattn": [cond_prompt]}
        uncond = {
            "c_concat": None if guess_mode else [hint_tensor],
            "c_crossattn": [uncond_prompt],
        }
        return cond, uncond

    @torch.no_grad()
    def _predict_eps(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: dict,
        uncond: Optional[dict],
        guidance_scale: float,
    ) -> torch.Tensor:
        if uncond is not None and guidance_scale > 1.0:
            eps_cond = self.model.apply_model(x, t, cond)
            eps_uncond = self.model.apply_model(x, t, uncond)
            return eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        return self.model.apply_model(x, t, cond)

    @torch.no_grad()
    def ddim_step(
        self,
        x: torch.Tensor,
        step_index: int,
        cond: dict,
        uncond: Optional[dict],
        guidance_scale: float,
    ) -> torch.Tensor:
        ddpm_t = self.time_range[step_index]
        alpha_index = self.total_steps - step_index - 1
        t = torch.full((x.shape[0],), ddpm_t, device=self.device, dtype=torch.long)
        e_t = self._predict_eps(x, t, cond, uncond, guidance_scale)

        alphas = self.ddim.ddim_alphas
        alphas_prev = self.ddim.ddim_alphas_prev
        sigmas = self.ddim.ddim_sigmas
        sqrt_one_minus_alphas = self.ddim.ddim_sqrt_one_minus_alphas

        b = x.shape[0]
        a_t = torch.full((b, 1, 1, 1), alphas[alpha_index], device=self.device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[alpha_index], device=self.device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[alpha_index], device=self.device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[alpha_index], device=self.device)

        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * torch.randn_like(x)
        return a_prev.sqrt() * pred_x0 + dir_xt + noise

    @torch.no_grad()
    def ddim_step_with_adapter(
        self,
        x: torch.Tensor,
        step_index: int,
        cond: dict,
        uncond: Optional[dict],
        guidance_scale: float,
    ) -> torch.Tensor:
        if self.adapter is None:
            return self.ddim_step(x, step_index, cond, uncond, guidance_scale)

        ddpm_t = self.time_range[step_index]
        alpha_index = self.total_steps - step_index - 1
        t = torch.full((x.shape[0],), ddpm_t, device=self.device, dtype=torch.long)

        cond_txt = torch.cat(cond["c_crossattn"], 1)
        hint = torch.cat(cond["c_concat"], 1) if cond["c_concat"] is not None else None
        control = None
        if hint is not None:
            control = self.model.control_model(x=x, hint=hint, timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.model.control_scales)]

        eps_cond = apply_unet_with_adapter(
            self.model.model.diffusion_model,
            self.adapter,
            x,
            t,
            cond_txt,
            control,
            self.model.only_mid_control,
        )
        if uncond is not None and guidance_scale > 1.0:
            eps_uncond = self.model.apply_model(x, t, uncond)
            e_t = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            e_t = eps_cond

        b = x.shape[0]
        a_t = torch.full((b, 1, 1, 1), self.ddim.ddim_alphas[alpha_index], device=self.device)
        a_prev = torch.full((b, 1, 1, 1), self.ddim.ddim_alphas_prev[alpha_index], device=self.device)
        sigma_t = torch.full((b, 1, 1, 1), self.ddim.ddim_sigmas[alpha_index], device=self.device)
        sqrt_one_minus_at = torch.full(
            (b, 1, 1, 1), self.ddim.ddim_sqrt_one_minus_alphas[alpha_index], device=self.device
        )
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * torch.randn_like(x)
        return a_prev.sqrt() * pred_x0 + dir_xt + noise

    @torch.no_grad()
    def ddim_steps(
        self,
        x: torch.Tensor,
        cond: dict,
        uncond: Optional[dict],
        guidance_scale: float,
        start_step: int = 0,
        end_step: Optional[int] = None,
    ) -> torch.Tensor:
        end_step = self.total_steps if end_step is None else end_step
        for step_index in range(start_step, end_step):
            x = self.ddim_step(x, step_index, cond, uncond, guidance_scale)
        return x

    @torch.no_grad()
    def generate_latent(
        self,
        cond: dict,
        uncond: Optional[dict],
        guidance_scale: float,
        seed: int,
        image_length: int,
        start_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if start_latent is None:
            torch.manual_seed(seed)
            start_latent = torch.randn((1, 4, image_length // 8, image_length // 8), device=self.device)
        return self.ddim_steps(start_latent, cond, uncond, guidance_scale)

    @torch.no_grad()
    def generate_latent_with_watermark(
        self,
        cond: dict,
        uncond: Optional[dict],
        guidance_scale: float,
        seed: int,
        image_length: int,
        wm_step: int,
        wm_config: ROBINWatermarkConfig,
        start_latent: Optional[torch.Tensor] = None,
        adapter_steps: int = 1,
        watermark_mode: str = "latent",
    ) -> Tuple[torch.Tensor, WatermarkArtifacts]:
        if not 0 <= wm_step <= self.total_steps:
            raise ValueError(f"wm_step must be in [0, {self.total_steps}], got {wm_step}")

        if start_latent is None:
            torch.manual_seed(seed)
            x = torch.randn((1, 4, image_length // 8, image_length // 8), device=self.device)
        else:
            x = start_latent.clone()

        x = self.ddim_steps(x, cond, uncond, guidance_scale, start_step=0, end_step=wm_step)

        wm_mask = build_watermark_mask(x, wm_config)
        wm_pattern = build_watermark_pattern(tuple(x.shape), wm_config, device=x.device)
        if watermark_mode == "latent":
            x = inject_watermark(x, wm_mask, wm_pattern, wm_config)
        elif watermark_mode == "adapter":
            if self.adapter is None:
                raise ValueError("watermark_mode='adapter' requires loading an adapter checkpoint")
        else:
            raise ValueError(f"Unsupported watermark_mode: {watermark_mode}")
        injected = x.clone()

        if self.adapter is not None and wm_step < self.total_steps and adapter_steps > 0:
            adapter_end = min(self.total_steps, wm_step + adapter_steps)
            for step_idx in range(wm_step, adapter_end):
                x = self.ddim_step_with_adapter(x, step_idx, cond, uncond, guidance_scale)
            x = self.ddim_steps(x, cond, uncond, guidance_scale, start_step=adapter_end)
        else:
            x = self.ddim_steps(x, cond, uncond, guidance_scale, start_step=wm_step)
        return x, WatermarkArtifacts(mask=wm_mask, pattern=wm_pattern, injected_latent=injected)

    @torch.no_grad()
    def generate_pair(
        self,
        prompt: str,
        hint_image: Optional[Union[Image.Image, np.ndarray]],
        wm_config: ROBINWatermarkConfig,
        wm_step: int = 35,
        guidance_scale: float = 7.5,
        conditioning_scale: float = 1.0,
        ddim_steps: int = 50,
        seed: int = 0,
        image_length: int = 512,
        negative_prompt: str = "",
        guess_mode: bool = False,
        eta: float = 0.0,
        adapter_steps: int = 1,
        watermark_mode: str = "latent",
    ) -> Tuple[Image.Image, Image.Image, WatermarkArtifacts]:
        """Generate no-watermark and watermarked images from the same initial noise."""

        self.make_schedule(ddim_steps, eta=eta)
        self.model.control_scales = (
            [conditioning_scale * (0.825 ** float(12 - i)) for i in range(13)]
            if guess_mode
            else [conditioning_scale] * 13
        )
        cond, uncond = self.get_conditioning(prompt, hint_image, negative_prompt, image_length, guess_mode)

        torch.manual_seed(seed)
        start_latent = torch.randn((1, 4, image_length // 8, image_length // 8), device=self.device)
        latent_no_wm = self.generate_latent(cond, uncond, guidance_scale, seed, image_length, start_latent)
        latent_wm, artifacts = self.generate_latent_with_watermark(
            cond,
            uncond,
            guidance_scale,
            seed,
            image_length,
            wm_step,
            wm_config,
            start_latent,
            adapter_steps,
            watermark_mode,
        )
        return self.decode_latent_to_image(latent_no_wm), self.decode_latent_to_image(latent_wm), artifacts

    @torch.no_grad()
    def encode_image_to_latent(
        self,
        image: Union[Image.Image, torch.Tensor],
        image_length: int = 512,
        sample_posterior: bool = True,
    ) -> torch.Tensor:
        if isinstance(image, Image.Image):
            tensor = transforms.Compose(
                [
                    transforms.Resize(image_length),
                    transforms.CenterCrop(image_length),
                    transforms.ToTensor(),
                ]
            )(image.convert("RGB"))
            tensor = (2.0 * tensor - 1.0).unsqueeze(0).to(self.device)
        else:
            tensor = image.to(self.device)
        posterior = self.model.encode_first_stage(tensor)
        if not sample_posterior and hasattr(posterior, "mode"):
            latent = posterior.mode()
            return (self.model.scale_factor * latent).detach()
        return self.model.get_first_stage_encoding(posterior).detach()

    @torch.no_grad()
    def decode_latent_to_image(self, latent: torch.Tensor) -> Image.Image:
        image = self.model.decode_first_stage(latent)
        image = torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)
        image_np = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return Image.fromarray((image_np[0] * 255).round().astype(np.uint8))

    @torch.no_grad()
    def ddim_inversion_step(
        self,
        x: torch.Tensor,
        step_index: int,
        cond: dict,
        uncond: Optional[dict] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        ddpm_t = self.time_range[step_index]
        alpha_index = self.total_steps - step_index - 1
        t = torch.full((x.shape[0],), ddpm_t, device=self.device, dtype=torch.long)
        e_t = self._predict_eps(x, t, cond, uncond, guidance_scale)

        alphas = self.ddim.ddim_alphas
        alphas_prev = self.ddim.ddim_alphas_prev
        b = x.shape[0]
        a_t = torch.full((b, 1, 1, 1), alphas[alpha_index], device=self.device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[alpha_index], device=self.device)

        pred_x0 = (x - (1.0 - a_t).sqrt() * e_t) / a_t.sqrt()
        dir_xt = (1.0 - a_prev).sqrt() * e_t
        return a_prev.sqrt() * pred_x0 + dir_xt

    @torch.no_grad()
    def invert_to_step(
        self,
        image: Union[Image.Image, torch.Tensor],
        wm_step: int,
        ddim_steps: int,
        image_length: int = 512,
        test_prompt: str = "",
        hint_image: Optional[Union[Image.Image, np.ndarray]] = None,
        use_controlnet_inversion: bool = False,
        conditioning_scale: float = 1.0,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """VAE encode an image and DDIM-invert it to the watermark step."""

        self.make_schedule(ddim_steps, eta=eta)
        self.model.control_scales = [conditioning_scale] * 13
        latent = self.encode_image_to_latent(image, image_length=image_length)

        if use_controlnet_inversion:
            cond, uncond = self.get_conditioning(test_prompt, hint_image, "", image_length)
        else:
            cond_txt = self.model.get_learned_conditioning([test_prompt])
            cond = {"c_concat": None, "c_crossattn": [cond_txt]}
            uncond = None

        for step_index in range(wm_step - 1, -1, -1):
            latent = self.ddim_inversion_step(latent, step_index, cond, uncond, guidance_scale=1.0)
        return latent

    def watermark_score(
        self,
        recovered_latent: torch.Tensor,
        wm_config: ROBINWatermarkConfig,
        reference_shape: Optional[Tuple[int, ...]] = None,
    ) -> float:
        shape = reference_shape or tuple(recovered_latent.shape)
        mask = build_watermark_mask(recovered_latent, wm_config)
        pattern = build_watermark_pattern(shape, wm_config, device=recovered_latent.device)
        return watermark_distance(recovered_latent, mask, pattern, wm_config)
