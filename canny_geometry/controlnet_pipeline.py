"""Plain ControlNet DDIM wrapper, extracted from the previously shared wrapper.
Only conditioning/scheduling/ordinary denoising are retained; no ROBIN imports.
"""
from typing import Optional, Tuple, Union
import einops
import numpy as np
import torch
from PIL import Image
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import hack_everything
from cldm.model import create_model, load_state_dict

class ControlNetPipeline:
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
