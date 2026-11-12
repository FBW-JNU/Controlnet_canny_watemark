"""Reusable frozen ControlNet channel for manifest-driven batch experiments."""
import os
import sys
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


class ControlNetChannel:
    def __init__(self, checkpoint, clip, device='cuda', config=None):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        os.environ['CANNY_WM_CLIP_PATH']=str(Path(clip).resolve())
        import torch
        import cv2
        from .api import ROOT
        clip_assets=next(a for a in json.loads((ROOT/'provenance/external_assets.json').read_text())['assets'] if a['id']=='clip')
        for name,record in clip_assets['files'].items():
            path=Path(clip)/name
            if not path.is_file():raise FileNotFoundError('Missing frozen CLIP asset: '+str(path))
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
            if digest.hexdigest()!=record['sha256']:raise ValueError('CLIP asset hash differs: '+name)
        sys.path.insert(0,str(ROOT/'vendor/controlnet'))
        from .controlnet_pipeline import ControlNetPipeline
        from cldm.ddim_hacked import DDIMSampler
        if not Path(checkpoint).is_file() or not Path(clip).is_dir():
            raise FileNotFoundError('Supply local checkpoint and CLIP directory.')
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable.')
        torch.set_num_threads(8);cv2.setNumThreads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        class DeviceSampler(DDIMSampler):
            def register_buffer(self,name,attr):
                setattr(self,name,attr.to(self.model.device) if isinstance(attr,torch.Tensor) else attr)
        self.device=device;self.torch=torch
        self.pipe=ControlNetPipeline(str(config or ROOT/'configs/cldm_v15.yaml'),str(checkpoint),device)
        self.pipe.model.requires_grad_(False).eval();self.pipe.model.cond_stage_model.device=device
        self.pipe.ddim=DeviceSampler(self.pipe.model);self.pipe.make_schedule(20,eta=0.)
        self.pipe.model.control_scales=[1.]*13

    def render(self, hint, prompt, seed):
        torch=self.torch;pipe=self.pipe
        image=np.asarray(Image.open(hint).convert('RGB'))
        if image.shape!=(512,512,3):raise ValueError('Use 512x512 condition images.')
        noise=torch.randn((1,4,64,64),generator=torch.Generator(device='cpu').manual_seed(seed))
        cond,uncond=pipe.get_conditioning(prompt,image,'',512,False)
        latent=noise.clone().to(self.device)
        with torch.inference_mode():
            for i in range(pipe.total_steps):latent=pipe.ddim_step(latent,i,cond,uncond,5.)
            rgb=((pipe.model.decode_first_stage(latent)+1)/2).clamp(0,1)
            rgb=(rgb[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
        return rgb
