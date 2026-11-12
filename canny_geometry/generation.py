"""Current ControlNet wrapper, plain paired DDIM path; no latent watermark."""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image


def generate(args):
    # Same deterministic setup as the experimental generator, before CUDA init.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['CANNY_WM_CLIP_PATH']=str(Path(args.clip).resolve())
    import torch
    import cv2
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'vendor/controlnet'))
    from controlnet_robin.pipeline import ControlNetROBINPipeline
    from cldm.ddim_hacked import DDIMSampler
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        raise RuntimeError('CUDA requested but unavailable.')
    inputs=sorted(Path(args.conditions).glob('*.png'))
    inputs=[p for p in inputs if p.stem not in ['mask']]
    if not inputs:raise ValueError('No condition PNGs found.')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if any((out/p.name).exists() for p in inputs):
        raise FileExistsError('Output exists; choose another output directory.')
    if not Path(args.checkpoint).is_file() or not Path(args.clip).is_dir():
        raise FileNotFoundError('Supply local ControlNet checkpoint and local CLIP directory.')
    torch.set_num_threads(8);cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    class DeviceSampler(DDIMSampler):
        def register_buffer(self,name,attr):
            setattr(self,name,attr.to(self.model.device) if isinstance(attr,torch.Tensor) else attr)
    pipe=ControlNetROBINPipeline(args.config,args.checkpoint,args.device)
    pipe.model.requires_grad_(False).eval()
    pipe.model.cond_stage_model.device=args.device
    pipe.ddim=DeviceSampler(pipe.model);pipe.make_schedule(20,eta=0.)
    pipe.model.control_scales=[1.]*13
    noise=torch.randn((1,4,64,64),generator=torch.Generator(device='cpu').manual_seed(args.seed))
    records=[]
    for path in inputs:
        image=np.array(Image.open(path).convert('RGB'))
        if image.shape!=(512,512,3):raise ValueError('Generation requires 512x512 Canny conditions.')
        cond,uncond=pipe.get_conditioning(args.prompt,image,'',512,False)
        latent=noise.clone().to(args.device)
        with torch.inference_mode():
            for i in range(pipe.total_steps):latent=pipe.ddim_step(latent,i,cond,uncond,5.)
            rgb=((pipe.model.decode_first_stage(latent)+1)/2).clamp(0,1)
            rgb=(rgb[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
        target=out/path.name;Image.fromarray(rgb).save(target)
        records.append(dict(condition=str(path),output=str(target),
            condition_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            rgb_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        print(target,flush=True)
    (out/'generation.json').write_text(json.dumps(dict(prompt=args.prompt,seed=args.seed,
        checkpoint=str(args.checkpoint),config=str(args.config),steps=20,CFG=5,eta=0,
        control=1,shared_initial_noise=True,no_latent_injection=True,records=records),indent=2)+'\n')
