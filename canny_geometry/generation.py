"""Frozen plain ControlNet generation; paired states share CPU initial noise."""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image


def generate(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['CANNY_WM_CLIP_PATH']=str(Path(args.clip).resolve())
    import torch
    import cv2
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'vendor/controlnet'))
    from .controlnet_pipeline import ControlNetPipeline
    from cldm.ddim_hacked import DDIMSampler
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        raise RuntimeError('CUDA requested but unavailable.')
    out=Path(args.out)
    if out.exists():raise FileExistsError('Choose a new output directory.')
    if not Path(args.checkpoint).is_file() or not Path(args.clip).is_dir():
        raise FileNotFoundError('Supply local ControlNet checkpoint and local CLIP directory.')
    if args.command=='probe-generate':
        from .stability import validate_plan
        plan=json.loads(Path(args.plan).read_text());validate_plan(plan)
        inputs=[Path(args.canny)]
        jobs=[dict(path=inputs[0],name=f'p{i//2}_s{i%2}.png',**r) for i,r in enumerate(plan)]
    else:
        inputs=sorted(p for p in Path(args.conditions).glob('*.png') if p.stem!='mask')
        jobs=[dict(path=p,name=p.name,prompt=args.prompt,seed=args.seed) for p in inputs]
    if not jobs:raise ValueError('No conditions found.')
    for p in inputs:
        image=np.asarray(Image.open(p).convert('RGB'))
        if image.shape!=(512,512,3):raise ValueError('Generation requires 512x512 conditions.')
        if not set(np.unique(image)).issubset({0,255}):raise ValueError('Use binary Canny conditions.')
    out.mkdir(parents=True)
    torch.set_num_threads(8);cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    class DeviceSampler(DDIMSampler):
        def register_buffer(self,name,attr):
            setattr(self,name,attr.to(self.model.device) if isinstance(attr,torch.Tensor) else attr)
    pipe=ControlNetPipeline(args.config,args.checkpoint,args.device)
    pipe.model.requires_grad_(False).eval();pipe.model.cond_stage_model.device=args.device
    pipe.ddim=DeviceSampler(pipe.model);pipe.make_schedule(20,eta=0.)
    pipe.model.control_scales=[1.]*13;records=[]
    for job in jobs:
        path=job['path'];image=np.array(Image.open(path).convert('RGB'))
        noise=torch.randn((1,4,64,64),generator=torch.Generator(device='cpu').manual_seed(job['seed']))
        cond,uncond=pipe.get_conditioning(job['prompt'],image,'',512,False)
        latent=noise.clone().to(args.device)
        with torch.inference_mode():
            for i in range(pipe.total_steps):latent=pipe.ddim_step(latent,i,cond,uncond,5.)
            rgb=((pipe.model.decode_first_stage(latent)+1)/2).clamp(0,1)
            rgb=(rgb[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
        target=out/job['name'];Image.fromarray(rgb).save(target)
        records.append(dict(condition=str(path),rgb=target.name,prompt=job['prompt'],seed=job['seed'],
            **({'prompt_group':job['prompt_group']} if 'prompt_group' in job else {}),
            hint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            rgb_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        print(target,flush=True)
    (out/'generation.json').write_text(json.dumps(dict(
        checkpoint=str(args.checkpoint),config=str(args.config),steps=20,CFG=5,eta=0,
        control=1,shared_initial_noise=True,backend='plain_ControlNet',records=records),indent=2)+'\n')
    if args.command=='probe-generate':
        (out/'probe_plan.json').write_text(json.dumps(records,indent=2)+'\n')
