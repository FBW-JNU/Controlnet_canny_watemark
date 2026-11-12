"""Frozen Train32 clean-generation manifest for the curve-level stage1 audit."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/canny_curve_stage1_train32_20261110'
OLD=ROOT/'outputs/canny_random10_stability_20261108'
REUSE=ROOT/'outputs/canny_mat_repair_transport10_20261109'
MANIFEST=ROOT/'coco/condition_validity/train2017/canny/normal_manifest.csv'
SEEDS=[2026110801,2026110802,2026110803,2026111004]

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def dump(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def save(p,a):Image.fromarray(a).save(p)

def prepare():
    if (OUT/'protocol.json').exists():
        p=read(OUT/'protocol.json')
        assert sha(OUT/'plan.json')==p['plan_sha256']
        assert sha(OUT/'selected.json')==p['selected_sha256']
        print('EXISTING FROZEN PREPARATION');return
    for name in ('sources','hints','noise','records','rgb','logs','source','curves','masks','panels','metrics','output_canny','graphs'):
        (OUT/name).mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(MANIFEST.open()));byid={int(r['image_id']):(i,r) for i,r in enumerate(rows)}
    old=read(OLD/'selected.json');previous=[int(r['image_id']) for r in old]
    remaining=[i for i,r in enumerate(rows) if int(r['image_id']) not in previous]
    additional=np.random.default_rng(20261110).choice(remaining,22,replace=False)
    chosen=[byid[cid][0] for cid in previous]+list(map(int,additional))
    rp=read(REUSE/'protocol.json');og=rp['generation']
    selected=[];plan=[];reuse_checks=[]
    for ci,idx in enumerate(chosen):
        row=rows[idx];cid=int(row['image_id']);source=Image.open(row['image_path']).convert('RGB')
        w,h=[round(v*512/max(source.size)) for v in source.size];x,y=(512-w)//2,(512-h)//2
        canvas=Image.new('RGB',(512,512),(127,127,127));canvas.paste(source.resize((w,h),Image.Resampling.LANCZOS),(x,y))
        source_path=OUT/'sources'/f'{cid:012d}.png';hint=OUT/'hints'/f'{cid:012d}.png'
        canvas.save(source_path);edge=cv2.Canny(np.asarray(canvas),100,200);save(hint,edge)
        roi=[x+8,y+8,x+w-8,y+h-8]
        if cid==85048:roi=[22,98,492,413] # previously documented native white photo frame
        selected.append(dict(**row,content_id=cid,content_index=ci,manifest_row_index=idx,stage_split='Train',
            previous_random10=cid in previous,source_sha256=sha(row['image_path']),prepared_source=str(source_path),
            hint=str(hint),hint_sha256=sha(hint),roi=roi))
        caption=row['prompt'].strip()
        prompts=[('caption',caption),('watercolor','A watercolor painting of the following scene: '+caption),
                 ('heldout_oil','A detailed oil painting of the following scene: '+caption+', expressive brushwork, dramatic natural lighting.')]
        for pi,(name,prompt) in enumerate(prompts):
            for si,seed in enumerate(SEEDS):
                role='discovery' if pi<2 and si<3 else 'heldout_joint' if pi==2 and si==3 else 'heldout_prompt' if pi==2 else 'heldout_seed'
                job=f'{cid:012d}_p{pi}_s{si}'
                noise=OUT/'noise'/f'{cid:012d}_s{si}.npy'
                p=dict(job=job,content_id=cid,content_index=ci,prompt_index=pi,prompt_name=name,prompt=prompt,
                    seed_index=si,seed=seed,role=role,hint=str(hint),hint_sha256=sha(hint),
                    noise=str(noise),rgb=str(OUT/'rgb'/f'{job}.png'),reused=False)
                if cid in previous and pi<2 and si<3:
                    record_path=REUSE/'records'/f'{job}_original.json';record=read(record_path)
                    assert record['prompt']==prompt and record['seed']==seed
                    assert record['hint_sha256']==sha(hint) and record['checkpoint_sha256']==og['checkpoint_sha256']
                    assert record['precision']=='float32, TF32 disabled' and record['device']=='physical GPU1'
                    assert record['adapter'] is None and record['all_parameters_frozen']
                    assert sha(record['rgb'])==record['rgb_sha256']
                    shutil.copyfile(record['rgb'],p['rgb'])
                    old_noise=OLD/'noise'/f'{cid:012d}_s{si}.npy'
                    assert sha(old_noise)==record['noise_sha256'];shutil.copyfile(old_noise,noise)
                    p.update(reused=True,source_record=str(record_path),source_record_sha256=sha(record_path))
                    dump(OUT/'records'/f'{job}.json',dict(**p,rgb_sha256=sha(p['rgb']),noise_sha256=sha(noise),
                        elapsed_seconds=0,device='physical GPU1',precision=record['precision'],checkpoint_sha256=record['checkpoint_sha256']))
                    reuse_checks.append(dict(job=job,source_record_sha256=sha(record_path),rgb_sha256=sha(p['rgb'])))
                plan.append(p)
    dump(OUT/'selected.json',selected);dump(OUT/'plan.json',plan)
    generation={k:og[k] for k in ('model_config','checkpoint','checkpoint_sha256','steps','eta','guidance','control_scale','negative_prompt','guess_mode')}
    generation.update(device='physical GPU1 (CUDA_VISIBLE_DEVICES=1)',precision='float32, TF32 disabled',
                      seeds=SEEDS,cpu_threads=8,total_pngs=384,reused_pngs=len(reuse_checks),new_pngs=384-len(reuse_checks))
    assert all(generation[k]==read(OLD/'protocol.json')['generation'][k] for k in ('model_config','checkpoint','checkpoint_sha256','steps','eta','guidance','control_scale','negative_prompt','guess_mode'))
    code={str(ROOT/f):sha(ROOT/f) for f in ('cldm/cldm.py','ldm/modules/encoders/modules.py','models/cldm_v15.yaml','controlnet_robin/pipeline.py')}
    assert all(code[k]==v for k,v in rp['frozen_source_code'].items() if k in code)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),stage='Stage1 clean curve stability; no geometric edits, MAT, payload or training',
        population_manifest=str(MANIFEST),population_manifest_sha256=sha(MANIFEST),population=len(rows),
        sampling='Nested probability sample: preserve original uniform random10, add22 uniform without replacement from remaining normal COCO train2017; supplemental RNG20261110. No output-based content exclusion.',
        split='All32 are development Train contents; holdouts are views, not unseen contents. Prior10 used earlier exploratory experiments. New heldout oil prompt and fourth seed not used for selection.',
        preprocessing='RGB Lanczos aspect fit512, gray127 center padding, cv2.Canny(RGB,100,200), valid photo inset8; documented baseball native-frame exclusion.',
        generation=generation,frozen_model_code=code,selected_sha256=sha(OUT/'selected.json'),plan_sha256=sha(OUT/'plan.json'),
        view_split=dict(discovery='p0,p1 x s0,s1,s2 (6)',heldout_seed='p0,p1 x s3 (2)',heldout_prompt='p2 x s0,s1,s2 (3)',heldout_joint='p2 x s3 (1)',
                        prohibition='Candidate filtering, ranking and masks use discovery ONLY; heldout metrics attached afterwards; do not retune thresholds on heldout.'),
        exploration_thresholds=dict(min_source_arc_px=24,max_source_arc_px=160,curvature_split_deg=50,resample_spacing_px=1,spline_smoothing_rms_px=.35,
            tube_radius_px=6,endpoint_search_px=4,anchor_tolerance_px=4,orientation_tolerance_deg=30,
            min_curve_coverage=.8,max_symmetric_median_px=3,max_symmetric_p90_px=6,min_length_ratio=.65,max_length_ratio=1.5,
            discovery_retention_min=.8,discovery_anchor_rate_min=.8,per_discovery_prompt_retention_min=2/3,
            source_duplicate_proximity_px=2,source_duplicate_overlap=.7,mask_half_width_px=4,mask_context_dilate_px=8),
        geometry_rule='Match an ordered path in the independently extracted output skeleton8-neighbour graph within6px tube; endpoint anchors4px; use symmetric curve distance, tangent orientation, coverage and arc-length ratio. Disconnected nearest-point clouds cannot pass graph-route match.',
        statistics='Save per-view matches, R_i, Wilson95% intervals, distance drift, length/curvature/anchor features, V_i; discovery-selected masks with confidence values. No forced minimum candidates.',
        reuse_protocol=str(REUSE/'protocol.json'),reuse_protocol_sha256=sha(REUSE/'protocol.json'),generator_sha256=sha(__file__))
    dump(OUT/'protocol.json',protocol);dump(OUT/'reuse_validation.json',dict(status='passed',count=len(reuse_checks),checks=reuse_checks))
    shutil.copyfile(__file__,OUT/'source'/Path(__file__).name)
    print('FROZEN',len(selected),'contents',len(plan),'views; reuse',len(reuse_checks),'new',384-len(reuse_checks),flush=True)

def worker():
    import torch
    from cldm.ddim_hacked import DDIMSampler
    from controlnet_robin.pipeline import ControlNetROBINPipeline
    protocol=read(OUT/'protocol.json');gen=protocol['generation']
    assert sha(OUT/'plan.json')==protocol['plan_sha256'] and sha(OUT/'selected.json')==protocol['selected_sha256']
    for path,digest in protocol['frozen_model_code'].items():assert sha(path)==digest,path
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and torch.cuda.is_available()
    assert sha(ROOT/gen['checkpoint'])==gen['checkpoint_sha256']
    torch.set_num_threads(gen['cpu_threads']);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    class DeviceSampler(DDIMSampler):
        def register_buffer(self,name,attr):setattr(self,name,attr.to(self.model.device) if isinstance(attr,torch.Tensor) else attr)
    pipe=ControlNetROBINPipeline(str(ROOT/gen['model_config']),str(ROOT/gen['checkpoint']),'cuda')
    pipe.model.requires_grad_(False).eval();pipe.model.cond_stage_model.device='cuda';pipe.ddim=DeviceSampler(pipe.model)
    pipe.make_schedule(gen['steps'],eta=gen['eta']);pipe.model.control_scales=[gen['control_scale']]*13
    conditioning={};plan=read(OUT/'plan.json')
    for p in plan:
        record=OUT/'records'/f'{p["job"]}.json'
        if record.exists():assert sha(p['rgb'])==read(record)['rgb_sha256'];continue
        assert not p['reused'];assert sha(p['hint'])==p['hint_sha256']
        start=torch.randn((1,4,64,64),generator=torch.Generator(device='cpu').manual_seed(p['seed']))
        noise=Path(p['noise'])
        if noise.exists():assert np.array_equal(np.load(noise),start.numpy())
        else:np.save(noise,start.numpy())
        key=(p['content_id'],p['prompt_index'])
        if key not in conditioning:
            conditioning.clear();conditioning[key]=pipe.get_conditioning(p['prompt'],np.asarray(Image.open(p['hint']).convert('RGB')),'',512,False)
        cond,uncond=conditioning[key];latent=start.clone().cuda();begun=time.monotonic()
        with torch.inference_mode():
            for step in range(pipe.total_steps):latent=pipe.ddim_step(latent,step,cond,uncond,gen['guidance'])
            rgb=((pipe.model.decode_first_stage(latent)+1)/2).clamp(0,1)
            array=(rgb[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
        save(p['rgb'],array)
        dump(record,dict(**p,rgb_sha256=sha(p['rgb']),noise_sha256=sha(noise),elapsed_seconds=time.monotonic()-begun,
            device='physical GPU1',precision=gen['precision'],checkpoint_sha256=gen['checkpoint_sha256'],adapter=None,all_parameters_frozen=True))
        print('SAVED',p['job'],'seconds',round(time.monotonic()-begun,2),'complete',len(list((OUT/'records').glob('*.json'))),'/384',flush=True)
    dump(OUT/'generation_status.json',dict(status='complete',completed=384,reused=sum(p['reused'] for p in plan),new=sum(not p['reused'] for p in plan)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','worker']);args=parser.parse_args()
    prepare() if args.action=='prepare' else worker()
