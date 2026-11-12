"""Frozen, paired single-bit geometric channel experiment (no training)."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from canny_curve_stage1_32 import ROOT, read, dump, sha, save

S1=ROOT/'outputs/canny_curve_stage1_train32_20261110'
OUT=ROOT/'outputs/canny_curve_stage2_bit_20261110'
MODES=['normal','sagitta','polynomial']
AMPLITUDE=1.5

def matcher():
    spec=importlib.util.spec_from_file_location('stage1_frozen_matcher',OUT/'frozen_stage1/analyze_canny_curve_stage1_32.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def canonical(samples):
    p=np.asarray(samples,float)
    return p if tuple(p[0])<tuple(p[-1]) else p[::-1].copy()

def fields(samples,mode,m=None):
    if m is None:m=matcher()
    p=canonical(samples);u=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];u/=u[-1]
    t=m.tangent(p);n=np.c_[-t[:,1],t[:,0]]
    if mode=='normal':
        v=np.minimum(u,1-u)/.25;b=np.sin(np.clip(v,0,1)*np.pi/2)**2
    elif mode=='sagitta':
        chord=p[-1]-p[0];chord/=np.linalg.norm(chord);n=np.tile([-chord[1],chord[0]],(len(p),1));b=4*u*(1-u)
    else:b=6*np.sqrt(3)*u*(1-u)*(2*u-1)
    return p,u,n,b

def coefficient(samples,path,mode,m=None):
    p,u,n,b=fields(samples,mode,m)
    q=(m or matcher()).resample(np.asarray(path,float),spacing=.25)
    _,near=cKDTree(q).query(p)
    return float(np.sum(np.sum((q[near]-p)*n,axis=1)*b)/np.sum(b*b))

def warp(edge,samples,mask,mode,amplitude,m):
    """Move actual binary edge pixels, preserving the identity map at zero dose."""
    p,u,n,b=fields(samples,mode,m)
    yy,xx=np.nonzero(edge);xy=np.c_[xx,yy];dist,near=cKDTree(p).query(xy)
    chosen=(dist<=1.25)&mask[yy,xx]
    moved=xy.copy();moved[chosen]=np.rint(xy[chosen]+amplitude*b[near[chosen],None]*n[near[chosen]]).astype(int)
    assert np.all((moved>=0)&(moved<512))
    out=edge.copy();out[yy[chosen],xx[chosen]]=0;out[moved[:,1],moved[:,0]]=255
    lookup=np.full(edge.shape,-1,int);lookup[yy,xx]=np.arange(len(xy))
    for dx,dy in [(1,0),(0,1),(1,1),(-1,1)]:
        nx=xx+dx;ny=yy+dy;valid=(nx>=0)&(nx<512)&(ny>=0)&(ny<512)
        ids=np.nonzero(valid)[0];other=lookup[ny[valid],nx[valid]];keep=(other>=0)
        ids=ids[keep];other=other[keep];keep=chosen[ids]|chosen[other]
        for a,z in zip(ids[keep],other[keep]):cv2.line(out,tuple(moved[a]),tuple(moved[z]),255,1,cv2.LINE_8)
    out[~mask]=edge[~mask]
    return out,dict(moving_edge_pixels=int(chosen.sum()),changed_pixels=int(np.count_nonzero(out!=edge)))

def stratum(r):
    t=r['source_geometry']['tortuosity']
    return 'straight' if t<1.01 else 'mild' if t<1.05 else 'bent'

def freeze_stage1():
    if (OUT/'stage1_freeze.json').exists():return
    for name in ('frozen_stage1','source','conditions','masks','curves','rgb','noise','records','logs','metrics','output_canny','panels','plots'):
        (OUT/name).mkdir(parents=True,exist_ok=True)
    # Full stage1 data tree, including generated PNGs and per-view matches.
    files=[p for p in S1.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
    for name in ['canny_curve_stage1_32.py','analyze_canny_curve_stage1_32.py','report_canny_curve_stage1_32.py','audit_canny_curve_stage1_32.py']:
        original=ROOT/name;snapshot=S1/'source'/name
        assert sha(original)==sha(snapshot),name
        hashes[str(original.relative_to(ROOT))]=sha(original)
        shutil.copyfile(snapshot,OUT/'frozen_stage1'/name)
    dump(OUT/'stage1_freeze.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),policy='Read-only SHA256 freeze; no stage1 files modified',files=hashes))

def prepare():
    if (OUT/'protocol.json').exists():
        verify();print('EXISTING FROZEN PREPARATION');return
    cv2.setNumThreads(1);freeze_stage1();m=matcher();t=read(S1/'protocol.json')['exploration_thresholds']
    eligible=[r for r in read(S1/'curve_results.json') if r['heldout_R_ge_80pct']]
    assert len(eligible)==751
    dump(OUT/'eligible751.json',eligible)
    cases={r['content_id']:r for r in read(S1/'selected.json')}
    rng=np.random.default_rng(2026111202);ids=np.array(sorted(cases));rng.shuffle(ids)
    splits={int(cid):'Cal' if i<16 else 'Dev' for i,cid in enumerate(ids)}
    dump(OUT/'content_split.json',splits)
    source={cid:read(S1/'curves'/f'{cid:012d}_source.json')['curves'] for cid in cases}
    chosen=[];rejections=[];used=set()
    for split in ['Cal','Dev']:
        for kind in ['bent','mild','straight']:
            pool=[r for r in eligible if splits[r['content_id']]==split and stratum(r)==kind]
            rng.shuffle(pool);count=0
            for r in pool:
                cid=r['content_id'];ci=r['curve_index']
                if cid in used:continue
                c=next(c for c in source[cid] if c['curve_index']==ci)
                edge=np.asarray(Image.open(cases[cid]['hint']).convert('L'))
                mask=np.asarray(Image.open(S1/r['mask']).convert('L'))>0
                cleanmatch=m.curve_match(c['samples'],m.skeleton_graph(edge),t,return_path=True)
                failure=None;inputs={};features={};checks={}
                if not cleanmatch['matched']:failure='clean_input_match'
                for mode in MODES:
                    zero,_=warp(edge,c['samples'],mask,mode,0,m)
                    assert np.array_equal(zero,edge),'zero-dose identity'
                    if failure:break
                    cc=coefficient(c['samples'],cleanmatch['output_path'],mode,m)
                    features[mode]={'clean':cc};checks[mode]={'zero_dose_identity':True}
                    for arm,sign in [('bit0',-1),('bit1',1)]:
                        img,info=warp(edge,c['samples'],mask,mode,sign*AMPLITUDE,m)
                        assert np.array_equal(img[~mask],edge[~mask]);assert set(np.unique(img))<={0,255}
                        match=m.curve_match(c['samples'],m.skeleton_graph(img),t,return_path=True)
                        info.update(matched=match['matched'],reason=match['reason'])
                        checks[mode][arm]=info;inputs[(mode,arm)]=img
                        if not match['matched']:failure=f'{mode}_{arm}_input_match';break
                        features[mode][arm]=coefficient(c['samples'],match['output_path'],mode,m)
                    if failure:break
                    f=features[mode]
                    if not (f['bit1']-f['bit0']>=1 and f['bit0']<=cc-.25 and f['bit1']>=cc+.25):
                        failure=f'{mode}_input_separation';break
                    f['delta_in']=f['bit1']-f['bit0'];f['midpoint']=(f['bit1']+f['bit0'])/2
                    f['half_separation']=f['delta_in']/2
                if failure:
                    rejections.append(dict(content_id=cid,curve_index=ci,split=split,stratum=kind,reason=failure,features=features,checks=checks));continue
                key=f'{cid:012d}_c{ci:04d}';folder=OUT/'conditions'/key;folder.mkdir(exist_ok=True)
                shutil.copyfile(cases[cid]['hint'],folder/'clean.png');shutil.copyfile(S1/r['mask'],OUT/'masks'/f'{key}.png')
                for (mode,arm),img in inputs.items():save(folder/f'{mode}_{arm}.png',img)
                entry=dict(key=key,content_id=cid,curve_index=ci,split=split,stratum=kind,
                    source_geometry=r['source_geometry'],source_arc_length_px=r['source_arc_length_px'],
                    source_trace_index=r['source_trace_index'],stage1_heldout_R=r['heldout']['R'],
                    samples=c['samples'],canonical_samples=canonical(c['samples']).tolist(),
                    mask=str(OUT/'masks'/f'{key}.png'),clean_hint=str(folder/'clean.png'),
                    input_features=features,input_checks=checks,caption=cases[cid]['prompt'].strip(),
                    source_rgb=cases[cid]['prepared_source'])
                chosen.append(entry);used.add(cid);count+=1
                print('SELECT',split,kind,key,flush=True)
                if count==4:break
            assert count==4,(split,kind,count)
    plan=[]
    for c in chosen:
        for view in range(2):
            prompt=('A crisp documentary photograph of the following scene: ' if view==0 else 'A realistic cinematic photograph of the following scene: ')+c['caption']
            seed=(2026112001 if c['split']=='Cal' else 2026112101)+view
            for mode,arm in [('shared','clean')]+[(mo,a) for mo in MODES for a in ['bit0','bit1']]:
                job=f'{c["key"]}_v{view}_{mode}_{arm}'
                hint=Path(c['clean_hint']) if arm=='clean' else Path(c['clean_hint']).parent/f'{mode}_{arm}.png'
                plan.append(dict(job=job,key=c['key'],content_id=c['content_id'],curve_index=c['curve_index'],split=c['split'],
                    stratum=c['stratum'],view=view,mode=mode,arm=arm,prompt=prompt,seed=seed,
                    hint=str(hint),hint_sha256=sha(hint),noise=str(OUT/'noise'/f'{c["key"]}_v{view}.npy'),rgb=str(OUT/'rgb'/f'{job}.png')))
    assert len(plan)==336 and len(used)==24
    old=read(S1/'plan.json');assert not ({p['seed'] for p in plan}&{p['seed'] for p in old})
    assert not ({p['prompt'] for p in plan}&{p['prompt'] for p in old})
    dump(OUT/'selection.json',chosen);dump(OUT/'input_rejections.json',rejections);dump(OUT/'plan.json',plan)
    oldprotocol=read(S1/'protocol.json');qproto=ROOT/'outputs/condition_carrier_screen_20261106/protocol.json'
    gen=oldprotocol['generation'].copy();gen.update(total_pngs=336,reused_pngs=0,new_pngs=336)
    weights=[Path('/home/hongwei/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth'),Path('/home/hongwei/.conda/envs/control/lib/python3.9/site-packages/lpips/weights/v0.1/alex.pth')]
    protected={str(p):sha(p) for p in OUT.joinpath('conditions').rglob('*.png')}
    protected.update({str(p):sha(p) for p in OUT.joinpath('masks').glob('*.png')})
    protected.update({str(p):sha(p) for p in [OUT/'selection.json',OUT/'content_split.json',OUT/'plan.json',OUT/'eligible751.json',OUT/'stage1_freeze.json',OUT/'input_rejections.json',OUT/'frozen_stage1/analyze_canny_curve_stage1_32.py',ROOT/'carrier_screen/common.py']+weights})
    shutil.copyfile(__file__,OUT/'source'/Path(__file__).name)
    protected[str(OUT/'source'/Path(__file__).name)]=sha(__file__)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),stage='Stage2 single-bit geometric writability, Oracle location, no training',
        stage1_freeze_sha256=sha(OUT/'stage1_freeze.json'),generation=gen,frozen_model_code=oldprotocol['frozen_model_code'],
        sampling='751 eligible heldout>=5/6. Content split RNG2026111202, 16Cal/16Dev; randomly ordered candidate pool per stratum, bent then mild then straight, skip previously used contents. Four per stratum per split, one curve per content. Input-only admissibility rejections retained. No output filtering.',
        strata=dict(straight='tortuosity<1.01',mild='1.01<=tortuosity<1.05',bent='tortuosity>=1.05'),
        independence='Dev is content-disjoint from Stage2 Cal, but all were Stage1 Train development contents. Two new prompt/seed joint views; prompt and seed effects cannot be disentangled.',
        modulation=dict(amplitude_px=AMPLITUDE,orientation='Source first endpoint lexicographically before last (x,y); bit0 negative, bit1 positive',
            normal='local normal, sin^2 end taper over25% arc, central plateau',sagitta='chord normal, 4u(1-u)',polynomial='local normal, 6sqrt(3)u(1-u)(2u-1)',
            rasterizer='Move original edge pixels <=1.25px from source curve and inside fixed Stage1 mask, round mapped locations, reconnect original 8-neighbour pairs, restore outside mask. Zero dose exactly original.',
            input_screen='All6 edited inputs match frozen Stage1 matcher; each delta>=1px and bit0<=clean-.25,bit1>=clean+.25. No amplitude search.'),
        matcher_thresholds=t,matcher_rule='Unchanged Stage1 connected graph matcher, always original source reference and fixed mask. No bit-specific templates. Whole RGB Canny100/200; binary skeleton8. Only matched routes yield valid features.',
        receiver='Known source curve/Oracle location and input codebook, no true message, prompt, seed or paired clean RGB. Least-squares signed displacement coefficient projected onto fixed basis; nearest route samples every0.25px. Score=(coefficient-input_midpoint)/input_half_separation. Fixed positive polarity. No trained receiver.',
        missing_policy='Strict failure is abstention. Primary ITT AUC uses neutral score0 for missing, alongside success-only AUC, coverage and adversarial missing-score bounds. No success filtering for quality.',
        calibration='Freeze primary mode using Cal only: eligible modes require match coverage>=.8 and ALL marked images pass joint quality. Highest neutral ITT bit AUC, tie normal,sagitta,polynomial. If none eligible, select highest AUC descriptively and mark no qualified mode. Clean presence threshold95th percentile of Cal abs(score). No polarity flips.',
        acceptance=dict(dev_bit_auc_min=.8,dev_marked_match_rate_min=.8,quality='ALL Dev marked attempts for chosen mode pass frozen joint gates',repeatable_contents_min=3,repeatability='both view pairs strictly matched and delta_out>0 on both; at least3 Dev contents',
                        presence='Report clean-vs-bit0, clean-vs-bit1 and clean-vs-combined AUC and Cal-threshold FPR/TPR; no predetermined presence acceptance threshold'),
        quality_gate=read(qproto)['quality_gate'],quality_provenance=str(qproto),quality_provenance_sha256=sha(qproto),
        uncertainty='2000 bootstrap replicates by content, preserve both views and labels. Small exploratory Dev12, not certification.',protected_sha256=protected)
    dump(OUT/'protocol.json',protocol)
    print('FROZEN Stage2',len(chosen),'curves',len(plan),'RGB; rejected',len(rejections),flush=True)

def verify(full=False):
    p=read(OUT/'protocol.json')
    for file,digest in p['protected_sha256'].items():assert sha(file)==digest,file
    for file,digest in p['frozen_model_code'].items():assert sha(file)==digest,file
    if full:
        for file,digest in read(OUT/'stage1_freeze.json')['files'].items():assert sha(ROOT/file)==digest,file
    return p

def worker():
    import torch
    from cldm.ddim_hacked import DDIMSampler
    from controlnet_robin.pipeline import ControlNetROBINPipeline
    p=verify();gen=p['generation'];assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and torch.cuda.is_available()
    assert sha(ROOT/gen['checkpoint'])==gen['checkpoint_sha256']
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    class DeviceSampler(DDIMSampler):
        def register_buffer(self,name,attr):setattr(self,name,attr.to(self.model.device) if isinstance(attr,torch.Tensor) else attr)
    pipe=ControlNetROBINPipeline(str(ROOT/gen['model_config']),str(ROOT/gen['checkpoint']),'cuda')
    pipe.model.requires_grad_(False).eval();pipe.model.cond_stage_model.device='cuda';pipe.ddim=DeviceSampler(pipe.model)
    pipe.make_schedule(gen['steps'],eta=gen['eta']);pipe.model.control_scales=[gen['control_scale']]*13
    plan=read(OUT/'plan.json')
    for idx,j in enumerate(plan):
        record=OUT/'records'/f'{j["job"]}.json'
        if record.exists():assert sha(j['rgb'])==read(record)['rgb_sha256'];continue
        assert sha(j['hint'])==j['hint_sha256']
        noise=torch.randn((1,4,64,64),generator=torch.Generator(device='cpu').manual_seed(j['seed']))
        if Path(j['noise']).exists():assert np.array_equal(np.load(j['noise']),noise.numpy())
        else:np.save(j['noise'],noise.numpy())
        cond,uncond=pipe.get_conditioning(j['prompt'],np.asarray(Image.open(j['hint']).convert('RGB')),'',512,False)
        latent=noise.clone().cuda();begun=time.monotonic()
        with torch.inference_mode():
            for step in range(pipe.total_steps):latent=pipe.ddim_step(latent,step,cond,uncond,gen['guidance'])
            rgb=((pipe.model.decode_first_stage(latent)+1)/2).clamp(0,1)
            array=(rgb[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
        save(j['rgb'],array)
        dump(record,dict(**j,rgb_sha256=sha(j['rgb']),noise_sha256=sha(j['noise']),elapsed_seconds=time.monotonic()-begun,
            device='physical GPU1',precision=gen['precision'],checkpoint_sha256=gen['checkpoint_sha256'],adapter=None,all_parameters_frozen=True))
        print('SAVED',idx+1,'/',len(plan),j['job'],'seconds',round(time.monotonic()-begun,2),flush=True)
    dump(OUT/'generation_status.json',dict(status='complete',completed=len(plan)))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','worker','verify']);a.add_argument('--full',action='store_true');args=a.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='worker':worker()
    else:verify(args.full);print('FREEZE VERIFIED',args.full)
