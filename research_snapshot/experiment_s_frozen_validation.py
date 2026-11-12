"""Frozen S encoder/receiver, source-only capacity audit and independent24."""
from broad_peak_runtime import ROOT,OUT as PREV,S1,read,digest,arr,THRESHOLDS,GATE,local_matcher
from canny_curve_stage1_32 import dump
from curve_broad_peak_code import render,receive,code_score,profile
from curve_receiver_stage2c import binary_target,geometry
from curve_qim_closed_loop_v2 import RasterWarp
from pathlib import Path
from PIL import Image
import numpy as np
import cv2,json,time,csv,shutil,re,subprocess,os,argparse

OUT=ROOT/'outputs/canny_s_frozen_validation_20261111'
MANIFEST=ROOT/'coco/condition_validity/train2017/canny/normal_manifest.csv'
SEEDS=[2026114101,2026114102]
PREFIX=['A realistic documentary photograph of the following scene: ','An expressive watercolor painting of the following scene: ']

def freeze():
    OUT.mkdir(exist_ok=True)
    if (OUT/'freeze.json').exists():check();return
    for n in ['source','sources','hints','conditions','masks','rgb','noise','records','measurements','capacity','panels','plots','diagnostics']:(OUT/n).mkdir(exist_ok=True)
    previous=read(PREV/'final_artifacts_sha256.json')
    for name,h in previous.items():assert digest(PREV/name)==h,name
    scripts=['experiment_s_frozen_validation.py','curve_broad_peak_code.py','curve_receiver_stage2c.py','curve_receiver_stage2b.py','curve_qim_closed_loop_v2.py','canny_curve_stage2_bit.py','broad_peak_runtime.py']
    for name in scripts:shutil.copyfile(ROOT/name,OUT/'source'/name)
    model=read(PREV/'protocol.json')['generation'];score=read(PREV/'score_freeze.json')['models']
    dump(OUT/'freeze.json',dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        source_sha256={str(ROOT/n):digest(ROOT/n) for n in scripts},parent_manifest_sha256=digest(PREV/'final_artifacts_sha256.json'),
        encoder=dict(basis='S(t)=6sqrt(3)t(1-t)(2t-1)',amplitude_px=1.5,orientation='Lexicographic endpoint canonicalization',parameterization='Normalized arc length, original frozen tangent and normal',
            raster='Original nearest-source-vertex binary pixel warp <=1.25px, round, reconnect8-neighbor pairs, preserve outside fixed mask, zero-dose identity'),
        receiver=dict(implementation='Exact S branch of preceding joint S/T Huber receiver which produced31/48. R0-first/A3 fallback; no threshold changes.',gate=GATE,thresholds=THRESHOLDS),
        score_models={h:score['S_'+h] for h in ['Auto','Oracle']},generation=model,quality=read(PREV/'protocol.json')['quality'],
        input_admissibility='Cal-frozen source-only rule: clean and both saved binary messages accepted by Auto and Oracle; distinct two-message codebook, frozen score decision recovers both. No output PNG used for carrier eligibility.',
        candidate_ranking='Greedy descending source arc length, curve_index tie-break. Independent carrier lower bound requires masks separated by2px and source curves >8px apart. Primary is first eligible; if none retain content and fallback, count failures.',
        sampling=dict(contents=24,rng=2026111141,policy='Uniform24 normal COCO train contents excluding all content IDs observable in surviving output paths and selection/plan metadata. No source/output based content replacement.'),
        prompts=PREFIX,seeds=SEEDS,planned_rgb=288,split='Fresh24, untouched for parameter selection;4views each, clean/bit0/bit1.',
        targets=dict(input_closed_loop=.99,all_attempt_bit_accuracy=.8,bit_auc=.8,presence_auc=.7),
        constraints='No new polynomial, amplitude search, receiver threshold tuning, eight/thirty-two-carrier real generation, or changes to sealed audit data.'))

def check():
    f=read(OUT/'freeze.json')
    for p,h in f['source_sha256'].items():assert digest(p)==h,p
    return f

def excluded_contents():
    files=subprocess.check_output(['rg','--files','outputs'],cwd=ROOT,text=True).splitlines();ids=set();sources=[]
    for name in files:
        ids.update(int(x) for x in re.findall(r'(?<!\d)(0\d{11})(?!\d)',name) if 0<int(x)<600000)
        if not name.endswith(('selection.json','selected.json','plan.json','content_sources.json')) or '/source/' in name:continue
        try:obj=read(ROOT/name)
        except (ValueError,OSError):continue
        def collect(x):
            if isinstance(x,dict):
                for k,v in x.items():
                    if k in ['content_id','image_id']:
                        try:
                            n=int(v)
                            if 0<n<600000:ids.add(n)
                        except (ValueError,TypeError):pass
                    elif k not in ['samples','canonical_samples','source_atlas','observations','codebooks']:collect(v)
            elif isinstance(x,list):
                for y in x:collect(y)
        collect(obj);sources.append(dict(path=name,sha256=digest(ROOT/name)))
    dump(OUT/'exclusions.json',dict(ids=sorted(ids),metadata=sources,path_inventory_sha256=__import__('hashlib').sha256('\n'.join(sorted(files)).encode()).hexdigest(),limitation='Missing historical stage2 directories cannot be enumerated; all available paths and metadata excluded.'))
    return ids

def mask_for(edge,samples,roi):
    valid=np.zeros_like(edge,bool);x0,y0,x1,y1=roi;valid[y0:y1,x0:x1]=True;line=np.zeros_like(edge)
    cv2.polylines(line,[np.rint(samples).astype('int32').reshape(-1,1,2)],False,255,1)
    return (cv2.dilate(line,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))>0)&valid

def input_measure(edge,source,mask):
    warp=RasterWarp(edge,source,mask);warp.vector=profile(warp.u,'S')[warp.near[warp.chosen],None]*warp.n[warp.near[warp.chosen]]
    images={arm:edge if sign==0 else warp.render(sign*1.5)[0] for arm,sign in [('clean',0),('bit0',-1),('bit1',1)]}
    assert np.array_equal(warp.render(0)[0],edge)
    rr={arm:receive(source,binary_target(img,source),'S',THRESHOLDS,GATE) for arm,img in images.items()};books={}
    for head in ['Auto','Oracle']:
        valid=all(rr[arm][head]['accepted'] for arm in ['bit0','bit1'])
        books[head]={arm:rr[arm][head]['coefficient'] for arm in ['clean','bit0','bit1']} if valid else None
        if valid:
            distance=np.linalg.norm(np.array(books[head]['bit1'])-books[head]['bit0'])
            if distance<1e-6:books[head]=None
    good=all(books[h] is not None and rr['clean'][h]['accepted'] for h in ['Auto','Oracle'])
    for head in ['Auto','Oracle']:
        for arm,bit in [('bit0',0),('bit1',1)]:
            value=code_score(rr[arm][head],books[head])
            if value is not None:
                model=read(OUT/'freeze.json')['score_models'][head];value=(value-model['intercept'])/model['gain']
            good=good and value is not None and int(value>0)==bit and np.count_nonzero(images[arm]!=edge)>0
    return images,rr,books,bool(good)

def prepare():
    freeze();f=check()
    if (OUT/'input_freeze.json').exists():return
    if (OUT/'content_sample.json').exists():sample=read(OUT/'content_sample.json')
    else:
        excluded=excluded_contents();rows=list(csv.DictReader(MANIFEST.open()));rows=[r for r in rows if int(r['image_id']) not in excluded]
        indices=np.random.default_rng(2026111141).choice(len(rows),24,replace=False);sample=[rows[int(i)] for i in indices];dump(OUT/'content_sample.json',sample)
    selected=[];inputs=[];plan=[];m=local_matcher();cv2.setNumThreads(1)
    for row in sample:
        cid=int(row['image_id']);checkpoint=OUT/'capacity'/f'{cid:012d}.json'
        if checkpoint.exists():result=read(checkpoint)
        else:
            original=Image.open(row['image_path']).convert('RGB');w,h=[round(v*512/max(original.size)) for v in original.size];x,y=(512-w)//2,(512-h)//2
            canvas=Image.new('RGB',(512,512),(127,127,127));canvas.paste(original.resize((w,h),Image.Resampling.LANCZOS),(x,y));source=OUT/'sources'/f'{cid:012d}.png';canvas.save(source)
            edge=cv2.Canny(np.array(canvas),100,200);hint=OUT/'hints'/f'{cid:012d}.png';Image.fromarray(edge).save(hint);roi=[x+8,y+8,x+w-8,y+h-8];valid=np.zeros_like(edge);valid[roi[1]:roi[3],roi[0]:roi[2]]=edge[roi[1]:roi[3],roi[0]:roi[2]]
            curves,_,_=m.source_curves(valid,THRESHOLDS);curves=sorted(curves,key=lambda c:(-c['arc_length_px'],c['curve_index']));audit=[];good=[];primary=None;occupied=np.zeros_like(edge,bool);independent=[]
            for curve in curves:
                samples=curve['samples'];mask=mask_for(edge,samples,roi);images,measures,books,admissible=input_measure(edge,samples,mask)
                entry=dict(curve_index=curve['curve_index'],arc_length_px=curve['arc_length_px'],samples=samples,admissible=admissible,
                    message_received={h:{a:measures[a][h]['accepted'] for a in ['clean','bit0','bit1']} for h in ['Auto','Oracle']},
                    input_gate_failures={a:[{k:v for k,v in q['gate_checks'].items() if not v} for q in measures[a]['Auto']['decisions']] for a in ['bit0','bit1']})
                audit.append(entry)
                if admissible:
                    good.append(entry);expanded=cv2.dilate(mask.astype('uint8'),np.ones((5,5),np.uint8))>0
                    from scipy.spatial import cKDTree
                    distant=all(cKDTree(q['samples']).query(samples)[0].min()>8 for q in independent)
                    if not np.any(expanded&occupied) and distant:independent.append(entry);occupied|=expanded
                if primary is None and (admissible or curve is curves[-1]):
                    if not admissible and curves:
                        curve=curves[0];samples=curve['samples'];mask=mask_for(edge,samples,roi);images,measures,books,admissible=input_measure(edge,samples,mask)
                    primary=dict(key=f'{cid:012d}_c{curve["curve_index"]:04d}',content_id=cid,curve_index=curve['curve_index'],samples=samples,caption=row['prompt'].strip(),source_rgb=str(source),roi=roi,
                        source_original=row['image_path'],source_sha256=digest(row['image_path']),arc_length_px=curve['arc_length_px'],codebooks={'S':books},input_admissible=admissible,split='Independent24',mode='S',stratum='straight' if curve['arc_length_px']/curve['chord_px']<1.01 else 'mild' if curve['arc_length_px']/curve['chord_px']<1.05 else 'bent')
                    path=OUT/'masks'/f'{primary["key"]}.png';Image.fromarray(mask.astype('uint8')*255).save(path);primary['mask']=str(path);inputrows=[]
                    for arm,image in images.items():
                        path=OUT/'conditions'/f'{primary["key"]}_{arm}.png';Image.fromarray(image).save(path);assert np.array_equal(image[~mask],edge[~mask])
                        inputrows.append(dict(key=primary['key'],mode='S',arm=arm,hint=str(path),hint_sha256=digest(path),changed_pixels=int(np.count_nonzero(image!=edge)),measurements=measures[arm]))
                    primary['clean_hint']=inputrows[0]['hint']
            assert primary is not None,('No source curves; retain content and implement explicit absent-carrier fallback',cid)
            result=dict(content_id=cid,primary=primary,inputrows=inputrows,source_candidates=len(curves),eligible=len(good),independent_lower_bound=len(independent),independent_curve_indices=[q['curve_index'] for q in independent],candidates=audit)
            dump(checkpoint,result)
        c=result['primary'];selected.append(c);inputs+=result['inputrows']
        for pi,prefix in enumerate(PREFIX):
            for si,seed in enumerate(SEEDS):
                view=pi*2+si;clean=str(OUT/'rgb'/f'{c["key"]}_p{pi}_s{si}_clean.png')
                for arm in ['clean','bit0','bit1']:
                    inp=next(r for r in result['inputrows'] if r['arm']==arm);job=f'{c["key"]}_p{pi}_s{si}_{arm}'
                    plan.append(dict(job=job,key=c['key'],content_id=cid,split='Independent24',mode='S',stratum=c['stratum'],arm=arm,view=view,prompt_index=pi,seed_index=si,
                        prompt=prefix+c['caption'],seed=seed,hint=inp['hint'],hint_sha256=inp['hint_sha256'],rgb=str(OUT/'rgb'/f'{job}.png'),clean_rgb=clean,noise=str(OUT/'noise'/f'{c["key"]}_s{si}.npy'),generate=True))
        print('PREPARED',cid,'curves',result['source_candidates'],'eligible',result['eligible'],'independent',result['independent_lower_bound'],flush=True)
    dump(OUT/'selection.json',selected);dump(OUT/'inputs.json',inputs);dump(OUT/'plan.json',plan)
    protected=[OUT/n for n in ['selection.json','inputs.json','plan.json','content_sample.json','exclusions.json']]+list((OUT/'conditions').glob('*.png'))+list((OUT/'masks').glob('*.png'))+list((OUT/'capacity').glob('*.json'))
    dump(OUT/'input_freeze.json',dict(protected_sha256={str(p):digest(p) for p in protected},freeze_sha256=digest(OUT/'freeze.json'),contents=24,unique_rgb=288,selected_input_admissible=sum(c['input_admissible'] for c in selected),policy='All input-only choices frozen before final generation;24 fixed contents, no replacement.'))
    print('INPUT_FREEZE_COMPLETE',len(plan),flush=True)

def worker():
    import torch
    from cldm.ddim_hacked import DDIMSampler
    from controlnet_robin.pipeline import ControlNetROBINPipeline
    f=check();gen=f['generation'];inputs=read(OUT/'input_freeze.json')
    for p,h in inputs['protected_sha256'].items():assert digest(p)==h,p
    for p,h in read(S1/'protocol.json')['frozen_model_code'].items():assert digest(p)==h,p
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and torch.cuda.is_available();assert digest(ROOT/gen['checkpoint'])==gen['checkpoint_sha256']
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    class DeviceSampler(DDIMSampler):
        def register_buffer(self,name,attr):setattr(self,name,attr.to(self.model.device) if isinstance(attr,torch.Tensor) else attr)
    pipe=ControlNetROBINPipeline(str(ROOT/gen['model_config']),str(ROOT/gen['checkpoint']),'cuda');pipe.model.requires_grad_(False).eval();pipe.model.cond_stage_model.device='cuda';pipe.ddim=DeviceSampler(pipe.model);pipe.make_schedule(gen['steps'],eta=gen['eta']);pipe.model.control_scales=[gen['control_scale']]*13
    plan=read(OUT/'plan.json')
    for ix,j in enumerate(plan):
        record=OUT/'records'/f'{j["job"]}.json'
        if record.exists():assert digest(j['rgb'])==read(record)['rgb_sha256'];continue
        assert digest(j['hint'])==j['hint_sha256'];noise=torch.randn((1,4,64,64),generator=torch.Generator(device='cpu').manual_seed(j['seed']))
        if Path(j['noise']).exists():assert np.array_equal(np.load(j['noise']),noise.numpy())
        else:np.save(j['noise'],noise.numpy())
        cond,uncond=pipe.get_conditioning(j['prompt'],np.array(Image.open(j['hint']).convert('RGB')),'',512,False);latent=noise.clone().cuda();start=time.monotonic()
        with torch.inference_mode():
            for step in range(pipe.total_steps):latent=pipe.ddim_step(latent,step,cond,uncond,gen['guidance'])
            rgb=((pipe.model.decode_first_stage(latent)+1)/2).clamp(0,1);im=(rgb[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
        Image.fromarray(im).save(j['rgb']);dump(record,dict(**j,rgb_sha256=digest(j['rgb']),noise_sha256=digest(j['noise']),freeze_sha256=digest(OUT/'freeze.json'),checkpoint_sha256=gen['checkpoint_sha256'],elapsed_seconds=time.monotonic()-start))
        print('GEN',ix+1,'/288',j['job'],flush=True)
    dump(OUT/'generation_status.json',dict(status='complete',unique_rgb=288))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare','worker']);a=p.parse_args();globals()[a.action]()
