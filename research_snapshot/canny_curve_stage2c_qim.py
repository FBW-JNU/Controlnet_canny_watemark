"""Cal-only rejection development and Stage3A/B single-bit QIM experiment."""
from __future__ import annotations
import argparse,itertools,math,os,time,shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.special import expit
from canny_curve_stage2b import ROOT,OUT as B,OLD,S1,read,dump,sha,save,now,presence_fit,presence_predict,metrics
from canny_curve_stage2_bit import matcher,canonical
from analyze_canny_curve_stage2_bit import auc,distribution
from curve_receiver_stage2c import geometry,candidates,receive,prepare_target,oracle_mask
from curve_qim_stage3 import DELTAS,KEY,phase,encode,decode

OUT=ROOT/'outputs/canny_curve_stage2c_qim1_20261110'
THRESHOLDS=read(OLD/'protocol.json')['matcher_thresholds']
GATES=[dict(coverage=c,residual=r,margin=m,offset=o) for c,r,m,o in itertools.product([.65,.75,.85],[.6,.9,1.2],[0.,.08,.18],[1.5,2.5])]
CONTROL_NAMES=['shift_x24','shift_y24','near_normal_plus8','near_normal_minus8','wrong_anchor','wrong_key_carrier']

def check():
    p=read(OUT/'protocol.json')
    for path,digest in p['protected_sha256'].items():assert sha(path)==digest,path
    return p

def old_jobs(split):return [j for j in read(OLD/'plan.json') if j['split']==split and (j['arm']=='clean' or j['mode']=='polynomial')]
def old_cases():return {c['key']:c for c in read(OLD/'selection.json')}

def prepare():
    if (OUT/'protocol.json').exists():check();return
    for name in ['source','logs','cal_conditions','cal_rgb','cal_records','cal_noise','cal_extract','diag_extract','fresh_extract','fresh','plots','panels']:(OUT/name).mkdir(parents=True,exist_ok=True)
    cc=[c for c in old_cases().values() if c['split']=='Cal'];rng=np.random.default_rng(2026111037);qcal=[]
    for kind in ['straight','mild','bent']:
        pool=[c for c in cc if c['stratum']==kind];rng.shuffle(pool);qcal+=pool[:2]
    dump(OUT/'qim_cal_selection.json',qcal);plan=[];inputs=[];original=old_jobs('Cal')
    for c in qcal:
        edge=np.array(Image.open(c['clean_hint']).convert('L'));mask=np.array(Image.open(c['mask']).convert('L'))>0
        for delta in DELTAS:
            for bit in [0,1]:
                image,r=encode(edge,c['samples'],mask,c['key'],delta,bit);path=OUT/'cal_conditions'/f'{c["key"]}_d{delta}_b{bit}.png';save(path,image)
                inputs.append(dict(key=c['key'],hint=str(path),hint_sha256=sha(path),**r))
                for view in [0,1]:
                    clean=next(j for j in original if j['key']==c['key'] and j['view']==view and j['arm']=='clean');job=f'{c["key"]}_v{view}_qim_d{delta}_bit{bit}'
                    plan.append(dict(job=job,key=c['key'],content_id=c['content_id'],stratum=c['stratum'],split='CalQIM',mode='QIM',delta=delta,arm=f'bit{bit}',view=view,
                        prompt=clean['prompt'],seed=clean['seed'],hint=str(path),hint_sha256=sha(path),rgb=str(OUT/'cal_rgb'/f'{job}.png'),noise=str(OUT/'cal_noise'/f'{c["key"]}_v{view}.npy'),clean_rgb=clean['rgb'],generate=True))
        for view in [0,1]:
            clean=next(j for j in original if j['key']==c['key'] and j['view']==view and j['arm']=='clean')
            plan.append(dict(**{k:v for k,v in clean.items() if k not in ['split','mode']},split='CalQIM',mode='clean',delta=None,clean_rgb=clean['rgb'],generate=False))
        print('INPUT QIM',c['key'],flush=True)
    dump(OUT/'qim_cal_inputs.json',inputs);dump(OUT/'qim_cal_plan.json',plan)
    files=[ROOT/name for name in ['curve_receiver_stage2c.py','curve_qim_stage3.py','canny_curve_stage2c_qim.py']]
    files+=[B/'final_artifacts_sha256.json',OLD/'protocol.json',OLD/'selection.json',OLD/'plan.json',OUT/'qim_cal_selection.json',OUT/'qim_cal_inputs.json',OUT/'qim_cal_plan.json']
    protected={str(p):sha(p) for p in files};protected.update({i['hint']:i['hint_sha256'] for i in inputs})
    for p in files[:3]:shutil.copyfile(p,OUT/'source'/p.name)
    protocol=dict(created_utc=now(),protected_sha256=protected,old_data='Stage1/2/2b read-only; old2b FreshTest now diagnostic, including exactly its60 disjoint translated controls.',
        receiver='Unchanged R0 search then confidence rejection; A3 fallback, also gated. Original3px/4px/6px strict parameters unchanged. Source-normal, order, tangent, information, Huber residual, offset, anchor support and distinct-best/runner-up margin. No bit templates.',
        ambiguity='Alternative Viterbi path after removing selected actual observations within1.25px. R0 and A3 compared only if geometrically distinct. Missing runner-up recorded, not a calibrated uniqueness certificate.',
        controls=CONTROL_NAMES,control_caveat='Wrong-key carrier is another real source curve: acceptance can be geometrically correct yet assigned to the wrong carrier. Report separately; geometry alone cannot authenticate a key.',
        Cal='12 original Cal contents/72 existing PNGs. Fixed54 gate settings. Eligible marked>=.70, bitAUC>=.85, worst acceptance over disjoint24px/near-normal8px/wrong-anchor <=1/6. Choose max coverage then min negative then AUC. If none, select max coverage-2*worst_negative diagnostically and flag unqualified. Presence density Cal-only after routing. No old Dev/2b FreshTest tuning.',
        gates=GATES,stage3_basis='16t²(1-t)², distinct from S basis used in Stage2/2b. Displacement is along fixed source normals.',
        QIM='Keyed coordinate z=c-dither(key,carrier). In z:Q0={2kΔ},Q1={(2k+1)Δ}. Nearest parity lattice; encoder raster feedback fixed target, alpha<=2.65px. One bit, no RS/RM or authentication claim.',
        QIM_Cal='6 originalCal contents,2/stratum,2 existingCal prompt/seed views; Delta=[1,1.5,2]px.72new marked PNGs,12 reused clean. All attempts retained. Input success>=99% is checked before generation.',
        QIM_Oracle='Fixed original curve/tube and actual RGB Canny observations, no bit-dependent location/template. OracleMask supplies source location only; it does not provide true correspondence. Raw and Cal-affine decoding, plus paired-clean reference-assisted diagnostic separately. Never claim perfect correspondence from knownMask.',
        QIM_selection='Cal only, input success>=.99 and joint quality>=.95; max OracleMask Cal-affine bit AUC then lower Delta. If no qualified, diagnostic best flagged. Affine c_out=intercept+gain*c_in, fit all acceptedCal clean andmarked for eachDelta; positivegain floor.1. Presence1D Gaussian on lattice residual, Cal only.',
        fresh='12 newcontents excluded all32Stage1+12Stage2bFresh.4/source stratum;6cleanprobe views/content, frozenR0 stable>=5/6; first source-admissible curve, fallback retained. Frozen newreceiver first; then2fresh prompt/seed pairedclean/Sbit0/Sbit1,72PNGs. SelectedQIM adds48marked, shares24clean. All content and messages fixed before final test.',
        quality_gate=read(OLD/'protocol.json')['quality_gate'],generation=read(OLD/'protocol.json')['generation'])
    dump(OUT/'protocol.json',protocol);dump(OUT/'prior_stage2b_freeze.json',read(B/'final_artifacts_sha256.json'));print('PROTOCOL FROZEN',flush=True)

def input_summary():
    rows=read(OUT/'qim_cal_inputs.json');s={}
    for d in DELTAS:
        rr=[r for r in rows if r['delta']==d];s[str(d)]=dict(attempts=len(rr),successes=sum(r['input_success'] for r in rr),success_rate=float(np.mean([r['input_success'] for r in rr])),
            raster_error=distribution(r.get('quantization_error') for r in rr),changed_pixels=distribution(r.get('changed_pixels') for r in rr))
    dump(OUT/'stage3a_cal_summary.json',s);return s

def controls(c):
    from scipy.spatial import cKDTree
    p,u,n,phi=geometry(c['samples']);out={}
    for axis,name in [(0,'shift_x24'),(1,'shift_y24')]:
        d=np.zeros(2);d[axis]=24 if p[:,axis].max()+24<504 else -24;q=p+d
        out[name]=dict(samples=q.tolist(),disjoint16=bool(cKDTree(p).query(q)[0].min()>16))
    out['near_normal_plus8']=dict(samples=(p+8*n).tolist());out['near_normal_minus8']=dict(samples=(p-8*n).tolist())
    jitter=np.interp(u,[0,.2,.4,.6,.8,1],[0,8,-8,8,-8,0]);out['wrong_anchor']=dict(samples=(p+jitter[:,None]*n).tolist())
    atlas=read(S1/'curves'/f'{c["content_id"]:012d}_source.json')['curves'] if c.get('split')!='Fresh2c' else c['source_atlas']
    other=[a for a in atlas if a['curve_index']!=c['curve_index']]
    if other:
        other.sort(key=lambda a:(float(np.linalg.norm(np.mean(a['samples'],0)-p.mean(0))),a['curve_index']));a=other[0]
        out['wrong_key_carrier']=dict(samples=a['samples'],other_curve_index=a['curve_index'],real_other_carrier=True)
    return out

def extract(split='Cal'):
    check();fresh=split=='Fresh2c';diag=split=='FixedDiag'
    cc={c['key']:c for c in read(OUT/'fresh/selection.json')} if fresh else {c['key']:c for c in read(B/'fresh/selection.json')} if diag else old_cases()
    plan=read(OUT/'fresh/s_plan.json') if fresh else read(B/'fresh/test_plan.json') if diag else old_jobs('Cal')
    folder=OUT/('fresh_extract' if fresh else 'diag_extract' if diag else 'cal_extract');m=matcher()
    for i,j in enumerate(plan):
        dest=folder/f'{j["job"]}.json'
        if dest.exists():continue
        c=cc[j['key']];target=prepare_target(np.array(Image.open(j['rgb']).convert('RGB')));main=candidates(c['samples'],target,THRESHOLDS,'S');negative={}
        for name,spec in controls(c).items():negative[name]=dict(**spec,extracted=candidates(spec['samples'],target,THRESHOLDS,'S'))
        dump(dest,dict(job=j,main=main,controls=negative,rgb_sha256=sha(j['rgb'])));print('EXTRACT',split,i+1,'/',len(plan),flush=True)

def score_rows(rows,gate,head=None):
    rr=[];neg={name:[] for name in CONTROL_NAMES};disjoint=[]
    for row in rows:
        r=receive(row['main'],gate);c=next(c for c in (read(OUT/'fresh/selection.json') if row['job'].get('split')=='Fresh2c' else read(B/'fresh/selection.json') if row['job'].get('split')=='FreshTest' else read(OLD/'selection.json')) if c['key']==row['job']['key'])
        # Bit polarity normalization is public input metadata, not matching.
        if r['accepted']:
            book=c['input_features']['polynomial'];r['bit_score']=(r['c_hat']-book['midpoint'])/book['half_separation'];r['normalized_se']=r['coefficient_se']/book['half_separation'];r['absolute_presence']=abs((r['c_hat']-book['clean'])/book['half_separation'])
        else:r.update(bit_score=0.,normalized_se=None,absolute_presence=0.)
        r['wrong_location_acceptance']=[]
        for name,spec in row['controls'].items():
            accepted=receive(spec['extracted'],gate)['accepted'];neg[name].append(accepted)
            if name in ['shift_x24','shift_y24']:
                r['wrong_location_acceptance'].append(accepted)
                if spec['disjoint16']:disjoint.append(accepted)
        if head:r['s_presence']=presence_predict(r,head)
        rr.append((row['job'],r))
    stats=metrics(rr);negative={name:dict(accepted=sum(v),attempts=len(v),rate=float(np.mean(v)) if v else None) for name,v in neg.items()}
    negative['disjoint24']=dict(accepted=sum(disjoint),attempts=len(disjoint),rate=float(np.mean(disjoint)) if disjoint else 0.)
    stats['negative_controls']=negative;stats['worst_negative']=max(negative[n]['rate'] or 0 for n in ['disjoint24','near_normal_plus8','near_normal_minus8','wrong_anchor'])
    return rr,stats

def freeze_receiver():
    check();assert not (OUT/'receiver_freeze.json').exists();rows=[read(p) for p in sorted((OUT/'cal_extract').glob('*.json'))];assert len(rows)==72;grid=[]
    for gi,g in enumerate(GATES):
        rr,s=score_rows(rows,g);eligible=s['marked_coverage']>=.7 and s['bit_auc_itt']>=.85 and s['worst_negative']<=1/6
        grid.append(dict(index=gi,gate=g,eligible=eligible,metrics=s))
    eligible=[r for r in grid if r['eligible']]
    chosen=max(eligible,key=lambda r:(r['metrics']['marked_coverage'],-r['metrics']['worst_negative'],r['metrics']['bit_auc_itt'])) if eligible else max(grid,key=lambda r:(r['metrics']['marked_coverage']-2*r['metrics']['worst_negative'],r['metrics']['bit_auc_itt']))
    rr,s=score_rows(rows,chosen['gate']);head=presence_fit(rr)
    for j,r in rr:r['s_presence']=presence_predict(r,head)
    chosen=dict(chosen,presence_model=head,Cal_calibrated_metrics=metrics(rr))
    dump(OUT/'receiver_freeze.json',dict(created_utc=now(),selected=chosen,qualified=bool(eligible),grid=grid,protocol_sha256=sha(OUT/'protocol.json'),Cal_result_sha256={str(p):sha(p) for p in sorted((OUT/'cal_extract').glob('*.json'))},policy='Frozen before fixed old60 diagnostic evaluation and newcontent selection. No Dev optimization.'))
    print('FROZEN RECEIVER',chosen['gate'],'qualified',bool(eligible),flush=True)

def receiver_summary(split):
    d=read(OUT/'receiver_freeze.json');folder=OUT/('fresh_extract' if split=='Fresh2c' else 'diag_extract' if split=='FixedDiag' else 'cal_extract');rows=[read(p) for p in sorted(folder.glob('*.json'))];assert len(rows)==72
    rr,s=score_rows(rows,d['selected']['gate'],d['selected']['presence_model']);dump(OUT/f'receiver_{split}_summary.json',s)
    dump(OUT/f'receiver_{split}_observations.json',[dict(job=j,result=r) for j,r in rr]);print('SUMMARY',split,s['marked_received'],s['bit_auc_itt'],flush=True)

def worker_cal():
    check();summary=input_summary();eligible={float(d) for d,s in summary.items() if s['success_rate']>=.99};plan=[j for j in read(OUT/'qim_cal_plan.json') if j['generate'] and j['delta'] in eligible]
    dump(OUT/'qim_generated_delta_eligibility.json',dict(created_utc=now(),eligible=sorted(eligible),policy='Only inputclosedloop>=99% candidate intervals generated. Failed interval inputattempts retained and reported.'))
    generate(plan,OUT/'cal_records');dump(OUT/'cal_generation_status.json',dict(status='complete',new_pngs=len(plan)))

def generate(plan,records):
    import torch
    from cldm.ddim_hacked import DDIMSampler
    from controlnet_robin.pipeline import ControlNetROBINPipeline
    gen=read(OLD/'protocol.json')['generation'];assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and torch.cuda.is_available();assert sha(ROOT/gen['checkpoint'])==gen['checkpoint_sha256']
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    class DeviceSampler(DDIMSampler):
        def register_buffer(self,name,attr):setattr(self,name,attr.to(self.model.device) if isinstance(attr,torch.Tensor) else attr)
    pipe=ControlNetROBINPipeline(str(ROOT/gen['model_config']),str(ROOT/gen['checkpoint']),'cuda');pipe.model.requires_grad_(False).eval();pipe.model.cond_stage_model.device='cuda';pipe.ddim=DeviceSampler(pipe.model);pipe.make_schedule(gen['steps'],eta=gen['eta']);pipe.model.control_scales=[gen['control_scale']]*13
    def run(items,folder):
        folder.mkdir(parents=True,exist_ok=True)
        for i,j in enumerate(items):
            record=folder/f'{j["job"]}.json'
            if record.exists():assert sha(j['rgb'])==read(record)['rgb_sha256'];continue
            assert sha(j['hint'])==j['hint_sha256'];noise=torch.randn((1,4,64,64),generator=torch.Generator(device='cpu').manual_seed(j['seed']))
            Path(j['noise']).parent.mkdir(parents=True,exist_ok=True)
            if Path(j['noise']).exists():assert np.array_equal(np.load(j['noise']),noise.numpy())
            else:np.save(j['noise'],noise.numpy())
            cond,uncond=pipe.get_conditioning(j['prompt'],np.array(Image.open(j['hint']).convert('RGB')),'',512,False);latent=noise.clone().cuda();start=time.monotonic()
            with torch.inference_mode():
                for step in range(pipe.total_steps):latent=pipe.ddim_step(latent,step,cond,uncond,gen['guidance'])
                rgb=((pipe.model.decode_first_stage(latent)+1)/2).clamp(0,1);image=(rgb[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
            save(j['rgb'],image);dump(record,dict(**j,rgb_sha256=sha(j['rgb']),noise_sha256=sha(j['noise']),checkpoint_sha256=gen['checkpoint_sha256'],elapsed_seconds=time.monotonic()-start,protocol_sha256=sha(OUT/'protocol.json')))
            print('GEN',i+1,'/',len(items),j['job'],flush=True)
    if callable(plan):plan(run)
    else:run(plan,records)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','input','extract','freeze_receiver','receiver_summary','worker_cal']);a.add_argument('--split',default='Cal');args=a.parse_args()
    if args.action=='extract':extract(args.split)
    elif args.action=='receiver_summary':receiver_summary(args.split)
    elif args.action=='input':print(input_summary())
    else:globals()[args.action]()
