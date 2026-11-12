"""Single-curve S/T four-word experiment. Cal-only channel estimation."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mat_transport_mpl')
from experiment_s_frozen_validation import (
    ROOT, PREV as BROAD, S1, THRESHOLDS, GATE, MANIFEST, PREFIX,
    read,dump,digest,arr,local_matcher,mask_for,geometry,
)
from curve_broad_peak_code import basis,receive
from curve_qim_closed_loop_v2 import RasterWarp
from curve_receiver_stage2c import binary_target,prepare_target
from carrier_screen.common import pixel_metrics
from analyze_canny_curve_stage2_bit import auc
from pathlib import Path
from PIL import Image
from scipy.spatial import cKDTree
from scipy.special import logsumexp
import numpy as np
import cv2,json,csv,re,subprocess,time,shutil,argparse

PRIOR=ROOT/'outputs/canny_s_frozen_validation_20261111'
OUT=ROOT/'outputs/canny_single_curve_2bit_stage4a_20261111'
SIGNS={'clean':[0,0],'S0':[-1,0],'S1':[1,0],'T0':[0,-1],'T1':[0,1],
       'm00':[-1,-1],'m01':[-1,1],'m10':[1,-1],'m11':[1,1]}
WORDS=['m00','m01','m10','m11']
DENSE=np.linspace(0,1,100001)
PEAK=float(np.max(np.sum(abs(basis(DENSE)),axis=1)))
AMPLITUDE=1.5/PEAK
SEEDS={'Cal':[2026114201,2026114202],'Eval':[2026114301,2026114302]}


def freeze():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():return check()
    for d in ['source','sources','hints','masks','conditions','inputs','rgb','noise','records','measurements','panels','plots','logs']:
        (OUT/d).mkdir(exist_ok=True)
    scripts=['curve_single_2bit_stage4a.py','curve_broad_peak_code.py','broad_peak_runtime.py',
             'curve_receiver_stage2c.py','curve_receiver_stage2b.py','curve_qim_closed_loop_v2.py','experiment_s_frozen_validation.py']
    for n in scripts:shutil.copyfile(ROOT/n,OUT/'source'/n)
    p=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        source_sha256={str(ROOT/n):digest(ROOT/n) for n in scripts},
        prior_manifest_sha256=digest(PRIOR/'final_artifacts_sha256.json'),
        basis=dict(S='6sqrt(3)t(1-t)(2t-1)',T='4t(1-t)(5(2t-1)^2-1)',
            S_unchanged=True,T_is_existing_joint_receiver_basis=True,
            orthogonality='Continuous uniform arc-length integral S*T=0; both have zero mean. Pixel observations need not stay orthogonal.'),
        encoder=dict(arms=SIGNS,coefficient_amplitude_px=AMPLITUDE,combined_peak_bound_px=1.5,normalization=PEAK,
            rule='Single normal displacement a1*S+a2*T, then one original binary rasterization; never sequential image warps.',
            orientation='Original lexicographic canonical endpoints and normalized arc length; fixed tube mask, exterior unchanged.'),
        receiver=dict(implementation='Exact prior joint S/T receiver, mode=S; bit-blind R0 priority/A3 fallback, unchanged gates.',gate=GATE,thresholds=THRESHOLDS,
            limitation='Existing information gate measures S coverage. Joint covariance guards both coefficients. T-specific coverage recorded descriptively, no new gate.'),
        input_admission=dict(rule='First descending-length curve with nine accepted actual raster states in both heads; four distinct marked PNGs; nearest ideal joint sign center recovers all four; minimum coefficient separation>=0.35px.',
            minimum_separation_px=.35,no_rgb_selection=True),
        design=dict(cal_contents=6,new_eval_contents=12,views_per_content=2,arms_per_view=9,total_rgb=324,
            cal_source='Six uniformly sampled prior Fresh24 contents, only new 2-bit PNG used for fitting.',
            eval_source='Twelve uniform COCO normal train contents excluding all currently observable historical IDs.',
            view0='Photo prompt and seed0',view1='Watercolor prompt and seed1',seeds=SEEDS,
            no_content_replacement=True,missing_historical_manifests_limit_exclusion=True),
        calibration=dict(features='Observed S/T coefficients minus source clean-input coefficients.',
            channel='OLS y=b+A*x on accepted Cal clean, axis and four-word states; actual raster-input x, no Eval fitting.',
            classifier='Four predicted actual-input centroids, shared Cal residual covariance Mahalanobis nearest centroid; no expected word in synchronization.',
            covariance_eigenvalue_floor_px2=.01,presence='Cal clean Gaussian vs uniform four-centroid Gaussian mixture; Cal-only residual covariances, clean95% threshold.',
            erasure='Geometry/input book unavailable: no word; full-message failure. Presence score=-1e6, rejected.',
            finite_difference='Central nominal-axis differences on Cal, plus joint-corner differences and input-domain response; missing pairs excluded with counts.'),
        generation=read(PRIOR/'freeze.json')['generation'],quality=read(PRIOR/'freeze.json')['quality'],
        targets=dict(input_full_word=.99,eval_all_attempt_full_word=.70),
        scope='Four-word modulation prototype; no RS/RM, no logical stitching, no claim that a 2-bit alphabet itself provides error correction.')
    dump(OUT/'protocol.json',p);return p


def check():
    f=read(OUT/'protocol.json')
    for p,h in f['source_sha256'].items():assert digest(p)==h,p
    return f


def render(edge,samples,mask,sign):
    w=RasterWarp(edge,samples,mask)
    vec=basis(w.u)@(AMPLITUDE*np.array(sign))
    w.vector=vec[w.near[w.chosen],None]*w.n[w.near[w.chosen]]
    return w.render(1.)[0]


def input_check(edge,samples,mask):
    images={arm:render(edge,samples,mask,s) for arm,s in SIGNS.items()}
    assert np.array_equal(images['clean'],edge)
    obs={arm:receive(samples,binary_target(im,samples),'S',THRESHOLDS,GATE) for arm,im in images.items()}
    ideal=AMPLITUDE*np.array([SIGNS[a] for a in WORDS]);details={};good=True
    for head in ['Auto','Oracle']:
        accepted=all(obs[a][head]['accepted'] for a in SIGNS)
        entry=dict(all_nine_received=accepted,decoded={},minimum_separation_px=None,correct=0)
        if accepted:
            origin=np.array(obs['clean'][head]['coefficient'])
            values=np.array([obs[a][head]['coefficient'] for a in WORDS])-origin
            pred=np.argmin(np.sum((values[:,None,:]-ideal[None,:,:])**2,axis=2),axis=1)
            sep=np.linalg.norm(values[:,None,:]-values[None,:,:],axis=2)+np.eye(4)*1e6
            entry.update(decoded={a:WORDS[int(i)] for a,i in zip(WORDS,pred)},correct=int(np.sum(pred==np.arange(4))),minimum_separation_px=float(sep.min()))
        details[head]=entry
        good=good and accepted and entry['correct']==4 and entry['minimum_separation_px']>=.35
    distinct=len({__import__('hashlib').sha256(images[a].tobytes()).hexdigest() for a in WORDS})==4
    active=all(np.any(images[a]!=edge) for a in WORDS)
    return images,obs,dict(admissible=bool(good and distinct and active),distinct_four=distinct,all_marked_active=active,heads=details)


def prepare():
    f=freeze()
    if (OUT/'input_freeze.json').exists():return
    if (OUT/'content_sample.json').exists():sample=read(OUT/'content_sample.json')
    else:
        old=read(PRIOR/'selection.json');rng=np.random.default_rng(2026111149)
        cal=[old[int(i)] for i in rng.choice(len(old),6,replace=False)]
        excluded=set(read(PRIOR/'exclusions.json')['ids'])|{c['content_id'] for c in old}
        inventory=subprocess.check_output(['rg','--files','outputs'],cwd=ROOT,text=True).splitlines()
        for path in inventory:excluded.update(int(i) for i in re.findall(r'(?<!\d)(0\d{11})(?!\d)',path) if 0<int(i)<600000)
        rows=[r for r in csv.DictReader(MANIFEST.open()) if int(r['image_id']) not in excluded]
        fresh=[rows[int(i)] for i in rng.choice(len(rows),12,replace=False)]
        sample=[dict(split='Cal',content_id=c['content_id'],image_path=c['source_original'],caption=c['caption']) for c in cal]
        sample += [dict(split='Eval',content_id=int(r['image_id']),image_path=r['image_path'],caption=r['prompt'].strip()) for r in fresh]
        dump(OUT/'exclusions.json',dict(ids=sorted(excluded),missing_historical_manifests=True))
        dump(OUT/'content_sample.json',sample)
    m=local_matcher();cv2.setNumThreads(1);selected=[];plan=[]
    for row in sample:
        cid=row['content_id'];dest=OUT/'inputs'/f'{cid:012d}.json'
        if dest.exists():c=read(dest)
        else:
            original=Image.open(row['image_path']).convert('RGB');w,h=[round(v*512/max(original.size)) for v in original.size];x,y=(512-w)//2,(512-h)//2
            canvas=Image.new('RGB',(512,512),(127,127,127));canvas.paste(original.resize((w,h),Image.Resampling.LANCZOS),(x,y))
            source=OUT/'sources'/f'{cid:012d}.png';canvas.save(source);edge=cv2.Canny(np.array(canvas),100,200)
            hint=OUT/'hints'/f'{cid:012d}.png';Image.fromarray(edge).save(hint);roi=[x+8,y+8,x+w-8,y+h-8];valid=np.zeros_like(edge);valid[roi[1]:roi[3],roi[0]:roi[2]]=edge[roi[1]:roi[3],roi[0]:roi[2]]
            curves=sorted(m.source_curves(valid,THRESHOLDS)[0],key=lambda c:(-c['arc_length_px'],c['curve_index']));tried=[];chosen=None
            for curve in curves:
                mask=mask_for(edge,curve['samples'],roi);images,observations,admission=input_check(edge,curve['samples'],mask)
                tried.append(dict(curve_index=curve['curve_index'],arc_length_px=curve['arc_length_px'],**admission))
                if admission['admissible']:
                    chosen=(curve,mask,images,observations,admission);break
            if chosen is None:
                assert curves,('No curve: explicit absent-carrier handling required',cid)
                curve=curves[0];mask=mask_for(edge,curve['samples'],roi);images,observations,admission=input_check(edge,curve['samples'],mask);chosen=curve,mask,images,observations,admission
            curve,mask,images,observations,admission=chosen
            key=f'{cid:012d}_c{curve["curve_index"]:04d}';mp=OUT/'masks'/f'{key}.png';Image.fromarray(mask.astype('uint8')*255).save(mp)
            states={}
            for arm,image in images.items():
                path=OUT/'conditions'/f'{key}_{arm}.png';Image.fromarray(image).save(path)
                assert set(np.unique(image))<={0,255} and np.array_equal(image[~mask],edge[~mask])
                states[arm]=dict(hint=str(path),hint_sha256=digest(path),changed_pixels=int(np.count_nonzero(image!=edge)),measurements=observations[arm])
            c=dict(**row,key=key,samples=curve['samples'],curve_index=curve['curve_index'],arc_length_px=curve['arc_length_px'],
                source_rgb=str(source),source_sha256=digest(row['image_path']),clean_hint=str(hint),mask=str(mp),roi=roi,
                admission=admission,source_candidate_count=len(curves),candidates_tried=tried,states=states)
            dump(dest,c)
        selected.append(c)
        for view in range(2):
            seed=SEEDS[c['split']][view];prompt=PREFIX[view]+c['caption'];clean=str(OUT/'rgb'/f'{c["key"]}_v{view}_clean.png')
            for arm in SIGNS:
                job=f'{c["key"]}_v{view}_{arm}'
                plan.append(dict(job=job,key=c['key'],content_id=cid,split=c['split'],view=view,arm=arm,
                    prompt=prompt,seed=seed,hint=c['states'][arm]['hint'],hint_sha256=c['states'][arm]['hint_sha256'],
                    rgb=str(OUT/'rgb'/f'{job}.png'),clean_rgb=clean,noise=str(OUT/'noise'/f'{c["key"]}_v{view}.npy')))
        print('PREPARED',c['split'],cid,'length',c['arc_length_px'],'tried',len(c['candidates_tried']),'admitted',c['admission']['admissible'],flush=True)
    dump(OUT/'selection.json',selected);dump(OUT/'plan.json',plan)
    protected=[OUT/n for n in ['selection.json','plan.json','content_sample.json','exclusions.json']]+list((OUT/'conditions').glob('*.png'))+list((OUT/'masks').glob('*.png'))+list((OUT/'inputs').glob('*.json'))
    dump(OUT/'input_freeze.json',dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),protected_sha256={str(p):digest(p) for p in protected},unique_rgb=len(plan)))
    print('INPUT_FREEZE_COMPLETE',len(plan),flush=True)


def worker():
    import torch
    from cldm.ddim_hacked import DDIMSampler
    from controlnet_robin.pipeline import ControlNetROBINPipeline
    f=check();gen=f['generation']
    for p,h in read(OUT/'input_freeze.json')['protected_sha256'].items():assert digest(p)==h,p
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and torch.cuda.is_available()
    assert digest(ROOT/gen['checkpoint'])==gen['checkpoint_sha256']
    for p,h in read(S1/'protocol.json')['frozen_model_code'].items():assert digest(p)==h,p
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    class DeviceSampler(DDIMSampler):
        def register_buffer(self,name,attr):setattr(self,name,attr.to(self.model.device) if isinstance(attr,torch.Tensor) else attr)
    pipe=ControlNetROBINPipeline(str(ROOT/gen['model_config']),str(ROOT/gen['checkpoint']),'cuda');pipe.model.requires_grad_(False).eval();pipe.model.cond_stage_model.device='cuda';pipe.ddim=DeviceSampler(pipe.model)
    pipe.make_schedule(gen['steps'],eta=gen['eta']);pipe.model.control_scales=[gen['control_scale']]*13
    for ix,j in enumerate(read(OUT/'plan.json')):
        if j['split']=='Eval':
            while not (OUT/'score_freeze.json').exists():time.sleep(2)
            assert read(OUT/'score_freeze.json')['protocol_sha256']==digest(OUT/'protocol.json')
        record=OUT/'records'/f'{j["job"]}.json'
        if record.exists():assert digest(j['rgb'])==read(record)['rgb_sha256'];continue
        assert digest(j['hint'])==j['hint_sha256'];noise=torch.randn((1,4,64,64),generator=torch.Generator(device='cpu').manual_seed(j['seed']))
        if Path(j['noise']).exists():assert np.array_equal(np.load(j['noise']),noise.numpy())
        else:np.save(j['noise'],noise.numpy())
        cond,uncond=pipe.get_conditioning(j['prompt'],np.array(Image.open(j['hint']).convert('RGB')),'',512,False);latent=noise.clone().cuda();start=time.monotonic()
        with torch.inference_mode():
            for step in range(pipe.total_steps):latent=pipe.ddim_step(latent,step,cond,uncond,gen['guidance'])
            rgb=((pipe.model.decode_first_stage(latent)+1)/2).clamp(0,1);im=(rgb[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
        Image.fromarray(im).save(j['rgb'])
        dump(record,dict(**j,rgb_sha256=digest(j['rgb']),noise_sha256=digest(j['noise']),checkpoint_sha256=gen['checkpoint_sha256'],
            protocol_sha256=digest(OUT/'protocol.json'),score_freeze_sha256=digest(OUT/'score_freeze.json') if j['split']=='Eval' else None,elapsed_seconds=time.monotonic()-start))
        print('GEN',ix+1,'/324',j['split'],j['job'],flush=True)
    dump(OUT/'generation_status.json',dict(status='complete',unique_rgb=324))


def extract():
    check();cc={c['key']:c for c in read(OUT/'selection.json')}
    for ix,j in enumerate(read(OUT/'plan.json')):
        dest=OUT/'measurements'/f'{j["job"]}.json'
        if dest.exists():continue
        while not (OUT/'records'/f'{j["job"]}.json').exists():time.sleep(2)
        assert digest(j['rgb'])==read(OUT/'records'/f'{j["job"]}.json')['rgb_sha256']
        rgb=np.array(Image.open(j['rgb']).convert('RGB'));obs=receive(cc[j['key']]['samples'],prepare_target(rgb),'S',THRESHOLDS,GATE)
        dump(dest,dict(job=j['job'],rgb_sha256=digest(j['rgb']),observations=obs))
        print('EXTRACT',ix+1,'/324',j['job'],flush=True)


def covariance(values):
    v=np.asarray(values,float)
    cov=np.cov(v.T,bias=True) if len(v)>1 else np.eye(2)
    eig,vec=np.linalg.eigh(cov);return (vec*np.maximum(eig,.01))@vec.T


def features(c,arm,head):
    return np.array(c['states'][arm]['measurements'][head]['coefficient'])-np.array(c['states']['clean']['measurements'][head]['coefficient'])


def output_feature(c,obs,head):
    if not c['admission']['admissible'] or not obs[head]['accepted']:return None
    return np.array(obs[head]['coefficient'])-np.array(c['states']['clean']['measurements'][head]['coefficient'])


def gaussian_log(x,cov):
    cov=np.array(cov);x=np.asarray(x)
    return -.5*(np.einsum('...i,ij,...j->...',x,np.linalg.inv(cov),x)+np.linalg.slogdet(cov)[1]+2*np.log(2*np.pi))


def infer(c,obs,model,head):
    y=output_feature(c,obs,head)
    if y is None:return dict(accepted=False,word=None,bits=None,bit_scores=[0.,0.],presence_score=-1e6,presence_accept=False)
    a=np.array(model['A']);b=np.array(model['intercept']);centers=np.array([a@features(c,arm,head)+b for arm in WORDS]);ll=gaussian_log(y-centers,model['residual_covariance']);i=int(np.argmax(ll))
    bit_scores=[float(logsumexp(ll[[k for k in range(4) if WORDS[k][1+bit]=='1']])-logsumexp(ll[[k for k in range(4) if WORDS[k][1+bit]=='0']])) for bit in [0,1]]
    marked=logsumexp(gaussian_log(y-centers,model['marked_covariance']))-np.log(4)
    clean=gaussian_log(y-b-np.array(model['clean_residual_mean']),model['clean_covariance']);presence=float(marked-clean)
    return dict(accepted=True,word=WORDS[i],bits=WORDS[i][1:],feature=y.tolist(),bit_scores=bit_scores,presence_score=presence,presence_accept=presence>model['presence_threshold95'],class_loglikelihoods=ll.tolist())


def finite_differences(plan,cc,measurements,head):
    rows=[]
    for c in cc.values():
        if c['split']!='Cal':continue
        for view in range(2):
            jobs={j['arm']:j for j in plan if j['key']==c['key'] and j['view']==view}
            values={a:output_feature(c,measurements[j['job']],head) for a,j in jobs.items()}
            pairs={a:values[a] for a in SIGNS}
            axis=None;corners=None;interaction=None
            if all(pairs[a] is not None for a in ['S0','S1','T0','T1']):
                axis=np.column_stack([(pairs['S1']-pairs['S0'])/(2*AMPLITUDE),(pairs['T1']-pairs['T0'])/(2*AMPLITUDE)])
            if all(pairs[a] is not None for a in WORDS):
                corners=np.column_stack([(pairs['m10']+pairs['m11']-pairs['m00']-pairs['m01'])/(4*AMPLITUDE),(pairs['m01']+pairs['m11']-pairs['m00']-pairs['m10'])/(4*AMPLITUDE)])
                interaction=(pairs['m11']-pairs['m10']-pairs['m01']+pairs['m00'])/(4*AMPLITUDE**2)
            inp=np.column_stack([(features(c,'S1',head)-features(c,'S0',head))/(2*AMPLITUDE),(features(c,'T1',head)-features(c,'T0',head))/(2*AMPLITUDE)]) if c['admission']['admissible'] else None
            def matrix(m):
                return None if m is None else dict(A=m.tolist(),singular_values=np.linalg.svd(m,compute_uv=False).tolist(),condition=float(np.linalg.cond(m)),determinant=float(np.linalg.det(m)))
            rows.append(dict(key=c['key'],view=view,input=matrix(inp),output_axis=matrix(axis),output_corners=matrix(corners),interaction=interaction.tolist() if interaction is not None else None))
    result=dict(rows=rows)
    for field in ['input','output_axis','output_corners']:
        mats=[np.array(r[field]['A']) for r in rows if r[field] is not None]
        mean=np.mean(mats,axis=0) if mats else None
        result[field]=dict(complete_groups=len(mats),attempts=len(rows),mean=mean.tolist() if mean is not None else None,
            singular_values=np.linalg.svd(mean,compute_uv=False).tolist() if mean is not None else None,condition=float(np.linalg.cond(mean)) if mean is not None else None)
    return result


def calibrate():
    check()
    if (OUT/'score_freeze.json').exists():return
    plan=[j for j in read(OUT/'plan.json') if j['split']=='Cal'];cc={c['key']:c for c in read(OUT/'selection.json')}
    for j in plan:
        while not (OUT/'measurements'/f'{j["job"]}.json').exists():time.sleep(2)
    assert not any(j['split']=='Eval' for p in (OUT/'records').glob('*.json') for j in [read(p)]),'Eval RGB existed before score freeze'
    measurements={j['job']:read(OUT/'measurements'/f'{j["job"]}.json')['observations'] for j in plan};models={};fd={}
    for head in ['Auto','Oracle']:
        rows=[]
        for j in plan:
            c=cc[j['key']];y=output_feature(c,measurements[j['job']],head)
            if y is not None:rows.append((j,features(c,j['arm'],head),y))
        assert len(rows)>=12,('Insufficient Cal geometry',head,len(rows))
        x=np.array([np.r_[1,r[1]] for r in rows]);y=np.array([r[2] for r in rows]);beta=np.linalg.lstsq(x,y,rcond=None)[0];res=y-x@beta
        clean=np.array([r[2]-beta[0] for r in rows if r[0]['arm']=='clean']);marked=np.array([res[i] for i,r in enumerate(rows) if r[0]['arm'] in WORDS]);assert len(clean)>=3 and len(marked)>=6
        model=dict(A=beta[1:].T.tolist(),intercept=beta[0].tolist(),cal_received=len(rows),cal_attempts=len(plan),
            residual_covariance=covariance(res).tolist(),marked_covariance=covariance(marked).tolist(),
            clean_residual_mean=clean.mean(0).tolist(),clean_covariance=covariance(clean).tolist(),presence_threshold95=0.)
        clean_scores=[infer(cc[j['key']],measurements[j['job']],model,head)['presence_score'] for j in plan if j['arm']=='clean']
        model['presence_threshold95']=float(np.percentile(clean_scores,95));model['condition']=float(np.linalg.cond(np.array(model['A'])));models[head]=model
        fd[head]=finite_differences(plan,cc,measurements,head)
    dump(OUT/'cal_response_matrices.json',fd)
    dump(OUT/'score_freeze.json',dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),protocol_sha256=digest(OUT/'protocol.json'),
        models=models,finite_differences_sha256=digest(OUT/'cal_response_matrices.json'),
        cal_measurement_sha256={str(OUT/'measurements'/f'{j["job"]}.json'):digest(OUT/'measurements'/f'{j["job"]}.json') for j in plan},
        no_eval_rgb_before_calibration=True))
    print('SCORE_FROZEN',models,flush=True)


def summarize(rows):
    msg=[r for r in rows if r['arm'] in WORDS];clean=[r for r in rows if r['arm']=='clean'];good=[r for r in msg if r['accepted']]
    conf=np.zeros((4,5),int)
    for r in msg:conf[WORDS.index(r['arm']),WORDS.index(r['word']) if r['accepted'] else 4]+=1
    result=dict(attempts=len(msg),received=len(good),correct=sum(r['accepted'] and r['word']==r['arm'] for r in msg),
        wrong=sum(r['accepted'] and r['word']!=r['arm'] for r in msg),erasures=len(msg)-len(good),
        full_word_accuracy=sum(r['accepted'] and r['word']==r['arm'] for r in msg)/len(msg),
        receive_rate=len(good)/len(msg),confusion_matrix=conf.tolist(),confusion_columns=WORDS+['erasure'],
        clean_attempts=len(clean),clean_presence_accepted=sum(r['presence_accept'] for r in clean),marked_presence_accepted=sum(r['presence_accept'] for r in msg),
        presence_auc=auc([int(r['arm']!='clean') for r in clean+msg],[r['presence_score'] for r in clean+msg]))
    result['bits']=[]
    for bit in [0,1]:
        result['bits'].append(dict(bit_index=bit,correct=sum(r['accepted'] and r['bits'][bit]==r['arm'][1+bit] for r in msg),
            all_attempt_accuracy=sum(r['accepted'] and r['bits'][bit]==r['arm'][1+bit] for r in msg)/len(msg),
            auc=auc([int(r['arm'][1+bit]) for r in msg],[r['bit_scores'][bit] for r in msg])))
    return result


def evaluate():
    check();score=read(OUT/'score_freeze.json');assert score['protocol_sha256']==digest(OUT/'protocol.json')
    cc={c['key']:c for c in read(OUT/'selection.json')};plan=read(OUT/'plan.json');decoded=[]
    for j in plan:
        measurement=read(OUT/'measurements'/f'{j["job"]}.json');assert measurement['rgb_sha256']==digest(j['rgb'])
        for head in ['Auto','Oracle']:decoded.append(dict(job=j['job'],key=j['key'],split=j['split'],view=j['view'],arm=j['arm'],head=head,**infer(cc[j['key']],measurement['observations'],score['models'][head],head)))
    dump(OUT/'decoded.json',decoded);summary={}
    for split in ['Cal','Eval']:
        for head in ['Auto','Oracle']:summary[split+'_'+head]=summarize([r for r in decoded if r['split']==split and r['head']==head])
    dump(OUT/'summary.json',summary)
    # Freeze-compatible full-domain input check: actual PNG reload, ideal joint decoder.
    closure=[]
    for c in cc.values():
        edge=arr(c['clean_hint']);mask=arr(c['mask'])>0
        for arm in SIGNS:
            image=arr(c['states'][arm]['hint']);assert np.array_equal(image,render(edge,c['samples'],mask,SIGNS[arm]));assert digest(c['states'][arm]['hint'])==c['states'][arm]['hint_sha256']
        _,_,admission=input_check(edge,c['samples'],mask)
        closure.append(dict(key=c['key'],split=c['split'],**admission))
    dump(OUT/'input_closure.json',dict(contents=len(closure),unique_marked_states=4*len(closure),records=closure,
        policy='Actual saved binary PNG reloaded; nominal joint four-sign decoder, not self-centroid identity check. Admission is source-only and conditional; raw attempts retained.'))
    rng=np.random.default_rng(2026111151);intervals={}
    keys=[c['key'] for c in cc.values() if c['split']=='Eval']
    for head in ['Auto','Oracle']:
        blocks=[[r for r in decoded if r['key']==k and r['head']==head] for k in keys];values=[]
        for _ in range(2000):
            r=summarize([r for i in rng.integers(0,len(blocks),len(blocks)) for r in blocks[i]])
            values.append([r['full_word_accuracy'],r['receive_rate'],r['presence_auc']])
        intervals[head]=dict(zip(['full_word_accuracy','receive_rate','presence_auc'],np.percentile(values,[2.5,97.5],axis=0).T.tolist()))
    dump(OUT/'bootstrap.json',dict(contents=len(keys),draws=2000,intervals=intervals))
    import torch,lpips
    torch.set_num_threads(4);cv2.setNumThreads(1);net=lpips.LPIPS(net='alex',version='0.1').eval().requires_grad_(False).cpu();g=check()['quality'];qr=[]
    for j in plan:
        if j['arm']=='clean':continue
        a=np.array(Image.open(j['clean_rgb']).convert('RGB'));b=np.array(Image.open(j['rgb']).convert('RGB'));q=pixel_metrics(a,b)
        ta=torch.from_numpy(a).permute(2,0,1)[None].float()/127.5-1;tb=torch.from_numpy(b).permute(2,0,1)[None].float()/127.5-1
        with torch.inference_mode():q['lpips_alex']=float(net(ta,tb))
        gates=dict(psnr=q['psnr_infinite'] or q['psnr_db']>=g['psnr_db_min'],ssim=q['ssim']>=g['ssim_min'],lpips=q['lpips_alex']<=g['lpips_alex_max'],edge_f1=q['edge_f1_3x3']>=g['rgb_pair_edge_f1_3x3_min'])
        qr.append(dict(job=j['job'],key=j['key'],split=j['split'],view=j['view'],arm=j['arm'],quality=q,gates=gates,pass_all=bool(all(gates.values()))));print('QUALITY',len(qr),'/288',flush=True)
    dump(OUT/'quality.json',dict(records=qr,gates=g))
    print('EVALUATION_COMPLETE',summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','extract','calibrate','evaluate']);a=p.parse_args();globals()[a.action]()
