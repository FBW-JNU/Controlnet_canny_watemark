"""Stage 4B: fixed two-bit S/T book, recovery from genuinely retained pixels.

No expected message is passed to correspondence, Huber fitting or gating.
All losses are receiver-side interventions; RGBs are not repainted.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mat_transport_mpl')
import curve_single_2bit_stage4a as a4
from curve_broad_peak_code import fit,receive,basis
from curve_receiver_stage2c import binary_target,geometry
from pathlib import Path
from PIL import Image
from scipy.linalg import eigvalsh
from scipy.spatial import cKDTree
from scipy.special import logsumexp
from scipy.stats import beta as beta_distribution
import numpy as np
import cv2,json,csv,time,shutil,argparse,hashlib,itertools

ROOT=a4.ROOT; PARENT=a4.OUT
OUT=ROOT/'outputs/canny_partial_st_stage4b_20261111'
read=a4.read;dump=a4.dump;digest=a4.digest
WORDS=a4.WORDS
ARMS=['clean']+WORDS
SCENARIOS=[dict(name='none',kind='none',loss=0.,repeat=0)]
for loss in [.25,.50,.75]:
    for repeat in range(3):SCENARIOS.append(dict(name=f'random{int(loss*100)}_r{repeat}',kind='random',loss=loss,repeat=repeat))
for kind in ['center','ends']:
    for loss in [.25,.50]:SCENARIOS.append(dict(name=f'{kind}{int(loss*100)}',kind=kind,loss=loss,repeat=0))
SCENARIOS += [dict(name='left50',kind='left',loss=.5,repeat=0),dict(name='blocks50',kind='blocks',loss=.5,repeat=0)]
REMATCH={'none','random50_r0','center25','center50','ends50','blocks50'}

def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())

def freeze():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():return check()
    for d in ['source','sources','conditions','masks','inputs','rgb','noise','records','measurements','loss','controls','logs','plots','panels']:(OUT/d).mkdir(exist_ok=True)
    scripts=['curve_partial_st_stage4b.py','curve_single_2bit_stage4a.py','curve_broad_peak_code.py','curve_receiver_stage2c.py','curve_receiver_stage2b.py','broad_peak_runtime.py']
    for n in scripts:shutil.copyfile(ROOT/n,OUT/'source'/n)
    p=dict(created_utc=now(),source_sha256={str(ROOT/n):digest(ROOT/n) for n in scripts},
       parent_manifest_sha256=digest(PARENT/'final_artifacts_sha256.json'),
       fixed_encoder=read(PARENT/'protocol.json')['encoder'],generation=read(PARENT/'protocol.json')['generation'],
       quality=read(PARENT/'protocol.json')['quality'],scenarios=SCENARIOS,rematch_scenarios=sorted(REMATCH),
       observation_drop='Freeze accepted Auto correspondences before thinning. Refit only surviving real Canny pixels; baseline-rejected remains erasure. Synchronization-controlled bound, not blind receiving.',
       edge_drop='Delete actual output Canny vertices in source tube<=8px according to source arc masks, rebuild graph retaining original RGB tangents; rerun unchanged R0/A3 proposals. No invented bridge pixels.',
       losses='Common deterministic masks across messages per content/view; random removes exact station fractions, contiguous masks in normalized original arc length. No RGB editing, no image quality change.',
       fit='Existing joint Huber [offset,S,T], same25 iterations. New information gate uses offset-projected 2x2 weighted Gram vs complete fixed station Gram, minimum generalized eigenvalue. Record S/T info separately, design condition and covariance.',
       gate_selection=dict(grid=dict(info_min=[.1,.2,.3],condition_max=[20.,50.,100.],ci95_max=[.75,1.,1.5]),
           unchanged_constraints=dict(residual=.6,offset=1.5,distance=2.5,tangent=.8,order=.95,margin=0.),
           objective='Cal-only maximize correct-2*wrong across equal scenario/track attempts; reject settings accepting >10% translated Cal-anchor negatives. Ties favor fewer wrong then stricter thresholds.',
           note='Coverage/span/anchor counts cannot alone certify partial two-dimensional identifiability. No gate uses expected bit, class likelihood or codeword residual.'),
       classifier='Frozen Stage4A Auto channel A,b,centers and residual covariance, no refitting using independent data. Four-state joint nearest Mahalanobis. Report high confidence at posterior>=.95, not calibrated correctness.',
       presence='Cal-only clean Gaussian vs four-center marked mixture, adding fit covariance. Clean distribution fit on prior Cal plus new32 Cal clean; 95th percentile Cal clean threshold. No new capacity or authentication.',
       wrong_key='XOR11 permutes all four legal labels; unordered four-center residual and mixture likelihood exactly invariant. Wrong-anchor translations are separate correspondence negatives, not clean watermark FPR.',
       sampling=dict(new_cal_clean_contents=32,independent_clean_contents=128,new_marked_contents=12,new_marked_states=48,
          views_per_content=1,total_new_rgb=208,rng=2026111161,source_admission='Same nine-state Stage4A source-only admission. No output-driven content replacement; missing carriers count as erasures.',
          alternating_prompts=True,seeds=dict(Cal=2026114600,Fresh=2026114700),
          exclusions='All currently observable historical paths and metadata; missing historical directories prevent complete historic-overlap proof.'),
       evaluation='Existing Stage4A Eval is diagnostic only. Fresh128 clean and Fresh12 four-state RGB generated after gate/presence freeze. Report unconditional recovery and wrong/erasure counts, clustered intervals; no threshold tuning on fresh.',
       limitations=['Masked residual recovery is real-valued robust regression, not finite-field RS decoding.','Only two bits; four entirely legal words have no intrinsic authentication redundancy.','Known source curve/tube and fixed orientation required. No global synchronization claim.'])
    dump(OUT/'protocol.json',p);return p

def check():
    p=read(OUT/'protocol.json')
    for f,h in p['source_sha256'].items():assert digest(f)==h,f
    assert digest(PARENT/'final_artifacts_sha256.json')==p['parent_manifest_sha256']
    return p

def prepare():
    freeze();cv2.setNumThreads(1)
    if (OUT/'input_freeze.json').exists():return
    if (OUT/'content_sample.json').exists():sample=read(OUT/'content_sample.json')
    else:
        import experiment_s_frozen_validation as sv
        # Reuse exclusion collector without touching its frozen output directory.
        previous=sv.OUT;sv.OUT=OUT
        try:excluded=sv.excluded_contents()
        finally:sv.OUT=previous
        rng=np.random.default_rng(2026111161);rows=[r for r in csv.DictReader(a4.MANIFEST.open()) if int(r['image_id']) not in excluded]
        chosen=[rows[int(i)] for i in rng.choice(len(rows),160,replace=False)]
        sample=[dict(split='Cal' if i<32 else 'Fresh',marked=(32<=i<44),content_id=int(r['image_id']),image_path=r['image_path'],caption=r['prompt'].strip(),view=i%2,
                    seed=(2026114600+i if i<32 else 2026114700+i-32)) for i,r in enumerate(chosen)]
        dump(OUT/'content_sample.json',sample)
    matcher=a4.local_matcher();selected=[];plan=[]
    for num,row in enumerate(sample):
        cid=row['content_id'];dest=OUT/'inputs'/f'{cid:012d}.json'
        if dest.exists():c=read(dest)
        else:
            im=Image.open(row['image_path']).convert('RGB');w,h=[round(v*512/max(im.size)) for v in im.size];x,y=(512-w)//2,(512-h)//2
            canvas=Image.new('RGB',(512,512),(127,127,127));canvas.paste(im.resize((w,h),Image.Resampling.LANCZOS),(x,y))
            sp=OUT/'sources'/f'{cid:012d}.png';canvas.save(sp);edge=cv2.Canny(np.array(canvas),100,200)
            roi=[x+8,y+8,x+w-8,y+h-8];valid=np.zeros_like(edge);valid[roi[1]:roi[3],roi[0]:roi[2]]=edge[roi[1]:roi[3],roi[0]:roi[2]]
            curves=sorted(matcher.source_curves(valid,a4.THRESHOLDS)[0],key=lambda q:(-q['arc_length_px'],q['curve_index']));tried=[];chosen=None
            for curve in curves:
                mask=a4.mask_for(edge,curve['samples'],roi);images,observations,admission=a4.input_check(edge,curve['samples'],mask)
                tried.append(dict(curve_index=curve['curve_index'],admissible=admission['admissible']))
                chosen=(curve,mask,images,observations,admission)
                if admission['admissible']:break
            if chosen is None:
                # Retain a content without a carrier, still generate clean PNG.
                curve=dict(samples=[],curve_index=-1,arc_length_px=0);mask=np.zeros_like(edge,dtype=bool);images={'clean':edge};observations={'clean':{}};admission=dict(admissible=False)
            else:curve,mask,images,observations,admission=chosen
            key=f'{cid:012d}_c{curve["curve_index"]:04d}';mp=OUT/'masks'/f'{key}.png';Image.fromarray(mask.astype('uint8')*255).save(mp)
            states={}
            for arm,image in images.items():
                path=OUT/'conditions'/f'{key}_{arm}.png';Image.fromarray(image).save(path)
                states[arm]=dict(hint=str(path),hint_sha256=digest(path),measurements=observations[arm])
            c=dict(**row,key=key,samples=curve['samples'],curve_index=curve['curve_index'],arc_length_px=curve['arc_length_px'],roi=roi,
                   source_rgb=str(sp),source_sha256=digest(row['image_path']),mask=str(mp),states=states,admission=admission,candidates_tried=tried,source_candidate_count=len(curves))
            dump(dest,c)
        selected.append(c)
        clean=OUT/'rgb'/f'{c["key"]}_v{c["view"]}_clean.png'
        for arm in (ARMS if row['marked'] else ['clean']):
            # If source has no valid carrier, four identical conditions retained and counted as failures.
            state=c['states'].get(arm,c['states']['clean']);job=f'{c["key"]}_v{c["view"]}_{arm}'
            plan.append(dict(job=job,key=c['key'],content_id=cid,split=c['split'],view=c['view'],arm=arm,prompt=a4.PREFIX[c['view']]+c['caption'],seed=c['seed'],
                hint=state['hint'],hint_sha256=state['hint_sha256'],rgb=str(OUT/'rgb'/f'{job}.png'),clean_rgb=str(clean),noise=str(OUT/'noise'/f'{c["key"]}_v{c["view"]}.npy')))
        print('PREPARE',num+1,'/160',cid,c['admission']['admissible'],'tries',len(c['candidates_tried']),flush=True)
    dump(OUT/'selection.json',selected);dump(OUT/'plan.json',plan)
    protected=[OUT/n for n in ['selection.json','plan.json','content_sample.json','exclusions.json']]+list((OUT/'conditions').glob('*.png'))+list((OUT/'inputs').glob('*.json'))
    dump(OUT/'input_freeze.json',dict(created_utc=now(),protected_sha256={str(p):digest(p) for p in protected},rgb_count=len(plan)))

def worker():
    import torch
    from cldm.ddim_hacked import DDIMSampler
    from controlnet_robin.pipeline import ControlNetROBINPipeline
    f=check();gen=f['generation'];assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and torch.cuda.is_available()
    assert digest(ROOT/gen['checkpoint'])==gen['checkpoint_sha256']
    for p,h in read(a4.S1/'protocol.json')['frozen_model_code'].items():assert digest(p)==h,p
    for p,h in read(OUT/'input_freeze.json')['protected_sha256'].items():assert digest(p)==h,p
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    class DeviceSampler(DDIMSampler):
        def register_buffer(self,name,attr):setattr(self,name,attr.to(self.model.device) if isinstance(attr,torch.Tensor) else attr)
    pipe=ControlNetROBINPipeline(str(ROOT/gen['model_config']),str(ROOT/gen['checkpoint']),'cuda');pipe.model.requires_grad_(False).eval();pipe.model.cond_stage_model.device='cuda';pipe.ddim=DeviceSampler(pipe.model)
    pipe.make_schedule(gen['steps'],eta=gen['eta']);pipe.model.control_scales=[gen['control_scale']]*13
    plan=read(OUT/'plan.json')
    for ix,j in enumerate(plan):
        if j['split']=='Fresh':
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
        Image.fromarray(im).save(j['rgb']);dump(record,dict(**j,rgb_sha256=digest(j['rgb']),noise_sha256=digest(j['noise']),checkpoint_sha256=gen['checkpoint_sha256'],
           protocol_sha256=digest(OUT/'protocol.json'),score_freeze_sha256=digest(OUT/'score_freeze.json') if j['split']=='Fresh' else None,elapsed_seconds=time.monotonic()-start))
        print('GEN',ix+1,'/',len(plan),j['split'],j['job'],flush=True)
    dump(OUT/'generation_status.json',dict(status='complete',unique_rgb=len(plan)))

def mask_keep(source,job,scenario,indices):
    _,u,_,_=geometry(source);ii=np.array(indices,int);v=u[ii];kind=scenario['kind'];loss=scenario['loss']
    if kind=='none':return np.ones(len(ii),bool)
    if kind=='center':return abs(v-.5)>=loss/2
    if kind=='ends':return (v>=loss/2)&(v<=1-loss/2)
    if kind=='left':return v>=loss
    if kind=='blocks':return (np.floor(v*8).astype(int)%2)==0
    stations=np.unique(np.r_[np.arange(0,len(u),2),len(u)-1]).astype(int)
    # Job arm deliberately omitted: all five messages get the same intervention.
    token=f'{job["key"]}_{job["view"]}_{scenario["repeat"]}'
    rng=np.random.default_rng(int(hashlib.sha256(token.encode()).hexdigest()[:16],16));gone=rng.permutation(len(stations))[:round(loss*len(stations))]
    allowed=np.ones(len(stations),bool);allowed[gone]=False;nearest=np.argmin(abs(ii[:,None]-stations[None,:]),axis=1)
    return allowed[nearest]

def information(source,f):
    p,u,_,_=geometry(source);ii=np.array(f['indices']);w=np.array(f['weights']);b=basis(u);v=b[ii]
    stations=np.unique(np.r_[np.arange(0,len(p),2),len(p)-1]).astype(int)
    def centered_gram(v,w):
        mean=np.average(v,axis=0,weights=w);q=v-mean
        return q.T@(w[:,None]*q)
    reference=centered_gram(b[stations],np.ones(len(stations)));gram=centered_gram(v,w)
    eig=eigvalsh(gram,reference);x=np.c_[np.ones(len(ii)),v];condition=float(np.linalg.cond(x.T@(w[:,None]*x)))
    coverage=np.sum(w[:,None]*v*v,axis=0)/np.maximum(np.sum(b[stations]**2,axis=0),1e-9)
    return dict(info_eigenvalues=eig.tolist(),info_min=float(eig.min()),condition=condition,ST_information=coverage.tolist(),actual_samples=len(ii))

def partial(source,old,job,scenario):
    if not old.get('accepted'):return dict(accepted=False,fitted=False,reason='baseline_correspondence_rejected')
    ii=np.array(old['indices']);keep=mask_keep(source,job,scenario,ii)
    f=fit(source,dict(indices=ii[keep],xy=np.array(old['observed_xy'])[keep],weights=np.array(old['weights'])[keep]),'S')
    if f is None:return dict(accepted=False,fitted=False,reason='insufficient_or_singular_observation',surviving_count=int(keep.sum()))
    f.update(information(source,f));f['association_ok']=True;f['baseline_received']=True
    f['baseline_partial_gate']=bool(f['coverage']>=.65 and f['span']>=.8 and f['r_phi']>=.65 and 1.96*f['coefficient_se_max']<=1.5 and f['residual_rms']<=.6)
    return f

def edge_delete(source,rgb,job,scenario):
    original=a4.prepare_target(rgb);edge=original['edge'].copy();p,_,_,_=geometry(source)
    yy,xx=np.nonzero(edge);coords=np.c_[xx,yy];dist,idx=cKDTree(p).query(coords);keep=mask_keep(source,job,scenario,idx)
    gone=(dist<=8)&(~keep);edge[yy[gone],xx[gone]]=0
    target=binary_target(edge,source)
    if len(target['coords']):
        # Preserve the tangents of observed RGB edges, never use fitted curves.
        _,near=cKDTree(original['coords']).query(target['coords']);target['tangents']=original['tangents'][near]
    return target,dict(deleted_edge_pixels=int(gone.sum()),retained_edge_pixels=int(np.count_nonzero(edge)),edge=edge)

def rematch(source,rgb,job,scenario):
    target,details=edge_delete(source,rgb,job,scenario)
    if not len(target['coords']):return dict(fitted=False,accepted=False,reason='empty_edge',deletion={k:v for k,v in details.items() if k!='edge'})
    obs=receive(source,target,'S',a4.THRESHOLDS,a4.GATE)['Auto'];pool=[]
    for f in obs.get('decisions',[]):
        q=dict(f);q.update(information(source,q))
        # The same geometric constraints; only fixed sample/endpoint coverage
        # requirements are replaced by joint identifiability gates.
        keys=['tangent','order','residual','offset','distance','margin']
        q['association_ok']=all(q['gate_checks'][k] for k in keys)
        q['baseline_partial_gate']=q['accepted'];pool.append(q)
    return dict(fitted=bool(pool),pool=pool,baseline_received=obs['accepted'],deletion={k:v for k,v in details.items() if k!='edge'})

def gate_one(f,gate):
    if not f.get('fitted') or not f.get('association_ok'):return dict(f,accepted=False)
    tests=dict(information=f['info_min']>=gate['info_min'],condition=f['condition']<=gate['condition_max'],
       uncertainty=1.96*f['coefficient_se_max']<=gate['ci95_max'],residual=f['residual_rms']<=.6,offset=abs(f['offset'])<=1.5)
    return dict(f,accepted=bool(all(tests.values())),recovery_checks=tests)

def select(raw,gate):
    if 'pool' not in raw:return gate_one(raw,gate)
    pool=[gate_one(f,gate) for f in raw['pool']];good=[f for f in pool if f['accepted']];strict=[f for f in good if f['branch']=='R0']
    return (strict or sorted(good,key=lambda f:f['cost']))[0] if good else dict(accepted=False,fitted=raw['fitted'],reason='partial_geometry_rejected')

def extract(split):
    check();cv2.setNumThreads(1)
    if split in ['Cal','Fresh']:
        plan=[j for j in read(OUT/'plan.json') if j['split']==split];cc={c['key']:c for c in read(OUT/'selection.json')};folder=OUT
    else:
        plan=[j for j in read(PARENT/'plan.json') if j['arm'] in ARMS and j['split']==('Cal' if split=='OldCal' else 'Eval')]
        cc={c['key']:c for c in read(PARENT/'selection.json')};folder=PARENT
    for ix,j in enumerate(plan):
        dest=OUT/'loss'/f'{j["job"]}.json'
        if dest.exists():continue
        record=folder/'records'/f'{j["job"]}.json'
        while not record.exists():time.sleep(2)
        assert digest(j['rgb'])==read(record)['rgb_sha256'];c=cc[j['key']]
        rgb=np.array(Image.open(j['rgb']).convert('RGB'));data={}
        if len(c['samples']):
            if folder==PARENT:obs=read(PARENT/'measurements'/f'{j["job"]}.json')['observations']['Auto']
            else:obs=receive(c['samples'],a4.prepare_target(rgb),'S',a4.THRESHOLDS,a4.GATE)['Auto']
            for s in SCENARIOS:
                row=dict(ObsDrop=partial(c['samples'],obs,j,s))
                if s['name'] in REMATCH:row['EdgeDrop']=rematch(c['samples'],rgb,j,s)
                data[s['name']]=row
        else:obs=dict(accepted=False,reason='no_source_carrier')
        dump(dest,dict(job=j['job'],key=j['key'],split=split,arm=j['arm'],view=j['view'],content_id=j['content_id'],rgb_sha256=digest(j['rgb']),original=obs,scenarios=data))
        print('EXTRACT',split,ix+1,'/',len(plan),j['job'],flush=True)
    if split in ['Cal','Fresh']:
        # Bit-independent wrong-anchor diagnostic, two predeclared translations.
        for j in plan:
            if j['arm']!='clean':continue
            dest=OUT/'controls'/f'{j["job"]}.json'
            if dest.exists():continue
            c=cc[j['key']];rgb=np.array(Image.open(j['rgb']).convert('RGB'));rows=[]
            for shift in [[32,0],[0,32]]:
                if not len(c['samples']):continue
                shifted=np.array(c['samples'])+shift
                # Same image/tube algorithm, wrong carrier. Not watermark false positives.
                rows.append(dict(shift=shift,raw=rematch(shifted,rgb,j,SCENARIOS[0])))
            dump(dest,dict(job=j['job'],split=split,controls=rows))

def channel_model():return read(PARENT/'score_freeze.json')['models']['Auto']

def decode(c,obs,model,presence=None):
    if not obs.get('accepted') or not c['admission']['admissible']:
        return dict(accepted=False,word=None,feature=None,posterior_max=0.,high_confidence=False,codeword_residual=None,presence_score=-1e6,presence_accept=False)
    y=np.array(obs['coefficient'])-np.array(c['states']['clean']['measurements']['Auto']['coefficient'])
    centers=np.array([np.array(model['A'])@a4.features(c,word,'Auto')+model['intercept'] for word in WORDS])
    cov=np.array(model['residual_covariance'])+np.array(obs['covariance'])
    ll=a4.gaussian_log(y-centers,cov);winner=int(np.argmax(ll));prob=np.exp(ll-logsumexp(ll));residual=float(np.min(np.einsum('...i,ij,...j->...',y-centers,np.linalg.inv(cov),y-centers)))
    score=-1e6
    if presence is not None:
        mcov=np.array(model['marked_covariance'])+np.array(obs['covariance'])
        clean_cov=np.array(presence['clean_covariance'])+np.array(obs['covariance'])
        score=float(logsumexp(a4.gaussian_log(y-centers,mcov))-np.log(4)-a4.gaussian_log(y-np.array(presence['clean_mean']),clean_cov))
    return dict(accepted=True,word=WORDS[winner],feature=y.tolist(),posterior_max=float(prob.max()),high_confidence=bool(prob.max()>=.95),
                codeword_residual=residual,wrong_key_residual=residual,wrong_key_word=WORDS[winner^3],presence_score=score,
                presence_accept=bool(presence is not None and score>presence['threshold95']))

def all_data(splits):
    result=[]
    for p in sorted((OUT/'loss').glob('*.json')):
        d=read(p)
        if d['split'] in splits:result.append(d)
    return result

def calibrate():
    check()
    if (OUT/'score_freeze.json').exists():return
    assert not any(read(p)['split']=='Fresh' for p in (OUT/'records').glob('*.json'))
    cc={c['key']:c for folder in [PARENT,OUT] for c in read(folder/'selection.json')};model=channel_model();data=all_data(['OldCal','Cal'])
    assert len([d for d in data if d['split']=='OldCal'])==60 and len([d for d in data if d['split']=='Cal'])==32
    negatives=[r['raw'] for p in (OUT/'controls').glob('*.json') for d in [read(p)] if d['split']=='Cal' for r in d['controls']]
    trials=[]
    for info,condition,ci in itertools.product([.1,.2,.3],[20.,50.,100.],[.75,1.,1.5]):
        gate=dict(info_min=info,condition_max=condition,ci95_max=ci);correct=wrong=attempts=0
        for d in data:
            if d['split']!='OldCal' or d['arm']=='clean':continue
            for s,tracks in d['scenarios'].items():
                for track,raw in tracks.items():
                    out=decode(cc[d['key']],select(raw,gate),model);attempts+=1
                    correct+=bool(out['accepted'] and out['word']==d['arm']);wrong+=bool(out['accepted'] and out['word']!=d['arm'])
        false=sum(select(r,gate)['accepted'] for r in negatives);eligible=false<=.1*len(negatives)
        trials.append(dict(gate=gate,correct=correct,wrong=wrong,attempts=attempts,negative_accepts=false,negative_attempts=len(negatives),eligible=eligible,objective=correct-2*wrong))
    eligible=[r for r in trials if r['eligible']]
    if not eligible:
        # Do not silently relax the negative-control requirement.
        dump(OUT/'cal_gate_grid.json',trials);raise RuntimeError('No Cal gate satisfies predeclared false-anchor constraint')
    selected=max(eligible,key=lambda r:(r['objective'],-r['wrong'],r['gate']['info_min'],-r['gate']['condition_max'],-r['gate']['ci95_max']))
    gate=selected['gate'];clean=[]
    for d in data:
        if d['arm']!='clean':continue
        raw=d['scenarios'].get('none',{}).get('ObsDrop',dict(fitted=False));obs=select(raw,gate);v=decode(cc[d['key']],obs,model)
        if v['accepted']:clean.append(v['feature'])
    assert len(clean)>=16
    presence=dict(clean_mean=np.mean(clean,axis=0).tolist(),clean_covariance=a4.covariance(clean).tolist(),threshold95=0.,fit_received_clean=len(clean),fit_attempt_clean=44)
    scores=[]
    # Threshold covers all declared loss scenarios/tracks on Cal clean only.
    # These repeated views are correlated; no binomial 5% guarantee is claimed.
    for d in data:
        if d['arm']!='clean':continue
        for tracks in d['scenarios'].values():
            for raw in tracks.values():
                v=decode(cc[d['key']],select(raw,gate),model,presence)
                if v['accepted']:scores.append(v['presence_score'])
    presence['threshold95']=float(np.percentile(scores,95));presence['cal_received_scores']=len(scores)
    dump(OUT/'cal_gate_grid.json',trials)
    dump(OUT/'score_freeze.json',dict(created_utc=now(),protocol_sha256=digest(OUT/'protocol.json'),gate=gate,selected=selected,
        presence=presence,channel_model=model,parent_score_sha256=digest(PARENT/'score_freeze.json'),
        cal_data_sha256={str(OUT/'loss'/f'{d["job"]}.json'):digest(OUT/'loss'/f'{d["job"]}.json') for d in data},
        cal_controls_sha256={str(p):digest(p) for p in (OUT/'controls').glob('*.json') if read(p)['split']=='Cal'},no_fresh_rgb_before_freeze=True))
    print('SCORE_FROZEN',gate,presence,flush=True)

def summarize(rows):
    marked=[r for r in rows if r['arm'] in WORDS];clean=[r for r in rows if r['arm']=='clean'];received=[r for r in marked if r['accepted']]
    correct=sum(r['word']==r['arm'] for r in received);wrong=len(received)-correct
    hc=[r for r in received if r['high_confidence']]
    def rate(x,n):return x/n if n else None
    return dict(attempts=len(marked),received=len(received),correct=correct,wrong=wrong,erasures=len(marked)-len(received),
       full_accuracy=rate(correct,len(marked)),received_accuracy=rate(correct,len(received)),receive_rate=rate(len(received),len(marked)),
       high_confidence_received=len(hc),high_confidence_wrong=sum(r['word']!=r['arm'] for r in hc),
       clean_attempts=len(clean),clean_received=sum(r['accepted'] for r in clean),clean_presence_accepted=sum(r['presence_accept'] for r in clean),
       marked_presence_accepted=sum(r['presence_accept'] for r in marked),
       presence_auc=a4.auc([int(r['arm'] in WORDS) for r in rows],[r['presence_score'] for r in rows]) if clean and marked else None)

def evaluate():
    check();freeze=read(OUT/'score_freeze.json');gate=freeze['gate'];model=freeze['channel_model'];presence=freeze['presence'];data=all_data(['OldCal','Cal','OldEval','Fresh'])
    cc={c['key']:c for folder in [PARENT,OUT] for c in read(folder/'selection.json')};rows=[]
    for d in data:
        for scenario in SCENARIOS:
            name=scenario['name']
            for track in ['ObsDrop']+(['EdgeDrop'] if name in REMATCH else []):
                raw=d['scenarios'].get(name,{}).get(track,dict(fitted=False));obs=select(raw,gate);out=decode(cc[d['key']],obs,model,presence)
                baseline=bool(raw.get('baseline_partial_gate',raw.get('baseline_received',False)))
                rows.append(dict(job=d['job'],key=d['key'],split=d['split'],view=d['view'],arm=d['arm'],scenario=name,track=track,
                    baseline_accept=baseline,original_accept=d['original']['accepted'],**out,
                    geometry={k:obs[k] for k in ['coverage','span','ST_information','info_min','condition','coefficient_se_max','residual_rms','indices','observed_xy','coefficient','covariance'] if k in obs}))
    dump(OUT/'decoded.json',rows);summary={}
    for split in ['OldCal','Cal','OldEval','Fresh']:
        summary[split]={}
        for scenario in SCENARIOS:
            name=scenario['name'];summary[split][name]={}
            for track in ['ObsDrop']+(['EdgeDrop'] if name in REMATCH else []):
                subset=[r for r in rows if r['split']==split and r['scenario']==name and r['track']==track]
                summary[split][name][track]=summarize(subset)
    dump(OUT/'summary.json',summary)
    controls=[]
    for path in (OUT/'controls').glob('*.json'):
        d=read(path)
        for control in d['controls']:
            obs=select(control['raw'],gate);controls.append(dict(job=d['job'],split=d['split'],shift=control['shift'],accepted=obs['accepted'],
                baseline_received=control['raw'].get('baseline_received',False),coefficient=obs.get('coefficient')))
    dump(OUT/'wrong_anchor.json',dict(records=controls,summary={s:dict(attempts=sum(r['split']==s for r in controls),accepted=sum(r['split']==s and r['accepted'] for r in controls),baseline_accepted=sum(r['split']==s and r['baseline_received'] for r in controls)) for s in ['Cal','Fresh']}))
    invariance=[r for r in rows if r['accepted']]
    dump(OUT/'wrong_key.json',dict(key_transform='XOR11 label permutation, all four states legal',attempts=len(invariance),
       maximum_residual_difference=max([abs(r['codeword_residual']-r['wrong_key_residual']) for r in invariance] or [0.]),
       conclusion='Exactly invariant codeword residual; this key does not authenticate. Wrong-key message labels change, but acceptance cannot reject the key.'))
    # Paired full-vs-partial drift, not truth error: full estimate itself can be wrong.
    full={(r['job'],r['track']):r for r in rows if r['scenario']=='none'};drift=[]
    for r in rows:
        base=full[(r['job'],r['track'])]
        if r['accepted'] and base['accepted']:
            drift.append(dict(job=r['job'],split=r['split'],arm=r['arm'],track=r['track'],scenario=r['scenario'],
                coefficient_delta=(np.array(r['feature'])-base['feature']).tolist(),same_word=r['word']==base['word']))
    dump(OUT/'coefficient_drift.json',drift)
    fresh_clean=[r for r in rows if r['split']=='Fresh' and r['arm']=='clean' and r['scenario']=='none' and r['track']=='ObsDrop']
    k=sum(r['presence_accept'] for r in fresh_clean);n=len(fresh_clean)
    dump(OUT/'clean_independent.json',dict(contents=n,false_accepts=k,false_accept_rate=k/n,
       exact_binomial95=[float(beta_distribution.ppf(.025,k,n-k+1)) if k else 0.,float(beta_distribution.ppf(.975,k+1,n-k)) if k<n else 1.],
       limitation='One image/content, alternating two prompts, new seeds. No 1% FPR claim; sample size and two-bit authentication limitations remain.'))
    # Grouped bootstrap on fresh marked contents, full attempt denominator.
    rng=np.random.default_rng(2026111162);intervals={}
    for scenario in SCENARIOS:
        name=scenario['name'];intervals[name]={}
        for track in ['ObsDrop']+(['EdgeDrop'] if name in REMATCH else []):
            subset=[r for r in rows if r['split']=='Fresh' and r['arm'] in WORDS and r['scenario']==name and r['track']==track];keys=sorted({r['key'] for r in subset})
            blocks=[[r for r in subset if r['key']==k] for k in keys];values=[]
            for _ in range(1000):
                draw=[r for i in rng.integers(0,len(blocks),len(blocks)) for r in blocks[i]];values.append(sum(r['accepted'] and r['word']==r['arm'] for r in draw)/len(draw))
            intervals[name][track]=np.percentile(values,[2.5,97.5]).tolist()
    dump(OUT/'bootstrap.json',dict(unit='12 independent marked contents; four correlated words per content',draws=1000,full_accuracy95=intervals))
    print('EVALUATE_COMPLETE',summary['Fresh']['none'],flush=True)

def quality():
    check();import broad_peak_runtime as runtime
    previous=runtime.OUT;runtime.OUT=OUT
    try:runtime.quality(read(OUT/'plan.json'),OUT/'quality.json')
    finally:runtime.OUT=previous

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','extract','calibrate','evaluate','quality']);p.add_argument('--split',default='OldCal');args=p.parse_args()
    if args.action=='extract':extract(args.split)
    else:globals()[args.action]()
