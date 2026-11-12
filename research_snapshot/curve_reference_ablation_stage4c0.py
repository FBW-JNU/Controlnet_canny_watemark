"""4C-0: progressively remove sender references without changing S/T encoding.

Only Cal labels fit public models. RGB-only feature functions accept no source
curve, source state, intended word, source admission flag or source metadata.
"""
import curve_partial_st_stage4b as b4
from curve_broad_peak_code import basis
from scipy.special import logsumexp
from scipy.stats import rankdata
from pathlib import Path
from PIL import Image
import numpy as np
import cv2, json, argparse, shutil, hashlib, itertools, collections, os

ROOT=b4.ROOT; A4=b4.PARENT; B4=b4.OUT
OUT=ROOT/'outputs/canny_reference_ablation_stage4c0_20261111'
read=b4.read;dump=b4.dump;digest=b4.digest;WORDS=b4.WORDS
DEGREES=[0,1,2,3,4]
OPTIONS=dict(min_arc_px=24.,max_arc_px=300.,min_chord_px=12.,station_spacing_px=2.,
             robust_iterations=25,rank_relative_tolerance=1e-8,max_condition=100.,
             residual_grid=[.75,1.5,3.],ci95_grid=[1.,2.,4.],mask_dilation_px=0,
             extraction='8-neighbor Canny skeleton primitive traces; no gap repair, curve smoothing or source-guided joins. Longest eligible trace, tie lexical endpoints. No message likelihood in trace selection.',
             orientation='Observed endpoints lexicographic; observed normalized arc length; lateral position relative to observed endpoint chord.',
             caveat='A deliberately simple first self-reference receiver, not an optimal blind receiver.')
KEY='4c0-public-mapping-0'
while hashlib.sha256(KEY.encode()).digest()[0]%4:KEY+='x'


def freeze():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():return check()
    for d in ['source','cal_features','eval_features','plots','panels','logs','metrics']:(OUT/d).mkdir(exist_ok=True)
    names=['curve_reference_ablation_stage4c0.py','curve_partial_st_stage4b.py','curve_single_2bit_stage4a.py','curve_broad_peak_code.py','analyze_canny_curve_stage1_32.py']
    for name in names:shutil.copyfile(ROOT/name,OUT/'source'/name)
    dump(OUT/'protocol.json',dict(created_utc=b4.now(),sources={str(ROOT/n):digest(ROOT/n) for n in names},
        parents={str(p/'final_artifacts_sha256.json'):digest(p/'final_artifacts_sha256.json') for p in [A4,B4]},
        options=OPTIONS,codebook=dict(words=WORDS,amplitude=b4.a4.AMPLITUDE,basis='unchanged S cubic / T quartic',key=KEY,
            key_role='Only relabels the four legal states; provides no authentication. Wrong-key presence is mathematically invariant.'),
        data=dict(Cal='Six4A Cal contents x2 views x(clean+4words), plus32 Stage4B Cal clean contents. No old4A Eval.',
                  Eval='Exact existing Stage4B48 marked RGB +128 clean RGB. Twelve marked contents; paired12 clean included in128. Reused held-out diagnostic evaluation, not a newly generated independent validation.'),
        methods=dict(R0='Frozen4B reference-assisted receiver, source geometry+clean coefficient+all4 actual source codewords+original channel/presence.',
          R0_public='Same geometry and clean coefficient, replace actual-input source book by nominal public book with original Cal channel.',
          R1_zero_clean='R0_public without clean subtraction, no recalibration; isolates the direct subtraction removal.',
          R1_public='Same source-assisted geometric synchronization; no clean coefficient, actual-input book or admission flag passed to classifier. Public absolute-coefficient centers fitted on Cal.',
          R2_tube='RGB + fixed source tube Mask + public Cal decoder + Key; source-shaped Mask is auxiliary side information.',
          R2_bbox='RGB + rectangular Mask bounds + public Cal decoder + Key; retains location/extent, removes detailed tube shape.',
          R3='RGB + public rules + Key; no Mask or per-image source fields. Longest geometric trace selected before coefficient decoding.'),
        tuning='Natural baseline degree0/1/2 and geometric fit residual/CI limits selected on Cal by leave-one-content-out full-message correct-2*wrong. Degrees3/4 recorded but rank-deficient => erasure. No Eval tuning.',
        presence='Cal public clean Gaussian vs4 marked centers; Cal clean95% threshold, all receive failures score -1e6. Empirical small-Cal diagnostic, not1%FPR certification.',
        identifiability='Project S/T out of natural polynomial baseline B=[1,x,...,x^r]; report residual energy, Gram eig/condition/rank and six public codeword residual distances. Robust real-valued estimation, not RS.',
        no_regeneration=True))
    return check()


def check():
    p=read(OUT/'protocol.json')
    for f,h in {**p['sources'],**p['parents']}.items():assert digest(f)==h,f
    return p


def information(t,w,degree):
    phi=basis(t);base=np.polynomial.legendre.legvander(2*t-1,degree);root=np.sqrt(w)
    projection=np.linalg.lstsq(base*root[:,None],phi*root[:,None],rcond=None)[0]
    residual=phi-base@projection;g=residual.T@(w[:,None]*residual);raw=phi.T@(w[:,None]*phi)
    e=np.maximum(0,np.linalg.eigvalsh(g));rawmax=max(np.linalg.eigvalsh(raw).max(),1e-12)
    rank=int(np.sum(e>OPTIONS['rank_relative_tolerance']*rawmax));cond=float(e[-1]/e[0]) if rank==2 else None
    pairs=[];book=b4.a4.AMPLITUDE*np.array([b4.a4.SIGNS[word] for word in WORDS])
    for i,j in itertools.combinations(range(4),2):
        d=book[i]-book[j];d2=max(0.,float(d@g@d));pairs.append(dict(pair=[WORDS[i],WORDS[j]],d2=d2,rms=float(np.sqrt(d2/w.sum()))))
    return dict(rank=rank,G=g.tolist(),eigenvalues=e.tolist(),relative_lambda_min=float(e[0]/rawmax),condition=cond,
       residual_energy_ratio=(np.sum(w[:,None]*residual**2,axis=0)/np.maximum(np.sum(w[:,None]*phi**2,axis=0),1e-20)).tolist(),
       min_codeword_d2=min(r['d2'] for r in pairs),min_codeword_rms=min(r['rms'] for r in pairs),pairs=pairs),base,residual


def fit_self(path,degree):
    p=np.array(path,float);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    if len(p)<max(8,degree+4) or arc[-1]<OPTIONS['min_arc_px']:return dict(fitted=False,reason='short_observation')
    if tuple(p[-1])<tuple(p[0]):p=p[::-1];arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    # Select existing pixels spaced in observed arc length; never invent fitting pixels.
    ids=[0]
    for i in range(1,len(p)):
        if arc[i]-arc[ids[-1]]>=OPTIONS['station_spacing_px']:ids.append(i)
    if ids[-1]!=len(p)-1:ids.append(len(p)-1)
    pfull=p;p=p[ids];t=arc[ids]/arc[-1];chord=p[-1]-p[0];chord/=np.linalg.norm(chord);normal=np.array([-chord[1],chord[0]])
    y=(p-p[0])@normal;w=np.ones(len(p));initial,base,_=information(t,w,degree)
    if initial['rank']<2:return dict(fitted=False,reason='ST_in_natural_baseline_span',information=initial,samples=len(p),degree=degree)
    x=np.c_[base,basis(t)];coef=None
    for _ in range(OPTIONS['robust_iterations']):
        coef=np.linalg.lstsq(x*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0];err=y-x@coef
        scale=max(.25,1.4826*float(np.median(abs(err-np.median(err)))));w=np.minimum(1.,1.345*scale/np.maximum(abs(err),1e-9))
    info,_,_=information(t,w,degree)
    sigma=max(.15,1.4826*float(np.median(abs(err-np.median(err)))));cov=np.linalg.pinv(x.T@(w[:,None]*x))*sigma**2*max(1,len(p)/(arc[-1]/4))
    return dict(fitted=True,degree=degree,coefficient=coef[-2:].tolist(),baseline=coef[:-2].tolist(),covariance=cov[-2:,-2:].tolist(),
        ci95_max=float(1.96*np.sqrt(np.linalg.eigvalsh(cov[-2:,-2:]).max())),residual_rms=float(np.sqrt(np.average(err**2,weights=w))),
        observed_xy=p.tolist(),t=t.tolist(),weights=w.tolist(),normal_position=y.tolist(),information=info,
        arc_length_px=float(arc[-1]),chord_px=float(np.linalg.norm(p[-1]-p[0])),samples=len(p))


_MATCHER=None
def extract_rgb(rgb,mask=None):
    """Public-input boundary: no source coordinates, reference coefficient or label."""
    global _MATCHER
    if _MATCHER is None:_MATCHER=b4.a4.local_matcher()
    edge=cv2.Canny(rgb,100,200)
    if mask is not None:edge=np.where(mask,edge,0).astype('uint8')
    graph=_MATCHER.skeleton_graph(edge);paths,stats=_MATCHER.traces(graph);pool=[]
    for p in paths:
        if len(p)<8:continue
        if tuple(p[-1])<tuple(p[0]):p=p[::-1]
        arc=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum());chord=float(np.linalg.norm(p[-1]-p[0]))
        if OPTIONS['min_arc_px']<=arc<=OPTIONS['max_arc_px'] and chord>=OPTIONS['min_chord_px']:
            pool.append((arc,tuple(p[0]),tuple(p[-1]),p))
    pool.sort(key=lambda r:(-r[0],r[1],r[2]))
    if not pool:return dict(selected_path=None,candidate_count=0,stats=stats,fits={str(d):dict(fitted=False,reason='no_eligible_observed_trace') for d in DEGREES})
    p=pool[0][3]
    return dict(selected_path=p.tolist(),candidate_count=len(pool),stats=stats,
        runner_up_arc_px=pool[1][0] if len(pool)>1 else None,selected_arc_px=pool[0][0],
        fits={str(d):fit_self(p,d) for d in DEGREES})


def dataset(split):
    old=read(A4/'selection.json');new=read(B4/'selection.json');cc={c['key']:c for c in old+new}
    if split=='Cal':
        jobs=[dict(j,folder=str(A4)) for j in read(A4/'plan.json') if j['split']=='Cal' and j['arm'] in ['clean']+WORDS]
        jobs += [dict(j,folder=str(B4)) for j in read(B4/'plan.json') if j['split']=='Cal']
    else:jobs=[dict(j,folder=str(B4)) for j in read(B4/'plan.json') if j['split']=='Fresh']
    return jobs,cc


def extract(split):
    check();cv2.setNumThreads(1);jobs,cc=dataset(split);folder=OUT/('cal_features' if split=='Cal' else 'eval_features')
    if split!='Cal':assert (OUT/'score_freeze.json').exists()
    for n,j in enumerate(jobs):
        dest=folder/f'{j["job"]}.json'
        if dest.exists():continue
        c=cc[j['key']];rgb=np.array(Image.open(j['rgb']).convert('RGB'));mask=np.array(Image.open(c['mask']).convert('L'))>0
        bbox=np.zeros_like(mask);yy,xx=np.nonzero(mask)
        if len(xx):bbox[yy.min():yy.max()+1,xx.min():xx.max()+1]=True
        data=read(B4/'loss'/f'{j["job"]}.json');obs=b4.select(data['scenarios']['none']['EdgeDrop'],read(B4/'score_freeze.json')['gate'])
        r0=b4.decode(c,obs,read(B4/'score_freeze.json')['channel_model'],read(B4/'score_freeze.json')['presence'])
        public=dict(R2_tube=extract_rgb(rgb,mask),R2_bbox=extract_rgb(rgb,bbox),R3=extract_rgb(rgb))
        # Mask-only audit is a side-information diagnostic, never an R2 feature.
        if len(c['samples']) and len(xx):
            from scipy.spatial import cKDTree
            sk=b4.a4.local_matcher().skeleton_graph(mask.astype('uint8')*255)['coords'];p=np.array(c['samples'])
            md=float(np.median(cKDTree(sk).query(p)[0])) if len(sk) else None
        else:md=None
        dump(dest,dict(job=j['job'],key=j['key'],content_id=j['content_id'],arm=j['arm'],split=split,rgb=j['rgb'],rgb_sha256=digest(j['rgb']),mask=c['mask'],mask_sha256=digest(c['mask']),
            reference_observation=obs,R0=r0,public=public,mask_information=dict(shape_is_source_tube=True,source_to_mask_skeleton_median_px=md,
            tube_pixels=int(mask.sum()),bbox_pixels=int(bbox.sum()),same_mask_for_all_messages=True)))
        print('EXTRACT',split,n+1,'/',len(jobs),j['job'],flush=True)


def load(split):return [read(p) for p in sorted((OUT/('cal_features' if split=='Cal' else 'eval_features')).glob('*.json'))]


def eligible(f,residual,ci):
    return bool(f.get('fitted') and f['information']['rank']==2 and f['information']['condition']<=OPTIONS['max_condition'] and f['residual_rms']<=residual and f['ci95_max']<=ci)


def cov(values):
    if len(values)<2:return np.eye(2)
    m=np.cov(np.array(values).T,bias=True);e,v=np.linalg.eigh(m);return (v*np.maximum(e,.04))@v.T


def model_fit(rows):
    """Only public feature vectors and training labels enter a public model."""
    centers=[];res=[];counts={}
    for word in WORDS:
        vals=[r['value'] for r in rows if r['arm']==word];counts[word]=len(vals)
        if not vals:return dict(available=False,counts=counts)
        center=np.mean(vals,axis=0);centers.append(center.tolist());res.extend(np.array(vals)-center)
    clean=[r['value'] for r in rows if r['arm']=='clean']
    if len(clean)<2:return dict(available=False,counts=counts,reason='insufficient_Cal_clean')
    return dict(available=True,centers=centers,covariance=cov(res).tolist(),clean_mean=np.mean(clean,axis=0).tolist(),clean_covariance=cov(clean).tolist(),counts=counts,clean_count=len(clean))


def classify(value,fit_cov,model,key=KEY):
    if value is None or not model['available']:return dict(accepted=False,word=None,score=-1e6)
    y=np.array(value);v=np.array(fit_cov);ll=b4.a4.gaussian_log(y-np.array(model['centers']),np.array(model['covariance'])+v)
    i=int(np.argmax(ll));xor=hashlib.sha256(key.encode()).digest()[0]%4
    score=float(logsumexp(ll)-np.log(4)-b4.a4.gaussian_log(y-np.array(model['clean_mean']),np.array(model['clean_covariance'])+v))
    post=np.exp(ll-logsumexp(ll))
    return dict(accepted=True,word=WORDS[i^xor],score=score,posterior_max=float(post.max()),high_confidence=bool(post.max()>=.95),feature=y.tolist())


def rows_for(records,method,degree=None,residual=None,ci=None):
    rows=[]
    for r in records:
        if method=='R1_public':
            f=r['reference_observation'];ok=f.get('accepted',False)
        else:
            f=r['public'][method]['fits'][str(degree)];ok=eligible(f,residual,ci)
        if ok:rows.append(dict(content_id=r['content_id'],arm=r['arm'],value=f['coefficient'],covariance=f['covariance']))
    return rows


def calibrate():
    check();assert not list((OUT/'eval_features').glob('*.json'))
    if (OUT/'score_freeze.json').exists():return
    records=load('Cal');assert len(records)==92;frozen={};trials={}
    for method in ['R1_public','R2_tube','R2_bbox','R3']:
        configs=[(None,None,None)] if method=='R1_public' else list(itertools.product([0,1,2],OPTIONS['residual_grid'],OPTIONS['ci95_grid']))
        results=[]
        for degree,residual,ci in configs:
            rows=rows_for(records,method,degree,residual,ci);correct=wrong=received=0
            for content in sorted({r['content_id'] for r in records if r['arm'] in WORDS}):
                train=[r for r in rows if r['content_id']!=content];model=model_fit(train)
                for row in [r for r in rows if r['content_id']==content and r['arm'] in WORDS]:
                    d=classify(row['value'],row['covariance'],model);received+=d['accepted'];correct+=bool(d['accepted'] and d['word']==row['arm']);wrong+=bool(d['accepted'] and d['word']!=row['arm'])
            results.append(dict(degree=degree,residual=residual,ci95=ci,attempts=48,correct=correct,wrong=wrong,received=received,erasures=48-received,objective=correct-2*wrong))
        selected=max(results,key=lambda r:(r['objective'],-r['wrong'],-(r['degree'] or 0),-(r['residual'] or 0),-(r['ci95'] or 0)))
        rows=rows_for(records,method,selected['degree'],selected['residual'],selected['ci95']);model=model_fit(rows)
        scores=[classify(r['value'],r['covariance'],model)['score'] for r in rows if r['arm']=='clean']
        model['presence_threshold95']=float(np.quantile(scores,.95)) if scores else 1e6
        frozen[method]=dict(selected=selected,model=model);trials[method]=results
        print('CAL',method,selected,flush=True)
    # Public nominal codebook using only the already-frozen global Cal channel.
    channel=read(B4/'score_freeze.json')['channel_model'];centers=np.array([np.array(channel['A'])@(b4.a4.AMPLITUDE*np.array(b4.a4.SIGNS[w]))+channel['intercept'] for w in WORDS])
    frozen['nominal_channel']=dict(available=True,centers=centers.tolist(),covariance=channel['residual_covariance'],clean_mean=read(B4/'score_freeze.json')['presence']['clean_mean'],clean_covariance=read(B4/'score_freeze.json')['presence']['clean_covariance'])
    frozen['created_utc']=b4.now();frozen['protocol_sha256']=digest(OUT/'protocol.json');frozen['cal_features_sha256']={p.name:digest(p) for p in sorted((OUT/'cal_features').glob('*.json'))}
    frozen['no_eval_features_before_freeze']=True;dump(OUT/'cal_trials.json',trials);dump(OUT/'score_freeze.json',frozen)


def theoretical():
    grid=np.linspace(0,1,401);results=[]
    for pattern in ['full','center25','center50','random50']:
        keep=np.ones(len(grid),bool)
        if pattern.startswith('center'):
            amount=int(pattern[6:])/100;keep=abs(grid-.5)>amount/2
        if pattern=='random50':keep=np.random.default_rng(2026111109).random(len(grid))>.5
        for degree in DEGREES:
            metrics,_,_=information(grid[keep],np.ones(keep.sum()),degree)
            results.append(dict(pattern=pattern,degree=degree,observations=int(keep.sum()),metrics=metrics))
    dump(OUT/'metrics/baseline_projection.json',results)
    return results


def eval_all():
    check();frozen=read(OUT/'score_freeze.json');records=load('Eval');assert len(records)==176
    _,cc=dataset('Eval');decisions=[]
    for r in records:
        obs=r['reference_observation'];row=dict(job=r['job'],key=r['key'],content_id=r['content_id'],arm=r['arm'],methods={})
        d=r['R0'];row['methods']['R0']=dict(accepted=d['accepted'],word=d['word'],score=d['presence_score'],presence_accept=d['presence_accept'])
        # These controls use common nominal channel without per-source codeword centers.
        for method,subtract in [('R0_public',True),('R1_zero_clean',False)]:
            val=np.array(obs['coefficient']) if obs.get('accepted') else None
            if val is not None and subtract:val=val-np.array(cc[r['key']]['states']['clean']['measurements']['Auto']['coefficient'])
            row['methods'][method]=classify(val,obs.get('covariance',np.eye(2)),frozen['nominal_channel'])
        for method in ['R1_public','R2_tube','R2_bbox','R3']:
            item=frozen[method];config=item['selected'];model=item['model']
            if method=='R1_public':f=obs;ok=f.get('accepted',False)
            else:f=r['public'][method]['fits'][str(config['degree'])];ok=eligible(f,config['residual'],config['ci95'])
            d=classify(f['coefficient'] if ok else None,f.get('covariance',np.eye(2)),model)
            d['presence_accept']=bool(d['accepted'] and d['score']>model['presence_threshold95'])
            wrong=classify(f['coefficient'] if ok else None,f.get('covariance',np.eye(2)),model,key=KEY+'wrong')
            d['wrong_key_score']=wrong['score'];d['wrong_key_word']=wrong['word'];d['fit_eligible']=ok
            row['methods'][method]=d
        decisions.append(row)
    dump(OUT/'decisions.json',decisions)
    summaries={}
    def auc(pos,neg):
        labels=np.r_[np.ones(len(pos)),np.zeros(len(neg))];values=np.r_[pos,neg]
        return float((rankdata(values)[labels==1].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
    for method in decisions[0]['methods']:
        marked=[r for r in decisions if r['arm'] in WORDS];clean=[r for r in decisions if r['arm']=='clean'];received=[r for r in marked if r['methods'][method]['accepted']]
        correct=sum(r['methods'][method]['word']==r['arm'] for r in received)
        summaries[method]=dict(attempts=48,received=len(received),correct=correct,wrong=len(received)-correct,erasures=48-len(received),full_accuracy=correct/48,
           high_confidence_wrong=sum(r['methods'][method].get('high_confidence',False) and r['methods'][method]['word']!=r['arm'] for r in received),
           clean_attempts=128,clean_received=sum(r['methods'][method]['accepted'] for r in clean),clean_presence_accept=sum(r['methods'][method].get('presence_accept',False) for r in clean) if method not in ['R0_public','R1_zero_clean'] else None,
           marked_presence_accept=sum(r['methods'][method].get('presence_accept',False) for r in marked) if method not in ['R0_public','R1_zero_clean'] else None,presence_AUC=auc([r['methods'][method]['score'] for r in marked],[r['methods'][method]['score'] for r in clean]))
    dump(OUT/'summary.json',summaries)
    print('SUMMARY',json.dumps(summaries),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','cal','calibrate','eval','theory']);a=p.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='cal':extract('Cal')
    elif a.action=='calibrate':calibrate()
    elif a.action=='eval':extract('Eval');eval_all()
    else:check();theoretical()
