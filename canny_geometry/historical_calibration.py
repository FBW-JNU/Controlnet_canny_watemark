"""Reconstruct the historical 4A Auto Cal model from actual binary and RGB fits."""
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from .benchmark import verify,read,dump,sha,location,image
from .api import observe,WORDS
from .research import covariance
from .historical_receiver import gaussian_log


def check(manifest_path,decoded_root,out):
    m=verify(manifest_path);decoded_root=Path(decoded_root)
    if read(decoded_root/'summary.json')['manifest_sha256']!=sha(manifest_path):raise ValueError('Decoded manifest differs.')
    observations={r['id']:r['decoded'] for r in read(decoded_root/'decoded.json')}
    jobs={j['id']:j for j in m['generation_jobs']};rows=[];clean_decisions=[]
    allrows=[r for r in m['evaluation_rows'] if r['group']=='st2/Cal6']
    cache={};input_fits=0
    for row in allrows:
        ref=m['references'][row['reference']]['reference'];job=jobs[row['generation_id']];arm=row['id'].rsplit('_',1)[-1];d=observations[row['id']]
        if arm=='clean':clean_decisions.append((row,ref,d))
        if not d['accepted']:continue
        token=(row['reference'],job['hint'])
        if token not in cache:
            observed=observe(ref['samples'],image(location(manifest_path,job['hint'])))['Auto']
            if not observed['accepted']:raise ValueError('Historical Cal input is no longer observable.')
            cache[token]=np.array(observed['coefficient']);input_fits+=1
        origin=np.array(ref['codebooks']['Auto']['clean'])
        rows.append(dict(row=row,arm=arm,x=cache[token]-origin,y=np.array(d['coefficient'])-origin))
    x=np.array([np.r_[1,r['x']] for r in rows]);y=np.array([r['y'] for r in rows]);beta=np.linalg.lstsq(x,y,rcond=None)[0];res=y-x@beta
    clean=np.array([r['y']-beta[0] for r in rows if r['arm']=='clean'])
    marked=np.array([res[i] for i,r in enumerate(rows) if r['arm'].startswith('m')])
    model=dict(A=beta[1:].T.tolist(),intercept=beta[0].tolist(),cal_received=len(rows),cal_attempts=len(allrows),residual_covariance=covariance(res).tolist(),marked_covariance=covariance(marked).tolist(),clean_residual_mean=clean.mean(0).tolist(),clean_covariance=covariance(clean).tolist(),condition=float(np.linalg.cond(beta[1:].T)))
    scores=[]
    for row,ref,d in clean_decisions:
        if not d['accepted']:scores.append(-1e6);continue
        book=ref['codebooks']['Auto'];feature=np.array(d['coefficient'])-book['clean'];centers=np.array([beta[1:].T@(np.array(book[w])-book['clean'])+beta[0] for w in WORDS])
        marked_score=logsumexp(gaussian_log(feature-centers,model['marked_covariance']))-np.log(4)
        clean_score=gaussian_log(feature-beta[0]-model['clean_residual_mean'],model['clean_covariance'])
        scores.append(float(marked_score-clean_score))
    model['presence_threshold95']=float(np.percentile(scores,95));expected=m['models']['4A'];deltas={}
    for k,v in model.items():
        delta=float(np.max(np.abs(np.asarray(v)-np.asarray(expected[k]))))
        if delta>1e-10:deltas[k]=delta
    report=dict(status='passed' if not deltas else 'failed',Cal_RGB_attempts=len(allrows),Cal_received=len(rows),unique_binary_input_fits=input_fits,absolute_tolerance=1e-10,mismatches=deltas,reconstructed_model=model,scope='4A Auto channel and presence parameters from Cal-only observations; no Eval fitting, original content discovery or 4B training reconstruction.')
    dump(out,report)
    if deltas:raise RuntimeError('Historical Cal model differs.')
    return report
