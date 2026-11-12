"""Historical 4B SourceRef, zero added loss; no historical output imports.

Retains the original geometric information gate and observation covariance in
four-symbol classification. Used only when the frozen manifest requests 4B.
"""
import numpy as np
from scipy.linalg import eigvalsh
from scipy.spatial import cKDTree
from scipy.special import logsumexp
from .api import THRESHOLDS, GATE
from .geometry import geometry
from .observations import prepare_target,binary_target
from .receiver import basis,receive
WORDS=['m00','m01','m10','m11']
# From curve_partial_st_stage4b.py:171; numerical body unchanged.
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

# From curve_partial_st_stage4b.py:214; numerical body unchanged.
def gate_one(f,gate):
    if not f.get('fitted') or not f.get('association_ok'):return dict(f,accepted=False)
    tests=dict(information=f['info_min']>=gate['info_min'],condition=f['condition']<=gate['condition_max'],
       uncertainty=1.96*f['coefficient_se_max']<=gate['ci95_max'],residual=f['residual_rms']<=.6,offset=abs(f['offset'])<=1.5)
    return dict(f,accepted=bool(all(tests.values())),recovery_checks=tests)

# From curve_partial_st_stage4b.py:220; numerical body unchanged.
def select(raw,gate):
    if 'pool' not in raw:return gate_one(raw,gate)
    pool=[gate_one(f,gate) for f in raw['pool']];good=[f for f in pool if f['accepted']];strict=[f for f in good if f['branch']=='R0']
    return (strict or sorted(good,key=lambda f:f['cost']))[0] if good else dict(accepted=False,fitted=raw['fitted'],reason='partial_geometry_rejected')


def rematch_none(source,rgb):
    # Exact edge_delete(..., scenario=none) target construction: crop the binary
    # graph to the source, then restore tangents measured on original RGB.
    original=prepare_target(rgb)
    target=binary_target(original['edge'],source)
    if not len(target['coords']):return dict(fitted=False,pool=[])
    _,near=cKDTree(original['coords']).query(target['coords'])
    target['tangents']=original['tangents'][near]
    obs=receive(source,target,'S',THRESHOLDS,GATE)['Auto'];pool=[]
    for f in obs.get('decisions',[]):
        q=dict(f);q.update(information(source,q))
        q['association_ok']=all(q['gate_checks'][k] for k in ['tangent','order','residual','offset','distance','margin'])
        q['baseline_partial_gate']=q['accepted'];pool.append(q)
    return dict(fitted=bool(pool),pool=pool,baseline_received=obs['accepted'])
# From curve_single_2bit_stage4a.py:216; numerical body unchanged.
def features(c,arm,head):
    return np.array(c['states'][arm]['measurements'][head]['coefficient'])-np.array(c['states']['clean']['measurements'][head]['coefficient'])

# From curve_single_2bit_stage4a.py:225; numerical body unchanged.
def gaussian_log(x,cov):
    cov=np.array(cov);x=np.asarray(x)
    return -.5*(np.einsum('...i,ij,...j->...',x,np.linalg.inv(cov),x)+np.linalg.slogdet(cov)[1]+2*np.log(2*np.pi))
# From curve_partial_st_stage4b.py:265; numerical body unchanged.
def decode(c,obs,model,presence=None):
    if not obs.get('accepted') or not c['admission']['admissible']:
        return dict(accepted=False,word=None,feature=None,posterior_max=0.,high_confidence=False,codeword_residual=None,presence_score=-1e6,presence_accept=False)
    y=np.array(obs['coefficient'])-np.array(c['states']['clean']['measurements']['Auto']['coefficient'])
    centers=np.array([np.array(model['A'])@features(c,word,'Auto')+model['intercept'] for word in WORDS])
    cov=np.array(model['residual_covariance'])+np.array(obs['covariance'])
    ll=gaussian_log(y-centers,cov);winner=int(np.argmax(ll));prob=np.exp(ll-logsumexp(ll));residual=float(np.min(np.einsum('...i,ij,...j->...',y-centers,np.linalg.inv(cov),y-centers)))
    score=-1e6
    if presence is not None:
        mcov=np.array(model['marked_covariance'])+np.array(obs['covariance'])
        clean_cov=np.array(presence['clean_covariance'])+np.array(obs['covariance'])
        score=float(logsumexp(gaussian_log(y-centers,mcov))-np.log(4)-gaussian_log(y-np.array(presence['clean_mean']),clean_cov))
    return dict(accepted=True,word=WORDS[winner],feature=y.tolist(),posterior_max=float(prob.max()),high_confidence=bool(prob.max()>=.95),
                codeword_residual=residual,wrong_key_residual=residual,wrong_key_word=WORDS[winner^3],presence_score=score,
                presence_accept=bool(presence is not None and score>presence['threshold95']))


def receive_4b(carrier,rgb,frozen):
    if carrier is None:return dict(accepted=False,bits=None,reason='no_carrier')
    obs=select(rematch_none(carrier['samples'],rgb),frozen['gate'])
    out=decode(carrier,obs,frozen['channel_model'])
    return dict(accepted=out['accepted'],bits=out['word'][1:] if out['word'] else None,
                coefficient=obs.get('coefficient'),posterior_max=out['posterior_max'])
