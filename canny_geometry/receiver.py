"""Frozen 1-bit/4A joint S/T receiver; no presence gate in public API."""
import numpy as np
from scipy.spatial import cKDTree
from . import matcher as M
from .geometry import geometry, canonical
from .observations import ordered_observations, path_observations
CONFIG = dict(tube=6, angle=30, skip=4, bridge=6, r_phi=.5)
MIX = .25

# From curve_broad_peak_code.py:14; numerical body unchanged.
def basis(u):
    return np.c_[6*np.sqrt(3)*u*(1-u)*(2*u-1),4*u*(1-u)*(5*(2*u-1)**2-1)]

_dense = np.linspace(0, 1, 100001)
NORMALIZATION = float(np.max(np.abs(basis(_dense) @ np.array([MIX, 1.]))))

# From curve_broad_peak_code.py:20; numerical body unchanged.
def profile(u,mode):
    b=basis(np.asarray(u))
    return b[:,0] if mode=='S' else b@np.array([MIX,1.])/NORMALIZATION

# From curve_broad_peak_code.py:29; numerical body unchanged.
def fit(source,obs,mode):
    if obs is None:return None
    p,u,n,_=geometry(source,'S');ii=np.array(obs['indices'],int);xy=np.array(obs['xy'],float)
    if len(ii)<7:return None
    b=basis(u);x=np.c_[np.ones(len(ii)),b[ii]];original=np.array(obs.get('weights',np.ones(len(ii))),float);w=original.copy()
    y=np.sum((xy-p[ii])*n[ii],axis=1)
    if np.linalg.cond(x.T@(w[:,None]*x))>500:return None
    for _ in range(25):
        beta=np.linalg.lstsq(x*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0];res=y-x@beta
        sigma=max(.25,1.4826*np.median(abs(res-np.median(res))));w=original*np.minimum(1,1.345*sigma/np.maximum(abs(res),1e-8))
    span=float(np.ptp(u[ii]));length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum());eff=max(2,min(len(ii),span*length/4))
    sigma=max(.15,1.4826*np.median(abs(res-np.median(res))));cov=np.linalg.inv(x.T@(w[:,None]*x))*sigma*sigma*len(ii)/eff
    stations=np.unique(np.r_[np.arange(0,len(p),2),len(p)-1]).astype(int);phi=profile(u,mode)
    info=np.sum(w*phi[ii]**2)/max(1e-9,np.sum(phi[stations]**2));rms=float(np.sqrt(np.average(res*res,weights=w)))
    return dict(fitted=True,coefficient=beta[1:].tolist(),offset=float(beta[0]),covariance=cov[1:,1:].tolist(),
        coefficient_se_max=float(np.sqrt(np.linalg.eigvalsh(cov[1:,1:]).max())),residual_rms=rms,r_phi=min(1,float(info)),coverage=len(ii)/len(stations),span=span,
        observed_xy=xy.tolist(),indices=ii.tolist(),weights=w.tolist(),normal_positions=y.tolist())

# From curve_broad_peak_code.py:47; numerical body unchanged.
def measure(source,target,obs,branch,mode):
    f=fit(source,obs,mode)
    if f is None:return None
    p,u,n,_=geometry(source,'S');ii=np.array(f['indices']);xy=np.array(f['observed_xy']);t=M.tangent(p)
    _,near=cKDTree(target['coords']).query(xy);angle=np.degrees(np.arccos(np.clip(np.abs(np.sum(t[ii]*target['tangents'][near],axis=1)),0,1)))
    distance=np.linalg.norm(xy-p[ii],axis=1);anchors=np.rint(np.array([.15,.3,.5,.7,.85])*(len(p)-1)).astype(int)
    ad,ai=cKDTree(xy).query(p[anchors]);projection=cKDTree(p).query(xy)[1]
    f.update(branch=branch,distance_median=float(np.median(distance)),angle_mean=float(angle.mean()),tangent_good=float(np.mean(angle<=30)),
        anchor_fraction=float(np.mean((ad<=3)&(angle[ai]<=30))),order_consistency=float(np.mean(np.diff(projection)>=0)))
    f['cost']=.35*f['distance_median']+f['residual_rms']+2*(1-min(1,f['coverage']))+.8*f['angle_mean']/30+.5*abs(f['offset'])
    return f

# From curve_broad_peak_code.py:59; numerical body unchanged.
def receive(source,target,mode,thresholds,gate):
    # The same source/tube, path search and pool are used for both messages.
    strict=M.curve_match(canonical(source),target,thresholds,True);pool=[]
    if strict['matched']:
        ii,xy,w=path_observations(source,strict['output_path'])
        r=measure(source,target,dict(indices=ii,xy=xy,weights=w),'R0',mode)
        if r is not None:pool.append(r)
    obs=ordered_observations(source,target,CONFIG);primary=measure(source,target,obs,'A3',mode)
    if primary is not None:pool.append(primary)
    if obs is not None:
        dist=cKDTree(obs['xy']).query(target['coords'])[0];keep=dist>1.25;alternative=dict(target)
        for k in ['coords','tangents','strength','components']:alternative[k]=target[k][keep]
        second=measure(source,target,ordered_observations(source,alternative,CONFIG),'A3_alternative',mode)
        if second is not None:pool.append(second)
    decisions=[]
    for r in pool:
        other=[]
        for q in pool:
            if q is r or q['coverage']<.55 or q['span']<.65:continue
            d=cKDTree(r['observed_xy']).query(q['observed_xy'])[0]
            if np.median(d)>=1 or np.mean(d>2)>=.2:other.append(q['cost'])
        cost=min(other) if other else None;r['q_margin']=max(0.,(cost-r['cost'])/(abs(cost)+1e-8)) if cost is not None else 1.
        checks=dict(coverage=r['coverage']>=gate['coverage'],span=r['span']>=.8,information=r['r_phi']>=.65,tangent=r['tangent_good']>=.8,
            order=r['order_consistency']>=.95,anchors=r['anchor_fraction']>=.6,residual=r['residual_rms']<=gate['residual'],
            offset=abs(r['offset'])<=gate['offset'],distance=r['distance_median']<=2.5,margin=r['q_margin']>=gate['margin'],uncertainty=1.96*r['coefficient_se_max']<=1.5)
        decisions.append(dict(**r,accepted=all(checks.values()),gate_checks=checks))
    good=[r for r in decisions if r['accepted']];preferred=[r for r in good if r['branch']=='R0']
    selected=(preferred or sorted(good,key=lambda r:r['cost']))
    auto=dict(selected[0],decisions=decisions) if selected else dict(accepted=False,coefficient=None,decisions=decisions,reason='no_confident_correspondence')
    if primary is None:oracle=dict(accepted=False,coefficient=None)
    else:oracle=dict(primary,accepted=primary['coverage']>=.45 and primary['span']>=.65 and primary['r_phi']>=.4 and primary['residual_rms']<=1.25 and 1.96*primary['coefficient_se_max']<=1.5)
    return dict(Auto=auto,Oracle=oracle)

# From curve_broad_peak_code.py:92; numerical body unchanged.
def code_score(observation,book):
    if not observation.get('accepted') or book is None:return None
    a=np.array(book['bit0']);b=np.array(book['bit1']);direction=b-a
    return float(np.dot(np.array(observation['coefficient'])-(a+b)/2,direction)/max(np.dot(direction,direction)/2,1e-9))
