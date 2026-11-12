"""Message-independent ambiguity rejection, frozen R0-first with A3 fallback."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from canny_curve_stage2_bit import matcher,canonical
from curve_receiver_stage2b import prepare_target,ordered_observations,path_observations

CONFIG=dict(tube=6,angle=30,skip=4,bridge=6,r_phi=.5)
M=matcher()

def geometry(source,basis_name='S'):
    p=canonical(source);u=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];u/=u[-1]
    tangent=M.tangent(p);normal=np.c_[-tangent[:,1],tangent[:,0]]
    phi=6*np.sqrt(3)*u*(1-u)*(2*u-1) if basis_name=='S' else 16*u*u*(1-u)**2
    return p,u,normal,phi

def fit(source,indices,xy,weights=None,basis_name='S'):
    p,u,n,phi=geometry(source,basis_name);ii=np.array(indices,int);xy=np.array(xy,float)
    if len(ii)<6:return dict(fitted=False)
    original=np.ones(len(ii)) if weights is None else np.array(weights,float)
    y=np.sum((xy-p[ii])*n[ii],1);x=np.c_[np.ones(len(ii)),phi[ii]];w=original.copy()
    if np.linalg.cond(x.T@(w[:,None]*x))>500:return dict(fitted=False)
    for _ in range(25):
        beta=np.linalg.lstsq(x*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0];r=y-x@beta
        sigma=max(.25,1.4826*np.median(np.abs(r-np.median(r))));w=original*np.minimum(1,1.345*sigma/np.maximum(abs(r),1e-8))
    stations=np.unique(np.r_[np.arange(0,len(p),2),len(p)-1]).astype(int);span=float(np.ptp(u[ii]));length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
    info=float(np.sum(w*phi[ii]**2)/max(np.sum(phi[stations]**2),1e-9));rms=float(np.sqrt(np.average(r*r,weights=w)))
    sigma=max(.15,1.4826*np.median(np.abs(r-np.median(r))));eff=max(2,min(len(ii),span*length/4));cov=np.linalg.inv(x.T@(w[:,None]*x))*sigma*sigma*len(ii)/eff
    se=float(np.sqrt(cov[1,1]));return dict(fitted=True,c_hat=float(beta[1]),offset=float(beta[0]),coefficient_se=se,ci95=[float(beta[1]-1.96*se),float(beta[1]+1.96*se)],
        residual_rms=rms,r_phi=min(1,info),coverage=len(ii)/len(stations),span=span,indices=ii.tolist(),observed_xy=xy.tolist(),weights=w.tolist(),normal_positions=y.tolist())

def binary_target(edge,source=None):
    if source is not None:
        p=np.array(source);lo=np.maximum(0,np.floor(p.min(0)-12)).astype(int);hi=np.minimum(edge.shape[::-1],np.ceil(p.max(0)+13)).astype(int)
        local=np.zeros_like(edge);local[lo[1]:hi[1],lo[0]:hi[0]]=edge[lo[1]:hi[1],lo[0]:hi[0]];edge=local
    g=M.skeleton_graph(edge);xy=g['coords'];tree=cKDTree(xy);tangents=[]
    for point in xy:
        ids=tree.query_ball_point(point,3.25);q=xy[ids]-point
        cov=q.T@q;_,vectors=np.linalg.eigh(cov);tangents.append(vectors[:,-1])
    from scipy.sparse.csgraph import connected_components
    _,components=connected_components(g['graph'],directed=False)
    return dict(**g,tangents=np.array(tangents).reshape(-1,2),strength=np.ones(len(xy)),components=components)

def measure(source,target,obs,branch,basis_name):
    if obs is None:return None
    f=fit(source,obs['indices'],obs['xy'],obs.get('weights'),basis_name)
    if not f['fitted']:return None
    p,u,n,phi=geometry(source,basis_name);ii=np.array(f['indices']);xy=np.array(f['observed_xy']);t=M.tangent(p)
    _,nearest=cKDTree(target['coords']).query(xy);angle=np.degrees(np.arccos(np.clip(np.abs(np.sum(t[ii]*target['tangents'][nearest],1)),0,1)))
    distance=np.linalg.norm(xy-p[ii],axis=1);anchors=np.rint(np.array([.15,.3,.5,.7,.85])*(len(p)-1)).astype(int)
    ad,ai=cKDTree(xy).query(p[anchors]);anchor_good=(ad<=3)&(angle[ai]<=30)
    projection=cKDTree(p).query(xy)[1];order=float(np.mean(np.diff(projection)>=0)) if len(ii)>1 else 0.
    cost=.35*float(np.median(distance))+f['residual_rms']+2*(1-min(1,f['coverage']))+.8*float(np.mean(angle))/30+.5*abs(f['offset'])
    f.update(branch=branch,distance_median=float(np.median(distance)),angle_mean=float(np.mean(angle)),tangent_good=float(np.mean(angle<=30)),anchor_fraction=float(anchor_good.mean()),order_consistency=order,cost=cost)
    return f

def candidates(source,target,thresholds,basis_name='S'):
    strict=M.curve_match(canonical(source),target,thresholds,True);pool=[]
    if strict['matched']:
        ii,xy,w=path_observations(source,strict['output_path']);r=measure(source,target,dict(indices=ii,xy=xy,weights=w),'R0',basis_name)
        if r is not None:pool.append(r)
    obs=ordered_observations(source,target,CONFIG)
    primary=measure(source,target,obs,'A3',basis_name)
    if primary is not None:pool.append(primary)
    if obs is not None:
        # Suppress the selected observed contour, never fit interpolated gap
        # pixels. An alternative must be a geometrically distinct observation.
        dist=cKDTree(obs['xy']).query(target['coords'])[0];keep=dist>1.25
        alternative=dict(target)
        for key in ['coords','tangents','strength','components']:alternative[key]=target[key][keep]
        second=measure(source,target,ordered_observations(source,alternative,CONFIG),'A3_alternative',basis_name)
        if second is not None:pool.append(second)
    for r in pool:
        alternatives=[];rp=np.array(r['observed_xy'])
        for other in pool:
            if other is r or other['coverage']<.55 or other['span']<.65:continue
            d=cKDTree(rp).query(np.array(other['observed_xy']))[0]
            if np.median(d)>=1 or np.mean(d>2)>=.2:alternatives.append(other['cost'])
        second=min(alternatives) if alternatives else None
        r['runner_up_cost']=second;r['runner_up_missing']=second is None
        r['q_margin']=max(0.,(second-r['cost'])/(abs(second)+1e-8)) if second is not None else 1.
    return dict(strict_matched=strict['matched'],strict_reason=strict['reason'],candidates=pool)

def receive(extracted,gate):
    decisions=[]
    for r in extracted['candidates']:
        checks=dict(coverage=r['coverage']>=gate['coverage'],span=r['span']>=.8,information=r['r_phi']>=.65,
            tangent=r['tangent_good']>=.8,order=r['order_consistency']>=.95,anchors=r['anchor_fraction']>=.6,
            residual=r['residual_rms']<=gate['residual'],offset=abs(r['offset'])<=gate['offset'],distance=r['distance_median']<=2.5,
            margin=r['q_margin']>=gate['margin'],uncertainty=1.96*r['coefficient_se']<=1.5)
        decisions.append(dict(**r,accepted=all(checks.values()),gate_checks=checks))
    eligible=[r for r in decisions if r['accepted']]
    strict=[r for r in eligible if r['branch']=='R0'];selected=(strict or sorted(eligible,key=lambda r:r['cost']))
    if not selected:return dict(accepted=False,c_hat=None,q_sync=0.,decisions=decisions,reason='no_confident_correspondence')
    r=dict(selected[0]);r['q_sync']=float(r['r_phi']*np.exp(-r['residual_rms'])*(.5+.5*r['q_margin']));r['decisions']=decisions
    return r

def oracle_mask(source,target,basis_name='QIM'):
    # The same source curve and search tube are used for clean and both bits.
    # This oracle supplies location, not true correspondence or message shape.
    obs=ordered_observations(source,target,CONFIG)
    r=measure(source,target,obs,'OracleMask',basis_name)
    if r is None:return dict(accepted=False,c_hat=None,q_sync=0.)
    accepted=r['coverage']>=.45 and r['span']>=.65 and r['r_phi']>=.4 and r['residual_rms']<=1.25 and 1.96*r['coefficient_se']<=1.5
    return dict(**r,accepted=accepted,q_sync=float(r['r_phi']*np.exp(-r['residual_rms'])))
