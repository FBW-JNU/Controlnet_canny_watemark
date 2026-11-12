"""R0 frozen baseline, R1 gap graph, R2 ordered observations and robust coefficient.

All correspondence scores are message-independent. No synthesized observations
are allowed in coefficient fitting, including across repaired graph gaps.
"""
from __future__ import annotations
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from canny_curve_stage2_bit import matcher,canonical,fields

M=None
def mathmod():
    global M
    if M is None:M=matcher()
    return M

def prepare_target(rgb,low=100,high=200,soft=None):
    m=mathmod();edge=cv2.Canny(rgb,low,high);g=m.skeleton_graph(edge);xy=g['coords'];yy=xy[:,1].astype(int);xx=xy[:,0].astype(int)
    gx=cv2.Sobel(rgb.astype(np.float32),cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(rgb.astype(np.float32),cv2.CV_32F,0,1,ksize=3)
    channel=np.argmax(gx*gx+gy*gy,axis=2);a=gx[yy,xx,channel[yy,xx]];b=gy[yy,xx,channel[yy,xx]]
    norm=np.maximum(np.hypot(a,b),1e-6);t=np.c_[-b/norm,a/norm]
    strength=np.ones(len(xy)) if soft is None else .25+.75*np.clip(soft[yy,xx],0,1)
    _,components=connected_components(g['graph'],directed=False)
    return dict(**g,tangents=t,strength=strength,components=components,edge=edge)

def basis(source):
    p,u,n,phi=fields(source,'polynomial',mathmod());return p,u,n,phi

def fit_observations(source,indices,xy,weights,robust=True):
    p,u,n,phi=basis(source);indices=np.array(indices,int);xy=np.array(xy,float);w=np.array(weights,float)
    if len(indices)<5:return dict(fitted=False,reason='fewer_than5_observations')
    y=np.sum((xy-p[indices])*n[indices],axis=1);x=np.c_[np.ones(len(indices)),phi[indices]]
    original_w=np.clip(w,0,1);w=original_w.copy();beta=np.zeros(2)
    for iteration in range(25 if robust else 1):
        gram=x.T@(w[:,None]*x)
        if np.linalg.cond(gram)>500:return dict(fitted=False,reason='ill_conditioned')
        new=np.linalg.solve(gram,x.T@(w*y));res=y-x@new
        scale=max(.25,1.4826*float(np.median(np.abs(res-np.median(res)))))
        beta=new
        if robust:w=original_w*np.minimum(1,1.345*scale/np.maximum(np.abs(res),1e-9))
    span=float(u[indices].max()-u[indices].min());arc=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
    effective_n=max(2.,min(len(indices),span*arc/4));sigma=max(.15,1.4826*float(np.median(np.abs(res-np.median(res)))))
    cov=np.linalg.inv(x.T@(w[:,None]*x))*sigma*sigma*len(indices)/effective_n
    se=float(np.sqrt(cov[1,1]));rms=float(np.sqrt(np.average(res*res,weights=w)))
    # Stations are a roughly uniform 2px subset; information denominator uses
    # the same fixed station grid, never only the visible observations.
    stations=np.unique(np.r_[np.arange(0,len(p),2),len(p)-1]).astype(int)
    information=float(np.sum(w*phi[indices]**2)/max(np.sum(phi[stations]**2),1e-9))
    left=indices[u[indices]<.5];right=indices[u[indices]>=.5]
    half=[]
    for low,high in [(0,.5),(.5,1.00001)]:
        obs=(u[indices]>=low)&(u[indices]<high);whole=(u[stations]>=low)&(u[stations]<high)
        half.append(float(np.sum(w[obs]*phi[indices[obs]]**2)/max(np.sum(phi[stations[whole]]**2),1e-9)))
    bins=len(set(np.minimum(5,(u[indices]*6).astype(int))))
    q=float(np.clip(information,0,1)*np.exp(-rms)*min(1,span/.8))
    return dict(fitted=True,c_hat=float(beta[1]),offset=float(beta[0]),coefficient_se=se,ci95=[float(beta[1]-1.96*se),float(beta[1]+1.96*se)],
        r_phi=min(1.,information),half_information=half,coverage=len(indices)/len(stations),span=span,bins=bins,
        residual_rms=rms,effective_n=effective_n,q_sync=q,observations=len(indices),
        indices=indices.tolist(),observed_xy=xy.tolist(),weights=w.tolist(),normal_positions=y.tolist(),residuals=res.tolist())

def gate(fit,config):
    if not fit.get('fitted'):return dict(**fit,accepted=False,q_sync=0.)
    checks=dict(coverage=fit['coverage']>=config.get('coverage',.45),span=fit['span']>=.65,
        information=fit['r_phi']>=config.get('r_phi',.4),both_lobes=min(fit['half_information'])>=.22,
        bins=fit['bins']>=4,residual=fit['residual_rms']<=1.25,uncertainty=1.96*fit['coefficient_se']<=1.5)
    return dict(**fit,accepted=all(checks.values()),gate_checks=checks)

def path_observations(source,path,max_distance=4):
    p,u,n,phi=basis(source);stations=np.unique(np.r_[np.arange(0,len(p),2),len(p)-1]).astype(int)
    path=np.array(path,float);d,near=cKDTree(path).query(p[stations]);keep=d<=max_distance
    indices=stations[keep];xy=path[near[keep]]
    # One real edge vertex cannot provide repeated independent evidence.
    _,unique=np.unique(xy,axis=0,return_index=True);unique=np.sort(unique)
    return indices[unique],xy[unique],np.ones(len(unique))

def input_codebook(source,paths,robust=True):
    features={}
    for arm,path in paths.items():
        i,xy,w=path_observations(source,path);f=fit_observations(source,i,xy,w,robust)
        if not f.get('fitted'):raise ValueError(('input fit',arm,f))
        features[arm]=f['c_hat']
    features['midpoint']=(features['bit0']+features['bit1'])/2
    features['half_separation']=(features['bit1']-features['bit0'])/2
    if features['half_separation']<=.25:raise ValueError(('input insufficient separation',features))
    return features

def r1(source,target,thresholds,config):
    m=mathmod();s=canonical(source);a=round(.1*(len(s)-1));z=round(.9*(len(s)-1));internal=s[a:z+1]
    g=target['graph'].copy();xy=target['coords'];degree=np.diff(g.indptr)
    local=((xy>=internal.min(0)-10)&(xy<=internal.max(0)+10)).all(1)
    ids=np.flatnonzero((degree<=1)&local);st=m.tangent(s);source_tree=cKDTree(s)
    endpoints=xy[ids];rr=[];cc=[];dd=[]
    if len(ids):
        tree=cKDTree(endpoints)
        for ai,bi in tree.query_pairs(config['gap']):
            aa,bb=ids[ai],ids[bi];delta=xy[bb]-xy[aa];distance=np.linalg.norm(delta)
            if distance<1.01:continue
            _,near=source_tree.query((xy[aa]+xy[bb])/2)
            if abs(np.dot(delta/distance,st[near]))<np.cos(np.deg2rad(50)):continue
            rr.extend([aa,bb]);cc.extend([bb,aa]);dd.extend([distance*1.5]*2)
        if rr:g=g+csr_matrix((dd,(rr,cc)),shape=g.shape)
    t=thresholds.copy();t.update(tube_radius_px=8,endpoint_search_px=config['endpoint'],anchor_tolerance_px=config['endpoint'],
        max_symmetric_median_px=4,max_symmetric_p90_px=8,min_curve_coverage=.7)
    match=m.curve_match(internal,dict(**{k:v for k,v in target.items() if k!='graph'},graph=g),t,True)
    if not match['matched']:return dict(accepted=False,fitted=False,q_sync=0.,reason=match['reason'],route=match.get('output_path'))
    # Bridging edges are used for localization, but only their observed end
    # pixels enter the feature estimator. No interpolated bridge pixels.
    i,xy,w=path_observations(source,match['output_path']);result=gate(fit_observations(source,i,xy,w,False),dict(r_phi=.3,coverage=.35))
    result['route']=match['output_path'];return result

def ordered_observations(source,target,config):
    """Viterbi over real edge vertices with skipped stations and shape costs."""
    m=mathmod();s,u,n,phi=basis(source);st=m.tangent(s);stations=np.unique(np.r_[np.arange(0,len(s),2),len(s)-1]).astype(int)
    coords=target['coords'];low=s.min(0)-config['tube'];high=s.max(0)+config['tube']
    ids=np.flatnonzero(((coords>=low)&(coords<=high)).all(1))
    if not len(ids):return None
    distances,projection=cKDTree(s).query(coords[ids]);keep=distances<=config['tube'];ids=ids[keep];distances=distances[keep];projection=projection[keep]
    if not len(ids):return None
    angle=np.degrees(np.arccos(np.clip(np.abs(np.sum(st[projection]*target['tangents'][ids],axis=1)),0,1)))
    keep=angle<=config['angle'];ids=ids[keep];distances=distances[keep];projection=projection[keep];angle=angle[keep]
    candidates=[]
    for j in stations:
        kk=np.flatnonzero(np.abs(projection-j)<=1.25)
        local=1-.3*(distances[kk]/config['tube'])**2-.25*(angle[kk]/config['angle'])**2
        local*=target['strength'][ids[kk]]
        kk=kk[np.argsort(-local,kind='stable')[:8]]
        # Every node is a real observed edge pixel. Its signed displacement
        # uses the fixed station normal, with tangent displacement penalized.
        q=coords[ids[kk]];offset=np.sum((q-s[j])*n[j],axis=1)
        reward=target['strength'][ids[kk]]*(1-.3*(distances[kk]/config['tube'])**2-.25*(angle[kk]/config['angle'])**2)
        candidates.append(dict(ids=ids[kk],xy=q,projection=projection[kk],offset=offset,reward=reward))
    def search(reverse=False):
        order=list(range(len(stations)))[::-1] if reverse else list(range(len(stations)))
        score={};back={};best=(-1.,None)
        for pos,k in enumerate(order):
            cur=candidates[k];size=len(cur['ids']);sc=cur['reward'].copy();bp=[None]*size
            for gap in range(1,min(config['skip']+1,pos)+1):
                previous=order[pos-gap];prev=candidates[previous]
                if not len(prev['ids']) or not size:continue
                ds=float(abs(stations[k]-stations[previous]));dp=cur['projection'][:,None]-prev['projection'][None,:]
                if reverse:dp=-dp
                normal=np.abs(cur['offset'][:,None]-prev['offset'][None,:]);physical=np.linalg.norm(cur['xy'][:,None,:]-prev['xy'][None,:,:],axis=2)
                same=target['components'][cur['ids']][:,None]==target['components'][prev['ids']][None,:]
                valid=(dp>.25)&(normal<=1.5+.35*ds)&(physical<=1.8*ds+2)&(same|(physical<=config['bridge']))
                cost=.16*(gap-1)+.12*normal**2/max(ds,1)+.12*(~same)
                values=np.where(valid,score[previous][None,:]-cost,-1e9)
                near=np.argmax(values,axis=1);v=values[np.arange(size),near]+cur['reward'];improve=v>sc
                for a in np.flatnonzero(improve):sc[a]=v[a];bp[a]=(previous,int(near[a]))
            score[k]=sc;back[k]=bp
            if size and sc.max()>best[0]:best=(float(sc.max()),(k,int(sc.argmax())))
        route=[];node=best[1]
        while node is not None:
            k,a=node;route.append((k,a));node=back[k][a]
        return best[0],route
    forward=search(False);backward=search(True);best=forward if forward[0]>=backward[0] else backward
    route=sorted(best[1]);indices=[];xy=[];weights=[];seen=set()
    for k,a in route:
        cid=int(candidates[k]['ids'][a])
        if cid in seen:continue
        seen.add(cid);indices.append(int(stations[k]));xy.append(coords[cid]);weights.append(float(target['strength'][cid]))
    if len(indices)<5:return None
    return dict(indices=indices,xy=np.array(xy),weights=weights,orientation='forward' if best is forward else 'reverse',path_score=best[0])

def r2(source,target,config,robust=True):
    obs=ordered_observations(source,target,config)
    if obs is None:return dict(accepted=False,fitted=False,q_sync=0.,reason='insufficient_ordered_observations')
    fit=fit_observations(source,obs['indices'],obs['xy'],obs['weights'],robust)
    result=gate(fit,config);result.update(orientation=obs['orientation'],path_score=obs['path_score'])
    return result
