"""Original real-edge observation search; no synthesized points."""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from .geometry import canonical, fields, matcher
M = matcher()

# From curve_receiver_stage2b.py:15; numerical body unchanged.
def mathmod():
    global M
    if M is None:M=matcher()
    return M

# From curve_receiver_stage2b.py:20; numerical body unchanged.
def prepare_target(rgb,low=100,high=200,soft=None):
    m=mathmod();edge=cv2.Canny(rgb,low,high);g=m.skeleton_graph(edge);xy=g['coords'];yy=xy[:,1].astype(int);xx=xy[:,0].astype(int)
    gx=cv2.Sobel(rgb.astype(np.float32),cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(rgb.astype(np.float32),cv2.CV_32F,0,1,ksize=3)
    channel=np.argmax(gx*gx+gy*gy,axis=2);a=gx[yy,xx,channel[yy,xx]];b=gy[yy,xx,channel[yy,xx]]
    norm=np.maximum(np.hypot(a,b),1e-6);t=np.c_[-b/norm,a/norm]
    strength=np.ones(len(xy)) if soft is None else .25+.75*np.clip(soft[yy,xx],0,1)
    _,components=connected_components(g['graph'],directed=False)
    return dict(**g,tangents=t,strength=strength,components=components,edge=edge)

# From curve_receiver_stage2b.py:29; numerical body unchanged.
def basis(source):
    p,u,n,phi=fields(source,'polynomial',mathmod());return p,u,n,phi

# From curve_receiver_stage2b.py:71; numerical body unchanged.
def path_observations(source,path,max_distance=4):
    p,u,n,phi=basis(source);stations=np.unique(np.r_[np.arange(0,len(p),2),len(p)-1]).astype(int)
    path=np.array(path,float);d,near=cKDTree(path).query(p[stations]);keep=d<=max_distance
    indices=stations[keep];xy=path[near[keep]]
    # One real edge vertex cannot provide repeated independent evidence.
    _,unique=np.unique(xy,axis=0,return_index=True);unique=np.sort(unique)
    return indices[unique],xy[unique],np.ones(len(unique))

# From curve_receiver_stage2b.py:114; numerical body unchanged.
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

# From curve_receiver_stage2c.py:32; numerical body unchanged.
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
