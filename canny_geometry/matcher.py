"""Extracted frozen graph geometry; no experiment-directory dependencies."""
from __future__ import annotations
import math
import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

# From outputs/canny_broad_peak_code_20261111/source/frozen_stage1_matcher.py:26; numerical body unchanged.
def thinning(edge):
    """Zhang-Suen thinning; OpenCV Canny input, no extra contrib dependency."""
    a=np.pad(edge.astype(np.uint8),1)
    while True:
        removed=0
        for phase in (0,1):
            p=a[1:-1,1:-1]
            ns=[a[:-2,1:-1],a[:-2,2:],a[1:-1,2:],a[2:,2:],a[2:,1:-1],a[2:,:-2],a[1:-1,:-2],a[:-2,:-2]]
            b=sum(ns);trans=sum(((ns[k]==0)&(ns[(k+1)%8]==1)).astype('uint8') for k in range(8))
            if phase==0:keep=(ns[0]*ns[2]*ns[4]==0)&(ns[2]*ns[4]*ns[6]==0)
            else:keep=(ns[0]*ns[2]*ns[6]==0)&(ns[0]*ns[4]*ns[6]==0)
            erase=(p==1)&(b>=2)&(b<=6)&(trans==1)&keep
            removed+=int(erase.sum());p[erase]=0
        if removed==0:break
    return a[1:-1,1:-1]>0

# From outputs/canny_broad_peak_code_20261111/source/frozen_stage1_matcher.py:42; numerical body unchanged.
def skeleton_graph(edge):
    sk=thinning(edge);yy,xx=np.nonzero(sk);coords=np.column_stack([xx,yy]).astype('float64')
    lookup=np.full(sk.shape,-1,np.int32);lookup[yy,xx]=np.arange(len(xx))
    rr=[];cc=[];ww=[]
    for dx,dy in ((1,0),(0,1),(1,1),(-1,1)):
        nx,ny=xx+dx,yy+dy;valid=(nx>=0)&(nx<sk.shape[1])&(ny>=0)&(ny<sk.shape[0])
        ids=np.nonzero(valid)[0];nid=lookup[ny[valid],nx[valid]];ok=nid>=0
        if dx and dy:
            # Drop diagonal triangles when a cardinal two-step route exists.
            ok&=~(sk[yy[ids],nx[valid]]|sk[ny[valid],xx[ids]])
        ids=ids[ok];nid=nid[ok]
        rr.extend([ids,nid]);cc.extend([nid,ids]);ww.extend([np.full(len(ids),math.hypot(dx,dy))]*2)
    rr=np.concatenate(rr);cc=np.concatenate(cc);ww=np.concatenate(ww)
    graph=csr_matrix((ww,(rr,cc)),shape=(len(xx),len(xx)))
    return dict(skeleton=sk,coords=coords,graph=graph,lookup=lookup)

# From outputs/canny_broad_peak_code_20261111/source/frozen_stage1_matcher.py:58; numerical body unchanged.
def traces(graph):
    g=graph['graph'];coords=graph['coords'];degree=np.diff(g.indptr);seen=set();result=[]
    def walk(a,b):
        path=[a];previous=a;current=b;seen.add((min(a,b),max(a,b)))
        while True:
            path.append(current)
            if degree[current]!=2:break
            options=g.indices[g.indptr[current]:g.indptr[current+1]]
            nxt=int(options[0] if options[0]!=previous else options[1]);key=(min(current,nxt),max(current,nxt))
            if key in seen:break
            seen.add(key);previous,current=current,nxt
        if len(path)>1:result.append(coords[path])
    for a in np.nonzero(degree!=2)[0]:
        for b in g.indices[g.indptr[a]:g.indptr[a+1]]:
            if (min(a,b),max(a,b)) not in seen:walk(int(a),int(b))
    for a in np.nonzero(degree==2)[0]:
        for b in g.indices[g.indptr[a]:g.indptr[a+1]]:
            if (min(a,b),max(a,b)) not in seen:walk(int(a),int(b))
    return result,dict(vertices=len(coords),edges=int(g.nnz//2),endpoints=int((degree==1).sum()),junction_pixels=int((degree>=3).sum()),primitive_traces=len(result))

# From outputs/canny_broad_peak_code_20261111/source/frozen_stage1_matcher.py:78; numerical body unchanged.
def resample(points,spacing=1):
    p=np.asarray(points,dtype=float);ds=np.linalg.norm(np.diff(p,axis=0),axis=1)
    p=p[np.r_[True,ds>1e-8]]
    if len(p)<2:return p
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    q=np.linspace(0,arc[-1],max(2,int(np.ceil(arc[-1]/spacing))+1))
    return np.column_stack([np.interp(q,arc,p[:,axis]) for axis in (0,1)])

# From outputs/canny_broad_peak_code_20261111/source/frozen_stage1_matcher.py:86; numerical body unchanged.
def tangent(p):
    smooth=gaussian_filter1d(np.asarray(p,float),1.5,axis=0,mode='nearest')
    v=np.gradient(smooth,axis=0);return v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-9)

# From outputs/canny_broad_peak_code_20261111/source/frozen_stage1_matcher.py:90; numerical body unchanged.
def length(p):return float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())

# From outputs/canny_broad_peak_code_20261111/source/frozen_stage1_matcher.py:92; numerical body unchanged.
def source_curves(edge,thresholds):
    graph=skeleton_graph(edge);raw,stats=traces(graph);pool=[]
    for ti,pixels in enumerate(raw):
        if length(pixels)<thresholds['min_source_arc_px']:continue
        p=resample(pixels)
        tt=tangent(p);window=5
        angles=np.zeros(len(p));angles[window:-window]=np.degrees(np.arccos(np.clip(np.sum(tt[:-2*window]*tt[2*window:],axis=1),-1,1)))
        peaks=find_peaks(angles,height=thresholds['curvature_split_deg'],distance=16)[0].tolist()
        if np.linalg.norm(p[0]-p[-1])<3:
            peaks+=list(range(len(p)//4,len(p),max(1,len(p)//4)))
        cuts=sorted(set([0,len(p)-1]+[int(i) for i in peaks if 12<=i<len(p)-12]))
        for a,b in zip(cuts[:-1],cuts[1:]):
            if b-a<thresholds['min_source_arc_px']:continue
            n=max(1,int(np.ceil((b-a)/thresholds['max_source_arc_px'])))
            local=np.linspace(a,b,n+1).astype(int)
            for start,end in zip(local[:-1],local[1:]):
                q=p[start:end+1];ll=length(q)
                if ll<thresholds['min_source_arc_px'] or np.linalg.norm(q[0]-q[-1])<12:continue
                degree=min(3,len(q)-1)
                try:
                    tck,u=splprep(q.T,u=np.linspace(0,1,len(q)),s=len(q)*thresholds['spline_smoothing_rms_px']**2,k=degree)
                    fit=np.column_stack(splev(np.linspace(0,1,len(q)),tck));fit=resample(fit)
                    fitted=True
                except (ValueError,TypeError):fit=q;fitted=False
                drift=cKDTree(q).query(fit)[0]
                if float(np.max(drift))>1.25:fit=q;fitted=False
                pix=np.unique(np.rint(q).astype(int),axis=0)
                pool.append(dict(primitive_trace_index=ti,source_points=q.tolist(),samples=fit.tolist(),spline_fitted=fitted,
                    arc_length_px=length(fit),chord_px=float(np.linalg.norm(fit[-1]-fit[0])),source_raster_points=pix.tolist()))
    kept=[];trees=[]
    for c in sorted(pool,key=lambda c:-c['arc_length_px']):
        pts=np.asarray(c['samples']);duplicate=False
        for old,tree in zip(kept,trees):
            if abs(np.asarray(old['samples']).mean(0)-pts.mean(0)).max()>(old['arc_length_px']+c['arc_length_px'])/2+4:continue
            if float((tree.query(pts)[0]<=thresholds['source_duplicate_proximity_px']).mean())>thresholds['source_duplicate_overlap']:
                duplicate=True;break
        if duplicate:continue
        c['curve_index']=len(kept);kept.append(c);trees.append(cKDTree(pts))
    stats.update(candidates_before_dedup=len(pool),source_candidates=len(kept))
    return kept,graph,stats

# From outputs/canny_broad_peak_code_20261111/source/frozen_stage1_matcher.py:133; numerical body unchanged.
def curve_match(source,target,thresholds,return_path=False):
    """Find a connected output graph route; never link disconnected clouds."""
    source=np.asarray(source,float);coords=target['coords'];tube=thresholds['tube_radius_px']
    result=dict(matched=False,path_found=False,reason='no_endpoint_route')
    if not len(coords):return result
    low=source.min(0)-tube;high=source.max(0)+tube
    ids=np.nonzero(((coords>=low)&(coords<=high)).all(1))[0]
    if not len(ids):return result
    dist,projection=cKDTree(source).query(coords[ids]);keep=dist<=tube
    ids=ids[keep];dist=dist[keep];projection=projection[keep];q=coords[ids]
    start_dist=np.linalg.norm(q-source[0],axis=1);end_dist=np.linalg.norm(q-source[-1],axis=1)
    starts=np.nonzero((start_dist<=thresholds['endpoint_search_px'])&(projection<=max(5,.15*len(source))))[0]
    ends=np.nonzero((end_dist<=thresholds['endpoint_search_px'])&(projection>=len(source)-1-max(5,.15*len(source))))[0]
    if not len(starts) or not len(ends):return result
    sub=target['graph'][ids][:,ids].tocoo();rr=sub.row;cc=sub.col
    allowed=projection[cc]>=projection[rr]-2
    rr=rr[allowed];cc=cc[allowed];weights=sub.data[allowed]*(1+.2*(dist[rr]+dist[cc])/2)
    count=len(ids)
    graph=csr_matrix((np.r_[weights,start_dist[starts]+1e-6],(np.r_[rr,np.full(len(starts),count)],np.r_[cc,starts])),shape=(count+1,count+1))
    costs,predecessor=dijkstra(graph,directed=True,indices=count,return_predecessors=True)
    scores=costs[ends]+end_dist[ends];best=int(ends[np.argmin(scores)])
    if not np.isfinite(costs[best]):return result
    route=[];current=best
    while current!=count:
        route.append(current);current=int(predecessor[current])
        if current<0:return result
    path=q[route[::-1]]
    if len(path)<3:return result
    sampled=resample(path);ds,near=cKDTree(sampled).query(source);dt,back=cKDTree(source).query(sampled)
    st=tangent(source);tt=tangent(sampled);angles=np.degrees(np.arccos(np.clip(np.abs(np.sum(st*tt[near],axis=1)),0,1)))
    normal=np.column_stack([-st[:,1],st[:,0]]);offset=sampled[near]-source
    signed=np.sum(offset*normal,axis=1)
    anchor_ids=np.rint(np.linspace(0,len(source)-1,5)).astype(int)
    anchor_dist=ds[anchor_ids];anchor_good=(anchor_dist<=thresholds['anchor_tolerance_px'])&(angles[anchor_ids]<=thresholds['orientation_tolerance_deg'])
    endpoint=np.array([np.linalg.norm(path[0]-source[0]),np.linalg.norm(path[-1]-source[-1])])
    ratio=length(sampled)/length(source)
    coverage=float((ds<=thresholds['anchor_tolerance_px']).mean());angle_fraction=float((angles<=thresholds['orientation_tolerance_deg']).mean())
    symmetric_median=max(float(np.median(ds)),float(np.median(dt)))
    symmetric_p90=max(float(np.percentile(ds,90)),float(np.percentile(dt,90)))
    heading=np.unwrap(np.arctan2(tt[:,1],tt[:,0]));total_turn=float(heading[-1]-heading[0])
    criteria=dict(distance=symmetric_median<=thresholds['max_symmetric_median_px'] and symmetric_p90<=thresholds['max_symmetric_p90_px'],
        coverage=coverage>=thresholds['min_curve_coverage'],orientation=angle_fraction>=.8,
        length=thresholds['min_length_ratio']<=ratio<=thresholds['max_length_ratio'],
        anchors=float(anchor_good.mean())>=.8,endpoints=bool(np.all(endpoint<=thresholds['anchor_tolerance_px'])))
    result.update(matched=bool(all(criteria.values())),path_found=True,reason='pass' if all(criteria.values()) else ','.join(k for k,v in criteria.items() if not v),
        symmetric_median_px=symmetric_median,symmetric_p90_px=symmetric_p90,source_coverage=coverage,
        tangent_mean_deg=float(np.mean(angles)),tangent_good_fraction=angle_fraction,length_ratio=ratio,
        anchor_good_fraction=float(anchor_good.mean()),anchor_distances_px=anchor_dist.tolist(),endpoint_distances_px=endpoint.tolist(),
        signed_normal_mean_px=float(signed.mean()),normal_rms_px=float(np.sqrt(np.mean(signed**2))),total_turn_rad=total_turn,
        mean_abs_curvature_per_px=float(np.abs(np.diff(heading)).sum()/max(1,length(sampled))),
        normal_profile_32_px=np.interp(np.linspace(0,1,32),np.linspace(0,1,len(signed)),signed).tolist(),
        route_vertices=len(path),route_arc_length_px=length(path),criteria=criteria)
    if return_path:result['output_path']=path.tolist()
    return result
