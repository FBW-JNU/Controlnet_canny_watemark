"""Graph-constrained curve matching, discovery-only masks and heldout audit."""
from __future__ import annotations
import argparse
import csv
import heapq
import json
import math
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy.interpolate import splprep,splev
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from canny_curve_stage1_32 import ROOT,OUT,read,dump,sha

METHOD_VERSION='graphcurve1'

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

def resample(points,spacing=1):
    p=np.asarray(points,dtype=float);ds=np.linalg.norm(np.diff(p,axis=0),axis=1)
    p=p[np.r_[True,ds>1e-8]]
    if len(p)<2:return p
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    q=np.linspace(0,arc[-1],max(2,int(np.ceil(arc[-1]/spacing))+1))
    return np.column_stack([np.interp(q,arc,p[:,axis]) for axis in (0,1)])

def tangent(p):
    smooth=gaussian_filter1d(np.asarray(p,float),1.5,axis=0,mode='nearest')
    v=np.gradient(smooth,axis=0);return v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-9)

def length(p):return float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())

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

def wilson(success,n):
    if not n:return [None,None]
    z=1.959963984540054;p=success/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,mid-half),min(1,mid+half)]

def aggregate(rows):
    n=len(rows);count=sum(r['matched'] for r in rows);found=[r for r in rows if r['path_found']];matched=[r for r in rows if r['matched']]
    result=dict(views=n,matched=count,R=count/n if n else None,R_wilson95=wilson(count,n),paths_found=len(found),
        anchor_view_rate=sum(r.get('anchor_good_fraction',0)>=.8 for r in rows)/n if n else None,
        per_prompt={str(pi):dict(views=len(rs),matched=sum(r['matched'] for r in rs),R=sum(r['matched'] for r in rs)/len(rs))
                    for pi in sorted({r['prompt_index'] for r in rows}) for rs in [[r for r in rows if r['prompt_index']==pi]]},
        D=dict(median_symmetric_px=float(np.median([r['symmetric_median_px'] for r in matched])) if matched else None,
               median_endpoint_px=float(np.median([v for r in matched for v in r['endpoint_distances_px']])) if matched else None,
               median_signed_normal_px=float(np.median([r['signed_normal_mean_px'] for r in matched])) if matched else None),
        V=dict(samples=len(matched),defined=len(matched)>=2,feature_variances={}))
    if len(matched)>=2:
        for key in ('length_ratio','total_turn_rad','mean_abs_curvature_per_px','signed_normal_mean_px','normal_rms_px'):
            result['V']['feature_variances'][key]=float(np.var([r[key] for r in matched],ddof=1))
        profile=np.asarray([r['normal_profile_32_px'] for r in matched])
        result['V']['normal_profile_variance_32_px2']=np.var(profile,axis=0,ddof=1).tolist()
        result['V']['mean_normal_profile_variance_px2']=float(np.var(profile,axis=0,ddof=1).mean())
    return result

def csvout(path,rows):
    if not rows:path.write_text('');return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));writer.writeheader();writer.writerows(rows)

def source_case(case):
    cid=case['content_id'];path=OUT/'curves'/f'{cid:012d}_source.json'
    if path.exists():return read(path)
    edge=np.asarray(Image.open(case['hint']))>0;x0,y0,x1,y1=case['roi'];valid=np.zeros_like(edge);valid[y0:y1,x0:x1]=True;edge&=valid
    thresholds=read(OUT/'protocol.json')['exploration_thresholds'];curves,graph,stats=source_curves(edge,thresholds)
    data=dict(content_id=cid,method_version=METHOD_VERSION,source_sha256=sha(case['hint']),stats=stats,curves=curves)
    dump(path,data);Image.fromarray(graph['skeleton'].astype('uint8')*255).save(OUT/'graphs'/f'{cid:012d}_source_skeleton.png')
    g=graph['graph']
    np.savez_compressed(OUT/'graphs'/f'{cid:012d}_source_graph.npz',coords_xy=graph['coords'],indptr=g.indptr,indices=g.indices,
                        edge_length=g.data,degree=np.diff(g.indptr),roi=case['roi'])
    print('SOURCE',cid,stats,flush=True);return data

def match_case(case,discovery_only=False):
    cv2.setNumThreads(1);cid=case['content_id'];thresholds=read(OUT/'protocol.json')['exploration_thresholds']
    source=source_case(case);jobs=[j for j in read(OUT/'plan.json') if j['content_id']==cid]
    for job in jobs:
        if discovery_only and job['role']!='discovery':continue
        cache=OUT/'metrics'/f'{job["job"]}_matches.json'
        if cache.exists():
            data=read(cache);assert data['method_version']==METHOD_VERSION and data['rgb_sha256']==sha(job['rgb']);continue
        if not (OUT/'records'/f'{job["job"]}.json').exists():continue
        record=read(OUT/'records'/f'{job["job"]}.json');assert sha(job['rgb'])==record['rgb_sha256']
        edge=cv2.Canny(np.asarray(Image.open(job['rgb']).convert('RGB')),100,200)>0
        x0,y0,x1,y1=case['roi'];valid=np.zeros_like(edge);valid[y0:y1,x0:x1]=True;edge&=valid
        Image.fromarray(edge.astype('uint8')*255).save(OUT/'output_canny'/f'{job["job"]}.png')
        graph=skeleton_graph(edge);results=[];begun=time.monotonic()
        for curve in source['curves']:
            result=curve_match(curve['samples'],graph,thresholds,return_path=True)
            result.update(curve_index=curve['curve_index'],job=job['job'],role=job['role'],prompt_index=job['prompt_index'],seed_index=job['seed_index'])
            results.append(result)
        dump(cache,dict(method_version=METHOD_VERSION,content_id=cid,job=job['job'],role=job['role'],rgb_sha256=sha(job['rgb']),
            source_curves_sha256=sha(OUT/'curves'/f'{cid:012d}_source.json'),matches=results))
        print('MATCH',job['job'],job['role'],'accepted',sum(r['matched'] for r in results),'/',len(results),'seconds',round(time.monotonic()-begun,1),flush=True)

def select_case(case):
    cid=case['content_id'];path=OUT/'curves'/f'{cid:012d}_selected.json'
    if path.exists():return read(path)
    jobs=[j for j in read(OUT/'plan.json') if j['content_id']==cid and j['role']=='discovery']
    if not all((OUT/'metrics'/f'{j["job"]}_matches.json').exists() for j in jobs):return None
    data=source_case(case);perview=[read(OUT/'metrics'/f'{j["job"]}_matches.json')['matches'] for j in jobs]
    thresholds=read(OUT/'protocol.json')['exploration_thresholds'];allrows=[];selected=[]
    for curve in data['curves']:
        index=curve['curve_index'];stats=aggregate([rs[index] for rs in perview])
        passed=stats['R']>=thresholds['discovery_retention_min'] and stats['anchor_view_rate']>=thresholds['discovery_anchor_rate_min'] and all(v['R']>=thresholds['per_discovery_prompt_retention_min'] for v in stats['per_prompt'].values())
        row=dict(curve_index=index,selected=passed,discovery=stats)
        allrows.append(row)
        if passed:selected.append(row)
    selected.sort(key=lambda r:(-r['discovery']['R'],r['discovery']['D']['median_symmetric_px'],r['curve_index']))
    # Source-only duplicate suppression was performed before seeing any outputs.
    confidence=np.zeros((512,512),np.float32);lower=np.zeros_like(confidence);lines=np.zeros((512,512),np.uint8)
    x0,y0,x1,y1=case['roi'];valid=np.zeros((512,512),bool);valid[y0:y1,x0:x1]=True
    individual=OUT/'masks'/f'{cid:012d}';individual.mkdir(exist_ok=True)
    for row in selected:
        c=data['curves'][row['curve_index']];mask=np.zeros((512,512),np.uint8)
        pts=np.rint(c['samples']).astype('int32');cv2.polylines(mask,[pts.reshape(-1,1,2)],False,255,1)
        lines|=mask;radius=thresholds['mask_half_width_px'];tube=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))>0
        tube&=valid
        confidence[tube]=np.maximum(confidence[tube],row['discovery']['R']);lower[tube]=np.maximum(lower[tube],row['discovery']['R_wilson95'][0])
        Image.fromarray(tube.astype('uint8')*255).save(individual/f'curve_{row["curve_index"]:04d}.png')
        row['mask']=str((individual/f'curve_{row["curve_index"]:04d}.png').relative_to(OUT))
    confidence[~valid]=0;lower[~valid]=0
    Image.fromarray(lines).save(OUT/'masks'/f'{cid:012d}_selected_curves.png')
    Image.fromarray((confidence>0).astype('uint8')*255).save(OUT/'masks'/f'{cid:012d}_region.png')
    Image.fromarray(np.rint(confidence*255).astype('uint8')).save(OUT/'masks'/f'{cid:012d}_confidence.png')
    context=cv2.dilate((confidence>0).astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))>0;context&=valid
    Image.fromarray(context.astype('uint8')*255).save(OUT/'masks'/f'{cid:012d}_context.png')
    np.savez_compressed(OUT/'masks'/f'{cid:012d}_confidence.npz',confidence=confidence,wilson95_lower=lower)
    result=dict(content_id=cid,method_version=METHOD_VERSION,selection_used_roles=['discovery'],discovery_jobs=[j['job'] for j in jobs],
        mask_meaning='4px tube around selected source curves, clipped to valid photo; filled pixels are candidate region, not certified stable edge pixels. Confidence is max discovery curve R in overlapping tubes; Wilson lower bounds stored separately.',
        source_curves_sha256=sha(OUT/'curves'/f'{cid:012d}_source.json'),discovery_match_hashes={j['job']:sha(OUT/'metrics'/f'{j["job"]}_matches.json') for j in jobs},
        candidates=len(allrows),selected_count=len(selected),selected=selected,all_candidates=allrows)
    dump(path,result);print('SELECTED',cid,len(selected),'/',len(allrows),flush=True);return result

def run(content_index=None,discovery_only=False):
    cases=read(OUT/'selected.json')
    for c in cases:
        if content_index is not None and c['content_index']!=content_index:continue
        source_case(c);match_case(c,True);select_case(c)
        if not discovery_only:match_case(c,False)

def watch():
    while True:
        run()
        ready=len(list((OUT/'metrics').glob('*_matches.json')))
        dump(OUT/'analysis_status.json',dict(status='complete' if ready==384 else 'running',completed_views=ready,planned_views=384,
            discovery_frozen_contents=len(list((OUT/'curves').glob('*_selected.json')))))
        print('ANALYSIS PROGRESS',ready,'/384',flush=True)
        if ready==384:break
        time.sleep(15)

def synthetic_checks():
    t=read(OUT/'protocol.json')['exploration_thresholds']
    source=np.column_stack([np.arange(20,101),np.full(81,64)])
    def target(points):
        a=np.zeros((128,128),np.uint8)
        for x,y in points:a[int(y),int(x)]=1
        return skeleton_graph(a>0)
    tests={}
    for name,pts,expected in [('identity',source,True),('two_px_shift',source+[0,2],True),('eight_px_shift',source+[0,8],False),
        ('disconnected_cloud',source[::3],False),('perpendicular_crossing',np.column_stack([np.full(81,60),np.arange(24,105)]),False)]:
        result=curve_match(source,target(pts),t);assert result['matched']==expected,(name,result)
        tests[name]=dict(expected=expected,actual=result['matched'],path_found=result['path_found'])
    theta=np.linspace(-1.1,1.1,110);curve=np.column_stack([64+38*np.sin(theta),30+38*(1-np.cos(theta))])
    raster=np.zeros((128,128),np.uint8);cv2.polylines(raster,[np.rint(curve+[0,2]).astype('int32').reshape(-1,1,2)],False,1,1)
    result=curve_match(resample(curve),skeleton_graph(raster>0),t)
    assert result['matched'],result;tests['curved_two_px_shift']=dict(expected=True,actual=result['matched'])
    dump(OUT/'synthetic_validation.json',dict(status='passed',tests=tests));print('SYNTHETIC CHECKS PASSED',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','source','test','watch']);parser.add_argument('--content-index',type=int);parser.add_argument('--discovery-only',action='store_true');a=parser.parse_args()
    if a.action=='test':synthetic_checks()
    elif a.action=='source':
        for c in read(OUT/'selected.json'):source_case(c)
    elif a.action=='watch':watch()
    else:run(a.content_index,a.discovery_only)
