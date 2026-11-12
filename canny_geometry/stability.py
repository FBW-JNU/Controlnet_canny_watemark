"""Screen source edge pixels before extracting any source curve candidates.

Nearest-edge support is a proximity proxy. Whole-curve graph matching provides
an additional check; neither step establishes semantic correspondence.
"""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from . import matcher
from .api import THRESHOLDS, validate_edge

POLICY = dict(version='region_first_v1', match_radius_px=3., min_views=5,
              min_per_prompt=1, max_displacement_dispersion_px=2.,
              curve_min_views=5, curve_min_anchor_views=5, curve_median_px=3.)


def validate_plan(plan):
    groups=sorted({str(r['prompt_group']) for r in plan})
    if len(plan)!=6 or len(groups)!=3 or any(sum(str(r['prompt_group'])==g for r in plan)!=2 for g in groups):
        raise ValueError('Use three prompt groups with two seed observations each.')
    if all('seed' in r for r in plan):
        if any(len({r['seed'] for r in plan if str(r['prompt_group'])==g})!=2 for g in groups):
            raise ValueError('Each prompt group requires two distinct seeds.')
    if all('prompt' in r for r in plan):
        prompts=[{r['prompt'] for r in plan if str(r['prompt_group'])==g} for g in groups]
        if any(len(p)!=1 for p in prompts) or len(set.union(*prompts))!=3:
            raise ValueError('Use one prompt per group and three distinct prompts.')
    return groups


def screen_pixels(edge, observed_edges, plan):
    """Return retained source pixels, support/drift maps; no source curve tracing."""
    edge=validate_edge(edge);groups=validate_plan(plan)
    if len(observed_edges)!=len(plan):raise ValueError('Observation count differs from plan.')
    yy,xx=np.nonzero(edge);source=np.column_stack([xx,yy])
    n=len(source);dist=np.full((6,n),np.inf);delta=np.full((6,n,2),np.nan)
    for i,out in enumerate(observed_edges):
        if out.shape!=edge.shape:raise ValueError('Dimensions differ; register probes explicitly before screening.')
        ty,tx=np.nonzero(out);target=np.column_stack([tx,ty])
        if len(target) and n:
            dist[i],idx=cKDTree(target).query(source)
            delta[i]=target[idx]-source
    hits=dist<=POLICY['match_radius_px'];count=hits.sum(0)
    median=np.full(n,np.inf);dispersion=np.full(n,np.inf)
    # Avoid NaN reductions on pixels with no observations.
    for j in np.flatnonzero(count>=POLICY['min_views']):
        valid=hits[:,j];offset=delta[valid,j]
        median[j]=np.median(dist[valid,j])
        dispersion[j]=np.median(np.linalg.norm(offset-np.median(offset,axis=0),axis=1))
    keep=count>=POLICY['min_views']
    for g in groups:
        keep&=hits[[str(r['prompt_group'])==g for r in plan]].sum(0)>=POLICY['min_per_prompt']
    keep&=dispersion<=POLICY['max_displacement_dispersion_px']
    stable=np.zeros_like(edge);stable[yy[keep],xx[keep]]=255
    support=np.zeros_like(edge);support[yy,xx]=count
    drift=np.full(edge.shape,np.inf,np.float32);drift[yy,xx]=dispersion
    return stable,support,drift


def probe(canny_path,plan_path,out_path,roi=None):
    canny_path=Path(canny_path);plan_path=Path(plan_path).resolve();out_path=Path(out_path)
    edge=validate_edge(np.asarray(Image.open(canny_path).convert('L')))
    plan=json.loads(plan_path.read_text());groups=validate_plan(plan)
    if len({r['rgb'] for r in plan})!=6:raise ValueError('Provide six different RGB paths.')
    sha=hashlib.sha256(canny_path.read_bytes()).hexdigest();observed=[];records=[]
    for row in plan:
        if row.get('hint_sha256',sha)!=sha:raise ValueError('Probe hint hash differs from original Canny.')
        path=Path(row['rgb']);path=path if path.is_absolute() else plan_path.parent/path
        image=np.asarray(Image.open(path).convert('RGB'))
        observed.append(cv2.Canny(image,100,200))
        records.append(dict(row,rgb_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    stable,support,drift=screen_pixels(edge,observed,plan)
    if roi is not None:
        x0,y0,x1,y1=map(int,roi)
        if not (0<=x0<x1<=edge.shape[1] and 0<=y0<y1<=edge.shape[0]):
            raise ValueError('ROI must be inside the original Canny.')
        valid=np.zeros_like(stable,bool);valid[y0:y1,x0:x1]=True
        stable[~valid]=0
    # This is the FIRST source curve extraction in this workflow.
    curves=matcher.source_curves(stable,THRESHOLDS)[0]
    targets=[matcher.skeleton_graph(e) for e in observed];rows=[]
    for c in curves:
        matches=[matcher.curve_match(c['samples'],t,THRESHOLDS) for t in targets]
        successful=[r for r in matches if r['matched']]
        bygroup={g:sum(m['matched'] for m,r in zip(matches,plan) if str(r['prompt_group'])==g) for g in groups}
        anchors=sum(r.get('anchor_good_fraction',0)>=.8 for r in matches)
        median=float(np.median([r['symmetric_median_px'] for r in successful])) if successful else None
        eligible=len(successful)>=5 and min(bygroup.values())>=1 and anchors>=5 and median<=3
        rows.append(dict(**c,stable_eligible=bool(eligible),probe_matched=len(successful),
                         per_prompt_matched=bygroup,anchor_views=anchors,median_distance=median))
    rows.sort(key=lambda r:(-r['arc_length_px'],r['curve_index']))
    out_path.parent.mkdir(parents=True,exist_ok=True)
    mask_path=out_path.with_name(out_path.stem+'_stable_edges.png')
    maps_path=out_path.with_name(out_path.stem+'_maps.npz')
    Image.fromarray(stable).save(mask_path)
    np.savez_compressed(maps_path,support_count=support,probability=support.astype('float32')/6,
                        displacement_dispersion_px=drift,source_edge=edge>0)
    result=dict(canny_sha256=sha,policy=POLICY,
        order=['original_Canny','clean_prompt_seed_probes','pixel_stability_screen',
               'ordered_curve_extraction','whole_curve_verification','input_writability_check_pending'],
        stable_edges=mask_path.name,maps=maps_path.name,roi=roi,source_pixels=int((edge>0).sum()),
        stable_pixels=int((stable>0).sum()),probe_plan=records,
        eligible=sum(r['stable_eligible'] for r in rows),candidates=rows,
        limitation='Proximity proxy plus graph verification; no semantic guarantee or validated RGB message accuracy.')
    out_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result
