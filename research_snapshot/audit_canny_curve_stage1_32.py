"""Independent checks of data provenance, connected routes and mask selection."""
from __future__ import annotations
import argparse
import collections
import math
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist

from canny_curve_stage1_32 import ROOT,OUT,read,dump,sha
from analyze_canny_curve_stage1_32 import resample,tangent,skeleton_graph,curve_match

def independent_geometry(source,path,thresholds):
    """Brute-force pairwise distances; no KD-tree or graph route construction."""
    s=np.asarray(source,float);p=resample(path)
    matrix=cdist(s,p);near=matrix.argmin(1);ds=matrix.min(1);dt=matrix.min(0)
    st=tangent(s);pt=tangent(p)
    angles=np.rad2deg(np.arccos(np.clip(np.abs((st*pt[near]).sum(1)),0,1)))
    sym=max(float(np.median(ds)),float(np.median(dt)));p90=max(float(np.quantile(ds,.9)),float(np.quantile(dt,.9)))
    sl=float(np.hypot(*np.diff(s,axis=0).T).sum());pl=float(np.hypot(*np.diff(p,axis=0).T).sum())
    anchor_ids=np.rint(np.linspace(0,len(s)-1,5)).astype(int)
    good=(ds[anchor_ids]<=thresholds['anchor_tolerance_px'])&(angles[anchor_ids]<=thresholds['orientation_tolerance_deg'])
    endpoints=[float(np.linalg.norm(path[0]-s[0])),float(np.linalg.norm(path[-1]-s[-1]))]
    ok=sym<=3 and p90<=6 and float((ds<=4).mean())>=.8 and float((angles<=30).mean())>=.8 and .65<=pl/sl<=1.5 and float(good.mean())>=.8 and max(endpoints)<=4
    return dict(matched=bool(ok),symmetric_median_px=sym,symmetric_p90_px=p90,source_coverage=float((ds<=4).mean()),
        tangent_good_fraction=float((angles<=30).mean()),length_ratio=pl/sl,anchor_good_fraction=float(good.mean()))

def audit(partial=False):
    cv2.setNumThreads(1)
    protocol=read(OUT/'protocol.json');plan=read(OUT/'plan.json');cases=read(OUT/'selected.json');t=protocol['exploration_thresholds']
    assert len(cases)==32 and len({c['content_id'] for c in cases})==32
    assert len(plan)==384 and sha(OUT/'plan.json')==protocol['plan_sha256']
    assert sha(OUT/'selected.json')==protocol['selected_sha256']
    assert sha(protocol['population_manifest'])==protocol['population_manifest_sha256']
    assert sha(ROOT/protocol['generation']['checkpoint'])==protocol['generation']['checkpoint_sha256']
    for f,h in protocol['frozen_model_code'].items():assert sha(f)==h,f
    counts=collections.Counter(j['role'] for j in plan)
    assert counts==dict(discovery=192,heldout_seed=64,heldout_prompt=96,heldout_joint=32)
    assert sum(j['reused'] for j in plan)==60
    checked_images=0;checked_paths=0;checked_selected=0;checked_failed=0;complete_contents=0;checks=[]
    for case in cases:
        cid=case['content_id'];jobs=[j for j in plan if j['content_id']==cid]
        if not all((OUT/'metrics'/f'{j["job"]}_matches.json').exists() for j in jobs):
            if partial:continue
            raise RuntimeError(f'Not complete {cid}')
        complete_contents+=1;source_file=OUT/'curves'/f'{cid:012d}_source.json';source=read(source_file)
        selection=read(OUT/'curves'/f'{cid:012d}_selected.json');ids={r['curve_index'] for r in selection['selected']}
        assert selection['source_curves_sha256']==sha(source_file)
        discovery_jobs=[j for j in jobs if j['role']=='discovery']
        assert len(discovery_jobs)==6 and set(selection['discovery_jobs'])=={j['job'] for j in discovery_jobs}
        assert selection['selection_used_roles']==['discovery']
        for j in discovery_jobs:assert sha(OUT/'metrics'/f'{j["job"]}_matches.json')==selection['discovery_match_hashes'][j['job']]
        assert sha(case['hint'])==case['hint_sha256'] and sha(case['image_path'])==case['source_sha256']
        prepared=np.asarray(Image.open(case['prepared_source']).convert('RGB'));hint=np.asarray(Image.open(case['hint']))>0
        assert np.array_equal(cv2.Canny(prepared,100,200)>0,hint)
        discovery_rows=[read(OUT/'metrics'/f'{j["job"]}_matches.json')['matches'] for j in discovery_jobs]
        for c in source['curves']:
            i=c['curve_index'];rs=[a[i] for a in discovery_rows]
            rate=sum(r['matched'] for r in rs)/6;anchor=sum(r.get('anchor_good_fraction',0)>=.8 for r in rs)/6
            per_prompt=[sum(r['matched'] for r in rs if r['prompt_index']==pi)/3 for pi in (0,1)]
            selected=rate>=t['discovery_retention_min'] and anchor>=t['discovery_anchor_rate_min'] and min(per_prompt)>=t['per_discovery_prompt_retention_min']
            assert selected==(i in ids),(cid,i)
        union=np.zeros((512,512),bool)
        for i in ids:
            mask=np.asarray(Image.open(OUT/'masks'/f'{cid:012d}'/f'curve_{i:04d}.png'))>0;union|=mask
        saved=np.asarray(Image.open(OUT/'masks'/f'{cid:012d}_region.png'))>0
        assert np.array_equal(union,saved)
        with np.load(OUT/'masks'/f'{cid:012d}_confidence.npz') as data:
            assert np.array_equal(data['confidence']>0,saved)
            assert np.all(data['wilson95_lower']<=data['confidence'])
            assert np.isfinite(data['confidence']).all()
        for j in jobs:
            record=read(OUT/'records'/f'{j["job"]}.json');assert sha(j['rgb'])==record['rgb_sha256']
            assert sha(j['noise'])==record['noise_sha256']
            assert record['checkpoint_sha256']==protocol['generation']['checkpoint_sha256']
            assert record['device']=='physical GPU1' and record['precision']=='float32, TF32 disabled'
            if j['reused']:
                assert sha(j['source_record'])==j['source_record_sha256']
                old=read(j['source_record']);assert old['rgb_sha256']==record['rgb_sha256']
                assert old['prompt']==j['prompt'] and old['seed']==j['seed'] and old['hint_sha256']==j['hint_sha256']
            else:assert record['adapter'] is None and record['all_parameters_frozen']
            rgb=np.asarray(Image.open(j['rgb']).convert('RGB'));edge=cv2.Canny(rgb,100,200)>0
            x0,y0,x1,y1=case['roi'];valid=np.zeros_like(edge);valid[y0:y1,x0:x1]=True;edge&=valid
            assert np.array_equal(edge,np.asarray(Image.open(OUT/'output_canny'/f'{j["job"]}.png'))>0)
            cached=read(OUT/'metrics'/f'{j["job"]}_matches.json')
            assert cached['rgb_sha256']==record['rgb_sha256'] and cached['source_curves_sha256']==sha(source_file)
            graph=None
            for i in ids:
                row=cached['matches'][i]
                if not row['path_found']:
                    assert not row['matched'];checked_failed+=1;continue
                p=row.get('output_path')
                if p is None:
                    if graph is None:graph=skeleton_graph(edge)
                    p=curve_match(source['curves'][i]['samples'],graph,t,return_path=True)['output_path']
                path=np.asarray(p);pix=np.rint(path).astype(int)
                assert np.array_equal(pix,path)
                assert np.all(edge[pix[:,1],pix[:,0]])
                delta=np.abs(np.diff(pix,axis=0));assert np.all(delta.max(1)==1)
                assert len(np.unique(pix,axis=0))==len(pix)
                check=independent_geometry(source['curves'][i]['samples'],path,t)
                for k,v in check.items():
                    if k=='matched':assert v==row[k],(j['job'],i,k,v,row[k])
                    else:assert abs(v-row[k])<1e-9,(j['job'],i,k,v,row[k])
                checked_paths+=1
            checked_images+=1
        checked_selected+=len(ids)
        checks.append(dict(content_id=cid,source_candidates=len(source['curves']),selected=len(ids),discovery_only_selection_verified=True,masks_verified=True))
        print('AUDIT',cid,'selected',len(ids),'paths',checked_paths,flush=True)
    result=dict(status='passed' if complete_contents==32 else 'partial_passed',complete_contents=complete_contents,
        checked_images=checked_images,checked_selected_curves=checked_selected,checked_connected_paths=checked_paths,
        selected_view_no_route_failures=checked_failed,reuse_verified=60,
        image_hashes_and_noise_hashes=True,actual_Canny_reextracted_from_RGB=True,discovery_only_selection_recomputed=True,
        independent_geometry='scipy.spatial.distance.cdist brute-force vs stored KD-tree metrics; path vertices must be actual output edge pixels, 8-connected, without repeated vertices; source/target coverage, symmetric distances, tangent, length, anchors independently recounted.',
        no_heldout_rescreening=True,checks=checks)
    if not partial:
        summary=read(OUT/'summary.json')
        assert summary['contents']==32 and summary['discovery_selected_curves']==checked_selected
        for case in cases:assert summary['selection_hashes'][str(case['content_id'])]==sha(OUT/'curves'/f'{case["content_id"]:012d}_selected.json')
        snapshot={}
        for name in ('canny_curve_stage1_32.py','analyze_canny_curve_stage1_32.py','report_canny_curve_stage1_32.py','audit_canny_curve_stage1_32.py'):
            shutil.copyfile(ROOT/name,OUT/'source'/name);snapshot[name]=sha(ROOT/name)
        result['script_snapshots_sha256']=snapshot
        dump(OUT/'audit.json',result)
    else:dump(OUT/'partial_audit.json',result)
    print(result['status'],checked_images,'images',checked_paths,'paths',flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args();audit(a.partial)
