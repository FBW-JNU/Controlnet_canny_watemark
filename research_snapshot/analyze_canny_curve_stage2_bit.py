"""Preregistered Stage2 scoring. Never changes the frozen Stage1 matcher."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image
from scipy.stats import rankdata
from canny_curve_stage2_bit import ROOT, OUT, MODES, read, dump, sha, save, matcher, coefficient, verify

def auc(y,s):
    y=np.asarray(y,int);s=np.asarray(s,float);n1=int(y.sum());n0=len(y)-n1
    if not n1 or not n0:return None
    return float((rankdata(s)[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def interval(values):
    v=[x for x in values if x is not None and np.isfinite(x)]
    return [float(x) for x in np.percentile(v,[2.5,97.5])] if v else [None,None]

def distribution(values):
    v=[float(x) for x in values if x is not None and np.isfinite(x)]
    return dict(n=len(v),mean=float(np.mean(v)) if v else None,median=float(np.median(v)) if v else None,
                p05=float(np.percentile(v,5)) if v else None,p95=float(np.percentile(v,95)) if v else None,
                min=min(v) if v else None,max=max(v) if v else None)

def input_code_score(c,mode,value):
    f=c['input_features'][mode]
    return (value-f['midpoint'])/f['half_separation']

def presence_score(c,mode,value):
    f=c['input_features'][mode]
    return abs((value-f['clean'])/f['half_separation'])

def labels_scores(rows,mode,cases):
    out=[]
    for r in rows:
        if r['arm']=='clean' or r['mode']!=mode:continue
        c=cases[r['key']];valid=r['match']['matched'];value=r['coefficients'][mode] if valid else None
        out.append(dict(key=r['key'],content_id=r['content_id'],view=r['view'],stratum=r['stratum'],
            label=int(r['arm']=='bit1'),valid=valid,score=input_code_score(c,mode,value) if valid else 0.,
            presence=presence_score(c,mode,value) if valid else 0.,quality_pass=r['quality']['quality_pass']))
    return out

def bit_metrics(data,bootstrap=True):
    y=[r['label'] for r in data];s=[r['score'] for r in data];valid=[r for r in data if r['valid']]
    low=[r['score'] if r['valid'] else -1e9 if r['label']==1 else 1e9 for r in data]
    high=[r['score'] if r['valid'] else 1e9 if r['label']==1 else -1e9 for r in data]
    result=dict(attempts=len(data),matched=len(valid),coverage=len(valid)/len(data) if data else None,
        auc_neutral_itt=auc(y,s),auc_success_only=auc([r['label'] for r in valid],[r['score'] for r in valid]),
        auc_missing_bounds=[auc(y,low),auc(y,high)],
        fixed_threshold_accuracy_itt=float(np.mean([(r['score']>0)==bool(r['label']) for r in data])) if data else None,
        strict_correct_fraction=float(np.mean([r['valid'] and ((r['score']>0)==bool(r['label'])) for r in data])) if data else None,
        quality_passes=sum(r['quality_pass'] for r in data),quality_pass_fraction=float(np.mean([r['quality_pass'] for r in data])) if data else None)
    if bootstrap and data:
        byid={cid:[r for r in data if r['content_id']==cid] for cid in sorted({r['content_id'] for r in data})}
        rng=np.random.default_rng(2026111203);ids=list(byid);values=[]
        for _ in range(2000):
            sample=[r for cid in rng.choice(ids,len(ids),replace=True) for r in byid[int(cid)]]
            values.append(auc([r['label'] for r in sample],[r['score'] for r in sample]))
        result['auc_cluster_bootstrap95']=interval(values)
    return result

def clean_data(rows,mode,cases):
    out=[]
    for r in rows:
        if r['arm']!='clean':continue
        valid=r['match']['matched'];c=cases[r['key']]
        value=r['coefficients'][mode] if valid else None
        out.append(dict(key=r['key'],content_id=r['content_id'],view=r['view'],valid=valid,
                        presence=presence_score(c,mode,value) if valid else 0.))
    return out

def presence_metrics(clean,marked,threshold):
    result=dict(clean_attempts=len(clean),clean_matched=sum(r['valid'] for r in clean),marked_attempts=len(marked),
        marked_matched=sum(r['valid'] for r in marked),cal_threshold=threshold)
    for label,name in [(0,'bit0'),(1,'bit1'),(None,'combined')]:
        marks=[r for r in marked if label is None or r['label']==label]
        y=[0]*len(clean)+[1]*len(marks);s=[r['presence'] for r in clean+marks]
        subset=[(0,r) for r in clean if r['valid']]+[(1,r) for r in marks if r['valid']]
        result[name]=dict(auc_neutral_itt=auc(y,s),auc_success_only=auc([z[0] for z in subset],[z[1]['presence'] for z in subset]),
            fpr_itt=float(np.mean([r['valid'] and r['presence']>threshold for r in clean])) if clean else None,
            tpr_itt=float(np.mean([r['valid'] and r['presence']>threshold for r in marks])) if marks else None,
            fpr_success_only=float(np.mean([r['presence']>threshold for r in clean if r['valid']])) if any(r['valid'] for r in clean) else None,
            tpr_success_only=float(np.mean([r['presence']>threshold for r in marks if r['valid']])) if any(r['valid'] for r in marks) else None)
    return result

def make_pairs(rows,mode,cases):
    out=[];lookup={(r['key'],r['view'],r['mode'],r['arm']):r for r in rows}
    for c in cases.values():
        for view in range(2):
            r0=lookup.get((c['key'],view,mode,'bit0'));r1=lookup.get((c['key'],view,mode,'bit1'));rc=lookup.get((c['key'],view,'shared','clean'))
            if not (r0 and r1 and rc):continue
            valid=r0['match']['matched'] and r1['match']['matched'];dc=None
            if valid:dc=r1['coefficients'][mode]-r0['coefficients'][mode]
            din=c['input_features'][mode]['delta_in']
            cleanvalid=rc['match']['matched']
            out.append(dict(key=c['key'],content_id=c['content_id'],stratum=c['stratum'],split=c['split'],view=view,mode=mode,
                both_matched=valid,clean_matched=cleanvalid,delta_in=din,delta_out=dc,gain=dc/din if valid else None,
                sign_consistent=bool(valid and dc>0),both_quality_pass=bool(r0['quality']['quality_pass'] and r1['quality']['quality_pass']),
                delta_bit0_vs_clean=r0['coefficients'][mode]-rc['coefficients'][mode] if r0['match']['matched'] and cleanvalid else None,
                delta_bit1_vs_clean=r1['coefficients'][mode]-rc['coefficients'][mode] if r1['match']['matched'] and cleanvalid else None))
    return out

def pair_metrics(pairs):
    valid=[p for p in pairs if p['both_matched']];ids=sorted({p['content_id'] for p in pairs})
    repeat=[cid for cid in ids if len([p for p in pairs if p['content_id']==cid])==2 and all(p['sign_consistent'] for p in pairs if p['content_id']==cid)]
    result=dict(pairs=len(pairs),both_matched=len(valid),positive_sign_pairs=sum(p['sign_consistent'] for p in pairs),
        positive_fraction_all_pairs=sum(p['sign_consistent'] for p in pairs)/len(pairs) if pairs else None,
        positive_fraction_matched=sum(p['sign_consistent'] for p in pairs)/len(valid) if valid else None,
        repeatable_contents=repeat,repeatable_count=len(repeat),total_contents=len(ids),
        delta_in=distribution(p['delta_in'] for p in pairs),delta_out=distribution(p['delta_out'] for p in pairs),gain=distribution(p['gain'] for p in pairs))
    if len(valid)>2:
        x=np.array([p['delta_in'] for p in valid]);y=np.array([p['delta_out'] for p in valid])
        result['pearson_delta_in_vs_out']=float(np.corrcoef(x,y)[0,1]) if x.std()>1e-9 and y.std()>1e-9 else None
        result['slope_through_origin']=float(np.dot(x,y)/np.dot(x,x))
        # Matched-only geometry summaries, with missing coverage reported separately.
        rng=np.random.default_rng(2026111204);boot=[];byid={cid:[p for p in pairs if p['content_id']==cid] for cid in ids}
        for _ in range(2000):
            sample=[p for cid in rng.choice(ids,len(ids),replace=True) for p in byid[int(cid)] if p['both_matched']]
            if sample:boot.append(float(np.mean([p['delta_out'] for p in sample])))
        result['mean_delta_out_cluster_bootstrap95']=interval(boot)
    return result

def quality_metrics(rows,mode):
    q=[r['quality'] for r in rows if r['mode']==mode]
    return dict(attempts=len(q),joint_passes=sum(r['quality_pass'] for r in q),joint_pass_fraction=float(np.mean([r['quality_pass'] for r in q])) if q else None,
        zero_change_outputs=sum(not r['active_output'] for r in q),
        metrics={k:distribution(r[k] for r in q) for k in ['psnr_db','ssim','lpips_alex','edge_f1_3x3']},
        gate_pass_counts={k:sum(r['gates'][k] for r in q) for k in ['psnr','ssim','lpips','edge_f1']})

def calibrate(rows,cases):
    assert all(r['split']=='Cal' for r in rows)
    assert len(rows)==168
    result={};eligible=[]
    for mode in MODES:
        data=labels_scores(rows,mode,cases);stats=bit_metrics(data);clean=clean_data(rows,mode,cases)
        valid=[r['presence'] for r in clean if r['valid']]
        threshold=float(np.percentile(valid,95)) if valid else 1e9
        qualified=stats['coverage']>=.8 and stats['quality_passes']==stats['attempts']
        result[mode]=dict(bit=stats,presence_threshold95=threshold,presence=presence_metrics(clean,data,threshold),qualified=qualified)
        if qualified:eligible.append(mode)
    pool=eligible or MODES
    selected=max(pool,key=lambda mo:result[mo]['bit']['auc_neutral_itt'])
    decision=dict(created_utc=datetime.now(timezone.utc).isoformat(),calibration_only=True,cal_contents=sorted({r['content_id'] for r in rows}),
        selected_mode=selected,qualified_modes=eligible,has_qualified_mode=bool(eligible),modes=result,
        protocol_sha256=sha(OUT/'protocol.json'),analysis_freeze_sha256=sha(OUT/'analysis_freeze.json'),
        cal_metric_sha256={r['job']:sha(OUT/'metrics'/f'{r["job"]}.json') for r in rows},
        policy='Primary mode frozen before any Dev metrics are computed; fixed input polarity; no retuning on Dev.')
    dump(OUT/'calibration_decision.json',decision)
    print('CAL FROZEN',selected,'quality-qualified',eligible,flush=True)
    return decision

def freeze_analysis():
    path=OUT/'analysis_freeze.json'
    if path.exists():
        for file,digest in read(path)['files'].items():assert sha(file)==digest,file
        return
    assert not list((OUT/'metrics').glob('*.json')),'Freeze before first scoring'
    shutil.copyfile(__file__,OUT/'source'/Path(__file__).name)
    files={str(p):sha(p) for p in [Path(__file__).resolve(),OUT/'source'/Path(__file__).name,ROOT/'canny_curve_stage2_bit.py',OUT/'protocol.json']}
    dump(path,dict(created_utc=datetime.now(timezone.utc).isoformat(),files=files,tests=tests()))

def tests():
    assert auc([0,0,1,1],[0,1,2,3])==1
    assert auc([0,0,1,1],[3,2,1,0])==0
    assert auc([0,1],[0,0])==.5
    data=[dict(label=y,score=s,valid=v,quality_pass=True,content_id=i//2) for i,(y,s,v) in enumerate([(0,-1,True),(1,1,True),(0,0,False),(1,0,False)])]
    a=bit_metrics(data,False);assert a['auc_missing_bounds'][0]<=a['auc_neutral_itt']<=a['auc_missing_bounds'][1]
    assert a['coverage']==.5 and a['auc_success_only']==1
    c=dict(input_features={mo:dict(clean=.2,midpoint=.1,half_separation=1.5) for mo in MODES})
    assert input_code_score(c,'normal',1.6)==1
    assert presence_score(c,'normal',.2)==0
    # Canonical coefficient and sign under reversed source storage.
    m=matcher();p=np.c_[np.linspace(100,200,101),np.full(101,100.)]
    from canny_curve_stage2_bit import fields
    for mode in MODES:
        q,u,n,b=fields(p,mode,m);path=q+1.5*b[:,None]*n
        v=coefficient(p,path,mode,m);r=coefficient(p[::-1],path,mode,m)
        assert abs(v-r)<1e-10 and v>1.3,(mode,v,r)
    return dict(status='passed',checks=['AUC polarity and ties','missing-score conservative bounds','normalization','presence zero','canonical orientation reversal','three synthetic modulation responses'])

def watch():
    import torch
    import lpips
    from carrier_screen.common import pixel_metrics
    p=verify();freeze_analysis();torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    net=lpips.LPIPS(net='alex',version='0.1').eval().requires_grad_(False).cpu()
    m=matcher();cases={c['key']:c for c in read(OUT/'selection.json')};plan=read(OUT/'plan.json');lookup={(j['key'],j['view'],j['arm'],j['mode']):j for j in plan}
    while True:
        calibration=(OUT/'calibration_decision.json').exists();progress=False
        for j in plan:
            target=OUT/'metrics'/f'{j["job"]}.json'
            if target.exists():continue
            if j['split']=='Dev' and not calibration:continue
            rec=OUT/'records'/f'{j["job"]}.json'
            if not rec.exists():continue
            r=read(rec);assert sha(j['rgb'])==r['rgb_sha256'];assert sha(j['hint'])==r['hint_sha256']
            c=cases[j['key']];rgb=np.asarray(Image.open(j['rgb']).convert('RGB')).copy()
            edge=cv2.Canny(rgb,100,200);graph=m.skeleton_graph(edge)
            match=m.curve_match(c['samples'],graph,p['matcher_thresholds'],return_path=True)
            coefficients={mo:coefficient(c['samples'],match['output_path'],mo,m) if match['matched'] else None for mo in MODES}
            q=None
            if j['arm']!='clean':
                cleanjob=lookup[(j['key'],j['view'],'clean','shared')]
                assert (OUT/'records'/f'{cleanjob["job"]}.json').exists()
                clean=np.asarray(Image.open(cleanjob['rgb']).convert('RGB')).copy()
                q=pixel_metrics(clean,rgb)
                aa=torch.from_numpy(clean).permute(2,0,1)[None].float()/127.5-1
                bb=torch.from_numpy(rgb).permute(2,0,1)[None].float()/127.5-1
                with torch.inference_mode():q['lpips_alex']=float(net(aa,bb).item())
                g=p['quality_gate'];gates=dict(psnr=bool(q['psnr_infinite'] or q['psnr_db']>=g['psnr_db_min']),
                    ssim=q['ssim']>=g['ssim_min'],lpips=q['lpips_alex']<=g['lpips_alex_max'],edge_f1=q['edge_f1_3x3']>=g['rgb_pair_edge_f1_3x3_min'])
                q.update(gates=gates,quality_pass=all(gates.values()),active_output=q['changed_pixels']>0)
            edgepath=OUT/'output_canny'/f'{j["job"]}.png';save(edgepath,edge)
            dump(target,dict(**j,rgb_sha256=r['rgb_sha256'],record_sha256=sha(rec),match=match,coefficients=coefficients,
                quality=q,output_canny=str(edgepath),output_canny_sha256=sha(edgepath),analysis_freeze_sha256=sha(OUT/'analysis_freeze.json')))
            progress=True
            print('SCORED',j['job'],'matched',match['matched'],'quality',q['quality_pass'] if q else 'reference',flush=True)
        calpaths=[OUT/'metrics'/f'{j["job"]}.json' for j in plan if j['split']=='Cal']
        if not calibration and all(f.exists() for f in calpaths):calibrate([read(f) for f in calpaths],cases);progress=True
        n=len(list((OUT/'metrics').glob('*.json')))
        dump(OUT/'analysis_status.json',dict(completed=n,total=len(plan),calibration_frozen=(OUT/'calibration_decision.json').exists()))
        if n==len(plan):break
        if not progress:time.sleep(5)
    summarize()

def csvout(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def summarize():
    freeze_analysis();p=verify();plan=read(OUT/'plan.json');rows=[read(OUT/'metrics'/f'{j["job"]}.json') for j in plan]
    decision=read(OUT/'calibration_decision.json');assert decision['protocol_sha256']==sha(OUT/'protocol.json')
    for job,digest in decision['cal_metric_sha256'].items():assert sha(OUT/'metrics'/f'{job}.json')==digest
    cases={c['key']:c for c in read(OUT/'selection.json')};summary={};pairs=[];percontent=[];scores=[]
    for split in ['Cal','Dev']:
        subset=[r for r in rows if r['split']==split];summary[split]={}
        for mode in MODES:
            data=labels_scores(subset,mode,cases);clean=clean_data(subset,mode,cases);pp=make_pairs(subset,mode,cases);pairs+=pp
            threshold=decision['modes'][mode]['presence_threshold95']
            summary[split][mode]=dict(bit=bit_metrics(data),pairs=pair_metrics(pp),quality=quality_metrics(subset,mode),
                presence=presence_metrics(clean,data,threshold),clean_match_rate=sum(r['valid'] for r in clean)/len(clean),
                strata={kind:dict(bit=bit_metrics([r for r in data if r['stratum']==kind]),pairs=pair_metrics([r for r in pp if r['stratum']==kind])) for kind in ['straight','mild','bent']})
            for cid in sorted({r['content_id'] for r in data}):
                d=[r for r in data if r['content_id']==cid];cp=[r for r in pp if r['content_id']==cid]
                percontent.append(dict(split=split,mode=mode,content_id=cid,stratum=d[0]['stratum'],**bit_metrics(d,False),
                    repeatable_both_views=len(cp)==2 and all(r['sign_consistent'] for r in cp),mean_delta_out=distribution(r['delta_out'] for r in cp)['mean']))
            for r in data:scores.append(dict(split=split,mode=mode,**r))
    mode=decision['selected_mode'];dev=summary['Dev'][mode]
    gates=dict(cal_has_qualified_mode=decision['has_qualified_mode'],dev_bit_auc_ge_80=dev['bit']['auc_neutral_itt']>=.8,
        dev_marked_match_ge_80=dev['bit']['coverage']>=.8,dev_all_quality_pass=dev['quality']['joint_passes']==dev['quality']['attempts'],
        dev_repeatable_at_least3=dev['pairs']['repeatable_count']>=3)
    summary.update(selected_mode=mode,acceptance_gates=gates,accepted=all(gates.values()),
        interpretation='AUC uses fixed input polarity and neutral0 on abstentions. Success-only AUC is conditional. All quality images retained. Shared clean counted once per mode; do not pool repeated clean across modes.',
        counts=dict(contents=24,Cal_contents=12,Dev_contents=12,curves=24,unique_rgb=336,clean_rgb=48,marked_rgb=288,pairs=144),
        protocol_sha256=sha(OUT/'protocol.json'),calibration_sha256=sha(OUT/'calibration_decision.json'))
    dump(OUT/'summary.json',summary);dump(OUT/'pair_responses.json',pairs);dump(OUT/'per_content.json',percontent)
    csvout(OUT/'pair_responses.csv',pairs);csvout(OUT/'per_content.csv',percontent);csvout(OUT/'scores.csv',scores)
    qualityrows=[dict(job=r['job'],content_id=r['content_id'],split=r['split'],mode=r['mode'],arm=r['arm'],view=r['view'],
        **{k:v for k,v in r['quality'].items() if k!='gates'},**{f'pass_{k}':v for k,v in r['quality']['gates'].items()}) for r in rows if r['quality']]
    csvout(OUT/'quality.csv',qualityrows)
    print('SUMMARY',mode,gates,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','watch','summarize','tests']);args=a.parse_args()
    if args.action=='watch':watch()
    elif args.action=='summarize':summarize()
    elif args.action=='freeze':freeze_analysis()
    else:print(tests())
