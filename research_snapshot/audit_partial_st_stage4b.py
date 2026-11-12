"""Denominator, pixel provenance, mask, freeze and classifier audit."""
from curve_partial_st_stage4b import *
from urllib.parse import unquote
import re

def run():
    cv2.setNumThreads(1)
    f=check();sf=read(OUT/'score_freeze.json');inf=read(OUT/'input_freeze.json')
    for name,h in read(PARENT/'final_artifacts_sha256.json').items():assert digest(PARENT/name)==h,(name,'parent changed')
    for p,h in inf['protected_sha256'].items():assert digest(p)==h,p
    for field in ['cal_data_sha256','cal_controls_sha256']:
        for p,h in sf[field].items():assert digest(p)==h,p
    assert sf['protocol_sha256']==digest(OUT/'protocol.json');assert sf['parent_score_sha256']==digest(PARENT/'score_freeze.json')
    assert sf['calibration_amendment_sha256']==digest(OUT/'calibration_amendment.json')
    assert sf['amended_grid_sha256']==digest(OUT/'cal_gate_grid_amended.json')
    amendment=read(OUT/'calibration_amendment.json');assert amendment['original_grid_sha256']==digest(OUT/'cal_gate_grid.json')
    assert amendment['source_sha256']==digest(ROOT/'calibrate_partial_st_stage4b.py')
    plan=read(OUT/'plan.json');selected=read(OUT/'selection.json');cc={c['key']:c for folder in [PARENT,OUT] for c in read(folder/'selection.json')}
    assert len(selected)==len({c['content_id'] for c in selected})==160
    assert sum(c['split']=='Cal' for c in selected)==32 and sum(c['split']=='Fresh' for c in selected)==128
    assert sum(c['marked'] for c in selected)==12
    excluded=set(read(OUT/'exclusions.json')['ids']);assert not excluded&{c['content_id'] for c in selected}
    assert len(plan)==len({j['job'] for j in plan})==len({j['rgb'] for j in plan})==208
    assert len(list((OUT/'records').glob('*.json')))==208
    sources=0
    for c in selected:
        assert digest(c['image_path'])==c['source_sha256'];sources+=1
        if not c['samples']:continue
        base=a4.arr(c['states']['clean']['hint']);mask=a4.arr(c['mask'])>0
        for arm,state in c['states'].items():
            image=a4.arr(state['hint']);assert digest(state['hint'])==state['hint_sha256'];assert set(np.unique(image))<={0,255}
            assert np.array_equal(image,a4.render(base,c['samples'],mask,a4.SIGNS[arm]));assert np.array_equal(image[~mask],base[~mask])
    groups={};noises={}
    for j in plan:
        record=read(OUT/'records'/f'{j["job"]}.json');assert record['rgb_sha256']==digest(j['rgb']);assert record['hint_sha256']==digest(j['hint'])
        assert record['protocol_sha256']==digest(OUT/'protocol.json');assert record['checkpoint_sha256']==f['generation']['checkpoint_sha256']
        assert record['prompt']==j['prompt'] and record['seed']==j['seed'];assert record['noise_sha256']==digest(j['noise'])
        if j['split']=='Fresh':assert record['score_freeze_sha256']==digest(OUT/'score_freeze.json')
        groups.setdefault(j['key'],[]).append(j);noises[j['noise']]=j['seed']
    assert len(groups)==len(noises)==160
    for key,jobs in groups.items():assert len(jobs)==(5 if cc[key]['marked'] else 1) and len({j['noise'] for j in jobs})==1
    import torch
    for path,seed in noises.items():assert np.array_equal(np.load(path),torch.randn((1,4,64,64),generator=torch.Generator(device='cpu').manual_seed(seed)).numpy())
    data=all_data(['OldCal','OldEval','Cal','Fresh']);assert len(data)==388
    assert {s:sum(d['split']==s for d in data) for s in ['OldCal','OldEval','Cal','Fresh']}==dict(OldCal=60,OldEval=120,Cal=32,Fresh=176)
    pixels=0;partial_checks=0;edge_checks=0
    plans={j['job']:j for folder in [PARENT,OUT] for j in read(folder/'plan.json')}
    for d in data:
        j=plans[d['job']];assert digest(j['rgb'])==d['rgb_sha256'];rgb=np.array(Image.open(j['rgb']).convert('RGB'));edge=cv2.Canny(rgb,100,200);c=cc[d['key']]
        if not c['samples']:continue
        p,_,_,_=geometry(c['samples']);old=d['original'];original_points=set(map(tuple,old.get('observed_xy',[])))
        for name,tracks in d['scenarios'].items():
            scenario=next(s for s in SCENARIOS if s['name']==name)
            for track,raw in tracks.items():
                for obs in (raw.get('pool',[]) if track=='EdgeDrop' else [raw]):
                    if not obs.get('fitted'):continue
                    xy=np.array(obs['observed_xy']);assert np.array_equal(xy,np.rint(xy));xy=xy.astype(int)
                    assert np.all(edge[xy[:,1],xy[:,0]]>0);pixels+=len(xy)
                    if track=='ObsDrop':
                        assert set(map(tuple,xy))<=original_points;assert mask_keep(c['samples'],j,scenario,obs['indices']).all();partial_checks+=1
                    else:
                        dist,idx=cKDTree(p).query(xy);assert np.all((dist>8)|mask_keep(c['samples'],j,scenario,idx));edge_checks+=1
                    assert len(set(map(tuple,xy)))==len(xy),'Repeated real pixel evidence'
                    assert len(set(obs['indices']))==len(obs['indices'])
    rows=read(OUT/'decoded.json');summary=read(OUT/'summary.json');assert len(rows)==388*22
    assert len({(r['job'],r['scenario'],r['track']) for r in rows})==len(rows)
    for split in summary:
        for name,tracks in summary[split].items():
            for track,value in tracks.items():
                rr=[r for r in rows if r['split']==split and r['scenario']==name and r['track']==track]
                expected=summarize(rr);assert value==expected,(split,name,track)
                assert value['correct']+value['wrong']+value['erasures']==value['attempts']
                if split=='Fresh':assert value['attempts']==48 and value['clean_attempts']==128
    # Recompute the four-state-set invariance rather than trusting stored equal fields.
    invariant=0
    for r in rows:
        if not r['accepted']:assert r['word'] is None and not r['presence_accept'];continue
        c=cc[r['key']];model=sf['channel_model'];centers=np.array([np.array(model['A'])@a4.features(c,word,'Auto')+model['intercept'] for word in WORDS]);y=np.array(r['feature'])
        cov=np.array(model['residual_covariance'])+np.array(r['geometry']['covariance']);ll=a4.gaussian_log(y-centers,cov);wrong_ll=a4.gaussian_log(y-centers[np.array([3,2,1,0])],cov)
        assert abs(logsumexp(ll)-logsumexp(wrong_ll))<1e-10;assert abs(float(ll.max())-float(wrong_ll.max()))<1e-10
        assert WORDS[int(ll.argmax())]==r['word'];invariant+=1
    quality=read(OUT/'quality.json');assert len(quality['records'])==48
    assert all(r['rgb_sha256']==digest(next(j['rgb'] for j in plan if j['job']==r['job'])) for r in quality['records'])
    controls=read(OUT/'wrong_anchor.json');assert len(controls['records'])==320
    clean=read(OUT/'clean_independent.json');rr=[r for r in rows if r['split']=='Fresh' and r['arm']=='clean' and r['scenario']=='none' and r['track']=='ObsDrop']
    assert clean['contents']==len(rr)==128 and clean['false_accepts']==sum(r['presence_accept'] for r in rr)
    for control in read(OUT/'algebraic_positive_control.json')['records']:
        assert control['rank']==3 and control['max_coefficient_error']<1e-8
    for href in re.findall(r'(?:href|src)="([^"]+)"',(OUT/'index.html').read_text()):
        if href.startswith(('http:','https:','#')):continue
        if href=='audit.json':continue  # Produced by this audit below.
        assert (OUT/unquote(href)).exists(),href
    manifest_sources={}
    for n in ['diagnose_partial_st_stage4b.py','report_partial_st_stage4b.py','audit_partial_st_stage4b.py','calibrate_partial_st_stage4b.py']:
        shutil.copyfile(ROOT/n,OUT/'source'/n);manifest_sources[n]=digest(ROOT/n)
    dump(OUT/'analysis_source_sha256.json',manifest_sources)
    for p in Path('/tmp').glob('stage4b_*.log'):shutil.copyfile(p,OUT/'logs'/p.name)
    dump(OUT/'audit.json',dict(status='passed',audited_utc=now(),new_contents=160,new_rgb=208,new_independent_clean_contents=128,new_marked_contents=12,
        protected_parent_files=len(read(PARENT/'final_artifacts_sha256.json')),loss_records=len(data),decoded_rows=len(rows),
        verified_real_observed_pixels=pixels,observation_subset_checks=partial_checks,edge_deletion_checks=edge_checks,
        wrong_key_invariance_checks=invariant,wrong_anchor_controls=320,quality_pairs=48,source_files=sources,
        checks=['parent and source hashes unchanged','independent content exclusions','input raster/exterior/maximum displacement unchanged',
                'shared paired noise','fresh generation binds Cal freeze','message-blind masks','all fitted pixels real and retained',
                'no duplicated real pixel evidence','all-attempt denominators','all four codewords and clean retained','wrong-Key set invariance independently recomputed',
                'noiseless full-rank control','viewer links valid']))
    manifest={str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='final_artifacts_sha256.json'}
    dump(OUT/'final_artifacts_sha256.json',manifest)
    for name,h in manifest.items():assert digest(OUT/name)==h,name
    print('AUDIT_PASSED',len(manifest),'files;',pixels,'real pixel checks',flush=True)

if __name__=='__main__':run()
