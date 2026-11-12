"""Documented Cal-only tightening after original negative-control grid failed.

Original protocol, source and failed grid are preserved. No Fresh RGB exists
when this amendment is recorded or when the final score is selected.
"""
from curve_partial_st_stage4b import *

def run():
    check()
    if (OUT/'score_freeze.json').exists():return
    assert not any(read(p)['split']=='Fresh' for p in (OUT/'records').glob('*.json'))
    original=read(OUT/'cal_gate_grid.json');assert not any(r['eligible'] for r in original)
    path=OUT/'calibration_amendment.json'
    grid=dict(info_min=[.1,.2,.3,.4,.5,.65,.8,.9,1.01],condition_max=[10.,20.,50.],ci95_max=[.35,.5,.75])
    if not path.exists():
        dump(path,dict(created_utc=now(),original_grid_sha256=digest(OUT/'cal_gate_grid.json'),
             reason='Original27 settings all failed fixed <=10% Cal wrong-anchor acceptance; minimum10/64. Tighten only on Cal before Fresh generation.',
             grid=grid,impossible_info_1_01='Explicit reject-all fallback if no useful setting passes; no relaxing false-control target.',
             unchanged=['encoder','generation','geometry proposals','bit-blind gates','Cal objective correct-2wrong','negative acceptance<=10%'],
             source_sha256=digest(ROOT/'calibrate_partial_st_stage4b.py'),no_fresh_rgb=True))
    cc={c['key']:c for folder in [PARENT,OUT] for c in read(folder/'selection.json')};model=channel_model();data=all_data(['OldCal','Cal'])
    negatives=[r['raw'] for p in (OUT/'controls').glob('*.json') for d in [read(p)] if d['split']=='Cal' for r in d['controls']]
    trials=[]
    for info,condition,ci in itertools.product(grid['info_min'],grid['condition_max'],grid['ci95_max']):
        gate=dict(info_min=info,condition_max=condition,ci95_max=ci);false=sum(select(r,gate)['accepted'] for r in negatives)
        if false>.1*len(negatives):
            trials.append(dict(gate=gate,negative_accepts=false,negative_attempts=len(negatives),eligible=False));continue
        correct=wrong=attempts=0
        for d in data:
            if d['split']!='OldCal' or d['arm']=='clean':continue
            for tracks in d['scenarios'].values():
                for raw in tracks.values():
                    out=decode(cc[d['key']],select(raw,gate),model);attempts+=1
                    correct+=bool(out['accepted'] and out['word']==d['arm']);wrong+=bool(out['accepted'] and out['word']!=d['arm'])
        trials.append(dict(gate=gate,correct=correct,wrong=wrong,attempts=attempts,negative_accepts=false,negative_attempts=len(negatives),eligible=True,objective=correct-2*wrong))
    selected=max([r for r in trials if r['eligible']],key=lambda r:(r['objective'],-r['wrong'],r['gate']['info_min'],-r['gate']['condition_max'],-r['gate']['ci95_max']))
    gate=selected['gate'];clean=[]
    for d in data:
        if d['arm']!='clean':continue
        v=decode(cc[d['key']],select(d['scenarios'].get('none',{}).get('ObsDrop',dict(fitted=False)),gate),model)
        if v['accepted']:clean.append(v['feature'])
    # Reject-all would prevent new presence fitting; preserve old clean model
    # and report no useful recovery, rather than selecting on Fresh.
    if len(clean)<3:
        parent=channel_model();mean=np.array(parent['intercept'])+parent['clean_residual_mean'];cov=parent['clean_covariance']
    else:mean=np.mean(clean,axis=0);cov=a4.covariance(clean).tolist()
    presence=dict(clean_mean=mean.tolist(),clean_covariance=cov,threshold95=0.,fit_received_clean=len(clean),fit_attempt_clean=44)
    scores=[]
    for d in data:
        if d['arm']!='clean':continue
        for tracks in d['scenarios'].values():
            for raw in tracks.values():
                v=decode(cc[d['key']],select(raw,gate),model,presence)
                if v['accepted']:scores.append(v['presence_score'])
    presence['threshold95']=float(np.percentile(scores,95)) if scores else 1e6;presence['cal_received_scores']=len(scores)
    dump(OUT/'cal_gate_grid_amended.json',trials)
    dump(OUT/'score_freeze.json',dict(created_utc=now(),protocol_sha256=digest(OUT/'protocol.json'),gate=gate,selected=selected,
        calibration_amendment_sha256=digest(path),amended_grid_sha256=digest(OUT/'cal_gate_grid_amended.json'),
        presence=presence,channel_model=model,parent_score_sha256=digest(PARENT/'score_freeze.json'),
        cal_data_sha256={str(OUT/'loss'/f'{d["job"]}.json'):digest(OUT/'loss'/f'{d["job"]}.json') for d in data},
        cal_controls_sha256={str(p):digest(p) for p in (OUT/'controls').glob('*.json') if read(p)['split']=='Cal'},no_fresh_rgb_before_freeze=True))
    print('SCORE_FROZEN',selected,presence,flush=True)

if __name__=='__main__':run()
