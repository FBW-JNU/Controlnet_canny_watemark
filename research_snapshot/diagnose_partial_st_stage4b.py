"""Read-only complementary diagnostics for the Stage4B frozen experiment."""
from curve_partial_st_stage4b import *

def run():
    check();f=read(OUT/'score_freeze.json');gate=f['gate'];rows=[]
    selected=read(PARENT/'selection.json')+[c for c in read(OUT/'selection.json') if c['marked']]
    for c in selected:
        if not c['admission']['admissible']:continue
        base=np.array(c['states']['clean']['measurements']['Auto']['coefficient'])
        ideal=a4.AMPLITUDE*np.array([a4.SIGNS[a] for a in WORDS])
        for arm in WORDS:
            old=c['states'][arm]['measurements']['Auto']
            for scenario in SCENARIOS:
                job=dict(key=c['key'],view=c.get('view',0));raw=partial(c['samples'],old,job,scenario);obs=select(raw,gate);word=None
                if obs['accepted']:
                    y=np.array(obs['coefficient'])-base;word=WORDS[int(np.argmin(np.sum((y-ideal)**2,axis=1)))]
                rows.append(dict(key=c['key'],split=c['split'],arm=arm,scenario=scenario['name'],accepted=obs['accepted'],word=word,
                    info_min=obs.get('info_min'),condition=obs.get('condition')))
    summary={}
    for split in ['Cal','Eval','Fresh']:
        summary[split]={}
        for s in SCENARIOS:
            rr=[r for r in rows if r['split']==split and r['scenario']==s['name']]
            summary[split][s['name']]=dict(attempts=len(rr),correct=sum(r['word']==r['arm'] for r in rr),wrong=sum(r['accepted'] and r['word']!=r['arm'] for r in rr),erasures=sum(not r['accepted'] for r in rr))
    dump(OUT/'input_partial_closure.json',dict(records=rows,summary=summary,
        note='Saved actual binary conditions, observed real edge coordinates; nominal four-sign decoder, not nearest to its own measured centroid. Random mask uses view0 for historical content. Conditional admitted carriers.'))
    # Frozen original coverage/uncertainty gating, same observation thinning.
    cc={c['key']:c for folder in [PARENT,OUT] for c in read(folder/'selection.json')};comparison=[]
    for d in all_data(['OldCal','Cal','OldEval','Fresh']):
        for scenario,tracks in d['scenarios'].items():
            raw=tracks['ObsDrop'];baseline=dict(raw,accepted=bool(raw.get('baseline_partial_gate',False)))
            baseline=decode(cc[d['key']],baseline,f['channel_model'],f['presence'])
            new=decode(cc[d['key']],select(raw,gate),f['channel_model'],f['presence'])
            comparison.append(dict(job=d['job'],split=d['split'],arm=d['arm'],scenario=scenario,
                baseline_received=baseline['accepted'],baseline_correct=baseline['accepted'] and baseline['word']==d['arm'],
                partial_received=new['accepted'],partial_correct=new['accepted'] and new['word']==d['arm']))
    dump(OUT/'coverage_gate_comparison.json',comparison)
    # Positive control: polynomial rank and recovery without pixel quantization.
    source=selected[0]['samples'];p,u,n,_=geometry(source);stations=np.unique(np.r_[np.arange(0,len(p),2),len(p)-1]).astype(int);truth=np.array([.71,-.63]);offset=.25
    controls=[]
    for scenario in SCENARIOS:
        keep=mask_keep(source,dict(key='synthetic_fixed',view=0),scenario,stations);ii=stations[keep]
        xy=p[ii]+(offset+basis(u[ii])@truth)[:,None]*n[ii]
        result=fit(source,dict(indices=ii,xy=xy,weights=np.ones(len(ii))),'S')
        error=float(np.max(abs(np.array(result['coefficient'])-truth))) if result else None
        if result:assert error<1e-8,(scenario,error)
        # Omitted-variable control: two separate offset+single-basis fits use
        # exactly the same noiseless observations. Partial intervals break
        # orthogonality and can confound the coefficients.
        y=offset+basis(u[ii])@truth;marginal=[]
        for k in range(2):marginal.append(float(np.linalg.lstsq(np.c_[np.ones(len(ii)),basis(u[ii])[:,k]],y,rcond=None)[0][1]))
        controls.append(dict(scenario=scenario['name'],samples=len(ii),max_coefficient_error=error,rank=int(np.linalg.matrix_rank(np.c_[np.ones(len(ii)),basis(u[ii])])),
            marginal_coefficients=marginal,marginal_max_error=float(np.max(abs(np.array(marginal)-truth)))))
    dump(OUT/'algebraic_positive_control.json',dict(records=controls,
        note='Noiseless continuous test checks algebra/parameter orientation only; not a diffusion or pixel-domain performance result. Rank failure must erase.'))
    # Accepted vs original strict rejections, including imperfect matches.
    report=[];decoded=read(OUT/'decoded.json')
    for split in ['OldEval','Fresh']:
        for scenario in SCENARIOS:
            for track in ['ObsDrop']+(['EdgeDrop'] if scenario['name'] in REMATCH else []):
                rr=[r for r in decoded if r['split']==split and r['arm'] in WORDS and r['scenario']==scenario['name'] and r['track']==track]
                new=[r for r in rr if r['accepted'] and not r['baseline_accept']]
                report.append(dict(split=split,scenario=scenario['name'],track=track,new_received=len(new),new_correct=sum(r['word']==r['arm'] for r in new),new_wrong=sum(r['word']!=r['arm'] for r in new)))
    dump(OUT/'newly_received.json',report)
    fresh_marked_keys={c['key'] for c in read(OUT/'selection.json') if c['marked']};paired=[]
    for scenario in SCENARIOS:
        for track in ['ObsDrop']+(['EdgeDrop'] if scenario['name'] in REMATCH else []):
            rr=[r for r in decoded if r['split']=='Fresh' and r['key'] in fresh_marked_keys and r['scenario']==scenario['name'] and r['track']==track]
            paired.append(dict(scenario=scenario['name'],track=track,**summarize(rr)))
    dump(OUT/'paired_presence.json',dict(records=paired,note='Same12 contents, clean12 + four messages48, same prompt/seed within content. Supplemental;128 clean bank remains primary independent clean validation.'))
    # Descriptive pre-gate diagnosis only: no new receiver/threshold is
    # selected from Fresh, and these predictions are not acceptance results.
    raw_rows=[]
    for d in all_data(['OldCal','OldEval','Fresh']):
        for name,tracks in d['scenarios'].items():
            raw=tracks['ObsDrop'];hypothesis=decode(cc[d['key']],dict(raw,accepted=bool(raw.get('fitted'))),f['channel_model'],f['presence'])
            gated=select(raw,gate)
            raw_rows.append(dict(job=d['job'],split=d['split'],arm=d['arm'],scenario=name,fitted=bool(raw.get('fitted')),original_accepted=d['original']['accepted'],
                predicted_word=hypothesis['word'],receiver_accepted=gated['accepted'],recovery_checks=gated.get('recovery_checks',{})))
    raw_summary=[]
    for split in ['OldCal','OldEval','Fresh']:
        for scenario in SCENARIOS:
            rr=[r for r in raw_rows if r['split']==split and r['arm'] in WORDS and r['scenario']==scenario['name']]
            raw_summary.append(dict(split=split,scenario=scenario['name'],attempts=len(rr),fitted=sum(r['fitted'] for r in rr),
                pre_gate_correct=sum(r['predicted_word']==r['arm'] for r in rr),pre_gate_wrong=sum(r['fitted'] and r['predicted_word']!=r['arm'] for r in rr),
                rejected_correct_hypotheses=sum(not r['receiver_accepted'] and r['predicted_word']==r['arm'] for r in rr),
                rejected_wrong_hypotheses=sum(not r['receiver_accepted'] and r['fitted'] and r['predicted_word']!=r['arm'] for r in rr)))
            raw_summary[-1]['original_correspondence_rejected']=sum(not r['original_accepted'] for r in rr)
            raw_summary[-1]['partial_fit_unavailable_after_original_accept']=sum(r['original_accepted'] and not r['fitted'] for r in rr)
            raw_summary[-1]['confidence_rejected_fitted']=sum(r['fitted'] and not r['receiver_accepted'] for r in rr)
            raw_summary[-1]['failed_recovery_checks']={k:sum(r['fitted'] and not r['recovery_checks'].get(k,False) for r in rr) for k in ['information','condition','uncertainty','residual','offset']}
    dump(OUT/'pre_gate_diagnosis.json',dict(records=raw_rows,summary=raw_summary,
        note='Descriptive only, classify every numerically fitted candidate before confidence rejection. Not an operational receiver, not a modified frozen gate, no valid correspondence guarantee.'))
    # Uniform continuous-arc information: a geometry-only explanation, not
    # a prediction of diffusion recovery or a reason to lower frozen gates.
    t=np.linspace(0,1,20001);bb=basis(t)
    def gram(v):
        centered=v-v.mean(0);return centered.T@centered/len(t)
    complete=gram(bb);theory=[]
    for kind in ['center','ends','left']:
        for loss in [.0,.1,.25,.4,.5,.6,.75]:
            if kind=='center':keep=abs(t-.5)>=loss/2
            elif kind=='ends':keep=(t>=loss/2)&(t<=1-loss/2)
            else:keep=t>=loss
            g=gram(bb[keep]);eig=eigvalsh(g,complete)
            corr=float(g[0,1]/np.sqrt(g[0,0]*g[1,1]));diag=np.sum(bb[keep]**2,axis=0)/np.sum(bb**2,axis=0)
            theory.append(dict(kind=kind,loss=loss,joint_information_eigenvalues=eig.tolist(),ST_uncentered_coverage=diag.tolist(),offset_projected_correlation=corr))
    dump(OUT/'geometry_information_theory.json',dict(records=theory,complete_gram=complete.tolist(),
        caveat='Uniform continuous arc, independent homoscedastic observations, nuisance offset projected out. Identifiability only, not an empirical channel/FPR guarantee.'))
    baseline={(r['job'],r['track']):r for r in decoded if r['scenario']=='none'};transitions=[]
    for scenario in SCENARIOS:
        for track in ['ObsDrop']+(['EdgeDrop'] if scenario['name'] in REMATCH else []):
            rr=[r for r in decoded if r['split']=='Fresh' and r['arm'] in WORDS and r['scenario']==scenario['name'] and r['track']==track]
            originally_correct=[r for r in rr if baseline[r['job'],track]['accepted'] and baseline[r['job'],track]['word']==r['arm']]
            paired=[r for r in rr if r['accepted'] and baseline[r['job'],track]['accepted']]
            drift=np.array([np.array(r['feature'])-baseline[r['job'],track]['feature'] for r in paired])
            transitions.append(dict(scenario=scenario['name'],track=track,all_attempts=len(rr),
                full_correct=len(originally_correct),retained_correct=sum(r['accepted'] and r['word']==r['arm'] for r in originally_correct),
                newly_wrong_from_full_correct=sum(r['accepted'] and r['word']!=r['arm'] for r in originally_correct),
                erased_from_full_correct=sum(not r['accepted'] for r in originally_correct),
                paired_received=len(paired),median_abs_ST_drift=np.median(abs(drift),axis=0).tolist() if len(drift) else None))
    dump(OUT/'full_to_partial_transitions.json',dict(records=transitions,
        note='Conditioning on successful full observations is descriptive and optimistic; all48 remains primary denominator. Coefficient drift is against full estimate, not unknown true output geometry.'))
    failure_rows=[]
    for d in all_data(['Fresh']):
        if d['arm'] not in WORDS:continue
        old=d['original'];decisions=old.get('decisions',[])
        failure_rows.append(dict(job=d['job'],key=d['key'],arm=d['arm'],original_accepted=old['accepted'],
            numerical_proposals=len(decisions),best_candidate_failed_checks=[k for k,v in min(decisions,key=lambda r:r['cost'])['gate_checks'].items() if not v] if decisions else [],
            best_candidate_coefficient=min(decisions,key=lambda r:r['cost'])['coefficient'] if decisions else None,
            warning='Rejected candidate coefficients are unverified correspondences, not received message evidence.'))
    counts={}
    for r in failure_rows:
        if r['original_accepted']:continue
        if not r['numerical_proposals']:counts['no_numerical_proposal']=counts.get('no_numerical_proposal',0)+1
        for k in r['best_candidate_failed_checks']:counts[k]=counts.get(k,0)+1
    dump(OUT/'original_sync_failures.json',dict(records=failure_rows,attempts=len(failure_rows),original_rejected=sum(not r['original_accepted'] for r in failure_rows),
       failed_checks_overlapping=counts,note='Descriptive only. No Fresh threshold/proposal changes; best proposal chosen by bit-blind cost. Failures can overlap.'))
    print('DIAGNOSTICS_COMPLETE',flush=True)

if __name__=='__main__':run()
