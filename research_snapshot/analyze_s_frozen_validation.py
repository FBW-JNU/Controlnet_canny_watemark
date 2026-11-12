"""No-tuning scoring and old-S erasure/quality diagnosis."""
from experiment_s_frozen_validation import *
from broad_peak_runtime import presence
from curve_receiver_stage2c import prepare_target
from analyze_canny_curve_stage2_bit import auc
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from carrier_screen.common import pixel_metrics

def analysis_freeze():
    check();p=OUT/'analysis_freeze.json';sha=digest(__file__)
    if p.exists():assert read(p)['script_sha256']==sha;return
    assert not list((OUT/'rgb').glob('*.png'))
    dump(p,dict(script_sha256=sha,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),policy='Uses preceding S Cal score models unchanged. No new thresholds, no coefficient-margin abstention. Fresh24 only evaluated. Bootstrap2000 resamples whole contents.'))
    shutil.copyfile(__file__,OUT/'source'/Path(__file__).name)

def decode(j,c,obs,models):
    result=[]
    for head in ['Auto','Oracle']:
        book=c['codebooks']['S'][head];raw=code_score(obs[head],book);model=models[head]
        for method in ['raw','affine']:
            s=(raw-model['intercept'])/model['gain'] if raw is not None and method=='affine' else raw
            sp=presence(abs(s),model['presence_model']) if s is not None and method=='affine' else 0.
            result.append(dict(job=j['job'],key=j['key'],content_id=j['content_id'],arm=j['arm'],view=j['view'],head=head,method=method,
                geometry_accepted=obs[head]['accepted'],codebook_available=book is not None,accepted=s is not None,score=s if s is not None else 0.,
                bit=int(s>0) if s is not None else None,s_presence=sp,presence_accept=bool(s is not None and method=='affine' and sp>model['threshold95'])))
    return result

def input_closure(rows,selection,models):
    cc={c['key']:c for c in selection};details=[]
    for row in rows:
        c=cc[row['key']];image=arr(row['hint']);r=receive(c['samples'],binary_target(image,c['samples']),'S',THRESHOLDS,GATE)
        for h in ['Auto','Oracle']:
            assert r[h]['accepted']==row['measurements'][h]['accepted']
            a=r[h]['coefficient'];b=row['measurements'][h]['coefficient'];assert (a is None and b is None) or np.allclose(a,b,atol=1e-10)
        if row['arm']=='clean':continue
        for h in ['Auto','Oracle']:
            book=c['codebooks']['S'][h];s=code_score(r[h],book);m=models[h];correct=s is not None and int(((s-m['intercept'])/m['gain'])>0)==int(row['arm'][-1])
            details.append(dict(key=c['key'],split=c['split'],arm=row['arm'],head=h,active=row['changed_pixels']>0,geometry_received=r[h]['accepted'],book_available=book is not None,score=s,correct=bool(correct)))
    summary={}
    for split in sorted({r['split'] for r in details})+['all']:
        for h in ['Auto','Oracle']:
            rr=[r for r in details if r['head']==h and (r['split']==split or split=='all')];n=len(rr);good=sum(r['active'] and r['correct'] for r in rr)
            summary[split+'_'+h]=dict(attempts=n,active=sum(r['active'] for r in rr),geometry_received=sum(r['geometry_received'] for r in rr),correct=good,rate=good/n if n else 0.,pass99=bool(n and good/n>=.99))
    return dict(summary=summary,records=details,policy='Reload actual saved PNG; include input-book failures and zero edits. Unique input states only, not inflated by repeating prompt/seed. Source-only admissible selection is reported separately from raw candidate denominator.')

def diagnose():
    analysis_freeze();f=check();models=f['score_models'];cc=read(PREV/'runtime_selection.json');rows=[r for r in read(PREV/'inputs.json') if r['mode']=='S']
    dump(OUT/'old_input_closure.json',input_closure(rows,cc,models))
    plan=[j for j in read(PREV/'runtime_plan.json') if j['mode']=='S' and j['split']=='Diagnostic'];bykey={c['key']:c for c in cc};details=[];all_decoded=[]
    for j in plan:
        c=bykey[j['key']];obs=read(PREV/'measurements'/f'{j["job"]}.json')['observations'];dd=decode(j,c,obs,models);all_decoded+=dd
        d=next(r for r in dd if r['head']=='Auto' and r['method']=='affine')
        if j['arm']=='clean':continue
        if not d['accepted'] or d['bit']!=int(j['arm'][-1]):
            rgb=np.array(Image.open(j['rgb']).convert('RGB'));edge=cv2.Canny(rgb,100,200);yy,xx=np.nonzero(edge);support=cKDTree(np.c_[xx,yy]).query(c['samples'])[0]
            r=obs['Auto'];candidates=r['decisions'];best=min(candidates,key=lambda q:sum(not x for x in q['gate_checks'].values())) if candidates else None
            reason='input_codebook_unavailable' if not d['codebook_available'] else 'trusted_correspondence_rejected' if not d['geometry_accepted'] else 'accepted_wrong_bit'
            detail=dict(**d,reason=reason,nearby_support_2px=float(np.mean(support<=2)),nearby_support_4px=float(np.mean(support<=4)),
                oracle_accepted=obs['Oracle']['accepted'],oracle_score=next(x['score'] for x in dd if x['head']=='Oracle' and x['method']=='affine'),
                failed_checks=[k for k,v in best['gate_checks'].items() if not v] if best else ['no_fit'],best_candidate=best,
                interpretation='Nearby/Oracle observations do not prove target identity; retain erasure without threshold relaxation.' if not d['accepted'] else 'Measured coefficient crossed frozen decision boundary; inspect noise and margin, no automatic correction.')
            details.append(detail)
    dd=[d for d in all_decoded if d['head']=='Auto' and d['method']=='affine' and d['arm']!='clean'];erasures=[r for r in details if not r['accepted']]
    assert len(erasures)==14 and sum(r['accepted'] and r['bit']==int(r['arm'][-1]) for r in dd)==31
    dump(OUT/'old_erasure_diagnosis.json',dict(marked=48,correct=31,wrong=3,erasures=14,input_book_unavailable=sum(r['reason']=='input_codebook_unavailable' for r in erasures),
        output_correspondence_rejected=sum(r['reason']=='trusted_correspondence_rejected' for r in erasures),coefficient_margin_erasures=0,
        note='Frozen decoder has no coefficient-margin abstention. An accepted coefficient always produces a bit;3 failures are wrong decisions, not erasures. Oracle accepted all6 remaining output-only erasures, but target identity cannot be certified.',records=details))
    quality=read(PREV/'quality_S.json');qrows=[]
    for q in quality['records']:
        if q['split']!='Diagnostic':continue
        j=next(j for j in plan if j['job']==q['job']);c=bykey[j['key']];clean=np.array(Image.open(j['clean_rgb']).convert('RGB'));image=np.array(Image.open(j['rgb']).convert('RGB'))
        a=cv2.Canny(clean,100,200)>0;b=cv2.Canny(image,100,200)>0;ad=cv2.dilate(a.astype('uint8'),np.ones((3,3),np.uint8))>0;bd=cv2.dilate(b.astype('uint8'),np.ones((3,3),np.uint8))>0
        unmatched=(a&~bd)|(b&~ad);mask=arr(c['mask'])>0;near=cv2.dilate(mask.astype('uint8'),np.ones((17,17),np.uint8))>0
        inp=next(r for r in rows if r['key']==c['key'] and r['arm']==j['arm']);p=np.array(c['samples']);shape=c.get('source_geometry',{});inputimage=arr(j['hint']);hinttree=cKDTree(np.c_[np.nonzero(inputimage)[1],np.nonzero(inputimage)[0]])
        r=dict(job=q['job'],key=c['key'],view=j['view'],stratum=j['stratum'],arm=j['arm'],pass_all=q['pass_all'],gates=q['gates'],quality=q['quality'],
            arc_length_px=c['arc_length_px'],source_geometry=shape,input_changed_pixels=inp['changed_pixels'],clean_edge_pixels=int(a.sum()),marked_edge_pixels=int(b.sum()),
            unmatched_edges=int(unmatched.sum()),unmatched_outside_mask=int((unmatched&~mask).sum()),unmatched_outside_near=int((unmatched&~near).sum()),
            unmatched_outside_near_fraction=float((unmatched&~near).sum()/max(1,unmatched.sum())))
        qrows.append(r)
    failed=[r for r in qrows if not r['pass_all']];assert len(failed)==4
    dump(OUT/'old_quality_diagnosis.json',dict(records=qrows,failures=failed,all_fail_edge_f1=all(not r['gates']['edge_f1'] for r in failed),
        note='4 failures are two content/view pairs, both messages; PSNR/SSIM/LPIPS pass. Global Canny differences outside edited region may explain Edge-F1 sensitivity; source geometry alone is not a demonstrated cause.'))
    print('OLD_DIAGNOSIS_COMPLETE',flush=True)

def extract():
    check();assert read(OUT/'analysis_freeze.json')['script_sha256']==digest(__file__);cc={c['key']:c for c in read(OUT/'selection.json')}
    for ix,j in enumerate(read(OUT/'plan.json')):
        dest=OUT/'measurements'/f'{j["job"]}.json'
        if dest.exists():continue
        record=OUT/'records'/f'{j["job"]}.json'
        while not record.exists():time.sleep(2)
        assert digest(j['rgb'])==read(record)['rgb_sha256'];rgb=np.array(Image.open(j['rgb']).convert('RGB'));target=prepare_target(rgb)
        obs=receive(cc[j['key']]['samples'],target,'S',THRESHOLDS,GATE);dump(dest,dict(job=j,observations=obs,rgb_sha256=digest(j['rgb'])))
        print('EXTRACT',ix+1,'/288',j['job'],flush=True)

def metrics(rows):
    marked=[r for r in rows if r['arm']!='clean'];clean=[r for r in rows if r['arm']=='clean'];good=[r for r in marked if r['accepted']]
    return dict(marked=len(marked),geometry_received=sum(r['geometry_accepted'] for r in marked),received=len(good),correct=sum(r['accepted'] and r['bit']==int(r['arm'][-1]) for r in marked),
        wrong=sum(r['accepted'] and r['bit']!=int(r['arm'][-1]) for r in marked),erasures=len(marked)-len(good),
        bit_auc=auc([int(r['arm']=='bit1') for r in marked],[r['score'] for r in marked]),all_attempt_accuracy=sum(r['accepted'] and r['bit']==int(r['arm'][-1]) for r in marked)/len(marked),
        presence_auc=auc([int(r['arm']!='clean') for r in rows],[r['s_presence'] for r in rows]),
        clean=len(clean),clean_presence_accepted=sum(r['presence_accept'] for r in clean),marked_presence_accepted=sum(r['presence_accept'] for r in marked))

def evaluate():
    f=check();assert read(OUT/'analysis_freeze.json')['script_sha256']==digest(__file__);cc=read(OUT/'selection.json');bykey={c['key']:c for c in cc};plan=read(OUT/'plan.json');decoded=[]
    for j in plan:
        row=read(OUT/'measurements'/f'{j["job"]}.json');assert digest(j['rgb'])==row['rgb_sha256'];decoded+=decode(j,bykey[j['key']],row['observations'],f['score_models'])
    summaries={}
    for head in ['Auto','Oracle']:
        for method in ['raw','affine']:summaries[head+'_'+method]=metrics([r for r in decoded if r['head']==head and r['method']==method])
    dump(OUT/'decoded.json',decoded);dump(OUT/'summary.json',summaries);dump(OUT/'new_input_closure.json',input_closure(read(OUT/'inputs.json'),cc,f['score_models']))
    grouped=[]
    for c in cc:
        rr=[r for r in decoded if r['key']==c['key'] and r['head']=='Auto' and r['method']=='affine'];grouped.append(dict(key=c['key'],content_id=c['content_id'],arc_length_px=c['arc_length_px'],stratum=c['stratum'],**metrics(rr)))
    dump(OUT/'per_content.json',grouped)
    rng=np.random.default_rng(2026111143);cis={}
    for head in ['Auto','Oracle']:
        rr=[r for r in decoded if r['head']==head and r['method']=='affine'];values=[]
        blocks=[[r for r in rr if r['key']==c['key']] for c in cc]
        for _ in range(2000):
            selected=[r for i in rng.integers(0,len(blocks),len(blocks)) for r in blocks[i]];s=metrics(selected);values.append([s['all_attempt_accuracy'],s['bit_auc'],s['presence_auc']])
        cis[head]=dict(zip(['all_attempt_accuracy','bit_auc','presence_auc'],np.percentile(values,[2.5,97.5],axis=0).T.tolist()))
    dump(OUT/'bootstrap.json',dict(contents=24,draws=2000,policy='Whole-content bootstrap retaining4 views and all arms',intervals=cis))
    # Quality uses exactly the frozen four gates, complete192 marked outputs.
    import torch,lpips
    torch.set_num_threads(4);cv2.setNumThreads(1);net=lpips.LPIPS(net='alex',version='0.1').eval().requires_grad_(False).cpu();qr=[];g=f['quality']
    for ix,j in enumerate([j for j in plan if j['arm']!='clean']):
        clean=np.array(Image.open(j['clean_rgb']).convert('RGB'));image=np.array(Image.open(j['rgb']).convert('RGB'));q=pixel_metrics(clean,image)
        a=torch.from_numpy(clean).permute(2,0,1)[None].float()/127.5-1;b=torch.from_numpy(image).permute(2,0,1)[None].float()/127.5-1
        with torch.inference_mode():q['lpips_alex']=float(net(a,b))
        gates=dict(psnr=q['psnr_infinite'] or q['psnr_db']>=g['psnr_db_min'],ssim=q['ssim']>=g['ssim_min'],lpips=q['lpips_alex']<=g['lpips_alex_max'],edge_f1=q['edge_f1_3x3']>=g['rgb_pair_edge_f1_3x3_min'])
        qr.append(dict(job=j['job'],key=j['key'],view=j['view'],arm=j['arm'],quality=q,gates=gates,pass_all=bool(all(gates.values()))));print('QUALITY',ix+1,'/192',flush=True)
    dump(OUT/'quality.json',dict(attempts=len(qr),joint_passes=sum(r['pass_all'] for r in qr),records=qr,gates=g))
    caps=[read(p) for p in sorted((OUT/'capacity').glob('*.json'))];dump(OUT/'capacity_summary.json',dict(contents=24,source_candidates=sum(c['source_candidates'] for c in caps),eligible=sum(c['eligible'] for c in caps),
        independent_lower_bounds=[dict(content_id=c['content_id'],count=c['independent_lower_bound']) for c in caps],at_least8=sum(c['independent_lower_bound']>=8 for c in caps),at_least32=sum(c['independent_lower_bound']>=32 for c in caps),
        limitation='Source-only writable and spatially isolated candidates. Greedy lower bound, not maximum independent set. No diffusion stability or statistical independence certification, no multi-carrier RGB generated.'))
    print('EVALUATION_COMPLETE',summaries,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['diagnose','extract','evaluate']);a=p.parse_args();globals()[a.action]()
