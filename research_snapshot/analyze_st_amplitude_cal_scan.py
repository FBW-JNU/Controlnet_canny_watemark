"""Frozen LOCO scoring rule, input diagnosis, and audits for peak sweep."""
from canny_st_amplitude_cal_scan import *
from scipy.special import logsumexp
from scipy.stats import rankdata
import math,collections

METHODS=['SourceRef','R2_tube','R2_bbox']


def analysis_freeze():
    check();dest=OUT/'analysis_freeze.json'
    if dest.exists():assert digest(__file__)==read(dest)['source_sha256'];return
    dump(dest,dict(created_utc=b4.now(),source_sha256=digest(__file__),protocol_sha256=digest(OUT/'protocol.json'),
        fold='Outer leave-one-content-out; both views all4words excluded from class/clean density fit. Same model_fit/classify functions for every peak. No hyperparameter search.',
        threshold='32 separate Cal clean, k=ceil(33*(1-alpha)), strict greater-than. Rates alpha=0.05/0.10. Receiver rejects retain -1e6 and all denominators.',
        locked_control='SourceRef additionally decoded with old global channel A/intercept/covariance and public nominal centers scaled to peak; no source-specific message coefficients.',
        caveat='Primary uses new fixed-rule strength-specific Cal score models under LOCO. Geometry and fit gates remain frozen. Old-global-channel control is in-sample diagnostic, not independent validation.'))
    shutil.copyfile(__file__,OUT/'source'/Path(__file__).name)


def input_receivers():
    analysis_freeze();cv2.setNumThreads(1);cc={c['key']:c for c in read(OUT/'selection.json')};rows=[]
    for n,r in enumerate(read(OUT/'input_closure.json')):
        dest=OUT/'input'/f'{r["key"]}_{tag(r["peak"])}_{r["word"]}.json'
        if dest.exists():rows.append(read(dest));continue
        c=cc[r['key']];image=np.array(Image.open(r['hint']).convert('L'))
        obs=b4.receive(c['samples'],b4.binary_target(image,c['samples']),'S',b4.a4.THRESHOLDS,b4.a4.GATE)
        nominal=(r['peak']/b4.a4.PEAK)*np.array([b4.a4.SIGNS[w] for w in WORDS]);out=dict(key=r['key'],peak=r['peak'],word=r['word'],heads={})
        for head in ['Auto','Oracle']:
            f=obs[head];value=np.array(f['coefficient'])-np.array(c['states']['clean']['measurements'][head]['coefficient']) if f.get('accepted') else None
            word=WORDS[int(np.argmin(np.sum((nominal-value)**2,axis=1)))] if value is not None else None
            out['heads'][head]=dict(accepted=bool(f.get('accepted')),word=word,correct=word==r['word'],feature=value.tolist() if value is not None else None)
        dump(dest,out);rows.append(out);print('INPUT',n+1,'/96',r['key'],r['peak'],flush=True)
    dump(OUT/'input_receiver_summary.json',{tag(peak):{head:dict(attempts=24,received=sum(r['heads'][head]['accepted'] for r in rows if r['peak']==peak),correct=sum(r['heads'][head]['correct'] for r in rows if r['peak']==peak)) for head in ['Auto','Oracle']} for peak in PEAKS})


def calibration_negatives():
    # Only original4B32 Cal clean, never Fresh/Eval data.
    cc={c['key']:c for c in read(b4.OUT/'selection.json') if c['split']=='Cal'};config=read(C0/'score_freeze.json');records=[]
    for p in sorted((C0/'cal_features').glob('*.json')):
        r=read(p)
        if r['key'] not in cc:continue
        assert r['arm']=='clean';c=cc[r['key']];obs=r['reference_observation']
        value=(np.array(obs['coefficient'])-np.array(c['states']['clean']['measurements']['Auto']['coefficient'])).tolist() if obs.get('accepted') else None
        methods=dict(SourceRef=dict(accepted=bool(obs.get('accepted')),value=value,covariance=obs.get('covariance',np.eye(2).tolist())))
        for method in ['R2_tube','R2_bbox']:
            conf=config[method]['selected'];f=r['public'][method]['fits'][str(conf['degree'])];ok=c0.eligible(f,conf['residual'],conf['ci95'])
            methods[method]=dict(accepted=ok,value=f['coefficient'] if ok else None,covariance=f.get('covariance',np.eye(2).tolist()))
        records.append(dict(job=r['job'],content_id=r['content_id'],methods=methods,source_feature=str(p),source_sha256=digest(p)))
    assert len(records)==32;dump(OUT/'threshold_clean32.json',records);return records


def auroc(pos,neg):
    if not len(pos) or not len(neg):return None
    values=np.r_[pos,neg];ranks=rankdata(values)
    return float((ranks[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))


def order_threshold(scores,alpha):
    k=int(np.ceil((len(scores)+1)*(1-alpha)))
    return dict(alpha=alpha,n=len(scores),k=k,value=float(sorted(scores)[k-1]) if k<=len(scores) else None,
        realized_cal_false_accepts=sum(s>sorted(scores)[k-1] for s in scores) if k<=len(scores) else 0)


def score(value,model):return c0.classify(value['value'] if value['accepted'] else None,value['covariance'],model)


def summarize(rows,method,peak):
    marked=[r for r in rows if r['method']==method and r['peak']==peak and r['arm'] in WORDS];clean=[r for r in rows if r['method']==method and r['peak']==peak and r['arm']=='clean']
    received=[r for r in marked if r['decoded']['accepted']];correct=sum(r['decoded']['word']==r['arm'] for r in received)
    result=dict(attempts=48,geometry_received=sum(r['geometry_accepted'] for r in marked),received=len(received),correct=correct,wrong=len(received)-correct,erasures=48-len(received),
        full_accuracy=correct/48,error_rate=(len(received)-correct)/48,erasure_rate=(48-len(received))/48,conditional_accuracy=correct/len(received) if received else None,
        high_confidence_wrong=sum(r['decoded'].get('high_confidence',False) and r['decoded']['word']!=r['arm'] for r in received),
        clean_attempts=12,clean_received=sum(r['decoded']['accepted'] for r in clean),presence_AUC=auroc([r['decoded']['score'] for r in marked],[r['decoded']['score'] for r in clean]),fixed_FPR={})
    for alpha in [.05,.10]:
        key=str(alpha);det=lambda r:bool(r['thresholds'][key]['value'] is not None and r['decoded']['score']>r['thresholds'][key]['value'])
        tp=sum(det(r) for r in marked);fp=sum(det(r) for r in clean)
        result['fixed_FPR'][key]=dict(marked_detected=tp,marked_attempts=48,TPR=tp/48,heldout_clean_false_accepts=fp,heldout_clean_attempts=12,realized_FPR=fp/12)
    folded=[]
    for cid in sorted({r['content_id'] for r in marked}):
        folded.append(auroc([r['decoded']['score'] for r in marked if r['content_id']==cid],[r['decoded']['score'] for r in clean if r['content_id']==cid]))
    result['mean_within_content_AUC']=float(np.mean(folded));return result


def evaluate():
    analysis_freeze();plan=read(OUT/'plan.json');records=[read(OUT/'features'/f'{j["job"]}.json') for j in plan];assert len(records)==204
    negatives=calibration_negatives();ids=sorted({r['content_id'] for r in records});assert len(ids)==6 and not set(ids)&{r['content_id'] for r in negatives}
    output=[];folds=[]
    for peak in PEAKS:
        subset=[r for r in records if r['peak']==peak or r['arm']=='clean'];assert len(subset)==60
        for method in METHODS:
            for cid in ids:
                train=[r for r in subset if r['content_id']!=cid];holdout=[r for r in subset if r['content_id']==cid]
                fitting=[dict(content_id=r['content_id'],arm=r['arm'],value=r['methods'][method]['value']) for r in train if r['methods'][method]['accepted']]
                model=c0.model_fit(fitting)
                calibration=[score(n['methods'][method],model)['score'] for n in negatives]
                thresholds={str(alpha):order_threshold(calibration,alpha) for alpha in [.05,.10]}
                folds.append(dict(peak=peak,method=method,heldout_content=cid,train_contents=sorted({r['content_id'] for r in train}),model=model,thresholds=thresholds,calibration_scores=calibration))
                for r in holdout:
                    d=score(r['methods'][method],model)
                    output.append(dict(job=r['job'],original_job=r['original_job'],key=r['key'],content_id=cid,view=r['view'],arm=r['arm'],peak=peak,method=method,
                        geometry_accepted=r['methods'][method]['accepted'],decoded=d,thresholds=thresholds))
    dump(OUT/'loco_folds.json',folds);dump(OUT/'decisions.json',output)
    summary={tag(peak):{method:summarize(output,method,peak) for method in METHODS} for peak in PEAKS};dump(OUT/'summary.json',summary)
    # Locked original channel, separate descriptive control, scaled nominal book.
    legacy=read(b4.OUT/'score_freeze.json');channel=legacy['channel_model'];base=read(C0/'score_freeze.json')['nominal_channel'];locked=[]
    for peak in PEAKS:
        centers=[(np.array(channel['A'])@((peak/b4.a4.PEAK)*np.array(b4.a4.SIGNS[w]))+channel['intercept']).tolist() for w in WORDS]
        model=dict(base,centers=centers);marked=[r for r in records if r['peak']==peak];dec=[(r,score(r['methods']['SourceRef'],model)) for r in marked];valid=[(r,d) for r,d in dec if d['accepted']];correct=sum(d['word']==r['arm'] for r,d in valid)
        locked.append(dict(peak=peak,attempts=48,correct=correct,wrong=len(valid)-correct,erasures=48-len(valid),full_accuracy=correct/48,
            caveat='Legacy global Cal model reused on its own Cal contents; descriptive control, not independent estimate. Nominal codebook scaled to actual peak.'))
    dump(OUT/'locked_channel_control.json',locked)
    # Quality: same clean for all peaks; no omission for decoding failures.
    quality=[read(p) for p in sorted((OUT/'quality').glob('*.json'))];assert len(quality)==192;qsummary={}
    for peak in PEAKS:
        rr=[r for r in quality if r['peak']==peak];assert len(rr)==48
        qsummary[tag(peak)]=dict(attempts=48,joint_passes=sum(r['pass_all'] for r in rr),per_gate_passes={g:sum(r['gates'][g] for r in rr) for g in ['psnr','ssim','lpips','edge_f1']},
             medians={k:float(np.median([r['quality'][k] for r in rr])) for k in ['psnr_db','ssim','lpips_alex','edge_f1_3x3']},
             worst={k:float((min if k!='lpips_alex' else max)(r['quality'][k] for r in rr)) for k in ['psnr_db','ssim','lpips_alex','edge_f1_3x3']},
             failed_jobs=[r['job'] for r in rr if not r['pass_all']])
    dump(OUT/'quality_summary.json',qsummary)
    # Content paired changes; do not treat two views/four words as independent.
    percontent=[]
    for cid in ids:
        for method in METHODS:
            for peak in PEAKS:
                rr=[r for r in output if r['content_id']==cid and r['method']==method and r['peak']==peak and r['arm'] in WORDS];ok=[r for r in rr if r['decoded']['accepted']]
                percontent.append(dict(content_id=cid,method=method,peak=peak,attempts=8,received=len(ok),correct=sum(r['decoded']['word']==r['arm'] for r in ok)))
    dump(OUT/'metrics/per_content.json',percontent)
    # Paired source-referenced finite differences, only all4 geometrically accepted.
    response=[]
    for peak in PEAKS:
        complete=[]
        for cid in ids:
            for view in [0,1]:
                rr={r['arm']:r for r in records if r['peak']==peak and r['content_id']==cid and r['view']==view}
                if all(rr[w]['methods']['SourceRef']['accepted'] for w in WORDS):
                    y={w:np.array(rr[w]['methods']['SourceRef']['value']) for w in WORDS};amp=peak/b4.a4.PEAK
                    matrix=np.column_stack([(y['m10']+y['m11']-y['m00']-y['m01'])/(4*amp),(y['m01']+y['m11']-y['m00']-y['m10'])/(4*amp)])
                    complete.append(dict(content_id=cid,view=view,matrix=matrix.tolist(),singular_values=np.linalg.svd(matrix,compute_uv=False).tolist()))
        response.append(dict(peak=peak,complete_four_word_groups=len(complete),attempted_groups=12,groups=complete))
    dump(OUT/'metrics/response_matrices.json',response)
    print('EVALUATION_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['inputs','evaluate']);a=p.parse_args();input_receivers() if a.action=='inputs' else evaluate()
