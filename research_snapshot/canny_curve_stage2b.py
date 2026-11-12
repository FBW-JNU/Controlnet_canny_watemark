"""Stage2b receiver-only ablations and a separately frozen fresh test."""
from __future__ import annotations
import argparse,ast,csv,math,os,shutil,time
from pathlib import Path
from datetime import datetime,timezone
import cv2
import numpy as np
from PIL import Image
from scipy.special import logsumexp,expit
from canny_curve_stage2_bit import ROOT,OUT as OLD,S1,read,dump,sha,save,matcher,coefficient,canonical,warp
from curve_receiver_stage2b import prepare_target,r1,r2,input_codebook
from analyze_canny_curve_stage2_bit import auc,distribution,interval

OUT=ROOT/'outputs/canny_curve_stage2b_20261110'
PIDICODE=ROOT/'fuxian_papers/Improved-Scribble/upstream/annotator/pidinet/model.py'
PIDIWEIGHT=ROOT/'fuxian_papers/checkpoints/improved_scribble/table5_pidinet.pth'
R1CONFIG=[dict(endpoint=6,gap=2),dict(endpoint=8,gap=3)]
R2CONFIG=[dict(tube=4,angle=35,skip=3,bridge=6,r_phi=.4),dict(tube=6,angle=40,skip=5,bridge=8,r_phi=.4),dict(tube=6,angle=30,skip=4,bridge=6,r_phi=.5)]
METHODS=['A0','A1','A2','A3','A3_low60','A3_low40','A4']

def now():return datetime.now(timezone.utc).isoformat()
def prepare():
    if (OUT/'protocol.json').exists():return
    for name in ['source','soft','soft_records','results','logs','plots','panels','fresh','fresh/records','fresh/rgb','fresh/noise','fresh/soft','fresh/soft_records','fresh/results']:(OUT/name).mkdir(parents=True,exist_ok=True)
    s2manifest=read(OLD/'final_artifacts_sha256.json')
    for rel,digest in s2manifest['files'].items():assert sha(OLD/rel)==digest,rel
    files=[ROOT/'curve_receiver_stage2b.py',ROOT/'canny_curve_stage2b.py',PIDICODE,PIDIWEIGHT,OLD/'protocol.json',OLD/'selection.json',OLD/'plan.json',OLD/'summary.json',OLD/'final_artifacts_sha256.json']
    protected={str(p):sha(p) for p in files}
    for p in files[:2]:shutil.copyfile(p,OUT/'source'/p.name)
    protocol=dict(created_utc=now(),scope='Stage2b receiver-only first; original Stage2 and Stage1 read-only. Existing Dev now diagnostic only.',
        original_data='24Cal clean +48Cal polynomial marked; 24Dev clean +48Dev polynomial marked; all exact same PNGs and original messages.',
        baseline='A0 reads frozen strict match/features. No R0 parameter, mask, candidate or score changes.',
        basis='ACTUAL embedded basis 6sqrt(3)t(1-t)(2t-1); zero at both endpoints AND center, two information lobes. Not16t²(1-t)².',
        methods=dict(A0='frozen strict matcher',A1='trim10%-90%, relaxed endpoint and observed gap links, unrobust coefficient',A2='ordered real local observations with skips + weighted least squares',
                     A3='A2 + Huber IRLS and intercept nuisance',A3_low60='A3 with Canny60/120',A3_low40='A3 with Canny40/80',A4='A3 Canny40/80 observed RGB edge vertices, weighted by PiDiNet soft response; no synthesized/soft-only vertices'),
        R1_configs=R1CONFIG,R2_configs=R2CONFIG,
        observation_rule='Fixed source and source orientation; no true bit templates. Station2px, top8 local real vertices, geometric Viterbi costs only. Both orders tested with geometry-only winner; coefficient always canonical. No interpolated gap pixels in fit.',
        confidence='Report c_hat,q_sync,s_presence separately. Erase if coverage<.45, span<.65, Rphi below config, either basis lobe information<.22, <4/6 bins, RMS>1.25px or CI95 halfwidth>1.5px; R1 coverage .35,Rphi .3. CI block4px effective sample heuristic, not certified frequentist interval.',
        calibration='Cal ONLY. Every method config evaluated on Cal72 and 2wrong-location controls/image. Eligible config: marked coverage>=.70, bit neutralITT AUC>=.80 and wrong-location acceptance<=max(.10,A0+.05). Select highest marked coverage then AUC then lower negative acceptance; if none, select best diagnostically and flag unqualified. Tie config order. Primary among A3/A4 uses same rule. No Dev retuning.',
        wrong_correspondence='Translate known curve24px horizontally and24px vertically, inward at boundary, same RGB. Diagnostic wrong-location acceptance, NOT certified semantic miscorrespondence ground truth.',
        presence='Cal-only per-class Gaussian density in input-normalized coefficient, variance floor .15², add fit SE². s_presence=sigmoid(log marked mixture likelihood -log clean likelihood). Original abs displacement statistic also reported. Failed receptions remain abstentions; rank ITT uses0; report coverage, success-only AUC, missing bounds and q_sync-only diagnostic.',
        gates=dict(marked_coverage_min=.70,bit_auc_min=.8,presence_auc_target=.7,wrong_location_max='max(.10,A0+.05)',quality='unchanged PNG quality, all attempts included'),
        fresh_test='After Cal freeze: 12 new contents excluded from Stage1 Train32,4/source curvature stratum.6 frozenR0 probe clean views/content for stable source-only-admissible candidate selection; then2 new prompt/seed paired clean,bit0,bit1. Probe cannot tune receiver; final test neither filters contents nor tunes anything. If no stable candidate in assigned stratum, record content as unencodable and generate source-preselected fallback transparently.',
        protected_sha256=protected)
    dump(OUT/'protocol.json',protocol);dump(OUT/'original_artifacts_freeze.json',s2manifest)
    print('FROZEN STAGE2b protocol',flush=True)

def check():
    p=read(OUT/'protocol.json')
    for path,digest in p['protected_sha256'].items():assert sha(path)==digest,path
    return p

def jobs(fresh=False):
    if fresh:return read(OUT/'fresh/test_plan.json')
    return [j for j in read(OLD/'plan.json') if j['arm']=='clean' or j['mode']=='polynomial']

def load_pidi(device):
    import torch
    tree=ast.parse(PIDICODE.read_text());tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and n.module=='basicsr.utils')]
    ns={'__name__':'frozen_stage2b_pidinet'};exec(compile(tree,str(PIDICODE),'exec'),ns)
    net=ns['pidinet']();state=torch.load(PIDIWEIGHT,map_location='cpu',weights_only=True)['state_dict']
    net.load_state_dict({k.replace('module.',''):v for k,v in state.items()},strict=True)
    return net.to(device).eval().requires_grad_(False)

def extract_soft(fresh=False):
    import torch
    check();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and torch.cuda.is_available()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    net=load_pidi('cuda');folder=OUT/'fresh' if fresh else OUT
    for i,j in enumerate(jobs(fresh)):
        dest=folder/'soft'/f'{j["job"]}.npz';record=folder/'soft_records'/f'{j["job"]}.json'
        if record.exists():assert read(record)['rgb_sha256']==sha(j['rgb']);continue
        rgb=np.asarray(Image.open(j['rgb']).convert('RGB'));x=torch.from_numpy(rgb[:,:,::-1].copy()).permute(2,0,1)[None].float().cuda()/255
        with torch.inference_mode():prob=net(x)[-1][0,0].cpu().numpy().astype('float32')
        np.savez_compressed(dest,probability=prob);save(dest.with_suffix('.png'),np.rint(prob*255).astype('uint8'))
        dump(record,dict(rgb=j['rgb'],rgb_sha256=sha(j['rgb']),soft_sha256=sha(dest),weight_sha256=sha(PIDIWEIGHT),code_sha256=sha(PIDICODE),representation='BGR/255 final float32 probability, no MAT, no binary-map reannotation'))
        if (i+1)%12==0:print('SOFT',i+1,'/',len(jobs(fresh)),flush=True)

def cases(fresh=False):return {c['key']:c for c in read(OUT/'fresh/selection.json' if fresh else OLD/'selection.json')}
def codebooks(c):
    m=matcher();t=read(OLD/'protocol.json')['matcher_thresholds'];paths={}
    for arm in ['clean','bit0','bit1']:
        path=Path(c['clean_hint']) if arm=='clean' else Path(c['clean_hint']).parent/f'polynomial_{arm}.png'
        r=m.curve_match(c['samples'],m.skeleton_graph(np.asarray(Image.open(path).convert('L'))),t,True)
        assert r['matched'];paths[arm]=r['output_path']
    return {'LS':input_codebook(c['samples'],paths,False),'Huber':input_codebook(c['samples'],paths,True),'R0':c['input_features']['polynomial']}

def method_result(method,c,target,config):
    if method=='A1':return r1(c['samples'],target,read(OLD/'protocol.json')['matcher_thresholds'],config)
    return r2(c['samples'],target,config,method!='A2')

def configs(method):return [None] if method=='A0' else R1CONFIG if method=='A1' else R2CONFIG

def evaluate(split='Cal',fresh=False):
    check();cc=cases(fresh);folder=OUT/'fresh' if fresh else OUT;m=matcher();t=read(OLD/'protocol.json')['matcher_thresholds']
    if split!='Cal':assert (OUT/'receiver_freeze.json').exists()
    decision=read(OUT/'receiver_freeze.json') if (OUT/'receiver_freeze.json').exists() else None
    books={k:codebooks(c) for k,c in cc.items()};dump(folder/'input_codebooks.json',books)
    plan=[j for j in jobs(fresh) if fresh or j['split']==split]
    for i,j in enumerate(plan):
        resultfile=folder/'results'/f'{j["job"]}.json'
        if resultfile.exists():continue
        c=cc[j['key']];rgb=np.asarray(Image.open(j['rgb']).convert('RGB'));softpath=folder/'soft'/f'{j["job"]}.npz'
        assert softpath.exists();soft=np.load(softpath)['probability']
        targets={'standard':prepare_target(rgb),'low60':prepare_target(rgb,60,120),'low40':prepare_target(rgb,40,80),'soft':prepare_target(rgb,40,80,soft)}
        outcomes={}
        for method in METHODS:
            opts=configs(method);selected=[decision['methods'][method]['config_index']] if decision and split!='Cal' else range(len(opts))
            outcomes[method]={}
            for ci in selected:
                if method=='A0':
                    if fresh:
                        mm=m.curve_match(c['samples'],targets['standard'],t,True);value=coefficient(c['samples'],mm['output_path'],'polynomial',m) if mm['matched'] else None
                    else:
                        oldr=read(OLD/'metrics'/f'{j["job"]}.json');mm=oldr['match'];value=oldr['coefficients']['polynomial']
                    result=dict(accepted=mm['matched'],fitted=mm['matched'],c_hat=value,q_sync=1. if mm['matched'] else 0.,coefficient_se=0.,reason=mm['reason'],r_phi=None)
                    tar=targets['standard']
                else:
                    name='low60' if method=='A3_low60' else 'low40' if method=='A3_low40' else 'soft' if method=='A4' else 'standard';tar=targets[name]
                    result=method_result(method,c,tar,opts[ci])
                negatives=[]
                for axis in [0,1]:
                    points=np.array(c['samples']);delta=np.zeros(2);delta[axis]=24 if points[:,axis].max()+24<504 else -24
                    negative=dict(c,samples=(points+delta).tolist())
                    if method=='A0':nr=m.curve_match(negative['samples'],tar,t,True);accepted=nr['matched']
                    else:accepted=method_result(method,negative,tar,opts[ci])['accepted']
                    negatives.append(bool(accepted))
                book=books[c['key']]['R0' if method=='A0' else 'LS' if method in ['A1','A2'] else 'Huber']
                if result['accepted']:
                    result['bit_score']=(result['c_hat']-book['midpoint'])/book['half_separation'];result['absolute_presence']=abs((result['c_hat']-book['clean'])/book['half_separation'])
                    result['normalized_se']=result.get('coefficient_se',0)/book['half_separation']
                else:result.update(bit_score=0.,absolute_presence=0.,normalized_se=None)
                result['wrong_location_acceptance']=negatives;outcomes[method][str(ci)]=result
        dump(resultfile,dict(**j,rgb_sha256=sha(j['rgb']),soft_sha256=sha(softpath),outcomes=outcomes))
        print('EVAL',split,i+1,'/',len(plan),j['job'],flush=True)
    print('EVAL DONE',split,flush=True)

def row_results(rows,method,ci):return [(r,r['outcomes'][method][str(ci)]) for r in rows]
def missing_bounds(y,s,valid):
    return [auc(y,[v if ok else -1e6 if label==1 else 1e6 for label,v,ok in zip(y,s,valid)]),auc(y,[v if ok else 1e6 if label==1 else -1e6 for label,v,ok in zip(y,s,valid)])]
def metrics(rr):
    marks=[(r,x) for r,x in rr if r['arm']!='clean'];clean=[(r,x) for r,x in rr if r['arm']=='clean']
    y=[int(r['arm']=='bit1') for r,x in marks];s=[x['bit_score'] for r,x in marks];valid=[x['accepted'] for r,x in marks]
    py=[int(r['arm']!='clean') for r,x in rr];ps=[x.get('s_presence',x['absolute_presence']) for r,x in rr];pv=[x['accepted'] for r,x in rr]
    wrong=[a for r,x in rr for a in x['wrong_location_acceptance']]
    return dict(marked_attempts=len(marks),marked_received=sum(valid),marked_coverage=float(np.mean(valid)),clean_attempts=len(clean),clean_received=sum(x['accepted'] for r,x in clean),
        bit_auc_itt=auc(y,s),bit_auc_success=auc([a for a,b in zip(y,valid) if b],[a for a,b in zip(s,valid) if b]),bit_auc_missing_bounds=missing_bounds(y,s,valid),
        presence_auc_itt=auc(py,ps),presence_auc_success=auc([a for a,b in zip(py,pv) if b],[a for a,b in zip(ps,pv) if b]),presence_missing_bounds=missing_bounds(py,ps,pv),
        absolute_presence_auc_itt=auc(py,[x['absolute_presence'] for r,x in rr]),q_sync_only_presence_auc=auc(py,[x['q_sync'] for r,x in rr]),
        wrong_location_acceptance=float(np.mean(wrong)),wrong_location_count=sum(wrong),wrong_location_attempts=len(wrong),
        coefficient_se=distribution(x.get('coefficient_se') for r,x in rr if x['accepted']),information=distribution(x.get('r_phi') for r,x in rr if x['accepted']))

def presence_fit(rr):
    params={}
    for arm in ['clean','bit0','bit1']:
        subset=[x['bit_score'] for r,x in rr if r['arm']==arm and x['accepted']]
        params[arm]=dict(n=len(subset),mean=float(np.mean(subset)) if subset else 0.,variance=max(.15**2,float(np.var(subset))) if subset else 1.)
    return params

def presence_predict(x,params):
    if not x['accepted']:return 0.
    z=x['bit_score'];se=x.get('normalized_se') or 0
    def logpdf(arm):
        p=params[arm];variance=p['variance']+se*se
        return -.5*(math.log(2*np.pi*variance)+(z-p['mean'])**2/variance)
    return float(expit(logsumexp([logpdf('bit0'),logpdf('bit1')])-math.log(2)-logpdf('clean')))

def calibrate():
    check();rows=[read(OUT/'results'/f'{j["job"]}.json') for j in jobs() if j['split']=='Cal'];assert len(rows)==72
    decisions={};baseline=metrics(row_results(rows,'A0',0));limit=max(.1,baseline['wrong_location_acceptance']+.05)
    for method in METHODS:
        options=[]
        for ci,cfg in enumerate(configs(method)):
            rr=row_results(rows,method,ci);stats=metrics(rr)
            eligible=stats['marked_coverage']>=.7 and stats['bit_auc_itt']>=.8 and stats['wrong_location_acceptance']<=limit
            options.append(dict(config_index=ci,config=cfg,metrics=stats,eligible=eligible))
        eligible=[o for o in options if o['eligible']];selected=dict(max(eligible or options,key=lambda o:(o['metrics']['marked_coverage'],o['metrics']['bit_auc_itt'],-o['metrics']['wrong_location_acceptance'])))
        rr=row_results(rows,method,selected['config_index']);head=presence_fit(rr)
        for r,x in rr:x['s_presence']=presence_predict(x,head)
        clean_scores=[x['s_presence'] for r,x in rr if r['arm']=='clean' and x['accepted']]
        selected.update(all_configs=options,presence_model=head,presence_threshold95=float(np.percentile(clean_scores,95)) if clean_scores else 1.,calibrated_metrics=metrics(rr),qualified=bool(eligible))
        decisions[method]=selected
    pool=[method for method in ['A3','A4'] if decisions[method]['qualified']] or ['A3','A4']
    primary=max(pool,key=lambda mo:(decisions[mo]['metrics']['marked_coverage'],decisions[mo]['metrics']['bit_auc_itt'],-decisions[mo]['metrics']['wrong_location_acceptance']))
    frozen=dict(created_utc=now(),methods=decisions,primary=primary,wrong_location_acceptance_limit=limit,has_qualified_primary=decisions[primary]['qualified'],
        protected_sha256=read(OUT/'protocol.json')['protected_sha256'],protocol_sha256=sha(OUT/'protocol.json'),Cal_result_sha256={r['job']:sha(OUT/'results'/f'{r["job"]}.json') for r in rows},
        policy='Frozen before Dev receiver scores and before fresh content selection. Old Dev diagnostic only.')
    dump(OUT/'receiver_freeze.json',frozen);print('RECEIVER FROZEN',primary,'qualified',frozen['has_qualified_primary'],flush=True)

def summarize(fresh=False):
    decision=read(OUT/'receiver_freeze.json');folder=OUT/'fresh' if fresh else OUT
    rows=[read(folder/'results'/f'{j["job"]}.json') for j in jobs(fresh)]
    splits=['FreshTest'] if fresh else ['Cal','Dev_diagnostic'];summary={};flat=[];pairlist=[]
    for split in splits:
        subset=rows if fresh else [r for r in rows if r['split']==('Cal' if split=='Cal' else 'Dev')];summary[split]={}
        for method in METHODS:
            spec=decision['methods'][method];rr=row_results(subset,method,spec['config_index'])
            for r,x in rr:
                x['s_presence']=presence_predict(x,spec['presence_model'])
                flat.append(dict(job=r['job'],key=r['key'],content_id=r['content_id'],split=split,method=method,arm=r['arm'],view=r['view'],
                    c_hat=x.get('c_hat'),q_sync=x['q_sync'],s_presence=x['s_presence'],accepted=x['accepted'],bit_score=x['bit_score'],r_phi=x.get('r_phi'),coefficient_se=x.get('coefficient_se')))
            stats=metrics(rr);threshold=spec['presence_threshold95'];clean=[x for r,x in rr if r['arm']=='clean'];marked=[x for r,x in rr if r['arm']!='clean']
            stats.update(presence_threshold95=threshold,presence_fpr_itt=float(np.mean([x['accepted'] and x['s_presence']>threshold for x in clean])),
                presence_tpr_itt=float(np.mean([x['accepted'] and x['s_presence']>threshold for x in marked])))
            pairs=[]
            for key in sorted({r['key'] for r in subset}):
                for view in range(2):
                    zero=next(x for r,x in rr if r['key']==key and r['view']==view and r['arm']=='bit0');one=next(x for r,x in rr if r['key']==key and r['view']==view and r['arm']=='bit1')
                    valid=zero['accepted'] and one['accepted'];delta=one['c_hat']-zero['c_hat'] if valid else None
                    item=dict(key=key,view=view,method=method,split=split,both_received=valid,delta_out=delta,positive=bool(valid and delta>0));pairs.append(item);pairlist.append(item)
            repeat=[key for key in {p['key'] for p in pairs} if all(p['positive'] for p in pairs if p['key']==key)]
            stats.update(pairs=len(pairs),both_received_pairs=sum(p['both_received'] for p in pairs),positive_pairs=sum(p['positive'] for p in pairs),repeatable_contents=repeat,delta_out=distribution(p['delta_out'] for p in pairs))
            original={(r['key'],r['view'],r['arm']):x for r,x in row_results(subset,'A0',0)}
            recovered=[(r,x) for r,x in rr if r['arm']!='clean' and not original[(r['key'],r['view'],r['arm'])]['accepted'] and x['accepted']]
            stats['recovered_R0_failures']=len(recovered);stats['recovered_correct_bit']=sum((x['bit_score']>0)==(r['arm']=='bit1') for r,x in recovered)
            # Cluster intervals: never treat both views and messages as independent.
            ids=sorted({r['content_id'] for r in subset});rng=np.random.default_rng(2026111017);boot=[];pb=[]
            for _ in range(1000):
                sample=[(r,x) for cid in rng.choice(ids,len(ids),replace=True) for r,x in rr if r['content_id']==cid]
                mark=[(r,x) for r,x in sample if r['arm']!='clean'];boot.append(auc([int(r['arm']=='bit1') for r,x in mark],[x['bit_score'] for r,x in mark]));pb.append(auc([int(r['arm']!='clean') for r,x in sample],[x['s_presence'] for r,x in sample]))
            stats['bit_auc_cluster95']=interval(boot);stats['presence_auc_cluster95']=interval(pb);summary[split][method]=stats
    dump(folder/'summary.json',dict(primary=decision['primary'],results=summary,receiver_freeze_sha256=sha(OUT/'receiver_freeze.json'),
        note='Existing Dev is diagnostic, not independent receiver test. Erasures scored0 only for explicit ITT ranking; validity and confidence separately retained. Wrong-location control not semantic ground truth.'))
    dump(folder/'coefficient_observations.json',flat);dump(folder/'pair_responses.json',pairlist)
    with (folder/'coefficient_observations.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,list(flat[0]));writer.writeheader();writer.writerows(flat)
    print('SUMMARY',folder,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','soft','evaluate','calibrate','summarize']);a.add_argument('--split',default='Cal');a.add_argument('--fresh',action='store_true');args=a.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='soft':extract_soft(args.fresh)
    elif args.action=='evaluate':evaluate(args.split,args.fresh)
    elif args.action=='calibrate':calibrate()
    else:summarize(args.fresh)
