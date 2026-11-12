"""Same-PNG frozen-receiver comparison and residual-codebook diagnostics.

Fresh48 is now a diagnostic set. No true output correspondence is inferred
from message labels. Missing true-Oracle annotations remain explicitly absent.
"""
import curve_partial_st_stage4b as b4
from curve_broad_peak_code import fit,receive,basis
from curve_receiver_stage2c import geometry
from scipy.linalg import eigvalsh
from scipy.spatial import cKDTree
from scipy.special import logsumexp
import numpy as np
import cv2,os,json,time,shutil,itertools,argparse
from pathlib import Path
from PIL import Image

a4=b4.a4;ROOT=b4.ROOT;PRIOR=b4.OUT;PARENT=b4.PARENT
OUT=ROOT/'outputs/canny_st_identifiability_comparison_20261111'
read=b4.read;dump=b4.dump;digest=b4.digest;WORDS=b4.WORDS
SCENARIOS=[s for s in b4.SCENARIOS if s['name'] in ['none','random25_r0','random50_r0','center25','center50','ends25','ends50','left50','blocks50']]
PAIRS=list(itertools.combinations(range(4),2))

def freeze():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():return check()
    for d in ['source','measurements','metrics','blind_review','plots','panels','logs']:(OUT/d).mkdir(exist_ok=True)
    for n in ['compare_st_identifiability_4b.py']:shutil.copyfile(ROOT/n,OUT/'source'/n)
    dump(OUT/'protocol.json',dict(created_utc=b4.now(),source_sha256={str(ROOT/'compare_st_identifiability_4b.py'):digest(ROOT/'compare_st_identifiability_4b.py')},
        parents={str(folder/'final_artifacts_sha256.json'):digest(folder/'final_artifacts_sha256.json') for folder in [PRIOR,PARENT]},
        dataset='Exactly48 Fresh marked PNG from Stage4B, no generation or image changes. Same source, canonical orientation and four actual-input codewords. All48 denominator including erasure.',
        methods=dict(A4='Exact original receive Auto + exact original a4.infer and original4A Auto Cal model.',
            B4='Exact current4B EdgeDrop:none pool/gate/decode; also report prior ObsDrop:none reproduction and common-decoder ablations.',
            OracleMask_proxy='Existing4A Oracle primary ordered observations, refit and same4B gates/decoder; known tube only, NOT known-correct correspondence.',
            TrueOracle='Requires externally verified original-station -> real output-edge correspondence. No such annotation in existing4A/4B artifacts; never substitute message-guided curve selection.'),
        identity_control='Common4A decoder and common4B decoder for every geometry head in addition to native frozen pipelines; this distinguishes receiver change from score change.',
        information='Raw Phi^T W Phi and offset-projected Schur complement; eigenvalues, condition, generalized full-station eigenvalues, S/T energy coverage. No size normalization ambiguity: also average weighted Gram.',
        distances='All6 codeword pairs on observed t: nominal input, actual-input coefficients, and frozen channel output centers; raw weighted squared and RMS; also subtract weighted mean for unknown offset. No interpolation used.',
        hypotheses='Retain all R0/A3/A3_alternative fits, full candidate coords, signed best/runner-up distinct-geometry gap. No second candidate means unknown ambiguity, not proven uniqueness.',
        exploratory_set_receiver='Frozen4B gate plus Cal-selected minimum output-codeword offset-projected RMS distance. Retain distinct eligible geometry hypotheses within0.10 cost of bit-blind minimum. If decoded words disagree, return set/erasure; never pick a hypothesis because it matches the known word.',
        cal='Old4A six Cal contents + original4B32 Cal wrong-anchor controls. Fresh48 is reused diagnostic data, no new independent performance claim. New distance threshold freezes before extracting Fresh in this run.',
        scenarios=SCENARIOS,quality='Same RGB files; previous quality measurements remain valid, no new quality generation.',
        limitations=['Diffusion has no stored point-identity ground truth. OracleMask cannot prove true correspondence.','Nominal/Cal-channel distance predicts separability, not true edge identity.','Four legal words give no authentication redundancy.']))
    return check()

def check():
    p=read(OUT/'protocol.json')
    for f,h in p['source_sha256'].items():assert digest(f)==h,f
    for f,h in p['parents'].items():assert digest(f)==h,f
    return p

def safe_cond(g):
    e=np.linalg.eigvalsh(g)
    return float(e[-1]/e[0]) if e[0]>1e-12 else None

def residual_metrics(c,f):
    if not f or not f.get('fitted'):return None
    p,u,_,_=geometry(c['samples']);ii=np.array(f['indices'],int);w=np.array(f['weights'],float);phi=basis(u[ii]);total=float(w.sum())
    stations=np.unique(np.r_[np.arange(0,len(p),2),len(p)-1]).astype(int);full=basis(u[stations])
    mean=np.average(phi,axis=0,weights=w);z=phi-mean;g=phi.T@(w[:,None]*phi);schur=z.T@(w[:,None]*z)
    full_g=full.T@full;full_z=full-full.mean(0);ref=full_z.T@full_z
    eigen=np.linalg.eigvalsh(g);effective_eigen=np.linalg.eigvalsh(schur);generalized=eigvalsh(schur,ref)
    a=np.array(b4.channel_model()['A']);intercept=np.array(b4.channel_model()['intercept'])
    codebooks=dict(nominal=a4.AMPLITUDE*np.array([a4.SIGNS[word] for word in WORDS]),
       actual_input=np.array([a4.features(c,word,'Auto') for word in WORDS]))
    codebooks['channel_output']=codebooks['actual_input']@a.T+intercept
    distances={}
    for domain,coefs in codebooks.items():
        pairs=[]
        for i,j in PAIRS:
            delta=coefs[i]-coefs[j];d2=max(0.,float(delta@g@delta));dc2=max(0.,float(delta@schur@delta))
            pairs.append(dict(pair=[WORDS[i],WORDS[j]],d2=d2,rms=float(np.sqrt(d2/total)),offset_projected_d2=dc2,offset_projected_rms=float(np.sqrt(dc2/total))))
        distances[domain]=dict(pairs=pairs,min_d2=min(r['d2'] for r in pairs),min_rms=min(r['rms'] for r in pairs),
            min_offset_projected_d2=min(r['offset_projected_d2'] for r in pairs),min_offset_projected_rms=min(r['offset_projected_rms'] for r in pairs),
            weakest_pair=min(pairs,key=lambda r:r['offset_projected_d2'])['pair'])
    y=np.array(f['normal_positions']);v=y-(f['offset']+phi@np.array(f['coefficient']));noise=max(.15,1.4826*float(np.median(abs(v-np.median(v)))))
    return dict(samples=len(ii),fixed_stations=len(stations),sample_coverage=len(ii)/len(stations),weighted_sample_coverage=total/len(stations),span=float(np.ptp(u[ii])),
        ST_energy_coverage=(np.sum(w[:,None]*phi**2,axis=0)/np.maximum(np.sum(full**2,axis=0),1e-12)).tolist(),
        G=g.tolist(),G_eigenvalues=eigen.tolist(),G_condition=safe_cond(g),G_average=(g/total).tolist(),
        G_offset_projected=schur.tolist(),G_offset_projected_eigenvalues=effective_eigen.tolist(),G_offset_projected_condition=safe_cond(schur),
        relative_joint_information=generalized.tolist(),joint_info_min=float(generalized.min()),
        weighted_ST_correlation=float(schur[0,1]/np.sqrt(max(1e-20,schur[0,0]*schur[1,1]))),
        distances=distances,robust_residual_noise_px=noise,output_min_distance_over_noise=distances['channel_output']['min_offset_projected_rms']/noise)

def distinct(a,b):
    d=cKDTree(a['observed_xy']).query(b['observed_xy'])[0]
    return bool(np.median(d)>=1 or np.mean(d>2)>=.2)

def hypotheses(raw):
    pool=sorted(raw.get('pool',[]),key=lambda f:(f['cost'],f['branch']));unique=[]
    for f in pool:
        if all(distinct(f,q) for q in unique):unique.append(f)
    first=unique[0] if unique else None;second=unique[1] if len(unique)>1 else None
    return dict(all_count=len(pool),distinct_count=len(unique),best_branch=first['branch'] if first else None,
        runner_up_branch=second['branch'] if second else None,best_cost=first['cost'] if first else None,runner_up_cost=second['cost'] if second else None,
        signed_relative_gap=float((second['cost']-first['cost'])/(abs(second['cost'])+1e-8)) if second else None,
        absolute_gap=float(second['cost']-first['cost']) if second else None,
        near_equivalent_branches=[q['branch'] for q in unique if q['cost']<=first['cost']+.10] if first else [],
        no_runner_up_does_not_prove_uniqueness=second is None)

def as_gate_fit(source,f):
    if not f or not f.get('fitted'):return dict(fitted=False,accepted=False)
    out=dict(f);out.update(b4.information(source,f));out['association_ok']=True
    return out

def multi(c,raw,frozen,threshold):
    eligible=[]
    for q in raw.get('pool',[]):
        f=b4.gate_one(q,frozen['gate'])
        if not f['accepted']:continue
        m=residual_metrics(c,f)
        if m['distances']['channel_output']['min_offset_projected_rms']<threshold:continue
        eligible.append(f)
    if not eligible:return dict(accepted=False,word=None,word_set=[],reason='no_identifiable_hypothesis',branches=[])
    eligible.sort(key=lambda q:(q['cost'],q['branch']));unique=[]
    for q in eligible:
        if all(distinct(q,r) for r in unique):unique.append(q)
    retained=[q for q in unique if q['cost']<=unique[0]['cost']+.10]
    outputs=[b4.decode(c,q,frozen['channel_model']) for q in retained];words=sorted({r['word'] for r in outputs})
    return dict(accepted=len(words)==1,word=words[0] if len(words)==1 else None,word_set=words,
       reason='geometry_hypotheses_agree' if len(words)==1 else 'ambiguous_correspondence',branches=[q['branch'] for q in retained],
       hypotheses=[dict(branch=q['branch'],cost=q['cost'],word=r['word'],coefficient=q['coefficient']) for q,r in zip(retained,outputs)])

def calibrate():
    freeze()
    if (OUT/'distance_freeze.json').exists():return
    cv2.setNumThreads(1);frozen=read(PRIOR/'score_freeze.json');cc={c['key']:c for c in read(PARENT/'selection.json')};cal=[]
    for j in read(PARENT/'plan.json'):
        if j['split']!='Cal' or j['arm'] not in WORDS:continue
        data=read(PRIOR/'loss'/f'{j["job"]}.json')
        for s in SCENARIOS:
            raw=data['scenarios'][s['name']].get('EdgeDrop')
            if raw:cal.append((j,raw))
    controls=[];newcc={c['key']:c for c in read(PRIOR/'selection.json')}
    for path in (PRIOR/'controls').glob('*.json'):
        d=read(path)
        if d['split']!='Cal':continue
        key=d['job'].rsplit('_v',1)[0]
        controls.extend((newcc[key],r['raw']) for r in d['controls'])
    trials=[]
    for threshold in [0.,.1,.2,.3,.4,.5,.65,.8,1.]:
        correct=wrong=received=0
        for j,raw in cal:
            out=multi(cc[j['key']],raw,frozen,threshold);received+=out['accepted']
            correct+=bool(out['accepted'] and out['word']==j['arm']);wrong+=bool(out['accepted'] and out['word']!=j['arm'])
        false=sum(multi(c,raw,frozen,threshold)['accepted'] for c,raw in controls)
        trials.append(dict(threshold=threshold,attempts=len(cal),correct=correct,wrong=wrong,received=received,erasures=len(cal)-received,
            negative_accepts=false,negative_attempts=len(controls),eligible=false<=.1*len(controls),objective=correct-2*wrong))
    selected=max([r for r in trials if r['eligible']],key=lambda r:(r['objective'],-r['wrong'],r['threshold']))
    dump(OUT/'distance_freeze.json',dict(created_utc=b4.now(),protocol_sha256=digest(OUT/'protocol.json'),
        original_4b_score_sha256=digest(PRIOR/'score_freeze.json'),trials=trials,selected=selected,
        stage4b_gate=frozen['gate'],geometry_near_cost=.10,no_new_Fresh_measurement_before_freeze=not list((OUT/'measurements').glob('*.json')),
        scope='Cal-selected extra distance gate and hypothesis consensus; exploratory diagnostic, not independent validation.'))
    print('DISTANCE_FROZEN',selected,flush=True)

def blind_sheets(jobs,cc,measurements):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng=np.random.default_rng(2026111183);mapping=[]
    for cnum,c in enumerate([c for c in cc.values() if c['marked']]):
        selected=[j for j in jobs if j['key']==c['key']];selected=[selected[i] for i in rng.permutation(len(selected))]
        p,_,_,_=geometry(c['samples']);lo=np.maximum(0,np.floor(p.min(0)-22)).astype(int);hi=np.minimum(512,np.ceil(p.max(0)+23)).astype(int)
        fig,axes=plt.subplots(4,5,figsize=(18,10));source=np.array(Image.open(c['source_rgb']).convert('RGB'))
        for row in range(4):
            axes[row,0].imshow(source);axes[row,0].plot(p[:,0],p[:,1],color='cyan',lw=2);axes[row,0].set_xlim(lo[0],hi[0]);axes[row,0].set_ylim(hi[1],lo[1]);axes[row,0].set_title('Original source RGB / target contour')
        for col,j in enumerate(selected,1):
            tag=f'C{cnum:02d}X{col}';mapping.append(dict(tag=tag,job=j['job'],key=c['key']))
            rgb=np.array(Image.open(j['rgb']).convert('RGB'));m=measurements[j['job']];edge=cv2.Canny(rgb,100,200)
            axes[0,col].imshow(rgb);axes[0,col].set_title(tag+' RGB (no message label)')
            for row in [1,2,3]:axes[row,col].imshow(edge,cmap='gray',vmin=0,vmax=255)
            for row,branch in [(1,'R0'),(2,'A3'),(3,'A3_alternative')]:
                q=next((q for q in m['native']['Auto'].get('decisions',[]) if q['branch']==branch),None)
                axes[row,col].plot(p[:,0],p[:,1],color='#2475b0',alpha=.5,lw=1)
                if q:
                    xy=np.array(q['observed_xy']);axes[row,col].scatter(xy[:,0],xy[:,1],s=9,color='#24b85a');axes[row,col].set_title(branch+' observed vertices')
                else:axes[row,col].set_title(branch+' no proposal')
            for row in range(4):axes[row,col].set_xlim(lo[0],hi[0]);axes[row,col].set_ylim(hi[1],lo[1])
        for ax in axes.flat:ax.grid(alpha=.15);ax.tick_params(labelsize=7)
        fig.suptitle('Bit-blind correspondence review. Source at left; output columns randomly permuted. No decoder scores or codeword shapes shown.\nGreen vertices are candidates, not ground truth. Human/semantic review does not prove exact diffusion point identity.',fontsize=11)
        fig.tight_layout();fig.savefig(OUT/'blind_review'/f'content_{cnum:02d}.png',dpi=130);plt.close(fig)
    dump(OUT/'blind_review_mapping.json',mapping)
    dump(OUT/'true_oracle_annotations.template.json',dict(status='unannotated',schema='records: {tag, status=verified/absent/ambiguous, indices:[original canonical station indices], observed_xy:[[x,y]], weights:[...], reviewer, rationale}. Every observed point must be an actual output Canny pixel; no expected bit used.',
        records=[dict(tag=r['tag'],status='unannotated',indices=[],observed_xy=[],weights=[],reviewer=None,rationale=None) for r in mapping]))

def run():
    check();calibrate();cv2.setNumThreads(1);frozen=read(PRIOR/'score_freeze.json');threshold=read(OUT/'distance_freeze.json')['selected']['threshold']
    cc={c['key']:c for c in read(PRIOR/'selection.json')};jobs=[j for j in read(PRIOR/'plan.json') if j['arm'] in WORDS]
    assert len(jobs)==48;allrows=[];metrics=[];measurements={}
    for ix,j in enumerate(jobs):
        c=cc[j['key']];dest=OUT/'measurements'/f'{j["job"]}.json';rgb=np.array(Image.open(j['rgb']).convert('RGB'))
        if dest.exists():m=read(dest)
        else:
            native=receive(c['samples'],a4.prepare_target(rgb),'S',a4.THRESHOLDS,a4.GATE)
            raw=b4.rematch(c['samples'],rgb,j,b4.SCENARIOS[0])
            original=read(PRIOR/'loss'/f'{j["job"]}.json')['original']
            assert native['Auto']==original,('frozen4A did not reproduce original raw fit',j['job'])
            m=dict(job=j['job'],rgb_sha256=digest(j['rgb']),native=native,raw4b=raw,distance_freeze_sha256=digest(OUT/'distance_freeze.json'))
            dump(dest,m)
        measurements[j['job']]=m;assert m['rgb_sha256']==digest(j['rgb'])
        original4a=a4.infer(c,dict(Auto=m['native']['Auto']),frozen['channel_model'],'Auto')
        current4b=b4.select(m['raw4b'],frozen['gate'])
        oldraw=read(PRIOR/'loss'/f'{j["job"]}.json')['scenarios']['none']['ObsDrop'];previous4b=b4.select(oldraw,frozen['gate'])
        proxy_raw=as_gate_fit(c['samples'],m['native']['Oracle']);proxy=b4.gate_one(proxy_raw,frozen['gate'])
        geometry_methods=dict(A4=m['native']['Auto'],B4=current4b,B4_PreviousObsDrop=previous4b,OracleMask_proxy=proxy)
        for method,obs in geometry_methods.items():
            native=original4a if method=='A4' else b4.decode(c,obs,frozen['channel_model'],frozen['presence'])
            row=dict(job=j['job'],key=j['key'],arm=j['arm'],method=method,head_accepted=obs.get('accepted',False),
                native=native,common_A4=a4.infer(c,dict(Auto=obs),frozen['channel_model'],'Auto'),
                common_B4=b4.decode(c,obs,frozen['channel_model'],frozen['presence']))
            if method=='OracleMask_proxy':row['oracle_correspondence_verified']=False
            allrows.append(row)
        hypothesis_summary=hypotheses(m['raw4b']);mf=[]
        for f in m['raw4b'].get('pool',[]):
            mf.append(dict(branch=f['branch'],accepted4a=f['accepted'],cost=f['cost'],old_clipped_q_margin=f.get('q_margin'),
                accepted4b=b4.gate_one(f,frozen['gate'])['accepted'],metrics=residual_metrics(c,f)))
        metrics.append(dict(job=j['job'],key=j['key'],arm=j['arm'],scenario='none',kind='RematchPool',hypotheses=hypothesis_summary,candidates=mf,
            set_receiver=multi(c,m['raw4b'],frozen,threshold)))
        # All residual observations including baseline rejects are retained;
        # missing fits have explicit null metrics, not fabricated information.
        previous=read(PRIOR/'loss'/f'{j["job"]}.json')
        for s in SCENARIOS:
            raw=previous['scenarios'][s['name']]['ObsDrop']
            metrics.append(dict(job=j['job'],key=j['key'],arm=j['arm'],scenario=s['name'],kind='ObsDrop',
                accepted4b=b4.select(raw,frozen['gate'])['accepted'],metrics=residual_metrics(c,raw)))
            if s['name']!='none' and 'EdgeDrop' in previous['scenarios'][s['name']]:
                eraw=previous['scenarios'][s['name']]['EdgeDrop'];metrics.append(dict(job=j['job'],key=j['key'],arm=j['arm'],scenario=s['name'],kind='EdgeDropPool',
                    hypotheses=hypotheses(eraw),candidates=[dict(branch=f['branch'],cost=f['cost'],metrics=residual_metrics(c,f)) for f in eraw.get('pool',[])],
                    set_receiver=multi(c,eraw,frozen,threshold)))
        print('COMPARE',ix+1,'/48',j['job'],flush=True)
    dump(OUT/'comparison.json',allrows);dump(OUT/'identifiability.json',metrics)
    summary={}
    for method in geometry_methods:
        rr=[r for r in allrows if r['method']==method];summary[method]={}
        for decoder in ['native','common_A4','common_B4']:
            received=[r for r in rr if r[decoder]['accepted']];correct=sum(r[decoder]['word']==r['arm'] for r in received)
            summary[method][decoder]=dict(attempts=len(rr),received=len(received),correct=correct,wrong=len(received)-correct,erasures=len(rr)-len(received),full_accuracy=correct/len(rr))
    summary['TrueOracle']=dict(status='not_available_without_verified_correspondences',attempts=48,correct=None,wrong=None,erasures=None,
        not_replaced_by_proxy=True)
    dump(OUT/'summary.json',summary)
    blind_sheets(jobs,cc,measurements)
    print('COMPARISON_COMPLETE',summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['calibrate','run']);args=p.parse_args();globals()[args.action]()
