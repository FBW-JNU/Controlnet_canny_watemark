"""Paired Cal experiment: source-only versus clean-probe-stable S/T carriers."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/st_carrier_mpl')
from pathlib import Path
import argparse, collections, html, json, shutil, time
import numpy as np
import cv2
from PIL import Image, ImageDraw
import canny_st_amplitude_cal_scan as old
import curve_reference_ablation_stage4c0 as c0
import curve_partial_st_stage4b as b4
from scipy.stats import rankdata

ROOT = old.ROOT
OUT = ROOT / 'outputs/canny_st_stable_carrier_compare_20261111'
read, dump, digest = old.read, old.dump, old.digest
WORDS = b4.WORDS
METHODS = ['SourceRef', 'R2_tube', 'R2_bbox']
POLICIES = ['source_only', 'clean_stable']
PROBE_PREFIX = [
    'A sharp natural-color photograph showing the following scene: ',
    'A detailed pencil drawing showing the following scene: ',
    'An oil painting showing the following scene: ',
]
PROBE_SEEDS = [2026115801, 2026115802]
EVAL_SEEDS = [2026115901, 2026115902]


def setup():
    OUT.mkdir(exist_ok=True)
    for name in ['source', 'probes', 'records', 'noise', 'conditions', 'masks',
                 'candidates', 'selection', 'rgb', 'features', 'quality', 'plots', 'panels', 'logs']:
        (OUT / name).mkdir(exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        return check()
    scripts = ['canny_st_stable_carrier_compare.py', 'curve_single_2bit_stage4a.py',
               'curve_partial_st_stage4b.py', 'curve_broad_peak_code.py',
               'curve_receiver_stage2c.py', 'curve_receiver_stage2b.py',
               'curve_qim_closed_loop_v2.py', 'curve_reference_ablation_stage4c0.py',
               'broad_peak_runtime.py']
    for name in scripts:
        shutil.copyfile(ROOT / name, OUT / 'source' / name)
    p = dict(created_utc=b4.now(), scripts={str(ROOT/n): digest(ROOT/n) for n in scripts},
             parents={str(x/'final_artifacts_sha256.json'): digest(x/'final_artifacts_sha256.json')
                      for x in [b4.PARENT, b4.OUT, c0.OUT, old.OUT]},
             data='Exactly six old4A Cal contents; no content replacement; exploratory paired diagnosis, not independent-content validation.',
             probe=dict(prefixes=PROBE_PREFIX, seeds=PROBE_SEEDS, clean_per_content=6,
                        matching='Exact frozen Stage1 strict graph matcher, original thresholds.',
                        eligibility='At least5/6 strict matches; at least1/2 in each of3 prompts; matched median symmetric distance<=3px; anchor-good-view rate>=5/6.',
                        ranking='Same descending source length / curve index as baseline, then unchanged nine-state input admission. No message/output-coded PNG in selection.'),
             evaluation=dict(prefixes=b4.a4.PREFIX, seeds=EVAL_SEEDS, views=2,
                             arms=['clean']+WORDS, attempts_per_policy=48,
                             isolation='Probe prompts and seeds disjoint from evaluation. Freeze carrier choices/conditions before any evaluation RGB. No holdout reselection.'),
             encoder=read(b4.PARENT/'protocol.json')['encoder'],
             receivers=dict(SourceRef='Frozen4B R0/A3 candidate pool, joint Huber fit, info gate and old channel/presence; all4 source raster centers, never true-message-guided selection.',
                            self_reference='Frozen4C0 tube and bbox extraction, fit gates and public Cal models. No retraining or recalibration.',
                            no_carrier='Four marked attempts remain erasures. Render identity clean conditions for quality bookkeeping, also report quality+correct joint count.'),
             generation=read(old.OUT/'protocol.json')['generation'],
             quality=read(old.OUT/'protocol.json')['quality'],
             limitations=['Six previously developed-on Cal contents, not independent validation.',
                          'Clean geometry stability does not guarantee S/T transfer.',
                          'Matched-pixel identity after diffusion is not known; no true Oracle claim.',
                          'Two-bit all-legal codebook has no authentication redundancy.',
                          'Two evaluation views cannot separate prompt effects from seed effects.'])
    dump(path, p)
    return check()


def check():
    p = read(OUT/'protocol.json')
    for name, h in {**p['scripts'], **p['parents']}.items():
        assert digest(name) == h, name
    return p


def prepare():
    setup()
    if (OUT/'probe_plan.json').exists():
        return
    contents = read(old.OUT/'selection.json')
    assert len(contents) == 6
    plan = []
    for c in contents:
        for pi, prefix in enumerate(PROBE_PREFIX):
            for si, seed in enumerate(PROBE_SEEDS):
                job = f'{c["content_id"]:012d}_probe_p{pi}_s{si}'
                plan.append(dict(job=job, content_id=c['content_id'], phase='probe', arm='clean',
                                 prompt_index=pi, seed_index=si, prompt=prefix+c['caption'], seed=seed,
                                 hint=c['states']['clean']['hint'], hint_sha256=digest(c['states']['clean']['hint']),
                                 noise=str(OUT/'noise'/f'probe_{seed}.npy'), rgb=str(OUT/'probes'/f'{job}.png')))
    dump(OUT/'contents.json', contents)
    dump(OUT/'probe_plan.json', plan)
    dump(OUT/'probe_freeze.json', dict(created_utc=b4.now(), plan_sha256=digest(OUT/'probe_plan.json'),
         contents_sha256=digest(OUT/'contents.json'), protocol_sha256=digest(OUT/'protocol.json')))
    print('PREPARED', len(plan), 'probe RGB', flush=True)


def worker(phase):
    p = check()
    if phase == 'eval':
        assert (OUT/'selection_freeze.json').exists()
        for name, h in read(OUT/'selection_freeze.json')['protected'].items():
            assert digest(name) == h, name
    import torch
    from cldm.ddim_hacked import DDIMSampler
    from controlnet_robin.pipeline import ControlNetROBINPipeline
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1' and torch.cuda.is_available()
    gen = p['generation']
    assert digest(ROOT/gen['checkpoint']) == gen['checkpoint_sha256']
    for name, h in read(b4.a4.S1/'protocol.json')['frozen_model_code'].items():
        assert digest(name) == h
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cv2.setNumThreads(1)
    class DeviceSampler(DDIMSampler):
        def register_buffer(self, name, attr):
            setattr(self, name, attr.to(self.model.device) if isinstance(attr, torch.Tensor) else attr)
    pipe = ControlNetROBINPipeline(str(ROOT/gen['model_config']), str(ROOT/gen['checkpoint']), 'cuda')
    pipe.model.requires_grad_(False).eval()
    pipe.model.cond_stage_model.device = 'cuda'
    pipe.ddim = DeviceSampler(pipe.model)
    pipe.make_schedule(gen['steps'], eta=gen['eta'])
    pipe.model.control_scales = [gen['control_scale']]*13

    def generate(j):
        assert digest(j['hint']) == j['hint_sha256']
        noise = torch.randn((1,4,64,64), generator=torch.Generator(device='cpu').manual_seed(j['seed']))
        npy = Path(j['noise'])
        if npy.exists():
            assert np.array_equal(noise.numpy(), np.load(npy))
        else:
            np.save(npy, noise.numpy())
        cond, uncond = pipe.get_conditioning(j['prompt'], np.array(Image.open(j['hint']).convert('RGB')), '', 512, False)
        latent = noise.clone().cuda()
        with torch.inference_mode():
            for step in range(pipe.total_steps):
                latent = pipe.ddim_step(latent, step, cond, uncond, gen['guidance'])
            rgb = ((pipe.model.decode_first_stage(latent)+1)/2).clamp(0,1)
        return (rgb[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')

    if not (OUT/'canary.json').exists():
        j = next(j for j in read(b4.PARENT/'plan.json') if j['split']=='Cal' and j['arm']=='clean')
        image = generate(j)
        assert np.array_equal(image, np.array(Image.open(j['rgb']).convert('RGB')))
        dump(OUT/'canary.json', dict(created_utc=b4.now(), pixels_identical=True, reference=j['rgb'], reference_sha256=digest(j['rgb'])))
    plan = read(OUT/f'{phase}_plan.json')
    for i, j in enumerate(plan):
        record = OUT/'records'/f'{j["job"]}.json'
        if record.exists():
            assert digest(j['rgb']) == read(record)['rgb_sha256']
            continue
        start = time.monotonic()
        Image.fromarray(generate(j)).save(j['rgb'])
        dump(record, dict(**j, rgb_sha256=digest(j['rgb']), noise_sha256=digest(j['noise']),
                          checkpoint_sha256=gen['checkpoint_sha256'], protocol_sha256=digest(OUT/'protocol.json'),
                          selection_freeze_sha256=digest(OUT/'selection_freeze.json') if phase=='eval' else None,
                          elapsed_seconds=time.monotonic()-start))
        print('GEN', phase, i+1, '/', len(plan), j['job'], round(time.monotonic()-start,2), flush=True)
    dump(OUT/f'{phase}_generation_status.json', dict(status='complete', unique_rgb=len(plan)))


def carrier(c, curve, edge):
    """Exact old source-only admission; no output image or output label enters."""
    key = f'{c["content_id"]:012d}_c{curve["curve_index"]:04d}'
    dest = OUT/'candidates'/f'{key}_input.json'
    if dest.exists():
        return read(dest)
    mask = b4.a4.mask_for(edge, curve['samples'], c['roi'])
    images, obs, admission = b4.a4.input_check(edge, curve['samples'], mask)
    mp = OUT/'masks'/f'{key}.png'
    Image.fromarray(mask.astype('uint8')*255).save(mp)
    states = {}
    for arm, image in images.items():
        path = OUT/'conditions'/f'{key}_{arm}.png'
        Image.fromarray(image).save(path)
        states[arm] = dict(hint=str(path), hint_sha256=digest(path), measurements=obs[arm])
    result = dict(content_id=c['content_id'], key=key, samples=curve['samples'], curve_index=curve['curve_index'],
                  arc_length_px=curve['arc_length_px'], mask=str(mp), admission=admission, states=states,
                  caption=c['caption'], roi=c['roi'], source_rgb=c['source_rgb'])
    dump(dest, result)
    return result


def select():
    check()
    if (OUT/'selection_freeze.json').exists():
        return
    assert not (OUT/'eval_generation_status.json').exists()
    assert not list((OUT/'rgb').glob('*.png'))
    cv2.setNumThreads(1)
    m = b4.a4.local_matcher()
    contents = read(OUT/'contents.json')
    probes = read(OUT/'probe_plan.json')
    choices = []
    for c in contents:
        cid = c['content_id']
        dest = OUT/'selection'/f'{cid:012d}.json'
        if dest.exists():
            choices.append(read(dest)); continue
        edge = np.array(Image.open(c['states']['clean']['hint']).convert('L'))
        valid = np.zeros_like(edge); x0,y0,x1,y1 = c['roi']; valid[y0:y1,x0:x1] = edge[y0:y1,x0:x1]
        curves = sorted(m.source_curves(valid,b4.a4.THRESHOLDS)[0], key=lambda r:(-r['arc_length_px'],r['curve_index']))
        assert curves[c['curve_index']]['curve_index'] == c['curve_index']
        matching = {q['curve_index']:[] for q in curves}
        for j in [j for j in probes if j['content_id']==cid]:
            assert digest(j['rgb']) == read(OUT/'records'/f'{j["job"]}.json')['rgb_sha256']
            target = b4.a4.prepare_target(np.array(Image.open(j['rgb']).convert('RGB')))
            for curve in curves:
                r = m.curve_match(curve['samples'],target,b4.a4.THRESHOLDS)
                matching[curve['curve_index']].append(dict(**r, prompt_index=j['prompt_index'], job=j['job']))
        stats = []
        for curve in curves:
            rs = matching[curve['curve_index']]
            success = [r for r in rs if r['matched']]
            counts = [sum(r['matched'] for r in rs if r['prompt_index']==p) for p in range(3)]
            anchor = sum(r.get('anchor_good_fraction',0)>=.8 for r in rs)
            distance = float(np.median([r['symmetric_median_px'] for r in success])) if success else None
            eligible = len(success)>=5 and min(counts)>=1 and anchor>=5 and distance<=3
            stats.append(dict(curve_index=curve['curve_index'], arc_length_px=curve['arc_length_px'],
                              matched=len(success), R=len(success)/6, per_prompt_matched=counts,
                              anchor_good_views=anchor, median_symmetric_distance=distance,
                              stable_eligible=bool(eligible), views=rs))
        byid = {r['curve_index']:r for r in stats}
        dump(OUT/'candidates'/f'{cid:012d}_probe_matches.json', stats)
        selected = {}; tried = {}
        for policy in POLICIES:
            tried[policy] = []
            for curve in curves:
                if policy=='clean_stable' and not byid[curve['curve_index']]['stable_eligible']:
                    continue
                candidate = carrier(c,curve,edge)
                tried[policy].append(dict(curve_index=curve['curve_index'],admissible=candidate['admission']['admissible']))
                if candidate['admission']['admissible']:
                    selected[policy] = candidate
                    break
            else:
                selected[policy] = None
        # Baseline must exactly reproduce the existing source-only choice, not a new favorable choice.
        assert selected['source_only'] is not None and selected['source_only']['key']==c['key']
        for arm in ['clean']+WORDS:
            assert np.array_equal(np.array(Image.open(selected['source_only']['states'][arm]['hint'])),
                                  np.array(Image.open(c['states'][arm]['hint'])))
        result = dict(content_id=cid, source_candidates=len(curves), stable_candidates=sum(r['stable_eligible'] for r in stats),
                      carriers=selected,tried=tried,
                      selected_probe={policy:byid[q['curve_index']] if q is not None else None for policy,q in selected.items()})
        dump(dest,result); choices.append(result)
        print('SELECT',cid,'stable',result['stable_candidates'],'/',len(curves),
              {p:q['key'] if q else None for p,q in selected.items()},flush=True)
    evalplan = []; logical = []; unique = {}
    for c in contents:
        choice = next(q for q in choices if q['content_id']==c['content_id'])
        for view, seed in enumerate(EVAL_SEEDS):
            prompt = b4.a4.PREFIX[view]+c['caption']
            cleanjob = f'{c["content_id"]:012d}_eval_v{view}_clean'
            cleanrgb = str(OUT/'rgb'/f'{cleanjob}.png')
            for policy in POLICIES:
                q = choice['carriers'][policy]
                for arm in ['clean']+WORDS:
                    state = q['states'][arm] if q else c['states']['clean']
                    identity = (c['content_id'],view,state['hint_sha256'])
                    if identity not in unique:
                        job = cleanjob if arm=='clean' or q is None else f'{q["key"]}_eval_v{view}_{arm}'
                        j = dict(job=job,content_id=c['content_id'],phase='eval',view=view,arm=arm,
                                 prompt=prompt,seed=seed,hint=state['hint'],hint_sha256=state['hint_sha256'],
                                 noise=str(OUT/'noise'/f'eval_{seed}.npy'),rgb=str(OUT/'rgb'/f'{job}.png'),clean_rgb=cleanrgb)
                        unique[identity] = j; evalplan.append(j)
                    j = unique[identity]
                    logical.append(dict(policy=policy,content_id=c['content_id'],view=view,arm=arm,
                                        job=j['job'],key=q['key'] if q else None,rgb=j['rgb'],hint=j['hint'],
                                        clean_rgb=cleanrgb,carrier_available=q is not None))
    dump(OUT/'selection.json',choices); dump(OUT/'eval_plan.json',evalplan); dump(OUT/'logical_plan.json',logical)
    files=[OUT/n for n in ['selection.json','eval_plan.json','logical_plan.json','probe_plan.json','contents.json']]
    files += list((OUT/'selection').glob('*.json'))+list((OUT/'candidates').glob('*.json'))
    files += list((OUT/'masks').glob('*.png'))+list((OUT/'conditions').glob('*.png'))
    files += list((OUT/'probes').glob('*.png'))+[OUT/'records'/f'{j["job"]}.json' for j in probes]
    dump(OUT/'selection_freeze.json',dict(created_utc=b4.now(),no_eval_rgb_before_freeze=True,
                                        eval_unique_rgb=len(evalplan),logical_marked_attempts_per_policy=48,
                                        protected={str(p):digest(p) for p in files}))
    print('FROZEN',len(evalplan),'unique evaluation RGB;96 logical marked attempts',flush=True)


def extract():
    check(); cv2.setNumThreads(1)
    frozen = read(b4.OUT/'score_freeze.json'); public = read(c0.OUT/'score_freeze.json')
    choices = {r['content_id']:r for r in read(OUT/'selection.json')}
    logical = read(OUT/'logical_plan.json')
    cache = {}
    for i,j in enumerate(logical):
        path = OUT/'features'/f'{j["policy"]}_{j["content_id"]:012d}_v{j["view"]}_{j["arm"]}.json'
        if path.exists():
            continue
        q = choices[j['content_id']]['carriers'][j['policy']]
        cachekey = (j['key'],j['job'])
        if cachekey not in cache:
            rgb = np.array(Image.open(j['rgb']).convert('RGB'))
            assert digest(j['rgb'])==read(OUT/'records'/f'{j["job"]}.json')['rgb_sha256']
            if q is None:
                obs = dict(accepted=False,reason='no_stable_carrier')
                heads = {method:dict(accepted=False,word=None,score=-1e6,posterior_max=0.) for method in METHODS}
                raw = {}; pp = {}; strict = dict(matched=False)
            else:
                raw = b4.rematch(q['samples'],rgb,j,b4.SCENARIOS[0])
                obs = b4.select(raw,frozen['gate'])
                decoded = b4.decode(q,obs,frozen['channel_model'],frozen['presence'])
                heads = dict(SourceRef=dict(**decoded,score=decoded['presence_score']))
                mask = np.array(Image.open(q['mask']).convert('L'))>0
                yy,xx=np.nonzero(mask); box=np.zeros_like(mask);box[yy.min():yy.max()+1,xx.min():xx.max()+1]=True
                pp = dict(R2_tube=c0.extract_rgb(rgb,mask),R2_bbox=c0.extract_rgb(rgb,box))
                for method in ['R2_tube','R2_bbox']:
                    conf = public[method]['selected']; f=pp[method]['fits'][str(conf['degree'])]
                    ok=c0.eligible(f,conf['residual'],conf['ci95'])
                    heads[method]=c0.classify(f['coefficient'] if ok else None,f.get('covariance',np.eye(2).tolist()),public[method]['model'])
                strict = b4.a4.local_matcher().curve_match(q['samples'],b4.a4.prepare_target(rgb),b4.a4.THRESHOLDS)
            cache[cachekey]=dict(raw=raw,observation=obs,public=pp,heads=heads,strict_match=strict)
        dump(path,dict(**j,**cache[cachekey],rgb_sha256=digest(j['rgb'])))
        print('EXTRACT',i+1,'/',len(logical),j['policy'],j['arm'],flush=True)


def quality():
    check()
    import torch, lpips
    from carrier_screen.common import pixel_metrics
    torch.set_num_threads(3); cv2.setNumThreads(1)
    net = None; limits = read(OUT/'protocol.json')['quality']; cache = {}
    for j in read(OUT/'logical_plan.json'):
        if j['arm']=='clean': continue
        dest=OUT/'quality'/f'{j["policy"]}_{j["content_id"]:012d}_v{j["view"]}_{j["arm"]}.json'
        if dest.exists():continue
        if j['job'] not in cache:
            if net is None:net=lpips.LPIPS(net='alex',version='0.1').eval().requires_grad_(False).cpu()
            clean=np.array(Image.open(j['clean_rgb']).convert('RGB'));rgb=np.array(Image.open(j['rgb']).convert('RGB'))
            q=pixel_metrics(clean,rgb)
            a=torch.from_numpy(clean).permute(2,0,1)[None].float()/127.5-1
            b=torch.from_numpy(rgb).permute(2,0,1)[None].float()/127.5-1
            with torch.inference_mode():q['lpips_alex']=float(net(a,b))
            gates=dict(psnr=bool(q['psnr_infinite'] or q['psnr_db']>=limits['psnr_db_min']),ssim=q['ssim']>=limits['ssim_min'],
                       lpips=q['lpips_alex']<=limits['lpips_alex_max'],edge_f1=q['edge_f1_3x3']>=limits['rgb_pair_edge_f1_3x3_min'])
            cache[j['job']]=dict(quality=q,gates=gates,pass_all=all(gates.values()))
        dump(dest,dict(**j,**cache[j['job']],rgb_sha256=digest(j['rgb']),clean_rgb_sha256=digest(j['clean_rgb'])))
    print('QUALITY_COMPLETE',flush=True)


def auc(pos,neg):
    ranks=rankdata(np.r_[pos,neg])
    return float((sum(ranks[:len(pos)])-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))


def analyze():
    check()
    rows=[read(p) for p in sorted((OUT/'features').glob('*.json'))]
    qualities=[read(p) for p in sorted((OUT/'quality').glob('*.json'))]
    assert len(rows)==120 and len(qualities)==96
    summary={}; percontent=[]
    for policy in POLICIES:
        ss={}
        marked=[r for r in rows if r['policy']==policy and r['arm']!='clean']
        clean=[r for r in rows if r['policy']==policy and r['arm']=='clean']
        assert len(marked)==48 and len(clean)==12
        for method in METHODS:
            ok=[r for r in marked if r['heads'][method]['accepted']]
            correct=sum(r['heads'][method]['word']==r['arm'] for r in ok)
            threshold=read(b4.OUT/'score_freeze.json')['presence']['threshold95'] if method=='SourceRef' else read(c0.OUT/'score_freeze.json')[method]['model']['presence_threshold95']
            ss[method]=dict(attempts=48,received=len(ok),correct=correct,wrong=len(ok)-correct,erasures=48-len(ok),full_accuracy=correct/48,
                            conditional_accuracy=correct/len(ok) if ok else None,
                            high_confidence_wrong=sum(r['heads'][method].get('high_confidence',False) and r['heads'][method]['word']!=r['arm'] for r in ok),
                            presence_AUC=auc([r['heads'][method]['score'] for r in marked],[r['heads'][method]['score'] for r in clean]),
                            frozen_threshold=threshold,marked_presence_accepts=sum(r['heads'][method]['score']>threshold for r in marked),
                            clean_presence_accepts=sum(r['heads'][method]['score']>threshold for r in clean),clean_attempts=12)
            for cid in sorted({r['content_id'] for r in rows}):
                subset=[r for r in marked if r['content_id']==cid]; received=[r for r in subset if r['heads'][method]['accepted']]
                percontent.append(dict(content_id=cid,policy=policy,method=method,attempts=8,received=len(received),
                                       correct=sum(r['heads'][method]['word']==r['arm'] for r in received),wrong=sum(r['heads'][method]['word']!=r['arm'] for r in received)))
        qq=[r for r in qualities if r['policy']==policy]
        correct_jobs={(r['content_id'],r['view'],r['arm']) for r in marked if r['heads']['SourceRef']['accepted'] and r['heads']['SourceRef']['word']==r['arm']}
        ss['quality']=dict(attempts=48,joint_pass=sum(r['pass_all'] for r in qq),
                           correct_and_quality_pass=sum(r['pass_all'] and (r['content_id'],r['view'],r['arm']) in correct_jobs for r in qq),
                           failures={key:sum(not r['gates'][key] for r in qq) for key in qq[0]['gates']},
                           medians={key:float(np.median([r['quality'][key] for r in qq])) for key in ['psnr_db','ssim','lpips_alex','edge_f1_3x3']})
        ss['strict_clean']=dict(matched=sum(r['strict_match']['matched'] for r in clean),attempts=12)
        ss['strict_marked']=dict(matched=sum(r['strict_match']['matched'] for r in marked),attempts=48)
        summary[policy]=ss
    ids=sorted({r['content_id'] for r in rows});rng=np.random.default_rng(2026115911);idx=rng.integers(0,6,(10000,6));paired=[]
    for method in METHODS:
        delta=np.array([(next(r['correct'] for r in percontent if r['content_id']==cid and r['policy']=='clean_stable' and r['method']==method)-next(r['correct'] for r in percontent if r['content_id']==cid and r['policy']=='source_only' and r['method']==method))/8 for cid in ids])
        paired.append(dict(method=method,accuracy_gain=float(delta.mean()),content_bootstrap95=np.quantile(delta[idx].mean(1),[.025,.975]).tolist(),
                           contents_improved=int(sum(delta>0)),contents_worse=int(sum(delta<0)),contents_equal=int(sum(delta==0))))
    dump(OUT/'summary.json',summary);dump(OUT/'per_content.json',percontent);dump(OUT/'paired_bootstrap.json',paired)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def audit():
    check(); counters=collections.Counter()
    freeze=read(OUT/'selection_freeze.json')
    for name,h in freeze['protected'].items():assert digest(name)==h,name;counters['protected_files']+=1
    probe=read(OUT/'probe_plan.json');evaluation=read(OUT/'eval_plan.json')
    assert len(probe)==36 and not set(j['prompt'] for j in probe)&set(j['prompt'] for j in evaluation)
    assert not set(j['seed'] for j in probe)&set(j['seed'] for j in evaluation)
    assert read(OUT/'canary.json')['pixels_identical']
    for j in probe+evaluation:
        r=read(OUT/'records'/f'{j["job"]}.json')
        assert digest(j['rgb'])==r['rgb_sha256'] and digest(j['hint'])==j['hint_sha256'] and digest(j['noise'])==r['noise_sha256']
        if j['phase']=='eval':assert r['selection_freeze_sha256']==digest(OUT/'selection_freeze.json')
        counters['RGB_records']+=1
    selection={r['content_id']:r for r in read(OUT/'selection.json')}
    for r in selection.values():
        for policy,q in r['carriers'].items():
            if q is None:continue
            edge=np.array(Image.open(q['states']['clean']['hint']).convert('L'));mask=np.array(Image.open(q['mask']).convert('L'))>0
            for arm in ['clean']+WORDS:
                actual=np.array(Image.open(q['states'][arm]['hint']).convert('L'))
                assert np.array_equal(actual,b4.a4.render(edge,q['samples'],mask,b4.a4.SIGNS[arm]))
                assert set(np.unique(actual))<={0,255} and np.array_equal(actual[~mask],edge[~mask])
                counters['Canny_states_reproduced']+=1
            if policy=='clean_stable':assert r['selected_probe'][policy]['stable_eligible']
    frozen=read(b4.OUT/'score_freeze.json')
    for path in (OUT/'features').glob('*.json'):
        r=read(path);assert digest(r['rgb'])==r['rgb_sha256']
        q=selection[r['content_id']]['carriers'][r['policy']]
        if q:
            obs=b4.select(r['raw'],frozen['gate']);assert obs==r['observation']
            decoded=b4.decode(q,obs,frozen['channel_model'],frozen['presence'])
            assert dict(**decoded,score=decoded['presence_score'])==r['heads']['SourceRef']
            edge=cv2.Canny(np.array(Image.open(r['rgb']).convert('RGB')),100,200)
            for method in ['R2_tube','R2_bbox']:
                xy=r['public'][method]['selected_path']
                if xy is not None:
                    xy=np.array(xy);assert np.all(edge[xy[:,1].astype(int),xy[:,0].astype(int)]>0)
                    counters['real_edge_pixels']+=len(xy)
        counters['decisions_replayed']+=1
    for folder in [b4.PARENT,b4.OUT,c0.OUT,old.OUT]:
        for rel,h in read(folder/'final_artifacts_sha256.json').items():assert digest(folder/rel)==h;counters['unchanged_parent_files']+=1
    dump(OUT/'audit.json',dict(status='passed',counts=dict(counters),probe_eval_prompt_seed_disjoint=True,
                              choices_frozen_before_eval=True,attempts_per_policy=48,no_receiver_retuning=True))
    print('AUDIT_PASSED',dict(counters),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','probe','select','eval','extract','quality','analyze','audit'])
    args=parser.parse_args()
    if args.action in ['probe','eval']:worker(args.action)
    else:globals()[args.action]()
