"""Cal-only S/T peak-displacement scan, paired prompts/seeds, frozen matchers."""
import curve_reference_ablation_stage4c0 as c0
import curve_partial_st_stage4b as b4
from curve_qim_closed_loop_v2 import RasterWarp
from curve_broad_peak_code import basis,fit
from pathlib import Path
from PIL import Image
import numpy as np
import cv2,os,json,time,argparse,shutil

ROOT=b4.ROOT;PARENT=b4.PARENT;C0=c0.OUT
OUT=ROOT/'outputs/canny_st_amplitude_cal_scan_20261111'
read=b4.read;dump=b4.dump;digest=b4.digest;WORDS=b4.WORDS
PEAKS=[1.5,2.,3.,4.]
def tag(x):return 'p'+str(x).replace('.','_')


def freeze():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():return check()
    for d in ['source','conditions','edit_masks','input','rgb','records','features','quality','logs','plots','panels','metrics','canary']:(OUT/d).mkdir(exist_ok=True)
    names=['canny_st_amplitude_cal_scan.py','curve_reference_ablation_stage4c0.py','curve_partial_st_stage4b.py','curve_single_2bit_stage4a.py','curve_broad_peak_code.py','curve_qim_closed_loop_v2.py']
    for name in names:shutil.copyfile(ROOT/name,OUT/'source'/name)
    parent=read(PARENT/'protocol.json')
    dump(OUT/'protocol.json',dict(created_utc=b4.now(),sources={str(ROOT/n):digest(ROOT/n) for n in names},
       parents={str(f/'final_artifacts_sha256.json'):digest(f/'final_artifacts_sha256.json') for f in [PARENT,b4.OUT,C0]},
       peaks_px=PEAKS,encoder='Same S/T basis, four sign words, original source curve, original normals, one combined warp and one integer rasterization. Coefficient amplitude=peak/old PEAK normalization.',
       masks='Single amplitude-independent edit mask: old tube dilated3px and clipped to original ROI. Needed to avoid4px movements hitting old4px tube boundary. Every1.5px raster must reproduce old PNG exactly. Receiver tube and bbox masks stay original, unchanged at every strength.',
       data='Exactly six original4A Cal contents, two original prompt/seed views.48 marked per peak. Reuse12 clean +48 peak1.5 RGB after hashes/provenance.144 new RGB; no Eval/Fresh marked used.',
       calibration='Additional32 old4B Cal clean are threshold-only. Primary classification: leave one of six marked contents out (both views and all words); fit fixed-rule Gaussian centers/covariance to other5 contents at that peak. Clean density fit to ten clean views of those5. Thresholds use separate32 clean scores, all failures included.',
       detection='Nominal5%/10% FPR thresholds: order statistic k=ceil((n+1)*(1-alpha)), strict score>threshold; k>n=>infinity. Report held-out12 clean realized false accepts and all48 marked TPR, no low-FPR certification.',
       matchers='Freeze4B rematch/gate for SourceRef. Freeze4C0 R2_tube/R2_bbox extraction, degree, residual and CI gates. No amplitude-specific geometry tuning. New public score centers adapt to known strength by identical LOCO rule, not held-out labels.',
       quality=parent['quality'],generation=parent['generation'],canary='Before new144 generation, independently reproduce one clean and one1.5 marked RGB with identical sampler/noise settings; require identical decoded PNG pixels.',
       input_closure='Report source-known moved actual-pixel joint coefficients and standard frozen input receiver separately. All four messages retained even on input or RGB failure; no content reselection.',
       scope='Exploratory Cal strength comparison only, not independent capacity, false-positive, or authentication validation. Same four legal words, no RS/RM, Key relabeling unchanged.'))
    return check()


def check():
    p=read(OUT/'protocol.json')
    for f,h in {**p['sources'],**p['parents']}.items():assert digest(f)==h,f
    return p


def prepare():
    freeze();cv2.setNumThreads(1)
    if (OUT/'input_freeze.json').exists():return
    selected=[c for c in read(PARENT/'selection.json') if c['split']=='Cal'];oldjobs=[j for j in read(PARENT/'plan.json') if j['split']=='Cal' and j['arm'] in ['clean']+WORDS]
    assert len(selected)==6 and len(oldjobs)==60;plan=[];inputs=[]
    for c in selected:
        edge=np.array(Image.open(c['states']['clean']['hint']).convert('L'));original_mask=np.array(Image.open(c['mask']).convert('L'))>0
        valid=np.zeros_like(original_mask);x0,y0,x1,y1=c['roi'];valid[y0:y1,x0:x1]=True
        mask=(cv2.dilate(original_mask.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))>0)&valid
        mp=OUT/'edit_masks'/f'{c["key"]}.png';Image.fromarray(mask.astype('uint8')*255).save(mp);warp=RasterWarp(edge,c['samples'],mask)
        # Fixed unique original-pixel stations, chosen without a message.
        ids=np.flatnonzero(warp.chosen);order=sorted(ids,key=lambda i:(warp.near[i],warp.distance[i],i));unique={}
        for i in order:unique.setdefault(int(warp.near[i]),i)
        stations=np.array(sorted(unique));pixel_ids=np.array([unique[i] for i in stations])
        origin=fit(c['samples'],dict(indices=stations,xy=warp.xy[pixel_ids],weights=np.ones(len(stations))),'S')
        for peak in PEAKS:
            for word in WORDS:
                displacement=basis(warp.u)@((peak/b4.a4.PEAK)*np.array(b4.a4.SIGNS[word]));warp.vector=displacement[warp.near[warp.chosen],None]*warp.n[warp.near[warp.chosen]]
                image,moved=warp.render(1.)
                assert set(np.unique(image))<={0,255};assert np.array_equal(image[~mask],edge[~mask])
                path=OUT/'conditions'/f'{c["key"]}_{tag(peak)}_{word}.png';Image.fromarray(image).save(path)
                if peak==1.5:assert np.array_equal(image,np.array(Image.open(c['states'][word]['hint']).convert('L')))
                xy=moved[pixel_ids];inside=mask[xy[:,1],xy[:,0]];keep=inside&(image[xy[:,1],xy[:,0]]>0)
                known=fit(c['samples'],dict(indices=stations[keep],xy=xy[keep],weights=np.ones(keep.sum())),'S')
                delta=np.array(known['coefficient'])-np.array(origin['coefficient']) if known else None
                nominal=(peak/b4.a4.PEAK)*np.array([b4.a4.SIGNS[w] for w in WORDS]);decoded=WORDS[int(np.argmin(np.sum((nominal-delta)**2,axis=1)))] if delta is not None else None
                shift=np.linalg.norm(moved[warp.chosen]-warp.xy[warp.chosen],axis=1);all_moved=moved[warp.chosen]
                inputs.append(dict(key=c['key'],peak=peak,word=word,hint=str(path),hint_sha256=digest(path),known_pixel_fit=known,origin_fit=origin,
                    decoded=decoded,input_correct=decoded==word,coefficient_delta=delta.tolist() if delta is not None else None,
                    selected_edge_pixels=int(warp.chosen.sum()),shifted_fraction=float(np.mean(shift>0)),max_integer_shift_px=float(shift.max()),mean_integer_shift_px=float(shift.mean()),
                    planned_moves_clipped=int(np.sum(~mask[all_moved[:,1],all_moved[:,0]])),changed_pixels=int(np.count_nonzero(image!=edge))))
                for old in [j for j in oldjobs if j['key']==c['key'] and j['arm']==word]:
                    job=f'{old["job"]}_{tag(peak)}';reuse=peak==1.5;rgb=old['rgb'] if reuse else str(OUT/'rgb'/f'{job}.png')
                    plan.append(dict(**{k:v for k,v in old.items() if k not in ['job','hint','hint_sha256','rgb']},job=job,original_job=old['job'],peak=peak,hint=str(path),hint_sha256=digest(path),rgb=rgb,reused=reuse))
    for old in [j for j in oldjobs if j['arm']=='clean']:plan.append(dict(**old,original_job=old['job'],peak=0.,reused=True))
    assert len(plan)==204 and sum(not j['reused'] for j in plan)==144
    for j in plan:
        if not j['reused']:continue
        old=read(PARENT/'records'/f'{j["original_job"]}.json');assert digest(j['rgb'])==old['rgb_sha256'];assert digest(j['noise'])==old['noise_sha256'];assert j['prompt']==old['prompt'] and j['seed']==old['seed']
        dump(OUT/'records'/f'{j["job"]}.json',dict(**j,rgb_sha256=digest(j['rgb']),noise_sha256=digest(j['noise']),reused_record=str(PARENT/'records'/f'{j["original_job"]}.json'),reused_record_sha256=digest(PARENT/'records'/f'{j["original_job"]}.json')))
    dump(OUT/'selection.json',selected);dump(OUT/'plan.json',plan);dump(OUT/'input_closure.json',inputs)
    protected=[OUT/n for n in ['selection.json','plan.json','input_closure.json']]+list((OUT/'conditions').glob('*.png'))+list((OUT/'edit_masks').glob('*.png'))
    dump(OUT/'input_freeze.json',dict(created_utc=b4.now(),protocol_sha256=digest(OUT/'protocol.json'),protected={str(p):digest(p) for p in protected},new_RGB=144,reused_RGB=60))
    print('PREPARE_COMPLETE',len(plan),'jobs',sum(r['input_correct'] for r in inputs),'/96 input messages',flush=True)


def worker():
    p=check();gen=p['generation'];assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
    import torch
    from cldm.ddim_hacked import DDIMSampler
    from controlnet_robin.pipeline import ControlNetROBINPipeline
    assert torch.cuda.is_available();assert digest(ROOT/gen['checkpoint'])==gen['checkpoint_sha256']
    for path,h in read(OUT/'input_freeze.json')['protected'].items():assert digest(path)==h
    for path,h in read(b4.a4.S1/'protocol.json')['frozen_model_code'].items():assert digest(path)==h
    torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    class DeviceSampler(DDIMSampler):
        def register_buffer(self,name,attr):setattr(self,name,attr.to(self.model.device) if isinstance(attr,torch.Tensor) else attr)
    pipe=ControlNetROBINPipeline(str(ROOT/gen['model_config']),str(ROOT/gen['checkpoint']),'cuda');pipe.model.requires_grad_(False).eval();pipe.model.cond_stage_model.device='cuda';pipe.ddim=DeviceSampler(pipe.model)
    pipe.make_schedule(gen['steps'],eta=gen['eta']);pipe.model.control_scales=[gen['control_scale']]*13
    def generate(j):
        assert digest(j['hint'])==j['hint_sha256'];noise=torch.randn((1,4,64,64),generator=torch.Generator(device='cpu').manual_seed(j['seed']))
        assert np.array_equal(noise.numpy(),np.load(j['noise']))
        cond,uncond=pipe.get_conditioning(j['prompt'],np.array(Image.open(j['hint']).convert('RGB')),'',512,False);latent=noise.clone().cuda()
        with torch.inference_mode():
            for step in range(pipe.total_steps):latent=pipe.ddim_step(latent,step,cond,uncond,gen['guidance'])
            rgb=((pipe.model.decode_first_stage(latent)+1)/2).clamp(0,1)
            return (rgb[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
    plan=read(OUT/'plan.json');canaries=[next(j for j in plan if j['arm']=='clean'),next(j for j in plan if j['peak']==1.5)]
    if not (OUT/'canary.json').exists():
        checks=[]
        for j in canaries:
            actual=generate(j);reference=np.array(Image.open(j['rgb']).convert('RGB'));same=np.array_equal(actual,reference)
            cp=OUT/'canary'/f'{j["job"]}.png';Image.fromarray(actual).save(cp)
            checks.append(dict(job=j['job'],pixels_identical=bool(same),maximum_difference=int(np.max(abs(actual.astype(int)-reference.astype(int)))),reference_sha256=digest(j['rgb']),canary_sha256=digest(cp)))
            assert same,('Runtime changed; do not mix old/new PNG',checks[-1])
        dump(OUT/'canary.json',dict(created_utc=b4.now(),checks=checks,checkpoint_sha256=gen['checkpoint_sha256']))
    new=[j for j in plan if not j['reused']]
    for n,j in enumerate(new):
        dest=OUT/'records'/f'{j["job"]}.json'
        if dest.exists():assert digest(j['rgb'])==read(dest)['rgb_sha256'];continue
        start=time.monotonic();rgb=generate(j);Image.fromarray(rgb).save(j['rgb'])
        dump(dest,dict(**j,rgb_sha256=digest(j['rgb']),noise_sha256=digest(j['noise']),checkpoint_sha256=gen['checkpoint_sha256'],protocol_sha256=digest(OUT/'protocol.json'),elapsed_seconds=time.monotonic()-start))
        print('GEN',n+1,'/144',j['job'],round(time.monotonic()-start,2),flush=True)
    dump(OUT/'generation_status.json',dict(status='complete',new_RGB=144,reused_RGB=60,total_RGB=204))


def extract():
    check();cv2.setNumThreads(1);cc={c['key']:c for c in read(OUT/'selection.json')};frozen=read(b4.OUT/'score_freeze.json');oldc0=read(C0/'score_freeze.json')
    plan=read(OUT/'plan.json')
    for n,j in enumerate(plan):
        dest=OUT/'features'/f'{j["job"]}.json'
        if dest.exists():continue
        record=OUT/'records'/f'{j["job"]}.json'
        while not record.exists():time.sleep(2)
        assert digest(j['rgb'])==read(record)['rgb_sha256'];c=cc[j['key']];rgb=np.array(Image.open(j['rgb']).convert('RGB'))
        if j['reused']:
            prev=read(C0/'cal_features'/f'{j["original_job"]}.json');assert prev['rgb_sha256']==digest(j['rgb']);obs=prev['reference_observation'];public=prev['public']
        else:
            raw=b4.rematch(c['samples'],rgb,j,b4.SCENARIOS[0]);obs=b4.select(raw,frozen['gate']);mask=np.array(Image.open(c['mask']).convert('L'))>0
            box=np.zeros_like(mask);yy,xx=np.nonzero(mask)
            if len(xx):box[yy.min():yy.max()+1,xx.min():xx.max()+1]=True
            public=dict(R2_tube=c0.extract_rgb(rgb,mask),R2_bbox=c0.extract_rgb(rgb,box))
        value=(np.array(obs['coefficient'])-np.array(c['states']['clean']['measurements']['Auto']['coefficient'])).tolist() if obs.get('accepted') else None
        methods=dict(SourceRef=dict(accepted=bool(obs.get('accepted')),value=value,covariance=obs.get('covariance',np.eye(2).tolist())))
        for method in ['R2_tube','R2_bbox']:
            config=oldc0[method]['selected'];f=public[method]['fits'][str(config['degree'])];ok=c0.eligible(f,config['residual'],config['ci95'])
            methods[method]=dict(accepted=ok,value=f['coefficient'] if ok else None,covariance=f.get('covariance',np.eye(2).tolist()))
        dump(dest,dict(job=j['job'],original_job=j['original_job'],key=j['key'],content_id=j['content_id'],view=j['view'],arm=j['arm'],peak=j['peak'],rgb_sha256=digest(j['rgb']),methods=methods,reference_observation=obs,public=public))
        print('EXTRACT',n+1,'/204',j['job'],flush=True)


def quality():
    check();import torch,lpips
    from carrier_screen.common import pixel_metrics
    torch.set_num_threads(3);cv2.setNumThreads(1);net=None;g=read(OUT/'protocol.json')['quality']
    old={r['job']:r for r in read(PARENT/'quality.json')['records']}
    for j in read(OUT/'plan.json'):
        if j['arm']=='clean':continue
        dest=OUT/'quality'/f'{j["job"]}.json'
        if dest.exists():continue
        while not (OUT/'records'/f'{j["job"]}.json').exists():time.sleep(2)
        if j['reused']:row=old[j['original_job']];q=row['quality'];gates=row['gates']
        else:
            if net is None:net=lpips.LPIPS(net='alex',version='0.1').eval().requires_grad_(False).cpu()
            clean=np.array(Image.open(j['clean_rgb']).convert('RGB'));rgb=np.array(Image.open(j['rgb']).convert('RGB'));q=pixel_metrics(clean,rgb)
            a=torch.from_numpy(clean).permute(2,0,1)[None].float()/127.5-1;b=torch.from_numpy(rgb).permute(2,0,1)[None].float()/127.5-1
            with torch.inference_mode():q['lpips_alex']=float(net(a,b))
            gates=dict(psnr=bool(q['psnr_infinite'] or q['psnr_db']>=g['psnr_db_min']),ssim=q['ssim']>=g['ssim_min'],lpips=q['lpips_alex']<=g['lpips_alex_max'],edge_f1=q['edge_f1_3x3']>=g['rgb_pair_edge_f1_3x3_min'])
        dump(dest,dict(job=j['job'],peak=j['peak'],key=j['key'],arm=j['arm'],view=j['view'],quality=q,gates=gates,pass_all=all(gates.values()),rgb_sha256=digest(j['rgb']),clean_rgb_sha256=digest(j['clean_rgb'])))
        print('QUALITY',j['job'],all(gates.values()),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','extract','quality']);a=p.parse_args();globals()[a.action]()
