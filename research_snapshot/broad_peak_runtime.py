"""Self-contained runtime adapter after old experiment folders disappeared.

No geometric, matching, mask, score, prompt or seed parameter is retuned.
"""
from pathlib import Path
import json,hashlib,importlib.util,os,time
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'outputs/canny_curve_stage2c_qim1_20261110'
OUT=ROOT/'outputs/canny_broad_peak_code_20261111'
S1=ROOT/'outputs/canny_curve_stage1_train32_20261110'
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arr(p):return np.array(Image.open(p).convert('L'))

def local_matcher():
    p=OUT/'source/frozen_stage1_matcher.py'
    if not p.exists():p=S1/'source/analyze_canny_curve_stage1_32.py'
    expected=read(S1/'audit.json')['source_sha256']['analyze_canny_curve_stage1_32.py'] if 'source_sha256' in read(S1/'audit.json') else '6bfbef83d95d3e970df4b9a954eab9e479d40306fcff5bb003543df76ef96392'
    assert digest(p)==expected
    spec=importlib.util.spec_from_file_location('broad_peak_frozen_stage1',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

# Redirect only the missing snapshot location, before any receiver imports.
import canny_curve_stage2_bit as _core
_core.matcher=local_matcher
THRESHOLDS=read(S1/'protocol.json')['exploration_thresholds']
GATE=dict(coverage=.65,residual=.6,margin=0.,offset=1.5)

def generate(items,folder):
    import torch,cv2
    from cldm.ddim_hacked import DDIMSampler
    from controlnet_robin.pipeline import ControlNetROBINPipeline
    gen=read(OUT/'protocol.json')['generation'];assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and torch.cuda.is_available()
    assert digest(ROOT/gen['checkpoint'])==gen['checkpoint_sha256']
    for p,h in read(S1/'protocol.json')['frozen_model_code'].items():assert digest(p)==h,p
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    class DeviceSampler(DDIMSampler):
        def register_buffer(self,name,attr):setattr(self,name,attr.to(self.model.device) if isinstance(attr,torch.Tensor) else attr)
    pipe=ControlNetROBINPipeline(str(ROOT/gen['model_config']),str(ROOT/gen['checkpoint']),'cuda');pipe.model.requires_grad_(False).eval();pipe.model.cond_stage_model.device='cuda';pipe.ddim=DeviceSampler(pipe.model)
    pipe.make_schedule(gen['steps'],eta=gen['eta']);pipe.model.control_scales=[gen['control_scale']]*13
    folder.mkdir(exist_ok=True)
    for ix,j in enumerate(items):
        record=folder/f'{j["job"]}.json'
        if record.exists():assert digest(j['rgb'])==read(record)['rgb_sha256'];continue
        assert digest(j['hint'])==j['hint_sha256'];noise=torch.randn((1,4,64,64),generator=torch.Generator(device='cpu').manual_seed(j['seed']))
        Path(j['noise']).parent.mkdir(exist_ok=True)
        if Path(j['noise']).exists():assert np.array_equal(np.load(j['noise']),noise.numpy())
        else:np.save(j['noise'],noise.numpy())
        cond,uncond=pipe.get_conditioning(j['prompt'],np.array(Image.open(j['hint']).convert('RGB')),'',512,False);latent=noise.clone().cuda();start=time.monotonic()
        with torch.inference_mode():
            for step in range(pipe.total_steps):latent=pipe.ddim_step(latent,step,cond,uncond,gen['guidance'])
            rgb=((pipe.model.decode_first_stage(latent)+1)/2).clamp(0,1);image=(rgb[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
        Image.fromarray(image).save(j['rgb']);dump(record,dict(**j,rgb_sha256=digest(j['rgb']),noise_sha256=digest(j['noise']),checkpoint_sha256=gen['checkpoint_sha256'],elapsed_seconds=time.monotonic()-start,
            protocol_sha256=digest(OUT/'protocol.json'),runtime_recovery_sha256=digest(OUT/'runtime_recovery_freeze.json')))
        print('GEN',ix+1,'/',len(items),j['job'],flush=True)

def density_fit(values):
    return dict(n=len(values),mean=float(np.mean(values)) if values else .25,variance=max(.05**2,float(np.var(values))) if values else .1)
def presence(value,model):
    from scipy.special import expit
    logs=[]
    for key in ['clean','marked']:
        p=model[key];logs.append(-.5*(np.log(2*np.pi*p['variance'])+(value-p['mean'])**2/p['variance']))
    return float(expit(logs[1]-logs[0]))
def quality(plan,dest):
    if dest.exists():return read(dest)
    import torch,lpips,cv2
    from carrier_screen.common import pixel_metrics
    torch.set_num_threads(4);cv2.setNumThreads(1);net=lpips.LPIPS(net='alex',version='0.1').eval().requires_grad_(False).cpu();g=read(OUT/'protocol.json')['quality'];rows=[]
    for j in plan:
        if j['arm']=='clean':continue
        clean=np.array(Image.open(j['clean_rgb']).convert('RGB'));image=np.array(Image.open(j['rgb']).convert('RGB'));q=pixel_metrics(clean,image)
        aa=torch.from_numpy(clean).permute(2,0,1)[None].float()/127.5-1;bb=torch.from_numpy(image).permute(2,0,1)[None].float()/127.5-1
        with torch.inference_mode():q['lpips_alex']=float(net(aa,bb))
        gates=dict(psnr=bool(q['psnr_infinite'] or q['psnr_db']>=g['psnr_db_min']),ssim=q['ssim']>=g['ssim_min'],lpips=q['lpips_alex']<=g['lpips_alex_max'],edge_f1=q['edge_f1_3x3']>=g['rgb_pair_edge_f1_3x3_min'])
        rows.append(dict(job=j['job'],key=j['key'],split=j['split'],arm=j['arm'],quality=q,gates=gates,pass_all=all(gates.values()),rgb_sha256=digest(j['rgb'])));print('QUALITY',len(rows),j['job'],all(gates.values()),flush=True)
    result=dict(frozen_gates=g,records=rows,attempts=len(rows),joint_passes=sum(r['pass_all'] for r in rows));dump(dest,result);return result
