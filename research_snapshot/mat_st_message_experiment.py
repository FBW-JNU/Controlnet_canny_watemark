"""Development-only G0--G4 local MAT/S,T experiment; gate RGB on real PNG closure."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/mat_st_message_mpl')
import argparse
import hashlib
import json
import sys
import time
import shutil
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
import broad_peak_runtime
from curve_broad_peak_code import basis,receive,fit
from curve_receiver_stage2c import geometry,binary_target
from curve_receiver_stage2b import prepare_target
from curve_qim_closed_loop_v2 import RasterWarp
from experiment_s_frozen_validation import THRESHOLDS,GATE

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/mat_st_message_experiment1_20261111'
S1=ROOT/'outputs/canny_curve_stage1_train32_20261110'
MAT=ROOT/'fuxian_papers/results/MAE_MAT_related/mat_hed_pidi_repair_20261108'
WORDS=('m00','m01','m10','m11')
SIGNS=np.array([[-1,-1],[-1,1],[1,-1],[1,1]],float)
PEAK=float(np.max(abs(basis(np.linspace(0,1,100001))).sum(1)))
AMP=1.5/PEAK
LENGTHS=(32,48)
SEED=2026111101


def read(p):return json.loads(Path(p).read_text())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):
    def default(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=default,allow_nan=False)+'\n')
def arr(p):return np.array(Image.open(p).convert('L'))
def save(p,a):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(a.astype(np.uint8)).save(p)


def local_segment(samples,length):
    p,u,_,_=geometry(samples);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    x=np.linspace((arc[-1]-length)/2,(arc[-1]+length)/2,length+1)
    return np.c_[np.interp(x,arc,p[:,0]),np.interp(x,arc,p[:,1])]


def hole_for(source):
    p,u,n,_=geometry(source);yy,xx=np.mgrid[:512,:512];xy=np.c_[xx.ravel(),yy.ravel()]
    d,near=cKDTree(p).query(xy);t=np.c_[n[:,1],-n[:,0]]
    keep=(d<=4)&(((xy-p[0])@t[0])>.5)&(((xy-p[-1])@t[-1])<-.5)
    return keep.reshape(512,512)


def observe(source,image,rgb=False):
    target=prepare_target(image) if rgb else binary_target(image,source)
    if not len(target['coords']):return dict(Auto=dict(accepted=False,coefficient=None),Oracle=dict(accepted=False,coefficient=None))
    return receive(source,target,'S',THRESHOLDS,GATE)


def decode(obs,origin):
    if not obs.get('accepted') or origin is None:return dict(accepted=False,word=None,presence=False,reason='geometry_or_clean_reference_rejected')
    v=np.array(obs['coefficient'])-origin;centers=AMP*SIGNS;ds=((v-centers)**2).sum(1);j=int(ds.argmin())
    # Fixed nearest-codeword-vs-origin diagnostic; no learned FPR claim.
    score=float(v@v-ds[j]);presence=score>0 and ds[j]<=(.75*AMP)**2
    return dict(accepted=True,word=WORDS[j],coefficient_delta=v,nearest_distance=float(np.sqrt(ds[j])),presence_score=score,presence=bool(presence))


def nominal(edge,source,mask,sign):
    w=RasterWarp(edge,source,mask);w.vector=(basis(w.u)@(AMP*sign))[w.near[w.chosen],None]*w.n[w.near[w.chosen]]
    return w.render(1)[0]


def project(edge,source,mask,prob,sign,origin):
    """Project a message-independent real MAT ridge, retaining orthogonal residual.

    No clean-hole edge pixels are used to construct the projected curve. Source
    reference geometry is public encoder side information. No fallback to G4.
    """
    p,u,n,_=geometry(source);coords=np.argwhere(mask)[:,::-1]
    raw=(prob>=.1).astype(np.uint8)*255;raw[~mask]=edge[~mask]
    o=observe(source,raw);candidates=o['Auto'].get('decisions',[])
    pool=[c for c in candidates if c['coverage']>=.65 and c['span']>=.8 and c['residual_rms']<=1.5]
    if not pool:return raw,dict(success=False,reason='no_supported_MAT_curve_for_projection',observations=o)
    chosen=min(pool,key=lambda c:c['cost']);ii=np.array(chosen['indices']);xy=np.array(chosen['observed_xy'])
    y=np.sum((xy-p[ii])*n[ii],1);res=y-chosen['offset']-basis(u[ii])@np.array(chosen['coefficient'])
    # Residual is from MAT. Shrink then remove its complete S/T/offset component.
    residual=np.interp(u,u[ii],res);x=np.c_[np.ones(len(u)),basis(u)]
    residual-=x@np.linalg.lstsq(x,residual,rcond=None)[0]
    residual*=min(.25,.25/max(np.max(abs(residual)),1e-6))
    taper=np.sin(np.pi*u)**2;residual*=taper
    desired=(np.zeros(2) if origin is None else np.array(origin))+AMP*sign
    shift=basis(u)@desired+residual
    q=p+n*shift[:,None]
    out=raw.copy();inside=out.copy();inside[mask]=0
    # Project only the target ridge. Other MAT-predicted edges in the hole stay.
    yy,xx=np.nonzero(raw);d,_=cKDTree(p).query(np.c_[xx,yy]);remove=mask[yy,xx]&(d<=3)
    out[yy[remove],xx[remove]]=0
    line=np.zeros_like(edge);cv2.polylines(line,[np.rint(q).astype(np.int32).reshape(-1,1,2)],False,255,1,cv2.LINE_8)
    out[mask]=np.maximum(out[mask],line[mask]);out[~mask]=edge[~mask]
    return out,dict(success=True,raw_branch=chosen['branch'],raw_coefficient=chosen['coefficient'],raw_coverage=chosen['coverage'],
        retained_MAT_residual_rms=float(np.sqrt(np.mean(residual**2))),continuous_shift_peak=float(np.max(abs(shift))),projected_path=q)


def prepare():
    assert not OUT.exists(),'Preserve prior run'
    OUT.mkdir();cv2.setNumThreads(1)
    for n in ('source','cases','inputs','conditions','mat_raw','logs','panels','rgb','noise','records','output_canny'):(OUT/n).mkdir()
    results=read(S1/'curve_results.json');selected=read(S1/'selected.json');cases=[]
    for c in sorted(selected,key=lambda x:x['content_id']):
        pool=sorted([r for r in results if r['content_id']==c['content_id'] and r['heldout_R_ge_80pct'] and r['source_arc_length_px']>=80],key=lambda r:(-r['source_arc_length_px'],r['curve_index']))
        if not pool:continue
        r=pool[0];cid=c['content_id'];sourcefile=S1/'curves'/f'{cid:012d}_source.json'
        curve=next(v for v in read(sourcefile)['curves'] if v['curve_index']==r['curve_index'])
        edge=arr(c['hint']);case=dict(content_id=cid,key=f'{cid:012d}',source_rgb=c['prepared_source'],clean_hint=c['hint'],
            full_samples=curve['samples'],curve_index=r['curve_index'],historical_heldout_retention=r['heldout']['R'],
            caption=c['prompt'].strip(),prompt='A realistic documentary photograph of the following scene: '+c['prompt'].strip(),seed=SEED,
            selection='First10 ascending content IDs with existing heldout stable>=5/6 and length>=80; longest stable curve per content, no message/RGB result selection',segments={})
        for length in LENGTHS:
            source=local_segment(curve['samples'],length);mask=hole_for(source);name=f'{cid:012d}_L{length}'
            save(OUT/'inputs'/f'{name}_clean.png',edge);save(OUT/'inputs'/f'{name}_mask.png',mask*255)
            save(OUT/'inputs'/f'{name}_damaged.png',np.where(mask,0,edge))
            o=observe(source,edge);origins={h:o[h]['coefficient'] if o[h]['accepted'] else None for h in ('Auto','Oracle')}
            case['segments'][str(length)]=dict(samples=source,mask=str(OUT/'inputs'/f'{name}_mask.png'),clean_observations=o,origins=origins)
            for word,sign in zip(WORDS,SIGNS):save(OUT/'conditions'/f'{name}_G4_{word}.png',nominal(edge,source,mask,sign))
        cases.append(case);dump(OUT/'cases'/f'{cid:012d}.json',case)
        print('PREPARE',cid,r['curve_index'],flush=True)
        if len(cases)==10:break
    assert len(cases)==10
    dump(OUT/'selection.json',cases)
    gen=read(ROOT/'outputs/canny_single_curve_2bit_stage4a_20261111/protocol.json')['generation']
    protocol=dict(stage='Experiment1 development pilot; local short-hole MAT messaging',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        contents=10,length_ladder=list(LENGTHS),message_bits=2,words=WORDS,signs=SIGNS,basis=dict(S='6sqrt(3)t(1-t)(2t-1)',T='4t(1-t)(5(2t-1)^2-1)',
        local_interval='Same S/T functions, normalized arc length on middle32/48px subcurve, NOT original whole curve',coefficient_amplitude=AMP,nominal_peak_bound=1.5),
        groups=dict(G0='original binary Canny',G1='old canny_native MAT ordinary binary repair, no message',
            G2='G1 supported ridge followed by fixed S/T coefficient projection, preserving small orthogonal MAT residual',
            G3='frozen MAT W+ latent optimized with message shape/coefficient constraints, followed by SAME projection as G2',G4='unchanged RasterWarp S/T on original local source, no MAT'),
        mat=dict(checkpoint=str(MAT/'checkpoints/canny_native.pt'),checkpoint_sha256=digest(MAT/'checkpoints/canny_native.pt'),threshold=.1,
            seed=240,precision='float32 params, CUDA bfloat16 forward; deterministic reset of intrinsic MAT dropout per forward',
            inference_optimization='Adam W+ only,48steps,lr0.04; model parameters frozen. Not a trained message conditional network.',
            constraint='0.8 soft ridge target BCE +2 normal-moment MSE +0.1 G1-preservation L1 +0.002 W+ displacement MSE; fixed anchors and known pixels copied',
            projection='Require real binary MAT path coverage>=.65/span>=.8/residual<=1.5, bit-blind path min-cost; set coefficients to clean-origin+nominal centers, residual shrink<=.25, no direct-G4 fallback'),
        receiver=dict(implementation='existing R0/A3 joint S/T Huber and geometry gates, unchanged',thresholds=THRESHOLDS,gate=GATE,
            source_reference=True,known_mask=True,blind=False,decoder='same four nominal centers +/-amplitude relative to original input clean coefficients; no expected-word path selection',
            presence='Fixed nearest-codeword-vs-clean-origin: d_clean^2>d_nearest^2 AND d_nearest<=.75*amplitude. Exploratory, no calibrated FPR guarantee.',
            old_long_curve_channel='Not refitted or silently transferred to short local geometry; nominal centers used consistently in input/RGB'),
        rgb_gate='G3 PNGs: ALL40 attempts have accepted Auto geometry, correct full2bit, active changes, four distinct per content, exterior unchanged. No failure filtering.',
        escalation='Test32 first; if gate fails, test48 with SAME hyperparameters and retain all32 failures. If neither passes, NO diffusion. All10 contents retained.',
        generation=dict(gen,device='physical GPU1',seed=SEED,views=1,total_per_length=140),
        quality=read(ROOT/'outputs/canny_single_curve_2bit_stage4a_20261111/protocol.json')['quality'],
        limitations='Existing historical stability selection + development10, no new independent test. Original ordered source reference supplied; not RGB-only blind decoding. Presence is not authenticated watermark.',
        script_sha256=digest(__file__))
    dump(OUT/'protocol.json',protocol)
    paths=[Path(__file__),MAT/'checkpoints/canny_native.pt',MAT/'thresholds.json',S1/'protocol.json',S1/'curve_results.json']
    paths += [ROOT/n for n in ('curve_broad_peak_code.py','curve_receiver_stage2b.py','curve_receiver_stage2c.py','curve_qim_closed_loop_v2.py','broad_peak_runtime.py','canny_curve_stage2_bit.py')]
    paths += [ROOT/'fuxian_papers/scripts'/n for n in ('mat_hed_pidi_repair.py','mat_canny_model.py','mat_common.py')]
    paths += [Path(c[k]) for c in cases for k in ('clean_hint','source_rgb')]
    paths += list((OUT/'inputs').glob('*.png'))+list((OUT/'cases').glob('*.json'))+[OUT/'selection.json',OUT/'protocol.json']
    dump(OUT/'freeze.json',{str(p):digest(p) for p in paths})
    shutil.copy2(__file__,OUT/'source'/Path(__file__).name)


def verify():
    for p,h in read(OUT/'freeze.json').items():assert digest(p)==h,p


def load_mat():
    import torch
    sys.path.insert(0,str(ROOT/'fuxian_papers/scripts'))
    from mat_hed_pidi_repair import RepairMAT
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=RepairMAT('native').cuda().eval().requires_grad_(False)
    ck=torch.load(MAT/'checkpoints/canny_native.pt',map_location='cpu',weights_only=False)
    assert ck['protocol_sha256']==digest(MAT/'protocol.json')
    model.load_state_dict(ck['state'],strict=True)
    return model


def run_mat():
    import torch
    import torch.nn.functional as F
    verify();cv2.setNumThreads(1);model=load_mat();log=[]
    for length in LENGTHS:
        if (OUT/f'input_gate_L{length}.json').exists():
            if read(OUT/f'input_gate_L{length}.json')['passed']:break
            continue
        for c in read(OUT/'selection.json'):
            key=c['key'];seg=c['segments'][str(length)];source=seg['samples'];edge=arr(c['clean_hint']);mask=arr(seg['mask'])>0
            prefix=f'{key}_L{length}';dest=OUT/'mat_raw'/prefix;dest.mkdir(exist_ok=True)
            if (dest/'complete.json').exists():print('REUSE',prefix,flush=True);continue
            p,u,n,_=geometry(source);b=basis(u);origin=seg['origins']['Auto']
            known=torch.from_numpy((~mask).astype(np.float32))[None,None].cuda()
            visible=torch.from_numpy(edge.astype(np.float32)/255)[None,None].cuda()*known
            inp=(visible*2-1)*known
            z=torch.from_numpy(np.random.RandomState(240).randn(1,512).astype(np.float32)).cuda()
            with torch.no_grad():w0=model.base.mapping(z,torch.zeros(1,0,device='cuda')).detach()
            def forward(w):
                torch.manual_seed(240);torch.cuda.manual_seed_all(240)
                with torch.autocast('cuda',dtype=torch.bfloat16):raw,_=model.base.synthesis(inp,known,w,noise_mode='const',return_stg1=True)
                return (raw.float()*3).sigmoid()
            with torch.no_grad():q0=forward(w0)
            prob0=q0[0,0].cpu().numpy();g1=np.where(mask,prob0>=.1,edge>0).astype(np.uint8)*255
            save(OUT/'conditions'/f'{prefix}_G0.png',edge);save(OUT/'conditions'/f'{prefix}_G1.png',g1)
            np.save(dest/'G1_prob.npy',prob0)
            # Explicit hidden-input invariance: neither hidden source pixels nor
            # any soft target are provided to MAT; signed masked input is zero.
            assert torch.count_nonzero(inp[torch.from_numpy(mask)[None,None].cuda()])==0
            d=np.linspace(-4,4,33);points=p[:,None,:]+n[:,None,:]*d[None,:,None]
            grid=torch.tensor(points/511*2-1,dtype=torch.float32,device='cuda')[None]
            dd=torch.tensor(d,dtype=torch.float32,device='cuda');x=torch.tensor(np.c_[np.ones(len(u)),b],dtype=torch.float32,device='cuda')
            pinv=torch.linalg.pinv(x);masktensor=torch.from_numpy(mask)[None,None].cuda()
            yy,xx=np.nonzero(mask);coords=np.c_[xx,yy]
            for wi,(word,sign) in enumerate(zip(WORDS,SIGNS)):
                g2,detail2=project(edge,source,mask,prob0,sign,origin)
                save(OUT/'conditions'/f'{prefix}_G2_{word}.png',g2)
                desired=(np.zeros(2) if origin is None else origin)+AMP*sign
                target_path=p+n*(b@desired)[:,None];near=cKDTree(target_path).query(coords)[0]
                target=np.zeros((512,512),np.float32);target[yy,xx]=np.exp(-.5*(near/.65)**2)
                target_t=torch.from_numpy(target)[None,None].cuda()
                desired_t=torch.tensor(desired,dtype=torch.float32,device='cuda')
                w=w0.clone().requires_grad_(True);opt=torch.optim.Adam([w],lr=.04);history=[]
                for step in range(49):
                    q=forward(w);cross=F.grid_sample(q,grid,align_corners=True)[0,0]
                    weights=torch.softmax(cross*8,dim=1);mean=(weights*dd).sum(1);coef=pinv@mean
                    shape=F.binary_cross_entropy(q[masktensor].clamp(1e-6,1-1e-6),target_t[masktensor])
                    coeff=((coef[1:]-desired_t)**2).mean();prior=abs(q[masktensor]-q0[masktensor]).mean();latent=((w-w0)**2).mean()
                    loss=.8*shape+2*coeff+.1*prior+.002*latent
                    if step in (0,12,24,48):history.append(dict(step=step,loss=float(loss.detach()),shape=float(shape.detach()),coefficient=float(coeff.detach()),soft_coefficient=coef.detach().cpu().numpy()))
                    if step==48:break
                    opt.zero_grad();loss.backward();assert torch.isfinite(w.grad).all();opt.step()
                prob=q.detach()[0,0].cpu().numpy();raw=np.where(mask,prob>=.1,edge>0).astype(np.uint8)*255
                g3,detail3=project(edge,source,mask,prob,sign,origin)
                save(OUT/'conditions'/f'{prefix}_G3_{word}.png',g3);save(dest/f'G3_preprojection_{word}.png',raw)
                np.save(dest/f'G3_probability_{word}.npy',prob);np.save(dest/f'G3_W_{word}.npy',w.detach().cpu().numpy())
                detail=dict(key=key,length=length,word=word,G2=detail2,G3=detail3,history=history,
                    latent_rms=float(torch.sqrt(((w-w0)**2).mean()).detach()),raw_vs_G1_pixels=int(np.count_nonzero(raw!=g1)),soft_vs_G1_mae=float(abs(prob[mask]-prob0[mask]).mean()),
                    G3_preprojection_observations=observe(source,raw))
                dump(dest/f'{word}.json',detail);log.append(detail)
                print('MAT',prefix,word,'raw changes',detail['raw_vs_G1_pixels'],'projection',detail3['success'],flush=True)
            dump(dest/'complete.json',dict(completed=True,hidden_input_erased=True))
        gate=score_input(length);dump(OUT/f'input_gate_L{length}.json',gate)
        print('INPUT_GATE',length,gate['passed'],gate['groups'],flush=True)
        if gate['passed']:break
    dump(OUT/'mat_status.json',dict(complete=True,gate_passed_lengths=[l for l in LENGTHS if (OUT/f'input_gate_L{l}.json').exists() and read(OUT/f'input_gate_L{l}.json')['passed']]))


def score_input(length):
    records=[];groups={};gate_all=True
    for c in read(OUT/'selection.json'):
        edge=arr(c['clean_hint']);s=c['segments'][str(length)];source=s['samples'];mask=arr(s['mask'])>0;prefix=f'{c["key"]}_L{length}'
        for group in ('G0','G1','G2','G3','G4','G3raw'):
            words=WORDS if group not in ('G0','G1') else ('none',);hashes=[];correct=0;presence=0;accepted=0
            for word in words:
                path=(OUT/'mat_raw'/prefix/f'G3_preprojection_{word}.png' if group=='G3raw' else OUT/'conditions'/f'{prefix}_{group}{"_"+word if word!="none" else ""}.png')
                image=arr(path);obs=observe(source,image);dec=decode(obs['Auto'],s['origins']['Auto']);oracle=decode(obs['Oracle'],s['origins']['Oracle'])
                active=bool(np.any(image!=edge));exterior=bool(np.array_equal(image[~mask],edge[~mask]));h=digest(path);hashes.append(h)
                right=dec['accepted'] and dec['word']==word
                if group=='G3':
                    detail=read(OUT/'mat_raw'/prefix/f'{word}.json')
                    gate_all=gate_all and right and active and exterior and detail['G3']['success']
                correct+=right;presence+=dec['presence'];accepted+=dec['accepted']
                records.append(dict(key=c['key'],length=length,group=group,word=word,path=str(path),sha256=h,decode=dec,oracle=oracle,correct=bool(right),active=active,exterior_unchanged=exterior,observations=obs))
            distinct=len(set(hashes))==4 if len(words)==4 else None
            if group=='G3':gate_all=gate_all and distinct
            g=groups.setdefault(group,dict(attempts=0,correct=0,accepted=0,presence=0,distinct_contents=0))
            for k,v in [('attempts',len(words)),('correct',correct),('accepted',accepted),('presence',presence),('distinct_contents',int(bool(distinct)))]:g[k]+=int(v)
    dump(OUT/f'input_measurements_L{length}.json',records)
    return dict(passed=bool(gate_all),length=length,groups=groups,rule=read(OUT/'protocol.json')['rgb_gate'])


def make_plan():
    verify();status=read(OUT/'mat_status.json');passed=status['gate_passed_lengths']
    if not passed:
        dump(OUT/'rgb_status.json',dict(status='SKIPPED_INPUT_GATE_FAILED',reason='No tested local length has complete G3 40/40 actual-PNG closure. Do not start diffusion.'))
        return
    length=passed[0];gate=read(OUT/f'input_gate_L{length}.json');assert gate['passed'];plan=[]
    for c in read(OUT/'selection.json'):
        prefix=f'{c["key"]}_L{length}'
        for group in ('G0','G1','G2','G3','G4'):
            for word in (('none',) if group in ('G0','G1') else WORDS):
                job=f'{prefix}_{group}'+('' if word=='none' else '_'+word);hint=OUT/'conditions'/f'{job}.png'
                plan.append(dict(job=job,key=c['key'],length=length,group=group,word=word,prompt=c['prompt'],seed=c['seed'],hint=str(hint),hint_sha256=digest(hint),rgb=str(OUT/'rgb'/f'{job}.png'),noise=str(OUT/'noise'/f'{c["key"]}.npy'),clean_rgb=str(OUT/'rgb'/f'{prefix}_G0.png')))
    assert len(plan)==140
    dump(OUT/'plan.json',plan);dump(OUT/'rgb_gate_freeze.json',dict(gate_sha256=digest(OUT/f'input_gate_L{length}.json'),protocol_sha256=digest(OUT/'protocol.json'),plan_sha256=digest(OUT/'plan.json'),conditions={j['hint']:j['hint_sha256'] for j in plan}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run_mat','make_plan'));a=p.parse_args();globals()[a.action]()
