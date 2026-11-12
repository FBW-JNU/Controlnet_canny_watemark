"""Paired native/refined MAT for Canny, genuine HED/PIDI soft and scribble.

All source RGB is used offline for teachers only. Repair receives one edge map
with a boolean missing mask. Original MAT/checkpoints remain unchanged.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from skimage.morphology import skeletonize

from mat_canny_model import native_mat
from mat_soft_threeway import ROOT,WEIGHT,SEED,digest,dump,seed_of,configure
from edge_data import prepare_rgb
from reproduce_scribble_usketch import official_definitions,hed_soft,pidi_soft,scribble_postprocess

OUT=ROOT/'results/MAE_MAT_related/mat_hed_pidi_repair_20261108'
KINDS=('canny','hed_soft','pidi_soft','hed_scribble','pidi_scribble')
STYLES=KINDS[1:]
STRUCTURES=('native','refined')


def rows(split):
    return json.loads((OUT/'manifest.json').read_text())['splits'][split]


def load_detectors(device):
    ns=official_definitions(ROOT/'Improved-Scribble/upstream/annotator/hed/__init__.py',
         ['DoubleConvBlock','ControlNetHED_Apache2'],{'torch':torch})
    hed=ns['ControlNetHED_Apache2']()
    hed.load_state_dict(torch.load(ROOT/'checkpoints/improved_scribble/ControlNetHED.pth',map_location='cpu',weights_only=True),strict=True)
    path=ROOT/'Improved-Scribble/upstream/annotator/pidinet/model.py'
    tree=ast.parse(path.read_text())
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='basicsr.utils')]
    ns={'__name__':'official_pidinet_offline'}
    exec(compile(tree,str(path),'exec'),ns)
    pidi=ns['pidinet']()
    state=torch.load(ROOT/'checkpoints/improved_scribble/table5_pidinet.pth',map_location='cpu',weights_only=True)['state_dict']
    pidi.load_state_dict({k.replace('module.',''):v for k,v in state.items()},strict=True)
    util=official_definitions(ROOT/'Improved-Scribble/upstream/annotator/util.py',['nms'],{'cv2':cv2,'np':np})
    return hed.to(device).eval().requires_grad_(False),pidi.to(device).eval().requires_grad_(False),util['nms']


def representations(hed,pidi,nms,rgb,device='cuda:0'):
    hs=hed_soft(hed,rgb,device);ps=pidi_soft(pidi,rgb,device)
    return dict(hed_soft=hs,pidi_soft=ps,hed_scribble=scribble_postprocess(hs,nms),
                pidi_scribble=scribble_postprocess(ps,nms))


def extract():
    configure();hed,pidi,nms=load_detectors('cuda:0');record={}
    for split in ('train','val','test'):
        for i,row in enumerate(rows(split)):
            rgb,_,_=prepare_rgb(row['source'],512)
            assert digest(row['source'])==row['source_sha256']
            with np.load(OUT/row['input_path']) as d:edge=d['edge'].copy()
            arrays=representations(hed,pidi,nms,rgb)
            arrays['canny']=edge.astype(np.uint8)*255
            # Teacher and direct-Canny conversions use exact pinned extractors.
            dest=(OUT/row['input_path']).parent
            for kind,array in arrays.items():
                Image.fromarray(array).save(dest/f'{kind}.png')
            derived={}
            for kind,array in arrays.items():
                binary=array>=128
                derived[kind+'_skeleton']=skeletonize(binary)
            np.savez_compressed(dest/'skeletons.npz',**derived)
            if split!='train':
                for polarity in ('white','black'):
                    x=(edge.astype(np.uint8)*255)
                    if polarity=='black':x=255-x
                    direct=representations(hed,pidi,nms,np.repeat(x[...,None],3,2))
                    for kind,array in direct.items():
                        Image.fromarray(array).save(dest/f'direct_{polarity}_{kind}.png')
            record[row['sample_id']]={k:digest(dest/f'{k}.png') for k in KINDS}
            if (i+1)%64==0:print('EXTRACT',split,i+1,flush=True)
    dump(OUT/'teacher_manifest.json',dict(images=record,source_manifest_sha256=digest(OUT/'manifest.json'),
         weights={str(p.relative_to(ROOT)):digest(p) for p in
                  (WEIGHT,ROOT/'checkpoints/improved_scribble/ControlNetHED.pth',ROOT/'checkpoints/improved_scribble/table5_pidinet.pth')},
         hed='ControlNet Apache HED, RGB 0..255, mean of resized five side logits then sigmoid, uint8 truncation',
         pidi='official pidinet, BGR /255, final output probability, uint8 truncation',
         scribble='official ControlNet nms(soft,127,3.0); GaussianBlur sigma3; values>4 ->255 else0; no inversion',
         note='Scribble is deliberately a thick stroke representation; compare width to its own teacher, not to thin Canny.'))


class Block(nn.Module):
    def __init__(self,ci,co):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(ci,co,3,padding=1),nn.GroupNorm(4,co),nn.SiLU(),
                               nn.Conv2d(co,co,3,padding=1),nn.GroupNorm(4,co),nn.SiLU())
    def forward(self,x):return self.net(x)


class Pyramid(nn.Module):
    def __init__(self,ci,co,zero_last=False):
        super().__init__();widths=[12,24,48,96,128]
        self.enc=nn.ModuleList([Block(ci if i==0 else widths[i-1],w) for i,w in enumerate(widths)])
        self.dec=nn.ModuleList([Block(widths[i+1]+widths[i],widths[i]) for i in range(3,-1,-1)])
        self.out=nn.Conv2d(widths[0],co,1)
        if zero_last:nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,x):
        skips=[]
        for i,block in enumerate(self.enc):
            if i:x=F.avg_pool2d(x,2)
            x=block(x);skips.append(x)
        for block,skip in zip(self.dec,reversed(skips[:-1])):
            x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x=block(torch.cat([x,skip],1))
        return self.out(x)


class RepairMAT(nn.Module):
    def __init__(self,structure):
        super().__init__();torch.manual_seed(SEED)
        self.base,_=native_mat(WEIGHT,'cpu')
        self.base.synthesis.requires_grad_(True);self.base.mapping.requires_grad_(False)
        self.structure=structure
        self.refine=Pyramid(5,1,zero_last=True) if structure=='refined' else None

    def forward(self,s,seed):
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        known=(~s['missing']).float();visible=s['input']*known
        z=torch.from_numpy(np.random.RandomState(seed).randn(1,512).astype(np.float32)).to(visible.device)
        final,coarse=self.base((s['input']*2-1)*known,known,z,
            torch.zeros(1,0,device=visible.device),noise_mode='const',return_stg1=True)
        logits=final.float()*3;coarse_logits=coarse.float()*3
        if self.refine is not None:
            features=torch.cat([visible,s['missing'].float(),logits.sigmoid(),coarse_logits.sigmoid(),s['distance']],1)
            residual=self.refine(features).float()
            logits=logits+4*torch.tanh(residual/4)
        return logits,coarse_logits


def distance_feature(array,missing):
    known_edges=(array>=.5)&~missing
    return np.minimum(cv2.distanceTransform((~known_edges).astype(np.uint8),cv2.DIST_L2,3),64)/64


def sample(row,kind,device='cuda:0',condition=None):
    dest=(OUT/row['input_path']).parent
    with np.load(OUT/row['input_path']) as d:missing=d['missing'].copy()
    target=np.asarray(Image.open(dest/f'{kind}.png')).astype(np.float32)/255
    inp=target.copy() if condition is None else condition.copy()
    assert inp.shape==target.shape==(512,512)
    # Everything, including derived distance, is based on erased visible input.
    inp[missing]=0
    with np.load(dest/'skeletons.npz') as d:sk=d[kind+'_skeleton'].copy()
    values=dict(target=target,input=inp,missing=missing,skeleton=sk,
                distance=distance_feature(inp,missing))
    result={k:torch.from_numpy(v.copy())[None,None].to(device=device) for k,v in values.items()}
    for k in result:
        if k!='missing':result[k]=result[k].float()
    result['box']=row['box']
    return result


def soft_skeleton(p,iterations=12):
    def erode(x):return -F.max_pool2d(-x,3,1,1)
    def opened(x):return F.max_pool2d(erode(x),3,1,1)
    sk=F.relu(p-opened(p))
    for _ in range(iterations):
        p=erode(p);delta=F.relu(p-opened(p));sk=sk+F.relu(delta-sk*delta)
    return sk


def representation_loss(logits,s):
    missing=s['missing'];y=s['target'];prob=logits.float().sigmoid()
    bce=F.binary_cross_entropy_with_logits(logits.float()[missing],y[missing])
    intensity=F.l1_loss(prob[missing],y[missing])
    # Masked spatial derivatives penalize unnecessary stroke widening.
    p=torch.where(missing,prob,y)
    dx=F.l1_loss((p[:,:,:,1:]-p[:,:,:,:-1]),(y[:,:,:,1:]-y[:,:,:,:-1]),reduction='none')
    dy=F.l1_loss((p[:,:,1:,:]-p[:,:,:-1,:]),(y[:,:,1:,:]-y[:,:,:-1,:]),reduction='none')
    mx=missing[:,:,:,1:]|missing[:,:,:,:-1];my=missing[:,:,1:,:]|missing[:,:,:-1,:]
    grad=(dx[mx].mean()+dy[my].mean())/2
    x0,y0,x1,y1=s['box'];pad=4
    patch=p[:,:,y0-pad:y1+pad,x0-pad:x1+pad]
    fg=(y[:,:,y0-pad:y1+pad,x0-pad:x1+pad]>=.5).float()
    gt_sk=s['skeleton'][:,:,y0-pad:y1+pad,x0-pad:x1+pad]
    pred_sk=soft_skeleton(patch)
    tprec=((pred_sk*fg).sum()+1)/(pred_sk.sum()+1)
    tsens=((gt_sk*patch).sum()+1)/(gt_sk.sum()+1)
    cldice=1-2*tprec*tsens/(tprec+tsens+1e-6)
    return bce+.5*intensity+.2*grad+.1*cldice


@torch.no_grad()
def prediction(model,s,seed):
    with torch.autocast('cuda',dtype=torch.bfloat16):logits,_=model(s,seed)
    p=torch.where(s['missing'],logits.float().sigmoid(),s['input'])
    assert torch.isfinite(p).all()
    return p[0,0].cpu().numpy()


@torch.no_grad()
def validation(model,kind):
    values=[]
    for r in rows('val'):
        s=sample(r,kind)
        with torch.autocast('cuda',dtype=torch.bfloat16):a,_=model(s,seed_of(r['sample_id']+'|forward'))
        values.append(float(representation_loss(a,s)))
    assert np.isfinite(values).all()
    return float(np.mean(values))


def freeze(steps):
    assert not (OUT/'protocol.json').exists()
    rng=np.random.default_rng(SEED);schedule=[]
    while len(schedule)<steps:schedule.extend(rng.permutation(len(rows('train'))).tolist())
    dump(OUT/'protocol.json',dict(steps_per_arm=steps,kinds=list(KINDS),structures=list(STRUCTURES),seed=SEED,
         schedule=schedule[:steps],lr=1e-4,validation_steps=[steps//2,steps],batch_size=1,device='GPU0',amp='bfloat16',
         data='512 train / 64 val /128 test, previous deterministic content and hole selection, native512, 64/96 holes',
         initialization='same official Places->native-gray MAT; mapping frozen; all synthesis trainable; refiner last layer zero',
         structure='native two-stage MAT vs same MAT + residual five-level UNet (12/24/48/96/128); inputs visible representation, missing, MAT final/coarse probabilities, visible-edge distance',
         loss='identical for both structures: unweighted soft-label BCE +0.5 L1 +0.2 first derivative L1 +0.1 soft-clDice; final +0.25 coarse; no pos_weight3',
         checkpoint='minimum fixed validation loss at same validation steps',
         threshold='validation only: maximize strict foreground F1 on nonempty holes; fixed teacher binary reference >=128; grid .10:.05:.90',
         conversion='geometry presets and direct detector polarity chosen on validation only; learned 4-head Canny->teacher Pyramid trained on train only; no source RGB at inference',
         converter_steps=1200,converter_batch_size=4,converter_lr=3e-4,
         pipeline='erase Canny hole BEFORE conversion; erase converted hole BEFORE MAT; compare hole only; known output equals converted visible input',
         limitation='each domain has its own teacher; cross-domain raw F1 is not a common-task ranking; teacher soft confidence/semantic attribution cannot be recovered uniquely from binary Canny',
         manifest_sha256=digest(OUT/'manifest.json'),teacher_manifest_sha256=digest(OUT/'teacher_manifest.json'),source_sha256=digest(__file__)))


def train(kind,structure):
    configure();protocol=json.loads((OUT/'protocol.json').read_text());name=kind+'_'+structure
    ckpt=OUT/'checkpoints'/f'{name}.pt';assert not ckpt.exists()
    assert digest(__file__)==protocol['source_sha256'],'Frozen training source changed'
    model=RepairMAT(structure).cuda().eval()
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(params,lr=protocol['lr'],betas=(.9,.99))
    data=rows('train');history=[];best=float('inf');started=time.monotonic()
    for step,idx in enumerate(protocol['schedule'],1):
        s=sample(data[idx],kind);opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            a,b=model(s,SEED+step)
            loss=representation_loss(a,s)+.25*representation_loss(b,s)
        assert torch.isfinite(loss),(name,step,'loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.)
        assert torch.isfinite(norm),(name,step,'gradient')
        opt.step()
        if step==1 or step%50==0:
            entry=dict(step=step,loss=float(loss),gradient_norm=float(norm),seconds=time.monotonic()-started)
            history.append(entry);print(name,json.dumps(entry),flush=True)
        if step in protocol['validation_steps']:
            score=validation(model,kind);entry=dict(step=step,validation_loss=score,seconds=time.monotonic()-started)
            history.append(entry);print('VAL',name,json.dumps(entry),flush=True)
            if score<best:
                best=score;ckpt.parent.mkdir(parents=True,exist_ok=True)
                torch.save(dict(state={n:v.detach().cpu() for n,v in model.state_dict().items()},kind=kind,structure=structure,
                           step=step,val_loss=score,protocol_sha256=digest(OUT/'protocol.json')),ckpt)
            dump(OUT/'logs'/f'{name}.json',dict(name=name,history=history,best_validation_loss=best,
                 completed_steps=step,all_gradients_finite=True,trainable_parameters=sum(p.numel() for p in params),
                 peak_gib=torch.cuda.max_memory_allocated()/2**30))
    print('COMPLETE',name,flush=True)


def audit_benchmark():
    configure();r=rows('val')[0];checks={}
    for structure in STRUCTURES:
        model=RepairMAT(structure).cuda().eval();s=sample(r,'hed_soft')
        a=prediction(model,s,123)
        changed={k:v.clone() if torch.is_tensor(v) else v for k,v in s.items()}
        changed['target']=torch.rand_like(changed['target']);changed['skeleton']=torch.rand_like(changed['skeleton'])
        # Arbitrary hidden input also cannot reach MAT or refiner.
        changed['input']=torch.where(s['missing'],torch.rand_like(s['input']),s['input'])
        b=prediction(model,changed,123)
        assert np.array_equal(a,b)
        mask=s['missing'][0,0].cpu().numpy();known=s['input'][0,0].cpu().numpy()
        assert np.array_equal(a[~mask],known[~mask])
        opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4)
        times=[]
        for i in range(2):
            start=time.monotonic();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                x,y=model(s,SEED+i);loss=representation_loss(x,s)+.25*representation_loss(y,s)
            assert torch.isfinite(loss);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
            opt.step();torch.cuda.synchronize();times.append(time.monotonic()-start)
        checks[structure]=dict(hidden_truth_invariance=0,known_unchanged=True,seconds=times,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        del model,opt;torch.cuda.empty_cache()
    dump(OUT/'audit_benchmark.json',checks);print(checks,flush=True)


def geometry(edge,kind,parameter):
    if 'scribble' in kind:
        k=int(parameter)
        return cv2.dilate(edge.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))).astype(np.float32)
    distance=cv2.distanceTransform((~edge).astype(np.uint8),cv2.DIST_L2,3)
    return np.exp(-.5*(distance/parameter)**2).astype(np.float32)


def converter_train(steps=1200):
    configure();assert not (OUT/'checkpoints/converter.pt').exists()
    torch.manual_seed(SEED);model=Pyramid(1,4).cuda().train()
    opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    data=rows('train');cache=[]
    for r in data:
        dest=(OUT/r['input_path']).parent
        x=np.asarray(Image.open(dest/'canny.png')).astype(np.float32)/255
        y=np.stack([np.asarray(Image.open(dest/f'{k}.png')).astype(np.float32)/255 for k in STYLES])
        cache.append((torch.from_numpy(x)[None],torch.from_numpy(y)))
    rng=np.random.default_rng(SEED);best=float('inf');history=[];start=time.monotonic()
    for step in range(1,steps+1):
        indices=rng.integers(len(cache),size=4)
        x=torch.stack([cache[i][0] for i in indices]).cuda();y=torch.stack([cache[i][1] for i in indices]).cuda()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(x).float();loss=F.binary_cross_entropy_with_logits(logits,y)+.5*F.l1_loss(logits.sigmoid(),y)
        assert torch.isfinite(loss);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step()
        if step%100==0:print('CONVERTER',step,float(loss),round(time.monotonic()-start,1),flush=True)
        if step in (steps//2,steps):
            values=[];model.eval()
            with torch.no_grad():
                for r in rows('val'):
                    dest=(OUT/r['input_path']).parent
                    x=torch.from_numpy(np.asarray(Image.open(dest/'canny.png')).copy()).float()[None,None].cuda()/255
                    y=torch.from_numpy(np.stack([np.asarray(Image.open(dest/f'{k}.png')).astype(np.float32)/255 for k in STYLES]))[None].cuda()
                    with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x).float()
                    values.append(float(F.binary_cross_entropy_with_logits(logits,y)+.5*F.l1_loss(logits.sigmoid(),y)))
            score=float(np.mean(values));assert np.isfinite(score)
            history.append(dict(step=step,validation_loss=score))
            if score<best:
                best=score;torch.save(dict(state={n:v.detach().cpu() for n,v in model.state_dict().items()},step=step,
                    validation_loss=score,styles=STYLES,protocol_sha256=digest(OUT/'protocol.json')),OUT/'checkpoints/converter.pt')
            model.train()
    dump(OUT/'converter_train.json',dict(steps=steps,batch_size=4,lr=3e-4,history=history,all_gradients_finite=True,
         parameters=sum(p.numel() for p in model.parameters()),task='complete binary Canny -> four RGB-derived teacher maps, train supervised, no RGB input',
         caveat='learned statistical prediction, not inverse recovery of original edge strength/ownership'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['extract','audit','freeze','train','converter'])
    p.add_argument('--kind',choices=KINDS);p.add_argument('--structure',choices=STRUCTURES);p.add_argument('--steps',type=int,default=800)
    a=p.parse_args()
    if a.action=='extract':extract()
    elif a.action=='audit':audit_benchmark()
    elif a.action=='freeze':freeze(a.steps)
    elif a.action=='train':train(a.kind,a.structure)
    elif a.action=='converter':converter_train()
