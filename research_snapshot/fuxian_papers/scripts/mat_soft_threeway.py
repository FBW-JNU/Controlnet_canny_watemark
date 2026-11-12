"""Isolated, paired MAT binary/soft/soft+orientation+visible-instance experiment.

Does not modify upstream MAT or retained baselines. GPU0 only.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from mat_canny_model import native_mat
from edge_data import prepare_rgb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/canny_middle50_global_local_20261106'
OUT = ROOT / 'results/MAE_MAT_related/mat_soft_canny_threeway_20261108'
WEIGHT = ROOT / 'checkpoints/mat/Places_512_FullData_G.pth'
ARMS = ('binary', 'soft', 'soft_dir_region')
SEED = 20261108


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_of(s):
    return int(hashlib.sha256(f'{SEED}|{s}'.encode()).hexdigest()[:8], 16)


def rows(split):
    return json.loads((OUT / 'manifest.json').read_text())['splits'][split]


def prepare():
    assert not (OUT / 'manifest.json').exists(), 'Frozen dataset already exists'
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    all_rows = {s: [json.loads(x) for x in (DATA / f'{s}.jsonl').read_text().splitlines()]
                for s in ('train', 'val', 'test')}
    wanted = {r['annotation_id'] for rs in all_rows.values() for r in rs if r.get('annotation_id')}
    annotations = {}
    for split in ('train2017', 'val2017'):
        obj = json.loads((ROOT.parent / 'coco/annotations' / f'instances_{split}.json').read_text())
        annotations.update({a['id']: a for a in obj['annotations']
                            if a['id'] in wanted and isinstance(a['segmentation'], list)})
    frozen, exclusions = {}, {}
    used_groups = set()
    # Small paired finetuning pilot; these counts are fixed before predictions.
    for split, count in (('train', 512), ('val', 64), ('test', 128)):
        selected, excluded = [], []
        candidates = sorted(all_rows[split], key=lambda r: seed_of(r['sample_id']))
        for r in candidates:
            if len(selected) == count:
                break
            aid = r.get('annotation_id')
            if aid not in annotations or r['content_group'] in used_groups:
                excluded.append([r['sample_id'], 'no_polygon_or_duplicate']); continue
            rgb, size, crop = prepare_rgb(r['source'], 512)
            edge = cv2.Canny(rgb, 100, 200, apertureSize=3, L2gradient=False) > 0
            assert hashlib.sha256((edge.astype(np.uint8)*255).tobytes()).hexdigest() == r['edge_pixel_sha256']
            region_img = Image.new('L', tuple(size), 0)
            draw = ImageDraw.Draw(region_img)
            for polygon in annotations[aid]['segmentation']:
                if len(polygon) >= 6:
                    draw.polygon(list(zip(polygon[::2], polygon[1::2])), fill=255)
            region = np.asarray(region_img.crop(tuple(crop)).resize((512,512), Image.Resampling.NEAREST)) > 0
            band = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((9,9),np.uint8)) > 0
            yy, xx = np.nonzero(band & edge)
            if len(xx) < 30:
                excluded.append([r['sample_id'], 'insufficient_visible_boundary']); continue
            side = (64, 96)[len(selected) % 2]
            rng = np.random.default_rng(seed_of(r['sample_id'] + '|mask'))
            chosen = None
            for idx in rng.permutation(len(xx))[:100]:
                left = int(np.clip(xx[idx] - side//2, 8, 504-side))
                top = int(np.clip(yy[idx] - side//2, 8, 504-side))
                if edge[top:top+side,left:left+side].sum() >= side:
                    chosen = left, top; break
            if chosen is None:
                excluded.append([r['sample_id'], 'insufficient_hole_edges']); continue
            left, top = chosen
            missing = np.zeros((512,512), bool)
            missing[top:top+side,left:left+side] = True
            soft = sum((cv2.Canny(rgb,t,2*t,apertureSize=3,L2gradient=False)>0).astype(np.float32)
                       for t in (40,60,80,100,120,140)) / 6
            # Match Canny L1 channel selection. Double angle removes contrast polarity.
            dx = cv2.Sobel(rgb, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(rgb, cv2.CV_32F, 0, 1, ksize=3)
            ix = np.argmax(np.abs(dx)+np.abs(dy), axis=2)[...,None]
            gx = np.take_along_axis(dx,ix,axis=2)[...,0]
            gy = np.take_along_axis(dy,ix,axis=2)[...,0]
            angle = np.arctan2(gy,gx)
            direction = np.stack([np.cos(2*angle)*soft, np.sin(2*angle)*soft])
            dest = OUT / 'data' / split / r['sample_id']
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest/'input.npz', edge=edge, soft=soft.astype(np.float16),
                                direction=direction.astype(np.float16), region=region, missing=missing)
            Image.fromarray(rgb).save(dest/'rgb.jpg', quality=92)
            record = {k:r[k] for k in ('sample_id','image_id','content_group','source','category','annotation_id','official_split','source_sha256')}
            record.update(split=split, crop=crop, size=size, box=[left,top,left+side,top+side], side=side,
                          input_path=str((dest/'input.npz').relative_to(OUT)), input_sha256=digest(dest/'input.npz'))
            selected.append(record); used_groups.add(r['content_group'])
        assert len(selected) == count, (split,len(selected))
        frozen[split] = selected; exclusions[split] = excluded
        print('PREPARED',split,len(selected),flush=True)
    dump(OUT/'manifest.json', dict(seed=SEED,splits=frozen,exclusions=exclusions,
        selection='stable hash order; polygon annotation; >=30 boundary Canny pixels; first feasible deterministic boundary-centred hole; no predictions used',
        soft='mean of RGB cv2.Canny(t,2t), t=40,60,80,100,120,140; not blurred binary',
        region='COCO selected instance polygon mask, visible pixels only; zero inside hole; oracle visible segmentation assistance',
        original_checkpoint_sha256=digest(WEIGHT)))


class ConditionalMAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)
        self.model, self.conversion = native_mat(WEIGHT, 'cpu')
        self.extra = None
        first = self.model.synthesis.first_stage.conv_first.conv
        second = self.model.synthesis.enc.EncConv_Block_512x512.conv0
        for conv in (first, second):
            old = conv.weight.detach()
            # Retain original fan-in scaling exactly; zero extra weights.
            conv.weight = torch.nn.Parameter(torch.cat([old,torch.zeros(old.shape[0],3,*old.shape[2:])],1))
            conv.register_forward_pre_hook(self._condition)
        self.model.synthesis.requires_grad_(True)
        self.model.mapping.requires_grad_(False)

    def _condition(self, module, args):
        return (torch.cat([args[0],self.extra.to(args[0].dtype)],1),)

    def forward(self, sample, arm, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        known = (~sample['missing']).float()
        edge = sample['edge'] if arm == 'binary' else sample['soft']
        extra = torch.cat([sample['direction'],sample['region']],1)
        self.extra = extra * known if arm == 'soft_dir_region' else torch.zeros_like(extra)
        z = torch.from_numpy(np.random.RandomState(seed).randn(1,512).astype(np.float32)).to(edge.device)
        image = (edge*2-1)*known
        return self.model(image, known, z, torch.zeros(1,0,device=edge.device),
                          noise_mode='const', return_stg1=True)


def load_sample(row, device='cuda:0'):
    with np.load(OUT/row['input_path']) as d:
        result = {k:torch.from_numpy(d[k].copy()).to(device) for k in d.files}
    return {k:(v[None] if k=='direction' else v[None,None]).bool() if k=='missing'
            else (v[None] if k=='direction' else v[None,None]).float() for k,v in result.items()}


def loss_for(raw, sample):
    selected = sample['missing']
    y = sample['edge'][selected]
    logits = raw.float()[selected]*3
    bce = F.binary_cross_entropy_with_logits(logits,y,pos_weight=logits.new_tensor(3.))
    p = logits.sigmoid()
    dice = 1-(2*(p*y).sum()+1)/(p.sum()+y.sum()+1)
    return bce + .5*dice


def configure():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True


def benchmark():
    configure()
    model = ConditionalMAT().cuda().eval()
    sample = load_sample(rows('train')[0])
    opt = torch.optim.Adam(model.model.synthesis.parameters(),lr=1e-4,betas=(.9,.99))
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    times=[]
    for step in range(3):
        start=time.monotonic(); opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            a,b=model(sample,'soft_dir_region',SEED+step)
            loss=loss_for(a,sample)+.5*loss_for(b,sample)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        assert torch.isfinite(norm), 'Nonfinite benchmark gradient'
        scaler.step(opt); scaler.update(); torch.cuda.synchronize()
        times.append(time.monotonic()-start)
        assert torch.isfinite(loss), 'Nonfinite benchmark loss'
        print('BENCH',step,float(loss),times[-1],torch.cuda.max_memory_allocated()/2**30,flush=True)
    dump(OUT/'benchmark.json',dict(seconds=times,peak_gib=torch.cuda.max_memory_allocated()/2**30,
                                  parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))


@torch.no_grad()
def predict(model, sample, arm, seed):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        raw,_=model(sample,arm,seed)
    return torch.where(sample['missing'],(raw.float()*3).sigmoid(),sample['edge'])


@torch.no_grad()
def validate(model, arm, subset=None):
    losses=[]
    for row in (rows('val') if subset is None else rows('val')[:subset]):
        s=load_sample(row)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            a,b=model(s,arm,seed_of(row['sample_id']+'|forward'))
        losses.append(float(loss_for(a,s)))
    return float(np.mean(losses))


def train(arm, steps):
    configure()
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert protocol['steps_per_arm']==steps
    checkpoint=OUT/'checkpoints'/f'{arm}.pt'
    assert not checkpoint.exists(), f'Already trained: {arm}'
    model=ConditionalMAT().cuda().eval()
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=protocol['lr'],betas=(.9,.99))
    scaler=torch.cuda.amp.GradScaler(enabled=False)
    schedule=protocol['schedule']; data=rows('train')
    best=float('inf'); history=[]; started=time.monotonic(); finite=True
    for step,idx in enumerate(schedule,1):
        s=load_sample(data[idx]); opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            a,b=model(s,arm,SEED+step)
            loss=loss_for(a,s)+.5*loss_for(b,s)
        assert torch.isfinite(loss), (arm,step,loss)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        finite = finite and bool(torch.isfinite(norm))
        assert finite, (arm,step,'Nonfinite gradient')
        scaler.step(opt); scaler.update()
        if step==1 or step%25==0:
            entry=dict(step=step,loss=float(loss),grad_norm=float(norm),scale=scaler.get_scale(),seconds=time.monotonic()-started)
            history.append(entry); print(arm,json.dumps(entry),flush=True)
        if step%protocol['validation_every']==0 or step==steps:
            val=validate(model,arm)
            entry=dict(step=step,val_loss=val,seconds=time.monotonic()-started)
            history.append(entry); print('VAL',arm,json.dumps(entry),flush=True)
            if val<best:
                best=val
                state={n:v.detach().cpu() for n,v in model.state_dict().items()}
                checkpoint.parent.mkdir(parents=True,exist_ok=True)
                torch.save(dict(state=state,step=step,val_loss=val,arm=arm,protocol_sha256=digest(OUT/'protocol.json')),checkpoint)
            dump(OUT/f'train_{arm}.json',dict(arm=arm,history=history,best_val_loss=best,
                 all_gradients_finite=finite,trainable_names=names,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                 peak_gib=torch.cuda.max_memory_allocated()/2**30))
    print('TRAIN COMPLETE',arm,flush=True)


def freeze(steps):
    assert not (OUT/'protocol.json').exists()
    rng=np.random.default_rng(SEED)
    schedule=[]
    while len(schedule)<steps:
        schedule.extend(rng.permutation(len(rows('train'))).tolist())
    dump(OUT/'protocol.json',dict(seed=SEED,steps_per_arm=steps,schedule=schedule[:steps],lr=1e-4,
         validation_every=steps//3,batch_size=1,device='GPU0',amp='bfloat16; float32 parameters and losses; no GradScaler',
         initialization='same original Places512 -> native single-channel conversion; extra condition weights zero; original fan-in scaling retained',
         trainable='all synthesis parameters; frozen mapping; same parameter shapes/counts in all arms',
         model='full two-stage MAT512; 3 extra channels injected into first-stage and second-stage input convolutions; no upstream edits',
         output='binary Canny100/200; sigmoid(3*MAT raw); known pixels replaced by original binary Canny',
         loss='masked pos_weight=3 BCEWithLogits + 0.5 soft Dice; final + 0.5 first stage; no RGB loss',
         checkpoint_selection='minimum fixed 64-image validation final-output loss; same steps evaluated',
         threshold_selection='per-arm best strict-F1 on validation, grid 0.10..0.90 step0.05; test never used',
         manifest_sha256=digest(OUT/'manifest.json'),script_sha256=digest(__file__)))


def audit():
    configure()
    m=ConditionalMAT().cuda().eval()
    row=rows('val')[0];s=load_sample(row)
    checks={}
    for arm in ARMS:
        p=predict(m,s,arm,123)
        changed={k:v.clone() for k,v in s.items()}
        for k in ('edge','soft','direction','region'):
            changed[k]=torch.where(s['missing'],torch.rand_like(changed[k]),changed[k])
        q=predict(m,changed,arm,123)
        # Known output is composited from binary GT; hidden GT is never read.
        delta=float((p-q).abs().max())
        checks[arm]=dict(hidden_input_invariance_max_abs=delta,
                         known_pixels_unchanged=bool(torch.equal(p[~s['missing']],s['edge'][~s['missing']])) )
        assert delta==0 and checks[arm]['known_pixels_unchanged']
    hashes={}
    for split in ('train','val','test'):
        hashes[split]=all(digest(OUT/r['input_path'])==r['input_sha256'] for r in rows(split))
    groups=[{r['content_group'] for r in rows(s)} for s in ('train','val','test')]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    dump(OUT/'audit.json',dict(checks=checks,input_hashes_valid=hashes,split_groups_disjoint=True,
                              original_checkpoint_sha256=digest(WEIGHT)))
    print('AUDIT',checks,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','benchmark','freeze','audit','train'])
    p.add_argument('--arm',choices=ARMS);p.add_argument('--steps',type=int,default=600)
    args=p.parse_args()
    if args.action=='prepare': prepare()
    elif args.action=='benchmark': benchmark()
    elif args.action=='freeze': freeze(args.steps)
    elif args.action=='audit': audit()
    elif args.action=='train': train(args.arm,args.steps)
