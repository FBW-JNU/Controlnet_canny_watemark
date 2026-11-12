"""Optional frozen domain MAT repair. No message-constrained W+ optimizer.
Architectures/forward extracted from mat_hed_pidi_repair.py; checkpoint state
is loaded directly, so public RGB initialization is unnecessary for inference.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

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

def distance_feature(array,missing):
    known_edges=(array>=.5)&~missing
    return np.minimum(cv2.distanceTransform((~known_edges).astype(np.uint8),cv2.DIST_L2,3),64)/64

class RepairMAT(nn.Module):
    def __init__(self,structure):
        super().__init__()
        from networks.mat import Generator
        self.base=Generator(z_dim=512,c_dim=0,w_dim=512,img_resolution=512,
                            img_channels=1,synthesis_kwargs={'use_noise':False})
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--canny',required=True);p.add_argument('--mask',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--protocol',required=True)
    p.add_argument('--threshold',type=float,required=True,help='Frozen validation threshold; not fitted on evaluation images')
    p.add_argument('--converter',help='Required for pidi_soft checkpoints; erase Canny before conversion')
    p.add_argument('--structure',choices=['native','refined'],default='native')
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=240)
    p.add_argument('--out',required=True);a=p.parse_args()
    if not 0<a.threshold<1:raise ValueError('Threshold must be between zero and one.')
    out=Path(a.out)
    if out.exists():raise FileExistsError('Choose a new output directory.')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor/MAT'))
    from torch_utils.ops import bias_act,upfirdn2d
    for op in [bias_act,upfirdn2d]:op._inited=True;op._plugin=None
    ck=torch.load(a.checkpoint,map_location='cpu')
    if ck.get('kind') not in ['canny','pidi_soft']:
        raise ValueError('Only canny or pidi_soft domain checkpoints are supported.')
    if ck.get('structure')!=a.structure:
        raise ValueError('Structure differs from checkpoint metadata.')
    if bool(a.converter)!=(ck['kind']=='pidi_soft'):
        raise ValueError('Use a converter exactly for the pidi_soft branch.')
    protocol_hash=hashlib.sha256(Path(a.protocol).read_bytes()).hexdigest()
    if ck['protocol_sha256']!=protocol_hash:raise ValueError('Training protocol hash mismatch.')
    model=RepairMAT(a.structure)
    model.load_state_dict(ck['state'],strict=True)
    model.to(a.device).eval().requires_grad_(False)
    edge=np.asarray(Image.open(a.canny).convert('L'))
    missing=np.asarray(Image.open(a.mask).convert('L'))>0
    if edge.shape!=(512,512) or missing.shape!=edge.shape:raise ValueError('Use matching 512x512 input and mask.')
    if not set(np.unique(edge)).issubset({0,255}):raise ValueError('Use binary Canny.')
    damaged=edge.astype(np.float32)/255;damaged[missing]=0
    with torch.inference_mode():
        if a.converter:
            cc=torch.load(a.converter,map_location='cpu')
            if cc['protocol_sha256']!=protocol_hash:raise ValueError('Converter protocol hash mismatch.')
            if tuple(cc['styles'])!=('hed_soft','pidi_soft','hed_scribble','pidi_scribble'):
                raise ValueError('Unexpected converter channel order.')
            converter=Pyramid(1,4).to(a.device).eval().requires_grad_(False)
            converter.load_state_dict(cc['state'],strict=True)
            # Teacher order: HED soft, PIDI soft, HED scribble, PIDI scribble.
            damaged=converter(torch.from_numpy(damaged)[None,None].to(a.device)).sigmoid()[0,1].cpu().numpy()
            damaged[missing]=0
        values=dict(input=damaged,missing=missing,distance=distance_feature(damaged,missing))
        sample={k:torch.from_numpy(v.copy())[None,None].to(a.device) for k,v in values.items()}
        logits,_=model(sample,a.seed);prob=logits.float().sigmoid()[0,0].cpu().numpy()
    # This CLI's output is a repaired binary Canny with exact known exterior.
    repaired=edge.copy();repaired[missing]=(prob[missing]>=a.threshold).astype(np.uint8)*255
    out.mkdir(parents=True)
    Image.fromarray(repaired).save(out/'repaired.png');np.save(out/'probability.npy',prob)
    (out/'repair.json').write_text(json.dumps(dict(
        checkpoint=Path(a.checkpoint).name,protocol_sha256=protocol_hash,
        threshold=a.threshold,seed=a.seed,structure=a.structure,
        converter=bool(a.converter),mask_white_means='missing',
        known_exterior_preserved=bool(np.array_equal(repaired[~missing],edge[~missing])),
        scope='Ordinary frozen repair diagnostic; no message constraint, no historical MAT benchmark reproduction'),indent=2)+'\n')


if __name__=='__main__':main()
