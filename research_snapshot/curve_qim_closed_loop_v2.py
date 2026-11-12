"""Discrete QIM input closure. Frozen old encoder/receiver remain unchanged.

The input decoder is bit-blind; only the encoder selects a feasible codeword.
An active write is stronger than parity correctness and is reported separately.
"""
from __future__ import annotations
import hashlib
import numpy as np
import cv2
from scipy.spatial import cKDTree
from curve_qim_stage3 import phase, decode, quantize
from curve_receiver_stage2c import geometry, binary_target, oracle_mask, candidates, receive

CONFIG=dict(amplitude_limit=2.65,amplitude_step=.05,target_residual_fraction=.25,
            minimum_coefficient_change=.5,minimum_changed_pixels=4,
            minimum_moved_phi_information=.25)

class RasterWarp:
    """Same rasterizer as frozen warp_quartic, with geometry cached."""
    def __init__(self,edge,source,mask):
        self.edge=edge;self.mask=mask;self.source=source
        self.p,self.u,self.n,self.phi=geometry(source,'QIM')
        self.yy,self.xx=np.nonzero(edge);self.xy=np.c_[self.xx,self.yy]
        self.distance,self.near=cKDTree(self.p).query(self.xy)
        self.chosen=(self.distance<=1.25)&mask[self.yy,self.xx]
        self.vector=self.phi[self.near[self.chosen],None]*self.n[self.near[self.chosen]]
        self.links=[];lookup=np.full(edge.shape,-1,int);lookup[self.yy,self.xx]=np.arange(len(self.xy));h,w=edge.shape
        for dx,dy in [(1,0),(0,1),(1,1),(-1,1)]:
            nx=self.xx+dx;ny=self.yy+dy;valid=(nx>=0)&(nx<w)&(ny>=0)&(ny<h)
            ids=np.flatnonzero(valid);other=lookup[ny[valid],nx[valid]];keep=other>=0;ids=ids[keep];other=other[keep]
            keep=self.chosen[ids]|self.chosen[other];self.links.extend(zip(ids[keep],other[keep]))
    def render(self,alpha):
        moved=self.xy.copy();moved[self.chosen]=np.rint(self.xy[self.chosen]+alpha*self.vector).astype(int)
        if np.any(moved<0) or np.any(moved[:,0]>=self.edge.shape[1]) or np.any(moved[:,1]>=self.edge.shape[0]):raise ValueError('outside canvas')
        out=self.edge.copy();out[self.yy[self.chosen],self.xx[self.chosen]]=0;out[moved[:,1],moved[:,0]]=255
        for a,b in self.links:cv2.line(out,tuple(moved[a]),tuple(moved[b]),255,1,cv2.LINE_8)
        out[~self.mask]=self.edge[~self.mask]
        return out,moved
    def moved_information(self,moved):
        stations=np.unique(np.r_[np.arange(0,len(self.p),2),len(self.p)-1]).astype(int)
        d,near=cKDTree(self.xy).query(self.p[stations]);valid=(d<=1.25)&self.chosen[near]
        shifted=np.any(moved[near]!=self.xy[near],axis=1)&valid
        return float(np.sum(self.phi[stations][shifted]**2)/max(1e-9,np.sum(self.phi[stations]**2)))

def measure(image,source,receiver,thresholds,gate):
    target=binary_target(image,source)
    return oracle_mask(source,target,'QIM') if receiver=='OracleMask' else receive(candidates(source,target,thresholds,'QIM'),gate)

def bank(edge,source,mask,thresholds,gate):
    warp=RasterWarp(edge,source,mask);rows=[];rasters={}
    for alpha in np.round(np.arange(-CONFIG['amplitude_limit'],CONFIG['amplitude_limit']+.001,CONFIG['amplitude_step']),8):
        image,moved=warp.render(float(alpha));key=hashlib.sha256(image.tobytes()).hexdigest()
        if key not in rasters:
            observations={name:measure(image,source,name,thresholds,gate) for name in ['OracleMask','Auto']}
            rasters[key]=dict(image=image,observations=observations)
        rows.append(dict(amplitude=float(alpha),raster_hash=key,changed_pixels=int(np.count_nonzero(image!=edge)),
                         moved_phi_information=warp.moved_information(moved)))
    return rows,rasters

def select(rows,rasters,c_original,content_key,delta,bit,receiver,policy):
    dither=phase(content_key,delta);target=quantize(c_original,bit,delta,dither);pool=[]
    for row in rows:
        r=rasters[row['raster_hash']]['observations'][receiver]
        if not r.get('accepted'):continue
        c=r['c_hat'];decision=decode(c,delta,dither);nearest=quantize(c,bit,delta,dither)
        residual=abs(c-nearest);correct=decision['bit']==bit
        strict_target=abs(c-target)<=CONFIG['target_residual_fraction']*delta
        active=(row['changed_pixels']>=CONFIG['minimum_changed_pixels'] and abs(c-c_original)>=CONFIG['minimum_coefficient_change']
                and row['moved_phi_information']>=CONFIG['minimum_moved_phi_information'])
        stable=correct and residual<=CONFIG['target_residual_fraction']*delta
        eligible=correct if policy=='parity_only' else correct and strict_target if policy=='nearest_target' else stable and active
        rr=dict(**row,c_measured=c,c_original=c_original,c_target=target,chosen_codeword=nearest,delta=delta,bit=bit,dither=dither,
                input_decode=decision,correct_bit=correct,stable_parity=stable,active_write=active,eligible=eligible,
                quantization_error=residual,nearest_target_error=abs(c-target),codeword_changed=abs(nearest-target)>1e-6,
                receiver=receiver,policy=policy,coefficient_change=c-c_original)
        pool.append(rr)
    good=[r for r in pool if r['eligible']]
    if not good:return dict(input_success=False,write_success=False,reason='no_reachable_codeword_with_required_guards',
        c_original=c_original,c_target=target,delta=delta,bit=bit,receiver=receiver,policy=policy,
        best_available=min(pool,key=lambda r:r['nearest_target_error']) if pool else None)
    if policy=='active_same_parity':
        chosen=min(good,key=lambda r:(abs(r['chosen_codeword']-c_original),abs(r['amplitude']),r['quantization_error'],r['changed_pixels']))
    else:chosen=min(good,key=lambda r:(r['nearest_target_error'],abs(r['amplitude'])))
    return dict(**chosen,input_success=True,write_success=chosen['stable_parity'] and chosen['active_write'])
