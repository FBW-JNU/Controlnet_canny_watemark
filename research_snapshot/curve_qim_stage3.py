"""Single-bit keyed-coordinate QIM, with a distinct quartic geometric basis."""
from __future__ import annotations
import hashlib
import numpy as np
import cv2
from scipy.spatial import cKDTree
from curve_receiver_stage2c import geometry,binary_target,oracle_mask

KEY='stage3-prototype-key-20261110'
DELTAS=[1.,1.5,2.]

def phase(content_key,delta,key=KEY):
    digest=hashlib.sha256((key+'|'+content_key).encode()).digest();u=int.from_bytes(digest[:8],'big')/2**64
    return (2*u-1)*delta

def quantize(c,bit,delta,dither):
    return float(dither+(2*np.floor(((c-dither)/delta-bit)/2+.5)+bit)*delta)

def decode(c,delta,dither):
    z=(c-dither)/delta;d0=abs(z-2*np.floor(z/2+.5));d1=abs(z-(2*np.floor((z-1)/2+.5)+1))
    return dict(bit=int(d1<d0),bit_score=float(d0-d1),lattice_residual=float(min(d0,d1)))

def warp_quartic(edge,source,mask,amplitude):
    p,u,n,phi=geometry(source,'QIM');yy,xx=np.nonzero(edge);xy=np.c_[xx,yy];dist,near=cKDTree(p).query(xy);chosen=(dist<=1.25)&mask[yy,xx]
    moved=xy.copy();moved[chosen]=np.rint(xy[chosen]+amplitude*phi[near[chosen],None]*n[near[chosen]]).astype(int)
    if np.any((moved<0)|(moved>=512)):raise ValueError('out-of-canvas geometric edit')
    out=edge.copy();out[yy[chosen],xx[chosen]]=0;out[moved[:,1],moved[:,0]]=255;lookup=np.full(edge.shape,-1,int);lookup[yy,xx]=np.arange(len(xy))
    for dx,dy in [(1,0),(0,1),(1,1),(-1,1)]:
        nx=xx+dx;ny=yy+dy;valid=(nx>=0)&(nx<512)&(ny>=0)&(ny<512);ids=np.flatnonzero(valid);other=lookup[ny[valid],nx[valid]];keep=other>=0;ids=ids[keep];other=other[keep];keep=chosen[ids]|chosen[other]
        for a,b in zip(ids[keep],other[keep]):cv2.line(out,tuple(moved[a]),tuple(moved[b]),255,1,cv2.LINE_8)
    out[~mask]=edge[~mask];assert np.array_equal(out[~mask],edge[~mask]);return out

def encode(edge,source,mask,content_key,delta,bit):
    initial=oracle_mask(source,binary_target(edge,source),'QIM')
    if not initial.get('accepted'):return edge.copy(),dict(input_success=False,reason='clean_input_not_measurable',c_original=initial.get('c_hat'),delta=delta,bit=bit)
    c=initial['c_hat'];dither=phase(content_key,delta);target=quantize(c,bit,delta,dither);ideal=target-c;attempts=[];best=None
    # Encoder feedback measures the raster with a bit-blind receiver. It never
    # changes the selected lattice point and never reads generated RGB.
    for correction in [0,-.15,.15,-.3,.3,-.45,.45,-.6,.6]:
        alpha=ideal+correction
        if abs(alpha)>2.65:continue
        image=warp_quartic(edge,source,mask,alpha);r=oracle_mask(source,binary_target(image,source),'QIM')
        if not r.get('accepted'):attempts.append(dict(amplitude=alpha,valid=False));continue
        decoded=decode(r['c_hat'],delta,dither);error=abs(r['c_hat']-target);attempts.append(dict(amplitude=alpha,valid=True,c_measured=r['c_hat'],error=error,**decoded))
        if best is None or error<best[0]:best=(error,image,r,alpha,decoded)
        if decoded['bit']==bit and error<=.2*delta:break
    if best is None:return edge.copy(),dict(input_success=False,reason='encoded_input_unmeasurable',delta=delta,bit=bit,attempts=attempts)
    error,image,r,alpha,decoded=best
    return image,dict(input_success=decoded['bit']==bit,c_original=c,c_target=target,c_measured=r['c_hat'],dither=dither,delta=delta,bit=bit,amplitude=alpha,
        quantization_error=error,input_decode=decoded,changed_pixels=int(np.count_nonzero(image!=edge)),attempts=attempts,basis='16t²(1-t)²',
        key_role='Key shifts the coefficient origin. Q0/Q1 are even/odd lattices in keyed coordinate c-dither. One bit is not authentication.')
