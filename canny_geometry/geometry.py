"""Original canonical coordinates and displacement fields."""
import numpy as np
from . import matcher as M
def matcher():
    return M

# From canny_curve_stage2_bit.py:26; numerical body unchanged.
def canonical(samples):
    p=np.asarray(samples,float)
    return p if tuple(p[0])<tuple(p[-1]) else p[::-1].copy()

# From canny_curve_stage2_bit.py:30; numerical body unchanged.
def fields(samples,mode,m=None):
    if m is None:m=matcher()
    p=canonical(samples);u=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];u/=u[-1]
    t=m.tangent(p);n=np.c_[-t[:,1],t[:,0]]
    if mode=='normal':
        v=np.minimum(u,1-u)/.25;b=np.sin(np.clip(v,0,1)*np.pi/2)**2
    elif mode=='sagitta':
        chord=p[-1]-p[0];chord/=np.linalg.norm(chord);n=np.tile([-chord[1],chord[0]],(len(p),1));b=4*u*(1-u)
    else:b=6*np.sqrt(3)*u*(1-u)*(2*u-1)
    return p,u,n,b

# From curve_receiver_stage2c.py:11; numerical body unchanged.
def geometry(source,basis_name='S'):
    p=canonical(source);u=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];u/=u[-1]
    tangent=M.tangent(p);normal=np.c_[-tangent[:,1],tangent[:,0]]
    phi=6*np.sqrt(3)*u*(1-u)*(2*u-1) if basis_name=='S' else 16*u*u*(1-u)**2
    return p,u,normal,phi
