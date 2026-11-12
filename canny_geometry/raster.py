"""Original binary pixel movement and neighbor reconnection."""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from .geometry import geometry

# From curve_qim_closed_loop_v2.py:18; numerical body unchanged.
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
