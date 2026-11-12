"""Recorded development amendment: threshold-aware thin-line constraints.

Preserves the complete first version. No receiver or content changes, no RGB
selection; frozen MAT weights, optimize W+ only, then audited projection.
"""
import mat_st_message_experiment as base
from mat_st_message_experiment import *
PARENT=OUT
OUT=PARENT/'constraints_v2'
LENGTHS=(32,)


def project_v2(edge,source,mask,prob,sign,origin):
    raw=(prob>=.1).astype(np.uint8)*255;raw[~mask]=edge[~mask]
    obs=observe(source,raw)
    pool=[r for r in obs['Auto'].get('decisions',[]) if r['coverage']>=.65 and r['span']>=.8 and r['residual_rms']<=1.5]
    if not pool:return raw,dict(success=False,reason='no_supported_MAT_curve_for_projection',observations=obs)
    chosen=min(pool,key=lambda r:r['cost']);p,u,n,_=geometry(source);ii=np.array(chosen['indices']);xy=np.array(chosen['observed_xy'])
    y=((xy-p[ii])*n[ii]).sum(1);res=y-chosen['offset']-basis(u[ii])@np.array(chosen['coefficient'])
    residual=np.interp(u,u[ii],res);x=np.c_[np.ones(len(u)),basis(u)]
    residual-=x@np.linalg.lstsq(x,residual,rcond=None)[0]
    residual*=min(.1,.1/max(np.max(abs(residual)),1e-6));residual*=np.sin(np.pi*u)**2
    desired=(np.zeros(2) if origin is None else np.array(origin))+AMP*sign
    shift=basis(u)@desired+residual;q=p+n*shift[:,None]
    line=np.zeros_like(edge);cv2.polylines(line,[np.rint(q).astype(np.int32).reshape(-1,1,2)],False,255,1)
    out=raw.copy();out[mask]=line[mask];out[~mask]=edge[~mask]
    return out,dict(success=True,raw_branch=chosen['branch'],raw_coefficient=chosen['coefficient'],raw_coverage=chosen['coverage'],
        retained_MAT_residual_rms=float(np.sqrt(np.mean(residual**2))),continuous_shift_peak=float(np.max(abs(shift))),projected_path=q,
        projection_scope='One target curve in entire missing tube; extra predicted branches removed. Known exterior copied.')


def prepare_v2():
    base.verify();assert not OUT.exists(),'Preserve prior version2'
    OUT.mkdir()
    for n in ('source','cases','inputs','conditions','mat_raw','logs','panels','rgb','noise','records','output_canny'):(OUT/n).mkdir()
    cases=read(PARENT/'selection.json');dump(OUT/'selection.json',cases)
    for c in cases:
        prefix=c['key']+'_L32'
        for group in ('G0','G1'):
            shutil.copy2(PARENT/'conditions'/f'{prefix}_{group}.png',OUT/'conditions'/f'{prefix}_{group}.png')
        for word in WORDS:shutil.copy2(PARENT/'conditions'/f'{prefix}_G4_{word}.png',OUT/'conditions'/f'{prefix}_G4_{word}.png')
    protocol=read(PARENT/'protocol.json')
    protocol.update(stage='Version2 development amendment after input-only v1; NO RGB observed',length_ladder=[32],script_sha256=digest(__file__),
        amendment=dict(reason='Version1 normal softmax coefficient loss improved while binary output became thick and codewords unreadable; projected outputs also retained competing branches',
            prior_results='Parent full32/48 results retained; no content/receiver/nominal amplitude changes',
            new_constraint='BCE with foreground weight3 on (MAT logits - logit(.1))/.5 against thin parametric message ridge; foreground-area penalty; calibrated foreground moment coefficient loss; W+96 steps lr.03',
            projection='Require same real-MAT-path coverage/span/residual; discard extra branches ONLY inside missing tube, retain <=.1px high-order MAT residual. Apply same revised projection to G2/G3.',
            attribution='Projection is explicit geometric reconstruction and may dominate. Compare preprojection G3raw and G2 separately. No direct-G4 fallback.'))
    protocol['mat'].update(inference_optimization='Adam W+ only96steps lr.03; MAT params frozen',
        constraint='threshold-aware BCE(pos_weight3)+0.5 calibrated foreground moment MSE+0.5 relative foreground-area MSE+0.003 W+ MSE',
        projection=protocol['amendment']['projection'])
    dump(OUT/'protocol.json',protocol)
    paths=[Path(__file__),PARENT/'protocol.json',PARENT/'selection.json',PARENT/'mat_status.json',OUT/'selection.json',OUT/'protocol.json']
    protected={str(p):digest(p) for p in paths};protected.update(read(PARENT/'freeze.json'))
    for p in list((OUT/'conditions').glob('*.png')):protected[str(p)]=digest(p)
    dump(OUT/'freeze.json',protected);shutil.copy2(__file__,OUT/'source'/Path(__file__).name)
    print('V2_PREPARED',flush=True)


def run_v2():
    import torch
    import torch.nn.functional as F
    base.OUT=OUT
    base.verify();cv2.setNumThreads(1);model=base.load_mat();thr=float(np.log(.1/.9))
    for c in read(OUT/'selection.json'):
        key=c['key'];seg=c['segments']['32'];source=seg['samples'];edge=arr(c['clean_hint']);mask=arr(seg['mask'])>0
        prefix=key+'_L32';dest=OUT/'mat_raw'/prefix;dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():print('V2_REUSE',prefix,flush=True);continue
        p,u,n,_=geometry(source);b=basis(u);origin=seg['origins']['Auto']
        known=torch.from_numpy((~mask).astype(np.float32))[None,None].cuda()
        visible=torch.from_numpy(edge.astype(np.float32)/255)[None,None].cuda()*known;inp=(visible*2-1)*known
        z=torch.from_numpy(np.random.RandomState(240).randn(1,512).astype(np.float32)).cuda()
        with torch.no_grad():w0=model.base.mapping(z,torch.zeros(1,0,device='cuda')).detach()
        def forward(w):
            torch.manual_seed(240);torch.cuda.manual_seed_all(240)
            with torch.autocast('cuda',dtype=torch.bfloat16):raw,_=model.base.synthesis(inp,known,w,noise_mode='const',return_stg1=True)
            return raw.float()*3
        with torch.no_grad():logit0=forward(w0);prob0=logit0.sigmoid()[0,0].cpu().numpy()
        np.save(dest/'G1_prob.npy',prob0)
        d=np.linspace(-4,4,33);points=p[:,None,:]+n[:,None,:]*d[None,:,None]
        grid=torch.tensor(points/511*2-1,dtype=torch.float32,device='cuda')[None];dd=torch.tensor(d,dtype=torch.float32,device='cuda')
        x=torch.tensor(np.c_[np.ones(len(u)),b],dtype=torch.float32,device='cuda');pinv=torch.linalg.pinv(x)
        mt=torch.from_numpy(mask)[None,None].cuda()
        for word,sign in zip(WORDS,SIGNS):
            g2,detail2=project_v2(edge,source,mask,prob0,sign,origin);save(OUT/'conditions'/f'{prefix}_G2_{word}.png',g2)
            desired=(np.zeros(2) if origin is None else origin)+AMP*sign
            qpath=p+n*(b@desired)[:,None];target=np.zeros_like(edge)
            cv2.polylines(target,[np.rint(qpath).astype(np.int32).reshape(-1,1,2)],False,255,1)
            target_t=torch.from_numpy((target>0).astype(np.float32))[None,None].cuda()
            desired_t=torch.tensor(desired,dtype=torch.float32,device='cuda')
            w=w0.clone().requires_grad_(True);opt=torch.optim.Adam([w],lr=.03);history=[]
            for step in range(97):
                logits=forward(w);centered=(logits-thr)/.5;binary_soft=centered.sigmoid()
                cross=F.grid_sample(binary_soft,grid,align_corners=True)[0,0]
                weights=(cross+1e-6)/(cross.sum(1,keepdim=True)+33e-6);mean=(weights*dd).sum(1);coef=pinv@mean
                shape=F.binary_cross_entropy_with_logits(centered[mt],target_t[mt],pos_weight=centered.new_tensor(3.))
                coeff=((coef[1:]-desired_t)**2).mean()
                area=((binary_soft[mt].sum()-target_t[mt].sum())/target_t[mt].sum().clamp_min(1))**2
                latent=((w-w0)**2).mean();loss=shape+.5*coeff+.5*area+.003*latent
                if step in (0,24,48,96):history.append(dict(step=step,loss=float(loss.detach()),shape=float(shape.detach()),coefficient=float(coeff.detach()),area=float(area.detach()),soft_coefficient=coef.detach().cpu().numpy()))
                if step==96:break
                opt.zero_grad();loss.backward();assert torch.isfinite(w.grad).all();opt.step()
            prob=logits.detach().sigmoid()[0,0].cpu().numpy();raw=np.where(mask,prob>=.1,edge>0).astype(np.uint8)*255
            g3,detail3=project_v2(edge,source,mask,prob,sign,origin)
            save(OUT/'conditions'/f'{prefix}_G3_{word}.png',g3);save(dest/f'G3_preprojection_{word}.png',raw)
            np.save(dest/f'G3_probability_{word}.npy',prob);np.save(dest/f'G3_W_{word}.npy',w.detach().cpu().numpy())
            g1=arr(OUT/'conditions'/f'{prefix}_G1.png')
            detail=dict(key=key,length=32,word=word,G2=detail2,G3=detail3,history=history,
                latent_rms=float(torch.sqrt(((w-w0)**2).mean()).detach()),raw_vs_G1_pixels=int(np.count_nonzero(raw!=g1)),soft_vs_G1_mae=float(abs(prob[mask]-prob0[mask]).mean()),
                G3_preprojection_observations=observe(source,raw),foreground_pixels=int(np.count_nonzero(raw[mask])),target_ridge_pixels=int(np.count_nonzero(target[mask])))
            dump(dest/f'{word}.json',detail)
            print('V2_MAT',key,word,'area',detail['foreground_pixels'],'/',detail['target_ridge_pixels'],'project',detail3['success'],flush=True)
        dump(dest/'complete.json',dict(completed=True,hidden_input_erased=True))
    gate=base.score_input(32);dump(OUT/'input_gate_L32.json',gate)
    dump(OUT/'mat_status.json',dict(complete=True,gate_passed_lengths=[32] if gate['passed'] else []))
    print('V2_INPUT_GATE',gate['passed'],gate['groups'],flush=True)
    base.make_plan()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare_v2','run_v2'));a=p.parse_args();globals()[a.action]()
