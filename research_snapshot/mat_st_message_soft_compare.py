"""Paired soft-MAT G1/G2/G3 development experiment on frozen local carriers."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import sys
import time
import mat_st_message_experiment as base
from mat_st_message_experiment import *
from mat_st_message_constraints_v2 import project_v2

PARENT=ROOT/'outputs/mat_st_message_experiment1_20261111'
BINARY=PARENT/'constraints_v2'
OUT=ROOT/'outputs/mat_st_message_soft_compare_20261111'
STRUCTURES=('native','refined')
CONVERTER=MAT/'checkpoints/converter.pt'


def prepare():
    base.verify()
    assert read(BINARY/'completion_audit.json')['complete']
    assert not OUT.exists(),'Preserve existing experiment'
    OUT.mkdir();(OUT/'source').mkdir();(OUT/'logs').mkdir()
    cases=read(PARENT/'selection.json');thresholds=read(MAT/'thresholds.json')
    weights=[CONVERTER]+[MAT/'checkpoints'/f'pidi_soft_{s}.pt' for s in STRUCTURES]
    protocol=dict(task='G1/G2/G3 soft MAT versus frozen binary V2; same10contents,32px holes,40messages',
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),structures=STRUCTURES,
        conversion='Erase Canny first -> frozen learned converter pidi_soft head -> erase converted hole again -> soft MAT',
        missing_information='Neither converter nor MAT sees original hidden Canny. No RGB or target soft is given at inference.',
        final_binary='Use original binary Canny outside missing tube; threshold MAT probability only inside it. Also archive continuous soft composite.',
        constraints='Same binary V2 loss,96 W+ updates lr.03; threshold logit aligned to fixed soft threshold. No extra continuity term or receiver tuning.',
        mapping_precision='Soft W+ initialized under bfloat16 autocast to match official old RepairMAT.forward exactly; binary V2 initialized mapping outside autocast. A numerical initialization difference remains in addition to checkpoint/domain changes.',
        projection='Same project_v2 function on hard-thresholded MAT output; actual supported path required; no G4 fallback.',
        interpretation='Different old domain checkpoints, shared constraint settings. Tests this pipeline; does not isolate input softness alone or all soft models.',
        thresholds={s:thresholds[f'pidi_soft_{s}']['threshold'] for s in STRUCTURES},
        weights={str(p):digest(p) for p in weights},binary_reference=str(BINARY),
        visual_review_indices=[0,1,8,9],script_sha256=digest(__file__))
    dump(OUT/'protocol.json',protocol)
    protected=read(PARENT/'freeze.json')
    protected.update({str(p):digest(p) for p in weights+[Path(__file__),OUT/'protocol.json',BINARY/'input_measurements_L32.json',BINARY/'input_gate_L32.json',ROOT/'mat_st_message_constraints_v2.py',ROOT/'evaluate_mat_st_message_experiment.py']})
    for n in ('mat_hed_pidi_repair.py','mat_soft_threeway.py','mat_canny_model.py','mat_common.py'):
        p=ROOT/'fuxian_papers/scripts'/n;protected[str(p)]=digest(p)
    for structure in STRUCTURES:
        arm=OUT/structure
        for n in ('source','conditions','inputs','mat_raw','logs','panels','rgb','noise','records','output_canny'):(arm/n).mkdir(parents=True)
        dump(arm/'selection.json',cases)
        ap=read(BINARY/'protocol.json');ap.update(stage='Soft MAT paired development: '+structure,length_ladder=[32],soft_comparison=protocol)
        ap['groups'].update(G1='masked Canny -> learned approximate PiDiNet soft -> old pidi_soft '+structure+' MAT -> threshold, no message',
            G2='G1 then identical binary-V2 projection',G3='soft MAT message-constrained W+ then identical binary-V2 projection')
        ck=MAT/'checkpoints'/f'pidi_soft_{structure}.pt';thr=protocol['thresholds'][structure]
        ap['mat'].update(checkpoint=str(ck),checkpoint_sha256=digest(ck),threshold=thr,
            structure=structure,input='converted damaged Canny, masked again; visible-only distance; no teacher or RGB',
            constraint='Same V2 loss: threshold BCE(pos_weight3)+.5 coefficient MSE+.5 relative-area MSE+.003 W+ MSE',
            inference_optimization='Adam W+96steps lr.03; all network weights frozen, including converter/refiner')
        ap['escalation']='Only32px for paired comparison; keep all40. Diffusion only if all40 G3 attempts pass original gate.'
        dump(arm/'protocol.json',ap)
        files={}
        for c in cases:
            prefix=c['key']+'_L32'
            for suffix in ['G0']+[f'G4_{w}' for w in WORDS]:
                p=arm/'conditions'/f'{prefix}_{suffix}.png';shutil.copy2(BINARY/'conditions'/p.name,p);files[str(p)]=digest(p)
        files.update(protected);files.update({str(p):digest(p) for p in (arm/'selection.json',arm/'protocol.json')})
        dump(arm/'freeze.json',files)
        shutil.copy2(__file__,arm/'source'/Path(__file__).name)
    shutil.copy2(__file__,OUT/'source'/Path(__file__).name)
    dump(OUT/'RUN_STATUS.json',dict(status='prepared',structures=STRUCTURES))
    print('SOFT_PREPARED',flush=True)


def run(structure):
    import torch
    import torch.nn.functional as F
    arm=OUT/structure;base.OUT=arm;base.LENGTHS=(32,);base.verify()
    sys.path.insert(0,str(ROOT/'fuxian_papers/scripts'))
    from mat_hed_pidi_repair import RepairMAT,Pyramid,STYLES,distance_feature
    torch.set_num_threads(8);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=RepairMAT(structure).cuda().eval().requires_grad_(False)
    ap=read(arm/'protocol.json');ck=torch.load(ap['mat']['checkpoint'],map_location='cpu',weights_only=False)
    assert ck['protocol_sha256']==digest(MAT/'protocol.json');assert ck['step']==800
    model.load_state_dict(ck['state'],strict=True)
    converter=Pyramid(1,4).cuda().eval().requires_grad_(False)
    cc=torch.load(CONVERTER,map_location='cpu',weights_only=False)
    assert cc['protocol_sha256']==digest(MAT/'protocol.json');converter.load_state_dict(cc['state'],strict=True)
    threshold=ap['mat']['threshold'];threshold_logit=float(np.log(threshold/(1-threshold)));audits=[]
    def convert(edge,mask):
        damaged=np.where(mask,0,edge).astype(np.float32)/255
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            soft=converter(torch.from_numpy(damaged)[None,None].cuda()).float().sigmoid()[0,STYLES.index('pidi_soft')].cpu().numpy()
        soft[mask]=0
        return soft
    def project(edge,source,mask,prob,sign,origin):
        # project_v2 hard-codes .1; passing an already hard-thresholded map
        # reuses its exact path selection and projection without changing it.
        return project_v2(edge,source,mask,(prob>=threshold).astype(np.float32),sign,origin)
    for c in read(arm/'selection.json'):
        key=c['key'];seg=c['segments']['32'];source=seg['samples'];edge=arr(c['clean_hint']);mask=arr(seg['mask'])>0
        prefix=key+'_L32';dest=arm/'mat_raw'/prefix;dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():
            audits.append(read(dest/'input_audit.json'));print('SOFT_REUSE',structure,key,flush=True);continue
        soft=convert(edge,mask);np.save(arm/'inputs'/f'{prefix}_masked_soft.npy',soft)
        save(arm/'inputs'/f'{prefix}_masked_soft.png',np.rint(soft*255))
        altered=edge.copy();altered[mask]=255-altered[mask]
        assert np.array_equal(soft,convert(altered,mask))
        missing=torch.from_numpy(mask)[None,None].cuda();known=(~missing).float()
        visible=torch.from_numpy(soft)[None,None].cuda()*known;inp=(visible*2-1)*known
        dist=torch.from_numpy(distance_feature(soft,mask))[None,None].cuda()
        z=torch.from_numpy(np.random.RandomState(240).randn(1,512).astype(np.float32)).cuda()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            w0=model.base.mapping(z,torch.zeros(1,0,device='cuda')).detach().float()
        def forward(w):
            torch.manual_seed(240);torch.cuda.manual_seed_all(240)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                raw,coarse=model.base.synthesis(inp,known,w,noise_mode='const',return_stg1=True)
                logits=raw.float()*3
                if model.refine is not None:
                    features=torch.cat([visible,missing.float(),logits.sigmoid(),(coarse.float()*3).sigmoid(),dist],1)
                    residual=model.refine(features).float();logits=logits+4*torch.tanh(residual/4)
            return logits
        with torch.no_grad():
            logit0=forward(w0);prob0=logit0.sigmoid()[0,0].cpu().numpy()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                check,_=model(dict(input=visible,missing=missing,distance=dist),240)
            forward_delta=float((check.float()-logit0).abs().max())
            assert forward_delta==0.,forward_delta
        np.save(dest/'G1_prob.npy',prob0)
        np.save(dest/'G1_soft_composite.npy',np.where(mask,prob0,soft))
        save(dest/'G1_soft_composite.png',np.rint(np.where(mask,prob0,soft)*255))
        g1=np.where(mask,prob0>=threshold,edge>0).astype(np.uint8)*255
        save(arm/'conditions'/f'{prefix}_G1.png',g1)
        audit=dict(key=key,hidden_canny_invariant=True,manual_W_forward_matches_model_exactly=True,forward_max_abs=forward_delta,source_RGB_not_model_input=True)
        dump(dest/'input_audit.json',audit);audits.append(audit)
        p,u,n,_=geometry(source);b=basis(u);origin=seg['origins']['Auto']
        d=np.linspace(-4,4,33);points=p[:,None,:]+n[:,None,:]*d[None,:,None]
        grid=torch.tensor(points/511*2-1,dtype=torch.float32,device='cuda')[None];dd=torch.tensor(d,dtype=torch.float32,device='cuda')
        pinv=torch.linalg.pinv(torch.tensor(np.c_[np.ones(len(u)),b],dtype=torch.float32,device='cuda'))
        for word,sign in zip(WORDS,SIGNS):
            started=time.monotonic();g2,detail2=project(edge,source,mask,prob0,sign,origin)
            save(arm/'conditions'/f'{prefix}_G2_{word}.png',g2)
            desired=(np.zeros(2) if origin is None else np.array(origin))+AMP*sign
            q=p+n*(b@desired)[:,None];target=np.zeros_like(edge)
            cv2.polylines(target,[np.rint(q).astype(np.int32).reshape(-1,1,2)],False,255,1)
            target_t=torch.from_numpy((target>0).astype(np.float32))[None,None].cuda();desired_t=torch.tensor(desired,dtype=torch.float32,device='cuda')
            w=w0.clone().requires_grad_(True);opt=torch.optim.Adam([w],lr=.03);history=[]
            for step in range(97):
                logits=forward(w);centered=(logits-threshold_logit)/.5;binary_soft=centered.sigmoid()
                cross=F.grid_sample(binary_soft,grid,align_corners=True)[0,0]
                weights=(cross+1e-6)/(cross.sum(1,keepdim=True)+33e-6);coef=pinv@((weights*dd).sum(1))
                shape=F.binary_cross_entropy_with_logits(centered[missing],target_t[missing],pos_weight=centered.new_tensor(3.))
                coeff=((coef[1:]-desired_t)**2).mean()
                area=((binary_soft[missing].sum()-target_t[missing].sum())/target_t[missing].sum().clamp_min(1))**2
                latent=((w-w0)**2).mean();loss=shape+.5*coeff+.5*area+.003*latent
                if step in (0,24,48,96):history.append(dict(step=step,loss=float(loss.detach()),shape=float(shape.detach()),coefficient=float(coeff.detach()),area=float(area.detach()),soft_coefficient=coef.detach().cpu().numpy()))
                if step==96:break
                opt.zero_grad();loss.backward();assert torch.isfinite(w.grad).all();opt.step()
            prob=logits.detach().sigmoid()[0,0].cpu().numpy();raw=np.where(mask,prob>=threshold,edge>0).astype(np.uint8)*255
            g3,detail3=project(edge,source,mask,prob,sign,origin)
            save(arm/'conditions'/f'{prefix}_G3_{word}.png',g3);save(dest/f'G3_preprojection_{word}.png',raw)
            np.save(dest/f'G3_probability_{word}.npy',prob);np.save(dest/f'G3_W_{word}.npy',w.detach().cpu().numpy())
            save(dest/f'G3_soft_{word}.png',np.rint(np.where(mask,prob,soft)*255))
            detail=dict(key=key,length=32,word=word,G2=detail2,G3=detail3,history=history,
                latent_rms=float(torch.sqrt(((w-w0)**2).mean()).detach()),raw_vs_G1_pixels=int(np.count_nonzero(raw!=g1)),soft_vs_G1_mae=float(abs(prob[mask]-prob0[mask]).mean()),
                G3_preprojection_observations=observe(source,raw),foreground_pixels=int(np.count_nonzero(raw[mask])),target_ridge_pixels=int(np.count_nonzero(target[mask])),seconds=time.monotonic()-started)
            dump(dest/f'{word}.json',detail)
            print('SOFT_MAT',structure,key,word,'area',detail['foreground_pixels'],'/',detail['target_ridge_pixels'],'project',detail3['success'],'seconds',round(detail['seconds'],1),flush=True)
        dump(dest/'complete.json',dict(completed=True,hidden_input_erased=True))
    gate=base.score_input(32);dump(arm/'input_gate_L32.json',gate)
    dump(arm/'mat_status.json',dict(complete=True,gate_passed_lengths=[32] if gate['passed'] else []))
    dump(arm/'input_audit.json',audits);base.make_plan();base.verify()
    print('SOFT_INPUT_GATE',structure,gate['passed'],gate['groups'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'));parser.add_argument('--structure',choices=STRUCTURES,default='native');args=parser.parse_args()
    prepare() if args.action=='prepare' else run(args.structure)
