"""Read-only tracing of the frozen QIM experiment; new outputs only."""
from pathlib import Path
import json, hashlib, argparse
import numpy as np
import cv2
from PIL import Image
from scipy.spatial import cKDTree
from curve_receiver_stage2c import geometry, binary_target, oracle_mask
from curve_qim_stage3 import decode, phase, warp_quartic

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'outputs/canny_curve_stage2c_qim1_20261110'
OUT=ROOT/'outputs/canny_qim_signal_trace_20261110'
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def arr(p):return np.array(Image.open(p).convert('L'))
def tensor(a):return np.ascontiguousarray(np.repeat(a[:,:,None],3,2).astype(np.float32).transpose(2,0,1)[None]/255.)
def harray(a):return hashlib.sha256(a.tobytes()).hexdigest()

def trace():
    OUT.mkdir(exist_ok=True)
    cc={c['key']:c for c in read(OLD/'fresh/selection.json')};states=[]
    for r in read(OLD/'fresh/qim_inputs.json'):
        c=cc[r['key']];edge=arr(c['clean_hint']);marked=arr(r['hint']);mask=arr(c['mask'])>0;p,u,n,phi=geometry(c['samples'],'QIM')
        yy,xx=np.nonzero(edge);xy=np.c_[xx,yy];distance,near=cKDTree(p).query(xy);chosen=(distance<=1.25)&mask[yy,xx]
        alpha=r['amplitude'];floating=alpha*phi[near[chosen],None]*n[near[chosen]];integer=np.rint(xy[chosen]+floating).astype(int)-xy[chosen]
        clean=oracle_mask(c['samples'],binary_target(edge,c['samples']),'QIM');obs=np.array(clean['observed_xy']);oi=np.array(clean['indices']);obs_dist,obs_near=cKDTree(p).query(obs);editable=(obs_dist<=1.25)&mask[obs[:,1].astype(int),obs[:,0].astype(int)]
        ideal=r['c_target']-r['c_original'];ideal_vector=ideal*phi[:,None]*n;initial_decode=decode(r['c_original'],r['delta'],r['dither'])
        actual=oracle_mask(c['samples'],binary_target(marked,c['samples']),'QIM')
        obs_shared=len(set(map(tuple,obs))&set(map(tuple,actual.get('observed_xy',[]))))/len(obs)
        assert np.array_equal(warp_quartic(edge,c['samples'],mask,alpha),marked)
        row=dict(**r,stratum=c['stratum'],ideal_amplitude=ideal,continuous_peak_displacement=float(np.max(np.linalg.norm(ideal_vector,axis=1))),
            chosen_original_pixels=int(chosen.sum()),pixels_with_nonzero_integer_move=int(np.any(integer!=0,axis=1).sum()),
            largest_selected_component_displacement=float(np.max(np.abs(floating))) if len(floating) else 0.,
            actual_peak_vector_displacement=float(np.max(np.linalg.norm(floating,axis=1))) if len(floating) else 0.,
            input_identity=bool(np.array_equal(edge,marked)),outside_mask_unchanged=bool(np.array_equal(edge[~mask],marked[~mask])),
            clean_already_decodes_intended=initial_decode['bit']==r['bit'],clean_decode=initial_decode,
            clean_observed_pixels_editable_fraction=float(editable.mean()),
            clean_observed_phi_information_editable_fraction=float(np.sum(phi[oi][editable]**2)/np.sum(phi[oi]**2)),
            old_input_observation_shared_fraction=obs_shared,
            clean_tensor_sha256=harray(tensor(edge)),marked_tensor_sha256=harray(tensor(marked)),
            old_trial_has_correct_bit=any(a.get('bit')==r['bit'] for a in r['attempts']))
        states.append(row)
    index={(r['key'],r['bit']):r for r in states};jobs=[]
    plan=read(OLD/'fresh/qim_plan.json');cleanplan={(r['key'],r['view']):r for r in plan if r['arm']=='clean'}
    for j in plan:
        if j['arm']=='clean':continue
        r=index[j['key'],int(j['arm'][-1])];clean=cleanplan[j['key'],j['view']]
        rec=read(OLD/'fresh/records'/f'{j["job"]}.json');cr=read(OLD/'fresh/records'/f'{clean["job"]}.json')
        rgb_equal=np.array_equal(np.array(Image.open(j['rgb'])),np.array(Image.open(clean['rgb'])))
        row=dict(job=j['job'],key=j['key'],view=j['view'],bit=r['bit'],input_success=r['input_success'],
            continuous_peak_displacement=r['continuous_peak_displacement'],chosen_amplitude=r['amplitude'],integer_moved_pixels=r['pixels_with_nonzero_integer_move'],changed_pixels=r['changed_pixels'],
            input_png_equal=digest(j['hint'])==digest(clean['hint']),condition_tensor_equal=r['clean_tensor_sha256']==r['marked_tensor_sha256'],
            prompt_equal=j['prompt']==clean['prompt'],seed_equal=j['seed']==clean['seed'],initial_noise_equal=rec['noise_sha256']==cr['noise_sha256'],
            checkpoint_equal=rec['checkpoint_sha256']==cr['checkpoint_sha256'],rgb_bytes_equal=rec['rgb_sha256']==cr['rgb_sha256'],rgb_pixels_equal=rgb_equal,
            internal_diffusion_trace='not saved in original run; identical deterministic model inputs imply identical trajectory' if rgb_equal else 'input changed; original intermediate trajectory unavailable',
            first_zero_stage='integer rasterization' if r['input_identity'] else None,
            cause='natural same-bit cell plus rounding' if r['input_identity'] and r['input_success'] else 'failed write, rounding plus feedback selection' if r['input_identity'] else 'nonzero raster edit')
        if rgb_equal:assert all(row[k] for k in ['input_png_equal','condition_tensor_equal','prompt_equal','seed_equal','initial_noise_equal','checkpoint_equal'])
        jobs.append(row)
    summary=dict(marked_pngs=len(jobs),identical_rgb=sum(r['rgb_bytes_equal'] for r in jobs),identical_input_states=sum(r['input_identity'] for r in states),
        zero_states_clean_already_intended=sum(r['input_identity'] and r['clean_already_decodes_intended'] for r in states),
        zero_states_wrong_bit=sum(r['input_identity'] and not r['clean_already_decodes_intended'] for r in states),
        nonzero_inputs_but_identical_rgb=sum(not r['input_png_equal'] and r['rgb_bytes_equal'] for r in jobs),
        input_successes=sum(r['input_success'] for r in states),
        failed_despite_correct_bit_trial=[r['key']+f'_b{r["bit"]}' for r in states if not r['input_success'] and r['old_trial_has_correct_bit']],
        scope='Old fresh set is now diagnostic, not independent validation. Original artifacts read-only.')
    dump(OUT/'trace_states24.json',states);dump(OUT/'trace_all48.json',jobs);dump(OUT/'trace_identical26.json',[r for r in jobs if r['rgb_bytes_equal']]);dump(OUT/'trace_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    for r in states:
        if not r['input_success']:print(r['key'],r['bit'],'editable',r['clean_observed_pixels_editable_fraction'],'phi',r['clean_observed_phi_information_editable_fraction'],'obs shared',r['old_input_observation_shared_fraction'],flush=True)

if __name__=='__main__':trace()
