"""Cal-first input repair and read-only diagnosis of the old fresh contents."""
from pathlib import Path
import argparse,json,hashlib,time,shutil
import numpy as np
from PIL import Image
from trace_qim_input_signal import ROOT,OLD,OUT,read,dump,digest,arr
from curve_qim_closed_loop_v2 import CONFIG,bank,select,measure,RasterWarp
from curve_qim_stage3 import warp_quartic,decode,phase
from canny_curve_stage2c_qim import THRESHOLDS

def setup():
    OUT.mkdir(exist_ok=True)
    path=OUT/'repair_protocol.json'
    protected={str(ROOT/n):digest(ROOT/n) for n in ['curve_qim_closed_loop_v2.py','repair_qim_input_closure.py','trace_qim_input_signal.py']}
    if path.exists():
        assert read(path)['protected_sha256']==protected
        return
    dump(path,dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),config=CONFIG,protected_sha256=protected,
        old_manifest_sha256=digest(OLD/'final_artifacts_sha256.json'),input_receiver_choices=['OracleMask old estimator','Auto frozen 2c estimator, also used after diffusion'],
        policies=dict(parity_only='dense feedback, parity correctness first; zero edits allowed as diagnostic',nearest_target='same originally nearest lattice target, <=.25Delta error',
                      active_same_parity='same parity lattice; choose nearest reachable guarded codeword; non-nearest deviation explicit'),
        primary='Auto, delta1, active_same_parity. Same binary rasterizer, mask, QIM basis, dither and existing 2c receiver remain unchanged.',
        scope='6 old Cal contents first, then all12 old fresh contents diagnostic only. No new independent claim. No output RGB used to tune input search. All failures retained.',
        thresholds='Engineering guards chosen before Cal run, not statistically calibrated transport guarantee. Coefficient shift .5px is an input diagnostic floor, not proven sufficient for diffusion.',
        change='Nearest-reachable parity codeword is an explicit constrained QIM variant; it is not claimed to preserve unconstrained nearest-target rule. Input failure returns no hint rather than silently sending clean.'))

def cases(split):return read(OLD/('qim_cal_selection.json' if split=='Cal' else 'fresh/selection.json'))

def run(split):
    setup();folder=OUT/split;folder.mkdir(exist_ok=True);gate=read(OLD/'receiver_freeze.json')['selected']['gate'];allresults=[]
    for c in cases(split):
        dest=folder/f'{c["key"]}.json'
        if dest.exists():allresults+=read(dest)['results'];continue
        start=time.monotonic();edge=arr(c['clean_hint']);mask=arr(c['mask'])>0
        rows,rasters=bank(edge,c['samples'],mask,THRESHOLDS,gate)
        warp=RasterWarp(edge,c['samples'],mask)
        # Bit-exact equivalence, including zero dose and negative/positive moves.
        for alpha in [0.,-.77,.81,2.65]:assert np.array_equal(warp.render(alpha)[0],warp_quartic(edge,c['samples'],mask,alpha))
        clean={receiver:measure(edge,c['samples'],receiver,THRESHOLDS,gate) for receiver in ['OracleMask','Auto']};results=[]
        for receiver in ['OracleMask','Auto']:
            for delta in [1.,1.5,2.]:
                for bit in [0,1]:
                    for policy in ['parity_only','nearest_target','active_same_parity']:
                        if not clean[receiver]['accepted']:
                            r=dict(input_success=False,write_success=False,reason='clean_not_measurable',receiver=receiver,delta=delta,bit=bit,policy=policy)
                        else:r=select(rows,rasters,clean[receiver]['c_hat'],c['key'],delta,bit,receiver,policy)
                        r.update(key=c['key'],content_id=c['content_id'],stratum=c['stratum'],split=split)
                        if r['input_success'] and policy=='active_same_parity':
                            path=folder/f'{c["key"]}_{receiver}_d{delta}_b{bit}.png';image=rasters[r['raster_hash']]['image'];Image.fromarray(image).save(path)
                            # Verify saved artifact, not an analytical target, with no message supplied to measurement.
                            actual=arr(path);v=measure(actual,c['samples'],receiver,THRESHOLDS,gate);assert v['accepted']
                            assert decode(v['c_hat'],delta,phase(c['key'],delta))['bit']==bit
                            assert np.array_equal(actual[~mask],edge[~mask]) and np.any(actual!=edge)
                            r.update(hint=str(path),hint_sha256=digest(path),saved_round_trip_verified=True,outside_mask_unchanged=True)
                        results.append(r)
        slim={h:{name:{k:v for k,v in r.items() if k in ['accepted','c_hat','offset','branch','residual_rms','r_phi','coefficient_se','coverage','span']} for name,r in state['observations'].items()} for h,state in rasters.items()}
        dump(dest,dict(key=c['key'],clean=clean,rows=rows,unique_rasters=slim,results=results,elapsed_seconds=time.monotonic()-start))
        allresults+=results
        print(split,c['key'],'unique',len(rasters),'primary',[(r['bit'],r['input_success'],r.get('amplitude'),r.get('codeword_changed')) for r in results if r['receiver']=='Auto' and r['delta']==1 and r['policy']=='active_same_parity'],flush=True)
    summaries={}
    for receiver in ['OracleMask','Auto']:
        for delta in [1.,1.5,2.]:
            for policy in ['parity_only','nearest_target','active_same_parity']:
                rr=[r for r in allresults if r['receiver']==receiver and r['delta']==delta and r['policy']==policy];good=[r for r in rr if r['input_success']]
                summaries[f'{receiver}_d{delta}_{policy}']=dict(attempts=len(rr),successes=len(good),active_writes=sum(r['write_success'] for r in rr),
                    zero_edits=sum(r.get('changed_pixels')==0 for r in good),nonnearest_targets=sum(r.get('codeword_changed',False) for r in good),
                    median_absolute_amplitude=float(np.median([abs(r['amplitude']) for r in good])) if good else None,
                    median_absolute_coefficient_change=float(np.median([abs(r['coefficient_change']) for r in good])) if good else None,
                    failures=[f'{r["key"]}_b{r["bit"]}' for r in rr if not r['input_success']])
    dump(folder/'results.json',allresults);dump(folder/'summary.json',summaries);print('DONE',split,json.dumps(summaries['Auto_d1.0_active_same_parity']),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['Cal','Diagnostic']);args=a.parse_args();run(args.action)
