"""Run portable historical batches or the complete region-first pilot.

Example (two GPUs): python tools/run_reproduction.py pilot --gpus 0,1 \
  --checkpoint models/control_sd15_canny.pth --clip models/clip-vit-large-patch14 \
  --out runs/pilot --replay
Logs and raw RGB stay under the requested run directory; no weights are fetched.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from canny_geometry.benchmark import read,dump,sha


def run(args):
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    devices=args.gpus.split(',')
    def cpu(module,arguments,tag):
        logfile=out/'logs'/(tag+'.log');logfile.parent.mkdir(exist_ok=True)
        print('STEP',tag,flush=True)
        with logfile.open('w') as log:
            p=subprocess.run([sys.executable,'-m',module]+list(map(str,arguments)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError(f'{tag} failed; inspect {logfile}')
    def generate(manifest,dest,tag):
        logs=[];processes=[]
        try:
            for index,device in enumerate(devices):
                logpath=out/'logs'/f'{tag}_gpu{index}.log';logpath.parent.mkdir(exist_ok=True);log=logpath.open('w');logs.append(log)
                command=[sys.executable,'-m','canny_geometry.benchmark','generate','--manifest',str(manifest),'--checkpoint',str(Path(args.checkpoint).resolve()),'--clip',str(Path(args.clip).resolve()),'--out',str(dest),'--shards',str(len(devices)),'--shard-index',str(index)]
                processes.append(subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=device),stdout=log,stderr=subprocess.STDOUT))
            print('STEP',tag,flush=True)
            while any(p.poll() is None for p in processes):
                if any(p.poll() not in (None,0) for p in processes):raise RuntimeError(f'{tag} failed; inspect {out/"logs"}')
                time.sleep(2)
            if any(p.returncode for p in processes):raise RuntimeError(f'{tag} failed; inspect {out/"logs"}')
        finally:
            for p in processes:
                if p.poll() is None:p.terminate()
            for p in processes:p.wait()
            for log in logs:log.close()
    def decode(manifest,rgb,dest,tag):
        cpu('canny_geometry.benchmark',['decode','--manifest',manifest,'--rgb-root',rgb,'--out',dest],tag)
    def quality(manifest,rgb,dest,tag):
        cpu('canny_geometry.benchmark',['quality','--manifest',manifest,'--rgb-root',rgb,'--out',dest],tag)
    if args.mode=='history':
        manifest=Path(args.manifest).resolve();rgb=out/'generated'
        cpu('canny_geometry.benchmark',['input-check','--manifest',manifest],'input_check')
        generate(manifest,rgb,'historical_generation');decode(manifest,rgb,out/'decoded','historical_decode')
        cpu('canny_geometry.benchmark',['calibration-check','--manifest',manifest,'--decoded',out/'decoded','--out',out/'calibration_check.json'],'historical_calibration')
        if not args.skip_quality:quality(manifest,rgb,out/'quality','historical_quality')
    else:
        protocol=Path(args.protocol).resolve()
        def pilot(dest,label):
            cpu('canny_geometry.research',['prepare','--protocol',protocol,'--out',dest],label+'_prepare')
            generate(dest/'probe_manifest.json',dest/'probe_generated',label+'_probes')
            cpu('canny_geometry.research',['select','--out',dest,'--probe-rgb-root',dest/'probe_generated'],label+'_select')
            generate(dest/'cal_manifest.json',dest/'cal_generated',label+'_Cal')
            decode(dest/'cal_manifest.json',dest/'cal_generated',dest/'cal_decoded',label+'_Cal_decode')
            cpu('canny_geometry.research',['calibrate','--out',dest,'--cal-decoded',dest/'cal_decoded','--cal-rgb-root',dest/'cal_generated'],label+'_calibrate')
            generate(dest/'eval_manifest.json',dest/'eval_generated',label+'_Eval')
            decode(dest/'eval_manifest.json',dest/'eval_generated',dest/'eval_decoded',label+'_Eval_decode')
            if not args.skip_quality:quality(dest/'eval_manifest.json',dest/'eval_generated',dest/'eval_quality',label+'_quality')
            cpu('canny_geometry.research',['report','--out',dest,'--eval-decoded',dest/'eval_decoded'],label+'_report')
        primary=out/'primary';pilot(primary,'primary')
        if args.replay:
            repeat=out/'repeat';pilot(repeat,'repeat');differences=[];total=0
            for phase in ['probe','cal','eval']:
                records=list((primary/(phase+'_generated')/'records').rglob('*.json'))
                for record in records:
                    other=repeat/(phase+'_generated')/'records'/record.relative_to(primary/(phase+'_generated')/'records')
                    a,b=read(record),read(other);total+=1
                    if a['rgb_sha256']!=b['rgb_sha256']:differences.append(a['id'])
            a,b=read(primary/'selection.json'),read(repeat/'selection.json')
            for cid in a:
                for field in ['reference','states','input','counts','rejected']:
                    if a[cid][field]!=b[cid][field]:differences.append('selection/'+cid+'/'+field)
                ca,cb=a[cid]['candidate'],b[cid]['candidate']
                if (ca is None)!=(cb is None) or ca is not None and (ca['samples']!=cb['samples'] or ca['curve_index']!=cb['curve_index']):differences.append('candidate/'+cid)
            if read(primary/'calibration.json')['model']!=read(repeat/'calibration.json')['model']:differences.append('calibration/model')
            for phase in ['cal','eval']:
                if read(primary/(phase+'_decoded')/'decoded.json')!=read(repeat/(phase+'_decoded')/'decoded.json'):differences.append('decoded/'+phase)
            if not args.skip_quality:
                if read(primary/'eval_quality/quality.json')!=read(repeat/'eval_quality/quality.json'):differences.append('quality')
            report=dict(status='passed' if not differences else 'failed',RGB_compared=total,differences=differences,selection_samples_codebooks_conditions_calibration_decisions_compared=True,scope='Independent complete rerun from raw Canny, clean probes and fixed protocol; timestamp and path metadata excluded.')
            dump(out/'replay_comparison.json',report)
            if differences:raise RuntimeError('Independent replay differs; see replay_comparison.json')
    dump(out/'completed.json',dict(status='passed',mode=args.mode,replay=args.replay,quality=not args.skip_quality))
    print('COMPLETE',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['history','pilot'])
    p.add_argument('--checkpoint',required=True);p.add_argument('--clip',required=True);p.add_argument('--out',required=True)
    p.add_argument('--gpus',default='0',help='Visible device list, e.g. 0 or 0,1; one process per GPU')
    p.add_argument('--manifest',default=str(ROOT/'benchmarks/history/manifest.json'))
    p.add_argument('--protocol',default=str(ROOT/'benchmarks/region_first/protocol.json'))
    p.add_argument('--replay',action='store_true',help='Rebuild and compare complete pilot twice')
    p.add_argument('--skip-quality',action='store_true',help='Explicitly skip the optional LPIPS quality stage')
    run(p.parse_args())
if __name__=='__main__':main()
