"""Rebuild the independent region-first pilot, with Cal before Eval freezing."""
from pathlib import Path
import argparse
import copy
import json
import os
import shutil
import time
import numpy as np
from PIL import Image
from .api import choose_stable_carrier
from .benchmark import read,dump,sha,image,location,verify
from .stability import probe


def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def manifest_base(protocol,assets):
    return dict(schema_version=1,channel=protocol['channel'],assets=assets,generation_jobs=[],evaluation_rows=[],references={},models={})
def asset_map(root):
    return {str(p.relative_to(root)):sha(p) for base in ['canny','conditions','masks'] for p in sorted((root/base).rglob('*.png'))}


def prepare(args):
    source=Path(args.protocol).resolve();protocol=read(source);out=Path(args.out)
    if (out/'protocol.json').exists():
        if sha(out/'protocol.json')!=sha(source):raise ValueError('Output protocol differs.')
        verify(out/'probe_manifest.json');print('Existing frozen probe plan verified.');return
    out.mkdir(parents=True,exist_ok=True);shutil.copy2(source,out/'protocol.json')
    for rel,digest in protocol['assets'].items():
        src=location(source,rel)
        if sha(src)!=digest:raise ValueError('Raw source Canny hash differs.')
        dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    m=manifest_base(protocol,asset_map(out));m['assets']['protocol.json']=sha(out/'protocol.json');m['name']='New region-first clean probes';m['protocol_sha256']=sha(out/'protocol.json')
    for c in protocol['contents']:
        for pi,prefix in enumerate(protocol['probe']['prefixes']):
            for si,seed in enumerate(protocol['probe']['seeds']):
                m['generation_jobs'].append(dict(id=f'probe/{c["content_id"]:012d}_p{pi}_s{si}',hint=c['canny'],prompt=prefix+c['caption'],seed=seed,prompt_group=pi,phase='probe',content_id=c['content_id']))
    dump(out/'probe_manifest.json',m);print(f'Frozen {len(m["generation_jobs"])} clean probe jobs.')


def phase_manifest(out,p,selection,split,calibration=None,calibrated=False):
    m=manifest_base(p,asset_map(out));m['assets']['protocol.json']=sha(out/'protocol.json');m['assets']['selection.json']=sha(out/'selection.json');m['assets']['selection_freeze.json']=sha(out/'selection_freeze.json');m['name']='New region-first '+split
    if calibrated:m['assets']['calibration.json']=sha(out/'calibration.json')
    for c in p['contents']:
        if c['split']!=split:continue
        chosen=selection[str(c['content_id'])];rid=str(c['content_id']);ref=copy.deepcopy(chosen['reference'])
        if ref is not None:
            if calibrated:ref['calibration']=calibration;ref['calibration_scope']='New Cal-only frozen channel' if calibration else 'Predeclared input-domain fallback'
            m['references'][rid]=dict(receiver='4A',reference=ref,input=chosen['input'])
        else:m['references'][rid]=None
        if split=='Eval':
            baseline=copy.deepcopy(m['references'][rid])
            if baseline is not None:baseline['reference']['calibration']=None
            m['references']['input/'+rid]=baseline
        seeds=p['evaluation'][split+'_seeds']
        for view,(prefix,seed) in enumerate(zip(p['evaluation']['prefixes'],seeds)):
            clean=f'{split}/{c["content_id"]:012d}_v{view}_clean'
            for arm in ['clean','00','01','10','11']:
                gid=f'{split}/{c["content_id"]:012d}_v{view}_{arm}'
                m['generation_jobs'].append(dict(id=gid,hint=chosen['states'][arm],prompt=prefix+c['caption'],seed=seed,phase=split,content_id=c['content_id']))
                row=dict(id=gid,generation_id=gid,clean_generation_id=clean,reference=rid,content_id=c['content_id'],group='region_first/'+split,truth=None if arm=='clean' else arm,arm=arm,view=view)
                m['evaluation_rows'].append(row)
                if split=='Eval':m['evaluation_rows'].append(dict(row,id='input/'+gid,reference='input/'+rid,group='region_first_input/Eval'))
    return m


def select(args):
    out=Path(args.out);p=read(out/'protocol.json');m=verify(out/'probe_manifest.json')
    if (out/'selection_freeze.json').exists():
        if sha(out/'selection.json')!=read(out/'selection_freeze.json')['selection_sha256']:raise ValueError('Frozen selection changed.')
        verify(out/'cal_manifest.json');print('Existing frozen selection verified.');return
    rgbroot=Path(args.probe_rgb_root).resolve();selection={};hashes={}
    for c in p['contents']:
        cid=c['content_id'];work=out/'screening'/str(cid);work.mkdir(parents=True,exist_ok=True);plan=[]
        for job in m['generation_jobs']:
            if job['content_id']!=cid:continue
            rgb=rgbroot/'rgb'/(job['id']+'.png');record=read(rgbroot/'records'/(job['id']+'.json'))
            if record['manifest_sha256']!=sha(out/'probe_manifest.json') or record['rgb_sha256']!=sha(rgb):raise ValueError('Probe does not belong to frozen clean plan.')
            hashes[job['id']]=sha(rgb);plan.append(dict(rgb=os.path.relpath(rgb,work.resolve()),prompt_group=job['prompt_group'],prompt=job['prompt'],seed=job['seed'],hint_sha256=c['canny_sha256']))
        dump(work/'plan.json',plan);report=probe(out/c['canny'],work/'plan.json',work/'candidates.json',c['roi'])
        edge=image(out/c['canny']);chosen=choose_stable_carrier(edge,report,'st2');states={};dest=out/'conditions'/str(cid);dest.mkdir(parents=True,exist_ok=True)
        for arm in ['clean','00','01','10','11']:
            path=dest/(arm+'.png');Image.fromarray(chosen['images'][arm] if chosen['images'] is not None else edge).save(path);states[arm]=str(path.relative_to(out))
        mask=chosen['mask'] if chosen['mask'] is not None else np.zeros_like(edge,bool);mp=out/'masks'/f'{cid}.png';mp.parent.mkdir(exist_ok=True);Image.fromarray(mask.astype('uint8')*255).save(mp)
        inp=dict(mode='st2',clean=states['clean'],mask=str(mp.relative_to(out)),states=states)
        selection[str(cid)]=dict(content_id=cid,split=c['split'],candidate=chosen['candidate'],reference=chosen['reference'],input=inp,states=states,rejected=chosen['rejected'],counts=dict(source_pixels=report['source_pixels'],stable_pixels=report['stable_pixels'],extracted=len(report['candidates']),verified=report['eligible']))
        print(f'SELECTED {cid} {c["split"]} carrier={chosen["candidate"] is not None} curves={report["eligible"]}',flush=True)
    dump(out/'selection.json',selection)
    dump(out/'selection_freeze.json',dict(created_utc=now(),protocol_sha256=sha(out/'protocol.json'),probe_manifest_sha256=sha(out/'probe_manifest.json'),probe_rgb_sha256=hashes,selection_sha256=sha(out/'selection.json'),condition_sha256=asset_map(out),no_eval_generated_yet=not (out/'eval_manifest.json').exists()))
    dump(out/'cal_manifest.json',phase_manifest(out,p,selection,'Cal'));print('Selection and Cal plan frozen before any message RGB.')


def covariance(values):
    v=np.asarray(values,float);cov=np.cov(v.T,bias=True) if len(v)>1 else np.eye(2);eig,vec=np.linalg.eigh(cov)
    return (vec*np.maximum(eig,.01))@vec.T


def calibrate(args):
    out=Path(args.out);p=read(out/'protocol.json');selection=read(out/'selection.json');m=verify(out/'cal_manifest.json')
    if (out/'calibration.json').exists():verify(out/'eval_manifest.json');print('Existing frozen calibration verified.');return
    decoded=read(Path(args.cal_decoded)/'decoded.json');byid={r['id']:r for r in decoded};received=[];rgbroot=Path(args.cal_rgb_root);hashes={}
    if read(Path(args.cal_decoded)/'summary.json')['manifest_sha256']!=sha(out/'cal_manifest.json'):raise ValueError('Cal decoding manifest differs.')
    for job in m['generation_jobs']:
        record=read(rgbroot/'records'/(job['id']+'.json'));path=rgbroot/'rgb'/(job['id']+'.png')
        if record['manifest_sha256']!=sha(out/'cal_manifest.json') or record['rgb_sha256']!=sha(path):raise ValueError('Invalid Cal generation record.')
        hashes[job['id']]=sha(path)
    for row in m['evaluation_rows']:
        d=byid[row['id']]['decoded'];ref=m['references'][row['reference']]
        if not d['accepted'] or ref is None:continue
        book=ref['reference']['codebooks']['Auto'];arm=row['arm'];received.append(dict(id=row['id'],clean=arm=='clean',x=(np.array(book[arm])-book['clean']).tolist(),y=(np.array(d['coefficient'])-book['clean']).tolist()))
    rules=p['calibration'];clean=sum(r['clean'] for r in received);marked=len(received)-clean;model=None
    enough=len(received)>=rules['minimum_received'] and clean>=rules['minimum_clean'] and marked>=rules['minimum_marked']
    if enough:
        x=np.array([np.r_[1,r['x']] for r in received]);y=np.array([r['y'] for r in received]);beta=np.linalg.lstsq(x,y,rcond=None)[0];res=y-x@beta;condition=float(np.linalg.cond(beta[1:].T))
        model=dict(A=beta[1:].T.tolist(),intercept=beta[0].tolist(),residual_covariance=covariance(res).tolist(),cal_received=len(received),cal_attempts=len(m['evaluation_rows']),condition=condition if np.isfinite(condition) else None)
    frozen=dict(created_utc=now(),selection_sha256=sha(out/'selection.json'),cal_manifest_sha256=sha(out/'cal_manifest.json'),cal_decoded_sha256=sha(Path(args.cal_decoded)/'decoded.json'),cal_rgb_sha256=hashes,received=len(received),clean_received=clean,marked_received=marked,status='calibrated' if enough else 'predeclared_input_domain_fallback',model=model,evaluation_fitting=False,no_eval_plan_before_calibration=not (out/'eval_manifest.json').exists())
    dump(out/'calibration.json',frozen);dump(out/'eval_manifest.json',phase_manifest(out,p,selection,'Eval',model,True))
    print(json.dumps(dict(status=frozen['status'],received=len(received),model=model),indent=2))


def report(args):
    out=Path(args.out);p=read(out/'protocol.json');selection=read(out/'selection.json');decoded=read(Path(args.eval_decoded)/'decoded.json');summary=read(Path(args.eval_decoded)/'summary.json')
    if summary['manifest_sha256']!=sha(out/'eval_manifest.json'):raise ValueError('Eval decoding manifest differs.')
    result=dict(summary=summary['summary'],calibration=read(out/'calibration.json'),encoding={},bootstrap95={},clean={},per_bit={},confusion={},per_content={},input_geometry={})
    quality_path=out/'eval_quality/quality.json';quality=read(quality_path) if quality_path.exists() else None
    if quality:result['quality']=dict(groups=quality['groups'],limits=quality['limits'],joint_correct_and_quality={})
    for split in ['Cal','Eval']:
        ids=[str(c['content_id']) for c in p['contents'] if c['split']==split];result['encoding'][split]=dict(contents=len(ids),with_carrier=sum(selection[c]['reference'] is not None for c in ids))
        for cid in ids:
            chosen=selection[cid]
            if chosen['reference'] is None:result['input_geometry'][cid]=None;continue
            book=chosen['reference']['codebooks']['Auto'];values=np.array([book[w] for w in ['00','01','10','11']]);dist=np.linalg.norm(values[:,None]-values[None,:],axis=2)+np.eye(4)*1e6
            result['input_geometry'][cid]=dict(split=split,curve_length_px=chosen['candidate']['arc_length_px'],actual_four_word_minimum_coefficient_separation_px=float(dist.min()),source_pixels=chosen['counts']['source_pixels'],stable_pixels=chosen['counts']['stable_pixels'],verified_candidates=chosen['counts']['verified'])
    from .benchmark import summarize
    rng=np.random.default_rng(2026111273)
    for group in ['region_first/Eval','region_first_input/Eval']:
        data=[r for r in decoded if r['group']==group];ids=sorted({r['content_id'] for r in data});acc=[];bacc=[]
        for _ in range(2000):
            sampled=[r for i in rng.integers(0,len(ids),len(ids)) for r in data if r['content_id']==ids[i]];metrics=summarize(sampled)[group];acc.append(metrics['full_message_accuracy']);bacc.append(metrics['all_attempt_bit_accuracy'])
        result['bootstrap95'][group]=dict(full_message=np.quantile(acc,[.025,.975]).tolist(),bit=np.quantile(bacc,[.025,.975]).tolist())
        marked=[r for r in data if r['truth'] is not None]
        result['per_bit'][group]=[dict(bit_index=b,attempts=len(marked),correct=sum(r['decoded']['accepted'] and r['decoded']['bits'][b]==r['truth'][b] for r in marked),wrong=sum(r['decoded']['accepted'] and r['decoded']['bits'][b]!=r['truth'][b] for r in marked),erased=sum(not r['decoded']['accepted'] for r in marked)) for b in [0,1]]
        words=['00','01','10','11'];matrix=np.zeros((4,5),int)
        for r in marked:matrix[words.index(r['truth']),words.index(r['decoded']['bits']) if r['decoded']['accepted'] else 4]+=1
        result['confusion'][group]=dict(rows=words,columns=words+['erasure'],counts=matrix.tolist())
        result['per_content'][group]={str(cid):summarize([r for r in data if r['content_id']==cid])[group] for cid in ids}
        if quality:
            byid={r['id']:r for r in quality['records']}
            joint=sum(r['decoded']['accepted'] and r['decoded']['bits']==r['truth'] and byid[r['id'].removeprefix('input/')]['pass_all'] for r in marked)
            result['quality']['joint_correct_and_quality'][group]=dict(attempts=len(marked),passed=joint)
        clean=[r for r in data if r['truth'] is None];counts={w:sum(r['decoded'].get('bits')==w for r in clean if r['decoded']['accepted']) for w in ['00','01','10','11']}
        result['clean'][group]=dict(attempts=len(clean),received=sum(r['decoded']['accepted'] for r in clean),predicted_words=counts,balanced_independent_target_bit_accuracy_on_received=.5,balanced_independent_target_word_accuracy_on_received=.25,interpretation='Expected baseline against all four independent balanced target words; constant predictions also reach this baseline.')
    result['scope']='Independent new-content pilot; source-reference-assisted; no multi-carrier ECC or mask-only claim.';dump(out/'result.json',result);print(json.dumps(result['summary'],indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--protocol',required=True);a.add_argument('--out',required=True)
    a=sub.add_parser('select');a.add_argument('--out',required=True);a.add_argument('--probe-rgb-root',required=True)
    a=sub.add_parser('calibrate');a.add_argument('--out',required=True);a.add_argument('--cal-decoded',required=True);a.add_argument('--cal-rgb-root',required=True)
    a=sub.add_parser('report');a.add_argument('--out',required=True);a.add_argument('--eval-decoded',required=True)
    a=p.parse_args();globals()[a.command](a)
if __name__=='__main__':main()
