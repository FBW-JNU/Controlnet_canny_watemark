"""Portable frozen-batch generation and independent RGB decoding.

Generation consumes only condition/prompt/seed; message labels and historical
expected decisions are used exclusively in evaluation. All paths in manifests
are relative to the exported repository or the manifest directory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import numpy as np
from PIL import Image
from .api import ROOT, decode, encode


def read(path):return json.loads(Path(path).read_text())
def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def location(manifest_path,relative):
    path=Path(relative)
    if path.is_absolute():raise ValueError('Manifest assets must use portable relative paths.')
    return (Path(manifest_path).resolve().parent/path).resolve()
def image(path,rgb=False):return np.asarray(Image.open(path).convert('RGB' if rgb else 'L'))


def verify(manifest_path):
    manifest=read(manifest_path)
    if manifest.get('schema_version')!=1:raise ValueError('Unsupported manifest schema.')
    fixed=dict(steps=20,CFG=5,eta=0,control=1,precision='float32')
    if any(manifest['channel'].get(k)!=v for k,v in fixed.items()):
        raise ValueError('Channel settings differ from this frozen generator.')
    if manifest['channel']['config_sha256']!=sha(ROOT/'configs/cldm_v15.yaml'):
        raise ValueError('Model configuration differs from frozen channel.')
    for path,digest in manifest['assets'].items():
        if sha(location(manifest_path,path))!=digest:raise ValueError('Asset hash mismatch: '+path)
    ids=[j['id'] for j in manifest['generation_jobs']]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate generation job identifiers.')
    for job in manifest['generation_jobs']:
        if Path(job['id']).is_absolute() or '..' in Path(job['id']).parts:
            raise ValueError('Job id must remain inside the output directory.')
        if job['hint'] not in manifest['assets']:raise ValueError('Unfrozen condition image.')
    rowids=[r['id'] for r in manifest['evaluation_rows']]
    if len(rowids)!=len(set(rowids)):raise ValueError('Duplicate evaluation row identifiers.')
    for row in manifest['evaluation_rows']:
        if row['generation_id'] not in ids:raise ValueError('Unknown generation job.')
        if row['reference'] not in manifest['references']:raise ValueError('Unknown reference.')
        if row.get('clean_generation_id') not in ids:raise ValueError('Unknown paired clean job.')
        if row['truth'] is not None and (not row['truth'] or set(row['truth'])-{'0','1'}):
            raise ValueError('Message labels must be binary strings.')
    return manifest


def generate(args):
    manifest=verify(args.manifest);frozen_sha=sha(args.manifest)
    if not 0<=args.shard_index<args.shards:raise ValueError('Invalid shard index.')
    out=Path(args.out);(out/'rgb').mkdir(parents=True,exist_ok=True)
    all_jobs=manifest['generation_jobs']
    if args.phase:all_jobs=[j for j in all_jobs if j.get('phase')==args.phase]
    jobs=[j for i,j in enumerate(all_jobs) if i%args.shards==args.shard_index]
    from .channel import ControlNetChannel
    channel=None;complete=[]
    for i,job in enumerate(jobs):
        target=out/'rgb'/(job['id']+'.png');record=out/'records'/(job['id']+'.json')
        if record.exists():
            old=read(record)
            if old['manifest_sha256']!=frozen_sha or sha(target)!=old['rgb_sha256']:
                raise ValueError('Resume record does not match frozen manifest/output.')
            complete.append(old);continue
        if target.exists():raise FileExistsError('PNG has no valid record: '+str(target))
        if channel is None:
            if sha(args.checkpoint)!=manifest['channel']['checkpoint_sha256']:
                raise ValueError('Checkpoint hash differs from frozen channel.')
            channel=ControlNetChannel(args.checkpoint,args.clip,args.device)
        hint=location(args.manifest,job['hint'])
        start=time.monotonic();rgb=channel.render(hint,job['prompt'],job['seed'])
        target.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(rgb).save(target)
        digest=sha(target);expected=job.get('expected_rgb_sha256')
        row=dict(id=job['id'],manifest_sha256=frozen_sha,hint_sha256=sha(hint),
                 rgb_sha256=digest,expected_png_identical=digest==expected if expected else None,
                 elapsed_seconds=time.monotonic()-start)
        dump(record,row);complete.append(row)
        print(f'GENERATED {i+1}/{len(jobs)} {job["id"]} {row["elapsed_seconds"]:.2f}s expected={row["expected_png_identical"]}',flush=True)
    dump(out/f'generation_shard{args.shard_index}_{args.phase or "all"}.json',dict(
        manifest_sha256=frozen_sha,shard=args.shard_index,shards=args.shards,jobs=len(complete),
        expected_matches=sum(r['expected_png_identical'] is True for r in complete),
        expected_mismatches=sum(r['expected_png_identical'] is False for r in complete)))


def summarize(rows):
    result={}
    for group in sorted({r['group'] for r in rows}):
        data=[r for r in rows if r['group']==group and r['truth'] is not None]
        correct=wrong=erased=bit_correct=bit_wrong=bit_erased=0
        for r in data:
            d=r['decoded'];truth=r['truth']
            if not d['accepted'] or d.get('bits') is None:erased+=1;bit_erased+=len(truth);continue
            correct+=d['bits']==truth;wrong+=d['bits']!=truth
            bit_correct+=sum(a==b for a,b in zip(d['bits'],truth));bit_wrong+=sum(a!=b for a,b in zip(d['bits'],truth))
        total_bits=bit_correct+bit_wrong+bit_erased
        result[group]=dict(attempts=len(data),correct=correct,wrong=wrong,erasures=erased,
            full_message_accuracy=correct/len(data) if data else None,
            bit_correct=bit_correct,bit_wrong=bit_wrong,bit_erased=bit_erased,
            all_attempt_bit_accuracy=bit_correct/total_bits if total_bits else None,
            received_bit_error_rate=bit_wrong/(bit_wrong+bit_correct) if bit_wrong+bit_correct else None)
    return result


def decode_batch(args):
    manifest=verify(args.manifest);root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    rgb_map=read(args.rgb_map) if args.rgb_map else None
    cache={};rows=[];expected_mismatches=[];png_mismatches=[]
    jobs={j['id']:j for j in manifest['generation_jobs']}
    for i,row in enumerate(manifest['evaluation_rows']):
        gid=row['generation_id'];path=Path(rgb_map[gid]) if rgb_map else Path(args.rgb_root)/'rgb'/(gid+'.png')
        if not path.is_file():raise FileNotFoundError(path)
        ref=manifest['references'].get(row['reference']);cache_key=(gid,row['reference'])
        if cache_key not in cache:
            if ref is None:d=dict(accepted=False,bits=None,reason='no_writable_carrier')
            elif ref['receiver']=='4B':
                from .historical_receiver import receive_4b
                d=receive_4b(ref['carrier'],image(path,True),manifest['models'][ref['model']])
            else:
                d=decode(ref['reference'],image(path,True))
            cache[cache_key]=d
        d=cache[cache_key]
        expected=row.get('expected_decoded')
        if expected is not None and (d['accepted']!=expected['accepted'] or d.get('bits')!=expected.get('bits')):
            expected_mismatches.append(dict(row_id=row['id'],expected=expected,actual=d))
        expected_png=jobs[gid].get('expected_rgb_sha256')
        if expected_png and sha(path)!=expected_png:png_mismatches.append(gid)
        rows.append(dict(id=row['id'],content_id=row['content_id'],group=row['group'],truth=row['truth'],
                         generation_id=gid,reference=row['reference'],decoded=d))
        if (i+1)%25==0:print(f'DECODED {i+1}/{len(manifest["evaluation_rows"])}',flush=True)
    summary=summarize(rows)
    expected_summary=manifest.get('expected_summary',{});metric_mismatches=[]
    for group,metrics in expected_summary.items():
        for key,value in metrics.items():
            if summary[group].get(key)!=value:metric_mismatches.append(dict(group=group,key=key,expected=value,actual=summary[group].get(key)))
    report=dict(manifest_sha256=sha(args.manifest),rows=len(rows),summary=summary,
        expected_decision_mismatches=expected_mismatches,expected_png_mismatches=sorted(set(png_mismatches)),
        expected_metric_mismatches=metric_mismatches,
        status='passed' if not expected_mismatches and not png_mismatches and not metric_mismatches else 'failed',
        scope='Fresh RGB observation/decoding; golden expected decisions used only for comparison')
    dump(root/'decoded.json',rows);dump(root/'summary.json',report)
    print(json.dumps(report,indent=2))
    if report['status']!='passed':raise RuntimeError('Historical reproduction differs; see summary.json.')


def input_check(args):
    manifest=verify(args.manifest);count=0
    for name,ref in manifest['references'].items():
        if not ref or 'input' not in ref:continue
        c=ref['input'];edge=image(location(args.manifest,c['clean']));mask=image(location(args.manifest,c['mask']))>0
        samples=ref['carrier']['samples'] if ref['receiver']=='4B' else ref['reference']['samples']
        generated,_=encode(edge,samples,c['mode'],mask)
        for word,path in c['states'].items():
            actual=image(location(args.manifest,path))
            if word in generated:
                if not np.array_equal(actual,generated[word]):raise AssertionError('Input raster differs: '+name+'/'+word)
                if not np.array_equal(actual[~mask],edge[~mask]):raise AssertionError('Mask exterior changed.')
                count+=1
    print(json.dumps(dict(status='passed',actual_binary_states_reproduced=count)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    v=sub.add_parser('verify');v.add_argument('--manifest',required=True)
    g=sub.add_parser('generate');g.add_argument('--manifest',required=True)
    g.add_argument('--checkpoint',required=True);g.add_argument('--clip',required=True);g.add_argument('--device',default='cuda')
    g.add_argument('--out',required=True);g.add_argument('--shards',type=int,default=1);g.add_argument('--shard-index',type=int,default=0)
    g.add_argument('--phase',choices=['probe','Cal','Eval'])
    d=sub.add_parser('decode');d.add_argument('--manifest',required=True);d.add_argument('--rgb-root')
    d.add_argument('--rgb-map',help='Optional audit-only map of ids to existing RGB; not required for fresh reproduction')
    d.add_argument('--out',required=True)
    q=sub.add_parser('quality');q.add_argument('--manifest',required=True);q.add_argument('--rgb-root');q.add_argument('--rgb-map');q.add_argument('--out',required=True)
    i=sub.add_parser('input-check');i.add_argument('--manifest',required=True)
    c=sub.add_parser('calibration-check');c.add_argument('--manifest',required=True);c.add_argument('--decoded',required=True);c.add_argument('--out',required=True)
    a=parser.parse_args()
    if a.command=='verify':
        m=verify(a.manifest);print(json.dumps(dict(status='passed',assets=len(m['assets']),generation_jobs=len(m['generation_jobs']),evaluation_rows=len(m['evaluation_rows']))))
    elif a.command=='generate':generate(a)
    elif a.command=='quality':
        if not a.rgb_root and not a.rgb_map:parser.error('quality requires --rgb-root or --rgb-map')
        from .quality import evaluate
        evaluate(a)
    elif a.command=='calibration-check':
        from .historical_calibration import check
        print(json.dumps(check(a.manifest,a.decoded,a.out),indent=2))
    elif a.command=='decode':
        if not a.rgb_root and not a.rgb_map:parser.error('decode requires --rgb-root or --rgb-map')
        decode_batch(a)
    else:input_check(a)
if __name__=='__main__':main()
