"""Compare a complete region-first pilot with the published frozen result."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from canny_geometry.benchmark import read,dump,sha


def signature(root):
    root=Path(root);selection=read(root/'selection.json');result={}
    result['protocol_sha256']=sha(root/'protocol.json')
    result['inputs']={str(p.relative_to(root)):sha(p) for base in ['canny','conditions','masks'] for p in sorted((root/base).rglob('*.png'))}
    result['selections']={cid:dict(curve_index=c['candidate']['curve_index'] if c['candidate'] else None,samples=c['reference']['samples'] if c['reference'] else None,reference=c['reference']) for cid,c in selection.items()}
    result['calibration_model']=read(root/'calibration.json')['model']
    result['rgb']={}
    for phase in ['probe','cal','eval']:
        base=root/(phase+'_generated')/'rgb'
        result['rgb'][phase]={str(p.relative_to(base).with_suffix('')):sha(p) for p in sorted(base.rglob('*.png'))}
    result['evaluation_summary']=read(root/'eval_decoded/summary.json')['summary']
    result['decisions']=[dict(id=r['id'],accepted=r['decoded']['accepted'],bits=r['decoded'].get('bits')) for r in read(root/'eval_decoded/decoded.json')]
    if (root/'eval_quality/quality.json').exists():
        q=read(root/'eval_quality/quality.json');result['quality_groups']=q['groups'];result['quality_pass']={r['id']:r['pass_all'] for r in q['records']}
    return result


def compare(actual,expected,path=''):
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):return [path+' keys']
        return [p for k in expected for p in compare(actual[k],expected[k],path+'/'+k)]
    if isinstance(expected,list):
        if not isinstance(actual,list) or len(actual)!=len(expected):return [path+' length']
        return [p for i,(a,b) in enumerate(zip(actual,expected)) for p in compare(a,b,path+'/'+str(i))]
    if isinstance(expected,float):
        return [] if actual is not None and np.isclose(actual,expected,rtol=0,atol=1e-10) else [path]
    return [] if actual==expected else [path]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--expected',default=str(ROOT/'benchmarks/region_first/expected.json'));p.add_argument('--out',required=True)
    a=p.parse_args();expected=read(a.expected);actual=signature(a.run);differences=compare(actual,expected)
    result=dict(status='passed' if not differences else 'failed',RGB_compared=sum(len(g) for g in expected['rgb'].values()),differences=differences,expected_sha256=sha(a.expected),scope='Compare raw input hashes, selected reference, Cal-only model, every generated RGB, Eval decisions, metrics and quality gates with published result.')
    dump(a.out,result);print(json.dumps(result,indent=2))
    if differences:raise SystemExit(1)
if __name__=='__main__':main()
