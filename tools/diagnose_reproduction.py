"""Observe failure cases and paired clean RGB without changing frozen selection.

Truth labels select cases for post-evaluation analysis only. They are never
passed to the geometric matcher or used to select an observation path.
"""
from pathlib import Path
import argparse
import collections
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from canny_geometry.api import observe
from canny_geometry.benchmark import read,dump,image,verify,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.run)
    manifest=verify(root/'eval_manifest.json');refs=manifest['references'];rows=read(root/'eval_decoded/decoded.json');plan={r['id']:r for r in manifest['evaluation_rows']};bad=[r for r in rows if r['group']=='region_first/Eval' and r['truth'] is not None and (not r['decoded']['accepted'] or r['decoded']['bits']!=r['truth'])]
    selected={r['id']:r for r in bad}
    for r in bad:
        clean=plan[r['id']]['clean_generation_id'];selected[clean]=next(v for v in rows if v['id']==clean)
    cases=[];counts=collections.Counter()
    for cid,row in selected.items():
        ref=refs[row['reference']];rgb=root/'eval_generated/rgb'/(row['generation_id']+'.png');record=read(root/'eval_generated/records'/(row['generation_id']+'.json'))
        if record['manifest_sha256']!=sha(root/'eval_manifest.json') or record['rgb_sha256']!=sha(rgb):raise ValueError('RGB record differs from frozen evaluation.')
        observed=observe(ref['reference']['samples'],image(rgb,True),True)['Auto'] if ref else dict(accepted=False,decisions=[],reason='no_writable_carrier')
        branches=[]
        for branch in observed.get('decisions',[]):
            failed=[k for k,v in branch['gate_checks'].items() if not v]
            branches.append(dict(branch=branch['branch'],accepted=branch['accepted'],failed_checks=failed,coefficient=branch['coefficient'],coverage=branch['coverage'],span=branch['span'],offset=branch['offset'],r_phi=branch['r_phi'],residual_rms=branch['residual_rms'],coefficient_se_max=branch['coefficient_se_max'],q_margin=branch['q_margin']))
            if row['truth'] is not None and not observed['accepted']:counts.update(failed)
        cases.append(dict(id=cid,content_id=row['content_id'],truth=row['truth'],decoded=row['decoded'],observed=branches,reason=observed.get('reason')))
    result=dict(scope='Post-evaluation failure diagnosis; branch-check counts are descriptive, not verified ground-truth correspondence causes. No parameter fitting or reselection.',wrong_or_erased_messages=len(bad),paired_clean_controls=len(selected)-len(bad),rejected_candidate_check_counts=dict(counts),cases=cases)
    dump(a.out,result);print(json.dumps(dict(wrong_or_erased_messages=len(bad),paired_clean_controls=len(selected)-len(bad),rejected_candidate_check_counts=dict(counts)),indent=2))
if __name__=='__main__':main()
