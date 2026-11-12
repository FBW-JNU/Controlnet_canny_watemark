"""Post-hoc diagnostic of projection support; never changes scores or gates."""
from pathlib import Path
import numpy as np
import mat_st_message_soft_compare as experiment
from mat_st_message_experiment import read,dump,arr,observe,digest,WORDS


def audit():
    arms={'binary':experiment.BINARY,**{s:experiment.OUT/s for s in experiment.STRUCTURES}}
    result={}
    for name,arm in arms.items():
        records=[]
        for c in read(arm/'selection.json'):
            prefix=c['key']+'_L32';mask=arr(c['segments']['32']['mask'])>0
            for word in WORDS:
                f=arm/'mat_raw'/prefix/f'{word}.json'
                if not f.exists():continue
                detail=read(f);prob=np.load(arm/'mat_raw'/prefix/f'G3_probability_{word}.npy')
                threshold=read(arm/'protocol.json')['mat']['threshold']
                obs=detail['G3_preprojection_observations']['Auto']
                pool=[r for r in obs.get('decisions',[]) if r['coverage']>=.65 and r['span']>=.8 and r['residual_rms']<=1.5]
                inside=0;points=0
                if detail['G3']['success']:
                    assert pool
                    chosen=min(pool,key=lambda r:r['cost'])
                    xy=np.rint(np.array(chosen['observed_xy'])).astype(int).clip(0,511)
                    points=len(xy);inside=int(mask[xy[:,1],xy[:,0]].sum())
                foreground=int((prob[mask]>=threshold).sum())
                records.append(dict(key=c['key'],word=word,projection_success=detail['G3']['success'],
                    hard_foreground_inside_hole=foreground,support_points_total=points,support_points_inside_hole=inside,
                    zero_hole_foreground_but_projection_success=detail['G3']['success'] and foreground==0,
                    zero_inside_support_but_projection_success=detail['G3']['success'] and inside==0))
        result[name]=dict(attempts=len(records),projection_successes=sum(r['projection_success'] for r in records),
            zero_hole_foreground_but_projection_success=sum(r['zero_hole_foreground_but_projection_success'] for r in records),
            zero_inside_support_but_projection_success=sum(r['zero_inside_support_but_projection_success'] for r in records),records=records)
    dump(experiment.OUT/'projection_support_audit.json',dict(
        diagnostic_only=True,reason='Observed some refined raw holes were empty while original project_v2 succeeded.',
        interpretation='Original pool allows a nearby curve outside missing tube to support reconstruction. Same algorithm retained for comparison; projected success is not proof that MAT completed the target ridge.',
        changes_to_receiver_projection_or_gate=False,arms=result,source_sha256=digest(__file__)))
    for name,d in result.items():print(name,{k:v for k,v in d.items() if k!='records'})


if __name__=='__main__':audit()
