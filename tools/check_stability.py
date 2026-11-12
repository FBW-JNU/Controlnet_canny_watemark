"""Check missing observations, inconsistent drift and source extraction order."""
from pathlib import Path
import json
import sys
import tempfile
from unittest.mock import patch
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from canny_geometry import stability
plan=[dict(prompt_group=i//2,seed=i%2,prompt=f'prompt {i//2}',rgb=f'{i}.png') for i in range(6)]
edge=np.zeros((128,128),np.uint8);edge[40,20:100]=255;edge[90,20:100]=255
outputs=[]
for i in range(6):
    e=np.zeros_like(edge);e[40,20:100]=255
    if i<2:e[90,20:100]=255
    outputs.append(e)
stable,support,drift=stability.screen_pixels(edge,outputs,plan)
assert (stable[40,20:100]==255).all() and not stable[90].any()
assert (support[40,20:100]==6).all() and (support[90,20:100]==2).all()
five=[e.copy() for e in outputs];five[-1][:]=0
assert stability.screen_pixels(edge,five,plan)[0][40].any()
assert not stability.screen_pixels(edge,[np.zeros_like(edge)]*6,plan)[0].any()
shifts=[]
for offset in [-3,-3,-3,3,3,3]:
    e=np.zeros_like(edge);e[40+offset,20:100]=255;shifts.append(e)
assert not stability.screen_pixels(edge,shifts,plan)[0].any()
original=stability.matcher.source_curves;calls=[]
def checked_source(value,thresholds):
    assert np.array_equal(value,stable), 'Curves extracted before stability screening'
    calls.append(True);return original(value,thresholds)
with tempfile.TemporaryDirectory(prefix='canny-screen-test-') as tmp:
    d=Path(tmp);Image.fromarray(edge).save(d/'E.png');(d/'plan.json').write_text(json.dumps(plan))
    for i in range(6):Image.fromarray(np.zeros((128,128,3),np.uint8)).save(d/f'{i}.png')
    with patch.object(stability.cv2,'Canny',side_effect=outputs),patch.object(stability.matcher,'source_curves',side_effect=checked_source):
        result=stability.probe(d/'E.png',d/'plan.json',d/'out.json')
    assert len(calls)==1 and result['stable_pixels']==80
    assert np.array_equal(np.asarray(Image.open(d/result['stable_edges'])),stable)
    with np.load(d/result['maps']) as maps:assert np.array_equal(maps['support_count'],support)
print(json.dumps(dict(status='passed',checks=['Screen pixels before first source curve extraction',
 'Reject low survival and inconsistent drift','Keep five-of-six support','Handle empty output edges',
 'Persist screened edges and support maps']),indent=2))
