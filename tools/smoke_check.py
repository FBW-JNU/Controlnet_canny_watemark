"""Integration replay against real frozen files, not a synthetic performance test."""
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from canny_geometry.api import decode,encode,summarize_records

def read(p):return json.loads((ROOT/p).read_text())
def load(p,rgb=False):return np.array(Image.open(ROOT/p).convert('RGB' if rgb else 'L'))

ref=read('examples/reference_1bit.json')
images,mask=encode(load('examples/conditions/clean.png'),ref['samples'],'s1')
for word,arm in [('clean','clean'),('0','bit0'),('1','bit1')]:
    assert np.array_equal(images[word],load(f'examples/conditions/{arm}.png')),arm
    assert np.array_equal(images[word][~mask],images['clean'][~mask]),arm
for expected in read('examples/expected_1bit.json'):
    actual=decode(ref,load(f'examples/rgb/{expected["arm"]}.png',True))
    assert actual['accepted']==expected['accepted']
    assert actual['bits']==str(expected['bit'])
    assert np.allclose(actual['coefficient'],expected['coefficient'],atol=1e-8)
    assert abs(actual['score']-expected['score'])<1e-8
assert not decode(ref,np.zeros((512,512,3),np.uint8))['accepted']

metrics=summarize_records(read('evidence/stable_carrier_decisions.json'))
summary=read('evidence/canny_st_stable_carrier_compare_20261111/summary.json')
for policy in ['source_only','clean_stable']:
    for method in ['SourceRef','R2_tube','R2_bbox']:
        m=metrics[f'{policy}/{method}'];old=summary[policy][method]
        for key in ['attempts','correct','wrong','erasures']:assert m[key]==old[key],(policy,method,key)
assert metrics['source_only/SourceRef']['bit_correct']==57
assert metrics['clean_stable/SourceRef']['bit_correct']==78
print(json.dumps(dict(status='passed',checks=[
    'Three real condition PNGs reproduced byte-for-byte as image arrays',
    'Encoder mask exterior unchanged',
    'Three real RGB observations, messages and frozen affine scores reproduced',
    'Empty RGB returns an erasure',
    'All six stable-comparison method/policy aggregates equal archived summary',
    'Message bit counts 57/96 and 78/96 reproduced'],
    scope='CPU replay and packaging integration only; no new GPU generation or model training'),indent=2))
