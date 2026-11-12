"""Real historical 4A/4B receiver regression plus portable asset/tamper checks."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from canny_geometry.benchmark import verify,read,image,decode,sha
from canny_geometry.historical_receiver import receive_4b
manifest=verify(ROOT/'benchmarks/history/manifest.json');byid={r['id']:r for r in manifest['evaluation_rows']}
for case in read(ROOT/'examples/history_receiver_checks/checks.json'):
    row=byid[case['row_id']];ref=manifest['references'][row['reference']];rgb=image(ROOT/case['rgb'],True)
    actual=receive_4b(ref['carrier'],rgb,manifest['models'][ref['model']]) if ref['receiver']=='4B' else decode(ref['reference'],rgb)
    assert actual['accepted']==case['expected']['accepted'] and actual['bits']==case['expected']['bits']
    if ref['receiver']=='4B':assert not receive_4b(ref['carrier'],np.zeros_like(rgb),manifest['models'][ref['model']])['accepted']
with tempfile.TemporaryDirectory(prefix='canny-manifest-check-') as tmp:
    d=Path(tmp);(d/'condition.png').write_bytes((ROOT/'examples/conditions/clean.png').read_bytes())
    small=copy.deepcopy(manifest);small['assets']={'condition.png':sha(d/'condition.png')};small['generation_jobs']=[];small['evaluation_rows']=[];small['references']={}
    (d/'manifest.json').write_text(json.dumps(small));verify(d/'manifest.json')
    original=copy.deepcopy(small)
    for changed in [dict(steps=21),dict(precision='float16')]:
        bad=copy.deepcopy(original);bad['channel'].update(changed);(d/'manifest.json').write_text(json.dumps(bad))
        try:verify(d/'manifest.json')
        except ValueError:pass
        else:raise AssertionError('Unsupported channel settings must not be silently ignored.')
    bad=copy.deepcopy(original);bad['generation_jobs']=[dict(id='unknown',hint='unfrozen.png')];(d/'manifest.json').write_text(json.dumps(bad))
    try:verify(d/'manifest.json')
    except ValueError:pass
    else:raise AssertionError('Unlisted generation inputs must not be accepted.')
    (d/'manifest.json').write_text(json.dumps(original))
    (d/'condition.png').write_bytes(b'corrupted')
    try:verify(d/'manifest.json')
    except ValueError:pass
    else:raise AssertionError('Changed input asset must not be accepted.')
print(json.dumps(dict(status='passed',checks=['Real 4A RGB message equals historical decision','Real 4B RGB message equals historical decision','4B empty output is an erasure','Portable manifest verifies outside research tree','Tampered binary input rejected','Unsupported channel settings rejected','Unfrozen generation inputs rejected']),indent=2))
