"""Run `python -m canny_geometry --help` from the exported repository."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import matcher
from .api import ROOT,THRESHOLDS,make_reference,decode,summarize_records,validate_edge


def read(path):
    return json.loads(Path(path).read_text())


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def load(path,rgb=False):
    return np.array(Image.open(path).convert('RGB' if rgb else 'L'))


def main():
    p=argparse.ArgumentParser(description='Canny geometry message tools; reference-assisted, no presence threshold')
    sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('demo',help='Replay the included real RGB example on CPU')
    d.add_argument('--out',default='runs/demo')
    e=sub.add_parser('embed',help='Write all 1-bit or 2-bit states on an explicit source curve')
    e.add_argument('--canny',required=True);e.add_argument('--curve',required=True)
    e.add_argument('--mask');e.add_argument('--mode',choices=['s1','st2'],default='s1')
    e.add_argument('--calibration',help='Frozen score JSON; models.Auto or the Auto model itself')
    e.add_argument('--out',required=True)
    d=sub.add_parser('decode',help='Read one RGB using an ordered source reference and codebook')
    d.add_argument('--reference',required=True);d.add_argument('--rgb',required=True)
    d.add_argument('--out')
    m=sub.add_parser('metrics',help='Aggregate existing message decisions, retaining erasures')
    m.add_argument('--records',default='evidence/stable_carrier_decisions.json');m.add_argument('--out')
    q=sub.add_parser('probe',help='Screen pixel stability first, then extract ordered source curves')
    q.add_argument('--canny',required=True);q.add_argument('--plan',required=True);q.add_argument('--out',required=True)
    pg=sub.add_parser('probe-generate',help='Generate six clean probes from the original Canny, loading the model once')
    pg.add_argument('--canny',required=True);pg.add_argument('--plan',required=True)
    pg.add_argument('--checkpoint',required=True);pg.add_argument('--clip',required=True)
    pg.add_argument('--config',default=str(ROOT/'configs/cldm_v15.yaml'))
    pg.add_argument('--device',default='cuda');pg.add_argument('--out',required=True)
    sel=sub.add_parser('select',help='Choose a stability-screened candidate passing actual binary input closure')
    sel.add_argument('--canny',required=True);sel.add_argument('--candidates',required=True)
    sel.add_argument('--mode',choices=['s1','st2'],default='s1');sel.add_argument('--out',required=True)
    g=sub.add_parser('generate',help='Frozen ControlNet generation with the bundled current backend')
    g.add_argument('--conditions',required=True,help='Directory of 512x512 condition PNGs')
    g.add_argument('--prompt',required=True);g.add_argument('--seed',type=int,default=2026114101)
    g.add_argument('--checkpoint',required=True);g.add_argument('--clip',required=True)
    g.add_argument('--config',default=str(ROOT/'configs/cldm_v15.yaml'))
    g.add_argument('--device',default='cuda');g.add_argument('--out',required=True)
    a=p.parse_args()
    if a.command=='demo':
        ref=read(ROOT/'examples/reference_1bit.json');expected=read(ROOT/'examples/expected_1bit.json');results=[]
        for old in expected:
            d=decode(ref,load(ROOT/f'examples/rgb/{old["arm"]}.png',True))
            assert d['accepted']==old['accepted'] and d['bits']==str(old['bit'])
            assert np.allclose(d['coefficient'],old['coefficient'],atol=1e-8)
            assert abs(d['score']-old['score'])<1e-8
            results.append(dict(arm=old['arm'],**d))
        dump(Path(a.out)/'decoded.json',results)
        print(json.dumps(dict(status='historical_example_reproduced',results=results),indent=2))
    elif a.command=='embed':
        curve=read(a.curve);samples=curve['samples'] if isinstance(curve,dict) else curve
        calibration=read(a.calibration) if a.calibration else None
        if calibration and 'models' in calibration:calibration=calibration['models']['Auto']
        images,mask,ref=make_reference(load(a.canny),samples,a.mode,
            load(a.mask)>0 if a.mask else None,calibration)
        out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
        for word,img in images.items():Image.fromarray(img).save(out/f'{word}.png')
        Image.fromarray(mask.astype('uint8')*255).save(out/'mask.png')
        dump(out/'reference.json',ref)
        print(f'Wrote {len(images)} conditions and reference to {out}; '+ref['calibration_scope'])
    elif a.command=='decode':
        result=decode(read(a.reference),load(a.rgb,True))
        if a.out:dump(a.out,result)
        print(json.dumps(result,indent=2))
    elif a.command=='metrics':
        result=summarize_records(read(a.records))
        if a.out:dump(a.out,result)
        print(json.dumps(result,indent=2))
    elif a.command=='probe':
        from .stability import probe
        result=probe(a.canny,a.plan,a.out)
        print(f'{result["stable_pixels"]}/{result["source_pixels"]} stable source pixels; '
              f'{result["eligible"]}/{len(result["candidates"])} verified curve candidates')
    elif a.command=='select':
        edge=load(a.canny);candidates=read(a.candidates)
        if candidates.get('policy',{}).get('version')!='region_first_v1':
            raise ValueError('Run the region-first probe before selecting a carrier.')
        if candidates['canny_sha256']!=hashlib.sha256(Path(a.canny).read_bytes()).hexdigest():
            raise ValueError('Original Canny hash differs from screening input.')
        out=Path(a.out)
        if out.exists():raise FileExistsError('Choose a new output directory.')
        rejected=[]
        for c in candidates['candidates']:
            if not c['stable_eligible']:continue
            try:images,mask,ref=make_reference(edge,c['samples'],a.mode)
            except ValueError as exc:
                rejected.append(dict(curve_index=c['curve_index'],reason=str(exc)));continue
            out.mkdir(parents=True)
            for word,img in images.items():Image.fromarray(img).save(out/f'{word}.png')
            Image.fromarray(mask.astype('uint8')*255).save(out/'mask.png')
            dump(out/'reference.json',ref)
            dump(out/'selection.json',dict(policy=candidates['policy'],selected=c,
                rejected=rejected,canny_sha256=candidates['canny_sha256'],
                calibration_scope=ref['calibration_scope']))
            print(f'Selected curve {c["curve_index"]}; input closure passed; RGB calibration unvalidated')
            break
        else:
            raise ValueError(f'No writable verified stable curve; no carrier selected. Rejections: {rejected}')
    elif a.command in ['generate','probe-generate']:
        from .generation import generate
        generate(a)


if __name__=='__main__':
    main()
