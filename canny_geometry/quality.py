"""Original paired image metrics plus frozen LPIPS quality gates."""
from pathlib import Path
import json
import numpy as np
import cv2
from .benchmark import read,dump,sha,image,verify

# From carrier_screen/common.py:117; numerical body unchanged.
def edge_f1(a, b):
    aa, bb = cv2.Canny(a, 100, 200) > 0, cv2.Canny(b, 100, 200) > 0
    if not aa.any() and not bb.any():
        return 1.0
    kernel = np.ones((3, 3), np.uint8)
    ad, bd = cv2.dilate(aa.astype(np.uint8), kernel), cv2.dilate(bb.astype(np.uint8), kernel)
    precision = float(ad[bb].mean()) if bb.any() else 0.0
    recall = float(bd[aa].mean()) if aa.any() else 0.0
    return 2 * precision * recall / max(precision + recall, 1e-12)

# From carrier_screen/common.py:128; numerical body unchanged.
def pixel_metrics(a, b):
    from skimage.metrics import structural_similarity
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64))**2))
    return dict(mse=mse, psnr_db=float(10 * np.log10(255**2 / mse)) if mse else None,
                psnr_infinite=mse == 0,
                ssim=float(structural_similarity(a, b, data_range=255, channel_axis=2)),
                edge_f1_3x3=edge_f1(a, b), changed_pixels=int(np.any(a != b, axis=2).sum()))


def evaluate(args):
    import torch,lpips
    torch.set_num_threads(4);cv2.setNumThreads(1)
    # Require the documented local cache; do not trigger a hidden model download.
    cache=Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth'
    if not cache.is_file():raise FileNotFoundError('Download the documented LPIPS AlexNet backbone first: '+str(cache))
    manifest=verify(args.manifest);jobs={j['id']:j for j in manifest['generation_jobs']}
    rgbmap=read(args.rgb_map) if args.rgb_map else None
    def path(gid):return Path(rgbmap[gid]) if rgbmap else Path(args.rgb_root)/'rgb'/(gid+'.png')
    expected={r['row_id']:r for r in manifest.get('quality_rows',[])}
    if expected:
        selected=[r for r in manifest['evaluation_rows'] if r['id'] in expected]
    else:
        selected=[r for r in manifest['evaluation_rows'] if r['truth'] is not None and not r['group'].startswith('region_first_input/')]
    limits=manifest.get('quality_limits',dict(psnr_db_min=30,ssim_min=.9,lpips_alex_max=.1,edge_f1_3x3_min=.95))
    net=lpips.LPIPS(net='alex',version='0.1').eval().requires_grad_(False).cpu()
    cache_values={};rows=[];mismatches=[]
    for i,row in enumerate(selected):
        gid=row['generation_id'];clean=row['clean_generation_id'];token=(gid,clean)
        if token not in cache_values:
            a=image(path(clean),True);b=image(path(gid),True);q=pixel_metrics(a,b)
            ta=torch.from_numpy(a.copy()).permute(2,0,1)[None].float()/127.5-1
            tb=torch.from_numpy(b.copy()).permute(2,0,1)[None].float()/127.5-1
            with torch.inference_mode():q['lpips_alex']=float(net(ta,tb))
            gates=dict(psnr=bool(q['psnr_infinite'] or q['psnr_db']>=limits['psnr_db_min']),ssim=bool(q['ssim']>=limits['ssim_min']),lpips=bool(q['lpips_alex']<=limits['lpips_alex_max']),edge_f1=bool(q['edge_f1_3x3']>=limits['edge_f1_3x3_min']))
            cache_values[token]=dict(quality=q,gates=gates,pass_all=all(gates.values()))
        measured=cache_values[token];old=expected.get(row['id'])
        if old is not None:
            delta={}
            for key in ['psnr_db','ssim','lpips_alex','edge_f1_3x3']:
                av=measured['quality'][key];bv=old['expected_metrics'][key]
                if av is None or bv is None:
                    if av!=bv:delta[key]=[av,bv]
                elif abs(av-bv)>1e-5:delta[key]=[av,bv]
            if measured['pass_all']!=old['expected_pass'] or delta:
                mismatches.append(dict(id=row['id'],pass_match=measured['pass_all']==old['expected_pass'],metric_differences=delta))
        rows.append(dict(id=row['id'],group=row['group'],truth=row['truth'],**measured))
        if (i+1)%50==0:print(f'QUALITY {i+1}/{len(selected)}',flush=True)
    groups={g:dict(attempts=sum(r['group']==g for r in rows),passed=sum(r['group']==g and r['pass_all'] for r in rows)) for g in sorted({r['group'] for r in rows})}
    result=dict(status='passed' if not mismatches else 'failed',groups=groups,records=rows,expected_mismatches=mismatches,limits=limits)
    dump(Path(args.out)/'quality.json',result)
    print(json.dumps(dict(status=result['status'],groups=groups,expected_mismatches=len(mismatches)),indent=2))
    if mismatches:raise RuntimeError('Quality reconstruction differs from frozen reference.')
