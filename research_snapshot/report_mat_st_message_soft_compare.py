"""Input-stage comparison and actual-PNG gallery; no receiver fitting."""
import argparse
import json
import html
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw
from skimage.morphology import skeletonize
import mat_st_message_soft_compare as experiment
import mat_st_message_experiment as core
import evaluate_mat_st_message_experiment as evaluator
from mat_st_message_experiment import read,dump,digest,arr,WORDS

OUT=experiment.OUT
BINARY=experiment.BINARY
GROUPS=('G0','G1','G2','G3raw','G3','G4')


def select(arm):
    core.OUT=arm;core.LENGTHS=(32,);evaluator.OUT=arm;evaluator.LENGTHS=(32,)


def freeze_analysis():
    path=OUT/'analysis_plan.json'
    if path.exists():
        assert read(path)['report_script_sha256']==digest(__file__)
        return
    dump(path,dict(report_script_sha256=digest(__file__),
        input_quality='All attempts: mask-only binary strict/tolerant1px F1 and skeleton tolerant1px F1. G1 reference G0; marked reference same-message unchanged G4. All diagnostic, no filtering.',
        tolerated_distance='3x3 square, Chebyshev1; skeletonize both operands after cropping with8px context, then mask.',
        pairing='Same content and word to binary V2. Bootstrap10000 by10content clusters, RNG20261111. Development only.',
        decoder='Unchanged original full-word Auto and fixed presence rule. No fit, no expected-word path selection.',
        RGB='Original G3 ALL40 actual-input gate; run diffusion only if passed, same original worker and quality evaluation.'))
    import shutil
    shutil.copy2(__file__,OUT/'source'/Path(__file__).name)


def overlap(pred,target,tolerance):
    npred=int(pred.sum());ntarget=int(target.sum())
    if not npred or not ntarget:return float(npred==ntarget)
    kernel=np.ones((2*tolerance+1,2*tolerance+1),np.uint8)
    pp=cv2.dilate(pred.astype(np.uint8),kernel)>0;tt=cv2.dilate(target.astype(np.uint8),kernel)>0
    precision=float((pred&tt).sum()/npred);recall=float((target&pp).sum()/ntarget)
    return 2*precision*recall/(precision+recall) if precision+recall else 0.


def quality(rows,cases):
    cc={c['key']:c for c in cases};result=[]
    for r in rows:
        c=cc[r['key']];mask=arr(c['segments']['32']['mask'])>0
        yy,xx=np.nonzero(mask);slices=(slice(max(0,yy.min()-8),min(512,yy.max()+9)),slice(max(0,xx.min()-8),min(512,xx.max()+9)))
        local=mask[slices];pred=arr(r['path'])[slices]>0
        if r['group'] in ('G0','G1'):
            reference=arr(c['clean_hint'])[slices]>0
        else:
            arm=Path(r['path']).parents[2 if r['group']=='G3raw' else 1]
            reference=arr(arm/'conditions'/f'{r["key"]}_L32_G4_{r["word"]}.png')[slices]>0
        ps=skeletonize(pred)&local;ts=skeletonize(reference)&local
        pred=pred&local;reference=reference&local
        result.append(dict(key=r['key'],group=r['group'],word=r['word'],
            f1=overlap(pred,reference,0),tolerant_f1_1px=overlap(pred,reference,1),skeleton_f1_1px=overlap(ps,ts,1),
            foreground_pixels=int(pred.sum()),target_foreground_pixels=int(reference.sum())))
    return dict(records=result,groups={g:{k:float(np.mean([r[k] for r in result if r['group']==g]))
        for k in ('f1','tolerant_f1_1px','skeleton_f1_1px','foreground_pixels','target_foreground_pixels')} for g in GROUPS})


def stats(arm):
    select(arm);core.verify();rows=read(arm/'input_measurements_L32.json');cases=read(arm/'selection.json')
    summary=read(BINARY/'summary.json') if arm==BINARY else evaluator.summarize()
    groups=summary['input']['32']['groups'];diagnostic=summary['input_diagnostics']['32']
    for g in groups:
        groups[g].update({k:diagnostic[g][k] for k in ('erased','presence_and_correct','oracle_correct')})
        groups[g]['all_four_correct_contents']=sum(all(r['correct'] for r in rows if r['key']==c['key'] and r['group']==g) for c in cases) if g not in ('G0','G1') else None
    q=quality(rows,cases);dump(OUT/'diagnostics'/f'{arm.name}_repair_quality.json',q)
    return dict(groups=groups,raw_mat_diagnostics=summary['raw_mat_diagnostics']['32'],quality=q['groups'])


def paired(arm):
    a=read(arm/'input_measurements_L32.json');b=read(BINARY/'input_measurements_L32.json');result={}
    for group in ('G2','G3raw','G3'):
        aa={(r['key'],r['word']):r['correct'] for r in a if r['group']==group}
        bb={(r['key'],r['word']):r['correct'] for r in b if r['group']==group};assert aa.keys()==bb.keys()
        ids=sorted({k[0] for k in aa});per=np.array([np.mean([int(aa[k])-int(bb[k]) for k in aa if k[0]==cid]) for cid in ids])
        rng=np.random.default_rng(20261111);ci=np.quantile(per[rng.integers(0,10,(10000,10))].mean(1),[.025,.975])
        result[group]=dict(mean=float(per.mean()),ci95=ci,unit='proportion difference; resample10contents, not40independentmessages')
    return result


def collage(path,labels,paths,crop=None,side=240):
    images=[np.asarray(Image.open(p).convert('RGB')) for p in paths]
    evaluator.panel(path,labels,images,crop=crop,side=side)


def gallery(arms):
    cases=read(BINARY/'selection.json');(OUT/'panels').mkdir(exist_ok=True)
    doc=['<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:16px/1.6 system-ui;background:#eee;margin:24px}article{background:white;margin:20px 0;padding:16px}img{max-width:100%;image-rendering:pixelated}td,th{border:1px solid #aaa;padding:7px}table{border-collapse:collapse}</style>',
        '<h1>G1–G3：二值MAT与PiDiNet-soft MAT</h1><p><a href="REPORT_ZH.md">报告</a> · <a href="comparison.json">汇总</a> · <a href="protocol.json">协议</a> · <a href="completion_audit.json">审计</a></p>',
        '<p>同10内容、32px缺口、四消息。先遮盖Canny→旧学习转换器→再次清空soft缺口→旧soft MAT→固定0.45硬化。最终Canny缺口外保留原二值输入。G3投影前已经使用消息约束；投影后另有显式几何编码贡献。已知原曲线参考，不是无参考盲检。所有失败均展示。</p>',
        '<p>连续soft输出单独展示，不能把灰度图的视觉平滑当作实际二值曲线可读。仅输入40/40通过后才运行RGB；RGB状态见报告。</p>']
    names={'binary':'Binary MAT V2','native':'Soft MAT native','refined':'Soft MAT refined'}
    for c in cases:
        prefix=c['key']+'_L32';p=np.array(c['segments']['32']['samples']);lo=np.maximum(0,np.floor(p.min(0)-14)).astype(int);hi=np.minimum(512,np.ceil(p.max(0)+15)).astype(int);crop=(*lo,*hi)
        edge=arr(c['clean_hint']);mask=arr(c['segments']['32']['mask'])>0
        marked=np.repeat(edge[:,:,None],3,2);marked[mask]=[220,60,60]
        evaluator.panel(OUT/'panels'/f'{prefix}_location.png',['Full clean Canny','Red: erased area'],[edge,marked],side=400)
        paths=[BINARY/'conditions'/f'{prefix}_G0.png']+[a/'conditions'/f'{prefix}_G1.png' for a in arms.values()]
        labels=['G0 clean']+[names[n]+' G1' for n in arms]
        collage(OUT/'panels'/f'{prefix}_G1.png',labels,paths,crop)
        doc += [f'<article id="{prefix}"><h2>{c["key"]} {html.escape(c["caption"])}</h2>',f'<details><summary>整图与缺口位置</summary><img loading="lazy" src="panels/{prefix}_location.png"></details>',f'<p>G1普通修复，硬化后的实际Canny</p><img loading="lazy" src="panels/{prefix}_G1.png">']
        for name,arm in arms.items():
            if name=='binary':continue
            paths=[arm/'inputs'/f'{prefix}_masked_soft.png',arm/'mat_raw'/prefix/'G1_soft_composite.png',arm/'conditions'/f'{prefix}_G1.png']
            dest=f'{prefix}_{name}_soft.png';collage(OUT/'panels'/dest,['Masked approximate soft','G1 repaired soft','G1 hardened Canny'],paths,crop)
            doc += [f'<p>{names[name]}：soft输入、修复soft、硬化结果</p><img loading="lazy" src="panels/{dest}">']
        for word in WORDS:
            doc.append(f'<h3>{word}</h3>')
            for group in ('G2','G3raw','G3'):
                paths=[a/'mat_raw'/prefix/f'G3_preprojection_{word}.png' if group=='G3raw' else a/'conditions'/f'{prefix}_{group}_{word}.png' for a in arms.values()]
                paths += [BINARY/'conditions'/f'{prefix}_G4_{word}.png']
                labels=[names[n]+' '+group for n in arms]+['G4 direct S/T']
                dest=f'{prefix}_{word}_{group}.png';collage(OUT/'panels'/dest,labels,paths,crop)
                doc += [f'<p>{group}</p><img loading="lazy" src="panels/{dest}">']
            paths=[a/'mat_raw'/prefix/f'G3_soft_{word}.png' for n,a in arms.items() if n!='binary']
            dest=f'{prefix}_{word}_soft.png';collage(OUT/'panels'/dest,[names[n]+' G3 soft' for n in arms if n!='binary'],paths,crop)
            doc += [f'<details><summary>G3投影前连续soft输出</summary><img loading="lazy" src="panels/{dest}"></details>']
        doc.append('</article>')
    (OUT/'index.html').write_text('\n'.join(doc))


def report():
    freeze_analysis();cv2.setNumThreads(1)
    arms={'binary':BINARY,**{s:OUT/s for s in experiment.STRUCTURES if (OUT/s/'input_gate_L32.json').exists()}}
    summary={n:stats(a) for n,a in arms.items()};differences={n:paired(a) for n,a in arms.items() if n!='binary'}
    dump(OUT/'comparison.json',dict(arms=summary,paired_vs_binary=differences,development_only=True))
    gallery(arms)
    lines=['# G1–G3使用PiDiNet-soft MAT的配对比较','',
        '[全部图像](index.html) · [指标](comparison.json) · [协议](protocol.json) · [审计](completion_audit.json)','',
        '沿用二值MAT V2的10个开发内容、原曲线、32px局部缺口、S/T四个2bit码字、约1.5px名义位移和相同接收器。软化使用旧学习转换器的PiDiNet-soft分支；先遮盖Canny，再转换，并清空转换图中的孔。两种soft结构各使用已有800步权重、原验证阈值0.45，无训练或阈值调优。模型未接收源RGB。','',
        'G1无消息，G2在普通修复后投影，G3在修复过程中以相同V2阈值对齐BCE/系数/面积损失优化W+96步后投影。G3raw表示已使用消息约束、硬化后但未投影的输出。所有模型权重冻结，最终二值图缺口外恢复原Canny；连续soft合成图另存。','',
        '## 实际PNG的完整2bit接收','',
        '|模型|组|正确/尝试|几何接收|判有消息|正确且判存在|全部四码字正确的内容/10|','|---|---|---:|---:|---:|---:|---:|']
    for name,s in summary.items():
        for group in GROUPS:
            d=s['groups'][group];marked=group not in ('G0','G1')
            lines.append(f'|{name}|{group}|{str(d["correct"])+"/"+str(d["attempts"]) if marked else "—"}|{d["accepted"]}/{d["attempts"]}|{d["presence"]}/{d["attempts"]}|{str(d["presence_and_correct"])+"/"+str(d["attempts"]) if marked else "—"}|{d["all_four_correct_contents"] if marked else "—"}|')
    lines += ['','G0/G1没有消息，“判有消息”为固定规则下的探索性误报；擦除计失败、分母不变，不将最近码字标签正确自动视为消息被确认。10个历史筛选内容只作开发，不证明普适性或低FPR。','',
        '## 相对二值V2的配对变化','',
        '|模型|组|完整消息正确率差|按10内容bootstrap95%区间|','|---|---|---:|---|']
    for name,groups in differences.items():
        for group,d in groups.items():lines.append(f'|{name}|{group}|{d["mean"]:+.3f}|[{d["ci95"][0]:+.3f},{d["ci95"][1]:+.3f}]|')
    lines += ['','## 缺口内二值形状诊断','',
        'G1以原Clean Canny为参考；G2/G3raw/G3以同消息G4直接编辑的二值图为参考。统计全部样本，不能与RGB质量等同。容差1px使用3×3方形邻域，骨架指标两侧均细化。与完整Canny相比，投影可能删除管道内其他真实分支，因此需同时观察完整图。','',
        '|模型|组|严格F1|容差1px F1|骨架容差1px F1|前景像素/参考像素均值|','|---|---|---:|---:|---:|---|']
    for name,s in summary.items():
        for group in ('G1','G2','G3raw','G3'):
            d=s['quality'][group];lines.append(f'|{name}|{group}|{d["f1"]:.4f}|{d["tolerant_f1_1px"]:.4f}|{d["skeleton_f1_1px"]:.4f}|{d["foreground_pixels"]:.1f} / {d["target_foreground_pixels"]:.1f}|')
    lines += ['','## RGB状态','']
    for name,arm in arms.items():
        state=read(arm/'rgb_status.json') if (arm/'rgb_status.json').exists() else {'status':'pending'}
        lines.append(f'- {name}: {state["status"]}')
    lines += ['','未达到原G3的40/40门槛的组不运行扩散。输入指标不能解释为RGB恢复率或RGB质量提升。','',
        '## 解释范围','',
        '- 比较的是不同旧领域权重及转换管线，不是仅改变同一模型输入的单因素实验。Refiner也有单独权重。',
        '- Soft模型W+映射在autocast内初始化，逐例与旧标准soft前向完全一致；二值V2映射在autocast外，因此还存在初始化数值精度差异。',
        '- 接收已知原始局部曲线和参考系数，不是随机RGB无参考盲检。四个合法码字没有认证冗余。',
        '- 投影显式重建S/T曲线，不能把投影后的成功全部归于MAT生成；必须查看G3raw。',
        '- 本次使用先遮盖再转换；没有测试完整Canny先转换后遮盖，也没有测试原图原生PiDiNet soft输入。','']
    for name in ('FINDINGS_ZH.md','VISUAL_REVIEW_ZH.md'):
        if (OUT/name).exists():lines += [(OUT/name).read_text(),'']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines))
    audit(arms);dump(OUT/'RUN_STATUS.json',dict(status='complete' if len(arms)==3 else 'partial',arms=list(arms),RGB={n:read(a/'rgb_status.json')['status'] for n,a in arms.items()}))
    print('SOFT_REPORT_COMPLETE',flush=True)


def audit(arms):
    import re
    cases=read(BINARY/'selection.json');result={}
    for name,arm in arms.items():
        select(arm);core.verify();rows=read(arm/'input_measurements_L32.json');assert len(rows)==180
        assert read(arm/'selection.json')==cases
        for r in rows:
            assert digest(r['path'])==r['sha256'];assert r['exterior_unchanged']
        for c in cases:
            prefix=c['key']+'_L32'
            for suffix in ['G0']+[f'G4_{w}' for w in WORDS]:
                assert digest(arm/'conditions'/f'{prefix}_{suffix}.png')==digest(BINARY/'conditions'/f'{prefix}_{suffix}.png')
        state=read(arm/'rgb_status.json')['status'];passed=read(arm/'input_gate_L32.json')['passed']
        if not passed:assert state=='SKIPPED_INPUT_GATE_FAILED' and not list((arm/'rgb').glob('*.png'))
        if name!='binary':
            assert len(read(arm/'input_audit.json'))==10
            assert all(r['hidden_canny_invariant'] and r['manual_W_forward_matches_model_exactly'] for r in read(arm/'input_audit.json'))
        result[name]=dict(records=180,all_input_exteriors_unchanged=True,same_contents_masks_G0_G4=True,freeze_verified=True,gate_passed=passed,RGB=state)
    for link in re.findall(r'(?:src|href)="([^"]+)"',(OUT/'index.html').read_text()):
        assert link=='completion_audit.json' or (OUT/link).exists(),link
    dump(OUT/'completion_audit.json',dict(complete=len(arms)==3,arms=result,all_page_links_valid=True,report_script_sha256=digest(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze_analysis','report','worker','evaluate'));p.add_argument('--structure',choices=experiment.STRUCTURES,default='native');args=p.parse_args()
    if args.action in ('worker','evaluate'):
        select(OUT/args.structure);getattr(evaluator,args.action)()
    else:globals()[args.action]()
