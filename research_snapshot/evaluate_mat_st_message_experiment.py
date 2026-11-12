"""Gate-respecting diffusion, fixed decoding, and review for MAT local S/T."""
import argparse
import html
import shutil
import time
import numpy as np
import cv2
from PIL import Image,ImageDraw
import mat_st_message_experiment as core
from mat_st_message_experiment import *


def freeze_analysis():
    path=OUT/'analysis_freeze.json'
    if path.exists():
        assert read(path)['script_sha256']==digest(__file__)
        return
    assert not list((OUT/'rgb').glob('*.png'))
    gen=read(OUT/'protocol.json')['generation']
    modelpaths=[ROOT/gen['checkpoint'],ROOT/gen['model_config']]
    protected={str(p):digest(p) for p in modelpaths}
    protected.update({p:digest(p) for p in read(S1/'protocol.json')['frozen_model_code']})
    dump(path,dict(script_sha256=digest(__file__),protocol_sha256=digest(OUT/'protocol.json'),models_and_code=protected,
        receiver='No fitting: nominal centers and presence rule fixed in original protocol; known local source curve. All erased samples are full-message failures.',
        quality='Same source, prompt, seed and exact latent noise G0 reference; all conditions counted including decode failures',
        uncertainty='10000 bootstrap by content;10development cases, not independent confirmation'))
    shutil.copy2(__file__,OUT/'source'/Path(__file__).name)


def worker():
    import torch
    verify();make_plan()
    if not (OUT/'plan.json').exists():return
    freeze_analysis();f=read(OUT/'analysis_freeze.json')
    for p,h in f['models_and_code'].items():assert digest(p)==h,p
    plan=read(OUT/'plan.json');gate=read(OUT/'rgb_gate_freeze.json')
    assert digest(OUT/'plan.json')==gate['plan_sha256']
    for p,h in gate['conditions'].items():assert digest(p)==h,p
    assert read(OUT/f'input_gate_L{plan[0]["length"]}.json')['passed']
    from cldm.ddim_hacked import DDIMSampler
    from controlnet_robin.pipeline import ControlNetROBINPipeline
    assert torch.cuda.is_available();torch.set_num_threads(8);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;cv2.setNumThreads(1)
    class DeviceSampler(DDIMSampler):
        def register_buffer(self,name,attr):setattr(self,name,attr.to(self.model.device) if isinstance(attr,torch.Tensor) else attr)
    gen=read(OUT/'protocol.json')['generation']
    pipe=ControlNetROBINPipeline(str(ROOT/gen['model_config']),str(ROOT/gen['checkpoint']),'cuda')
    pipe.model.requires_grad_(False).eval();pipe.model.cond_stage_model.device='cuda';pipe.ddim=DeviceSampler(pipe.model)
    pipe.make_schedule(gen['steps'],eta=gen['eta']);pipe.model.control_scales=[gen['control_scale']]*13
    for i,j in enumerate(plan):
        record=OUT/'records'/f'{j["job"]}.json'
        if record.exists():assert digest(j['rgb'])==read(record)['rgb_sha256'];continue
        assert digest(j['hint'])==j['hint_sha256']
        noise=torch.randn((1,4,64,64),generator=torch.Generator(device='cpu').manual_seed(j['seed']))
        if Path(j['noise']).exists():assert np.array_equal(noise.numpy(),np.load(j['noise']))
        else:np.save(j['noise'],noise.numpy())
        cond,uncond=pipe.get_conditioning(j['prompt'],np.array(Image.open(j['hint']).convert('RGB')),'',512,False)
        latent=noise.clone().cuda();start=time.monotonic()
        with torch.inference_mode():
            for step in range(pipe.total_steps):latent=pipe.ddim_step(latent,step,cond,uncond,gen['guidance'])
            rgb=((pipe.model.decode_first_stage(latent)+1)/2).clamp(0,1)
            image=(rgb[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
        save(j['rgb'],image)
        dump(record,dict(**j,rgb_sha256=digest(j['rgb']),noise_sha256=digest(j['noise']),protocol_sha256=digest(OUT/'protocol.json'),
            gate_freeze_sha256=digest(OUT/'rgb_gate_freeze.json'),analysis_freeze_sha256=digest(OUT/'analysis_freeze.json'),elapsed_seconds=time.monotonic()-start))
        print('RGB',i+1,'/140',j['job'],flush=True)
    dump(OUT/'rgb_status.json',dict(status='complete',images=len(plan),gate_passed=True,device='GPU1'))


def evaluate():
    import torch,lpips
    from carrier_screen.common import pixel_metrics
    verify();freeze_analysis()
    if not (OUT/'plan.json').exists():return
    torch.set_num_threads(4);cv2.setNumThreads(1)
    net=lpips.LPIPS(net='alex',version='0.1').eval().requires_grad_(False).cpu()
    cc={c['key']:c for c in read(OUT/'selection.json')};records=[];quality=[]
    gates=read(OUT/'protocol.json')['quality']
    for i,j in enumerate(read(OUT/'plan.json')):
        dest=OUT/'records'/f'{j["job"]}_evaluation.json'
        if dest.exists():
            d=read(dest);records.append(d['measurement'])
            if d['quality'] is not None:quality.append(d['quality'])
            continue
        assert digest(j['rgb'])==read(OUT/'records'/f'{j["job"]}.json')['rgb_sha256']
        rgb=np.array(Image.open(j['rgb']).convert('RGB'));c=cc[j['key']];s=c['segments'][str(j['length'])]
        obs=observe(s['samples'],rgb,rgb=True);auto=decode(obs['Auto'],s['origins']['Auto']);oracle=decode(obs['Oracle'],s['origins']['Oracle'])
        edge=cv2.Canny(rgb,100,200);save(OUT/'output_canny'/f'{j["job"]}.png',edge)
        m=dict(job=j['job'],key=j['key'],length=j['length'],group=j['group'],word=j['word'],decode=auto,oracle=oracle,observations=obs,
            correct=bool(j['word'] in WORDS and auto['accepted'] and auto['word']==j['word']),
            presence_and_correct=bool(j['word'] in WORDS and auto['accepted'] and auto['word']==j['word'] and auto['presence']),
            rgb_sha256=digest(j['rgb']))
        q=None
        if j['group']!='G0':
            clean=np.array(Image.open(j['clean_rgb']).convert('RGB'));v=pixel_metrics(clean,rgb)
            ta=torch.from_numpy(clean).permute(2,0,1)[None].float()/127.5-1;tb=torch.from_numpy(rgb).permute(2,0,1)[None].float()/127.5-1
            with torch.inference_mode():v['lpips_alex']=float(net(ta,tb))
            passed=dict(psnr=bool(v['psnr_infinite'] or v['psnr_db']>=gates['psnr_db_min']),ssim=v['ssim']>=gates['ssim_min'],
                lpips=v['lpips_alex']<=gates['lpips_alex_max'],edge_f1=v['edge_f1_3x3']>=gates['rgb_pair_edge_f1_3x3_min'])
            q=dict(job=j['job'],key=j['key'],group=j['group'],word=j['word'],metrics=v,gates=passed,pass_all=all(passed.values()))
            quality.append(q)
        records.append(m);dump(dest,dict(measurement=m,quality=q));print('EVAL',i+1,j['job'],auto['word'],auto['presence'],flush=True)
    dump(OUT/'rgb_measurements.json',records);dump(OUT/'quality.json',dict(gates=gates,records=quality))


def panel(path,labels,images,crop=None,side=230):
    canvas=Image.new('RGB',(side*len(images),side+32),'#eee');draw=ImageDraw.Draw(canvas)
    for i,(label,image) in enumerate(zip(labels,images)):
        im=Image.fromarray(image).convert('RGB')
        if crop:im=im.crop(crop)
        canvas.paste(im.resize((side,side),Image.Resampling.NEAREST),(i*side,32));draw.text((i*side+3,7),label,fill='black')
    canvas.save(path)


def summarize():
    result=dict(input={},input_diagnostics={},rgb={},raw_mat_diagnostics={},no_independent_test=True)
    for length in LENGTHS:
        if not (OUT/f'input_gate_L{length}.json').exists():continue
        result['input'][str(length)]=read(OUT/f'input_gate_L{length}.json')
        measurements=read(OUT/f'input_measurements_L{length}.json')
        diagnostics={}
        for group in ('G0','G1','G2','G3','G4','G3raw'):
            rows=[r for r in measurements if r['group']==group]
            diagnostics[group]=dict(
                erased=sum(not r['decode']['accepted'] for r in rows),
                presence_and_correct=sum(r['correct'] and r['decode']['presence'] for r in rows),
                oracle_correct=sum(r['oracle']['accepted'] and r['oracle']['word']==r['word'] for r in rows),
                word_confusion={word:{label:sum(r['word']==word and (r['decode']['word'] or 'erasure')==label for r in rows)
                    for label in (*WORDS,'erasure')} for word in (('none',) if group in ('G0','G1') else WORDS)},
                per_content={c:dict(attempts=sum(r['key']==c for r in rows),correct=sum(r['key']==c and r['correct'] for r in rows))
                    for c in sorted({r['key'] for r in rows})})
        result['input_diagnostics'][str(length)]=diagnostics
        details=[read(p) for p in (OUT/'mat_raw').glob(f'*_L{length}/m*.json')]
        result['raw_mat_diagnostics'][str(length)]=dict(attempts=len(details),
            raw_changes_vs_G1_median=float(np.median([x['raw_vs_G1_pixels'] for x in details])),
            raw_soft_change_vs_G1_median=float(np.median([x['soft_vs_G1_mae'] for x in details])),
            latent_rms_median=float(np.median([x['latent_rms'] for x in details])),
            coefficient_loss_step0_mean=float(np.mean([x['history'][0]['coefficient'] for x in details])),
            coefficient_loss_final_mean=float(np.mean([x['history'][-1]['coefficient'] for x in details])),
            projection_failures=sum(not x['G3']['success'] for x in details))
    if (OUT/'rgb_measurements.json').exists():
        rr=read(OUT/'rgb_measurements.json');qq=read(OUT/'quality.json')['records']
        for group in ('G0','G1','G2','G3','G4'):
            rows=[r for r in rr if r['group']==group];q=[r for r in qq if r['group']==group]
            result['rgb'][group]=dict(attempts=len(rows),correct=sum(r['correct'] for r in rows),
                erased=sum(not r['decode']['accepted'] for r in rows),wrong=sum(r['decode']['accepted'] and not r['correct'] for r in rows) if group not in ('G0','G1') else None,
                presence=sum(r['decode']['presence'] for r in rows),presence_and_correct=sum(r['presence_and_correct'] for r in rows),
                oracle_correct=sum(r['oracle']['accepted'] and r['oracle']['word']==r['word'] for r in rows) if group not in ('G0','G1') else None,
                quality_pass=sum(x['pass_all'] for x in q),quality_attempts=len(q),
                quality_failures={k:sum(not x['gates'][k] for x in q) for k in ('psnr','ssim','lpips','edge_f1')})
        delta={};rng=np.random.default_rng(20261111);ids=sorted({r['key'] for r in rr})
        for base in ('G2','G4'):
            per=[]
            for cid in ids:
                a=[r['correct'] for r in rr if r['key']==cid and r['group']=='G3'];b=[r['correct'] for r in rr if r['key']==cid and r['group']==base]
                per.append(float(np.mean(a)-np.mean(b)))
            per=np.array(per);delta['G3_minus_'+base]=dict(mean=float(per.mean()),ci95=np.quantile(per[rng.integers(0,len(per),(10000,len(per)))].mean(1),[.025,.975]))
        result['paired_rgb_full_word']=delta
    dump(OUT/'summary.json',result);return result


def report():
    cv2.setNumThreads(1);verify();s=summarize();cases=read(OUT/'selection.json')
    version2=OUT.name=='constraints_v2'
    docs=['<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:16px/1.6 system-ui;margin:24px;background:#eee}article{background:white;margin:24px 0;padding:16px}img{max-width:100%;image-rendering:pixelated}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #aaa}</style>',
        '<h1>实验1：局部MAT消息曲线，G0～G4</h1><p><a href="REPORT_ZH.md">完整报告</a> · <a href="summary.json">指标</a> · <a href="protocol.json">冻结定义</a> · <a href="completion_audit.json">完成审计</a></p>',
        '<p>10个既有稳定曲线开发内容，S/T四码字2bit。上方展示G0/破损输入/G1，下方每个消息对照G2、G3投影前、G3投影后和G4。所有图均为实际PNG。仅当G3输入40/40闭环通过后才启动扩散；没有通过的长度只展示输入阶段。</p>',
        '<p>原曲线参考和mask已知，不是未知RGB全图盲检测。G3优化冻结MAT的W+潜变量；强制投影的成功不单独证明MAT能按消息生成。误报来自预定近码字判别规则，没有长期FPR保证。</p>']
    if version2:docs.append('<p><a href="../index.html">初版32/48px完整对照</a></p>')
    elif (OUT/'constraints_v2/index.html').exists():docs.append('<p><a href="constraints_v2/index.html">修订约束V2：32px对照</a></p>')
    for length in LENGTHS:
        if str(length) not in s['input']:continue
        gate=s['input'][str(length)];docs.append(f'<h2>局部长度{length}px，输入门槛通过={gate["passed"]}</h2>')
        docs.append('<table><tr><th>组</th><th>尝试</th><th>完整2bit正确</th><th>几何接收</th><th>判有消息</th><th>消息正确且判存在</th></tr>')
        for group,v in gate['groups'].items():
            confirmed=s['input_diagnostics'][str(length)][group]['presence_and_correct'] if group not in ('G0','G1') else '—'
            docs.append(f'<tr><td>{group}</td><td>{v["attempts"]}</td><td>{v["correct"] if group not in ("G0","G1") else "—"}</td><td>{v["accepted"]}</td><td>{v["presence"]}</td><td>{confirmed}</td></tr>')
        docs.append('</table>')
        for c in cases:
            prefix=f'{c["key"]}_L{length}';seg=c['segments'][str(length)];p=np.array(seg['samples'])
            lo=np.maximum(0,np.floor(p.min(0)-14)).astype(int);hi=np.minimum(512,np.ceil(p.max(0)+15)).astype(int);crop=tuple([*lo,*hi])
            clean=arr(c['clean_hint']);mask=arr(seg['mask'])>0;g1=arr(OUT/'conditions'/f'{prefix}_G1.png')
            marked=np.repeat(clean[:,:,None],3,axis=2);marked[mask]=[220,60,60]
            panel(OUT/'panels'/f'{prefix}_overview.png',['G0 full Canny','Red: missing area','G1 full Canny'],[clean,marked,g1],side=320)
            panel(OUT/'panels'/f'{prefix}_reference.png',['G0 clean Canny','Missing segment','G1 MAT: no message'],[clean,np.where(mask,0,clean),g1],crop)
            docs += [f'<article id="{prefix}"><h3>{c["key"]}, curve{c["curve_index"]}, 历史稳定率{c["historical_heldout_retention"]:.3f}</h3>',f'<details><summary>整图和缺口位置</summary><img loading="lazy" src="panels/{prefix}_overview.png"></details>',f'<img loading="lazy" src="panels/{prefix}_reference.png">']
            for word in WORDS:
                labels=['G2 post-edit','G3 before projection','G3 constrained+projected','G4 direct S/T']
                paths=[OUT/'conditions'/f'{prefix}_G2_{word}.png',OUT/'mat_raw'/prefix/f'G3_preprojection_{word}.png',OUT/'conditions'/f'{prefix}_G3_{word}.png',OUT/'conditions'/f'{prefix}_G4_{word}.png']
                panel(OUT/'panels'/f'{prefix}_{word}_input.png',labels,[arr(v) for v in paths],crop)
                docs += [f'<p>{word}</p><img loading="lazy" src="panels/{prefix}_{word}_input.png">']
                if (OUT/'rgb'/f'{prefix}_G3_{word}.png').exists():
                    paths=[OUT/'rgb'/f'{prefix}_G0.png',OUT/'rgb'/f'{prefix}_G1.png']+[OUT/'rgb'/f'{prefix}_{group}_{word}.png' for group in ('G2','G3','G4')]
                    labels=['G0 clean','G1 MAT only','G2 post-edit','G3 constrained','G4 direct']
                    panel(OUT/'panels'/f'{prefix}_{word}_rgb.png',labels,[np.array(Image.open(v).convert('RGB')) for v in paths],side=200)
                    panel(OUT/'panels'/f'{prefix}_{word}_rgb_zoom.png',labels,[np.array(Image.open(v).convert('RGB')) for v in paths],crop,200)
                    edges=[cv2.Canny(np.array(Image.open(v).convert('RGB')),100,200) for v in paths]
                    panel(OUT/'panels'/f'{prefix}_{word}_output_canny.png',labels,edges,crop,200)
                    docs += [f'<img loading="lazy" src="panels/{prefix}_{word}_rgb_zoom.png"><img loading="lazy" src="panels/{prefix}_{word}_output_canny.png">',f'<details><summary>完整RGB</summary><img loading="lazy" src="panels/{prefix}_{word}_rgb.png"></details>']
            docs.append('</article>')
    (OUT/'index.html').write_text('\n'.join(docs))
    lines=['# 实验1：MAT能否按消息生成不同曲线'+('（约束修订V2）' if version2 else ''),'',
        '[全部图像](index.html) · [汇总数据](summary.json) · [冻结协议](protocol.json)','',
        '固定10个已有稳定曲线开发内容，从历史heldout匹配≥5/6、长度≥80px的内容按ID取前10，各取最长稳定曲线；没有按本轮消息/生成RGB更换内容。'+('V2只检验32px局部弧段，初版32/48px结果完整保留在父目录。' if version2 else '原曲线中段局部弧长先32px，预定失败后扩大48px；')+'两端锚点、缺口外像素保留。数据已经过历史稳定性筛选，只是开发实验。','',
        'S/T公式和四个符号不变，组合名义位移峰值1.5px，系数幅度约0.899px；将t重新归一到局部短弧段，这是相对旧整条曲线调制的明确变化。原曲线、规范方向、法向和clean输入系数用于接收，不属于无原图参考的盲识别。','',
        ('G1使用旧canny_native800步权重，输入缺损二值Canny+mask，阈值固定0.1；G3网络权重也完全冻结。V2对每消息优化W+96步，使用与0.1阈值对齐的细线BCE、前景面积和系数约束，取代初版易产生宽白带的软重心约束。V2投影仅在缺口管道内保留目标曲线并保留≤0.1px MAT高阶残差；G2/G3使用同一投影规则。初版32/48px全部结果在父目录保留。' if version2 else 'G1使用旧canny_native800步权重，输入缺损二值Canny+mask，阈值固定0.1；G3网络权重也完全冻结，仅对每消息优化W+48步。G2使用G1后相同的S/T码字投影，G3先施加消息形状/系数约束再相同投影，保留小幅MAT正交残差。')+'无受支持MAT路径时记投影失败，不使用G4兜底。G4沿用实际二值像素RasterWarp。','',
        '## 输入Canny的真实PNG闭环','',
        '|局部长度|组|完整2bit正确/尝试|几何接收|判有消息|消息正确且判存在|四图不同的内容数/10|','|---|---|---:|---:|---:|---:|---:|']
    for length,g in s['input'].items():
        for group,v in g['groups'].items():
            confirmed=str(s['input_diagnostics'][length][group]['presence_and_correct'])+'/'+str(v['attempts']) if group not in ('G0','G1') else '—'
            lines.append(f'|{length}|{group}|{str(v["correct"])+"/"+str(v["attempts"]) if group not in ("G0","G1") else "—"}|{v["accepted"]}/{v["attempts"]}|{v["presence"]}/{v["attempts"]}|{confirmed}|{v["distinct_contents"] if group not in ("G0","G1") else "—"}|')
    lines += ['','G3raw是潜变量优化后、投影前的二值输出。错误/拒绝均计失败；从落盘PNG重新读取，按相同无消息路径搜索及四个理想中心解码，不把当前PNG自身的系数用作类别中心。G0/G1没有发送消息，判有消息列分别是clean和普通修复的探索性误报。','',
        '## 潜变量约束与投影贡献','',
        'G3闭环通过本身不能证明生成式消息控制优于事后编辑，必须同时查看投影前的可读率、像素变化和约束损失。'+('V2软系数来自阈值对齐前景权重的法向重心。' if version2 else '初版软系数来自法向网格softmax重心。')+'这些系数是优化代理，不是实际二值曲线拟合真值。','',
        '|长度|优化消息数|投影失败|相对G1二值改变像素中位数|软图孔内MAE变化中位数|初始/最终软系数损失均值|','|---|---:|---:|---:|---:|---|']
    for length,v in s['raw_mat_diagnostics'].items():lines.append(f'|{length}|{v["attempts"]}|{v["projection_failures"]}|{v["raw_changes_vs_G1_median"]:.1f}|{v["raw_soft_change_vs_G1_median"]:.6f}|{v["coefficient_loss_step0_mean"]:.4f} / {v["coefficient_loss_final_mean"]:.4f}|')
    if s['rgb']:
        lines += ['','## 固定prompt/seed的RGB结果','',
            '每内容仅一组固定prompt/seed，G0/G1各1张，G2/G3/G4各四消息，共140张。20步DDIM、CFG5、eta0、ControlNet1、GPU1 float32且TF32关闭；同内容14条件共用完全相同初始latent。接收不读取目标消息、prompt、seed或配对clean RGB。未用这10个内容拟合解码/存在性阈值。','',
            '|组|完整2bit正确|错误|擦除|判有消息|消息正确且判存在|四项质量通过|','|---|---:|---:|---:|---:|---:|---:|']
        for group,v in s['rgb'].items():lines.append(f'|{group}|{str(v["correct"])+"/"+str(v["attempts"]) if group not in ("G0","G1") else "—"}|{v["wrong"] if group not in ("G0","G1") else "—"}|{v["erased"]}/{v["attempts"]}|{v["presence"]}/{v["attempts"]}|{v["presence_and_correct"] if group not in ("G0","G1") else "—"}|{str(v["quality_pass"])+"/"+str(v["quality_attempts"]) if group!="G0" else "参考"}|')
        lines += ['','全消息恢复要求两个bit同时正确；几何拒绝为擦除，分母不变。即使最近码字标签正确，若没有通过存在性规则，也不算消息被确认存在。存在性为固定近码字距离诊断，没有足够独立负例证明FPR控制或水印认证。','',
            '质量相对同prompt/seed的G0生成图：PSNR≥30dB、SSIM≥0.90、LPIPS-Alex≤0.10、RGB配对Canny Edge-F1(3×3)≥0.95，全部marked及G1均评价，未按恢复成败筛图。','',
            '|G3相对基准|完整消息正确率差|按10个内容bootstrap95%区间|','|---|---:|---|']
        for name,v in s['paired_rgb_full_word'].items():lines.append(f'|{name}|{v["mean"]:+.3f}|[{v["ci95"][0]:+.3f},{v["ci95"][1]:+.3f}]|')
    else:
        lines += ['','## 扩散阶段','',
            '**未运行扩散生成。** G3尚未满足全部10内容×4消息的输入闭环要求，遵守先解决MAT约束的门槛。输入失败不能被RGB筛选或改换内容掩盖。下一步优先诊断投影失败、原始MAT路径缺失、竞争边缘和2bit系数可辨性。']
    lines += ['','## 限制','',
        '- G3属于已知源曲线参考、冻结MAT的推理时潜变量优化＋显式几何投影。不能称为已训练的2bit消息条件MAT，也不能把投影输出解释成网络独立生成。',
        ('- V2投影在缺口管道内重建单条目标曲线，保留至多0.1px的MAT高阶残差，可能丢掉管道内其他真实分支。' if version2 else '- 初版投影保留的高阶MAT残差和原有预测杂线可能影响拟合。')+'名义S/T预算1.5px不等于全部MAT像素都被该预算约束。逐图投影峰值与实际PNG保存供复查。',
        '- 局部32/48px的S/T比原整条曲线频率更高；原长曲线的RGB信道拟合未直接套用，使用固定名义中心。因此此轮同时检验短载体、生成约束与未校准信道，不是原长曲线成绩的复现。',
        '- 开发内容只有10个且已按历史稳定性筛选；一prompt/seed不足以证明普适性。四个全部合法码字没有认证冗余。','']
    for name,title in [('FINDINGS_ZH.md','结果判断'),('VISUAL_REVIEW_ZH.md','固定样本目视观察')]:
        p=OUT/name
        if p.exists():lines += ['## '+title,'',p.read_text(),'']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines))
    audit()
    print('REPORT_COMPLETE',flush=True)


def audit():
    import re
    verify();status=read(OUT/'mat_status.json');counts={}
    for length in LENGTHS:
        f=OUT/f'input_measurements_L{length}.json'
        if not f.exists():continue
        rr=read(f);assert len(rr)==180
        assert all(r['exterior_unchanged'] for r in rr)
        for r in rr:assert digest(r['path'])==r['sha256']
        counts[str(length)]=len(rr)
    rgb_complete=False;rgb_skipped=False
    if (OUT/'rgb_status.json').exists():
        state=read(OUT/'rgb_status.json')['status'];rgb_skipped=state=='SKIPPED_INPUT_GATE_FAILED'
        if state=='complete':
            plan=read(OUT/'plan.json');assert len(plan)==140
            assert len(read(OUT/'rgb_measurements.json'))==140
            assert len(read(OUT/'quality.json')['records'])==130
            assert read(OUT/f'input_gate_L{plan[0]["length"]}.json')['passed']
            for j in plan:
                record=read(OUT/'records'/f'{j["job"]}.json')
                assert digest(j['rgb'])==record['rgb_sha256']
                assert digest(j['noise'])==record['noise_sha256']
            rgb_complete=True
    for link in re.findall(r'(?:src|href)="([^"]+)"',(OUT/'index.html').read_text()):
        assert link=='completion_audit.json' or (OUT/link).exists(),link
    dump(OUT/'completion_audit.json',dict(complete=status['complete'] and (rgb_complete or rgb_skipped),input_records=counts,
        rgb_complete=rgb_complete,rgb_skipped_by_gate=rgb_skipped,input_gate_status=status,all_condition_exteriors_unchanged=True,
        input_and_source_hashes_unchanged=True,all_page_links_valid=True,report_script_sha256=digest(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('worker','evaluate','report'));p.add_argument('--variant',choices=('v1','v2'),default='v1');a=p.parse_args()
    if a.variant=='v2':
        OUT=OUT/'constraints_v2';LENGTHS=(32,);core.OUT=OUT;core.LENGTHS=LENGTHS
    globals()[a.action]()
