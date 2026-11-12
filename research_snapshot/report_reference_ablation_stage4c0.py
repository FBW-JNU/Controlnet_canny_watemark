"""Descriptive report/audit for the frozen reference-removal experiment."""
from curve_reference_ablation_stage4c0 import *
import inspect, html, re, ast

LABELS={'R0':'R0 原始参考辅助','R0_public':'对照：仅换公共码本','R1_zero_clean':'对照：再去 clean、不重校准',
        'R1_public':'R1 去 clean＋公共码本（Cal 校准）','R2_tube':'R2 原管道 Mask 自参考','R2_bbox':'R2 矩形 Mask 自参考','R3':'R3 全图自参考'}


def checked_summary():
    result=read(OUT/'summary.json');records=load('Eval')
    # The frozen decision export omits R0 posterior; read it from raw receiver.
    count=sum(bool(r['arm'] in WORDS and r['R0']['accepted'] and r['R0'].get('high_confidence',False) and r['R0']['word']!=r['arm']) for r in records)
    before=result['R0']['high_confidence_wrong'];result['R0']['high_confidence_wrong']=count
    dump(OUT/'metrics/summary_export_correction.json',dict(field='R0.high_confidence_wrong',raw_export=before,checked=count,
        reason='Frozen decisions export omitted R0 posterior/high_confidence; checked directly in eval_features/* R0 raw records. No receiver, threshold, accepted status, word, score or accuracy changed.',
        raw_summary_sha256=digest(OUT/'summary.json')))
    dump(OUT/'summary_checked.json',result);return result


def audit():
    check();counts=collections.Counter();cal=load('Cal');ev=load('Eval');frozen=read(OUT/'score_freeze.json')
    assert len(cal)==92 and len(ev)==176
    assert len([r for r in ev if r['arm'] in WORDS])==48 and len([r for r in ev if r['arm']=='clean'])==128
    assert not {r['content_id'] for r in cal}&{r['content_id'] for r in ev}
    for name,h in frozen['cal_features_sha256'].items():assert digest(OUT/'cal_features'/name)==h
    assert frozen['protocol_sha256']==digest(OUT/'protocol.json') and frozen['no_eval_features_before_freeze']
    assert list(inspect.signature(extract_rgb).parameters)==['rgb','mask']
    assert list(inspect.signature(classify).parameters)==['value','fit_cov','model','key']
    # Analytic degree/rank and all residual pair distances independently checked.
    for row in read(OUT/'metrics/baseline_projection.json'):
        metrics=row['metrics'];degree=row['degree'];assert metrics['rank']==(2 if degree<=2 else 1 if degree==3 else 0)
    for record in cal+ev:
        assert digest(record['rgb'])==record['rgb_sha256'];assert digest(record['mask'])==record['mask_sha256'];counts['unchanged_RGB']+=1
        edge=cv2.Canny(np.array(Image.open(record['rgb']).convert('RGB')),100,200);mask=np.array(Image.open(record['mask']).convert('L'))>0
        for method,data in record['public'].items():
            if data['selected_path'] is not None:
                p=np.array(data['selected_path']);assert np.array_equal(p,p.round());assert tuple(p[0])<tuple(p[-1]);assert np.all(edge[p[:,1].astype(int),p[:,0].astype(int)]>0)
                if method=='R2_tube':assert np.all(mask[p[:,1].astype(int),p[:,0].astype(int)])
            for degree,f in data['fits'].items():
                if not f.get('fitted') or 'coefficient' not in f:continue
                p=np.array(f['observed_xy']);assert np.array_equal(p,p.round());assert np.all(edge[p[:,1].astype(int),p[:,0].astype(int)]>0);counts['actual_pixels_checked']+=len(p)
                t=np.array(f['t']);w=np.array(f['weights']);info,base,residual=information(t,w,int(degree))
                np.testing.assert_allclose(base.T@(w[:,None]*residual),0.,atol=1e-7)
                for pair in info['pairs']:
                    a,b=pair['pair'];d=b4.a4.AMPLITUDE*(np.array(b4.a4.SIGNS[a])-np.array(b4.a4.SIGNS[b]));profile=residual@d
                    np.testing.assert_allclose(np.sum(w*profile**2),pair['d2'],atol=1e-7);counts['direct_distances_checked']+=1
                counts['projected_Gram_checked']+=1
    # Continuous positive control: exact coefficients recover under valid degree.
    t=np.linspace(0,1,301);phi=basis(t);truth=np.array([.7,-.6])
    for degree in [0,1,2]:
        base=np.polynomial.legendre.legvander(2*t-1,degree);baseline=np.linspace(-1,1,degree+1)
        y=base@baseline+phi@truth;estimated=np.linalg.lstsq(np.c_[base,phi],y,rcond=None)[0][-2:]
        np.testing.assert_allclose(estimated,truth,atol=1e-10);counts['continuous_positive_controls']+=1
    # R0 reproduces previous same-PNG decisions, public label-free receiver replay.
    prior=read(ROOT/'outputs/canny_st_identifiability_comparison_20261111/comparison.json');prior={r['job']:r for r in prior if r['method']=='B4'}
    decisions=read(OUT/'decisions.json');di={r['job']:r for r in decisions}
    for record in ev:
        d=di[record['job']]['methods']['R0']
        if record['arm'] in WORDS:
            p=prior[record['job']]['native'];assert (d['accepted'],d['word'])==(p['accepted'],p['word']);counts['R0_same_PNG_reproductions']+=1
        for method in ['R1_public','R2_tube','R2_bbox','R3']:
            item=frozen[method];conf=item['selected'];model=item['model']
            f=record['reference_observation'] if method=='R1_public' else record['public'][method]['fits'][str(conf['degree'])]
            ok=f.get('accepted',False) if method=='R1_public' else eligible(f,conf['residual'],conf['ci95'])
            d=classify(f['coefficient'] if ok else None,f.get('covariance',np.eye(2)),model)
            stored=di[record['job']]['methods'][method]
            assert d['accepted']==stored['accepted'] and d['word']==stored['word'] and d['score']==stored['score']
            assert stored['score']==stored['wrong_key_score'];counts['public_decoder_replays']+=1
    # Re-extract public inputs for fixed predeclared examples without reference data.
    for record in ev[:2]:
        rgb=np.array(Image.open(record['rgb']).convert('RGB'));mask=np.array(Image.open(record['mask']).convert('L'))>0
        assert extract_rgb(rgb,mask)==record['public']['R2_tube'];assert extract_rgb(rgb)==record['public']['R3'];counts['public_extraction_replays']+=2
    for path,h in read(OUT/'protocol.json')['parents'].items():assert digest(path)==h
    dump(OUT/'audit.json',dict(status='passed',counts=dict(counts),Cal_Eval_content_disjoint=True,
        public_input_signatures={'extract_rgb':str(inspect.signature(extract_rgb)),'classify':str(inspect.signature(classify))},
        no_new_RGB=True,all48_marked_denominator=True,all128_clean_denominator=True,
        limitations='Input boundary and numeric tests do not prove selected curve identity. Reused Eval is diagnostic, not new independent acceptance.'))


def source_audit():
    _,cc=dataset('Eval');marked={r['key'] for r in load('Eval') if r['arm'] in WORDS};rows=[]
    for key in sorted(marked):
        c=cc[key];origin=np.array(c['states']['clean']['measurements']['Auto']['coefficient']);differences=[]
        for word in WORDS:
            actual=b4.a4.features(c,word,'Auto');nominal=b4.a4.AMPLITUDE*np.array(b4.a4.SIGNS[word]);differences.append(np.linalg.norm(actual-nominal))
        rows.append(dict(key=key,clean_coefficient=origin.tolist(),all4_actual_vs_nominal_distances_px=list(map(float,differences))))
    records=load('Eval');maskdist=[r['mask_information']['source_to_mask_skeleton_median_px'] for r in records if r['mask_information']['source_to_mask_skeleton_median_px'] is not None]
    same_masks=all(len({r['mask_sha256'] for r in records if r['key']==key})==1 for key in marked)
    out=dict(codebook_source='Original a4.features takes each of all4 input-state coefficients minus clean. b4.decode constructs all4 centers. It does not select the actual message codeword before classification.',
        consequence='All4 conditional source-derived centers are sender-side reference information. R0_public/R1 use no per-image input-state centers.',
        public_book='Nominal fixed amplitude/signs, original global Cal model for controls; global Cal output class centers for public R1/R2/R3. No Eval per-content model.',
        masks=dict(same_file_across_all4_messages=same_masks,source_curve_to_mask_skeleton_median_px=float(np.median(maskdist)),
                   message='Tube masks carry fine source position/shape but no per-message change; bbox is location+extent, R3 has no mask.'),
        source_stats=rows,median_clean_coefficient_norm_px=float(np.median([np.linalg.norm(r['clean_coefficient']) for r in rows])),
        median_actual_vs_nominal_distance_px=float(np.median([x for r in rows for x in r['all4_actual_vs_nominal_distances_px']])))
    dump(OUT/'metrics/source_information_audit.json',out);return out


def figures(records,frozen,projection):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for pattern in ['full','center25','center50','random50']:
        rr=[r for r in projection if r['pattern']==pattern]
        axes[0].plot(DEGREES,[r['metrics']['residual_energy_ratio'][0] for r in rr],'o-',label=pattern)
        axes[1].plot(DEGREES,[r['metrics']['residual_energy_ratio'][1] for r in rr],'o-',label=pattern)
        axes[2].plot(DEGREES,[r['metrics']['min_codeword_rms'] for r in rr],'o-',label=pattern)
    for ax,title in zip(axes,['S energy after natural-baseline projection','T energy after natural-baseline projection','Minimum projected codeword RMS (px)']):
        ax.set_title(title,fontsize=10);ax.set_xlabel('Natural baseline polynomial degree');ax.set_xticks(DEGREES);ax.grid(alpha=.3);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'plots/baseline_projection.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(18,4));colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728']
    for ax,method in zip(axes,['R1_public','R2_tube','R2_bbox','R3']):
        item=frozen[method];config=item['selected']
        for word,color in [('clean','#888888')]+list(zip(WORDS,colors)):
            values=[]
            for r in records:
                if r['arm']!=word:continue
                f=r['reference_observation'] if method=='R1_public' else r['public'][method]['fits'][str(config['degree'])]
                if not f.get('fitted') or 'coefficient' not in f:continue
                values.append(f['coefficient'])
            if values:
                v=np.array(values);ax.scatter(v[:,0],v[:,1],s=12,alpha=.55,color=color,label=word)
        centers=np.array(item['model'].get('centers',[]))
        if len(centers):ax.scatter(centers[:,0],centers[:,1],marker='X',s=70,c=colors,edgecolors='black')
        ax.set_title(method+(' degree='+str(config['degree']) if config['degree'] is not None else ''));ax.set_xlabel('S coefficient (px)');ax.set_ylabel('T coefficient (px)');ax.grid(alpha=.25)
    axes[0].legend(fontsize=7);fig.suptitle('Existing Eval: raw fitted coefficients including gate rejects; X = frozen Cal centers')
    fig.tight_layout();fig.savefig(OUT/'plots/coefficient_distributions.png',dpi=160);plt.close(fig)
    # Four predeclared examples, no output-driven selection.
    rr=[r for r in records if r['arm']=='m00'][:4]
    for index,r in enumerate(rr):
        rgb=np.array(Image.open(r['rgb']).convert('RGB'));mask=np.array(Image.open(r['mask']).convert('L'))>0
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,method in zip(axes,['R2_tube','R2_bbox','R3']):
            ax.imshow(rgb)
            if method=='R2_tube':ax.contour(mask,levels=[.5],colors=['cyan'],linewidths=.7)
            if method=='R2_bbox':
                yy,xx=np.nonzero(mask)
                if len(xx):ax.plot([xx.min(),xx.max(),xx.max(),xx.min(),xx.min()],[yy.min(),yy.min(),yy.max(),yy.max(),yy.min()],color='cyan')
            data=r['public'][method]
            if data['selected_path'] is not None:
                p=np.array(data['selected_path']);ax.plot(p[:,0],p[:,1],color='lime',lw=1)
            ax.set_title(method+' observed trace');ax.axis('off')
        fig.suptitle(r['job']+'; green is a geometric candidate, not verified target identity',fontsize=10);fig.tight_layout();fig.savefig(OUT/'panels'/f'example_{index}.png',dpi=150);plt.close(fig)


def render():
    check();summary=checked_summary();frozen=read(OUT/'score_freeze.json');records=load('Eval');decisions=read(OUT/'decisions.json');projection=read(OUT/'metrics/baseline_projection.json');src=source_audit()
    figures(records,frozen,projection)
    table='| 接收器 | 接收/48 | 正确 | 错误 | 擦除 | 完整 2-bit 正确率 | 高置信错误 | 存在性 AUC | clean 判存在/128 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for method,s in summary.items():
        fpr=str(s['clean_presence_accept']) if s['clean_presence_accept'] is not None else '未校准'
        table+=f"| {LABELS[method]} | {s['received']} | {s['correct']} | {s['wrong']} | {s['erasures']} | {s['full_accuracy']:.1%} | {s['high_confidence_wrong']} | {s['presence_AUC']:.3f} | {fpr} |\n"
    theory='| 自然基线阶数 | 投影后 S 能量比例 | 投影后 T 能量比例 | S/T 联合秩 | 最小残余码字 RMS |\n|---|---:|---:|---:|---:|\n'
    for r in [r for r in projection if r['pattern']=='full']:
        m=r['metrics'];theory+=f"| {r['degree']} | {m['residual_energy_ratio'][0]:.3g} | {m['residual_energy_ratio'][1]:.3g} | {m['rank']} | {m['min_codeword_rms']:.3g} px |\n"
    caltable='| 方法 | Cal 选择自然基线阶数 | 残差门槛 px | 95% CI 门槛 px | LOCO 正确/48 | 错误/48 | 擦除/48 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for method in ['R1_public','R2_tube','R2_bbox','R3']:
        s=frozen[method]['selected'];caltable+=f"| {method} | {s['degree']} | {s['residual']} | {s['ci95']} | {s['correct']} | {s['wrong']} | {s['erasures']} |\n"
    content=[]
    for cid in sorted({r['content_id'] for r in decisions if r['arm'] in WORDS}):
        rr=[r for r in decisions if r['content_id']==cid and r['arm'] in WORDS];entry=dict(content_id=cid,methods={})
        for method in summary:
            received=[r for r in rr if r['methods'][method]['accepted']];correct=sum(r['methods'][method]['word']==r['arm'] for r in received)
            entry['methods'][method]=dict(attempts=4,correct=correct,wrong=len(received)-correct,erasures=4-len(received))
        content.append(entry)
    dump(OUT/'metrics/per_content.json',content)
    report=f"""# 阶段 4C-0：逐步取消原曲线参考

本轮冻结 S/T 编码器和已有 PNG，仅更换接收端条件。R0、R1、R2、R3 均完成固定参数重放。主要结论必须区分：去掉 clean 栅格参考系数，与去掉原曲线几何，是不同难度的步骤；前者不意味着实现盲解码。自参考方案目前只能作为失败诊断原型。

## 同一评估分母

{table}
全部 marked 分母为 48（12 个内容×4 消息），clean 分母为 128，其中 12 个与 marked 同内容同生成条件。Cal 仅使用旧4A六个内容×两视图×五状态，以及32个4B Cal clean，共92张。Cal 与本轮评估内容不重叠；但评估 PNG 已被前轮研究使用，本轮属于冻结参数后的复用留出诊断，不是全新内容独立验收。marked 的四个消息同内容相关，48次不能当作48个独立内容。

R0 保留原冻结存在性阈值。R1/R2/R3 的新存在性统计和阈值只在 Cal 设计；两个删除项对照未校准二元存在性阈值，故不报告其阈值误报数。全样本 AUC 将拒绝计为最低分，报告中不以 AUC 替代恢复率。“高置信”仅指模型后验≥0.95，不是经过可靠性校准的正确概率；几何拒绝样本也没有核验其轮廓身份，clean几何接收不等于水印误报。

R0 使用冻结4B的 EdgeDrop:none 未删除重匹配分支，与上一轮同PNG对比中的当前4B一致。旧4B主表使用 ObsDrop:none 的重复鲁棒拟合，可能产生小幅系数差异；因此本轮 clean 为9/128，旧表为10/128，不是阈值变化或改进。R0导出判决未携带后验字段，原始 summary.json 的高置信错误计数漏记；summary_checked.json 已从完整原始接收记录核对为1，并保存字段修正记录，不改变任何判决或分数。

## 实际输入码本是否泄露消息

原实现 b4.decode 会遍历四个候选码字，由 a4.features 读取每个候选的“实际栅格系数减 clean 系数”，随后联合比较四个中心。**它没有只取当前真实消息的输入系数，也未把真实消息作为候选选择条件。** 但整套四码字系数和 clean 系数依赖发送端原始 Canny，确实属于每载体参考开销，不能叫公共码本。

本轮 R0_public 用全局名义 S/T 幅度与原 Cal 信道替代每图四码字输入系数；R1_zero_clean 再去掉 clean 系数而不重新校准；R1_public 则在 Cal 上学习全局绝对输出系数中心。公共分类器不接收原 Canny、clean 系数、实际输入码本、源图合格标志或当前真实消息。训练标签只用于 Cal 学习与最终计分。每图实际输入与名义码本差距中位数为 {src['median_actual_vs_nominal_distance_px']:.3f} px，clean 参考系数范数中位数为 {src['median_clean_coefficient_norm_px']:.3f} px；详细逐内容数值见 source_information_audit.json。

**R1 仍然使用原曲线计算法向位移。** 所以即使 R1 性能接近 R0，也只能说明 clean 栅格系数和每图码本可省，不能说明原始几何可省。

## 自参考分解的数学条件

使用 p(t)=B(t)beta+S(t)aS+T(t)aT 的加权 Huber 拟合，先计算 Phi_perp=(I-P_B)Phi，再检查 Phi_perpᵀ W Phi_perp 的秩、最小特征值、条件数、两个基函数残余能量与六对码字距离。表为401个均匀站位、固定原基函数幅度的无噪声计算。

{theory}
S(t)=-6√3 t+18√3 t²-12√3 t³，T(t)=16t-96t²+160t³-80t⁴。因而三次自然基线精确包含 S，四次自然基线精确包含 S 与 T。两种情况分别联合秩1和0，必须擦除，不能用正则化或更强拟合器宣称分离出真实消息。0至2阶虽然满秩，分离仍依赖“自然轮廓没有对应高阶分量”的建模假设；自然曲线真实包含这些分量时，仍会与水印混淆。

![去除自然基线后的信息](plots/baseline_projection.png)

中央缺口与随机缺失的投影信息同时保存。其作用是改变有效信息而非增加消息冗余；本轮不将插值补点计为真实观测，也没有引入有限域 RS/RM。

## R2 与 R3 的具体接收规则

只从 RGB 提取100/200 Canny、骨架化、建立8邻接图，在端点与分叉点拆分真实像素轨迹。符合长度24～300px、弦长≥12px的轨迹按弧长选择最长者；不依据消息概率寻找“最像水印”的轨迹。由观测端点的字典序固定方向，用观测弧长重新参数化，以观测弦为局部坐标，自参考 Huber 联合拟合自然基线和 S/T。每次拟合只采实际边缘像素，无平滑插值与假桥接点。

R2_tube 使用原管道 Mask 约束像素范围；R2_bbox 只保留 Mask 的外接矩形；R3 不读取 Mask或任何每图源数据，在全图按同一公共规则选轨迹。所选轨迹并未经过正确载体身份核验，因此可能是背景轮廓。这是第一版简单基线，未成功不等于证明所有盲接收器都不可能。

管道 Mask 四码字共享同一文件，没有随消息变化，但它本身由原曲线扩张而成。源曲线到 Mask 骨架的距离中位数仅 {src['masks']['source_curve_to_mask_skeleton_median_px']:.3f} px，具有明显形状参考信息。即使不显式传原采样点，也不能把管道 Mask 辅助结果称为严格 RGB 盲检测。矩形仍提供位置/尺度，只有 R3 的运行输入达到 RGB＋Key＋公共规则。

## Cal 选择与失败边界

{caltable}
阶数0/1/2和有限门槛网格均在 Cal 选择，按留一内容交叉验证的 correct-2×wrong 评分。R2/R3 的最优 Cal 得分仍为负，说明目前错误接收较多，这些候选设置不具备可信部署条件，未被包装为合格接收器。之所以保留复用评估，是为了观察去参考后何处失效；没有根据 Eval 结果更换基线、曲线、门槛或码本。

![实际系数分布](plots/coefficient_distributions.png)

R0/R1 的坐标以原曲线为参考，R2/R3 以输出端弦为参考，数值不能直接相减解释为传输增益。图中的 raw fitted 包括置信度门槛拒绝样本，只用于诊断重叠。新坐标下端点变化、弧长变化、不同片段选择、自然曲线高阶形状都会与 S/T 系数混合。

## Key 与存在性

本轮没有更改编码器，Key 只重排原有四个合法码字的标签。wrong Key 的四中心集合完全相同，存在性分数和码字集合残差因此严格不变；它只能改变读出的标签，不能认证水印归属。这个限制有逐样本数值审计，不能把 wrong-Key 消息改变当作认证成功。

下一步若要兼顾2bit消息和Key认证，需要设计可观测的内生参考及额外合法性约束，并先验证投影后信息是否充分。可比较单条逻辑曲线不同片段间的相对系数、参考/消息分区、Key相关的受限合法集合或校验自由度。当前四个全部合法的码字没有认证冗余，本轮没有新增编码机制或容量宣称。应先研究低自由度自然基线下的可辨识统计与载体同步，再决定是否牺牲净载荷以加入认证。

## 文件与复现

- [逐图查看](index.html)、[核对汇总](summary_checked.json)、[全部判决](decisions.json)、[分内容结果](metrics/per_content.json)。
- [冻结协议](protocol.json)、[Cal 网格](cal_trials.json)、[冻结模型与阈值](score_freeze.json)。
- [基线投影信息](metrics/baseline_projection.json)、[发送端信息审计](metrics/source_information_audit.json)、[数值与输入边界审计](audit.json)。
- cal_features 与 eval_features 保存所有轨迹、各阶数拟合、真实像素坐标、信息矩阵与文件哈希。

执行顺序：`curve_reference_ablation_stage4c0.py freeze` → `cal` → `calibrate` → `eval`；`theory`保存解析投影诊断。使用既有 control Python 环境与单线程BLAS，不需要GPU。未修改任何父实验 PNG/源码/阈值，原48张 marked 的质量结果继续沿用先前48/48联合通过；本轮无新的生成质量收益。
"""
    (OUT/'REPORT_ZH.md').write_text(report)
    page=['<!doctype html><meta charset="utf-8"><title>4C-0 reference removal</title><style>body{font:15px sans-serif;max-width:1450px;margin:25px auto}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:6px}img{max-width:100%}.warn{background:#fff2ce;padding:14px}pre{white-space:pre-wrap}</style>',
      '<h1>4C-0：逐步取消原曲线参考</h1><p class="warn">R1仍使用原曲线几何。R2管道Mask透露原始形状；矩形Mask仍透露位置/尺度。R3只依赖RGB与公共规则，但当前为未达标原型。Key仅重排四合法标签，不提供认证。</p>',
      '<p><a href="REPORT_ZH.md">完整报告</a> · <a href="summary_checked.json">核对统计</a> · <a href="audit.json">审计</a> · <a href="protocol.json">冻结协议</a></p>',
      '<img src="plots/baseline_projection.png"><img src="plots/coefficient_distributions.png"><h2>同48张marked逐图判决，擦除计失败</h2><table><tr><th>PNG/真值</th>']
    for method in summary:page.append('<th>'+html.escape(LABELS[method])+'</th>')
    page.append('</tr>')
    recordmap={r['job']:r for r in records}
    for r in decisions:
        if r['arm']=='clean':continue
        link=os.path.relpath(recordmap[r['job']]['rgb'],OUT);page.append(f'<tr><td><a href="{html.escape(link)}">{html.escape(r["job"])}</a></td>')
        for method in summary:
            d=r['methods'][method];color='#d6f5dc' if d['accepted'] and d['word']==r['arm'] else '#ffd6d6' if d['accepted'] else '#eee'
            page.append(f'<td style="background:{color}">'+html.escape(d['word'] if d['accepted'] else '擦除')+'</td>')
        page.append('</tr>')
    page.append('</table><h2>固定前四个m00示例：绿色仅表示候选，未核验身份</h2>')
    for f in sorted((OUT/'panels').glob('*.png')):page.append(f'<img loading="lazy" src="panels/{f.name}">')
    page.append('<h2>Cal 选择（未按评估结果调参）</h2><pre>'+html.escape(json.dumps({m:frozen[m]['selected'] for m in ['R1_public','R2_tube','R2_bbox','R3']},indent=2))+'</pre>')
    (OUT/'index.html').write_text('\n'.join(page))


def finalize():
    names=['curve_reference_ablation_stage4c0.py','report_reference_ablation_stage4c0.py']
    for name in names:shutil.copyfile(ROOT/name,OUT/'source'/name)
    dump(OUT/'analysis_source_sha256.json',{name:digest(ROOT/name) for name in names})
    dump(OUT/'final_artifacts_sha256.json',{str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='final_artifacts_sha256.json'})


if __name__=='__main__':
    audit();render();print('REPORT_AUDIT_COMPLETE',OUT,flush=True);finalize()
