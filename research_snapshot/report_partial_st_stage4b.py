"""Plots, per-image viewer and Chinese interpretation of frozen Stage4B."""
from curve_partial_st_stage4b import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from html import escape

COLORS=['#888888','#2878b5','#f28e2b','#59a14f','#b445a2']
DISPLAY=['none','random25_r0','random50_r0','random75_r0','center25','center50','ends25','ends50','left50','blocks50']

def rel(p):return os.path.relpath(p,OUT)
def pc(v):return '—' if v is None else f'{100*v:.1f}%'
def counts(r):return f'{r["correct"]}/{r["attempts"]} ({pc(r["full_accuracy"])})；错 {r["wrong"]}，擦除 {r["erasures"]}'

def plots(rows,summary):
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    for track,color in [('ObsDrop','#2475b0'),('EdgeDrop','#d15b31')]:
        names=[s for s in DISPLAY if s in summary['Fresh'] and track in summary['Fresh'][s]];xx=[DISPLAY.index(s) for s in names]
        axes[0].plot(xx,[summary['Fresh'][s][track]['full_accuracy'] for s in names],'o-',color=color,label=track+' correct/all48')
        axes[1].plot(xx,[summary['Fresh'][s][track]['receive_rate'] for s in names],'o-',color=color,label=track+' received/all48')
        axes[1].plot(xx,[summary['Fresh'][s][track]['wrong']/48 for s in names],'x--',color=color,label=track+' wrong/all48')
    for ax in axes:ax.set_ylim(-.03,1.03);ax.grid(alpha=.25);ax.legend(fontsize=9)
    axes[0].set_ylabel('Complete 2-bit accuracy');axes[1].set_ylabel('Reception / wrong-word rate');axes[1].set_xticks(range(len(DISPLAY)),DISPLAY,rotation=30,ha='right')
    fig.suptitle('Fresh12 contents, 48 marked RGBs reused under each receiver-side loss');fig.tight_layout();fig.savefig(OUT/'plots/loss_recovery.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,split in zip(axes,['OldCal','Fresh']):
        for arm,color in zip(ARMS,COLORS):
            rr=[r for r in rows if r['split']==split and r['arm']==arm and r['track']=='ObsDrop' and r['scenario']=='none' and r['accepted']]
            if rr:
                v=np.array([r['feature'] for r in rr]);ax.scatter(v[:,0],v[:,1],s=20,alpha=.65,color=color,label=f'{arm} n={len(rr)}')
        ax.axhline(0,color='grey',lw=.5);ax.axvline(0,color='grey',lw=.5);ax.set_xlabel('Observed S - source clean S (px)');ax.set_ylabel('Observed T - source clean T (px)');ax.set_title(split+' accepted geometry only');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'plots/coefficient_distributions.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for arm,color in [('clean','#777777'),('marked','#216cb0')]:
        rr=[r for r in rows if r['split']=='Fresh' and (r['arm']=='clean' if arm=='clean' else r['arm'] in WORDS) and r['track']=='ObsDrop' and r['scenario']=='none' and r['accepted']]
        axes[0].hist([r['presence_score'] for r in rr],bins=20,alpha=.5,density=True,color=color,label=f'{arm} n={len(rr)}')
    axes[0].axvline(read(OUT/'score_freeze.json')['presence']['threshold95'],color='red',ls='--',label='Cal-frozen threshold');axes[0].set_xlabel('Presence log-likelihood ratio');axes[0].legend(fontsize=8)
    rr=[r for r in rows if r['split']=='Fresh' and r['track']=='ObsDrop' and r['scenario']=='none' and r['accepted']]
    if rr:
        bins=np.linspace(0,max(r['codeword_residual'] for r in rr)+1e-6,26)
        axes[1].hist([r['codeword_residual'] for r in rr],bins=bins,histtype='step',lw=3,label='True label key')
        axes[1].hist([r['wrong_key_residual'] for r in rr],bins=bins,histtype='step',ls='--',lw=2,label='Wrong XOR11 label key')
    axes[1].set_xlabel('Nearest-codeword Mahalanobis residual');axes[1].set_title('Same four-state set: exact overlap');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'plots/presence_and_wrong_key.png',dpi=150);plt.close(fig)
    comparison=read(OUT/'coverage_gate_comparison.json');fig,ax=plt.subplots(figsize=(11,4))
    for field,label in [('baseline_correct','Original coverage gate'),('partial_correct','Joint information gate')]:
        values=[]
        for s in DISPLAY:
            rr=[r for r in comparison if r['split']=='Fresh' and r['arm'] in WORDS and r['scenario']==s];values.append(np.mean([r[field] for r in rr]))
        ax.plot(range(len(DISPLAY)),values,'o-',label=label)
    ax.set_xticks(range(len(DISPLAY)),DISPLAY,rotation=30,ha='right');ax.set_ylabel('Correct complete messages / all48');ax.set_ylim(-.03,1.03);ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'plots/gate_comparison.png',dpi=150);plt.close(fig)
    theory=read(OUT/'geometry_information_theory.json');fig,axes=plt.subplots(1,2,figsize=(11,4))
    for kind in ['center','ends','left']:
        rr=[r for r in theory['records'] if r['kind']==kind];axes[0].plot([r['loss'] for r in rr],[r['joint_information_eigenvalues'][0] for r in rr],'o-',label=kind)
    axes[0].plot([0,.75],[1,.25],'k:',label='Random loss expectation');axes[0].axhline(read(OUT/'score_freeze.json')['gate']['info_min'],ls='--',color='red',label='Frozen info gate');axes[0].set_xlabel('Missing arc fraction');axes[0].set_ylabel('Minimum joint information eigenvalue');axes[0].legend(fontsize=8)
    rr=[r for r in theory['records'] if r['kind']=='center']
    for k,label in enumerate(['S uncentered energy','T uncentered energy']):axes[1].plot([r['loss'] for r in rr],[r['ST_uncentered_coverage'][k] for r in rr],'o-',label=label)
    axes[1].plot([r['loss'] for r in rr],[r['joint_information_eigenvalues'][0] for r in rr],'ko-',label='Joint, offset projected');axes[1].set_xlabel('Missing central arc fraction');axes[1].set_ylabel('Information relative to complete arc');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2);ax.set_ylim(-.03,1.03)
    fig.tight_layout();fig.savefig(OUT/'plots/geometry_information.png',dpi=150);plt.close(fig)

def panels(rows):
    f=read(OUT/'score_freeze.json');lookup={(r['job'],r['scenario'],r['track']):r for r in rows}
    plan={(j['key'],j['arm']):j for j in read(OUT/'plan.json') if j['split']=='Fresh'};cards=[]
    for c in [c for c in read(OUT/'selection.json') if c['marked']]:
        p,u,n,_=geometry(c['samples']);lo=np.maximum(0,np.floor(p.min(0)-15)).astype(int);hi=np.minimum(512,np.ceil(p.max(0)+16)).astype(int)
        if (hi-lo).max()<70:
            mid=(lo+hi)/2;lo=np.maximum(0,(mid-35).astype(int));hi=np.minimum(512,(mid+35).astype(int))
        for scenario_name in ['none','random50_r0','center25','center50']:
            path=OUT/'panels'/f'{c["key"]}_{scenario_name}.png'
            if not path.exists():
                fig,axes=plt.subplots(4,5,figsize=(17,11),squeeze=False)
                for col,arm in enumerate(ARMS):
                    j=plan[c['key'],arm];rgb=np.array(Image.open(j['rgb']).convert('RGB'));d=read(OUT/'loss'/f'{j["job"]}.json');raw=d['scenarios'].get(scenario_name,{}).get('ObsDrop',dict(fitted=False));obs=select(raw,f['gate']);decoded=lookup[j['job'],scenario_name,'ObsDrop']
                    axes[0,col].imshow(np.array(Image.open(j['hint']))[lo[1]:hi[1],lo[0]:hi[0]],cmap='gray',vmin=0,vmax=255);axes[0,col].set_title(arm+' actual input Canny')
                    axes[1,col].imshow(rgb[lo[1]:hi[1],lo[0]:hi[0]]);axes[1,col].set_title('Frozen ControlNet RGB')
                    edge=cv2.Canny(rgb,100,200);axes[2,col].imshow(edge,cmap='gray',vmin=0,vmax=255);axes[2,col].set_xlim(lo[0],hi[0]);axes[2,col].set_ylim(hi[1],lo[1]);axes[2,col].plot(p[:,0],p[:,1],color='#507dc0',alpha=.5,lw=1)
                    old=d['original']
                    if old.get('accepted'):
                        ii=np.array(old['indices']);xy=np.array(old['observed_xy']);scenario=next(s for s in SCENARIOS if s['name']==scenario_name);keep=mask_keep(c['samples'],j,scenario,ii)
                        axes[2,col].scatter(xy[~keep,0],xy[~keep,1],s=16,facecolors='none',edgecolors='#f28e2b');axes[2,col].scatter(xy[keep,0],xy[keep,1],s=10,color='#24c365')
                    if obs.get('fitted'):
                        ii=np.array(obs['indices']);y=np.array(obs['normal_positions']);axes[3,col].scatter(u[ii],y,s=12,color='#179755')
                        fitted=obs['offset']+basis(u)@np.array(obs['coefficient']);axes[3,col].plot(u,fitted,color='#8146b5',ls='--',lw=1.5)
                        axes[3,col].set_title(f'S={obs["coefficient"][0]:.2f} T={obs["coefficient"][1]:.2f}\ninfoMin={obs["info_min"]:.2f} cond={obs["condition"]:.1f}',fontsize=9)
                    else:axes[3,col].text(.1,.5,'No reliable observation',transform=axes[3,col].transAxes)
                    axes[2,col].set_title(f'received={decoded["accepted"]}, word={decoded["word"]}\nPmax={decoded["posterior_max"]:.2f}; presence={decoded["presence_accept"]}',fontsize=9)
                    axes[3,col].set_xlim(0,1);axes[3,col].set_xlabel('Original arc parameter t');axes[3,col].set_ylabel('Normal position (px)');axes[3,col].grid(alpha=.2)
                    for row in range(3):axes[row,col].set_axis_off()
                fig.suptitle(f'Fresh {c["key"]} | {scenario_name} | green=retained actual pixels, orange=withheld; purple dashed=fit prediction\nClean forced message is not watermark presence. Partial fit uses no withheld points.',fontsize=11)
                fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)
            cards.append(dict(key=c['key'],scenario=scenario_name,path=rel(path)))
        print('PANELS',c['key'],flush=True)
    return cards

def report():
    check();rows=read(OUT/'decoded.json');summary=read(OUT/'summary.json');f=read(OUT/'score_freeze.json');clean=read(OUT/'clean_independent.json');quality=read(OUT/'quality.json');wrong=read(OUT/'wrong_anchor.json');boot=read(OUT/'bootstrap.json')
    plots(rows,summary);cards=panels(rows);selected=read(OUT/'selection.json');fresh=summary['Fresh'];b=fresh['none']['ObsDrop'];e=fresh['none']['EdgeDrop'];input_stats=read(OUT/'input_partial_closure.json')['summary']['Fresh']
    table=['| 缺失规则 | 已定位后联合恢复：正确/全部，错误，擦除 | 删除边缘后重匹配 | 输入 Canny 闭环 |','|---|---|---|---|']
    for s in SCENARIOS:
        name=s['name'];r=fresh[name]['ObsDrop'];edge=fresh[name].get('EdgeDrop');inp=input_stats[name]
        table.append(f'| {name} | {counts(r)} | {counts(edge) if edge else "未运行"} | {inp["correct"]}/{inp["attempts"]}；错 {inp["wrong"]}，擦除 {inp["erasures"]} |')
    hc=['| 缺失规则 | 路线 | 高置信度错误 | 全部错误 | 擦除 |','|---|---|---:|---:|---:|']
    for name in DISPLAY:
        for track,r in fresh[name].items():hc.append(f'| {name} | {track} | {r["high_confidence_wrong"]}/{r["high_confidence_received"]} | {r["wrong"]}/{r["attempts"]} | {r["erasures"]}/{r["attempts"]} |')
    quality_records=quality['records'];fails=[r['job'] for r in quality_records if not r['pass_all']]
    fpr_table=['| 缺失规则 | 路线 | clean 被判有水印/128 | Marked 被判有水印/48 | 存在性 AUC |','|---|---|---:|---:|---:|']
    for name in DISPLAY:
        for track,r in fresh[name].items():fpr_table.append(f'| {name} | {track} | {r["clean_presence_accepted"]}/128 | {r["marked_presence_accepted"]}/48 | {r["presence_auc"]:.3f} |')
    transitions=read(OUT/'full_to_partial_transitions.json')['records'];transition_table=['| 缺失规则 | 无缺失时判对的25个中仍判对 | 新错消息 | 新擦除 |','|---|---:|---:|---:|']
    for name in DISPLAY:
        r=next(r for r in transitions if r['scenario']==name and r['track']=='ObsDrop')
        transition_table.append(f'| {name} | {r["retained_correct"]}/{r["full_correct"]} | {r["newly_wrong_from_full_correct"]} | {r["erased_from_full_correct"]} |')
    pregate=read(OUT/'pre_gate_diagnosis.json');pretable=['| 缺失规则 | 数值拟合成功 | 拒绝前候选判对 | 被门槛拒绝的正确/错误候选 |','|---|---:|---:|---:|']
    for name in DISPLAY:
        r=next(r for r in pregate['summary'] if r['split']=='Fresh' and r['scenario']==name)
        pretable.append(f'| {name} | {r["fitted"]}/48 | {r["pre_gate_correct"]}/48 | {r["rejected_correct_hypotheses"]} / {r["rejected_wrong_hypotheses"]} |')
    report_text=f'''# 阶段 4B：不完整单曲线的 S/T 联合恢复

固定 4A 的 2 bit、四个码字、S/T 基函数、组合最大连续位移 1.5 px、像素编辑规则和 ControlNet。没有扩容、重新设计峰谷或使用 RS/RM。所有缺失仅作用于接收端观测；同一条件下复用完全相同的 RGB。

## 独立结果

全新 12 个内容、每内容一个新 prompt/seed 组合、四个消息，共 48 张 marked RGB。无人工缺失时，已定位路线完整消息恢复 {counts(b)}；删边重匹配路线 {counts(e)}。不同路线的接收条件不同，不能将已定位路线当作盲定位成绩。

**本轮结论：部分支持“分散观测可联合恢复系数”，尚未通过可靠端到端恢复或存在性验收。** 随机删去 25% 的三个重复均恢复 24/48；但无缺失本身仅恢复 25/48。中央 25% 缺口被冻结信息门槛全部擦除；随机 50% 删边后重匹配也全部擦除。不能把已定位后残余拟合的收益解释为自动接收问题已经解决，也不能把本轮结果写成单曲线纠错容量已得到证明。

{chr(10).join(table)}

random 为固定站点随机缺失；相同内容/view 的四种消息与 clean 使用相同掩码。r0/r1/r2 为三个掩码重复，复用同一张 RGB，不能当作新独立生成。center 删除中央相应弧长比例；ends 删除两端合计相应比例；left50 删除前半；blocks50 删除交替的四段 1/8 弧长。标称比例指原曲线参数区域，不保证刚好删掉等比例输出像素。实际保留样本数、加权信息和图像哈希均已保存。

下表只跟踪无缺失时已经判对的子集，帮助区分新增损失；它是条件分析，不能替代上表以全部 48 个为分母的恢复率。

{chr(10).join(transition_table)}

![恢复率](plots/loss_recovery.png)

## 两条路线与实际观测

- **ObsDrop：先固定正确性未知但已通过原 Auto 的对应，再删观测。** 只使用残留真实 Canny 像素同时拟合 offset、S、T。原定位失败继续记擦除，未用真消息挑选片段。这是控制同步后的恢复实验。
- **EdgeDrop：先在 RGB Canny 中删掉指定弧段附近 8 px 管内的边缘像素，重建图，再运行原 R0/A3 候选搜索。** RGB 切线保留原估计；不把样条插值、跨缺口连线当成观测。搜索管宽、方向和几何代价未放宽。用联合信息门槛替代原固定覆盖/端点门槛；这是新接收规则，旧 4A 仍作为冻结基线。

图中绿色是实际保留且进入拟合的像素，橙色空心点是人工扣留的旧观测，紫色虚线是拟合预测。紫色虚线不意味着模型输出在缺口内产生了真实边缘。逐图查看见 [index.html](index.html)。

具体恢复模型是原曲线法向形变：\(q(t)=p_0(t)+n_0(t)[o+c_S S(t)+c_T T(t)]\)。联合估计三个实参数 \((o,c_S,c_T)\)，并按冻结四码字分类。这里的低阶多项式约束属于**法向形变**；原始 COCO 轮廓 \(p_0\) 本身不必是多项式。需要原始曲线、方向、参数化与局部搜索管，不是未知 RGB 的全图盲恢复。

## Cal 冻结与联合可辨识性

使用旧 6 Cal 内容的既有 RGB 做缺失诊断，增加 32 个独立 Cal clean 内容，未用旧 24 clean 反复选阈值。保留原通道响应矩阵 A、b 和四码字中心；仅在 Cal 选择恢复信息门槛及新的存在性统计量，然后才生成独立内容。冻结时间 {f['created_utc']}。

原始 27 组 Cal 网格全部未达到错锚点 ≤10% 的要求，最少仍接受 10/64；未据此启动独立生成。随后记录 [Cal 修订](calibration_amendment.json)，仅在 Cal 扩展更严格的信息量、条件数和不确定度门槛，保留原失败网格。独立 RGB 仍在最终冻结后才生成。

选定门槛：最小联合信息特征值 ≥{f['gate']['info_min']}，设计矩阵条件数 ≤{f['gate']['condition_max']}，95% 系数最大半宽 ≤{f['gate']['ci95_max']} px。残差 ≤0.6 px、offset ≤1.5 px 等原几何约束保留。Cal 网格目标是 correct−2×wrong，同时要求错锚点接受率 ≤10%，不是按照预期 bit 筛选。

联合信息矩阵在消去 offset 后计算，并相对于完整固定站点的 S/T Gram 归一化。S 与 T 在完整连续区间正交，不保证在残缺区间正交。点数或 S 覆盖率高，都不保证 T 可恢复；联合信息低、条件数高或不确定度过大时擦除。Cal 仍只有 6 个 marked 内容，参数选择有较大统计不确定性。

![缺失位置与可辨识性](plots/geometry_information.png)

上图为连续、均匀弧长、独立同方差观测的几何信息分析，不是扩散性能预测。中央缺失可能仍留下大量 S 能量，但 T 与 offset 的区分能力下降；单侧缺失还会引入 S/T 相关。实际冻结的信息量门槛可能直接拒绝仍具有满秩拟合的残余观测，因此“全部擦除”不能自动解释为几何信号已经消失。

![固定覆盖门槛与联合信息门槛](plots/gate_comparison.png)

输入闭环使用已保存的真实二值条件图，在局部真实像素上做同样删点与联合拟合，以名义四符号码本解码，不是用自身实测中心证明自己。连续无噪声代数对照另见 [algebraic_positive_control.json](algebraic_positive_control.json)，它只检查实现，不作为扩散成绩。

## Clean、四码字与 Key

新的 clean 分布模型仅由旧 Cal 和新 32 Cal 内容拟合。新的 clean-vs-marked 统计量比较 clean Gaussian 与均匀四码字混合分布，并计入拟合协方差；阈值取 Cal 已接收缺失观测的 95 分位数，**不具有严格 5% FPR 保证**。同一内容的多次缺失观测具有相关性。

全新 128 内容、每内容一张 clean PNG 的无人工缺失验证：{clean['false_accepts']}/128 被判有水印，比例 {pc(clean['false_accept_rate'])}，二项精确 95% 区间 {pc(clean['exact_binomial95'][0])}–{pc(clean['exact_binomial95'][1])}。每图只检验一条选定曲线；不是 family-wise 多曲线误报率。不能据此声称达到 1% FPR。

{chr(10).join(fpr_table)}

![系数分布](plots/coefficient_distributions.png)

散点仅显示接收成功的几何特征；拒绝样本的原始候选、坐标、权重和系数仍保留于 loss JSON，未从分母移除。存在性 AUC 使用全部尝试，擦除拒绝；clean 128 与 marked 48 的 AUC 涉及不平衡内容集合，另保存原始逐图分数供配对子集检查。

![存在性及错误 Key](plots/presence_and_wrong_key.png)

**当前码本没有可验证的密钥认证冗余。** 将四个全部合法的码字按 XOR11 置换标签，真/错 Key 的无序码字集合完全相同，最近码字残差及混合似然完全相同。这个错误 Key 检验不能拒绝无水印或错误归属；不能虚构 Key 分离效果。要获得认证能力，需额外约束或牺牲容量，本轮未增加。

另设同图错锚点：原曲线分别平移 (32,0)、(0,32)，用相同接收器搜索。Cal 接受 {wrong['summary']['Cal']['accepted']}/{wrong['summary']['Cal']['attempts']}；独立组接受 {wrong['summary']['Fresh']['accepted']}/{wrong['summary']['Fresh']['attempts']}，原覆盖门槛独立组接受 {wrong['summary']['Fresh']['baseline_accepted']}/{wrong['summary']['Fresh']['attempts']}。它们是几何错对应诊断，不是 clean 水印误报率；部分平移区域可能与真实背景交叠。

## 高置信度错误与擦除

{chr(10).join(hc)}

高置信度指四类相对后验最大值 ≥0.95，**不是已校准的正确概率，也不是存在性证据**。四类全都不像观测时，仍可能出现某一类相对后验很高。不能因四分类自信就取消擦除或宣称有水印。

为区分“系数无法估计”和“系数可算出但被置信度拒绝”，另外列出门槛前诊断：

{chr(10).join(pretable)}

这张表把所有数值拟合成功的候选送入同一冻结四分类器，仅用于解释擦除；**不是另一套已验证接收器，不能将其中判对的候选直接改判为有效消息**。它没有提供可靠对应保证，也未通过错锚点拒绝约束。

## 质量、样本量与结论边界

新 48 张 marked 中，PSNR/SSIM/LPIPS/Edge-F1 四项联合通过 {sum(r['pass_all'] for r in quality_records)}/{len(quality_records)}。门槛沿用 PSNR≥30 dB、SSIM≥0.90、LPIPS≤0.10、Edge-F1≥0.95；不通过：{', '.join(fails) if fails else '无'}。接收端缺失操作不改变任何 RGB，故不能算作新的生成质量实验。

恢复率的按内容 bootstrap 见 [bootstrap.json](bootstrap.json)，12 个 marked 内容的区间较宽；所有人工缺失重复不增加独立内容数。新 clean 128 内容全部保留，没有按生成结果替换。可核验历史路径和元数据已排除，但早期缺失目录导致无法证明完整历史无重叠。

这轮验证的是**实数多项式几何在受控缺失下的可恢复性**：足够分散且可靠的残余观测，使低维系数仍可辨识；缺失位置导致秩退化、关联错误或输出噪声过大时，仍须擦除。只有先排除错误对应，才能把实验收益解释为几何冗余。它尚不是有限域 RS 纠错，也不能证明任意缺失比例、任意内容或盲定位下都能恢复。

原始材料：[协议](protocol.json)、[Cal 冻结](score_freeze.json)、[全部统计](summary.json)、[逐次解码](decoded.json)、[新增接收](newly_received.json)、[输入闭环](input_partial_closure.json)、[审计](audit.json)。旧 4A 文件保持冻结。
'''
    (OUT/'REPORT_ZH.md').write_text(report_text)
    clean_cards=[]
    byjob={r['job']:r for r in rows if r['split']=='Fresh' and r['scenario']=='none' and r['track']=='ObsDrop'}
    for j in read(OUT/'plan.json'):
        if j['split']!='Fresh' or j['arm']!='clean':continue
        r=byjob[j['job']];score=f'{r["presence_score"]:.3f}' if r['accepted'] else '擦除'
        clean_cards.append(f'<tr><td>{j["content_id"]:012d}</td><td>{r["accepted"]}</td><td>{score}</td><td>{r["presence_accept"]}</td><td><a href="{rel(j["rgb"])}">RGB</a> · <a href="loss/{j["job"]}.json">原始观测</a></td></tr>')
    articles=''.join(f'<article data-loss="{c["scenario"]}"><h3>{c["key"]} / {c["scenario"]}</h3><a href="{c["path"]}"><img loading="lazy" src="{c["path"]}"></a></article>' for c in cards)
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>阶段4B：S/T残余观测联合恢复</title><style>body{font:16px/1.6 system-ui;background:#eee;margin:24px}section,article{padding:18px;background:white;margin:18px 0}img{max-width:100%}a{color:#1765a1}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}select{font-size:17px}</style><h1>4B：不完整单曲线的2-bit恢复</h1><p><a href="REPORT_ZH.md">完整中文报告</a> · <a href="summary.json">全部统计</a> · <a href="audit.json">审计</a></p><section><p>五列依次 clean、00、01、10、11。四行依次实际输入Canny、生成RGB、输出Canny与观测、联合系数拟合。绿色=保留的真实像素；橙色空心=扣留的观测；紫色虚线=预测，不是缺口内真实边缘。clean判出某个码字不等于存在水印。</p><p>ObsDrop 已固定原来的对应；EdgeDrop 另在统计中报告。不能把这两条路线混为盲检测。</p><img src="plots/loss_recovery.png"><img src="plots/gate_comparison.png"><img src="plots/coefficient_distributions.png"><img src="plots/presence_and_wrong_key.png"></section><label>缺失规则 <select id="loss"><option value="center25">中央25%</option><option value="random50_r0">随机50%</option><option value="center50">中央50%</option><option value="none">无缺失</option><option value="all">全部</option></select></label>'''+articles+'''<section><h2>128个独立 clean 内容：无人工缺失</h2><p>每图一条曲线；显示存在性分数，几何接收和水印存在分开。所有内容均保留。</p><table><tr><th>内容</th><th>几何接收</th><th>存在性分数</th><th>判断有水印</th><th>文件</th></tr>'''+''.join(clean_cards)+'''</table></section><script>function show(){let v=document.getElementById('loss').value;document.querySelectorAll('article').forEach(a=>a.hidden=v!=='all'&&a.dataset.loss!==v)}document.getElementById('loss').onchange=show;show()</script></html>'''
    (OUT/'index.html').write_text(html);dump(OUT/'viewer_data.json',dict(cards=cards,clean_rows=128))
    print('REPORT_COMPLETE',flush=True)

if __name__=='__main__':report()
