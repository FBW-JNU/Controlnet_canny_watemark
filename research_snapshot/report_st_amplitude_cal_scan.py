"""Report and audit the completed Cal strength scan without tuning receivers."""
from analyze_st_amplitude_cal_scan import *
from scipy.spatial import cKDTree
import html,ast,re

LABELS={'SourceRef':'原曲线参考辅助','R2_tube':'管道 Mask 自参考','R2_bbox':'矩形 Mask 自参考'}


def bootstrap():
    rows=read(OUT/'metrics/per_content.json');ids=sorted({r['content_id'] for r in rows});rng=np.random.default_rng(2026111175);indices=rng.integers(0,6,size=(10000,6));result=[]
    for method in METHODS:
        base=np.array([next(r['correct']/8 for r in rows if r['content_id']==cid and r['method']==method and r['peak']==1.5) for cid in ids])
        for peak in PEAKS[1:]:
            value=np.array([next(r['correct']/8 for r in rows if r['content_id']==cid and r['method']==method and r['peak']==peak) for cid in ids]);delta=value-base;dist=delta[indices].mean(1)
            result.append(dict(method=method,peak=peak,accuracy_gain=float(delta.mean()),paired_content_bootstrap95=np.quantile(dist,[.025,.975]).tolist(),
                contents_improved=int(np.sum(delta>0)),contents_worse=int(np.sum(delta<0)),contents_equal=int(np.sum(delta==0))))
    dump(OUT/'metrics/paired_gain_bootstrap.json',dict(records=result,n_bootstrap=10000,clusters=6,
        caveat='Resamples six contents with both views/all messages kept together. Fixed decoded outputs/models, no LOCO refitting inside bootstrap; exploratory interval, not independent validation.'))
    return result


def audit():
    analysis_freeze();check();counts=collections.Counter();plan=read(OUT/'plan.json');cc={c['key']:c for c in read(OUT/'selection.json')}
    assert len(plan)==204 and sum(not j['reused'] for j in plan)==144 and len(cc)==6
    for p,h in read(OUT/'input_freeze.json')['protected'].items():assert digest(p)==h;counts['protected_input_hashes']+=1
    for checkrow in read(OUT/'canary.json')['checks']:assert checkrow['pixels_identical'] and checkrow['maximum_difference']==0
    for j in plan:
        r=read(OUT/'records'/f'{j["job"]}.json');f=read(OUT/'features'/f'{j["job"]}.json')
        assert digest(j['rgb'])==r['rgb_sha256']==f['rgb_sha256'];assert digest(j['hint'])==j['hint_sha256'];assert digest(j['noise'])==r['noise_sha256'];counts['RGB_records_checked']+=1
        original=read(PARENT/'records'/f'{j["original_job"]}.json')
        assert j['seed']==original['seed'] and j['prompt']==original['prompt'] and r['noise_sha256']==original['noise_sha256']
        assert j['clean_rgb']==next(k['rgb'] for k in plan if k['key']==j['key'] and k['view']==j['view'] and k['arm']=='clean')
        if j['reused']:assert r['rgb_sha256']==original['rgb_sha256'];counts['verified_reused_RGB']+=1
        if j['arm']!='clean':
            q=read(OUT/'quality'/f'{j["job"]}.json');assert q['rgb_sha256']==r['rgb_sha256'] and q['clean_rgb_sha256']==digest(j['clean_rgb']);assert q['quality']['changed_pixels']>0;counts['quality_pairs_checked']+=1
        edge=cv2.Canny(np.array(Image.open(j['rgb']).convert('RGB')),100,200)
        for method in ['R2_tube','R2_bbox']:
            selected=f['public'][method]['selected_path']
            if selected is None:continue
            xy=np.array(selected);assert np.array_equal(xy,np.round(xy));assert np.all(edge[xy[:,1].astype(int),xy[:,0].astype(int)]>0);counts['actual_candidate_pixels_checked']+=len(xy)
    # All dose conditions reproduce from source, fixed ROI and fixed normals.
    for r in read(OUT/'input_closure.json'):
        c=cc[r['key']];edge=np.array(Image.open(c['states']['clean']['hint']).convert('L'));mask=np.array(Image.open(OUT/'edit_masks'/f'{c["key"]}.png').convert('L'))>0
        warp=RasterWarp(edge,c['samples'],mask);v=basis(warp.u)@((r['peak']/b4.a4.PEAK)*np.array(b4.a4.SIGNS[r['word']]))
        warp.vector=v[warp.near[warp.chosen],None]*warp.n[warp.near[warp.chosen]];rendered,moved=warp.render(1.)
        assert np.array_equal(rendered,np.array(Image.open(r['hint']).convert('L')));assert r['input_correct']
        if r['peak']==1.5:assert np.array_equal(rendered,np.array(Image.open(c['states'][r['word']]['hint']).convert('L')))
        counts['reproduced_Canny_states']+=1
    folds=read(OUT/'loco_folds.json');neg=read(OUT/'threshold_clean32.json');assert len(neg)==32 and not {c['content_id'] for c in cc.values()}&{r['content_id'] for r in neg}
    records=[read(OUT/'features'/f'{j["job"]}.json') for j in plan];recordmap={r['job']:r for r in records};foldmap={}
    for f in folds:
        assert f['heldout_content'] not in f['train_contents'] and len(f['train_contents'])==5
        assert len(f['calibration_scores'])==32
        for alpha in [.05,.10]:assert f['thresholds'][str(alpha)]==order_threshold(f['calibration_scores'],alpha)
        train=[r for r in records if r['content_id'] in f['train_contents'] and (r['peak']==f['peak'] or r['arm']=='clean')]
        fitted=[dict(content_id=r['content_id'],arm=r['arm'],value=r['methods'][f['method']]['value']) for r in train if r['methods'][f['method']]['accepted']]
        assert c0.model_fit(fitted)==f['model']
        expected=[score(n['methods'][f['method']],f['model'])['score'] for n in neg]
        assert expected==f['calibration_scores'];foldmap[f['peak'],f['method'],f['heldout_content']]=f
        counts['LOCO_folds_checked']+=1
    decisions=read(OUT/'decisions.json');assert len(decisions)==720
    for d in decisions:
        f=foldmap[d['peak'],d['method'],d['content_id']];record=recordmap[d['job']]
        assert score(record['methods'][d['method']],f['model'])==d['decoded'];counts['heldout_decisions_replayed']+=1
    summary=read(OUT/'summary.json')
    for peak in PEAKS:
        for method in METHODS:
            assert summary[tag(peak)][method]==summarize(decisions,method,peak);counts['summary_cells_checked']+=1
    for folder in [PARENT,b4.OUT,C0]:
        for rel,h in read(folder/'final_artifacts_sha256.json').items():assert digest(folder/rel)==h;counts['unchanged_parent_manifest_entries']+=1
    dump(OUT/'audit.json',dict(status='passed',counts=dict(counts),same_prompt_seed_noise=True,matching_rules_unchanged=True,all_marked_denominator=48,
       heldout_clean_denominator=12,separate_threshold_clean=32,no_Eval_or_Fresh_RGB_in_scan=True,
       note='Cal exploration only. Same-content four words and two views are correlated. Source-known input mapping is available only before diffusion.'))


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=read(OUT/'summary.json');quality=read(OUT/'quality_summary.json')
    fig,axes=plt.subplots(2,3,figsize=(15,8));colors=['#1565c0','#e65100','#2e7d32']
    for method,color in zip(METHODS,colors):
        rows=[summary[tag(p)][method] for p in PEAKS]
        for ax,key in [(axes[0,0],'full_accuracy'),(axes[0,1],'error_rate'),(axes[0,2],'erasure_rate'),(axes[1,0],'presence_AUC')]:ax.plot(PEAKS,[r[key] for r in rows],'o-',label=method,color=color)
        for ax,alpha in [(axes[1,1],'.05'),(axes[1,2],'.1')]:
            real=str(float(alpha));ax.plot(PEAKS,[r['fixed_FPR'][real]['TPR'] for r in rows],'o-',label=method,color=color)
    for ax,title in zip(axes.flat,['Full2bit accuracy (all48)','Wrong words /48','Erasures /48','Clean vs marked pooled AUC','TPR at nominal5% FPR','TPR at nominal10% FPR']):
        ax.set_title(title);ax.set_xticks(PEAKS);ax.set_xlabel('Maximum continuous displacement (px)');ax.set_ylim(-.03,1.03);ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Cal six-content LOCO scoring; geometry rules fixed; thresholds from separate32 Cal clean')
    fig.tight_layout();fig.savefig(OUT/'plots/recovery_and_presence.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(16,4));g=read(OUT/'protocol.json')['quality']
    for ax,key,title,threshold in zip(axes,['psnr_db','ssim','lpips_alex','edge_f1_3x3'],['PSNR (dB)','SSIM','LPIPS Alex','Paired RGB Canny Edge-F1'],[g['psnr_db_min'],g['ssim_min'],g['lpips_alex_max'],g['rgb_pair_edge_f1_3x3_min']]):
        ax.plot(PEAKS,[quality[tag(p)]['medians'][key] for p in PEAKS],'o-',label='median');ax.plot(PEAKS,[quality[tag(p)]['worst'][key] for p in PEAKS],'s--',label='worst');ax.axhline(threshold,color='red',ls=':',label='frozen limit');ax.set_xticks(PEAKS);ax.set_title(title);ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'plots/quality.png',dpi=160);plt.close(fig)
    plan=read(OUT/'plan.json');cc={c['key']:c for c in read(OUT/'selection.json')};lookup={(j['key'],j['view'],j['arm'],j['peak']):j for j in plan};panels=[]
    for c in cc.values():
        p=np.array(c['samples']);lo=np.maximum(0,np.floor(p.min(0)-15)).astype(int);hi=np.minimum(512,np.ceil(p.max(0)+16)).astype(int)
        for view in [0,1]:
            clean=lookup[c['key'],view,'clean',0.]
            for word in WORDS:
                fig,axes=plt.subplots(3,5,figsize=(14,7));jobs=[clean]+[lookup[c['key'],view,word,peak] for peak in PEAKS]
                for col,j in enumerate(jobs):
                    hint=np.array(Image.open(j['hint']).convert('L'));rgb=np.array(Image.open(j['rgb']).convert('RGB'));edge=cv2.Canny(rgb,100,200)
                    for row,image in enumerate([hint,rgb,edge]):
                        axes[row,col].imshow(image,cmap='gray',vmin=0,vmax=255);axes[row,col].set_xlim(lo[0],hi[0]);axes[row,col].set_ylim(hi[1],lo[1]);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
                    axes[0,col].set_title('Clean' if col==0 else str(j['peak'])+'px')
                for ax,label in zip(axes[:,0],['Actual input Canny','Generated RGB','RGB extracted Canny']):ax.set_ylabel(label)
                name=f'{c["key"]}_v{view}_{word}';fig.suptitle(name+' — same prompt, seed and noise; fixed crop, no synthetic edge overlay',fontsize=10)
                fig.tight_layout();fig.savefig(OUT/'panels'/f'{name}.png',dpi=130);plt.close(fig);panels.append(dict(name=name,key=c['key'],view=view,word=word,path=f'panels/{name}.png'))
    dump(OUT/'panels.json',panels)


def render():
    check();summary=read(OUT/'summary.json');quality=read(OUT/'quality_summary.json');inp=read(OUT/'input_receiver_summary.json');locked=read(OUT/'locked_channel_control.json');gains=bootstrap();figures()
    plan=read(OUT/'plan.json');generation=read(OUT/'protocol.json')['generation']
    dump(OUT/'effective_generation_config.json',dict(sampler={k:generation[k] for k in ['model_config','checkpoint','checkpoint_sha256','steps','eta','guidance','control_scale']},
        actual_seeds_by_view={str(v):sorted({j['seed'] for j in plan if j['view']==v}) for v in [0,1]},
        CUDA_VISIBLE_DEVICES='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',precision='float32, TF32 disabled',new_unique_RGB=144,reused_unique_RGB=60,canary_duplicates=2,
        note='Actual prompts/seeds come from frozen plan.json and original Cal records, rather than historical count/seed fields in inherited generation metadata.'))
    table='| 最大位移 | 接收器 | 正确/48 | 错误/48 | 擦除/48 | 全样本完整2bit正确率 | 存在性AUC | 5%阈值TPR | 对应clean误报/12 | 10%阈值TPR | 对应clean误报/12 |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for peak in PEAKS:
        for method in METHODS:
            s=summary[tag(peak)][method];a=s['fixed_FPR']['0.05'];b=s['fixed_FPR']['0.1']
            table+=f"| {peak:g}px | {LABELS[method]} | {s['correct']} | {s['wrong']} | {s['erasures']} | {s['full_accuracy']:.1%} | {s['presence_AUC']:.3f} | {a['TPR']:.1%} | {a['heldout_clean_false_accepts']} | {b['TPR']:.1%} | {b['heldout_clean_false_accepts']} |\n"
    qt='| 最大位移 | PSNR中位数/最差 dB | SSIM中位数/最差 | LPIPS中位数/最差 | Edge-F1中位数/最差 | 四项联合通过/48 |\n|---|---:|---:|---:|---:|---:|\n'
    for peak in PEAKS:
        q=quality[tag(peak)];m=q['medians'];w=q['worst'];qt+=f"| {peak:g}px | {m['psnr_db']:.2f}/{w['psnr_db']:.2f} | {m['ssim']:.4f}/{w['ssim']:.4f} | {m['lpips_alex']:.4f}/{w['lpips_alex']:.4f} | {m['edge_f1_3x3']:.4f}/{w['edge_f1_3x3']:.4f} | {q['joint_passes']} |\n"
    input_rows=read(OUT/'input_closure.json');it='| 最大位移 | 系数幅度a | 源已知对应闭环/24 | 冻结输入Auto正确/24 | Auto错误/擦除 | 最大整数移动（全部样本） | 选中像素移动比例中位数 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for peak in PEAKS:
        rr=[r for r in input_rows if r['peak']==peak];auto=inp[tag(peak)]['Auto'];it+=f"| {peak:g}px | {peak/b4.a4.PEAK:.3f}px | {sum(r['input_correct'] for r in rr)} | {auto['correct']} | {auto['received']-auto['correct']}/{24-auto['received']} | {max(r['max_integer_shift_px'] for r in rr):.3f}px | {np.median([r['shifted_fraction'] for r in rr]):.1%} |\n"
    lt='| 最大位移 | 旧全局信道正确/48 | 错误/48 | 擦除/48 |\n|---|---:|---:|---:|\n'
    for r in locked:lt+=f"| {r['peak']:g}px | {r['correct']} | {r['wrong']} | {r['erasures']} |\n"
    gt='| 接收器 | 强度相对1.5px | 正确率变化 | 按内容配对Bootstrap95%区间 | 改善/变差/不变内容数 |\n|---|---:|---:|---:|---:|\n'
    for r in gains:
        ci=r['paired_content_bootstrap95'];gt+=f"| {LABELS[r['method']]} | {r['peak']:g}px | {r['accuracy_gain']:+.1%} | [{ci[0]:+.1%}, {ci[1]:+.1%}] | {r['contents_improved']}/{r['contents_worse']}/{r['contents_equal']} |\n"
    best={method:max(PEAKS,key=lambda p:summary[tag(p)][method]['full_accuracy']) for method in METHODS}
    findings=[]
    for method in METHODS:
        peak=best[method];s=summary[tag(peak)][method];base=summary[tag(1.5)][method]
        findings.append(f"{LABELS[method]}本批完整消息正确率最高为{peak:g}px的{s['correct']}/48（{s['full_accuracy']:.1%}），相对1.5px变化{s['full_accuracy']-base['full_accuracy']:+.1%}；该强度质量联合通过{quality[tag(peak)]['joint_passes']}/48。")
    report=f"""# Cal S/T最大位移扫描：1.5 / 2 / 3 / 4 px

{' '.join(findings)} 这些是六个 Cal 内容上的研发诊断，不能据此宣布独立验收通过或自动更换冻结幅度。

## 实验一致性与分母

固定原六个Cal内容、原曲线、四个S/T正负码字、两组prompt/seed及每组初始噪声。每强度48张marked，12张配对clean跨强度共用。实际新增144张RGB，复用48张1.5px和12张clean；另外32张旧4B Cal clean仅用于存在性阈值校准，不混入48张消息分母。

生成器仍为冻结SD1.5+ControlNet Canny，20步、CFG5、control1、eta0、float32、TF32关闭、CuBLAS确定性设置。当前环境先重生成1张clean与1张1.5px，两张PNG像素及哈希都与旧图完全相同，才开始144张新生成。

全部强度使用同一编辑Mask：原Mask扩张3px并限制在原ROI，以避免大位移撞到旧4px管道边界。所有1.5px条件图已逐像素证明与旧条件图一致。接收器的管道/矩形Mask仍为原Mask，不随强度扩张。完整2bit消息错误和擦除均计为失败，全部48次均保留；输入不闭环、质量失败或定位失败均不删除或替换内容。

## 消息恢复与存在性

{table}
参考辅助使用冻结4B源曲线同步和几何门槛，特征为观测S/T减源clean Canny系数。自参考两分支使用冻结4C0的轨迹提取、方向、自然基线阶数与残差/置信区间门槛（tube阶数0，bbox阶数1）。公共分类器都不接收真实消息，也不读取当前消息的发送端系数。

为让不同强度的码字中心具有公平的尺度，每个强度都按**同一训练规则**拟合评分模型，采用六折留一内容：五内容的40marked+10clean训练，另一个内容的8marked+2clean计分；整内容的两视图和四消息一起排除。模型或几何缺失仍为擦除。注意自然基线阶数/匹配规则过去已在这些Cal上开发；留一内容只隔离本轮评分模型，不消除历史开发复用，不是完全独立测试。

存在性统计为clean密度与四marked中心混合的对数似然比。32张额外Cal clean只用于每折阈值，不参与该折密度拟合。名义5%使用32个分数的最大值，名义10%使用第30顺序统计量，严格大于阈值才判存在。表内同时报告被留出的12张clean的实际误报次数。这里没有足够数据证明精确5%/10%长期FPR；尤其0/12不能解释为零误报保证，更不支持1%认证。失败分数计-1e6并保留在AUC和TPR分母。

![恢复与存在性](plots/recovery_and_presence.png)

## 冻结旧信道模型对照

{lt}
此对照保持旧全局Cal信道A、截距与协方差，只按新强度缩放公开名义码本；匹配器与主表完全相同。不重新拟合新强度分布。它也复用了原模型训练过的Cal，属于描述性对照。主表与此表的差异提示评分中心/噪声模型是否适配，不能把评分模型改善全部归为几何信道改善。

## 输入端：真实修改与接收规则需要区分

{it}
24是每强度六内容×四消息的独立条件图数；生成RGB的两视图复用这些条件，所以消息评估分母为48。源已知对应闭环直接使用栅格编辑时记录的原像素→实际移动后像素，拟合真实观测并扣除clean，四强度均24/24正确。这不是扩散输出Oracle，不能转移成输出对应真值。

冻结输入Auto在2/3/4px已经出现错误或擦除，虽然发送端已知对应仍能正确区分四状态。说明增加幅度并不保证原匹配规则仍然选择同一条边缘。这里不放宽匹配范围、不重新挑曲线；输出失败可能叠加固定门槛、邻近轮廓竞争、轨迹截断和扩散变化，不能全部解释为“强度不够”或“几何信息彻底丢失”。已知输入对应明细、标准输入Auto与旧Oracle管道代理结果分别保留，后者同样不是扩散输出的正确对应Oracle。

计划移动被编辑边界裁掉的像素总数：{sum(r['planned_moves_clipped'] for r in input_rows)}。实际位移、移动比例、条件二值XOR修改数与源已知系数均逐码字保存。连续位移是最大预算；平均绝对偏移约为峰值的0.486倍，RMS约为0.575倍，整数取整后不等于每点都移动该峰值。

## 冻结质量约束

{qt}
每张marked均与**相同内容、prompt、seed和初始噪声的clean RGB**比较。冻结要求：PSNR≥30dB、SSIM≥0.90、LPIPS-Alex≤0.10、RGB配对Canny Edge-F1≥0.95（原3×3容差规则）。每个强度全部48张均计入，包括解码失败。逐项通过数与失败job见quality_summary.json及quality目录，不能只看中位数。

![质量统计](plots/quality.png)

## 跨内容一致性与解释限制

{gt}
Bootstrap以六个内容为块，同时保留同内容的两视图和四消息，重采样10000次；它只重采已固定的判决，不在每次重采样重新训练LOCO模型。区间用于探索，六个内容不能支撑强泛化结论。所有参考辅助与自参考差距都同时包含定位、片段选择、方向/弧长参数化和特征基线差异，不能将两者差距完全归因于系数拟合。

如果参考辅助随强度改善且质量合格，而自参考仍低，支持把强度不足与参考依赖分开研究；如果参考辅助也不改善，需要先查看固定接收规则能否接纳更大形变，以及输入/输出曲线对应是否正确，不能继续只增加幅度。管道Mask本身透露源形状，bbox仍透露位置/尺度。四个全部合法的码字仍无认证冗余，存在性AUC也不能替代Key认证。本轮不更改峰谷形式、不增加消息容量，也未引入RS/RM。

## 查看与复现

- [逐图页面](index.html)：每内容、视图、消息都显示clean、1.5/2/3/4px；三行分别为实际输入Canny、生成RGB、RGB提取Canny。固定同一裁剪窗口，无假边缘补点或支持颜色覆盖。
- [恢复汇总](summary.json)、[质量汇总](quality_summary.json)、[逐图LOCO判决](decisions.json)、[六折模型和阈值](loco_folds.json)。
- [输入闭环](input_closure.json)、[冻结输入接收器结果](input_receiver_summary.json)、[配对内容统计](metrics/per_content.json)、[响应矩阵](metrics/response_matrices.json)。
- [冻结协议](protocol.json)、[评分规则冻结](analysis_freeze.json)、[运行复现核验](canary.json)、[审计](audit.json)。

运行顺序：canny_st_amplitude_cal_scan.py prepare → worker（GPU1）；extract与quality可在CPU随生成运行；analyze_st_amplitude_cal_scan.py inputs → evaluate；最后report_st_amplitude_cal_scan.py。所有父实验保持只读，文件及输入哈希均核验。
"""
    (OUT/'REPORT_ZH.md').write_text(report)
    panels=read(OUT/'panels.json');lookup={j['job']:j for j in read(OUT/'plan.json')};decisions=read(OUT/'decisions.json');by={(r['job'],r['method']):r for r in decisions if r['arm'] in WORDS}
    page=['<!doctype html><meta charset="utf-8"><title>S/T Cal strength scan</title><style>body{font:15px sans-serif;max-width:1450px;margin:25px auto}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:6px}img{max-width:100%}pre{white-space:pre-wrap}.warn{background:#fff2ce;padding:14px}select{font-size:16px;padding:6px}</style>',
      '<h1>Cal S/T强度扫描：1.5 / 2 / 3 / 4px</h1><p class="warn">6个Cal内容、两组原prompt/seed。匹配规则和接收端Mask冻结；评分模型按留一内容规则校准。marked分母48，held-out clean分母12，额外32 clean仅校准阈值。属于Cal探索，非独立验收。</p>',
      '<p><a href="REPORT_ZH.md">完整报告</a> · <a href="summary.json">恢复</a> · <a href="quality_summary.json">质量</a> · <a href="audit.json">审计</a></p>',
      '<img src="plots/recovery_and_presence.png"><img src="plots/quality.png"><h2>同内容、同视图、同消息逐强度对比</h2><p>列：clean、1.5、2、3、4px；行：实际输入Canny、生成RGB、RGB提取Canny。三行均无真假支持颜色覆盖。</p>',
      '<select id="choose">']
    for panel in panels:page.append('<option value="'+html.escape(panel['path'])+'">'+html.escape(panel['name'])+'</option>')
    page.append('</select><p><img id="panel" src="'+panels[0]['path']+'"></p><script>document.getElementById("choose").addEventListener("change",function(){document.getElementById("panel").src=this.value;});</script>')
    page.append('<h2>192次marked判决（错误与擦除均计失败）</h2><table><tr><th>RGB / 真消息 / 强度</th>')
    for method in METHODS:page.append('<th>'+LABELS[method]+'</th>')
    page.append('<th>四项质量</th></tr>')
    for j in [j for j in lookup.values() if j['arm'] in WORDS]:
        page.append('<tr><td><a href="'+html.escape(os.path.relpath(j['rgb'],OUT))+'">'+html.escape(j['job'])+'</a></td>')
        for method in METHODS:
            r=by[j['job'],method];d=r['decoded'];color='#d6f5dc' if d['accepted'] and d['word']==j['arm'] else '#ffd6d6' if d['accepted'] else '#eee'
            page.append('<td style="background:'+color+'">'+html.escape(d['word'] if d['accepted'] else '擦除')+'</td>')
        q=read(OUT/'quality'/f'{j["job"]}.json');page.append('<td>'+('通过' if q['pass_all'] else '失败：'+','.join(k for k,v in q['gates'].items() if not v))+'</td></tr>')
    page.append('</table>');(OUT/'index.html').write_text('\n'.join(page))


def finalize():
    names=['canny_st_amplitude_cal_scan.py','analyze_st_amplitude_cal_scan.py','report_st_amplitude_cal_scan.py']
    for name in names:shutil.copyfile(ROOT/name,OUT/'source'/name)
    dump(OUT/'analysis_source_sha256.json',{name:digest(ROOT/name) for name in names})
    dump(OUT/'final_artifacts_sha256.json',{str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='final_artifacts_sha256.json'})


if __name__=='__main__':
    audit();render();print('REPORT_AUDIT_COMPLETE',OUT,flush=True);finalize()
