"""Build a descriptive report and audit; never refit thresholds on Fresh48."""
from compare_st_identifiability_4b import *
import collections, html, subprocess, sys


def aggregate(records):
    result=[]
    for scenario in SCENARIOS:
        rows=[r for r in records if r['kind']=='ObsDrop' and r['scenario']==scenario['name']]
        mm=[r['metrics'] for r in rows if r['metrics'] is not None]
        def med(fn):return float(np.median([fn(m) for m in mm])) if mm else None
        result.append(dict(scenario=scenario['name'],attempts=len(rows),fitted=len(mm),received=sum(r['accepted4b'] for r in rows),
            coverage=med(lambda m:m['sample_coverage']),S_energy=med(lambda m:m['ST_energy_coverage'][0]),T_energy=med(lambda m:m['ST_energy_coverage'][1]),
            raw_lambda_min=med(lambda m:m['G_eigenvalues'][0]),projected_lambda_min=med(lambda m:m['G_offset_projected_eigenvalues'][0]),
            raw_condition=med(lambda m:m['G_condition']),projected_condition=med(lambda m:m['G_offset_projected_condition']),
            relative_info_min=med(lambda m:m['joint_info_min']),
            distances={d:{k:med(lambda m,d=d,k=k:m['distances'][d][k]) for k in ['min_d2','min_rms','min_offset_projected_d2','min_offset_projected_rms']} for d in ['nominal','actual_input','channel_output']}))
    return result


def audit():
    check();counts=collections.Counter();frozen=read(PRIOR/'score_freeze.json')
    assert frozen['channel_model']==read(PARENT/'score_freeze.json')['models']['Auto']
    cc={c['key']:c for c in read(PRIOR/'selection.json')};jobs={j['job']:j for j in read(PRIOR/'plan.json') if j['arm'] in WORDS}
    assert len(jobs)==48 and len({j['key'] for j in jobs.values()})==12
    rows=read(OUT/'comparison.json');index={(r['job'],r['method']):r for r in rows}
    assert len(index)==192
    for job,j in jobs.items():
        c=cc[j['key']];m=read(OUT/'measurements'/f'{job}.json')
        assert digest(j['rgb'])==m['rgb_sha256'];counts['unchanged_rgb_hashes']+=1
        assert m['native']['Auto']==read(PRIOR/'loss'/f'{job}.json')['original'];counts['exact_original_fit_reproductions']+=1
        aa=index[job,'A4'];bb=index[job,'B4'];old=index[job,'B4_PreviousObsDrop']
        for decoder in ['native','common_A4','common_B4']:
            assert (aa[decoder]['accepted'],aa[decoder]['word'])==(bb[decoder]['accepted'],bb[decoder]['word'])
            assert (bb[decoder]['accepted'],bb[decoder]['word'])==(old[decoder]['accepted'],old[decoder]['word'])
        rgb=np.array(Image.open(j['rgb']).convert('RGB'));edge=cv2.Canny(rgb,100,200)
        p,u,n,_=geometry(c['samples']);model=frozen['channel_model'];a=np.array(model['A']);intercept=np.array(model['intercept'])
        books=dict(nominal=a4.AMPLITUDE*np.array([a4.SIGNS[w] for w in WORDS]),actual_input=np.array([a4.features(c,w,'Auto') for w in WORDS]))
        books['channel_output']=books['actual_input']@a.T+intercept
        for f in m['raw4b'].get('pool',[]):
            ii=np.array(f['indices'],int);xy=np.array(f['observed_xy']);w=np.array(f['weights']);phi=basis(u[ii]);metric=residual_metrics(c,f)
            assert np.array_equal(xy,np.round(xy));assert np.all(edge[xy[:,1].astype(int),xy[:,0].astype(int)]>0)
            np.testing.assert_allclose(np.sum((xy-p[ii])*n[ii],axis=1),f['normal_positions'])
            counts['actual_observed_pixels_checked']+=len(ii)
            fullx=np.c_[np.ones(len(ii)),phi];gx=fullx.T@(w[:,None]*fullx)
            schur=gx[1:,1:]-np.outer(gx[1:,0],gx[0,1:])/gx[0,0]
            np.testing.assert_allclose(schur,metric['G_offset_projected'],atol=1e-10)
            for domain,book in books.items():
                for pair,(i,k) in zip(metric['distances'][domain]['pairs'],PAIRS):
                    profile=phi@(book[i]-book[k]);direct=float(np.sum(w*profile**2));centered=profile-np.average(profile,weights=w)
                    np.testing.assert_allclose(direct,pair['d2'],atol=1e-9)
                    np.testing.assert_allclose(np.sum(w*centered**2),pair['offset_projected_d2'],atol=1e-9)
                    counts['direct_codeword_distance_checks']+=1
            counts['schur_complement_checks']+=1
    for method in ['A4','B4','B4_PreviousObsDrop','OracleMask_proxy']:
        for dec in ['native','common_A4','common_B4']:
            rr=[r for r in rows if r['method']==method];accepted=[r for r in rr if r[dec]['accepted']]
            correct=sum(r[dec]['word']==r['arm'] for r in accepted)
            assert read(OUT/'summary.json')[method][dec]==dict(attempts=48,received=len(accepted),correct=correct,wrong=len(accepted)-correct,erasures=48-len(accepted),full_accuracy=correct/48)
    for folder in [PARENT,PRIOR]:
        for rel,h in read(folder/'final_artifacts_sha256.json').items():
            assert digest(folder/rel)==h,(folder,rel);counts['parent_manifest_entries_verified']+=1
    oracle_before=(OUT/'verified_oracle_result.json').exists()
    proc=subprocess.run([sys.executable,str(ROOT/'apply_verified_oracle_st_4b.py'),str(OUT/'true_oracle_annotations.template.json')],capture_output=True,text=True)
    assert proc.returncode!=0 and 'Unannotated entries' in proc.stderr
    assert (OUT/'verified_oracle_result.json').exists()==oracle_before
    dump(OUT/'logs/oracle_template_rejection.json',dict(returncode=proc.returncode,expected=True,message=proc.stderr.splitlines()[-1]))
    dump(OUT/'audit.json',dict(status='passed',counts=dict(counts),all48_same_png=True,all48_denominator=True,
        same_A4_B4_decisions=True,original_4a_channel_model_equal=True,unannotated_oracle_template_rejected=True,
        true_oracle_status='requires_verified_annotations',limitations='Pixel membership and numeric consistency do not establish curve identity.'))


def render():
    check();rows=read(OUT/'comparison.json');records=read(OUT/'identifiability.json');summary=read(OUT/'summary.json');agg=aggregate(records)
    dump(OUT/'metrics/scenario_summary.json',agg)
    pools=[r for r in records if r['kind']=='RematchPool'];set_rows=[r['set_receiver'] for r in pools]
    set_correct=sum(r['set_receiver']['accepted'] and r['set_receiver']['word']==r['arm'] for r in pools)
    multi_summary=dict(attempts=48,received=sum(r['accepted'] for r in set_rows),correct=set_correct,
        wrong=sum(r['accepted'] for r in set_rows)-set_correct,ambiguous_erasure=sum(r['reason']=='ambiguous_correspondence' for r in set_rows),
        raw_near_equivalent_pools=sum(len(r['hypotheses']['near_equivalent_branches'])>=2 for r in pools))
    dump(OUT/'metrics/multi_hypothesis_summary.json',multi_summary)
    frozen=read(PRIOR/'score_freeze.json');candidate_rows=[];cc={c['key']:c for c in read(PRIOR/'selection.json')}
    for r in pools:
        m=read(OUT/'measurements'/f'{r["job"]}.json');fits=[]
        for f in m['raw4b'].get('pool',[]):
            g=b4.gate_one(f,frozen['gate']);decoded=b4.decode(cc[r['key']],g,frozen['channel_model'])
            fits.append(dict(branch=f['branch'],cost=f['cost'],accepted=g['accepted'],word=decoded['word'],coefficient=f['coefficient'],
                failed_geometry=[k for k,v in f.get('gate_checks',{}).items() if not v],failed_identifiability=[k for k,v in g.get('recovery_checks',{}).items() if not v],metrics=residual_metrics(cc[r['key']],f)))
        candidate_rows.append(dict(job=r['job'],hypotheses=r['hypotheses'],candidates=fits,set_receiver=r['set_receiver']))
    dump(OUT/'metrics/correspondence_hypotheses.json',candidate_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(16,8));labels=[r['scenario'] for r in agg];xx=np.arange(len(agg))
    for ax,fn,title in zip(axes.flat,[lambda r:r['coverage'],lambda r:r['relative_info_min'],lambda r:r['projected_condition'],lambda r:r['projected_lambda_min'],lambda r:r['distances']['channel_output']['min_offset_projected_d2'],lambda r:r['distances']['channel_output']['min_offset_projected_rms']],
            ['Observed station coverage','Relative joint information: min eigenvalue','Offset-projected condition number','Offset-projected min eigenvalue','Min output codeword distance squared (weighted px^2)','Min output codeword RMS (px)']):
        ax.plot(xx,[fn(r) for r in agg],'o-');ax.set_xticks(xx,labels,rotation=35,ha='right');ax.set_title(title,fontsize=10);ax.grid(alpha=.25)
    fig.suptitle('Fixed-observation loss diagnostics: medians over 28 fitted cases; decoding denominator remains 48')
    fig.tight_layout();fig.savefig(OUT/'plots/identifiability.png',dpi=160);plt.close(fig)
    table='| 接收器 | 接收 | 正确 | 错误 | 擦除 | 完整 2-bit 正确率 |\n|---|---:|---:|---:|---:|---:|\n'
    for key,label in [('A4','冻结 4A 原始'),('B4','当前 4B 联合拟合'),('OracleMask_proxy','已知曲线管道 Oracle 代理')]:
        s=summary[key]['native'];table+=f"| {label} | {s['received']}/48 | {s['correct']} | {s['wrong']} | {s['erasures']} | {s['full_accuracy']:.1%} |\n"
    table+='| 正确对应 Oracle | 缺标注 | — | — | — | 不可报告 |\n'
    diag='| 缺失模式 | 拟合/48 | 接收/48 | 采样覆盖 | S 能量覆盖 | T 能量覆盖 | 原始最小特征值 | 消 offset 后最小特征值 | 消 offset 后条件数 | 最小码字 d² | 最小码字 RMS |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in agg:
        d=r['distances']['channel_output'];diag+=f"| {r['scenario']} | {r['fitted']} | {r['received']} | {r['coverage']:.1%} | {r['S_energy']:.1%} | {r['T_energy']:.1%} | {r['raw_lambda_min']:.3f} | {r['projected_lambda_min']:.3f} | {r['projected_condition']:.3f} | {d['min_offset_projected_d2']:.3f} | {d['min_offset_projected_rms']:.3f} px |\n"
    report=f"""# 阶段 4B：同 PNG 接收器对比与 S/T 可辨识性

## 同一批 48 张无人工删除 RGB

{table}
冻结 4A 与当前 4B 的逐图接收状态和消息完全一致（变化 0/48）。同时统一使用 4A 或 4B 分类器，计数均不改变。旧 4A 的 79/96 与本批 25/48 来自不同内容，不能据此断言 4B 接收器退化。原始 4A 本身已联合 Huber 拟合 offset、S、T；4B 新增联合可辨识性门槛和拟合协方差感知分类。

48 张图来自 12 个内容，每内容四个消息；方向按原曲线端点规范、原弧长参数与法向固定。所有拒绝均计入 48 分母，没有丢弃失败图。没有重新生成或编辑 RGB。原冻结质量结果仍为 48/48 联合通过。重新匹配与旧 ObsDrop:none 的重复鲁棒拟合可能产生小幅系数和分数差异，但逐图接收状态及消息一致。

当前解码除了 RGB 与 Mask，还需要原曲线有序采样点、规范方向/弧长/法向、S/T 基函数与码本、clean Canny 参考系数、该载体四种码字的实际栅格输入系数、Cal 冻结信道模型和拒绝阈值。这些可以预先保存，不必每次重新读取原 RGB。属于依赖原曲线参考的解码，尚不是仅凭 RGB、Mask 和 Key 的盲解码。不使用生成 prompt、seed 或真实消息作为接收条件；真实消息仅用于计分。

**正确对应 Oracle 尚不能运行：现有文件没有经过核验的原曲线采样位置→输出边缘像素对应。** 已运行的代理只知道原曲线附近搜索管道，26/48 不是正确对应 Oracle 上界，不能据此判断几何信息已经丢失。不能用真消息选择最符合码字的路径来替代 Oracle。

## 带可辨识性约束的恢复

保持同一码本与 Huber 联合拟合，记录真实观测点的最终鲁棒权重。除原始 G=PhiᵀWPhi，还报告消除未知 offset 的加权中心化 Gram（等价于完整 [1,S,T] 信息矩阵的 Schur 补）。输出几何有整体偏移，单看原始二阶矩可能低估 S/T 与 offset 的混淆。

逐图保存原始及消 offset 后的特征值、条件数、相对于完整固定站位的联合信息特征值、S/T 能量覆盖、采样覆盖与跨度。四码字六个配对距离同时在连续输入码本、实际栅格输入拟合码本、冻结 Cal 信道输出中心三个域计算。每对都保存加权 d² 及除以总权重后的 RMS；二者不能互相替代。距离使用实际观测位置，拟合插值不计为观测。

{diag}
表内连续指标为 **28 个有可用原始对应的样本的中位数**，接收统计的分母始终为 48。其余 20 个没有被悄悄移出恢复率分母。缺失模式作用于已匹配观测，主要隔离估计可辨识性；另有 EdgeDrop 重新匹配结果保存在 identifiability.json，不能混作同一实验。

中央删除 25% 时仍保留约 70% 采样点，但 T 能量覆盖约 48%；中央删除 50% 后消 offset 的最小特征值约 1.00、条件数约 18.74，输出域最小码字 RMS 降至约 0.196 px。相比之下，随机删除 50% 的每点 RMS 可保持，但总 d² 与信息量减少。这支持“看剩余观测能否区分消息”，不支持“点数足够就能恢复”的判断。中央缺口下原冻结门槛拒绝解码，应保留擦除。

![信息矩阵和残余码字距离](plots/identifiability.png)

## 多候选与拒绝

保留 R0、A3、A3_alternative 的坐标、代价和系数；按几何路径差异去重后报告最优/次优有符号相对差距。没有第二候选时差距为 null，不能视为唯一性证据。原实现 q_margin 会截零，且 margin≥0 的检查不能拒绝零间隔，本轮不改动冻结 baseline。

48 个原始候选池中，{multi_summary['raw_near_equivalent_pools']} 个具有代价差≤0.10 的多个不同几何假设。新增试验保留通过原几何与可辨识性门槛的近似等价假设；只有各假设消息一致才接收，否则保留消息集合并擦除。选择依据与目标 bit 无关。

残余输出域最小 RMS 门槛在旧 Cal 的 288 次标记尝试、64 次错误锚点上选择并冻结为 0.5 px。门槛 0 至 0.5 的结果相同：29 正确、3 错误、256 擦除，错误锚点接受 6/64；0.65 会降至 12 正确、3 错误、273 擦除，负例 1/64。此门槛尚无可测新增收益。当前 48 张多假设试验仍为正确 25、错误 3、擦除 20，没有通过简单放宽门槛增加接收；没有出现已通过门槛却消息不一致的新增擦除。

## 失败与实验边界

原 20 次拒绝中 3 次无数值候选；另有覆盖不足 10、跨度不足 13、信息不足 12、锚点不足 9、残差过大 7、不确定度过大 5 等（原因可重叠，见父实验 original_sync_failures.json）。这些是几何诊断，不足以证明对应轮廓不存在；同样，满秩信息矩阵只说明给定对应可估计，不证明对应正确。

本批已用于接收器诊断，不是新独立验收。四个全部合法的 2-bit 码字仍无认证冗余，码字可辨不能替代 clean-vs-marked 检测。本轮未新增 clean 阈值或误报结论，也未引入 RS/RM。

## 查看与正确对应标注

- [逐图页面](index.html)：三头并排、候选差距和完整候选 JSON；Oracle 代理明确标识。
- [机器可读逐图对比](comparison.json)、[全部可辨识性记录](identifiability.json)、[场景汇总](metrics/scenario_summary.json)。
- [冻结协议](protocol.json)、[Cal 距离门槛](distance_freeze.json)、[审计](audit.json)。
- blind_review/content_00.png 至 content_11.png 为不显示消息的核验图：左列源 RGB 与目标曲线，右侧四列随机排列输出；绿色点只是候选，不能作为真值。
- [正确对应标注模板](true_oracle_annotations.template.json)：48 项都需明确 verified/absent/ambiguous，记录原规范索引、真实 Canny 像素、权重、核验者和不依赖消息的理由。未标注条目会被拒绝，不能自动算擦除。语义相似也不能证明精确点对应。

具备独立核验标注后，可运行：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/hongwei/.conda/envs/control/bin/python apply_verified_oracle_st_4b.py /absolute/path/verified_annotations.json
```

该入口保持同一码字、方向、4B 门槛与解码器，absent/ambiguous 计为擦除、分母固定 48，并保存标注哈希和来源。它不会伪造“已知正确”对应。
"""
    (OUT/'REPORT_ZH.md').write_text(report)
    def esc(x):return html.escape(str(x))
    page=['<!doctype html><meta charset="utf-8"><title>S/T receiver diagnostics</title><style>body{font:15px sans-serif;max-width:1400px;margin:25px auto}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:6px}img{max-width:100%}pre{white-space:pre-wrap}details{margin:15px 0}.warn{background:#fff2ce;padding:14px}</style>',
        '<h1>同一批 48 张 RGB：S/T 接收器诊断</h1><p class="warn">正确对应 Oracle 缺少核验标注，尚未运行。Oracle proxy 为已知原曲线管道搜索，不能当作正确对应上界。全部恢复率分母为 48；连续信息指标中位数仅来自 28 个有拟合的样本。</p>',
        '<p><a href="REPORT_ZH.md">完整中文报告</a> · <a href="audit.json">审计</a> · <a href="true_oracle_annotations.template.json">标注模板</a></p>',
        '<img src="plots/identifiability.png"><h2>逐图对比</h2><table><tr><th>PNG / 真消息</th><th>冻结 4A</th><th>当前 4B</th><th>Oracle proxy</th><th>原图</th></tr>']
    jobmap={j['job']:j for j in read(PRIOR/'plan.json')};idx={(r['job'],r['method']):r for r in rows}
    for r in [r for r in rows if r['method']=='A4']:
        j=r['job'];page.append('<tr><td>'+esc(j)+'</td>')
        for method in ['A4','B4','OracleMask_proxy']:
            d=idx[j,method]['native'];value=d['word'] if d['accepted'] else '擦除';page.append('<td>'+esc(value)+'</td>')
        page.append('<td><a href="'+esc(os.path.relpath(jobmap[j]['rgb'],OUT))+'">RGB</a></td></tr>')
    page.append('</table><h2>几何候选池（候选消息仅作诊断，不用于选择对应）</h2>')
    for r in candidate_rows:
        page.append('<details><summary>'+esc(r['job'])+' — '+esc(r['hypotheses'])+'</summary><pre>'+esc(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details>')
    page.append('<h2>不显示消息的对应核验图</h2>')
    for f in sorted((OUT/'blind_review').glob('*.png')):page.append(f'<details><summary>{f.name}</summary><img loading="lazy" src="blind_review/{f.name}"></details>')
    (OUT/'index.html').write_text('\n'.join(page))


def finalize():
    sources=['compare_st_identifiability_4b.py','apply_verified_oracle_st_4b.py','report_st_identifiability_4b.py']
    for name in sources:shutil.copyfile(ROOT/name,OUT/'source'/name)
    dump(OUT/'analysis_source_sha256.json',{name:digest(ROOT/name) for name in sources})
    dump(OUT/'final_artifacts_sha256.json',{str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='final_artifacts_sha256.json'})


if __name__=='__main__':
    audit();render();finalize();print('REPORT_AND_AUDIT_COMPLETE',OUT,flush=True)
