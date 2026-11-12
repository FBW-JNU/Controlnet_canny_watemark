"""Human-readable paired carrier comparison; no tuning or image generation."""
from canny_st_stable_carrier_compare import *


def probe_coefficients():
    """Post-freeze diagnostic only; never used to select a carrier or threshold."""
    choices=read(OUT/'selection.json');plan=read(OUT/'probe_plan.json');frozen=read(b4.OUT/'score_freeze.json')
    records=[];cache={}
    for c in choices:
        for policy in POLICIES:
            q=c['carriers'][policy]
            if q is None:continue
            for j in [j for j in plan if j['content_id']==c['content_id']]:
                cachekey=(q['key'],j['job'])
                if cachekey not in cache:
                    rgb=np.array(Image.open(j['rgb']).convert('RGB'))
                    raw=b4.rematch(q['samples'],rgb,j,b4.SCENARIOS[0]);obs=b4.select(raw,frozen['gate'])
                    value=(np.array(obs['coefficient'])-np.array(q['states']['clean']['measurements']['Auto']['coefficient'])).tolist() if obs.get('accepted') else None
                    cache[cachekey]=dict(observation=obs,coefficient_delta=value)
                records.append(dict(policy=policy,content_id=c['content_id'],key=q['key'],job=j['job'],**cache[cachekey]))
    summary={}
    for policy in POLICIES:
        rs=[r for r in records if r['policy']==policy];values=[r['coefficient_delta'] for r in rs if r['coefficient_delta'] is not None]
        summary[policy]=dict(attempts=len(rs),geometry_received=len(values),
            coefficient_std=np.std(values,axis=0).tolist() if values else None,
            coefficient_rms=np.sqrt(np.mean(np.array(values)**2,axis=0)).tolist() if values else None)
    dump(OUT/'probe_coefficient_diagnostics.json',dict(records=records,summary=summary,
         note='Post-selection descriptive clean-probe statistics, biased by carrier selection; not held-out evidence or a true-correspondence Oracle. No tuning.'))


def diagnostics():
    choices=read(OUT/'selection.json');features=[read(p) for p in sorted((OUT/'features').glob('*.json'))]
    rows=[]
    for c in choices:
        entry=dict(content_id=c['content_id'],source_candidates=c['source_candidates'],stable_candidates=c['stable_candidates'],policies={})
        for policy in POLICIES:
            q=c['carriers'][policy];probe=c['selected_probe'][policy]
            if q is None:
                entry['policies'][policy]=dict(available=False);continue
            marked=[r for r in features if r['content_id']==c['content_id'] and r['policy']==policy and r['arm']!='clean']
            clean=[r for r in features if r['content_id']==c['content_id'] and r['policy']==policy and r['arm']=='clean']
            entry['policies'][policy]=dict(available=True,key=q['key'],length_px=q['arc_length_px'],probe_matches=probe['matched'],
                probe_R=probe['R'],probe_distance_px=probe['median_symmetric_distance'],probe_prompt_matches=probe['per_prompt_matched'],
                clean_strict_matches=sum(r['strict_match']['matched'] for r in clean),clean_geometry_received=sum(r['heads']['SourceRef']['accepted'] for r in clean),
                marked_strict_matches=sum(r['strict_match']['matched'] for r in marked),marked_geometry_received=sum(r['heads']['SourceRef']['accepted'] for r in marked),
                marked_correct=sum(r['heads']['SourceRef']['accepted'] and r['heads']['SourceRef']['word']==r['arm'] for r in marked))
        entry['carrier_changed']=c['carriers']['clean_stable'] is None or c['carriers']['source_only']['key']!=c['carriers']['clean_stable']['key']
        rows.append(entry)
    transitions=collections.Counter()
    for a in features:
        if a['policy']!='source_only' or a['arm']=='clean':continue
        b=next(r for r in features if r['policy']=='clean_stable' and (r['content_id'],r['view'],r['arm'])==(a['content_id'],a['view'],a['arm']))
        def state(r):
            d=r['heads']['SourceRef']
            return 'erasure' if not d['accepted'] else 'correct' if d['word']==r['arm'] else 'wrong'
        transitions[f'{state(a)} -> {state(b)}']+=1
    failed=collections.Counter()
    for r in features:
        if r['arm']=='clean' or r['heads']['SourceRef']['accepted']:continue
        pool=r['raw'].get('pool',[])
        if not pool:failed[r['policy']+':no_candidate']+=1;continue
        # Diagnostic best candidate is cost-only, never selected using the message.
        best=min(pool,key=lambda q:q['cost'])
        checks={k:best['gate_checks'][k] for k in ['tangent','order','residual','offset','distance','margin']}
        gated=b4.gate_one(best,read(b4.OUT/'score_freeze.json')['gate'])
        checks.update(gated.get('recovery_checks',{}))
        for name,ok in checks.items():
            if not ok:failed[r['policy']+':'+name]+=1
    dump(OUT/'carrier_diagnostics.json',dict(contents=rows,transitions=dict(transitions),
         best_cost_rejected_candidate_failed_checks=dict(failed),note='Checks may overlap. Correct output correspondences are unannotated; gate failure is not proof of information loss.'))


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=read(OUT/'summary.json');x=np.arange(len(METHODS));fig,axes=plt.subplots(1,3,figsize=(14,4))
    for i,p in enumerate(POLICIES):
        for ax,key,title in [(axes[0],'full_accuracy','Full 2-bit accuracy / all48'),(axes[1],'erasures','Erasures / all48'),(axes[2],'presence_AUC','Clean vs marked AUC')]:
            ys=[summary[p][m][key]/48 if key=='erasures' else summary[p][m][key] for m in METHODS]
            ax.bar(x+(i-.5)*.34,ys,width=.34,label=p)
            ax.set_title(title);ax.set_xticks(x,METHODS,rotation=12);ax.set_ylim(0,1.05);ax.grid(axis='y',alpha=.2)
    axes[0].legend();fig.suptitle('Six Cal contents; new held-out generation pairs; all receiver parameters frozen')
    fig.tight_layout();fig.savefig(OUT/'plots/paired_results.png',dpi=150);plt.close(fig)


def viewer():
    rows=[read(p) for p in sorted((OUT/'features').glob('*.json'))];choices=read(OUT/'selection.json')
    entries=[]
    for choice in choices:
        cid=choice['content_id'];qs=[q for q in choice['carriers'].values() if q is not None]
        guide=Image.new('RGB',(1024,554),'#eeeeee');gd=ImageDraw.Draw(guide)
        for column,policy in enumerate(POLICIES):
            q=choice['carriers'][policy]
            if q is None:continue
            original=np.array(Image.open(q['source_rgb']).convert('RGB')).astype(float)
            mask=np.array(Image.open(q['mask']).convert('L'))>0
            color=np.array([0,180,255]) if column==0 else np.array([255,100,0])
            original[mask]=.4*original[mask]+.6*color
            guide.paste(Image.fromarray(original.round().astype('uint8')),(column*512,42))
            gd.text((column*512+5,5),policy+' / '+q['key'],fill='black')
            gd.text((column*512+5,22),'Original source RGB + colored carrier Mask (location guide only)',fill='black')
        guide_name=f'{cid:012d}_mask_locations.png';guide.save(OUT/'panels'/guide_name)
        points=np.concatenate([q['samples'] for q in qs]);lo=np.floor(points.min(0)-24).astype(int);hi=np.ceil(points.max(0)+25).astype(int)
        center=(lo+hi)/2;side=max(112,int(max(hi-lo)));side=min(512,side)
        x0=int(np.clip(center[0]-side/2,0,512-side));y0=int(np.clip(center[1]-side/2,0,512-side));box=(x0,y0,x0+side,y0+side)
        for view in range(2):
            for arm in WORDS:
                rs=[next(r for r in rows if (r['policy'],r['content_id'],r['view'],r['arm'])==(policy,cid,view,arm)) for policy in POLICIES]
                cleans=[next(r for r in rows if (r['policy'],r['content_id'],r['view'],r['arm'])==(policy,cid,view,'clean')) for policy in POLICIES]
                columns=[cleans[0],rs[0],cleans[1],rs[1]]
                tile=256;canvas=Image.new('RGB',(4*tile,3*tile+118),'#eeeeee');draw=ImageDraw.Draw(canvas)
                draw.text((8,5),f'content={cid:012d} view={view} true message={arm}; identical crop for both policies',fill='black')
                for col,r in enumerate(columns):
                    d=r['heads']['SourceRef'];word=d.get('word') if d['accepted'] else 'ERASURE'
                    draw.text((col*tile+5,26),r['policy']+' / '+r['arm'],fill='black')
                    draw.text((col*tile+5,43),f'SourceRef: {word}',fill='black')
                    condition=Image.open(r['hint']).convert('RGB')
                    rgb=Image.open(r['rgb']).convert('RGB');edge=Image.fromarray(cv2.Canny(np.array(rgb),100,200)).convert('RGB')
                    for row,image in enumerate([condition,rgb,edge]):
                        canvas.paste(image.crop(box).resize((tile,tile),Image.Resampling.NEAREST if row!=1 else Image.Resampling.LANCZOS),(col*tile,70+row*tile))
                draw.text((8,3*tile+76),'Rows: actual input Canny / generated RGB / Canny extracted from generated RGB. No fitted/synthetic edge overlay.',fill='black')
                name=f'{cid:012d}_v{view}_{arm}.png';canvas.save(OUT/'panels'/name)
                entries.append(dict(content_id=cid,view=view,arm=arm,panel='panels/'+name,
                                    mask_guide='panels/'+guide_name,
                                    source_key=rs[0]['key'],stable_key=rs[1]['key'],source_decision=rs[0]['heads']['SourceRef'],stable_decision=rs[1]['heads']['SourceRef']))
    dump(OUT/'viewer_entries.json',entries)
    options=''.join(f'<option value="{i}">{e["content_id"]:012d} / view{e["view"]} / {e["arm"]}</option>' for i,e in enumerate(entries))
    cards=''.join(f'<details><summary>{e["content_id"]:012d} view{e["view"]} {e["arm"]}</summary><a href="{e["panel"]}"><img loading="lazy" src="{e["panel"]}"></a></details>' for e in entries)
    page='''<!doctype html><meta charset="utf-8"><title>S/T 稳定载体对照</title>
<style>body{font:16px sans-serif;max-width:1250px;margin:24px auto;padding:0 12px;background:#fafafa;color:#222}img{max-width:100%;height:auto}select,button{font-size:16px;padding:8px}pre{white-space:pre-wrap}details{padding:8px;border-bottom:1px solid #ddd}</style>
<h1>S/T 稳定载体筛选对照</h1><p>6 个既有 Cal 内容；clean 探测与评价 prompt/seed 分开。码本、1.5 px 幅度、接收器和评分模型冻结。</p>
<p>四列：原方法 clean、原方法 marked、稳定筛选 clean、稳定筛选 marked。三行：实际输入 Canny、生成 RGB、RGB 提取 Canny。所有列使用相同裁剪；无拟合边缘补点。</p>
<p><a href="REPORT_ZH.md">中文报告</a> · <a href="summary.json">汇总</a> · <a href="carrier_diagnostics.json">逐内容载体与失败诊断</a> · <a href="audit.json">审计</a></p>
<img src="plots/paired_results.png"><p><button onclick="move(-1)">上一张</button> <select id="choice">OPTIONS</select> <button onclick="move(1)">下一张</button></p>
<a id="full"><img id="panel"></a><p>载体位置示意：原始源图上叠加 Mask；蓝色为原方法，橙色为稳定筛选。彩色仅用于标记位置，不是输出边缘观测。</p><a id="maskfull"><img id="maskguide"></a><pre id="info"></pre><h2>全部逐图结果</h2>CARDS
<script>const entries=ENTRIES;const selector=document.getElementById('choice');function show(){const e=entries[+selector.value];document.getElementById('panel').src=e.panel;document.getElementById('full').href=e.panel;document.getElementById('maskguide').src=e.mask_guide;document.getElementById('maskfull').href=e.mask_guide;document.getElementById('info').textContent=JSON.stringify(e,null,2)}function move(d){selector.value=(+selector.value+d+entries.length)%entries.length;show()}selector.onchange=show;show();</script>'''
    (OUT/'index.html').write_text(page.replace('OPTIONS',options).replace('CARDS',cards).replace('ENTRIES',json.dumps(entries)),encoding='utf8')


def report():
    summary=read(OUT/'summary.json');diagnostic=read(OUT/'carrier_diagnostics.json');paired=read(OUT/'paired_bootstrap.json')
    source=summary['source_only']['SourceRef'];stable=summary['clean_stable']['SourceRef'];gain=stable['correct']-source['correct']
    changed=sum(r['carrier_changed'] for r in diagnostic['contents']);missing=sum(not r['policies']['clean_stable']['available'] for r in diagnostic['contents'])
    lines=['# S/T：原始选曲线与 clean 扩散稳定性筛选配对对照','',
           f'原方法完整2-bit恢复 {source["correct"]}/48（{source["full_accuracy"]:.1%}），增加稳定性筛选后 {stable["correct"]}/48（{stable["full_accuracy"]:.1%}），变化 {gain:+d}/48。6个内容中载体改变{changed}个，稳定且输入合格载体缺失{missing}个。结果来自既有Cal内容，不能视为独立内容验收。','',
           '## 固定条件与评价隔离','',
           '复用原6个Cal内容，每内容3个prompt×2个seed生成6张clean探测RGB，共36张。使用冻结阶段1严格图路径匹配，对原始Canny全部候选计算保留率；至少5/6成功、每个prompt至少1/2成功、锚点合格视图至少5/6、成功匹配对称距离中位数≤3px，才通过稳定性筛选。','',
           '两种方法都按源曲线长度递减、curve_index打破并列，选择第一个通过原九状态输入准入的候选；稳定方法在此之前增加上述过滤。原方法逐像素复现旧载体和条件图。无载体内容不换图、不删分母：四个消息计擦除，身份条件仍保留质量记录。','',
           '两种策略的实际输入四码字均通过原Auto及局部搜索代理头的闭环：各24/24唯一输入状态（每状态在两个评价视图复用）。这是载体准入条件，不是对任意曲线的总体≥99%保证；名为Oracle的旧输入代理头也不等于扩散输出的真实对应Oracle。','',
           '评价使用另外两组prompt/seed；与探测prompt和seed均无交集。每方法6内容×2视图×4消息=48次marked，另有12次clean几何观测；两种方法共享同一clean RGB。相同内容/视图的两种载体、四消息均共享prompt、seed和初始噪声。选曲线及条件图在第一张评价RGB前冻结。','',
           f'实际生成36张探测RGB和{len(read(OUT/"eval_plan.json"))}张唯一评价RGB；两种方法相同条件只生成一次，不重复计算生成工作量。','',
           'S/T基函数与四码字、最大组合位移1.5px、离散像素编辑规则、SD1.5+ControlNet Canny及20步/CFG5/control1保持不变。主要接收器是冻结4B联合拟合、信息门槛和旧全局信道评分；管道/矩形Mask自参考使用冻结4C0规则和Cal评分模型。所有接收器无微调、无新评分拟合、无新阈值搜索。原曲线参考辅助仍使用原几何、clean输入系数和全部四个源栅格码字，未传入真实消息。','',
           '## 全样本恢复与存在性','',
           '| 载体选择 | 接收器 | 正确 | 错误 | 擦除 | 完整2-bit正确率 | Clean-vs-marked AUC | 冻结阈值检出/48 | clean误接收/12 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for policy in POLICIES:
        for method in METHODS:
            r=summary[policy][method]
            lines.append(f'| {policy} | {method} | {r["correct"]} | {r["wrong"]} | {r["erasures"]} | {r["full_accuracy"]:.1%} | {r["presence_AUC"]:.3f} | {r["marked_presence_accepts"]} | {r["clean_presence_accepts"]} |')
    lines += ['', '存在性使用既有冻结Cal阈值。这里只报告12张clean上的实际接收数，不将其称为经过认证的固定5%误报率；12个clean视图来自6个内容，不能证明低误报性能。四码字全部合法，错误Key标签置换仍不提供认证。','',
              '## 逐内容的探测与留出结果','',
              '| 内容 | 稳定候选/全部 | 原曲线长度 | 稳定曲线长度 | 原/稳定探测成功 | 原/稳定留出clean严格匹配 | 原/稳定正确消息（各8） |',
              '|---|---:|---:|---:|---|---|---|']
    for r in diagnostic['contents']:
        a=r['policies']['source_only'];b=r['policies']['clean_stable']
        lines.append(f'| {r["content_id"]:012d} | {r["stable_candidates"]}/{r["source_candidates"]} | {a["length_px"]:.1f} | '+(f'{b["length_px"]:.1f} | {a["probe_matches"]}/6 / {b["probe_matches"]}/6 | {a["clean_strict_matches"]}/2 / {b["clean_strict_matches"]}/2 | {a["marked_correct"]}/8 / {b["marked_correct"]}/8 |' if b['available'] else f'无 | {a["probe_matches"]}/6 / 无 | {a["clean_strict_matches"]}/2 / 无 | {a["marked_correct"]}/8 / 0/8 |'))
    lines += ['', '稳定性筛选会改变曲线长度与位置分布；配对实验测量整个筛选策略的作用，不证明相同曲线仅因被贴上“稳定”标签而改善。clean轮廓匹配与S/T消息可传输性分别报告。','',
              '## 质量','', '| 选择 | 四项联合通过/48 | 正确恢复且质量通过/48 | PSNR中位数 | SSIM中位数 | LPIPS中位数 | Edge-F1中位数 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for policy in POLICIES:
        q=summary[policy]['quality'];m=q['medians']
        lines.append(f'| {policy} | {q["joint_pass"]} | {q["correct_and_quality_pass"]} | {m["psnr_db"]:.2f} | {m["ssim"]:.4f} | {m["lpips_alex"]:.5f} | {m["edge_f1_3x3"]:.4f} |')
    lines += ['', '冻结要求：PSNR≥30dB、SSIM≥0.90、LPIPS-Alex≤0.10、配对RGB Canny Edge-F1≥0.95（原3×3容差）。所有marked包括擦除和错误均计入。','',
              '## 本批观察','',
              f'参考辅助有效接收由{source["received"]}/48变为{stable["received"]}/48；错误消息由{source["wrong"]}变为{stable["wrong"]}，擦除由{source["erasures"]}变为{stable["erasures"]}。完整恢复率变化{gain/48:+.1%}，存在性AUC变化{stable["presence_AUC"]-source["presence_AUC"]:+.3f}。','',
              f'留出clean严格曲线匹配由{summary["source_only"]["strict_clean"]["matched"]}/12变为{summary["clean_stable"]["strict_clean"]["matched"]}/12；marked严格匹配由{summary["source_only"]["strict_marked"]["matched"]}/48变为{summary["clean_stable"]["strict_marked"]["matched"]}/48。严格路径匹配与4B有效接收是不同指标，后者允许可信残余片段。','',
              f'质量联合通过由{summary["source_only"]["quality"]["joint_pass"]}/48变为{summary["clean_stable"]["quality"]["joint_pass"]}/48；同时正确恢复且质量通过由{summary["source_only"]["quality"]["correct_and_quality_pass"]}/48变为{summary["clean_stable"]["quality"]["correct_and_quality_pass"]}/48。','',
              '筛选所用clean探测的系数接收及离散程度另见probe_coefficient_diagnostics.json。该统计在选曲线冻结后计算，受筛选偏差影响，只作描述，不作为留出验证或新门槛。','',
              '## 结论边界','',
              f'参考辅助正确率变化为{gain/48:+.1%}；按内容配对bootstrap95%区间为{paired[0]["content_bootstrap95"][0]:+.1%}至{paired[0]["content_bootstrap95"][1]:+.1%}。仅6个既有Cal内容，区间只反映本批固定判决的内容重采样，不能支撑强泛化结论。','',
              '若clean轮廓定位改善但消息未改善，说明稳定定位仍不足以保证细小S/T形变传输；若两者都改善，只说明该固定筛选策略在本批内容上有帮助。无论结果如何，不依据评价结果回调筛选门槛或替换曲线。没有独立正确对应标注，因此不能把候选门控失败解释成几何信息彻底丢失。','',
              '## 逐图与原始记录','',
              '- [交互逐图页面](index.html)：48组配对面板，实际条件Canny、生成RGB、输出Canny。',
              '- [汇总](summary.json)、[逐内容恢复](per_content.json)、[载体与失败诊断](carrier_diagnostics.json)、[配对区间](paired_bootstrap.json)。',
              '- [冻结协议](protocol.json)、[筛选结果](selection.json)、[选曲线冻结及哈希](selection_freeze.json)、[审计](audit.json)。',
              '- candidates/*_probe_matches.json 保存全部候选逐探测图匹配结果；features/ 保存每次候选池、拟合系数、拒绝条件和消息判决；quality/ 保存全部质量记录。']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def extra_audit():
    p=read(OUT/'protocol.json');logical=read(OUT/'logical_plan.json');counts=collections.Counter()
    for cid in sorted({r['content_id'] for r in logical}):
        for view in range(2):
            group=[r for r in logical if r['content_id']==cid and r['view']==view]
            records=[read(OUT/'records'/f'{r["job"]}.json') for r in group]
            assert len({r['prompt'] for r in records})==len({r['seed'] for r in records})==len({r['noise_sha256'] for r in records})==1
            counts['paired_noise_groups']+=1
    for path in (OUT/'features').glob('*.json'):
        r=read(path);edge=cv2.Canny(np.array(Image.open(r['rgb']).convert('RGB')),100,200)
        for q in r['raw'].get('pool',[]):
            xy=np.array(q['observed_xy']);assert np.allclose(xy,np.round(xy))
            assert np.all(edge[xy[:,1].astype(int),xy[:,0].astype(int)]>0)
            counts['source_receiver_actual_observed_pixels']+=len(xy)
    auditrow=read(OUT/'audit.json');auditrow['extra_counts']=dict(counts)
    auditrow['report_script_sha256']=digest(ROOT/'report_st_stable_carrier_compare.py')
    dump(OUT/'audit.json',auditrow)


def main():
    shutil.copyfile(ROOT/'report_st_stable_carrier_compare.py',OUT/'source/report_st_stable_carrier_compare.py')
    diagnostics();figures();viewer();report();audit();extra_audit()
    files=[p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'logs' not in p.parts and p.name!='final_artifacts_sha256.json']
    dump(OUT/'final_artifacts_sha256.json',{str(p.relative_to(OUT)):digest(p) for p in sorted(files)})
    print('REPORT_COMPLETE',len(files),'artifacts',flush=True)


if __name__=='__main__':main()
