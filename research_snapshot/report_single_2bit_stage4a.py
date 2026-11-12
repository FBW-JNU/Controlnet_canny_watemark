"""Read-only diagnostics and review artifacts for single-curve four words."""
from curve_single_2bit_stage4a import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import html


def rate(n,d):return f'{n}/{d} ({n/max(1,d):.1%})'
def matrix_text(m):return '—' if m is None else '['+'; '.join(', '.join(f'{x:.3f}' for x in row) for row in m)+']'


def diagnostics():
    f=check();plan=read(OUT/'plan.json');cc={c['key']:c for c in read(OUT/'selection.json')};models=read(OUT/'score_freeze.json')['models']
    dd=read(OUT/'decoded.json');lookup={(r['job'],r['head']):r for r in dd};axisrows=[];interference=[];independent=[]
    for head in ['Auto','Oracle']:
        model=models[head];a=np.array(model['A']);b=np.array(model['intercept'])
        for j in plan:
            c=cc[j['key']];d=lookup[j['job'],head]
            if j['arm'] in ['S0','S1','T0','T1']:
                family=j['arm'][0];arms=[family+'0',family+'1'];centers=np.array([a@features(c,s,head)+b for s in arms]) if c['admission']['admissible'] else None
                guess=int(np.argmax(gaussian_log(np.array(d['feature'])-centers,model['residual_covariance']))) if d['accepted'] else None
                axisrows.append(dict(job=j['job'],key=j['key'],split=j['split'],view=j['view'],head=head,arm=j['arm'],family=family,accepted=d['accepted'],bit=guess,correct=d['accepted'] and guess==int(j['arm'][-1])))
            if j['arm'] in WORDS:
                guess=''.join(str(int(v>0)) for v in np.array(d['feature'])-b) if d['accepted'] else None
                independent.append(dict(job=j['job'],head=head,split=j['split'],correct=d['accepted'] and guess==j['arm'][1:],bits=guess))
        axislookup={(r['key'],r['view'],r['arm']):r for r in axisrows if r['head']==head}
        for j in plan:
            if j['arm'] not in WORDS:continue
            d=lookup[j['job'],head]
            for bit,family in enumerate(['S','T']):
                ax=axislookup[j['key'],j['view'],family+j['arm'][1+bit]]
                interference.append(dict(job=j['job'],key=j['key'],split=j['split'],view=j['view'],head=head,family=family,
                    axis_accepted=ax['accepted'],axis_correct=ax['correct'],joint_accepted=d['accepted'],
                    joint_bit_correct=d['accepted'] and d['bits'][bit]==j['arm'][1+bit],
                    axis_correct_but_joint_wrong_or_erased=ax['correct'] and (not d['accepted'] or d['bits'][bit]!=j['arm'][1+bit])))
    summaries={}
    for head in ['Auto','Oracle']:
        summary={}
        for family in ['S','T']:
            rr=[r for r in axisrows if r['head']==head and r['split']=='Eval' and r['family']==family]
            both=[r for r in interference if r['head']==head and r['split']=='Eval' and r['family']==family]
            summary[family]=dict(axis_attempts=len(rr),axis_received=sum(r['accepted'] for r in rr),axis_correct=sum(r['correct'] for r in rr),
                joint_bit_attempts=len(both),joint_bit_correct=sum(r['joint_bit_correct'] for r in both),
                axis_correct_contexts=sum(r['axis_correct'] for r in both),axis_correct_but_joint_wrong_or_erased=sum(r['axis_correct_but_joint_wrong_or_erased'] for r in both))
        rr=[r for r in independent if r['head']==head and r['split']=='Eval'];summary['independent_coefficient_sign']=dict(attempts=len(rr),correct=sum(r['correct'] for r in rr),note='Descriptive raw coordinate-sign comparison, not a fitted competing decoder; no inverse response matrix.')
        summaries[head]=summary
    coverage=[]
    for j in plan:
        c=cc[j['key']];p,u,_,_=geometry(c['samples']);bb=basis(u);stations=np.unique(np.r_[np.arange(0,len(p),2),len(p)-1]).astype(int)
        observations=read(OUT/'measurements'/f'{j["job"]}.json')['observations']
        for head in ['Auto','Oracle']:
            r=observations[head]
            if not r.get('observed_xy'):continue
            ii=np.array(r['indices']);w=np.array(r['weights']);information=np.sum(w[:,None]*bb[ii]**2,axis=0)/np.maximum(1e-9,np.sum(bb[stations]**2,axis=0))
            coverage.append(dict(job=j['job'],split=j['split'],head=head,arm=j['arm'],accepted=r['accepted'],R_S=float(min(1,information[0])),R_T=float(min(1,information[1]))))
    dump(OUT/'basis_information_coverage.json',dict(records=coverage,note='Both basis information coverages are descriptive; existing S gate unchanged.'))
    failures=[]
    for j in plan:
        if j['arm'] not in WORDS:continue
        d=lookup[j['job'],'Auto']
        if d['accepted'] and d['word']==j['arm']:continue
        c=cc[j['key']];rgb=np.array(Image.open(j['rgb']).convert('RGB'));edge=cv2.Canny(rgb,100,200);yy,xx=np.nonzero(edge);distance=cKDTree(np.c_[xx,yy]).query(c['samples'])[0]
        obs=read(OUT/'measurements'/f'{j["job"]}.json')['observations']['Auto'];candidates=obs.get('decisions',[]);best=min(candidates,key=lambda r:sum(not v for v in r['gate_checks'].values())) if candidates else None
        od=lookup[j['job'],'Oracle']
        failures.append(dict(job=j['job'],key=j['key'],split=j['split'],view=j['view'],true_word=j['arm'],decoded=d['word'],
            kind='wrong_word' if d['accepted'] else 'geometry_erasure',support_2px=float(np.mean(distance<=2)),support_4px=float(np.mean(distance<=4)),
            failed_checks=[k for k,v in best['gate_checks'].items() if not v] if best else ['no_fit'],
            oracle_received=od['accepted'],oracle_word=od['word'],oracle_correct=od['accepted'] and od['word']==j['arm']))
    dump(OUT/'failure_diagnosis.json',dict(records=failures,note='Post-freeze descriptive analysis. Nearby support and Oracle do not certify target identity; no rejected sample is forcibly decoded.'))
    dump(OUT/'axis_and_interference.json',dict(summary=summaries,axis_rows=axisrows,paired_contexts=interference,independent_sign=independent,
        interpretation='Same prompt/noise matched arms. Conditioning and receiver rejection also affect comparisons; counts do not prove physical cross-talk alone. No Eval retuning.'))
    # Response uncertainty: resample complete Cal contents, retaining both views.
    fd=read(OUT/'cal_response_matrices.json');intervals={};rng=np.random.default_rng(2026111153)
    discrete={}
    for head in ['Auto','Oracle']:
        inputrows=[]
        for c in cc.values():
            if c['split']!='Cal' or not c['admission']['admissible']:continue
            z={a:features(c,a,head) for a in SIGNS}
            axis=np.column_stack([(z['S1']-z['S0'])/(2*AMPLITUDE),(z['T1']-z['T0'])/(2*AMPLITUDE)])
            joint=np.column_stack([(z['m10']+z['m11']-z['m00']-z['m01'])/(4*AMPLITUDE),(z['m01']+z['m11']-z['m00']-z['m10'])/(4*AMPLITUDE)])
            interaction=(z['m11']-z['m10']-z['m01']+z['m00'])/(4*AMPLITUDE**2)
            inputrows.append(dict(key=c['key'],axis=axis.tolist(),joint=joint.tolist(),bilinear_interaction=interaction.tolist()))
        common=[r for r in fd[head]['rows'] if r['output_axis'] is not None and r['output_corners'] is not None]
        discrete[head]=dict(input_rows=inputrows,input_axis_mean=np.mean([r['axis'] for r in inputrows],axis=0).tolist(),input_joint_mean=np.mean([r['joint'] for r in inputrows],axis=0).tolist(),
            output_common_complete_groups=len(common),output_axis_common_mean=np.mean([r['output_axis']['A'] for r in common],axis=0).tolist() if common else None,
            output_joint_common_mean=np.mean([r['output_corners']['A'] for r in common],axis=0).tolist() if common else None,
            interpretation='Input rasterization can change axis and joint gains. Compare outputs on the same complete groups; all are reception-selected diagnostics, not causal proof of cross-talk.')
    dump(OUT/'cal_discrete_interaction.json',discrete)
    for head in ['Auto','Oracle']:
        intervals[head]={}
        for name in ['output_axis','output_corners']:
            rows=fd[head]['rows'];keys=sorted({r['key'] for r in rows});est=[]
            for _ in range(2000):
                mats=[r[name]['A'] for key in rng.choice(keys,len(keys),replace=True) for r in rows if r['key']==key and r[name] is not None]
                if mats:est.append(np.mean(mats,axis=0))
            intervals[head][name]=dict(valid_bootstraps=len(est),interval95=np.percentile(est,[2.5,97.5],axis=0).tolist() if est else None)
    dump(OUT/'cal_response_uncertainty.json',intervals)
    fig,axes=plt.subplots(2,3,figsize=(11,6))
    for ri,head in enumerate(['Auto','Oracle']):
        for col,name in enumerate(['input','output_axis','output_corners']):
            ax=axes[ri,col];r=fd[head][name]
            if r['mean'] is not None:
                matrix=np.array(r['mean']);ax.imshow(matrix,cmap='coolwarm',vmin=-1,vmax=1)
                for i in range(2):
                    for j in range(2):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center')
            ax.set(xticks=[0,1],xticklabels=['aS','aT'],yticks=[0,1],yticklabels=['cS','cT'],title=f'{head}: {name}\nn={r["complete_groups"]}/12, cond={r["condition"]:.2f}' if r['condition'] is not None else f'{head}: {name}: no complete groups')
    fig.suptitle('Cal finite differences: px observed coefficient / px nominal coefficient; no Eval fitting');fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'plots/cal_response.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));s=read(OUT/'summary.json')
    for col,head in enumerate(['Auto','Oracle']):
        matrix=np.array(s['Eval_'+head]['confusion_matrix']);axes[col].imshow(matrix,cmap='Blues')
        for i in range(4):
            for j in range(5):axes[col].text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()*.6 else 'black')
        axes[col].set(xticks=range(5),xticklabels=['00','01','10','11','erase'],yticks=range(4),yticklabels=['00','01','10','11'],xlabel='Decoded word',ylabel='True word',title=head+' all Eval attempts')
    fig.tight_layout();fig.savefig(OUT/'plots/eval_confusion.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5));u=np.linspace(0,1,1001);bb=basis(u)
    axes[0].plot(u,bb[:,0],label='S');axes[0].plot(u,bb[:,1],label='T');axes[0].legend();axes[0].set(xlabel='Normalized arc length t',ylabel='Unit basis')
    for arm in WORDS:axes[1].plot(u,bb@(AMPLITUDE*np.array(SIGNS[arm])),label=arm[1:])
    axes[1].axhline(1.5,c='gray',ls='--');axes[1].axhline(-1.5,c='gray',ls='--');axes[1].legend();axes[1].set(xlabel='Normalized arc length t',ylabel='Normal displacement(px)');fig.tight_layout();fig.savefig(OUT/'plots/codebook_profiles.png',dpi=160);plt.close(fig)
    print('DIAGNOSTICS_COMPLETE',summaries,flush=True)


def panels():
    plan=read(OUT/'plan.json');cc=read(OUT/'selection.json');dd={(r['job'],r['head']):r for r in read(OUT/'decoded.json')};viewer=[]
    lookup={(j['key'],j['view'],j['arm']):j for j in plan}
    for c in cc:
        p,u,n,_=geometry(c['samples']);lo=np.maximum(0,p.min(0)-12);hi=np.minimum(512,p.max(0)+13)
        for view in range(2):
            dest=OUT/'panels'/f'{c["key"]}_v{view}.jpg'
            if not dest.exists():
                fig,axes=plt.subplots(4,5,figsize=(15,10),gridspec_kw={'height_ratios':[1,1.5,1,1]})
                for col,arm in enumerate(['clean']+WORDS):
                    j=lookup[c['key'],view,arm];d=dd[j['job'],'Auto'];rgb=np.array(Image.open(j['rgb']).convert('RGB'));edge=cv2.Canny(rgb,100,200);obs=read(OUT/'measurements'/f'{j["job"]}.json')['observations'];r=obs['Auto'] if obs['Auto']['accepted'] else obs['Oracle']
                    axes[0,col].imshow(arr(j['hint']),cmap='gray',vmin=0,vmax=255);axes[0,col].set_title(arm+' actual Canny')
                    axes[1,col].imshow(rgb);axes[1,col].plot(p[:,0],p[:,1],c='#17aaff',lw=.6);axes[1,col].axis('off')
                    axes[2,col].imshow(edge,cmap='gray',vmin=0,vmax=255);axes[2,col].plot(p[:,0],p[:,1],c='#17aaff',lw=.6)
                    if r.get('observed_xy'):
                        xy=np.array(r['observed_xy']);ii=np.array(r['indices']);color='#38b34b' if obs['Auto']['accepted'] else '#ee9e2b';axes[2,col].scatter(xy[:,0],xy[:,1],s=4,c=color)
                        axes[3,col].scatter(u[ii],r['normal_positions'],s=5,c=color);axes[3,col].plot(u,r['offset']+basis(u)@r['coefficient'],c='purple',lw=.8)
                    ir=c['states'][arm]['measurements']['Oracle']
                    if ir.get('coefficient') is not None:axes[3,col].plot(u,ir['offset']+basis(u)@ir['coefficient'],c='black',ls='--',lw=.8)
                    presence_text=f'{d["presence_score"]:.2f}' if d['accepted'] else 'rejected'
                    axes[2,col].set_title(f'Auto accepted={d["accepted"]}\nword={d["bits"]}, presence={presence_text}',fontsize=8)
                    for row in [0,2]:axes[row,col].set(xlim=(lo[0],hi[0]),ylim=(hi[1],lo[1]));axes[row,col].axis('off')
                    axes[3,col].axhline(0,c='gray',lw=.5);axes[3,col].set(xlim=(0,1),ylim=(-5,5),xlabel='t',ylabel='Normal position(px)')
                j=lookup[c['key'],view,'clean'];fig.suptitle(f'{c["split"]} {c["key"]} | {c["arc_length_px"]:.1f}px | view{view} seed{j["seed"]}\n'+j['prompt'][:150],fontsize=10)
                fig.tight_layout(rect=(0,0,1,.95));fig.savefig(dest,dpi=110);plt.close(fig)
            j=lookup[c['key'],view,'clean'];viewer.append(dict(key=c['key'],split=c['split'],view=view,panel=str(dest.relative_to(OUT)),prompt=j['prompt'],seed=j['seed'],
                links=[dict(arm=arm,hint=str(Path(lookup[c['key'],view,arm]['hint']).relative_to(OUT)),rgb=str(Path(lookup[c['key'],view,arm]['rgb']).relative_to(OUT))) for arm in SIGNS]))
        print('PANELS',c['key'],flush=True)
    dump(OUT/'viewer_data.json',viewer)


def report():
    f=check();summary=read(OUT/'summary.json');quality=read(OUT/'quality.json')['records'];closure=read(OUT/'input_closure.json');fd=read(OUT/'cal_response_matrices.json');models=read(OUT/'score_freeze.json')['models'];diag=read(OUT/'axis_and_interference.json')['summary'];boot=read(OUT/'bootstrap.json');cc=read(OUT/'selection.json')
    dump(OUT/'effective_generation_config.json',dict(sampler={k:f['generation'][k] for k in ['model_config','checkpoint','checkpoint_sha256','steps','eta','guidance','control_scale','negative_prompt','guess_mode','precision']},
        actual_seeds=SEEDS,actual_rgb=324,protocol_sha256=digest(OUT/'protocol.json'),input_plan_sha256=digest(OUT/'plan.json'),
        note='Inherited generation metadata contains historical seed/count fields. This experiment uses frozen design.seeds, design.total_rgb and plan.json; sampler/model fields stay identical.'))
    lines=['# 单曲线 2-bit：S/T 四码字实验','',
        '本轮研究对象是一条曲线上的两个几何自由度；没有将多条曲线组成纠错码，也没有逻辑拼接、RS或RM。本轮四状态调制是后续几何码字的基础，四状态本身不提供额外纠错冗余。','',
        '## 冻结设计与输入闭环','',
        f'S保持已验证的6√3t(1−t)(2t−1)，新增T=4t(1−t)(5(2t−1)²−1)，为原联合接收器已包含的低频偶对称基函数。连续均匀弧长积分下S与T正交，且均值为0；离散像素、部分观测和扩散后不保证正交。四码字为(aS,aT)∈{{−a,+a}}²，a={AMPLITUDE:.6f}px，组合最大法向位移≤1.5px。一次合成法向位移后栅格化，未连续做两次图像变形。','',
        '1.5px是连续几何设计的位移预算；原栅格器仍进行整数像素四舍五入与8邻接重连，实际离散修改不等于每个像素精确移动该幅度。闭环和信道拟合始终读取实际PNG及其测得系数，不以名义幅度代替实际输入。','',
        '每组包括clean、S0/S1、T0/T1、00/01/10/11九种条件，同prompt/seed共享初始噪声。S-only和T-only使用相同系数幅度a，供串扰与质量比较；a小于此前单S的1.5px，因此不能把前后恢复率差异全部归因于第二基函数。','',
        'Cal为从上一轮24个内容中均匀选出的6个，Eval为12个新COCO normal内容；均为2个视图：照片prompt+seed0、水彩prompt+seed1，合计324张新RGB。Eval使用新seed，内容与当前可核验历史不重叠；缺失历史名单仍限制完整排除证明。prompt与seed在本轮视图中绑定，无法分别估计二者的影响。SD1.5/ControlNet Canny、20步DDIM、CFG5、eta0、control1、float32均沿用冻结配置。有效seed与数量以预先冻结的design及plan.json为准；继承generation对象中的旧seed/数量元数据仅为历史参考，整理配置见effective_generation_config.json。','',
        '每内容从长到短选第一个满足实际输入四码字准入的曲线，不按输出RGB重筛，不更换内容。保留所有尝试记录。接收器沿用既有R0优先/A3回退、联合Huber S/T拟合及2c门槛；候选选择不访问目标消息。现有信息覆盖门槛仍使用S，联合拟合协方差约束两个系数，这对T定位的局限在本轮不作参数修改。','',
        '|实际PNG输入集合|四码字完整恢复|四种像素图均不同|','|---|---:|---:|']
    for split in ['Cal','Eval']:
        rows=[r for r in closure['records'] if r['split']==split]
        for h in ['Auto','Oracle']:
            lines.append(f'|{split}/{h}|{rate(sum(r["heads"][h]["correct"] for r in rows),4*len(rows))}|{sum(r["distinct_four"] for r in rows)}/{len(rows)}条|')
    lines+=['','输入解码器逐张重读实际二值PNG，使用clean输入系数作原点和预先定义的四个理想符号中心进行联合最近邻判决，不是以同一PNG自身作为类别中心来证明100%。但载体准入已经要求该检查通过，因此结果是筛选后可写集合的闭环，不是任意曲线的总体成功率保证。完整输入状态数72，不按prompt/seed重复计算。','',
        '## Cal局部几何响应矩阵','',
        '矩阵行是观测系数(cS,cT)，列是名义输入幅度(aS,aT)。axis用S±和T±作中心有限差分；corners用四角码字平均差分，观察联合调制时的响应。仅统计所需观测均通过接收的配对组，并明确分母；缺失不填0。','']
    for h in ['Auto','Oracle']:
        lines += [f'### {h}','', '|矩阵|完整组数|均值矩阵（按行）|奇异值|条件数|','|---|---:|---|---|---:|']
        for field in ['input','output_axis','output_corners']:
            r=fd[h][field];sv=', '.join(f'{x:.3f}' for x in r['singular_values']) if r['singular_values'] is not None else '—';condition=f'{r["condition"]:.3f}' if r['condition'] is not None else '—'
            lines.append(f'|{field}|{r["complete_groups"]}/{r["attempts"]}|{matrix_text(r["mean"])}|{sv}|{condition}|')
        lines += ['',f'基于实际栅格输入系数、所有Cal可接收状态OLS估计的传输矩阵A={matrix_text(models[h]["A"])}，条件数{models[h]["condition"]:.3f}。它与上表“相对名义幅度”的有限差分矩阵定义不同；输入栅格本身会造成缩放和串扰。','']
    lines+=['按Cal内容整组bootstrap得到的矩阵区间见cal_response_uncertainty.json。样本仅6个Cal内容，完整配对均值也受到接收筛选影响；较好的均值条件数不能保证每条曲线都不退化。','',
        '另外在cal_discrete_interaction.json保留实际输入的axis与joint矩阵、双线性交互项，以及输出仅取axis/joint同时完整组的对比。栅格化本身可能改变联合状态的有效幅度，不能把两个均值之差直接全部归因于ControlNet串扰。','',
        '![Cal响应矩阵](plots/cal_response.png)','',
        '## 冻结四类联合解码','',
        'Cal上拟合二维仿射信道和共享残差协方差；每条曲线已知实际四个输入码字，经冻结信道预测四个输出中心，以Mahalanobis距离联合分类。只读取Cal结果拟合，score_freeze.json写入前禁止生成任何Eval RGB。没有根据Eval旋转符号、调节幅度、重新选曲线、调整阈值或训练生成模型。Auto和Oracle均使用已知源曲线与固定局部搜索范围，本轮不是未知载体的全图盲同步测试；二者比较的是局部对应与接收条件。','',
        '|集合/接收器|完整2-bit正确|错误码字|擦除|有效接收|bit1正确|bit2正确|','|---|---:|---:|---:|---:|---:|---:|']
    for split in ['Cal','Eval']:
        for h in ['Auto','Oracle']:
            r=summary[split+'_'+h];lines.append(f'|{split}/{h}|{rate(r["correct"],r["attempts"])}|{r["wrong"]}|{r["erasures"]}|{rate(r["received"],r["attempts"])}|{rate(r["bits"][0]["correct"],r["attempts"])}|{rate(r["bits"][1]["correct"],r["attempts"])}|')
    lines+=['','Cal为拟合集，不能当作独立性能。全样本完整恢复中，任一bit错误或擦除均记失败；不以两个bit的平均正确率替代完整消息恢复。Oracle只是已知曲线位置下较宽的局部观测，不提供真实对应点。','',
        '|Eval/接收器|完整恢复95%内容bootstrap区间|bit1 AUC|bit2 AUC|','|---|---|---:|---:|']
    for h in ['Auto','Oracle']:
        r=summary['Eval_'+h];interval=boot['intervals'][h]['full_word_accuracy'];lines.append(f'|{h}|{interval[0]:.1%}–{interval[1]:.1%}|{r["bits"][0]["auc"]:.3f}|{r["bits"][1]["auc"]:.3f}|')
    lines+=['','![Eval全部尝试混淆矩阵](plots/eval_confusion.png)','',
        '## 同曲线串扰与单基函数对照','',
        '|Eval接收器/系数|单独调制bit正确|四码字中该bit正确|对应单调制已正确、联合却错/擦除|','|---|---:|---:|---:|']
    for h in ['Auto','Oracle']:
        for family in ['S','T']:
            r=diag[h][family];lines.append(f'|{h}/{family}|{rate(r["axis_correct"],r["axis_attempts"])}|{rate(r["joint_bit_correct"],r["joint_bit_attempts"])}|{r["axis_correct_but_joint_wrong_or_erased"]}/{r["axis_correct_contexts"]}个对应上下文|')
    lines+=['','单基函数解码使用相同Cal信道、同协方差，在对应两个轴码字中判决；联合解码在四类中判决。上下文比较共享噪声和prompt，但也受匹配拒绝、形变叠加及信道噪声影响，不能把所有下降都解释为纯系数串扰。直接对两个输出系数取正负的描述性对照保留于axis_and_interference.json，不替代联合分类。本轮Auto的简单系数符号对照也为79/96，与联合分类的完整正确数相同；本次没有证明联合分类优于该简单对照。','',
        '## Clean存在性与质量','',
        '存在性以Cal拟合的clean二维高斯对四码字混合分布的似然比评分，阈值为Cal clean评分95分位；擦除拒绝。没有依据Eval重新校准。','',
        '|Eval接收器|存在性AUC|clean接受|marked接受|','|---|---:|---:|---:|']
    for h in ['Auto','Oracle']:
        r=summary['Eval_'+h];lines.append(f'|{h}|{r["presence_auc"]:.3f}|{rate(r["clean_presence_accepted"],r["clean_attempts"])}|{rate(r["marked_presence_accepted"],r["attempts"])}|')
    decoded=read(OUT/'decoded.json')
    lines+=['','Eval clean的几何接收数：'+', '.join(f'{h} {sum(r["accepted"] for r in decoded if r["split"]=="Eval" and r["arm"]=="clean" and r["head"]==h)}/24' for h in ['Auto','Oracle'])+'。存在性接受与几何接收是两个不同量。只有24张Eval clean，Cal clean更少，不能证明1%或family-wise误报控制。','',
        '|Eval条件|四项质量联合通过|PSNR失败|SSIM失败|LPIPS失败|Edge-F1失败|','|---|---:|---:|---:|---:|---:|']
    for label,arms in [('S-only',['S0','S1']),('T-only',['T0','T1']),('双bit四码字',WORDS)]:
        rr=[r for r in quality if r['split']=='Eval' and r['arm'] in arms];lines.append(f'|{label}|{rate(sum(r["pass_all"] for r in rr),len(rr))}|'+ '|'.join(str(sum(not r['gates'][g] for r in rr)) for g in ['psnr','ssim','lpips','edge_f1'])+'|')
    lines+=['','|Eval四码字质量失败|未通过项|逐图查看|','|---|---|---|']
    for r in quality:
        if r['split']=='Eval' and r['arm'] in WORDS and not r['pass_all']:
            lines.append(f'|{r["key"]}/view{r["view"]}/{r["arm"]}|'+', '.join(k for k,v in r['gates'].items() if not v)+f'|[面板](panels/{r["key"]}_v{r["view"]}.jpg)|')
    lines+=['','始终相对同prompt/seed的clean：PSNR≥30dB、SSIM≥0.90、LPIPS≤0.10、Edge-F1(3×3)≥0.95。所有marked输出均计算，不排除擦除或质量失败。','']
    r=summary['Eval_Auto'];evalq=[q for q in quality if q['split']=='Eval' and q['arm'] in WORDS]
    acceptance=dict(input99=all(r['heads']['Auto']['correct']==4 for r in closure['records']),eval_full_word70=r['full_word_accuracy']>=.7,
        eval_full_word_accuracy=r['full_word_accuracy'],eval_received=r['received'],eval_attempts=r['attempts'],
        quality_joint_passes=sum(q['pass_all'] for q in evalq),quality_attempts=len(evalq),all_quality_pass=all(q['pass_all'] for q in evalq),
        no_real_error_correction_implemented=True)
    dump(OUT/'acceptance.json',acceptance)
    lines += ['## 探索性结论','',f'冻结联合解码在12个新内容上的完整2-bit恢复：{rate(r["correct"],r["attempts"])}；70%探索目标：{"达到" if acceptance["eval_full_word70"] else "未达到"}。四码字质量联合通过{rate(acceptance["quality_joint_passes"],len(evalq))}。本轮报告可分性与信道响应，不将四状态结果直接解释为已证明的纠错能力或容量。','',
        '全部输入、36组逐图面板、九条件原始PNG、拟合模型、接收细节和质量记录均保留。[失败逐例诊断](failure_diagnosis.json)记录几何门槛失败项、2/4px邻近支持和Oracle判词；[质量记录](quality.json)保留全部条件。邻近支持不保证正确对应，未据此强行恢复消息。结果只用于评价；后续若调整编码或接收规则，应重新冻结并使用新验证数据。','']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines))
    cards=[]
    for v in read(OUT/'viewer_data.json'):
        links=' · '.join(f'<a href="{l["hint"]}">{l["arm"]} Canny</a> <a href="{l["rgb"]}">RGB</a>' for l in v['links'])
        cards.append(f'<article data-split="{v["split"]}"><h2>{v["split"]} {v["key"]} view{v["view"]} seed{v["seed"]}</h2><p>{html.escape(v["prompt"])}</p><a href="{v["panel"]}"><img loading="lazy" src="{v["panel"]}"></a><p>{links}</p></article>')
    (OUT/'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>单曲线2-bit四码字</title><style>body{font:16px/1.6 system-ui;margin:24px;background:#eee}article,section{background:white;padding:18px;margin:20px 0}img{max-width:100%}a{color:#1766a7}h2{font-size:19px}</style><h1>单曲线S/T：四码字</h1><p><a href="REPORT_ZH.md">完整中文报告</a> · <a href="summary.json">统计</a> · <a href="audit.json">审计</a></p><p>五列依次clean、00、01、10、11；四行依次实际条件Canny、生成RGB、输出Canny及局部观测、法向位置。绿点只表示Auto几何接收，不表示消息正确；橙点为Auto拒绝后的Oracle候选，不保证正确对应。蓝线为原曲线，黑虚线为实际输入拟合，紫线为输出拟合。clean列强制输出的码字不表示携带消息。每组下方链接包含S-only/T-only原始对照。</p><section><img src="plots/codebook_profiles.png"><img src="plots/cal_response.png"><img src="plots/eval_confusion.png"></section><p><select id="split"><option value="all">全部</option><option>Cal</option><option>Eval</option></select></p>'''+''.join(cards)+'''<script>document.getElementById('split').onchange=e=>document.querySelectorAll('article').forEach(a=>a.hidden=e.target.value!=='all'&&a.dataset.split!==e.target.value)</script></html>''')
    print('REPORT_COMPLETE',acceptance,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['diagnostics','panels','report']);a=p.parse_args();globals()[a.action]()
