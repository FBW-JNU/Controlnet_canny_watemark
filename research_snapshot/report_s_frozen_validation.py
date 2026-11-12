"""Chinese report and complete viewer for the frozen S validation."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mat_transport_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment_s_frozen_validation import *
from curve_broad_peak_code import basis
from PIL import ImageDraw,ImageFont
import html

def frac(a,b):return f'{a}/{b} ({100*a/b:.1f}%)' if b else '0/0'
def fnum(x):return '—' if x is None else f'{x:.3f}'
def font(size):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)

def old():
    f=check();oldselection=read(PREV/'runtime_selection.json');oldplan={j['job']:j for j in read(PREV/'runtime_plan.json')};bykey={c['key']:c for c in oldselection}
    dump(OUT/'S_codebooks_frozen.json',dict(parent_selection_sha256=digest(PREV/'runtime_selection.json'),freeze_sha256=digest(OUT/'freeze.json'),
        carrier_codebooks=[{k:c[k] for k in ['key','content_id','samples','split']}|dict(S=c['codebooks']['S']) for c in oldselection],
        score_models=f['score_models'],note='Actual S-only books copied unchanged. Input-unavailable carriers retained with null book. Fresh books are source-specific and fixed in input_freeze before any Fresh RGB.'))
    diagnosis=read(OUT/'old_erasure_diagnosis.json');rr=[r for r in diagnosis['records'] if r['reason']=='trusted_correspondence_rejected']
    fig,axes=plt.subplots(len(rr),3,figsize=(12,len(rr)*2.3))
    for row,r in enumerate(rr):
        j=oldplan[r['job']];c=bykey[j['key']];p,u,n,_=geometry(c['samples']);lo=p.min(0)-10;hi=p.max(0)+11;rgb=np.array(Image.open(j['rgb']).convert('RGB'));edge=cv2.Canny(rgb,100,200);best=r['best_candidate']
        axes[row,0].imshow(arr(j['hint']),cmap='gray',vmin=0,vmax=255);axes[row,0].set_title(j['key']+f' v{j["view"]} {j["arm"]}',fontsize=9)
        axes[row,1].imshow(edge,cmap='gray',vmin=0,vmax=255);axes[row,1].plot(p[:,0],p[:,1],c='#19aaff',lw=.8)
        if best:
            xy=np.array(best['observed_xy']);axes[row,1].scatter(xy[:,0],xy[:,1],s=7,c='#ffae31');ii=np.array(best['indices']);axes[row,2].scatter(u[ii],best['normal_positions'],s=12,c='#bf791e');axes[row,2].plot(u,best['offset']+basis(u)@best['coefficient'],c='purple')
        axes[row,1].set_title('4px support='+f'{r["nearby_support_4px"]:.1%}; rejected: '+','.join(r['failed_checks']),fontsize=8)
        for col in [0,1]:axes[row,col].set(xlim=(lo[0],hi[0]),ylim=(hi[1],lo[1]));axes[row,col].axis('off')
        axes[row,2].set(xlim=(0,1),ylim=(-6,6),xlabel='t',ylabel='Normal position(px)');axes[row,2].axhline(0,c='gray',lw=.5)
    fig.suptitle('6 output-only erasures: actual observations exist, target identity unconfirmed; gates unchanged',fontsize=11);fig.tight_layout(rect=(0,0,1,.98));fig.savefig(OUT/'diagnostics/old_output_erasures.jpg',dpi=140);plt.close(fig)
    q=read(OUT/'old_quality_diagnosis.json')['failures'];tile=300;canvas=Image.new('RGB',(tile*4,(tile+65)*len(q)+45),'white');draw=ImageDraw.Draw(canvas)
    for row,r in enumerate(q):
        j=oldplan[r['job']];c=bykey[j['key']];a=np.array(Image.open(j['clean_rgb']).convert('RGB'));b=np.array(Image.open(j['rgb']).convert('RGB'));aa=cv2.Canny(a,100,200)>0;bb=cv2.Canny(b,100,200)>0
        ad=cv2.dilate(aa.astype('uint8'),np.ones((3,3),np.uint8))>0;bd=cv2.dilate(bb.astype('uint8'),np.ones((3,3),np.uint8))>0;over=np.zeros_like(a);over[(aa&bd)|(bb&ad)]=255;over[aa&~bd]=[255,75,75];over[bb&~ad]=[60,235,75]
        contours,_=cv2.findContours(arr(c['mask']),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(over,contours,-1,(40,150,255),1)
        images=[a,b,over,np.clip(np.abs(a.astype(float)-b)*8,0,255).astype('uint8')]
        for col,(label,img) in enumerate(zip(['paired clean RGB','marked RGB','Canny support red/green','RGB abs difference x8'],images)):
            canvas.paste(Image.fromarray(img).resize((tile,tile)),(col*tile,row*(tile+65)+45));draw.text((col*tile+6,row*(tile+65)+20),label,font=font(13),fill='black')
        text=f'{r["key"]} v{r["view"]} {r["arm"]}: EdgeF1={r["quality"]["edge_f1_3x3"]:.3f}; unmatched outside near-mask={r["unmatched_outside_near_fraction"]:.1%}'
        draw.text((6,(row+1)*(tile+65)-12),text,font=font(13),fill='black')
    canvas.save(OUT/'diagnostics/old_quality_failures.jpg',quality=93)
    rows=read(OUT/'offline_code_results.json');fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for col,decoder in enumerate(['unique_nearest','bounded_unique']):
        for name in ['repeat8','RM(1,3)']:
            rr=[r for r in rows if r['code']==name and r['decoder']==decoder and abs(r['requested_erasure_rate']-14/48)<1e-9];axes[col].plot([r['correlation'] for r in rr],[r['correct'] for r in rr],marker='o',label=name)
        axes[col].set(xlabel='Within-block state correlation',ylabel='Complete message correct probability',ylim=(0,1.02),title=decoder);axes[col].legend()
    fig.suptitle('Synthetic symbols only: repeat8=1bit, RM(1,3)=4bits; base31 correct/3 wrong/14 erased');fig.tight_layout();fig.savefig(OUT/'plots/offline_codes.png',dpi=160);plt.close(fig)
    print('OLD_PANELS_COMPLETE',flush=True)

def fresh_panels():
    cc=read(OUT/'selection.json');plan=read(OUT/'plan.json');decoded=read(OUT/'decoded.json');byjob={(r['job'],r['head'],r['method']):r for r in decoded};lookup={(j['key'],j['view'],j['arm']):j for j in plan};viewer=[]
    for c in cc:
        p,u,n,_=geometry(c['samples']);lo=np.maximum(0,p.min(0)-12);hi=np.minimum(512,p.max(0)+13)
        for view in range(4):
            name=f'{c["key"]}_v{view}.jpg';dest=OUT/'panels'/name
            if not dest.exists():
                fig,axes=plt.subplots(4,3,figsize=(10,10),gridspec_kw={'height_ratios':[1,1.5,1,1]})
                for col,arm in enumerate(['clean','bit0','bit1']):
                    j=lookup[c['key'],view,arm];d=byjob[j['job'],'Auto','affine'];obs=read(OUT/'measurements'/f'{j["job"]}.json')['observations'];rgb=np.array(Image.open(j['rgb']).convert('RGB'));edge=cv2.Canny(rgb,100,200);r=obs['Auto'] if obs['Auto']['accepted'] else obs['Oracle']
                    axes[0,col].imshow(arr(j['hint']),cmap='gray',vmin=0,vmax=255);axes[0,col].set_title(arm+' actual input Canny')
                    axes[1,col].imshow(rgb);axes[1,col].plot(p[:,0],p[:,1],c='#16aaff',lw=.7);axes[1,col].axis('off')
                    axes[2,col].imshow(edge,cmap='gray',vmin=0,vmax=255);axes[2,col].plot(p[:,0],p[:,1],c='#16aaff',lw=.7)
                    if r.get('observed_xy'):
                        xy=np.array(r['observed_xy']);ii=np.array(r['indices']);color='#35ba44' if obs['Auto']['accepted'] else '#f4a532';axes[2,col].scatter(xy[:,0],xy[:,1],s=5,c=color)
                        axes[3,col].scatter(u[ii],r['normal_positions'],s=7,c=color);axes[3,col].plot(u,r['offset']+basis(u)@r['coefficient'],c='purple',lw=.9)
                    axes[2,col].set_title(f'Auto geom={obs["Auto"]["accepted"]}, coded={d["accepted"]}\nbit={d["bit"]}, score={d["score"]:.2f}, presence={d["s_presence"]:.2f}',fontsize=8)
                    for row in [0,2]:axes[row,col].set(xlim=(lo[0],hi[0]),ylim=(hi[1],lo[1]));axes[row,col].axis('off')
                    ir=next(r for r in read(OUT/'inputs.json') if r['key']==c['key'] and r['arm']==arm)['measurements']['Oracle']
                    if ir.get('coefficient') is not None:axes[3,col].plot(u,ir['offset']+basis(u)@ir['coefficient'],c='black',ls='--',lw=.8)
                    axes[3,col].axhline(0,c='gray',lw=.5);axes[3,col].set(xlim=(0,1),ylim=(-7,7),xlabel='t',ylabel='Normal position(px)')
                j=lookup[c['key'],view,'clean'];fig.suptitle(f'{c["key"]} | {c["stratum"]} | {c["arc_length_px"]:.1f}px | p{j["prompt_index"]} seed{j["seed"]}\n'+j['prompt'][:130],fontsize=10)
                fig.tight_layout(rect=(0,0,1,.95));fig.savefig(dest,dpi=110);plt.close(fig)
            j=lookup[c['key'],view,'clean'];links=[dict(arm=a,hint=str(Path(lookup[c['key'],view,a]['hint']).relative_to(OUT)),rgb=str(Path(lookup[c['key'],view,a]['rgb']).relative_to(OUT))) for a in ['clean','bit0','bit1']]
            viewer.append(dict(key=c['key'],view=view,stratum=c['stratum'],prompt=j['prompt'],seed=j['seed'],panel='panels/'+name,links=links,mask=str(Path(c['mask']).relative_to(OUT))))
        print('FRESH_PANEL',c['key'],flush=True)
    dump(OUT/'viewer_data.json',viewer)

def report():
    f=check();s=read(OUT/'summary.json');q=read(OUT/'quality.json');cap=read(OUT/'capacity_summary.json');boot=read(OUT/'bootstrap.json');oldinput=read(OUT/'old_input_closure.json');newinput=read(OUT/'new_input_closure.json');diag=read(OUT/'old_erasure_diagnosis.json');qualityold=read(OUT/'old_quality_diagnosis.json');selection=read(OUT/'selection.json');neg=read(OUT/'negative_summary.json')
    a=s['Auto_affine'];o=s['Oracle_affine'];acceptance=dict(input99=newinput['summary']['all_Auto']['pass99'],all_attempt_accuracy80=a['all_attempt_accuracy']>=.8,
        bit_auc80=a['bit_auc']>=.8,presence_auc70=a['presence_auc']>=.7,all_quality_pass=q['joint_passes']==q['attempts'])
    dump(OUT/'acceptance.json',dict(gates=acceptance,all_pass=all(acceptance.values()),frozen_before_test=True,no_real_multicarrier_generation=True))
    oldfails=qualityold['failures'];lengths=[c['arc_length_px'] for c in selection];lb=[c['count'] for c in cap['independent_lower_bounds']]
    fig,axes=plt.subplots(1,2,figsize=(10,3.5));axes[0].hist(lb,bins=12,color='#4c799c');axes[0].axvline(8,c='gray',ls='--');axes[0].axvline(32,c='gray',ls=':');axes[0].set(xlabel='Source-only isolated writable carriers (greedy lower bound)',ylabel='Contents')
    axes[1].scatter([r['arc_length_px'] for r in read(OUT/'per_content.json')],[r['all_attempt_accuracy'] for r in read(OUT/'per_content.json')]);axes[1].axhline(.8,c='gray',ls='--');axes[1].set(xlabel='Selected source arc length(px)',ylabel='All-attempt bit accuracy',ylim=(-.03,1.03));fig.tight_layout();fig.savefig(OUT/'plots/capacity_and_accuracy.png',dpi=160);plt.close(fig)
    lines=['# 冻结 S 形编码：输入闭环、擦除诊断、Fresh24 独立验收与离线码设计','',
        f'Fresh24 自动链路：输入恢复48/48；RGB 全样本正确 {a["correct"]}/{a["marked"]}={a["all_attempt_accuracy"]:.1%}，错误{a["wrong"]}，擦除{a["erasures"]}；Bit AUC={a["bit_auc"]:.3f}；存在性 AUC={a["presence_auc"]:.3f}；四项质量联合通过{q["joint_passes"]}/{q["attempts"]}。验收全项通过：{all(acceptance.values())}。','',
        '## 冻结范围','',
        'S(t)=6√3·t(1−t)(2t−1)，幅度固定±1.5px，端点位移0。沿冻结的弧长参数与法向编辑，端点按字典序规范方向，采用原二值像素移动/四舍五入/8邻接重连规则，mask外不变。未更换多项式、增大幅度、做幅度搜索、训练模型或启动真实8段/32段编码。','',
        '本次接收器严格继承上轮产生31/48的 S 分支：R0优先、A3回退及原2c几何门槛，联合Huber拟合 offset+aS·S+aT·T。T仅为既有接收器的拟合自由度，编码器仍只写S；没有恢复到缺失历史目录中的另一版单系数接收器。特征、bit阈值、存在性模型均沿用上轮6个Cal内容的S模型，不用Fresh24重新拟合。','',
        'freeze.json记录实现与模型哈希；S_codebooks_frozen.json复制原18个载体的S码本，包含不可用码本；新内容的源码本在input_freeze.json中于RGB生成前冻结。analysis_freeze.json和offline_freeze.json分别保护评估和模拟。','',
        '## A：实际 Canny 输入闭环','',
        '|输入集合|唯一标记状态|Auto几何通过|Auto完整恢复|Oracle完整恢复|','|---|---:|---:|---:|---:|']
    for label,obj in [('既有Cal+诊断',oldinput),('Fresh24选定载体',newinput)]:
        x=obj['summary']['all_Auto'];y=obj['summary']['all_Oracle'];lines.append(f'|{label}|{x["attempts"]}|{frac(x["geometry_received"],x["attempts"])}|{frac(x["correct"],x["attempts"])}|{frac(y["correct"],y["attempts"])}|')
    lines+=['','逐张重读实际有效PNG，再由冻结的盲消息接收器提取系数和判bit。零修改、无双消息码本及接收失败均计错，不按prompt/seed复制输入分母。既有S输入端Auto为32/36，未达到99%；不能追溯性宣称原编码链路闭环成功。','',
        f'Fresh24采用预先冻结的输入端合格规则：clean和两个消息均可测、实际像素非零修改、码本可分并经冻结判决恢复；每内容从长到短取第一个合格载体。所有24个内容都保留，不因难编码或输出失败换图。原始{cap["source_candidates"]}条源候选中，{cap["eligible"]}条满足完整输入端条件。选定24条的48个输入状态恢复100%，达到本轮观测门槛；这是筛选后可写集合的闭环，不是对任意曲线的99%保证。48次全对也不足以统计证明总体成功率≥99%。','',
        f'新选定曲线长 {min(lengths):.1f}–{max(lengths):.1f}px，中位数{np.median(lengths):.1f}px；上轮大量诊断曲线较短。载体选择只读取原图Canny，不增加clean探测生成，也不按独立RGB稳定性重筛，因此不能把前后结果差异都归因于接收器。','',
        '## A：14次擦除与3次错误','',
        '|根因|次数|处理|','|---|---:|---|',
        '|源端双bit码本不可用|8/48|保留擦除；这是编码准入问题，不能用输出匹配放宽补救|',
        '|有输入码本，但输出几何置信度未通过|6/48|保持拒绝；审查残差、平移、锚点和错误对应|',
        '|正确匹配后因判决裕量主动擦除|0/48|当前冻结判决器没有此种擦除规则|','',
        '6个纯输出擦除的4px邻近支持为82.8%–100%；Oracle也均给出观测，但强制采用Oracle bit后只有3/6正确。高支持不能证明目标载体身份，也不能证明水印几何可恢复。既有8个输入码本擦除若用Oracle，仅6/8正确。两组均不能直接无条件回收。','',
        '三个已接收但判错样本中，仅1个拟合系数的95%投影区间覆盖bit判决边界，另两个在拟合内有较高置信度却仍判错。该区间只描述拟合误差，不包括错配与信道系统噪声，因此不能视作校准后的bit可靠度。未新增裕量阈值或强行改bit。','',
        '[查看六个输出擦除的局部观测](diagnostics/old_output_erasures.jpg)。橙点是被拒绝的候选，不代表正确恢复。完整17个擦除/错误案例保留于old_erasure_diagnosis.json与old_wrong_bit_margins.json。','',
        '## B：24个新内容独立验收','',
        '从筛选后的COCO Train normal总体均匀抽取24个新内容；排除所有现存输出路径和计划/选择元数据中可识别的历史内容。每内容2个prompt×2个seed×clean/bit0/bit1，共288张真实RGB、192个marked尝试和96个clean。SD1.5+ControlNet Canny使用原20步DDIM、CFG5、eta0、control1、float32配置；同一prompt/seed三组共享初始噪声。','',
        '独立性限制：旧Stage2三个目录仍缺失，无法枚举其全部历史内容或重跑原固定60负例。本轮与所有当前可核验的历史内容不重叠；无法对缺失名单提供完整排除证明。未打开、调整或扩充原封存审计集。所有Fresh结果只用于评价，没有再次调节参数。','',
        '|接收器/判决|几何接收|有效bit接收|全样本正确|错误|擦除|Bit AUC|存在性AUC|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for head in ['Auto','Oracle']:
        for method in ['raw','affine']:
            r=s[head+'_'+method];lines.append(f'|{head}/{method}|{r["geometry_received"]}/{r["marked"]}|{r["received"]}/{r["marked"]}|{frac(r["correct"],r["marked"])}|{r["wrong"]}|{r["erasures"]}|{r["bit_auc"]:.3f}|{fnum(r["presence_auc"]) if method=="affine" else "—"}|')
    lines+=['','主要结果是Cal正增益校准后的affine；raw仅为bit消融，没有定义raw存在性检测器。所有bit AUC包含擦除，擦除分数中性0；正确率中擦除计错。Oracle只提供位置和较宽的局部匹配条件，不提供正确对应点，不保证更可靠。全体内容使用相同的两个新seed及两个prompt模板，支持对这组冻结生成条件的跨内容验证，不能据此声称覆盖广泛prompt/seed分布。','',
        '|指标|Auto的内容bootstrap 95%区间|Oracle区间|','|---|---:|---:|']
    for key in ['all_attempt_accuracy','bit_auc','presence_auc']:
        x=boot['intervals']['Auto'][key];y=boot['intervals']['Oracle'][key];lines.append(f'|{key}|[{x[0]:.3f}, {x[1]:.3f}]|[{y[0]:.3f}, {y[1]:.3f}]|')
    lines+=['','按24个内容整组重采样2000次，保留同内容的prompt/seed/消息相关性，不能把192个消息当192个独立内容。逐内容统计见per_content.json。','',
        '|接收器|冻结Cal阈值下clean接受|marked接受|','|---|---:|---:|']
    for h in ['Auto','Oracle']:
        r=s[h+'_affine'];lines.append(f'|{h}|{frac(r["clean_presence_accepted"],r["clean"])}|{frac(r["marked_presence_accepted"],r["marked"])}|')
    lines+=['','存在性阈值仍取原Cal clean评分的95分位，Cal只有12张clean；实际Fresh clean接受率照实报告，不宣称达到1%或family-wise FPR。bit可分性与水印存在性分开验收。','',
        '## 质量约束与旧四个失败','',
        f'新192个marked中，四项联合通过{frac(q["joint_passes"],q["attempts"])}。门槛始终是相对同prompt/seed clean的PSNR≥30dB、SSIM≥0.90、LPIPS≤0.10、Edge-F1(3×3)≥0.95，不剔除失败样本，也不针对失败修改Canny阈值。','',
        '|质量项|新marked失败数|','|---|---:|']
    for key in ['psnr','ssim','lpips','edge_f1']:lines.append(f'|{key}|{sum(not r["gates"][key] for r in q["records"])}|')
    lines+=['','|新质量失败的内容/视图|消息|失败项|','|---|---|---|']
    for r in q['records']:
        if not r['pass_all']:
            lines.append(f'|{r["key"]}/view{r["view"]}|{r["arm"]}|'+', '.join(k for k,v in r['gates'].items() if not v)+'|')
    lines+=['','旧4个失败均为Edge-F1，PSNR/SSIM/LPIPS全部通过；分别来自071137/view1和494199/view0的bit0/bit1。前者直线约80.1px，后者弯曲约27.8px，prompt风格也不同，未发现“全是同一种曲率/形变”证据。','',
        '95.5%–99.0%的未对应边缘位于编辑mask及其8px邻域之外，说明全图边缘变化是主要组成；但通过样本的区域外比例也高，不能仅凭此比例证明失败原因。失败样本未对应边缘数中位数1877.5，通过样本749。局部Canny编辑可经扩散影响远处纹理，低幅RGB差异也可能改变Canny阈值结果；这是需后续Cal检验的解释，未据此放宽质量要求。','',
        '[查看质量四例](diagnostics/old_quality_failures.jpg)：白色为3×3范围内双方支持，红色为clean有而marked无支持，绿色为marked新增且clean无支持；蓝线标固定mask。这与接收器面板中的绿色“几何通过点”含义不同。','',
        '## 曲线容量与错误对应负例','',
        f'每图满足完整S输入条件且空间隔离的贪心载体数：范围{min(lb)}–{max(lb)}，中位数{np.median(lb):.1f}；≥8条为{cap["at_least8"]}/24，≥32条为{cap["at_least32"]}/24。该计数要求mask间隔并保持曲线距离，属于源端可写候选下界，既不是扩散后稳定容量，也不是统计独立性证明。没有生成真实多载体图像。','',
        '|错位幅度/接收器|几何错误接受|','|---|---:|']
    for k,r in neg.items():lines.append(f'|{k}|{frac(r["accepted"],r["attempts"])}|')
    lines+=['','12px错误锚点要求离源曲线>8px；24px要求>16px。不符合分离条件的查询不计入分母。它们是本轮新的同图相关负对照，不能和旧6/60直接等同，也不是水印存在性FPR。','']
    freshdiag=read(OUT/'fresh_failure_diagnosis.json');pair=freshdiag['pair_summary']
    lines+=[ '## Fresh24：信号与失败诊断','',
        f'96组配对中，Auto两个消息均接收{pair["auto_both_received"]}组，其中bit1−bit0分数为正{pair["auto_positive_response"]}组。分数为冻结码本归一化量，不能当成像素位移；像素单位的S系数差另存fresh_failure_diagnosis.json。配对已接收子集有选择偏差，不能取代192次完整链路统计。','',
        '|新擦除的4px邻近支持|次数|Oracle接收|Oracle判bit正确|','|---|---:|---:|---:|']
    for r in freshdiag['erasure_nearby_support_buckets']:
        lines.append(f'|{"≥80%" if r["support_4px_at_least80"] else "<80%"}|{r["n"]}|{r["oracle_accepted"]}|{r["oracle_correct"]}|')
    lines+=['','低支持不能严格证明原曲线消失，高支持也不能证明对应身份正确。这些分组只用于定位失败环节，未用来恢复bit或修改冻结门槛。每个错误/擦除的门槛失败项、邻近支持与Oracle结果均可查询。','',
        '|Prompt模板|全样本正确|错误|擦除|存在性AUC|','|---|---:|---:|---:|---:|']
    for pi,r in freshdiag['by_prompt'].items():
        lines.append(f'|{pi}|{frac(r["correct"],r["marked"])}|{r["wrong"]}|{r["erasures"]}|{r["presence_auc"]:.3f}|')
    lines+=['','[信号分布图](plots/fresh_signal_diagnostics.png)分别显示接收后的bit分数、全样本存在性评分以及双消息均接收时的成对响应；这些图用于描述独立集，不参与重新校准。','',
        '## C：密钥多载体存在性原型，仅离线','',
        '给第i条载体固定密钥符号k_i∈{−1,+1}，同一消息b的几何符号为(2b−1)k_i。先用冻结、bit无关的单载体接收器产生软分数和擦除，再检测这些观测是否同时满足{key,−key}码字。不得为迎合Key或消息而重新挑选背景轮廓。','',
        '初始软相关检测只用|Σk_i z_i w_i|/√Σw_i。Cal重采样模拟中，clean区分很好，但错误Key接受约54%–60%，说明软相关本身不足以验证归属。新增的离线对照联合检查软相关、有效载体数以及到{key,−key}的最小Hamming不一致数。','',
        '|Cal合成设置（独立抽样）|marked接受|错误Key接受|clean接受|','|---|---:|---:|---:|']
    keyresults=read(OUT/'key_consistency_results.json')['rows']
    for r in keyresults:
        if r['shared_pair_probability']==0 and r['allowed_disagreement']==0:
            lines.append(f'|至少{r["minimum_received"]}/8可接收、零符号不一致|{r["marked_accept"]:.1%}|{r["wrong_key_accept"]:.2%}|{r["clean_accept"]:.2%}|')
    lines+=['','这些数值是仅由12个Cal视图重采样的合成8载体块，不是新的独立RGB证据；小样本重复采样不会增加真实clean数量。所有阈值仅在模拟内展示，未部署到冻结接收器。未知bit下key与−key等价，错误Key还可能与合法码字空间碰撞，因此不能把模拟低接受率当认证安全性或真实1%FPR。','',
        '## 阶段4：符号模拟','',
        '基础暂用31/48正确、3/48错误、14/48擦除。保留接收后错误率3/34，扫描擦除率0.10/0.292/0.45/0.60和块内状态相关性0/0.1/0.3/0.6。相关性用Dirichlet混合通道生成，保持边际概率但改变块内共同状态，不假设已从单曲线数据测得多曲线相关性。','',
        '|基础擦除率、相关性|8次重复：1bit完整恢复|RM(1,3)：4bit完整恢复|','|---|---:|---:|']
    codes=read(OUT/'offline_code_results.json')
    for rho in [0.,.1,.3,.6]:
        rr=[r for r in codes if abs(r['requested_erasure_rate']-14/48)<1e-9 and r['correlation']==rho and r['decoder']=='unique_nearest'];rep=next(r for r in rr if r['code']=='repeat8');rm=next(r for r in rr if r['code']=='RM(1,3)');lines.append(f'|{14/48:.3f}, {rho:.1f}|{rep["correct"]:.1%}|{rm["correct"]:.1%}|')
    lines+=['','两种码每块传递的消息位数不同，不能只按恢复百分比比较效率。RM使用GF(2)仿射多项式产生8个符号，码距4；分别报告唯一最近译码与满足2e+s<4的保守有界译码。保守译码可降低错译并增加擦除。真实相关错误模型尚未建立；源端候选多也不证明真实8段可同时可靠传输。S的实数域几何多项式与有限域RM/RS码多项式分属不同层次，本轮没有实施RS或真实多载体扩散。','',
        '## 验收','', '|冻结门槛|结果|','|---|---|']
    for key,value in acceptance.items():lines.append(f'|{key}|{"通过" if value else "未通过"}|')
    lines+=['','无论本轮排序AUC是否达到探索目标，推进真实多段纠错编码仍需同时满足全样本bit恢复、可靠对应、存在性与质量要求；本次保持S基函数与位移不变，未以放宽阈值来换接收率。','']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines))
    viewer=read(OUT/'viewer_data.json');cards=[]
    for v in viewer:
        links=' · '.join(f'<a href="{l["hint"]}">{l["arm"]} 条件Canny</a> <a href="{l["rgb"]}">RGB</a>' for l in v['links'])
        cards.append(f'<article data-key="{v["key"]}"><h2>{v["key"]} / {v["stratum"]} / view{v["view"]} / seed{v["seed"]}</h2><p>{html.escape(v["prompt"])}</p><a href="{v["panel"]}"><img loading="lazy" src="{v["panel"]}"></a><p>{links} · <a href="{v["mask"]}">Mask</a></p></article>')
    options=''.join(f'<option>{c["key"]}</option>' for c in selection)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>冻结S形：Fresh24独立验证</title><style>body{font:16px/1.6 system-ui;margin:24px;background:#eee;color:#222}article,section{background:white;padding:18px;margin:20px 0;border-radius:8px}img{max-width:100%}h2{font-size:19px}a{color:#1766a7}select{font-size:16px}</style><h1>冻结S形：输入闭环与Fresh24独立验证</h1><p><a href="REPORT_ZH.md">完整中文报告</a> · <a href="acceptance.json">验收</a> · <a href="summary.json">指标</a> · <a href="audit.json">审计</a></p><p>每组同prompt/seed三列clean、bit0、bit1。四行依次为实际条件Canny、生成RGB、输出Canny与观测、法向位置（黑虚线输入拟合；点为实际输出；紫线为输出拟合）。蓝线=源曲线，绿点=Auto几何通过，橙点=被拒绝时的Oracle候选；均不保证bit正确或目标身份。clean列bit为接收器强制输出，不表示携带消息。view0/1为照片prompt的两seed，view2/3为水彩prompt的两seed。</p><section><h2>旧样本诊断与离线模拟</h2><p><a href="diagnostics/old_output_erasures.jpg">六个纯输出擦除</a> · <a href="diagnostics/old_quality_failures.jpg">四个质量失败</a> · <a href="offline_code_results.json">全部纠错模拟</a> · <a href="key_consistency_results.json">密钥一致性模拟</a></p><img src="plots/offline_codes.png"><img src="plots/capacity_and_accuracy.png"></section><p>选择内容：<select id="pick"><option value="all">全部24个内容、96个面板</option>'''+options+'''</select></p>'''+''.join(cards)+'''<script>const pick=document.getElementById('pick');pick.onchange=()=>document.querySelectorAll('article').forEach(a=>a.hidden=pick.value!=='all'&&a.dataset.key!==pick.value);</script></html>'''
    (OUT/'index.html').write_text(page);print('REPORT_COMPLETE',acceptance,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['old','fresh_panels','report']);a=p.parse_args();globals()[a.action]()
