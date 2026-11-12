"""Persist final curve/mask assets and an offline review UI; no rescreening."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree

from canny_curve_stage1_32 import ROOT,OUT,read,dump,sha
from analyze_canny_curve_stage1_32 import METHOD_VERSION,aggregate,curve_match,skeleton_graph,csvout,tangent

def load_edge(case,job):
    rgb=np.asarray(Image.open(job['rgb']).convert('RGB'));edge=cv2.Canny(rgb,100,200)>0
    x0,y0,x1,y1=case['roi'];mask=np.zeros_like(edge);mask[y0:y1,x0:x1]=True
    return edge&mask

def collect(partial=False):
    cv2.setNumThreads(1)
    cases=read(OUT/'selected.json');plan=read(OUT/'plan.json');content_results=[];curve_results=[];flat=[];browser=[]
    all_selected_hashes={}
    for case in cases:
        cid=case['content_id'];jobs=[j for j in plan if j['content_id']==cid]
        if not all((OUT/'metrics'/f'{j["job"]}_matches.json').exists() for j in jobs):
            if partial:continue
            raise RuntimeError(f'Not ready: {cid}')
        selected_file=OUT/'curves'/f'{cid:012d}_selected.json';selection=read(selected_file)
        all_selected_hashes[str(cid)]=sha(selected_file)
        source=read(OUT/'curves'/f'{cid:012d}_source.json');selected=selection['selected'];selected_ids=[r['curve_index'] for r in selected]
        job_data={j['job']:read(OUT/'metrics'/f'{j["job"]}_matches.json') for j in jobs}
        controls={};point_metrics={};paths={}
        for job in jobs:
            edge=load_edge(case,job);graph=None
            output_tree=cKDTree(np.column_stack(np.nonzero(edge))[:,::-1]) if edge.any() else None
            control_path=OUT/'metrics'/f'{job["job"]}_shift64_control.json'
            if job['role']!='discovery':
                if control_path.exists():controls[job['job']]=read(control_path)['matches']
                else:
                    shift=np.zeros_like(edge);shift[64:,64:]=edge[:-64,:-64];shift_graph=skeleton_graph(shift)
                    rows={str(i):dict(curve_match(source['curves'][i]['samples'],shift_graph,read(OUT/'protocol.json')['exploration_thresholds']),curve_index=i,
                                     prompt_index=job['prompt_index'],seed_index=job['seed_index'],role=job['role']) for i in selected_ids}
                    dump(control_path,dict(job=job['job'],shift_xy=[64,64],no_wrap=True,matches=rows));controls[job['job']]=rows
            point_metrics[job['job']]={}
            for index in selected_ids:
                curve=source['curves'][index];r=job_data[job['job']]['matches'][index]
                ds=output_tree.query(curve['samples'])[0] if output_tree is not None else np.full(len(curve['samples']),1e6)
                point_metrics[job['job']][index]=float((ds<=4).mean())
                if r['path_found'] and 'output_path' not in r:
                    if graph is None:graph=skeleton_graph(edge)
                    rr=curve_match(curve['samples'],graph,read(OUT/'protocol.json')['exploration_thresholds'],return_path=True)
                    assert rr['matched']==r['matched'];paths[job['job'],index]=rr.get('output_path',[])
                else:paths[job['job'],index]=r.get('output_path',[])
        viewer_curves=[]
        for selected_row in selected:
            index=selected_row['curve_index'];curve=source['curves'][index]
            rows=[job_data[j['job']]['matches'][index] for j in jobs]
            discovery=aggregate([r for r in rows if r['role']=='discovery'])
            assert discovery==selected_row['discovery']
            heldout=aggregate([r for r in rows if r['role']!='discovery'])
            all12=aggregate(rows)
            subgroups={role:aggregate([r for r in rows if r['role']==role]) for role in ('heldout_seed','heldout_prompt','heldout_joint')}
            shifted=aggregate([controls[j['job']][str(index)] for j in jobs if j['role']!='discovery'])
            point_r=float(np.mean([point_metrics[j['job']][index]>=.8 for j in jobs if j['role']!='discovery']))
            st=tangent(curve['samples']);heading=np.unwrap(np.arctan2(st[:,1],st[:,0]))
            source_geometry=dict(tortuosity=curve['arc_length_px']/max(1e-9,curve['chord_px']),
                total_turn_deg=float(np.rad2deg(heading[-1]-heading[0])),total_absolute_turn_deg=float(np.rad2deg(np.abs(np.diff(heading)).sum())))
            result=dict(content_id=cid,curve_index=index,source_trace_index=curve['primitive_trace_index'],source_arc_length_px=curve['arc_length_px'],source_chord_px=curve['chord_px'],
                        source_geometry=source_geometry,
                        mask=selected_row['mask'],discovery=discovery,heldout=heldout,all12_descriptive=all12,heldout_subgroups=subgroups,
                        heldout_shift64_control=shifted,heldout_point_only_80pct_support_R=point_r,
                        heldout_R_ge_80pct=heldout['R']>=.8,
                        heldout_anchor_rate_ge_80pct=heldout['anchor_view_rate']>=.8)
            curve_results.append(result)
            flat.append(dict(content_id=cid,curve_index=index,source_trace_index=curve['primitive_trace_index'],source_arc_length_px=curve['arc_length_px'],
                source_tortuosity=source_geometry['tortuosity'],source_total_turn_deg=source_geometry['total_turn_deg'],source_total_absolute_turn_deg=source_geometry['total_absolute_turn_deg'],
                discovery_R=discovery['R'],discovery_CI95_low=discovery['R_wilson95'][0],discovery_CI95_high=discovery['R_wilson95'][1],
                discovery_D_median_px=discovery['D']['median_symmetric_px'],heldout_R=heldout['R'],heldout_matched=heldout['matched'],heldout_views=heldout['views'],
                all12_descriptive_R=all12['R'],all12_V_mean_normal_px2=all12['V'].get('mean_normal_profile_variance_px2'),
                heldout_CI95_low=heldout['R_wilson95'][0],heldout_CI95_high=heldout['R_wilson95'][1],
                heldout_seed_R=subgroups['heldout_seed']['R'],heldout_prompt_R=subgroups['heldout_prompt']['R'],heldout_joint_R=subgroups['heldout_joint']['R'],
                heldout_anchor_view_rate=heldout['anchor_view_rate'],heldout_D_median_px=heldout['D']['median_symmetric_px'],
                heldout_V_mean_normal_px2=heldout['V'].get('mean_normal_profile_variance_px2'),
                heldout_V_length_ratio=heldout['V']['feature_variances'].get('length_ratio'),
                heldout_shift64_R=shifted['R'],heldout_point_only_R=point_r,mask=selected_row['mask']))
            viewrows=[]
            for job,row in zip(jobs,rows):
                keys=('matched','path_found','reason','symmetric_median_px','symmetric_p90_px','source_coverage','tangent_good_fraction','length_ratio',
                      'anchor_good_fraction','endpoint_distances_px','signed_normal_mean_px','normal_rms_px')
                viewrows.append(dict(job=job['job'],prompt_index=job['prompt_index'],seed_index=job['seed_index'],role=job['role'],
                    **{k:row[k] for k in keys if k in row},point_support_4px=point_metrics[job['job']][index],output_path=paths[job['job'],index]))
            viewer_curves.append(dict(curve_index=index,samples=curve['samples'],arc_length_px=curve['arc_length_px'],mask=selected_row['mask'],
                discovery=discovery,heldout=heldout,views=viewrows))
        rs=[r for r in curve_results if r['content_id']==cid]
        item=dict(content_id=cid,content_index=case['content_index'],previous_random10=case['previous_random10'],
            source_candidates=len(source['curves']),discovery_selected=len(selected),
            selected_parent_trace_groups=len({r['source_trace_index'] for r in rs}),
            heldout_R_ge_80pct=sum(r['heldout_R_ge_80pct'] for r in rs),
            heldout_mean_R=float(np.mean([r['heldout']['R'] for r in rs])) if rs else None,
            heldout_point_only_mean_R=float(np.mean([r['heldout_point_only_80pct_support_R'] for r in rs])) if rs else None,
            heldout_shift64_mean_R=float(np.mean([r['heldout_shift64_control']['R'] for r in rs])) if rs else None,
            selected_source_arc_px=sum(r['source_arc_length_px'] for r in rs),
            selected_mask_area_px=int((np.asarray(Image.open(OUT/'masks'/f'{cid:012d}_region.png'))>0).sum()),
            roi=case['roi'],selected_file_sha256=sha(selected_file))
        content_results.append(item)
        browser.append(dict(**item,caption=case['prompt'],curves=viewer_curves,
            jobs=[{k:j[k] for k in ('job','prompt_index','seed_index','seed','prompt','role','reused')} for j in jobs]))
        print('REPORT CONTENT',cid,'selected',len(selected),'heldout80',item['heldout_R_ge_80pct'],flush=True)
    finite=[r for r in content_results if r['discovery_selected']]
    summary=dict(status='complete' if len(content_results)==32 else 'partial',contents=len(content_results),planned_contents=32,
        source_candidates=sum(r['source_candidates'] for r in content_results),discovery_selected_curves=len(curve_results),
        selected_parent_trace_groups=sum(r['selected_parent_trace_groups'] for r in content_results),
        discovery_selected_bent_curves_tortuosity_ge_1p05=sum(r['source_geometry']['tortuosity']>=1.05 for r in curve_results),
        heldout80_bent_curves_tortuosity_ge_1p05=sum(r['source_geometry']['tortuosity']>=1.05 and r['heldout_R_ge_80pct'] for r in curve_results),
        contents_with_discovery_candidates=sum(r['discovery_selected']>0 for r in content_results),
        heldout_R_ge_80pct_curves=sum(r['heldout_R_ge_80pct'] for r in curve_results),
        contents_with_heldout_R_ge_80pct=sum(r['heldout_R_ge_80pct']>0 for r in content_results),
        heldout_mean_R_curve_weighted=float(np.mean([r['heldout']['R'] for r in curve_results])) if curve_results else None,
        all12_descriptive_mean_R=float(np.mean([r['all12_descriptive']['R'] for r in curve_results])) if curve_results else None,
        all12_descriptive_per_prompt_R={str(pi):float(np.mean([r['all12_descriptive']['per_prompt'][str(pi)]['R'] for r in curve_results])) if curve_results else None for pi in (0,1,2)},
        heldout_mean_R_content_weighted=float(np.mean([r['heldout_mean_R'] for r in finite])) if finite else None,
        heldout_point_only_mean_R_curve_weighted=float(np.mean([r['heldout_point_only_80pct_support_R'] for r in curve_results])) if curve_results else None,
        heldout_shift64_mean_R_curve_weighted=float(np.mean([r['heldout_shift64_control']['R'] for r in curve_results])) if curve_results else None,
        subgroup_mean_R={role:float(np.mean([r['heldout_subgroups'][role]['R'] for r in curve_results])) if curve_results else None for role in ('heldout_seed','heldout_prompt','heldout_joint')},
        selection_hashes=all_selected_hashes,geometry_edit_transport_tested=False,
        interpretation='Descriptive curve retention in finite heldout views; no codeword or bit capacity inferred. Masks remain discovery-selected, including candidates that fail heldout.')
    if finite:
        rng=np.random.default_rng(20261110)
        a=np.asarray([c['heldout_mean_R']*c['discovery_selected'] for c in finite]);b=np.asarray([c['discovery_selected'] for c in finite])
        samples=rng.integers(0,len(finite),(5000,len(finite)))
        boot=a[samples].sum(1)/b[samples].sum(1)
        summary['heldout_curve_R_content_bootstrap95']=np.quantile(boot,[.025,.975]).tolist()
        summary['bootstrap_method']='5000 resamples of content clusters, RNG20261110; selected curves weighted within each resampled content. Descriptive development-sample interval.'
        summary['discovery_curve_count_min_median_max']=[int(min(b)),float(np.median(b)),int(max(b))]
    summary['content_groups']={}
    for legacy in (True,False):
        group=[c for c in content_results if c['previous_random10']==legacy]
        ids={c['content_id'] for c in group};rs=[r for r in curve_results if r['content_id'] in ids]
        summary['content_groups']['previous10' if legacy else 'new22']=dict(contents=len(group),selected_curves=len(rs),
            heldout_mean_R=float(np.mean([r['heldout']['R'] for r in rs])) if rs else None)
    dvalues=[r['heldout']['D']['median_symmetric_px'] for r in curve_results if r['heldout']['D']['median_symmetric_px'] is not None]
    vvalues=[r['heldout']['V']['mean_normal_profile_variance_px2'] for r in curve_results if r['heldout']['V']['defined']]
    summary['matched_heldout_geometry']=dict(
        D_symmetric_px_median_across_curves=float(np.median(dvalues)) if dvalues else None,
        V_defined_curves=len(vvalues),V_mean_normal_px2_median=float(np.median(vvalues)) if vvalues else None,
        V_mean_normal_px2_p90=float(np.quantile(vvalues,.9)) if vvalues else None,
        note='D/V are conditional on successful heldout matches; V needs at least2 matched views. These are descriptive quantities, not extra screening gates.')
    prefix='partial_' if partial else ''
    dump(OUT/f'{prefix}summary.json',summary);dump(OUT/f'{prefix}content_results.json',content_results);dump(OUT/f'{prefix}curve_results.json',curve_results)
    csvout(OUT/f'{prefix}curve_results.csv',flat);csvout(OUT/f'{prefix}content_results.csv',content_results)
    if partial:
        dump(OUT/'partial_viewer_data.json',dict(cases=browser,summary=summary))
        render(browser,summary,'preview.html')
    else:
        dump(OUT/'viewer_data.json',dict(cases=browser,summary=summary));render(browser,summary);plots(content_results,curve_results);examples(browser);report(summary,content_results)
    print(json.dumps({k:v for k,v in summary.items() if k!='selection_hashes'},ensure_ascii=False,indent=2),flush=True)
    return summary

def render(cases,summary,filename='index.html'):
    data=json.dumps(dict(cases=cases,summary=summary),ensure_ascii=False,separators=(',',':')).replace('<',chr(92)+'u003c')
    doc='''<!doctype html><meta charset="utf-8"><title>Stage1 Train32 曲线稳定性</title>
<style>body{font:16px/1.6 system-ui;margin:20px;background:#eee;color:#222}article{background:white;padding:16px;margin:15px 0}select{padding:5px;margin:5px;max-width:95%;font-size:16px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.tile{position:relative}.tile img{width:100%;display:block}.tile svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.tile svg polyline{fill:none;stroke-width:2;stroke-linejoin:round}.masks{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.masks img{width:100%}code{word-break:break-all}table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:5px 10px}</style>
<h1>Stage1：32个Train内容的曲线级保留与Mask</h1><p id="completion"></p>
<p>每内容3个prompt×4个seed。仅用caption／水彩×前三个seed的6张发现集筛选；新seed、新油画prompt及其组合的6张用于独立验证。全部使用完整原始二值Canny条件；没有MAT修复或几何修改。</p>
<p><a href="REPORT_ZH.md">报告</a> · <a href="summary.json">汇总</a> · <a href="curve_results.csv">逐曲线R/D/V</a> · <a href="content_results.csv">逐内容</a> · <a href="protocol.json">固定协议</a> · <a href="audit.json">审计</a></p>
<article><select id="content"></select><select id="curve"></select><select id="prompt"><option value="0">发现prompt：caption</option><option value="1">发现prompt：水彩</option><option value="2">验证prompt：油画</option></select><select id="seed"><option value="0">seed 2026110801</option><option value="1">seed 2026110802</option><option value="2">seed 2026110803</option><option value="3">验证seed 2026111004</option></select><label><input id="overlay" type="checkbox" checked>显示曲线</label>
<p id="meta"></p><p>黄：选中的源参考曲线；青：在输出骨架上找到的实际连通路径。输出RGB／Canny中的源曲线绿／红表示整条曲线通过／未通过本次几何匹配，非逐点支持。路径存在但未通过方向、距离等检查时，仍用青色显示以便检查。</p>
<div class="grid"><div>原图 RGB<div class="tile"><img id="source_rgb"><svg id="source_rgb_svg" viewBox="0 0 512 512"></svg></div></div><div>实际原始条件 Canny<div class="tile"><img id="source_canny"><svg id="source_canny_svg" viewBox="0 0 512 512"></svg></div></div><div>生成 RGB<div class="tile"><img id="rgb"><svg id="rgb_svg" viewBox="0 0 512 512"></svg></div></div><div>生成 RGB 提取 Canny<div class="tile"><img id="canny"><svg id="canny_svg" viewBox="0 0 512 512"></svg></div></div></div>
<p id="statistics"></p><p id="features"></p><div class="masks"><div>当前曲线区域Mask<img id="individual_mask"></div><div>全部发现集候选区域<img id="region_mask"></div><div>发现集置信度（0–255）<img id="confidence_mask"></div></div>
<p>区域Mask是参考曲线周围4px搜索区域；填充像素本身未被证明稳定。置信度图保存发现集R，重叠处取最大值；95% Wilson下界在对应confidence.npz中。验证失败的候选仍保留，避免用验证集反向筛选。</p></article>
<article><h2>32内容概览</h2><img src="overview.png" style="width:100%"><img src="retention_summary.png" style="max-width:100%"><img src="geometry_stability.png" style="max-width:100%"><p>以下为验证完成后用于解释的成功／失败示例；不改变候选或Mask。</p><img src="curve_match_examples.png" style="max-width:100%"></article>
<script>const DATA=__DATA__;
const $=id=>document.getElementById(id),pct=x=>x==null?'不适用':(100*x).toFixed(1)+'%',num=x=>x==null?'不适用':x.toFixed(2);
$('completion').textContent=(DATA.summary.status==='complete'?'完整结果：':'中间预览：')+DATA.cases.length+'/32个内容已完成，'+DATA.summary.discovery_selected_curves+'条发现集候选。';
DATA.cases.forEach((c,i)=>{let e=document.createElement('option');e.value=i;e.textContent='COCO '+String(c.content_id).padStart(12,'0')+'：'+c.discovery_selected+'候选';$('content').append(e)});
function fillCurves(){let c=DATA.cases[+$('content').value];$('curve').replaceChildren();c.curves.forEach((v,i)=>{let e=document.createElement('option');e.value=i;e.textContent='曲线 '+v.curve_index+'，源长'+v.arc_length_px.toFixed(0)+'px，发现R='+pct(v.discovery.R);$('curve').append(e)});$('curve').value='0';update()}
function line(svg,points,color){if(!points?.length)return;let p=document.createElementNS('http://www.w3.org/2000/svg','polyline');p.setAttribute('points',points.map(q=>q[0].toFixed(2)+','+q[1].toFixed(2)).join(' '));p.setAttribute('stroke',color);svg.append(p)}
function update(){let c=DATA.cases[+$('content').value],cid=String(c.content_id).padStart(12,'0'),v=c.curves[+$('curve').value],pi=+$('prompt').value,si=+$('seed').value,j=c.jobs.find(j=>j.prompt_index===pi&&j.seed_index===si),r=v?.views.find(r=>r.job===j.job);$('source_rgb').src='sources/'+cid+'.png';$('source_canny').src='hints/'+cid+'.png';$('rgb').src='rgb/'+j.job+'.png';$('canny').src='output_canny/'+j.job+'.png';$('region_mask').src='masks/'+cid+'_region.png';$('confidence_mask').src='masks/'+cid+'_confidence.png';$('individual_mask').src=v?v.mask:'masks/'+cid+'_region.png';
['source_rgb_svg','source_canny_svg','rgb_svg','canny_svg'].forEach(id=>$(id).replaceChildren());if(v&&$('overlay').checked){['source_rgb_svg','source_canny_svg'].forEach(id=>line($(id),v.samples,'#ffdb00'));['rgb_svg','canny_svg'].forEach(id=>{line($(id),r.output_path,'#00ddff');line($(id),v.samples,r.matched?'#00ed70':'#ff4040')})}
$('meta').textContent='COCO '+cid+'；'+j.role+'；'+j.prompt+(j.reused?'（核验复用PNG）':'');
$('statistics').textContent=v?'源长 '+num(v.arc_length_px)+'px；发现集 '+v.discovery.matched+'/6，R='+pct(v.discovery.R)+'，95%CI '+v.discovery.R_wilson95.map(pct).join('–')+'；独立验证 '+v.heldout.matched+'/6，R='+pct(v.heldout.R)+'；本次：'+(r.matched?'匹配通过':'未通过：'+r.reason):'此内容没有通过发现集筛选的候选，未强行补足。';
$('features').textContent=v?'本次对称距离中位数 '+num(r.symmetric_median_px)+'px；方向合格比例 '+pct(r.tangent_good_fraction)+'；长度比 '+num(r.length_ratio)+'；锚点合格比例 '+pct(r.anchor_good_fraction)+'；仅点位4px支持 '+pct(r.point_support_4px)+'。验证集匹配成功样本的平均法向位置方差 '+num(v.heldout.V.mean_normal_profile_variance_px2)+'px²；样本数 '+v.heldout.V.samples+'。':''}
$('content').addEventListener('change',fillCurves);['curve','prompt','seed','overlay'].forEach(id=>$(id).addEventListener('change',update));fillCurves();</script>'''
    (OUT/filename).write_text(doc.replace('__DATA__',data))

def plots(contents,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(14,5),layout='constrained');xx=np.arange(len(contents));width=.4
    ax.bar(xx-width/2,[c['discovery_selected'] for c in contents],width,label='Discovery selected')
    ax.bar(xx+width/2,[c['heldout_R_ge_80pct'] for c in contents],width,label='Heldout R >= 80% (descriptive)')
    ax.set_xticks(xx,[str(c['content_id']) for c in contents],rotation=75);ax.set_ylabel('Curve count');ax.legend();ax.set_title('Frozen discovery masks; heldout evaluation does not change masks')
    fig.savefig(OUT/'retention_summary.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    axes[0].scatter([r['source_arc_length_px'] for r in curves],[r['heldout']['R'] for r in curves],s=12,alpha=.5);axes[0].set(xlabel='Source arc length (px)',ylabel='Heldout curve R',ylim=(-.05,1.05))
    good=[r for r in curves if r['heldout']['D']['median_symmetric_px'] is not None]
    axes[1].hist([r['heldout']['D']['median_symmetric_px'] for r in good],bins=25);axes[1].set(xlabel='Median symmetric distance in matched heldout views (px)',ylabel='Curves')
    good=[r for r in curves if r['heldout']['V']['defined']]
    axes[2].scatter([r['heldout']['V']['mean_normal_profile_variance_px2'] for r in good],[r['heldout']['R'] for r in good],s=12,alpha=.5);axes[2].set(xlabel='Normal-profile variance across matched heldout views (px²)',ylabel='Heldout curve R')
    fig.savefig(OUT/'geometry_stability.png',dpi=150);plt.close(fig)
    overview=Image.new('RGB',(320*4,354*8),'#eee');draw=ImageDraw.Draw(overview)
    for ci,c in enumerate(contents):
        cid=c['content_id'];im=Image.open(OUT/'sources'/f'{cid:012d}.png').convert('RGB');d=ImageDraw.Draw(im)
        source=read(OUT/'curves'/f'{cid:012d}_source.json');sel=read(OUT/'curves'/f'{cid:012d}_selected.json')
        for r in sel['selected']:d.line([tuple(p) for p in source['curves'][r['curve_index']]['samples']],fill=(0,255,90),width=2)
        x=(ci%4)*320;y=(ci//4)*354;overview.paste(im.resize((320,320)),(x,y+34))
        draw.text((x+3,y+2),f'COCO {cid:012d}: selected {c["discovery_selected"]}; heldout80 {c["heldout_R_ge_80pct"]}',fill='black')
        draw.text((x+3,y+17),'Discovery-only selection; green = candidate source curves',fill='black')
    overview.save(OUT/'overview.png')

def examples(cases):
    picks=[]
    for passed in (True,False):
        found=None
        for c in cases:
            for v in c['curves']:
                row=next((r for r in v['views'] if r['role']=='heldout_joint' and r['matched']==passed),None)
                if row is not None:found=(c,v,row);break
            if found is not None:break
        if found is not None:picks.append(found)
    canvas=Image.new('RGB',(960,250*len(picks)),'#ddd');draw=ImageDraw.Draw(canvas)
    for n,(c,v,r) in enumerate(picks):
        cid=c['content_id'];p=np.asarray(v['samples']);lo=p.min(0)-16;hi=p.max(0)+16;side=max(48,float((hi-lo).max()))
        center=(lo+hi)/2;box=tuple(np.rint(np.r_[center-side/2,center+side/2]).astype(int))
        srgb=Image.open(OUT/'sources'/f'{cid:012d}.png').convert('RGB');scanny=Image.open(OUT/'hints'/f'{cid:012d}.png').convert('RGB')
        rgb=Image.open(OUT/'rgb'/f'{r["job"]}.png').convert('RGB');canny=Image.open(OUT/'output_canny'/f'{r["job"]}.png').convert('RGB');mask=Image.open(OUT/v['mask']).convert('RGB')
        for im in (srgb,scanny):ImageDraw.Draw(im).line([tuple(q) for q in v['samples']],fill=(255,220,0),width=1)
        for im in (rgb,canny):
            d=ImageDraw.Draw(im)
            if r['output_path']:d.line([tuple(q) for q in r['output_path']],fill=(0,220,255),width=1)
            d.line([tuple(q) for q in v['samples']],fill=(0,255,90) if r['matched'] else (255,50,50),width=1)
        for col,(im,label) in enumerate(zip((srgb,scanny,rgb,canny,mask),('Source RGB','Source Canny / yellow reference','Generated RGB','Output Canny / cyan route','Curve region mask'))):
            canvas.paste(im.crop(box).resize((192,192),Image.Resampling.NEAREST),(col*192,n*250+58))
            draw.text((col*192+3,n*250+39),label,fill='black')
        status='PASS' if r['matched'] else 'FAIL: '+r['reason']
        draw.text((3,n*250+3),f'COCO {cid:012d} / curve{v["curve_index"]} / new oil prompt + new seed: {status}',fill='black')
        draw.text((3,n*250+20),f'Source arc {v["arc_length_px"]:.1f}px; discovery {v["discovery"]["matched"]}/6; heldout {v["heldout"]["matched"]}/6; green/red = WHOLE-curve match',fill='black')
    canvas.save(OUT/'curve_match_examples.png')

def report(summary,contents):
    s=summary;protocol=read(OUT/'protocol.json');g=protocol['generation']
    f=lambda x:f'{x:.1%}' if x is not None else '不适用'
    lines=['# Stage1：Train32 Canny 曲线候选与Mask','',
      f'32个内容中，{s["contents_with_discovery_candidates"]}个有通过发现集筛选的候选；共{s["source_candidates"]}条源候选中选出{s["discovery_selected_curves"]}条。未经筛选的6张独立验证图上，曲线平均保留率为{f(s["heldout_mean_R_curve_weighted"])}（按候选平均）；{s["heldout_R_ge_80pct_curves"]}条达到至少5/6次匹配，分布于{s["contents_with_heldout_R_ge_80pct"]}/32个内容。', '',
      '## 固定数据与隔离验证','',
      f'- 数据：筛选后COCO train2017 normal清单（94625个内容）；保留前序均匀随机10个，再从其余内容无放回随机补22个。全部属于本阶段Train开发内容，未声称是未见内容泛化。',
      f'- 3个prompt×4个seed×32＝384张clean RGB；核验复用{g["reused_pngs"]}张原始条件的caption／水彩PNG，补生成{g["new_pngs"]}张。未复用MAT修复或遮挡条件的输出。',
      '- 发现集：caption、水彩×前三个seed，共6张/内容（192张）。只使用这6张计算筛选、排序、置信度与Mask。',
      '- 验证集：相同两个prompt×新seed（2张）、新油画prompt×旧seed（3张）、新prompt×新seed（1张），共6张/内容（192张）。新prompt、新seed均未参与本阶段筛选；验证结果不改变Mask。前序10内容曾参与探索，22个补抽内容为新增开发内容。',
      '- GPU1 RTX3090；SD1.5 Canny ControlNet，DDIM20、eta0、CFG5、control1、空负面prompt、FP32、禁用TF32，CPU固定初始噪声；无MAT、无几何编辑、无水印、无模型训练。', '',
      '## 曲线与匹配定义','',
      '1. OpenCV Canny(RGB,100,200)，有效照片内缩8px，排除padding；棒球内容沿用已记录的原生白框排除。当前主检测器是Canny，没有混入PiDiNet或SoftEdge结果。',
      '2. 对Canny做Zhang–Suen细化，建立8邻接像素图。删除可由两步正交路径替代的冗余对角边，防止台阶像素产生三角分叉；不增加新连接。按端点／分叉点追踪链，闭环分段，高曲率（约10px范围转角≥50°）进一步分段。',
      '3. 源曲线长度24–160px，按约1px弧长重采样，三次平滑B样条拟合；拟合偏离原轨迹超过1.25px时退回原重采样曲线。源候选独立去重：优先长段，已有曲线2px内重合超过70%的短段删除。去重不使用生成结果。',
      '4. 每个输出独立提取Canny、细化并建立像素图。在源曲线6px管道内，以距两端4px内的输出节点寻找有序连通路径，允许局部投影回退2个采样点以容纳栅格噪声；不跨越断开的骨架、不插值补线。',
      '5. 路径通过需同时满足：源曲线≥80%在输出路径4px内；双向距离各自中位数的最大值≤3px、90分位最大值≤6px；≥80%切线夹角≤30°（忽略方向正反）；长度比0.65–1.5；两端均≤4px；5个锚点中至少4个满足位置和方向。不是把多段散点拼成一次成功匹配。',
      '6. 发现集筛选：R≥80%（6张中至少5张），锚点合格视图比例≥80%，两个prompt分别至少2/3次匹配。没有最低候选数量要求。', '',
      '这里的保留率是按固定检测器与匹配规则估计的几何可重复定位率。匹配器先选一条偏好接近源曲线的最短代价输出路径，再检查几何条件，没有穷举全部备选路线；检测断边、骨架拓扑及路线选择都可能造成漏检。失败不等于视觉上绝对不存在类似轮廓，成功也不证明语义轮廓身份。', '',
      '## R、D、V与置信度','',
      '- R：满足全部曲线几何条件的视图比例，分别保存发现集、验证集及三个验证子组。每条曲线均保存Wilson 95%区间；6次视图的区间较宽，不能把5/6或6/6当作真实概率保证。',
      '- all12_descriptive：另外保存全部12个视图的R/D/V和每个prompt的4-seed结果，供描述总体变化；由于包含发现集，不能把这个指标当成独立验证。',
      '- D：成功匹配样本的对称距离、端点漂移、带符号法向位移；逐次失败原因、路径存在状态及几何量也保留，避免只汇总成功图。',
      '- V：成功匹配视图之间的长度比、总转角、平均曲率、法向偏移等特征样本方差，以及沿源曲线32个位置的法向位移方差。少于2个成功样本时标记未定义。V是对成功匹配条件化的描述，失败率仍由R体现。',
      '- R与V均来自有限视图，同一内容的视图并非严格独立抽样；Wilson区间为描述性区间。未根据验证集调节R/D/V阈值。', '',
      '## 验证结果','',
      '| 指标 | 结果 |','|---|---:|',
      f'| 有发现集候选的内容 | {s["contents_with_discovery_candidates"]}/32 |',
      f'| 发现集选中曲线 | {s["discovery_selected_curves"]} |',
      f'| 验证R≥80%的曲线／覆盖内容 | {s["heldout_R_ge_80pct_curves"]}／{s["contents_with_heldout_R_ge_80pct"]}/32 |',
      f'| 弯曲候选（源弧长/弦长≥1.05）／其中验证R≥80% | {s["discovery_selected_bent_curves_tortuosity_ge_1p05"]}／{s["heldout80_bent_curves_tortuosity_ge_1p05"]} |',
      f'| 验证平均曲线R（按曲线） | {f(s["heldout_mean_R_curve_weighted"])} |',
      f'| 全部12视图平均R（含发现集，仅描述） | {f(s["all12_descriptive_mean_R"])} |',
      f'| 以内容为簇重采样的95%区间 | {"–".join(f(v) for v in s.get("heldout_curve_R_content_bootstrap95",[]))} |',
      f'| 验证平均曲线R（按内容） | {f(s["heldout_mean_R_content_weighted"])} |',
      f'| 新seed／新prompt／联合新组合R | {" / ".join(f(s["subgroup_mean_R"][r]) for r in ("heldout_seed","heldout_prompt","heldout_joint"))} |',
      f'| 仅要求≥80%点在4px内的参考指标 | {f(s["heldout_point_only_mean_R_curve_weighted"])} |',
      f'| 输出整体平移64px的曲线匹配对照 | {f(s["heldout_shift64_mean_R_curve_weighted"])} |','',
      '在成功匹配的验证样本上，逐曲线对称距离中位数的总体中位数为 '+f'{s["matched_heldout_geometry"]["D_symmetric_px_median_across_curves"]:.2f}px；'+
      f'{s["matched_heldout_geometry"]["V_defined_curves"]}条曲线有至少2个成功验证样本，平均法向位移方差V的中位数为{s["matched_heldout_geometry"]["V_mean_normal_px2_median"]:.3f}px²、90分位为{s["matched_heldout_geometry"]["V_mean_normal_px2_p90"]:.3f}px²。上述D/V条件化于匹配成功，不包含失败视图的几何误差；失败另由R及逐次记录体现。', '',
      '平移对照是描述性空间重合检查，不是统计学误报率。点位支持参考指标不要求连通、方向或端点，不能替代曲线匹配。', '',
      '| COCO内容 | 源候选 | 发现集选中 | 验证R≥80% | 验证平均R |',
      '|---|---:|---:|---:|---:|']
    for c in contents:lines.append(f'| {c["content_id"]:012d} | {c["source_candidates"]} | {c["discovery_selected"]} | {c["heldout_R_ge_80pct"]} | {f(c["heldout_mean_R"])} |')
    lines+=['','## Mask与后续容量含义','',
      'masks/{content}_selected_curves.png保存发现集选中的拟合曲线；_region.png为其周围4px区域并裁到有效照片；_context.png再向外扩展8px供后续选patch使用。每条曲线有独立mask。区域填充像素是搜索区域，不是逐像素稳定证明。', '',
      '_confidence.png将发现集曲线R映射到0–255，重叠处取最大值；_confidence.npz保留未量化float置信度及Wilson95%下界。原始Canny骨架单独保存在graphs/。验证集失败的候选不会从mask中删除，后续可通过独立验证标签判断是否需要另设新发现实验。', '',
      f'源曲线已做空间重叠去重，同时记录source_trace_index：选中曲线来自{s["selected_parent_trace_groups"]}个源骨架链分组，同一长链被曲率或长度切分的候选可据此关联。该分组也不等于语义独立轮廓或独立信道。', '',
      '**本阶段只测clean条件下已有几何的保留，未测试修改几何能否传递。** 曲线多、R高或V小，都不能直接推导可编码比特数；相邻曲线仍可能相关，匹配具有参考位置约束，也不等于可盲定位。后续需要冻结候选后做正负几何扰动、可分辨性及新的prompt/seed验证。数量不足的内容不补凑码字。', '',
      '## 文件','',
      '[交互图册](index.html) · [32内容概览](overview.png) · [数量与验证图](retention_summary.png) · [几何稳定性](geometry_stability.png) · [成功／失败示例](curve_match_examples.png)',
      '[逐曲线JSON（完整R/D/V）](curve_results.json) · [逐曲线CSV](curve_results.csv) · [逐内容CSV](content_results.csv) · [固定协议](protocol.json) · [源内容](selected.json) · [384视图计划](plan.json)',
      '[独立审计](audit.json) · [合成检查](synthetic_validation.json) · [复用核验](reuse_validation.json) · [逐次曲线匹配目录](metrics/) · [Mask目录](masks/) · [源骨架与8邻接图](graphs/)', '']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args();collect(args.partial)
