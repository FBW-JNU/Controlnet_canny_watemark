"""Independent consistency and provenance audit for the S/T four-word pilot."""
from curve_single_2bit_stage4a import *
from urllib.parse import unquote


def rank_auc(labels,scores):
    y=np.asarray(labels);v=np.asarray(scores);d=v[y==1,None]-v[y==0][None,:]
    return float(np.mean((d>0)+.5*(d==0)))


def run():
    f=check();freeze=read(OUT/'input_freeze.json');score=read(OUT/'score_freeze.json')
    assert digest(ROOT/f['generation']['checkpoint'])==f['generation']['checkpoint_sha256']
    for path,h in read(S1/'protocol.json')['frozen_model_code'].items():assert digest(path)==h,path
    for path,h in freeze['protected_sha256'].items():assert digest(path)==h,path
    assert score['protocol_sha256']==digest(OUT/'protocol.json')
    for path,h in score['cal_measurement_sha256'].items():assert digest(path)==h,path
    assert score['finite_differences_sha256']==digest(OUT/'cal_response_matrices.json')
    for parent in [PRIOR,BROAD]:
        for name,h in read(parent/'final_artifacts_sha256.json').items():assert digest(parent/name)==h,(parent,name)
    assert digest(PRIOR/'final_artifacts_sha256.json')==f['prior_manifest_sha256']
    plan=read(OUT/'plan.json');cc=read(OUT/'selection.json');bykey={c['key']:c for c in cc};decoded=read(OUT/'decoded.json')
    assert len(cc)==18 and sum(c['split']=='Cal' for c in cc)==6 and sum(c['split']=='Eval' for c in cc)==12
    assert len({c['content_id'] for c in cc})==18
    excluded=set(read(OUT/'exclusions.json')['ids'])
    assert not excluded&{c['content_id'] for c in cc if c['split']=='Eval'}
    assert len(plan)==len({j['rgb'] for j in plan})==324 and len(decoded)==648
    assert len(list((OUT/'records').glob('*.json')))==len(list((OUT/'measurements').glob('*.json')))==324
    groups={};noises={};verified_points=0;identical=[];inputs=0
    for c in cc:
        assert c['admission']['admissible'];edge=arr(c['clean_hint']);mask=arr(c['mask'])>0
        p,u,_,_=geometry(c['samples']);assert np.max(np.abs(basis(u)),axis=0).max()<=1.000001
        for arm,sign in SIGNS.items():
            path=c['states'][arm]['hint'];image=arr(path);assert digest(path)==c['states'][arm]['hint_sha256'];assert set(np.unique(image))<={0,255}
            assert np.array_equal(image,render(edge,c['samples'],mask,sign));assert np.array_equal(image[~mask],edge[~mask]);inputs+=1
            assert np.max(np.abs(basis(u)@(AMPLITUDE*np.array(sign))))<=1.500001
    for j in plan:
        assert j['seed']==SEEDS[j['split']][j['view']]
        rec=read(OUT/'records'/f'{j["job"]}.json');assert digest(j['rgb'])==rec['rgb_sha256'];assert digest(j['hint'])==rec['hint_sha256'];assert rec['protocol_sha256']==digest(OUT/'protocol.json')
        assert rec['checkpoint_sha256']==f['generation']['checkpoint_sha256'];assert rec['prompt']==j['prompt'] and rec['seed']==j['seed'];assert rec['noise_sha256']==digest(j['noise'])
        if j['split']=='Eval':assert rec['score_freeze_sha256']==digest(OUT/'score_freeze.json')
        groups.setdefault((j['key'],j['view']),[]).append(j);noises[j['noise']]=j['seed']
        rgb=np.array(Image.open(j['rgb']).convert('RGB'));assert rgb.shape==(512,512,3)
        if j['arm']!='clean' and rec['rgb_sha256']==digest(j['clean_rgb']):identical.append(j['job'])
        measurement=read(OUT/'measurements'/f'{j["job"]}.json');assert measurement['rgb_sha256']==rec['rgb_sha256'];edge=cv2.Canny(rgb,100,200);p,u,n,_=geometry(bykey[j['key']]['samples'])
        for head in ['Auto','Oracle']:
            obs=measurement['observations'][head]
            if obs.get('observed_xy'):
                xy=np.array(obs['observed_xy']);assert np.array_equal(xy,np.rint(xy));xy=xy.astype(int);ii=np.array(obs['indices']);assert np.all(edge[xy[:,1],xy[:,0]]>0)
                assert np.allclose(np.sum((xy-p[ii])*n[ii],axis=1),obs['normal_positions'],atol=1e-10);verified_points+=len(xy)
    assert len(groups)==36 and len(noises)==36
    for group in groups.values():
        assert len(group)==9 and {j['arm'] for j in group}==set(SIGNS)
        for field in ['prompt','seed','noise','clean_rgb']:assert len({j[field] for j in group})==1
    import torch
    for path,seed in noises.items():assert np.array_equal(np.load(path),torch.randn((1,4,64,64),generator=torch.Generator(device='cpu').manual_seed(seed)).numpy())
    for d in decoded:
        j=next(j for j in plan if j['job']==d['job']);c=bykey[j['key']];obs=read(OUT/'measurements'/f'{j["job"]}.json')['observations'][d['head']]
        assert d['accepted']==(obs['accepted'] and c['admission']['admissible'])
        if not d['accepted']:
            assert d['word'] is None and d['presence_score']==-1e6 and not d['presence_accept'];continue
        model=score['models'][d['head']];values=np.array(d['feature']);centers=np.array([np.array(model['A'])@features(c,a,d['head'])+model['intercept'] for a in WORDS]);inv=np.linalg.inv(model['residual_covariance'])
        distance=np.array([(values-v)@inv@(values-v) for v in centers]);assert WORDS[int(distance.argmin())]==d['word']
    summaries=read(OUT/'summary.json')
    for split in ['Cal','Eval']:
        for head in ['Auto','Oracle']:
            rr=[r for r in decoded if r['split']==split and r['head']==head];msg=[r for r in rr if r['arm'] in WORDS];clean=[r for r in rr if r['arm']=='clean'];s=summaries[split+'_'+head]
            assert s['correct']+s['wrong']+s['erasures']==s['attempts']==(48 if split=='Cal' else 96)
            assert s['correct']==sum(r['accepted'] and r['word']==r['arm'] for r in msg)
            assert abs(rank_auc([int(r['arm']!='clean') for r in msg+clean],[r['presence_score'] for r in msg+clean])-s['presence_auc'])<1e-12
            for bit in [0,1]:assert abs(rank_auc([int(r['arm'][1+bit]) for r in msg],[r['bit_scores'][bit] for r in msg])-s['bits'][bit]['auc'])<1e-12
    # Independently recompute the pooled actual-input Cal regression.
    for head in ['Auto','Oracle']:
        x=[];y=[]
        for j in plan:
            if j['split']!='Cal':continue
            c=bykey[j['key']];obs=read(OUT/'measurements'/f'{j["job"]}.json')['observations'][head]
            if not obs['accepted']:continue
            origin=np.array(c['states']['clean']['measurements'][head]['coefficient'])
            x.append(np.r_[1,np.array(c['states'][j['arm']]['measurements'][head]['coefficient'])-origin]);y.append(np.array(obs['coefficient'])-origin)
        beta=np.linalg.lstsq(x,y,rcond=None)[0];assert np.allclose(beta[0],score['models'][head]['intercept']);assert np.allclose(beta[1:].T,score['models'][head]['A'])
    viewer=read(OUT/'viewer_data.json');assert len(viewer)==36
    for v in viewer:
        with Image.open(OUT/v['panel']) as im:im.verify()
    page=(OUT/'index.html').read_text()
    for link in re.findall(r'(?:href|src)="([^"]+)"',page):
        if link=='audit.json':continue
        assert (OUT/unquote(link)).exists(),link
    for name in ['report_single_2bit_stage4a.py','audit_single_2bit_stage4a.py']:shutil.copyfile(ROOT/name,OUT/'source'/name)
    result=dict(status='passed',contents=18,cal_contents=6,new_eval_contents=12,rgb=324,matched_groups=36,arms_per_group=9,
        unique_binary_inputs=inputs,unique_four_word_input_states=72,eval_four_word_attempts=96,eval_clean=24,
        actual_edge_points_verified=verified_points,identical_nonclean_rgb=identical,
        frozen_sources_verified=True,input_exterior_unchanged=True,combined_continuous_displacement_bound_px=1.5,
        eval_calibration_hash_verified=True,all_attempt_denominators_and_auc_verified=True,cal_channel_matrix_verified=True,
        previous_outputs_unchanged=True,known_eval_content_overlap=0,missing_historical_manifests=True,
        viewer_panels=36,real_error_correction_implemented=False,
        limitation='12 Eval contents, two prompt/seed combinations per content; source-only admissible selection; Oracle is not true point correspondence.')
    dump(OUT/'audit.json',result)
    dump(OUT/'final_artifacts_sha256.json',{str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='final_artifacts_sha256.json'})
    print('AUDIT_PASSED',result,flush=True)


if __name__=='__main__':run()
