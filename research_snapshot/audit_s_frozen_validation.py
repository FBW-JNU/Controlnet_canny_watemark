"""Independent artifact, geometry, denominator and finite-code checks."""
from analyze_s_frozen_validation import *
from s_frozen_offline_codes import rm_codebook
from urllib.parse import unquote

def exact_symbol_checks():
    states=np.indices((3,)*8).reshape(8,-1).T;probabilities=np.array([31/48,3/48,14/48]);weights=np.prod(probabilities[states],axis=1);assert abs(weights.sum()-1)<1e-12
    obs=np.where(states==2,-1,states);result={};codebooks={'repeat8':(np.array([[0]*8,[1]*8]),8),'RM(1,3)':(rm_codebook()[1],4)}
    simulations=read(OUT/'offline_code_results.json')
    for name,(codes,distance) in codebooks.items():
        costs=np.sum((obs[:,None,:]!=codes[None,:,:])&(obs[:,None,:]>=0),axis=2);best=costs.min(1);guess=costs.argmin(1);unique=np.sum(costs==best[:,None],axis=1)==1;s=np.sum(obs<0,axis=1)
        for bounded in [False,True]:
            accepted=unique&((2*best+s<distance) if bounded else True);row=dict(correct=float(weights[accepted&(guess==0)].sum()),wrong=float(weights[accepted&(guess!=0)].sum()),erased=float(weights[~accepted].sum()));method='bounded_unique' if bounded else 'unique_nearest'
            result[name+'_'+method]=row
            mc=next(r for r in simulations if r['code']==name and r['decoder']==method and r['correlation']==0 and abs(r['requested_erasure_rate']-14/48)<1e-9)
            for k in row:assert abs(row[k]-mc[k])<5*np.sqrt(max(.00001,row[k]*(1-row[k]))/mc['blocks'])+.0005,(name,k,row,mc)
        guaranteed=2*np.sum(states==1,axis=1)+np.sum(states==2,axis=1)<distance
        assert np.all(unique[guaranteed]&(guess[guaranteed]==0)),name
    dump(OUT/'offline_exact_validation.json',dict(status='passed',enumerated_states=3**8,exact_iid_results=result,policy='Enumerate every ternary error/erasure pattern; code linearity reduces transmitted message to zero. Verify5sigma MonteCarlo agreement and2e+s<d guarantee.'))
    return result

def run():
    f=check();assert read(OUT/'analysis_freeze.json')['script_sha256']==digest(ROOT/'analyze_s_frozen_validation.py')
    for name in ['offline_freeze.json','key_consistency_freeze.json','negative_freeze.json']:
        p={'offline_freeze.json':'s_frozen_offline_codes.py','key_consistency_freeze.json':'s_key_consistency_offline.py','negative_freeze.json':'validate_s_negative_controls.py'}[name]
        assert read(OUT/name)['script_sha256']==digest(ROOT/p)
    for p,h in read(OUT/'input_freeze.json')['protected_sha256'].items():assert digest(p)==h,p
    for name,h in read(PREV/'final_artifacts_sha256.json').items():assert digest(PREV/name)==h,name
    assert digest(PREV/'final_artifacts_sha256.json')==f['parent_manifest_sha256']
    for p,h in read(S1/'protocol.json')['frozen_model_code'].items():assert digest(p)==h,p
    plan=read(OUT/'plan.json');cc=read(OUT/'selection.json');inputs=read(OUT/'inputs.json');decoded=read(OUT/'decoded.json');bykey={c['key']:c for c in cc};excluded=set(read(OUT/'exclusions.json')['ids'])
    assert len(cc)==len({c['content_id'] for c in cc})==24 and not excluded&{c['content_id'] for c in cc}
    assert len(plan)==len({j['rgb'] for j in plan})==288 and len(decoded)==1152 and len(inputs)==72
    assert len(list((OUT/'records').glob('*.json')))==288 and len(list((OUT/'measurements').glob('*.json')))==288
    from curve_broad_peak_code import render
    for c in cc:
        mask=arr(c['mask'])>0;edge=arr(c['clean_hint'])
        assert c['input_admissible']
        for r in [r for r in inputs if r['key']==c['key']]:
            image=arr(r['hint']);assert digest(r['hint'])==r['hint_sha256'];assert set(np.unique(image))<={0,255};assert np.array_equal(image[~mask],edge[~mask])
            amplitude={'clean':0,'bit0':-1.5,'bit1':1.5}[r['arm']];assert np.array_equal(render(edge,c['samples'],mask,'S',amplitude),image)
    groups={};noise={};observed_points=0;identical=[]
    for j in plan:
        rec=read(OUT/'records'/f'{j["job"]}.json');assert rec['rgb_sha256']==digest(j['rgb']);assert rec['hint_sha256']==digest(j['hint']);assert rec['freeze_sha256']==digest(OUT/'freeze.json')
        assert rec['checkpoint_sha256']==f['generation']['checkpoint_sha256'];assert rec['seed']==j['seed'] and rec['prompt']==j['prompt'];assert rec['noise_sha256']==digest(j['noise'])
        groups.setdefault((j['key'],j['view']),[]).append(j);noise[j['noise']]=j['seed'];rgb=np.array(Image.open(j['rgb']).convert('RGB'));assert rgb.shape==(512,512,3)
        measure=read(OUT/'measurements'/f'{j["job"]}.json');assert measure['rgb_sha256']==digest(j['rgb']);edge=cv2.Canny(rgb,100,200);p,u,n,_=geometry(bykey[j['key']]['samples'])
        for h in ['Auto','Oracle']:
            obs=measure['observations'][h]
            if not obs.get('observed_xy'):continue
            xy=np.array(obs['observed_xy']);assert np.array_equal(xy,np.rint(xy));xy=xy.astype(int);assert np.all(edge[xy[:,1],xy[:,0]]>0)
            ii=np.array(obs['indices']);assert np.allclose(np.sum((xy-p[ii])*n[ii],axis=1),obs['normal_positions'],atol=1e-10);observed_points+=len(xy)
        if j['arm']!='clean' and digest(j['rgb'])==digest(j['clean_rgb']):identical.append(j['job'])
    import torch
    for path,seed in noise.items():assert np.array_equal(np.load(path),torch.randn((1,4,64,64),generator=torch.Generator(device='cpu').manual_seed(seed)).numpy())
    for key,jj in groups.items():
        assert len(jj)==3 and {j['arm'] for j in jj}=={'clean','bit0','bit1'}
        for k in ['seed','prompt','noise','clean_rgb']:assert len({j[k] for j in jj})==1
    assert len(groups)==96 and len(noise)==48
    for r in decoded:
        assert r['accepted']==(r['geometry_accepted'] and r['codebook_available'])
        if not r['accepted']:assert r['score']==0 and r['bit'] is None and not r['presence_accept']
    summary=read(OUT/'summary.json')
    for head in ['Auto','Oracle']:
        for method in ['raw','affine']:
            rr=[r for r in decoded if r['head']==head and r['method']==method];mark=[r for r in rr if r['arm']!='clean'];s=summary[head+'_'+method]
            a=np.array([r['score'] for r in mark if r['arm']=='bit0']);b=np.array([r['score'] for r in mark if r['arm']=='bit1']);delta=b[:,None]-a[None,:]
            assert abs(np.mean((delta>0)+.5*(delta==0))-s['bit_auc'])<1e-12
            assert s['correct']+s['wrong']+s['erasures']==s['marked']==192
            if method=='affine':
                a=np.array([r['s_presence'] for r in rr if r['arm']=='clean']);b=np.array([r['s_presence'] for r in mark]);delta=b[:,None]-a[None,:]
                assert abs(np.mean((delta>0)+.5*(delta==0))-s['presence_auc'])<1e-12
    # Spatial separation is a source-geometry property, not error independence.
    for path in (OUT/'capacity').glob('*.json'):
        data=read(path);assert data['eligible']==sum(c['admissible'] for c in data['candidates']);c=data['primary'];edge=arr(c['clean_hint']);occupied=np.zeros_like(edge,bool);used=[]
        for idx in data['independent_curve_indices']:
            curve=next(q for q in data['candidates'] if q['curve_index']==idx);assert curve['admissible'];mask=mask_for(edge,curve['samples'],c['roi']);wide=cv2.dilate(mask.astype('uint8'),np.ones((5,5),np.uint8))>0
            assert not np.any(wide&occupied);assert all(cKDTree(q).query(curve['samples'])[0].min()>8 for q in used);occupied|=wide;used.append(curve['samples'])
        assert len(used)==data['independent_lower_bound']
    exact_symbol_checks();viewer=read(OUT/'viewer_data.json');assert len(viewer)==96 and len({(v['key'],v['view']) for v in viewer})==96
    for v in viewer:
        with Image.open(OUT/v['panel']) as image:image.verify()
    scripts=['experiment_s_frozen_validation.py','analyze_s_frozen_validation.py','s_frozen_offline_codes.py','s_key_consistency_offline.py','validate_s_negative_controls.py','report_s_frozen_validation.py','diagnose_s_frozen_fresh.py','audit_s_frozen_validation.py']
    for name in scripts:shutil.copyfile(ROOT/name,OUT/'source'/name)
    audit=dict(status='passed',unique_rgb=288,contents=24,marked_attempts=192,clean_attempts=96,unique_input_marked_states=48,matched_prompt_seed_groups=96,noise_files=48,unique_global_seeds=2,
        observed_edge_points_verified=observed_points,identical_marked_rgb=identical,all_input_rasters_exactly_original_S=True,parent_artifacts_unchanged=True,
        frozen_implementations_and_model_code_verified=True,all_attempt_auc_verified=True,input_denominator_and_erasures_verified=True,
        known_historical_content_overlap=0,unknown_missing_historical_manifests=True,source_capacity_is_only_spatial_independence=True,
        rm_exact_ternary_patterns=6561,real_multicarrier_rgb_generated=0,viewer_panels=96,
        primary_acceptance=read(OUT/'acceptance.json'),limitation='Only2 globally fixed new seeds and2 prompt templates; missing historical directories limit full exclusion audit. Cal synthetic Key and code results do not establish real multi-carrier performance/FPR.')
    dump(OUT/'audit.json',audit)
    page=(OUT/'index.html').read_text()
    for link in re.findall(r'(?:href|src)="([^"]+)"',page):assert (OUT/unquote(link)).exists(),link
    dump(OUT/'final_artifacts_sha256.json',{str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='final_artifacts_sha256.json'})
    print('AUDIT_PASSED',audit,flush=True)

if __name__=='__main__':run()
