"""Compare keyed alignment with explicit binary codeword consistency, Cal only."""
from s_frozen_offline_codes import *

def run():
    check();config=dict(script_sha256=digest(__file__),seed=2026111149,blocks=50000,minimum_received=[4,6,7,8],allowed_disagreement=[0,1],shared_pair_probability=[0.,.5,1.],
        policy='Additional offline-only ablation; old alignment prototype preserved. No new RGB fitting, no change to frozen single-carrier decoder. Complementary keys equivalent for unknown message.')
    path=OUT/'key_consistency_freeze.json'
    if path.exists():assert read(path)==config
    else:dump(path,config)
    shutil.copyfile(__file__,OUT/'source'/Path(__file__).name)
    dd=[r for r in read(PREV/'decoded.json') if r['split']=='Cal' and r['mode']=='S' and r['head']=='Auto' and r['method']=='affine'];keys=sorted({(r['key'],r['view']) for r in dd});table={}
    for arm in ['clean','bit0','bit1']:
        rr=[next(r for r in dd if (r['key'],r['view'])==k and r['arm']==arm) for k in keys]
        table[arm]=(np.array([r['score'] for r in rr]),np.array([r['accepted'] for r in rr]))
    clean=table['clean'][0][table['clean'][1]];mean=clean.mean();std=max(.25,clean.std());rng=np.random.default_rng(config['seed']);key=rng.choice([-1,1],8);rows=[]
    def sample(n,rho,marked):
        index=rng.integers(0,len(keys),(n,8));shared=rng.random(n)<rho;index[shared]=index[shared,:1];message=rng.choice([-1,1],n);s=np.zeros((n,8));v=np.zeros((n,8),bool)
        for i in range(8):
            if not marked:s[:,i]=table['clean'][0][index[:,i]];v[:,i]=table['clean'][1][index[:,i]]
            else:
                bit=message*key[i]>0
                for b in [0,1]:
                    choose=bit==b;s[choose,i]=table[f'bit{b}'][0][index[choose,i]];v[choose,i]=table[f'bit{b}'][1][index[choose,i]]
        return s,v,message
    def features(s,v,k):
        n=v.sum(1);t=abs(np.sum((s-mean)/std*v*k,axis=1))/np.sqrt(np.maximum(1,n));t[n<4]=0
        mismatch=np.sum(((s>0)!=(k>0))&v,axis=1);return n,t,np.minimum(mismatch,n-mismatch)
    for rho in config['shared_pair_probability']:
        sc,vc,_=sample(20000,rho,False);threshold=float(np.percentile(features(sc,vc,key)[1],95))
        clean,cv,_=sample(50000,rho,False);marked,mv,message=sample(50000,rho,True);wrongkey=rng.choice([-1,1],marked.shape)
        fc=features(clean,cv,key);fm=features(marked,mv,key);fw=features(marked,mv,wrongkey)
        for minimum in config['minimum_received']:
            for mistakes in config['allowed_disagreement']:
                def accept(f):return (f[0]>=minimum)&(f[1]>threshold)&(f[2]<=mistakes)
                ac,am,aw=map(accept,[fc,fm,fw]);guess=np.where(np.sum(np.where(marked>0,1,-1)*mv*key,axis=1)>0,1,-1)
                rows.append(dict(shared_pair_probability=rho,minimum_received=minimum,allowed_disagreement=mistakes,blocks=50000,threshold=threshold,
                    clean_accept=float(ac.mean()),marked_accept=float(am.mean()),wrong_key_accept=float(aw.mean()),correct_message_and_presence=float(np.mean(am&(guess==message)))))
    dump(OUT/'key_consistency_results.json',dict(rows=rows,key_signs=key.tolist(),
        detector='Frozen per-carrier bit decisions. Require abs(keyed soft alignment)>Cal clean95 threshold, n>=minimum, min Hamming distance to{key,-key}<=allowed disagreements.',
        limitation='No threshold is promoted to real receiver. Synthetic blocks derived only from12 Cal views; ownership and low real FPR remain unvalidated.'))
    print('KEY_CONSISTENCY_COMPLETE',len(rows),flush=True)

if __name__=='__main__':run()
