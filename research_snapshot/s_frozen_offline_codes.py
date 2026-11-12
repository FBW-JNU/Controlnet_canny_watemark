"""Cal-only keyed-coherence prototype and correlated error/erasure simulations.

These are offline synthetic blocks, never real eight-carrier RGB evidence.
"""
from experiment_s_frozen_validation import *
from analyze_canny_curve_stage2_bit import auc

N=50000
SEED=2026111147
CORRELATIONS=[0.,.1,.3,.6]
ERASURES=[.1,14/48,.45,.6]

def rm_codebook():
    messages=((np.arange(16)[:,None]>>np.arange(4))&1).astype('int8')
    points=((np.arange(8)[:,None]>>np.arange(3))&1).astype('int8')
    codes=(messages[:,:1]+messages[:,1:]@points.T)%2
    distances=np.sum(codes[:,None]!=codes[None,:],axis=2);assert distances[~np.eye(16,dtype=bool)].min()==4
    return messages,codes

def block_channel(truth,rng,pe,wrong_given_received,rho):
    probs=np.array([(1-pe)*(1-wrong_given_received),(1-pe)*wrong_given_received,pe])
    if rho==0:p=np.tile(probs,(len(truth),1))
    else:p=rng.dirichlet(probs*(1/rho-1),len(truth))
    u=rng.random(truth.shape);state=(u>=p[:,0,None]).astype('int8')+(u>=(p[:,0]+p[:,1])[:,None]).astype('int8')
    obs=truth.copy();obs[state==1]^=1;obs[state==2]=-1
    return obs,state

def decode_blocks(observed,codes,true_indices,distance,bounded):
    costs=np.sum((observed[:,None,:]!=codes[None,:,:])&(observed[:,None,:]>=0),axis=2)
    best=costs.min(1);guess=costs.argmin(1);unique=np.sum(costs==best[:,None],axis=1)==1;erasures=np.sum(observed<0,axis=1)
    accepted=unique & ((2*best+erasures<distance) if bounded else True)
    return dict(correct=float(np.mean(accepted&(guess==true_indices))),wrong=float(np.mean(accepted&(guess!=true_indices))),erased=float(np.mean(~accepted)))

def simulate_symbols(rng):
    _,rm=rm_codebook();rep=np.array([[0]*8,[1]*8],dtype='int8');rows=[]
    for pe in ERASURES:
        for rho in CORRELATIONS:
            for name,codes,distance,bits in [('repeat8',rep,8,1),('RM(1,3)',rm,4,4)]:
                true=rng.integers(0,len(codes),N);observed,state=block_channel(codes[true],rng,pe,3/34,rho)
                for bounded in [False,True]:
                    rows.append(dict(code=name,message_bits=bits,block_symbols=8,decoder='bounded_unique' if bounded else 'unique_nearest',
                        requested_erasure_rate=pe,wrong_given_received=3/34,correlation=rho,blocks=N,
                        observed_symbol_correct=float(np.mean(state==0)),observed_symbol_wrong=float(np.mean(state==1)),observed_symbol_erased=float(np.mean(state==2)),
                        **decode_blocks(observed,codes,true,distance,bounded)))
    return rows

def simulate_presence(rng):
    decoded=[r for r in read(PREV/'decoded.json') if r['split']=='Cal' and r['mode']=='S' and r['head']=='Auto' and r['method']=='affine']
    keys=sorted({(r['key'],r['view']) for r in decoded});table={}
    for arm in ['clean','bit0','bit1']:
        rr=[next(r for r in decoded if (r['key'],r['view'])==k and r['arm']==arm) for k in keys]
        table[arm]=(np.array([r['score'] for r in rr]),np.array([r['accepted'] for r in rr]))
    clean_values=table['clean'][0][table['clean'][1]];mean=float(clean_values.mean());std=max(.25,float(clean_values.std()));key=rng.choice([-1,1],8)
    def sample(rng,n,rho,marked):
        index=rng.integers(0,len(keys),(n,8));shared=rng.random(n)<rho;index[shared]=index[shared,:1]
        message=rng.choice([-1,1],n);scores=np.zeros((n,8));valid=np.zeros((n,8),bool)
        for i in range(8):
            if not marked:
                scores[:,i]=table['clean'][0][index[:,i]];valid[:,i]=table['clean'][1][index[:,i]]
            else:
                bit=(message*key[i]>0)
                for b in [0,1]:
                    choose=bit==b;scores[choose,i]=table[f'bit{b}'][0][index[choose,i],];valid[choose,i]=table[f'bit{b}'][1][index[choose,i],]
        return (scores-mean)/std,valid
    def statistic(z,w,k):
        value=abs(np.sum(z*w*k,axis=1))/np.sqrt(np.maximum(1,w.sum(1)));value[w.sum(1)<4]=0;return value
    rows=[]
    for rho in [0.,.25,.5,.75,1.]:
        z,w=sample(rng,20000,rho,False);threshold=float(np.percentile(statistic(z,w,key),95))
        zc,wc=sample(rng,N,rho,False);zm,wm=sample(rng,N,rho,True);sc=statistic(zc,wc,key);sm=statistic(zm,wm,key);wrongkey=rng.choice([-1,1],(N,8));sw=statistic(zm,wm,wrongkey)
        rows.append(dict(shared_cal_pair_probability=rho,blocks=N,clean_vs_marked_auc=auc([0]*N+[1]*N,np.r_[sc,sm]),
            model_clean95_threshold=threshold,clean_accept=float(np.mean(sc>threshold)),marked_accept=float(np.mean(sm>threshold)),wrong_key_accept=float(np.mean(sw>threshold)),
            wrong_key_equivalent_fraction=float(np.mean(np.all(wrongkey==key,axis=1)|np.all(wrongkey==-key,axis=1)))))
    return dict(cal_clean_valid=len(clean_values),cal_pairs=len(keys),cal_clean_mean=mean,cal_clean_std=std,key_signs=key.tolist(),rows=rows,
        detector='Unknown1bit keyed repetition: abs(sum k_i*z_i*w_i)/sqrt(sum w_i), require>=4/8 observations, binary acceptance weights, Cal-only centering/scaling. k and -k are equivalent for unknown bit.',
        limitations='Synthetic8carrier blocks resample only12 Cal content/view pairs. Shared-pair probability is a stress model, not measured within-image carrier correlation. Model95% threshold is not validated real FPR; no new real multi-carrier images used.')

def run():
    check();freeze=OUT/'offline_freeze.json';config=dict(script_sha256=digest(__file__),seed=SEED,blocks=N,erasure_rates=ERASURES,correlations=CORRELATIONS,
        old_symbol_counts=dict(correct=31,wrong=3,erasure=14),policy='Theory and synthetic symbols only. Parameters fixed before new test scoring. No real8/32carrier generation, no new-test model fitting.')
    if freeze.exists():assert read(freeze)==config
    else:dump(freeze,config)
    shutil.copyfile(__file__,OUT/'source'/Path(__file__).name);rng=np.random.default_rng(SEED)
    result=simulate_symbols(rng);dump(OUT/'offline_code_results.json',result);dump(OUT/'offline_key_presence.json',simulate_presence(rng))
    dump(OUT/'rm13_codebook.json',dict(messages=rm_codebook()[0].tolist(),codewords=rm_codebook()[1].tolist(),field='GF(2)',minimum_distance=4,
        distinction='RM affine Boolean polynomials over GF(2) encode4 message bits into8 symbols. S geometric polynomial remains a real-valued scalar carrier, not an RS polynomial.'))
    print('OFFLINE_COMPLETE',len(result),flush=True)

if __name__=='__main__':run()
