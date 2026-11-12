"""Ingest explicit verified correspondences without ever fabricating them.

Annotations must be produced without consulting intended message labels.
The annotation file hash/provenance is retained; correctness remains a supplied
annotation claim, not something a numeric pixel-membership test can establish.
"""
from compare_st_identifiability_4b import *

def run(path):
    check();annotation=read(path);mapping={r['tag']:r for r in read(OUT/'blind_review_mapping.json')}
    records=annotation['records'];assert len(records)==48 and {r['tag'] for r in records}==set(mapping)
    assert all(r['status'] in ['verified','absent','ambiguous'] for r in records),'Unannotated entries cannot be treated as erasures or ground truth'
    assert all(r.get('reviewer') and r.get('rationale') for r in records),'Reviewer and message-independent rationale required'
    cc={c['key']:c for c in read(PRIOR/'selection.json')};jobs={j['job']:j for j in read(PRIOR/'plan.json')};frozen=read(PRIOR/'score_freeze.json');rows=[]
    for r in records:
        mapped=mapping[r['tag']];j=jobs[mapped['job']];c=cc[j['key']]
        if r['status']=='verified':
            ii=np.array(r['indices'],int);xy=np.array(r['observed_xy'],float);weights=np.array(r['weights'],float);p,_,_,_=geometry(c['samples'])
            assert len(ii)==len(xy)==len(weights)>=7
            assert len(set(ii))==len(ii) and len(set(map(tuple,xy)))==len(xy)
            assert np.all((ii>=0)&(ii<len(p))) and np.all(np.diff(ii)>0)
            assert np.all((weights>0)&(weights<=1));assert np.array_equal(xy,xy.round())
            edge=cv2.Canny(np.array(Image.open(j['rgb']).convert('RGB')),100,200);xx=xy[:,0].astype(int);yy=xy[:,1].astype(int)
            assert np.all((xx>=0)&(xx<512)&(yy>=0)&(yy<512));assert np.all(edge[yy,xx]>0)
            result=fit(c['samples'],dict(indices=ii,xy=xy,weights=weights),'S');obs=b4.gate_one(as_gate_fit(c['samples'],result),frozen['gate'])
        else:result=None;obs=dict(fitted=False,accepted=False,reason='verified_'+r['status'])
        rows.append(dict(job=j['job'],key=j['key'],arm=j['arm'],method='TrueOracle_SuppliedAnnotations',annotation_status=r['status'],
           native=b4.decode(c,obs,frozen['channel_model'],frozen['presence']),
           common_A4=a4.infer(c,dict(Auto=obs),frozen['channel_model'],'Auto'),geometry=result,metrics=residual_metrics(c,result)))
    rr=[r for r in rows if r['native']['accepted']];correct=sum(r['native']['word']==r['arm'] for r in rr)
    dump(OUT/'verified_oracle_result.json',dict(annotation_file=str(Path(path).resolve()),annotation_sha256=digest(path),created_utc=b4.now(),records=rows,
        summary=dict(attempts=48,received=len(rr),correct=correct,wrong=len(rr)-correct,erasures=48-len(rr),full_accuracy=correct/48),
        caveat='Verified contour identity and station mapping are supplied annotation assertions. Pixel-membership audit checks actual observations but cannot independently prove diffusion point identity.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('annotations');run(p.parse_args().annotations)
