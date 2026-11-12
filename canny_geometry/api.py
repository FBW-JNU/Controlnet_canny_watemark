"""Portable IO around the frozen numerical encoder and 1-bit/4A receiver.

The public decoder returns message decisions/erasures. It does not run a
clean-versus-marked threshold, and it never receives the true message.
"""
from pathlib import Path
import json

import cv2
import numpy as np
from scipy.special import logsumexp

from .geometry import canonical
from .observations import binary_target, prepare_target
from .raster import RasterWarp
from .receiver import basis, receive, code_score

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/'configs/geometry.json').read_text())
THRESHOLDS = CONFIG['thresholds']
GATE = CONFIG['gate']
WORDS = ['00', '01', '10', '11']
SIGNS = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])


def validate_edge(edge):
    edge = np.asarray(edge)
    if edge.ndim != 2 or min(edge.shape) < 2:
        raise ValueError('Expected a two-dimensional binary Canny map.')
    if not set(np.unique(edge)).issubset({0,255}):
        raise ValueError('Canny must contain 0/255; threshold explicitly before encoding.')
    return edge.astype(np.uint8)


def mask_for(edge, samples):
    mask = np.zeros_like(edge)
    cv2.polylines(mask,[np.rint(samples).astype(np.int32).reshape(-1,1,2)],False,255,1)
    return cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))) > 0


def observe(samples, image, rgb=False):
    target = prepare_target(image) if rgb else binary_target(image,samples)
    if not len(target['coords']):
        return {h:dict(accepted=False,coefficient=None,reason='no_observed_edges')
                for h in ['Auto','Oracle']}
    return receive(samples,target,'S',THRESHOLDS,GATE)


def encode(edge, samples, mode='s1', mask=None):
    edge = validate_edge(edge)
    samples = canonical(samples)
    if len(samples) < 7 or not np.isfinite(samples).all():
        raise ValueError('Supply at least seven finite ordered source samples.')
    if mask is None:
        mask = mask_for(edge,samples)
    mask = np.asarray(mask,bool)
    if mask.shape != edge.shape:
        raise ValueError('Mask and Canny dimensions differ.')
    warp = RasterWarp(edge,samples,mask)
    b = basis(warp.u)
    if mode == 's1':
        vectors = {'0':-1.5*b[:,0], '1':1.5*b[:,0]}
    elif mode == 'st2':
        peak = np.abs(basis(np.linspace(0,1,100001))).sum(1).max()
        vectors = {word:b@(1.5/peak*sign) for word,sign in zip(WORDS,SIGNS)}
    else:
        raise ValueError('mode must be s1 or st2')
    images = {'clean':edge.copy()}
    for word, displacement in vectors.items():
        warp.vector = displacement[warp.near[warp.chosen],None]*warp.n[warp.near[warp.chosen]]
        images[word] = warp.render(1.)[0]
        if not np.array_equal(images[word][~mask],edge[~mask]):
            raise AssertionError('Encoder changed pixels outside the mask.')
    return images,mask


def make_reference(edge, samples, mode='s1', mask=None, calibration=None):
    images,mask = encode(edge,samples,mode,mask)
    measured = {word:observe(samples,img) for word,img in images.items()}
    words = ['0','1'] if mode=='s1' else WORDS
    books = {}
    for head in ['Auto','Oracle']:
        if not all(measured[w][head]['accepted'] for w in ['clean']+words):
            raise ValueError(f'{head}: source states fail input correspondence; choose another curve.')
        book = {w:measured[w][head]['coefficient'] for w in ['clean']+words}
        if mode == 's1':
            book = {'clean':book['clean'],'bit0':book['0'],'bit1':book['1']}
        books[head] = book
    if any(np.array_equal(images[w],images['clean']) for w in words):
        raise ValueError('A marked raster is unchanged; input is not writable.')
    if len({img.tobytes() for w,img in images.items() if w!='clean'}) != len(words):
        raise ValueError('Messages collapse to identical pixel images.')
    ref = dict(schema_version=1,mode=mode,samples=np.asarray(samples).tolist(),
               codebooks=books,calibration=calibration,
               receiver='Frozen S/4A local reference-assisted joint Huber receiver',
               calibration_scope='Frozen Cal model' if calibration else 'Input-domain diagnostic; unvalidated for RGB')
    for word in words:
        if decode_observation(ref,measured[word]['Auto'])['bits'] != word:
            raise ValueError('Input message does not close under the supplied classifier.')
    return images,mask,ref


def decode_observation(reference, observation):
    if not observation.get('accepted'):
        return dict(accepted=False,bits=None,reason=observation.get('reason','geometry_rejected'))
    mode = reference['mode']
    book = reference['codebooks']['Auto']
    calibration = reference.get('calibration')
    if mode == 's1':
        score = code_score(observation,book)
        if calibration:
            score = (score-calibration['intercept'])/calibration['gain']
        return dict(accepted=True,bits=str(int(score>0)),score=float(score),
                    coefficient=observation['coefficient'])
    y = np.array(observation['coefficient'])-np.array(book['clean'])
    x = np.array([np.array(book[w])-book['clean'] for w in WORDS])
    if calibration:
        centers = x@np.array(calibration['A']).T+calibration['intercept']
        precision = np.linalg.inv(np.array(calibration['residual_covariance']))
    else:
        centers,precision = x,np.eye(2)
    residual = y-centers
    scores = -.5*np.einsum('ni,ij,nj->n',residual,precision,residual)
    bit_scores = [float(logsumexp(scores[[i for i,w in enumerate(WORDS) if w[b]=='1']])-
                        logsumexp(scores[[i for i,w in enumerate(WORDS) if w[b]=='0']])) for b in [0,1]]
    return dict(accepted=True,bits=WORDS[int(np.argmax(scores))],bit_scores=bit_scores,
                coefficient=observation['coefficient'])


def decode(reference, image, rgb=True):
    return decode_observation(reference,observe(reference['samples'],image,rgb)['Auto'])


def summarize_records(records):
    """Group frozen decisions; true labels are only used here for evaluation."""
    result = {}
    policies = sorted({r['policy'] for r in records})
    methods = sorted({m for r in records for m in r['heads']})
    for policy in policies:
        for method in methods:
            rows = [r for r in records if r['policy']==policy and r['truth'] is not None]
            correct = wrong = erased = bit_correct = bit_wrong = bit_erased = 0
            for r in rows:
                truth = r['truth']; d = r['heads'][method]
                if not d['accepted'] or d.get('bits') is None:
                    erased += 1; bit_erased += len(truth)
                    continue
                if len(d['bits']) != len(truth):
                    raise ValueError('Decoded message length mismatch.')
                correct += d['bits']==truth
                wrong += d['bits']!=truth
                bit_correct += sum(a==b for a,b in zip(truth,d['bits']))
                bit_wrong += sum(a!=b for a,b in zip(truth,d['bits']))
            nbits = bit_correct+bit_wrong+bit_erased
            result[f'{policy}/{method}'] = dict(attempts=len(rows),correct=correct,wrong=wrong,
                erasures=erased,full_message_accuracy=correct/len(rows) if rows else None,
                bit_correct=bit_correct,bit_wrong=bit_wrong,bit_erased=bit_erased,
                all_attempt_bit_accuracy=bit_correct/nbits if nbits else None,
                received_bit_error_rate=bit_wrong/(bit_wrong+bit_correct) if bit_wrong+bit_correct else None)
    return result
