"""Paired parent-video bootstrap of held-out low-score box-error discrimination."""
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import qr_settings as settings
from r_quality import RPolicy, collect


SEED = 20261127
REPLICATES = 2000
MODES = ('full', 'no_image', 'score')


class Ranking:
    """One frozen ordering with tied scores grouped at the same decision threshold."""
    def __init__(self, scores, labels):
        values = np.asarray(scores, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all() or len(values) != len(labels):
            raise ValueError('Invalid discrimination scores')
        self.order = np.argsort(-values, kind='stable')
        self.labels = np.asarray(labels, dtype=bool)[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(values[self.order])) + 1]

    def evaluate(self, weights):
        """AUC with half credit for ties and non-interpolated average precision."""
        w = weights[self.order]
        positives = np.add.reduceat(w*self.labels, self.starts)
        negatives = np.add.reduceat(w*(~self.labels), self.starts)
        tp = np.cumsum(positives, dtype=np.float64)
        fp = np.cumsum(negatives, dtype=np.float64)
        if tp[-1] <= 0 or fp[-1] <= 0:
            raise ValueError('Undefined AUC in a one-class sample')
        concordant = np.sum(positives*(fp[-1]-fp+.5*negatives))
        precision = np.divide(tp, tp+fp, out=np.zeros_like(tp), where=tp+fp>0)
        return np.array([concordant/(tp[-1]*fp[-1]), np.dot(positives,precision)/tp[-1]])


def bootstrap(parents, labels, scores, replicates=REPLICATES, seed=SEED):
    unique, membership = np.unique(parents, return_inverse=True)
    y = np.asarray(labels, dtype=bool)
    groups = len(unique)
    if groups < 2 or not 0 < y.sum() < len(y) or replicates < 1000:
        raise ValueError('Need two outcome classes, multiple parents and at least 1000 replicates')
    ranks = [Ranking(scores[m], y) for m in MODES]
    unit = np.ones(len(y),dtype=np.int64)
    point = np.array([rank.evaluate(unit) for rank in ranks])
    rng = np.random.default_rng(seed)
    samples = []; prevalences = []; rejected = 0
    while len(samples) < replicates:
        # Every draw selects the same parents for all models. Each selected
        # parent contributes all its sequences, frames and targets together.
        frequency = np.bincount(rng.integers(groups, size=groups), minlength=groups)
        weights = frequency[membership]
        pos = np.dot(weights,y.astype(np.int64)); total = weights.sum()
        if pos==0 or pos==total:
            rejected += 1
            if rejected > replicates:
                raise ValueError('Too many one-class parent-video draws for a stable interval')
            continue
        samples.append([rank.evaluate(weights) for rank in ranks])
        prevalences.append(float(pos/total))
    samples = np.asarray(samples)

    def interval(a):
        return [float(v) for v in np.percentile(a, [2.5,97.5])]

    names=('auc','ap')
    measurements = {m:{n:{'value':float(point[i,j]),'ci95':interval(samples[:,i,j])}
                       for j,n in enumerate(names)} for i,m in enumerate(MODES)}
    contrasts = {f'full_minus_{m}':{n:{'difference':float(point[0,j]-point[i,j]),
                     'ci95':interval(samples[:,0,j]-samples[:,i,j])} for j,n in enumerate(names)}
                 for i,m in enumerate(MODES) if i}
    return {'bootstrap':{'unit':'held-out parent video with qualifying low-score observations',
                'replicates':replicates,'seed':seed,'one_class_draws_redrawn':rejected,
                'interval':'paired percentile bootstrap, two-sided 95%, prediction weights fixed'},
            'sample_count':len(y),'positive_count':int(y.sum()),'parents':unique.tolist(),
            'parent_counts':{str(k):int(v) for k,v in zip(unique,np.bincount(membership))},
            'positive_prevalence':{'value':float(y.mean()),'ci95':interval(prevalences)},
            'pooled_methods':measurements,'paired_contrasts':contrasts}


def main():
    out=settings.OUT
    policy=out/'r_policy.json'; weights_hash=settings.sha(policy)
    frozen=settings.read(policy)
    original_inputs=settings.read(out/'r_training_inputs.json')
    # collect() normally writes its input audit. Redirect that write to scratch;
    # compare it to the frozen audit instead of changing any previous artifact.
    with tempfile.TemporaryDirectory() as scratch, patch.object(settings, 'OUT', Path(scratch)):
        x,y,parents,split,*_ = collect()
        if settings.read(Path(scratch)/'r_training_inputs.json') != original_inputs:
            raise ValueError('Frozen data and recollected image/detection provenance differ')
    fit=(split=='fit'); check=(split=='check')
    if (sorted(set(parents[fit]))!=frozen['fit_parents'] or sorted(set(parents[check]))!=frozen['check_parents']
            or int(fit.sum())!=frozen['fit_rows'] or int(check.sum())!=frozen['check_rows']):
        raise ValueError('Frozen observation cohort differs from the rebuilt one')
    low=x[:,0]<.5  # collect() retains the original cached scores > .10.
    denom=np.maximum(1.,np.exp(x[:,1:3]).min(1))
    error=np.linalg.norm(y[:,:2],axis=1)/denom
    cutoff=float(np.quantile(error[low & fit],.75))
    selected=low & check
    target=error[selected]>cutoff
    scores={}
    for mode in MODES:
        # Use the original cohort batch shape as well as its inputs to preserve
        # floating-point ties in the previously audited pooled rankings.
        predicted=RPolicy(policy,mode).variance(x)
        scores[mode]=(np.sqrt(predicted[:,0]+predicted[:,1])/denom)[selected]
    report=bootstrap(parents[selected],target,scores)
    with (out/'r_low_bias.csv').open(encoding='utf-8-sig',newline='') as f:
        locked={r['method']:r for r in csv.DictReader(f) if r['split']=='check' and r['size']=='all'}
    for mode in MODES:
        row=locked[mode]
        if (int(row['n'])!=report['sample_count'] or int(row['large_bias'])!=report['positive_count']
                or not np.isclose(float(row['bias_cut_fit_norm_center']),cutoff,rtol=0.,atol=1e-12)
                or not np.isclose(report['pooled_methods'][mode]['auc']['value'],
                                  float(row['roc_auc']),rtol=0.,atol=1e-10)):
            raise ValueError(f'Recalculated outcomes disagree with frozen {mode} discrimination audit')
    if settings.sha(policy)!=weights_hash:
        raise ValueError('R weights changed during uncertainty estimation')
    report.update(model_sha256=weights_hash, input_audit_sha256=settings.sha(out/'r_training_inputs.json'),
                  frozen_discrimination_sha256=settings.sha(out/'r_low_bias.csv'),
                  target='center error divided by measured short side above fit-low 75th percentile',
                  training_cutoff=cutoff, score='predicted center standard deviation divided by measured short side',
                  scope='observation-pooled held-out training-source parents, uncertainty conditional on frozen training and weights')
    dest=out/'r_discrimination_uncertainty.json'
    settings.write(dest,report)
    print('PAIRED PARENT BOOTSTRAP COMPLETE',dest,report['paired_contrasts'],flush=True)


if __name__=='__main__':
    main()
