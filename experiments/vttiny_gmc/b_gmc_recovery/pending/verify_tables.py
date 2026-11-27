"""Reconcile full and pilot scores, count identical TRACK files and publish the per-sequence deltas."""
import csv
import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parent/'results'
focus = 'DJI_0030_2'
ref = 'evidence_both'
arms = (ref, 'pending_strict', 'pending_temporal', 'pending_guarded')
scores = ('HOTA','DetA','AssA','IDF1','MOTA','FP','FN','IDSW')


def data(path):
    return json.loads((root/path).read_bytes())


def table(path):
    with (root/path).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def output(name, rows):
    with (root/name).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)


summary = data('full/metrics/summary.json')['results']
official = {(r['tracker'],r['sequence']):r for r in summary}
audit = data('full/metrics/evaluation_audit.json')['validation']
protocols = audit['protocols']
seqs = sorted(audit['sequences'])
assert len(seqs) == 30 and audit['subset'] is False
cases = data('focus/target.json') | data('focus2/target.json')
parity = data('focus2/parity.json')
assert parity[ref]['focus_sha256'] == parity['pending_shadow']['focus_sha256']
old = data('full/target.json')
assert all(old[m] == cases[m] for m in arms)
totals, each = [], []
wait = table('full/pending_by_sequence.csv')
decisions = table('full/release_outcomes.csv')
for m in arms:
    p, b = protocols[m], protocols[ref]
    for key in ('sequences','detection_fingerprints','feature_hashes','decoded_image_hashes',
                'native_config','rgb_encoder_sha256','detector_sha256','encoder_contracts','boxmot_source_hashes'):
        assert p[key] == b[key], (m,key)
    combined = official[m,'COMBINED_SEQ']
    delta = {k:combined[k]-official[ref,'COMBINED_SEQ'][k] for k in scores}
    for k in ('FP','FN','IDSW'):
        assert combined[k] == sum(official[m,s][k] for s in seqs)
    unchanged=0
    for s in seqs:
        same = p['output_prediction_hashes'][s]==b['output_prediction_hashes'][s]
        unchanged += same
        diffs = {k:official[m,s][k]-official[ref,s][k] for k in scores}
        assert not same or all(x == 0 for x in diffs.values())
        each.append(dict(method=m,sequence=s,trajectory_identical=same,**{k:official[m,s][k] for k in scores},
                         **{'delta_'+k:v for k,v in diffs.items()}))
    select = [r for r in wait if r['method']==m]
    outcomes = Counter(r['outcome'] for r in decisions if r['method']==m)
    nrel = sum(int(r['release']) for r in select)
    assert nrel == sum(outcomes.values())
    totals.append(dict(method=m,frames=sum(audit['sequences'].values()),sequences=len(seqs),
                       identical_trajectories=unchanged,deferred=sum(int(r['defer']) for r in select),releases=nrel,
                       same_gt=outcomes['same_gt'],other_gt=outcomes['other_gt'],FP_at_release=outcomes['FP'],
                       prior_unknown=outcomes['prior_unknown'],not_emitted=outcomes['not_emitted'],
                       **{k:combined[k] for k in scores},**{'delta_'+k:v for k,v in delta.items()}))
output('30段_逐序列对照.csv',each)
output('30段_总体指标与接回去向.csv',totals)
print(json.dumps(totals,indent=2,ensure_ascii=False))
