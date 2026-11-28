"""Explain the locked R3 history gates without changing their calibration result."""

import json
from pathlib import Path

from remote_exec import remote


HERE = Path(__file__).resolve().parent
REMOTE_CODE = """
import gzip
import hashlib
from collections import Counter

root = r.parent/'vttiny_wem_r3_v1'
features = root/'feature_analysis/embeddings/full'
labels = root/'labels_live/full'
lock = json.loads((root/'feature_analysis/scores/full/calibration_lock.json').read_bytes())
manifest = json.loads((features/'manifest.json').read_bytes())

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        yield from map(json.loads,f)

def key(row):
    return tuple(row[k] for k in ('sequence','frame','track_id','det_index'))

assert sha(features/'manifest.json') == lock['feature_manifest_sha256']
report = {}
for part in ('fit','calibration','holdout'):
    label_file = labels/part/'candidate_edges.jsonl.gz'
    assert sha(label_file) == lock['offline_label_sha256'][part]
    by_key = {key(row):row for row in rows(label_file)}
    scoped = []
    for seq, proof in manifest['sequences'].items():
        if proof['partition'] != part:
            continue
        score_file = features/(seq+'.jsonl.gz')
        assert sha(score_file) == proof['scores_sha256']
        scoped.extend((edge,by_key[key(edge)]) for edge in rows(score_file) if edge['geometry_group_rank'] <= 5)
    report[part] = {}
    for cutoff in (.75,.85,.92):
        counts = Counter()
        targets = {}
        for edge,label in scoped:
            if not label['opportunity']:
                continue
            target = tuple(label[k] for k in ('sequence','frame','gt_id'))
            stage = 0
            if edge['history_score'] is not None:
                stage = 1
                if edge['cosine'] >= cutoff:
                    stage = 2
                    if edge['motion'] >= .25:
                        stage = 3
                        if edge['history_score'] >= .55:
                            stage = 4
                            if (not edge['row_unknown'] and not edge['column_unknown'] and
                                all(edge[field] is None or edge[field] >= .05 for field in ('row_margin','column_margin'))):
                                stage = 5
            targets[target] = max(targets.get(target, 0), stage)
        names = ('eligible','usable_history','cosine','motion','joint_score','competition')
        for stage in targets.values():
            counts.update(names[:stage+1])
        report[part][str(cutoff)] = {name:counts[name] for name in names}
result = dict(status='POSTHOC_READ_ONLY', unit='deduplicated official missing target frame, before assignment',
              calibration_lock_sha256=sha(root/'feature_analysis/scores/full/calibration_lock.json'),
              sequential_gates=report)
print(json.dumps(result,indent=2))
"""


if __name__ == "__main__":
    packet = remote(REMOTE_CODE)
    result = json.loads(packet)
    path = HERE / "results" / "full" / "gate_breakdown.json"
    if path.is_file() and path.read_bytes() != packet:
        raise ValueError("Existing gate breakdown differs")
    path.write_bytes(packet)
    print(json.dumps(result, ensure_ascii=False))
