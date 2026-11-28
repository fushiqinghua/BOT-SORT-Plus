"""Extract the four locked WEM-R3 visual-review pools from offline labels."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
R3 = HERE.parent
PARTITIONS = {'calibration': 'CAL', 'holdout': 'HOLD'}
POOLS = {'background': 'BG', 'identity': 'ID'}
EXPECTED_COUNTS = {
    'calibration': {'background': 29, 'identity': 35},
    'holdout': {'background': 28, 'identity': 21},
}


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def classify(row):
    if (row['identity_label'] != 'unknown' or not row['formal_geometry_allowed']
            or not row['gmc_guard_allowed'] or row['hard_ownership_conflict']
            or row['near_duplicate_ownership_risk']):
        return None
    if row['best_gt_iou'] < 0.05:
        return 'background'
    if row['best_gt_iou'] >= 0.5 and row['identity_rule'] == 'track_identity_unanchored':
        return 'identity'
    return None


def extract(path, partition):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        selected = []
        for row in map(json.loads, stream):
            if row['partition'] != partition:
                raise ValueError(f'Bad partition in {path}: {row["partition"]}')
            pool = classify(row)
            if pool:
                selected.append((pool, row))

    selected.sort(key=lambda item: tuple(item[1][k] for k in ('sequence', 'frame', 'track_id', 'det_index')))
    seen = {}
    totals = Counter()
    duplicates = Counter()
    cases = []
    for pool, row in selected:
        key = (row['sequence'], row['frame'], row['track_id'], tuple(row['raw_box_xyxy']))
        if key in seen:
            if seen[key] != pool:
                raise ValueError(f'Conflicting review pools for candidate {key}')
            duplicates[pool] += 1
            continue
        seen[key] = pool
        totals[pool] += 1
        case_id = f'WEM_R3_{PARTITIONS[partition]}_{POOLS[pool]}_{totals[pool]:03d}'
        cases.append(dict(**row, case_id=case_id, review_pool=pool))

    counts = {pool: totals[pool] for pool in POOLS}
    if counts != EXPECTED_COUNTS[partition]:
        raise ValueError(f'Candidate count changed for {partition}: {counts}')
    return cases, counts, {pool: duplicates[pool] for pool in POOLS}


def select(labels, lock, output):
    frozen = json.loads(lock.read_bytes())['offline_label_sha256']
    candidates = []
    counts = {}
    duplicates = {}
    inputs = {}
    for partition in PARTITIONS:
        source = labels / partition / 'candidate_edges.jsonl.gz'
        digest = sha256(source)
        if digest != frozen[partition]:
            raise ValueError(f'Offline labels differ from the calibration lock: {source}')
        inputs[partition] = dict(file=source.name, sha256=digest)
        rows, counts[partition], duplicates[partition] = extract(source, partition)
        candidates.extend(rows)
    packet = dict(schema='wem_r3_review_candidates_v1',
                  source_files=inputs, calibration_lock_sha256=sha256(lock),
                  common_filter=('identity_label == unknown and formal_geometry_allowed and gmc_guard_allowed '
                                 'and not hard_ownership_conflict and not near_duplicate_ownership_risk'),
                  pool_filters={'background': 'best_gt_iou < 0.05',
                                'identity': 'best_gt_iou >= 0.5 and identity_rule == track_identity_unanchored'},
                  deduplication_key=['partition', 'sequence', 'frame', 'track_id', 'raw_box_xyxy'],
                  duplicate_policy='Keep the lowest original det_index among exact raw-coordinate matches',
                  counts=counts, duplicates_removed=duplicates, candidates=candidates)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if not output.is_file() or output.read_bytes() != data:
        temp = output.with_suffix('.partial')
        temp.write_bytes(data)
        temp.replace(output)
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=R3 / 'labels/results/full')
    parser.add_argument('--lock', type=Path, default=R3 / 'results/full/calibration_lock.json')
    parser.add_argument('--output', type=Path, default=HERE / 'inputs/candidates.json')
    args = parser.parse_args()
    print(json.dumps(select(args.labels, args.lock, args.output)))
