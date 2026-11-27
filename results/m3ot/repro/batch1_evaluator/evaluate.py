"""Score the M3OT factorial replay with the original single-vehicle TrackEval protocol."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


TRACKERS = ('native', 'B_early_moderate', 'dual_low', 'evidence_both')
DETECTORS = ('lower', 'higher')
COUNT_METRICS = ('TP', 'FP', 'FN', 'IDSW')
METRICS = ('HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', *COUNT_METRICS)
TRACK_EVAL_COMMIT = '12c8791b303e0a0b50f753af204249e622d0281a'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ground_truth(project, output, convert_file):
    source = project / 'data/raw/M3OT'
    manifest = json.loads((project / 'reports/usable_splits.json').read_bytes())
    gt = {}
    for row in manifest['sequences']:
        if (row['split'], row['modality']) != ('val', 'rgb'):
            continue
        name = row['sequence_id'].replace('/', '_')
        src = source / row['sequence_id'] / 'gt/gt.txt'
        dst = output / 'gt' / name / 'gt/gt.txt'
        frames = len(row['frame_ids'])
        if row['frame_ids'] != list(range(1, frames + 1)):
            raise ValueError(f'Frame sequence has holes: {name}')
        converted = convert_file(src, dst)
        gt[name] = dict(frames=frames, source_relative_path=src.relative_to(project).as_posix(),
                        raw_gt_sha256=sha(src), evaluated_gt_sha256=sha(dst),
                        vehicle_rows=converted['kept'], ignored_rows=converted['skipped'])
    if set(gt) != {'1_rgb_val_1-08', '2_rgb_val_2-08'} or sum(x['frames'] for x in gt.values()) != 1200:
        raise ValueError('M3OT RGB Val scope differs from the archived benchmark')
    return gt


def check_prediction(path, n):
    seen = set()
    with path.open() as stream:
        for row in csv.reader(stream):
            if (len(row) != 10 or not np.isfinite(list(map(float, row))).all()
                    or not 1 <= int(row[0]) <= n or int(row[1]) < 0 or min(map(float, row[4:6])) <= 0):
                raise ValueError(f'Invalid MOT row in {path}: {row}')
            key = (int(row[0]), int(row[1]))
            if key in seen:
                raise ValueError(f'Duplicate frame and track ID in {path}: {key}')
            seen.add(key)
    return len(seen)


def clear_gaps(dataset, clear, arm, seq, expected):
    """Observe the pinned CLEAR assignments and verify the reconstructed raw counts."""
    import trackeval.metrics.clear as clear_module

    raw = dataset.get_raw_seq_data(arm, seq)
    data = dataset.get_preprocessed_seq_data(raw, 'vehicle')
    for key in ('gt_dets', 'tracker_dets'):
        if any(not np.array_equal(a,b) for a,b in zip(raw[key],data[key],strict=True)):
            raise ValueError(f'Unexpected row filtering or reordering for {arm} {seq} {key}')
    times = iter(t for t, (g, p) in enumerate(zip(data['gt_ids'], data['tracker_ids'], strict=True))
                 if len(g) and len(p))
    history = {}
    gaps = []
    ids = []
    original_solver = clear_module.linear_sum_assignment

    def capture(cost):
        t = next(times)
        rows, cols = original_solver(cost)
        valid = -cost[rows, cols] > np.finfo(float).eps
        for i, j in zip(rows[valid], cols[valid], strict=True):
            gt, pred = int(raw['gt_ids'][t][i]), int(raw['tracker_ids'][t][j])
            ids.append((t, gt))
            if gt in history:
                last_frame, last_pred = history[gt]
                gaps.append(dict(gt_id=gt, previous_frame=last_frame+1, frame=t+1,
                                 gap=t-last_frame-1, previous_id=last_pred, current_id=pred,
                                 same_id=int(last_pred == pred)))
            history[gt] = (t, pred)
        return rows, cols

    clear_module.linear_sum_assignment = capture
    try:
        official = clear.eval_sequence(data)
    finally:
        clear_module.linear_sum_assignment = original_solver
    count = {k: int(official[v]) for k, v in
             dict(TP='CLR_TP', FP='CLR_FP', FN='CLR_FN', IDSW='IDSW').items()}
    if (next(times, None) is not None or count != {k: expected[k] for k in COUNT_METRICS}
            or len(set(ids)) != expected['TP'] or sum(1-r['same_id'] for r in gaps) != expected['IDSW']):
        raise ValueError(f'Unexpected CLEAR match reconstruction for {arm} {seq}')
    return gaps


def evaluate(project, tracker_root, output):
    sys.path.insert(0, str(project / 'tools'))
    from validate_evaluation import M3OTVehicle, trackeval
    from convert_gt_for_trackeval import convert_file

    track_eval = Path(trackeval.__file__).resolve().parents[1]
    commit = subprocess.check_output(['git', '-C', str(track_eval), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != TRACK_EVAL_COMMIT:
        raise ValueError(f'Unexpected TrackEval revision: {commit}')
    gt = ground_truth(project, output, convert_file)
    lengths = {seq: row['frames'] for seq, row in gt.items()}
    all_rows, all_gaps, fingerprints = [], [], {}
    for mode in DETECTORS:
        copied = output / mode / 'trackers'
        fingerprints[mode] = {}
        for arm in TRACKERS:
            src = tracker_root / mode / 'trackers' / arm / 'data'
            if {p.stem for p in src.glob('*.txt')} != set(gt):
                raise ValueError(f'Tracker sequence set differs from official Val: {src}')
            dst = copied / arm / 'data'
            dst.mkdir(parents=True, exist_ok=True)
            fingerprints[mode][arm] = {}
            for seq, frames in lengths.items():
                path = src / (seq+'.txt')
                count = check_prediction(path, frames)
                target = dst / path.name
                shutil.copyfile(path, target)
                if sha(path) != sha(target):
                    raise ValueError(f'Track file changed while copying: {path}')
                fingerprints[mode][arm][seq] = {'sha256': sha(target), 'rows': count}
        ds = M3OTVehicle(dict(GT_FOLDER=str(output/'gt'), TRACKERS_FOLDER=str(copied),
                             OUTPUT_FOLDER=str(output/mode/'metrics'), TRACKERS_TO_EVAL=list(TRACKERS),
                             BENCHMARK='M3OT', SPLIT_TO_EVAL='val', SKIP_SPLIT_FOL=True, SEQ_INFO=lengths,
                             DO_PREPROC=False, PRINT_CONFIG=False))
        clear = trackeval.metrics.CLEAR({'PRINT_CONFIG': False})
        ev = trackeval.Evaluator(dict(PRINT_CONFIG=False, PRINT_RESULTS=False, TIME_PROGRESS=False,
                                      DISPLAY_LESS_PROGRESS=True, PLOT_CURVES=False, BREAK_ON_ERROR=True,
                                      LOG_ON_ERROR=str(output/mode/'trackeval_errors.log')))
        result, _ = ev.evaluate([ds], [trackeval.metrics.HOTA(), clear, trackeval.metrics.Identity()])
        for method, values in result['M3OTVehicle'].items():
            for seq, classes in values.items():
                raw = classes['vehicle']
                hota, counts = raw['HOTA'], raw['CLEAR']
                row = dict(detector=mode, arm=method, sequence=seq,
                           **{k: float(np.mean(hota[k])*100) for k in ('HOTA','DetA','AssA')},
                           IDF1=float(raw['Identity']['IDF1']*100), MOTA=float(counts['MOTA']*100),
                           TP=int(counts['CLR_TP']), FP=int(counts['CLR_FP']),
                           FN=int(counts['CLR_FN']), IDSW=int(counts['IDSW']))
                if seq in gt:
                    if row['TP']+row['FN'] != gt[seq]['vehicle_rows']:
                        raise ValueError(f'GT count mismatch: {seq} {method}')
                    if row['TP']+row['FP'] != fingerprints[mode][method][seq]['rows']:
                        raise ValueError(f'Prediction count mismatch: {seq} {method}')
                    all_gaps.extend(dict(detector=mode, arm=method, sequence=seq, **g)
                                    for g in clear_gaps(ds, clear, method, seq, row))
                all_rows.append(row)

    lookup = {(r['detector'], r['arm'], r['sequence']):r for r in all_rows}
    deltas = []
    for mode in DETECTORS:
        for arm in TRACKERS:
            pool = lookup[mode,arm,'COMBINED_SEQ']
            for k in COUNT_METRICS:
                if pool[k] != sum(lookup[mode,arm,s][k] for s in gt):
                    raise ValueError(f'Combined count mismatch: {mode} {arm} {k}')
            for seq in (*gt,'COMBINED_SEQ'):
                a, b = lookup[mode,arm,seq], lookup[mode,'native',seq]
                deltas.append(dict(detector=mode, arm=arm, sequence=seq,
                                   **{k: a[k]-b[k] for k in METRICS}))
    delta_lookup = {(r['detector'],r['arm'],r['sequence']):r for r in deltas}
    interaction = [dict(arm=arm, sequence=seq, **{
        k: delta_lookup['higher',arm,seq][k]-delta_lookup['lower',arm,seq][k] for k in METRICS})
        for arm in TRACKERS[1:] for seq in (*gt,'COMBINED_SEQ')]
    write_csv(output/'metrics.csv', all_rows)
    write_csv(output/'delta_vs_native.csv', deltas)
    write_csv(output/'detector_interaction.csv', interaction)
    if all_gaps:
        write_csv(output/'clear_identity_intervals.csv', all_gaps)
    gap_counts = Counter()
    for item in all_gaps:
        n = item['gap']
        group = '0' if n == 0 else '1-3' if n <= 3 else '4-10' if n <= 10 else '11-30' if n <= 30 else '31+'
        for seq in (item['sequence'], 'COMBINED_SEQ'):
            key = (item['detector'], item['arm'], seq, group)
            gap_counts[key, 'pairs'] += 1
            gap_counts[key, 'id_switches'] += 1-item['same_id']
    gap_rows = []
    for key, name in sorted(gap_counts):
        if name != 'pairs':
            continue
        pairs, switches = gap_counts[key,'pairs'], gap_counts[key,'id_switches']
        mode, arm, seq, group = key
        gap_rows.append(dict(detector=mode, arm=arm, sequence=seq, gap_group=group, pairs=pairs,
                             id_switches=switches, same_id_rate=(pairs-switches)/pairs))
    if gap_rows:
        write_csv(output/'gap_summary.csv', gap_rows)
    write_json(output/'evaluation.json', dict(status='PASS', trackeval_commit=commit,
               gt=gt, prediction_hashes=fingerprints, metrics=all_rows, paired_deltas=deltas,
               detector_interaction=interaction, clear_gap_summary=gap_rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--tracker-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    evaluate(args.project, args.tracker_root, args.output)
