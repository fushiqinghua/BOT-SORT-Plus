"""Verify and score four frozen-input Q/R runs using official VT-Tiny-MOT RGB GT."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from qr_settings import INT, MOTION, ORIG, OUT, ROOT, SETTINGS, environment, read, sha, table, write


METHODS = ('native', 'learned_q', 'learned_r', 'learned_qr')
HORIZONS = (3, 5, 10)
METRICS = ('HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'FP', 'FN', 'IDSW')
POSITION = ('center_error_px', 'center_error_gt_diagonal', 'width_abs_error_px', 'height_abs_error_px',
            'width_abs_error_relative', 'height_abs_error_relative', 'iou')
COMMON_INPUTS = ('sequences', 'subset', 'detection_fingerprints', 'feature_hashes', 'decoded_image_hashes',
                 'source_protocol_sha256', 'detector_sha256', 'rgb_encoder_sha256', 'encoder_contracts',
                 'native_config', 'boxmot_source_hashes')
POLICY_FILES = {'native': (), 'learned_q': ('q_policy.json',), 'learned_r': ('r_policy.json',),
                'learned_qr': ('q_policy.json','r_policy.json')}


def verify_inputs(run_folder, historical):
    protocols = {m: read(run_folder/'protocols'/f'{m}.json') for m in METHODS}
    prior_path = historical/'protocols/native.json'
    prior = read(prior_path)
    seqs = prior['sequences']
    if len(seqs) != 30 or sum(seqs.values()) != 12590 or prior['subset'] is not False:
        raise ValueError('The historical reference must be the full VT-Tiny-MOT official test run')
    source_sha = sha(Path(__file__).with_name('replay.py'))
    policy_sha = {filename: sha(OUT/filename) for filename in ('q_policy.json', 'r_policy.json')}
    for name, protocol in protocols.items():
        if protocol['status'] != 'COMPLETE':
            raise ValueError(f'{name} has incomplete predictions')
        for key in COMMON_INPUTS:
            if protocol[key] != prior[key]:
                raise ValueError(f'{name} has different frozen inputs or native configuration: {key}')
        expected = {filename: policy_sha[filename] for filename in POLICY_FILES[name]}
        if protocol['learned_covariance'] != dict(method=name, policies=expected, replay_sha256=source_sha):
            raise ValueError(f'{name} has different learned covariance inputs or replay source')
        event_folder = run_folder/'qr_events'/name
        if {f.name for f in event_folder.glob('*.jsonl.gz')} != {s+'.jsonl.gz' for s in seqs}:
            raise ValueError(f'{name} has missing or extraneous detection-index logs')
    native = protocols['native']['output_prediction_hashes']
    if native != prior['output_prediction_hashes']:
        raise ValueError('Native predictions differ from the historical run')
    for seq, expected in native.items():
        local = run_folder/'trackers/native/data'/f'{seq}.txt'
        if sha(local) != expected:
            raise ValueError(f'{seq}: native MOT file differs from its matches in the historical protocol')
    return protocols, dict(sequences=seqs, frames=sum(seqs.values()), native_exact=len(seqs),
                           historical_protocol_sha256=sha(prior_path), compared_fields=list(COMMON_INPUTS),
                           replay_sha256=source_sha, policy_sha256=policy_sha)


def install_evaluation_paths(env):
    os.environ['TRACKEVAL_ROOT'] = env['TRACKEVAL_ROOT']
    sys.path[:0] = [str(ORIG/'tools'), str(ORIG), *env['PYTHONPATH'].split(os.pathsep)]


def score_official(run_folder, output, gt_root, env):
    args = [sys.executable, '-Bu', str(ORIG/'tools/evaluate_botsort_plus.py'), '--run-dir', str(run_folder),
            '--trackers', *METHODS, '--reference-tracker', 'native', '--gt-root', str(gt_root),
            '--trackeval-root', env['TRACKEVAL_ROOT'], '--output-dir', str(output)]
    subprocess.run(args, env=env, check=True)
    rows = read(output/'summary.json')['results']
    return {(r['tracker'], r['sequence']): r for r in rows}


def event_frames(path, frame_count):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        frames = [json.loads(line) for line in f]
    if [r['frame'] for r in frames] != list(range(1, frame_count+1)):
        raise ValueError(f'{path}: expected one readable event record for every video frame')
    return frames


def clear_assignments(dataset, method, sequence):
    from trackeval.metrics import CLEAR
    import trackeval.metrics.clear as clear_module

    raw = dataset.get_raw_seq_data(method, sequence)
    data = dataset.get_preprocessed_seq_data(raw, 'object')
    gt_masks = [np.asarray(extra['zero_marked']) != 0 for extra in raw['gt_extras']]
    active = iter(t for t, (ground, pred) in enumerate(zip(data['gt_ids'], data['tracker_ids'], strict=True))
                  if len(ground) and len(pred))
    matches = [{} for _ in gt_masks]
    solve = clear_module.linear_sum_assignment

    def record(costs):
        t = next(active)
        rows, cols = solve(costs)
        good = -costs[rows, cols] > np.finfo(float).eps
        matches[t] = {int(col): int(row) for row, col in zip(rows[good], cols[good], strict=True)}
        return rows, cols

    clear_module.linear_sum_assignment = record
    try:
        result = CLEAR({'PRINT_CONFIG': False}).eval_sequence(data)
    finally:
        clear_module.linear_sum_assignment = solve
    if next(active, None) is not None or result['CLR_TP'] != sum(map(len, matches)):
        raise ValueError(f'{method}/{sequence}: captured assignment differs from official CLEAR counts')
    for t, mask in enumerate(gt_masks):
        if (not np.array_equal(data['tracker_dets'][t], raw['tracker_dets'][t])
                or not np.array_equal(data['gt_dets'][t], raw['gt_dets'][t][mask])):
            raise ValueError(f'{method}/{sequence}/{t+1}: unexpected official coordinate preprocessing')
    return raw, data, gt_masks, matches, result


def geometry(tracker_box, truth_box, overlap):
    x, y, width, height = tracker_box
    tx, ty, twidth, theight = truth_box
    dx = x - tx + (width - twidth)/2
    dy = y - ty + (height - theight)/2
    center = np.hypot(dx, dy)
    werr, herr = abs(width - twidth), abs(height - theight)
    return dict(zip(POSITION, (float(center), float(center/np.hypot(twidth, theight)), float(werr), float(herr),
                               float(werr/twidth), float(herr/theight), float(overlap)), strict=True))


def observe_sequence(dataset, method, seq, detections, events, official):
    raw, data, gt_masks, matches, clear = clear_assignments(dataset, method, seq)
    if tuple(int(clear[k]) for k in ('CLR_FP','CLR_FN','IDSW')) != tuple(official[k] for k in ('FP','FN','IDSW')):
        raise ValueError(f'{method}/{seq}: official scores differ from replayed CLEAR assignments')
    positions, ground, by_gt, by_track, det_by_track = {}, [], [], [], []
    for offset, (det_rows, ledger, mask, assignments) in enumerate(zip(detections, events, gt_masks, matches, strict=True)):
        orig_ids = raw['tracker_ids'][offset]
        tracker_xywh = data['tracker_dets'][offset]
        scores = raw['tracker_confidences'][offset]
        gts = raw['gt_ids'][offset][mask]
        ledger_rows = {int(tid): int(det) for tid, det in ledger['rows']}
        if (len(ledger_rows) != len(ledger['rows']) or len(set(ledger_rows.values())) != len(ledger_rows)
                or set(ledger_rows) != set(map(int, orig_ids))):
            raise ValueError(f'{method}/{seq}/{offset+1}: event detection-index ledger does not match MOT IDs')
        ground.append(set(map(int, gts)))
        by_gt.append({})
        by_track.append({})
        det_by_track.append(ledger_rows)
        for col, (tid, output_score) in enumerate(zip(orig_ids, scores, strict=True)):
            tid = int(tid)
            det = ledger_rows[tid]
            if not 0 <= det < len(det_rows):
                raise ValueError(f'{method}/{seq}/{offset+1}: MOT output has no frozen detection row {det}')
            score = float(det_rows[det, 4])
            if not np.isclose(output_score, score, rtol=1e-6, atol=1e-8):
                raise ValueError(f'{method}/{seq}/{offset+1}: frozen detection score does not match output row')
            if col not in assignments:
                continue
            row = assignments[col]
            gt = int(gts[row])
            similarity = data['similarity_scores'][offset][row, col]
            positions[(offset + 1, gt, det)] = dict(score=score, **geometry(tracker_xywh[col], data['gt_dets'][offset][row], similarity))
            by_gt[offset][gt] = tid
            by_track[offset][tid] = gt
    if len(positions) != clear['CLR_TP']:
        raise ValueError(f'{method}/{seq}: localization records do not match the official true positives')
    return positions, ground, by_gt, by_track, det_by_track


def follow_recoveries(method, seq, events, ground, by_gt, by_track, det_rows):
    event_rows, window_rows = [], []
    for record in events:
        frame = record['frame']
        for tid, anchor, anchor_det, current_det in record['recovered']:
            if not 1 <= anchor < frame:
                raise ValueError(f'{method}/{seq}/{frame}: invalid pre-gap anchor frame {anchor}')
            start, end = anchor-1, frame-1
            for index, logged in ((start, anchor_det), (end, current_det)):
                observed = det_rows[index].get(tid)
                if observed is not None and observed != logged:
                    raise ValueError(f'{method}/{seq}/{frame}: recovery event conflicts with the output detection index')
            before, after = by_track[start].get(tid), by_track[end].get(tid)
            if before is None:
                status = 'anchor_unmatched'
            elif tid not in det_rows[end]:
                status = 'recovery_not_emitted'
            elif after is None:
                status = 'recovery_unmatched'
            else:
                status = 'same_gt' if before == after else 'different_gt'
            identity = dict(method=method, sequence=seq, frame=frame, tid=tid, anchor_frame=anchor,
                            anchor_det=anchor_det, current_det=current_det, gap=frame-anchor, gt=before)
            event_rows.append(dict(**identity, recovery_gt=after, status=status))
            if status != 'same_gt':
                continue
            for horizon in HORIZONS:
                last = end+horizon
                if last >= len(ground):
                    complete, endpoint = 'clip_end', 'clip_end'
                elif before not in ground[last]:
                    complete, endpoint = 'gt_unobservable', 'gt_unobservable'
                else:
                    endpoint = retention_at(last, before, tid, by_gt, by_track)
                    if any(before not in ids for ids in ground[end:last+1]):
                        complete = 'gt_unobservable'
                    else:
                        statuses = (retention_at(t, before, tid, by_gt, by_track) for t in range(end+1, last+1))
                        complete = next((s for s in statuses if s != 'held'), 'held')
                window_rows.append(dict(**identity, horizon=horizon, endpoint=endpoint, continuous=complete))
    return event_rows, window_rows


def retention_at(frame, gid, tid, by_gt, by_track):
    if by_gt[frame].get(gid) == tid:
        return 'held'
    return 'different_id' if gid in by_gt[frame] or tid in by_track[frame] else 'missed'


def strata(rows, low, high):
    split = min(high, max(low, .3))
    return {'all': rows, 'low': [r for r in rows if low < r['score'] <= high],
            'low_le_0p30': [r for r in rows if low < r['score'] <= split],
            'low_gt_0p30': [r for r in rows if split < r['score'] <= high],
            'high': [r for r in rows if r['score'] > high]}


def stat_rows(scope, methods, low, high, paired=False):
    buckets = {m: strata(rows, low, high) for m, rows in methods.items()}
    report = []
    for method, groups in buckets.items():
        for label, rows in groups.items():
            target = {'cohort': 'four_way_same_detection' if paired else 'all_official_clear_tp',
                      'sequence': scope, 'method': method, 'score_stratum': label, 'n': len(rows)}
            for key in POSITION:
                values = np.asarray([row[key] for row in rows])
                target[key+'_mean'] = float(values.mean()) if len(values) else None
                target[key+'_median'] = float(np.median(values)) if len(values) else None
                target[key+'_p90'] = float(np.quantile(values, .9)) if len(values) else None
                if paired:
                    baseline = np.asarray([row[key] for row in buckets['native'][label]])
                    if len(values) != len(baseline):
                        raise ValueError('Paired localization strata differ in size')
                    delta = values-baseline
                    target[key+'_mean_delta_native'] = float(delta.mean()) if len(delta) else None
                    target[key+'_median_delta_native'] = float(np.median(delta)) if len(delta) else None
            report.append(target)
    return report


def report_recoveries(output, events, windows):
    event_columns = ('method','sequence','frame','tid','anchor_frame','anchor_det','current_det','gap','gt','recovery_gt','status')
    window_columns = (*event_columns[:-2], 'horizon','endpoint','continuous')
    write_csv(output/'recovery_events.csv', events, event_columns)
    write_csv(output/'recovery_windows.csv', windows, window_columns)
    summary = []
    for method in METHODS:
        counts = Counter(row['status'] for row in events if row['method'] == method)
        for horizon in HORIZONS:
            for policy in ('endpoint','continuous'):
                results = Counter(r[policy] for r in windows if r['method'] == method and r['horizon'] == horizon)
                held = results['held']
                eligible = sum(results[k] for k in ('held','different_id','missed'))
                summary.append(dict(method=method,horizon=horizon,policy=policy,recoveries=sum(counts.values()),
                                    same_gt_at_recovery=counts['same_gt'],different_gt_at_recovery=counts['different_gt'],
                                    recovery_unmatched=counts['recovery_unmatched'],recovery_not_emitted=counts['recovery_not_emitted'],
                                    anchor_unmatched=counts['anchor_unmatched'],eligible=eligible,held=held,
                                    retention_rate=held/eligible if eligible else None,different_id=results['different_id'],
                                    missed=results['missed'],gt_unobservable=results['gt_unobservable'],clip_end=results['clip_end']))
    table(output/'recovery_retention.csv', summary)

    exact = {m: {} for m in METHODS}
    key_fields = ('sequence','anchor_frame','frame','gt','anchor_det','current_det','horizon')
    for row in windows:
        lookup = exact[row['method']]
        key = tuple(row[f] for f in key_fields)
        if key in lookup:
            raise ValueError(f'Duplicate common recovery window in {row["method"]}: {key}')
        lookup[key] = row
    common = sorted(set.intersection(*(set(rows) for rows in exact.values())))
    paired_windows = []
    for key in common:
        for method in METHODS:
            row = exact[method][key]
            paired_windows.append(dict(**dict(zip(key_fields,key,strict=True)), method=method, tid=row['tid'],
                                       endpoint=row['endpoint'], continuous=row['continuous']))
    write_csv(output/'recovery_windows_four_way.csv', paired_windows, (*key_fields,'method','tid','endpoint','continuous'))
    pooled = []
    for horizon in HORIZONS:
        for policy in ('endpoint', 'continuous'):
            available = [key for key in common if key[-1]==horizon and exact['native'][key][policy] not in ('clip_end','gt_unobservable')]
            for method in METHODS:
                base = [exact['native'][key][policy]=='held' for key in available]
                other = [exact[method][key][policy]=='held' for key in available]
                pooled.append(dict(method=method,horizon=horizon,policy=policy,eligible=len(available),held=sum(other),
                                   retained_in_both=sum(a and b for a,b in zip(base,other,strict=True)),
                                   gained=sum(not a and b for a,b in zip(base,other,strict=True)),
                                   lost=sum(a and not b for a,b in zip(base,other,strict=True)),
                                   retention_rate=sum(other)/len(available) if available else None))
    table(output/'recovery_retention_four_way.csv', pooled)


def write_csv(path, rows, fields):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare_localization(run_folder, output, gt_root, protocols, official):
    from evaluate_vttiny import VTTinyObjects
    from botsort_plus.io import detection_fingerprint, read_detection_cache

    protocol = protocols['native']
    seqs = protocol['sequences']
    settings = read(SETTINGS)
    cache_folder = Path(settings['baseline_root'])/'detections'
    gt_folder = str(gt_root)
    tracker_folder = str(run_folder/'trackers')
    dataset = VTTinyObjects(dict(GT_FOLDER=gt_folder, TRACKERS_FOLDER=tracker_folder, OUTPUT_FOLDER=str(output/'clear'),
        TRACKERS_TO_EVAL=list(METHODS), BENCHMARK='VT-Tiny-MOT', SPLIT_TO_EVAL='test', SKIP_SPLIT_FOL=True,
        SEQ_INFO=seqs, DO_PREPROC=False, PRINT_CONFIG=False))
    low, high = (protocol['native_config'][k] for k in ('track_low_thresh','track_high_thresh'))
    pooled, paired = ({m: [] for m in METHODS} for _ in range(2))
    memberships, loc_summary, recovered, windows = [], [], [], []
    for seq, frame_count in sorted(seqs.items()):
        det_file = cache_folder/f'{seq}.json.gz'
        if not det_file.is_file():
            det_file = det_file.with_suffix('')
        cache = read_detection_cache(det_file)
        if detection_fingerprint(cache) != protocol['detection_fingerprints'][seq]:
            raise ValueError(f'{seq}: current cache differs from the countersigned detection protocol')
        detections = [np.asarray(frame['detections'], np.float32).reshape(-1,6) for frame in cache['frames']]
        by_method = {}
        for method in METHODS:
            logs = event_frames(run_folder/'qr_events'/method/f'{seq}.jsonl.gz', frame_count)
            positions, *tracks = observe_sequence(dataset, method, seq, detections, logs, official[method,seq])
            events, method_windows = follow_recoveries(method, seq, logs, *tracks)
            by_method[method] = positions
            recovered.extend(events)
            windows.extend(method_windows)
        common = sorted(set.intersection(*(set(rows) for rows in by_method.values())))
        seq_values = {m: list(rows.values()) for m, rows in by_method.items()}
        seq_paired = {m: [rows[k] for k in common] for m, rows in by_method.items()}
        for key in common:
            for method in METHODS:
                value = by_method[method][key]
                if value['score'] != by_method['native'][key]['score']:
                    raise ValueError(f'{seq}/{key}: shared detection row has different scores')
                frame, gid, det = key
                memberships.append(dict(sequence=seq, frame=frame, gt_id=gid, det_index=det, method=method, **value))
        for method in METHODS:
            pooled[method].extend(seq_values[method])
            paired[method].extend(seq_paired[method])
        loc_summary.extend(stat_rows(seq, seq_values, low, high))
        loc_summary.extend(stat_rows(seq, seq_paired, low, high, paired=True))
    combined = stat_rows('COMBINED_SEQ', pooled, low, high) + stat_rows('COMBINED_SEQ', paired, low, high, paired=True)
    native_gt_count = len(pooled['native']) + official['native','COMBINED_SEQ']['FN']
    for method in METHODS:
        if len(pooled[method]) + official[method,'COMBINED_SEQ']['FN'] != native_gt_count:
            raise ValueError('The official GT count differs across tracker arms')
    table(output/'localization_combined.csv', combined)
    table(output/'localization_by_sequence.csv', loc_summary)
    write_csv(output/'localization_paired_rows.csv.gz', memberships, ('sequence','frame','gt_id','det_index','method','score',*POSITION))
    report_recoveries(output, recovered, windows)
    return dict(low_score_threshold=low, high_score_threshold=high, common_rows=len(paired['native']),
                official_true_positives={m:len(pooled[m]) for m in METHODS}, recovery_events=len(recovered))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-run', type=Path, default=INT/'native_botsort_gmc_factorial_v1/full')
    args = parser.parse_args()
    run_folder = ROOT/'full'
    output = OUT
    settings = read(SETTINGS)
    gt_root = Path(settings['gt_root'])
    output.mkdir(parents=True, exist_ok=True)
    protocols, audit = verify_inputs(run_folder, args.historical_run)
    env = environment()
    official = score_official(run_folder, output/'metrics', gt_root, env)
    historical_summary = args.historical_run/'metrics/summary.json'
    if not historical_summary.is_file():
        historical_summary = MOTION/'full/metrics/summary.json'
    previous_native = {r['sequence']:r for r in read(historical_summary)['results'] if r['tracker']=='native'}
    metric_rows = []
    for seq in [*sorted(audit['sequences']), 'COMBINED_SEQ']:
        reference = official['native',seq]
        if any(abs(reference[k]-previous_native[seq][k]) > 1e-10 for k in METRICS):
            raise ValueError(f'{seq}: native metrics differ from the historical evaluation')
        for method in METHODS:
            row = official[method,seq]
            metric_rows.append(dict(**row, **{f'delta_{k}':row[k]-reference[k] for k in METRICS}))
    table(output/'tracking_by_sequence.csv', metric_rows)
    table(output/'tracking_combined.csv', [row for row in metric_rows if row['sequence']=='COMBINED_SEQ'])
    install_evaluation_paths(env)
    diagnosis = compare_localization(run_folder, output, gt_root, protocols, official)
    evaluator = read(output/'metrics/evaluation_audit.json')
    write(output/'score_audit.json', dict(status='PASS', **audit, **diagnosis, native_metric_parity=True,
          historical_summary_sha256=sha(historical_summary), trackeval_commit=evaluator['evaluator']['commit'],
          gt_sha256=evaluator['ground_truth']['sha256'], scope='full_official_test_development',
          location_cohorts=['all_official_clear_tp','four_way_same_detection'], recovery_horizons=list(HORIZONS)))
    print(json.dumps(dict(status='COMPLETE', frames=audit['frames'], **diagnosis)), flush=True)


if __name__ == '__main__':
    main()
