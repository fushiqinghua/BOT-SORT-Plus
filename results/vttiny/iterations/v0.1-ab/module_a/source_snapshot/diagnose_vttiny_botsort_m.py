"""Attribute frozen BoT-SORT test errors. CLEAR matching follows the pinned TrackEval release."""
import argparse
import csv
import gzip
import json
import math
import os
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from vttiny_tracking_common import ROOT, TRACK_EVAL


RUN = Path(os.environ.get('VTTINY_BASELINE_RESULTS', ROOT/'outputs/yolo11m_rgb_rgbt_mot_test_v1'))
OUTPUT = Path(os.environ.get('VTTINY_DIAGNOSIS_OUTPUT', ROOT/'outputs/botsort_m_test_diagnosis_v1'))
GT_FOLDER = Path(os.environ.get('VTTINY_GT_FOLDER', ROOT/'data/processed_v1/mot/rgb/test'))
IMAGE_ROOT = Path(os.environ.get('VTTINY_FRAME_ROOT', ROOT/'data/raw/VT-Tiny-MOT'))
VARIANTS = {'main': ('rgbt', 'botsort_reid'), 'no_reid': ('rgbt', 'botsort_no_reid'),
            'rgb': ('rgb', 'botsort_reid')}
EPS = np.finfo(float).eps
sys.path.insert(0, str(TRACK_EVAL))
from trackeval.datasets._base_dataset import _BaseDataset


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_detections(directory, sequence):
    plain = directory / (sequence + '.json')
    if plain.is_file():
        return load(plain)
    with gzip.open(plain.with_suffix('.json.gz'), 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def frame_path(name):
    saved_prefix = 'VTTINY_ROOT/data/raw/VT-Tiny-MOT/'
    return IMAGE_ROOT / Path(name.removeprefix(saved_prefix))


def mot(path, n, ground_truth=False):
    rows = [[] for _ in range(n)]
    with path.open() as fh:
        for row in csv.reader(fh):
            if not row or (ground_truth and int(float(row[6])) == 0):
                continue
            rows[int(row[0]) - 1].append([float(x) for x in row[1:7]])
    return [np.asarray(frame, dtype=float).reshape(-1, 6) for frame in rows]


def overlaps(a, b):
    return _BaseDataset._calculate_box_ious(a, b, box_format='xywh')


def clear_step(gt, tracks, previous):
    iou = overlaps(gt[:, 1:5], tracks[:, 1:5])
    if not len(gt) or not len(tracks):
        return {}, iou
    ids = gt[:, 0].astype(int)
    prev = np.asarray([previous.get(g, -1) for g in ids])
    score = (prev[:, None] == tracks[:, 0]) * 1000 + iou
    score[iou < .5 - EPS] = 0
    a, b = linear_sum_assignment(-score)
    matches = {i: j for i, j in zip(a, b) if score[i, j] > EPS}
    previous.clear()
    previous.update({ids[i]: int(tracks[j, 0]) for i, j in matches.items()})
    return matches, iou


def candidate(iou, scores):
    best = float(iou.max(initial=0))
    ok = iou >= .5 - EPS
    conf = float(scores[ok].max()) if ok.any() else -1.
    if conf < 0:
        label = 'overlap_lt10' if best < .1 else ('overlap_10_30' if best < .3 else 'overlap_30_50')
    elif conf >= .6:
        label = 'birth_eligible'
    elif conf > .5:
        label = 'high_below_birth'
    elif conf > .1 and conf != .5:
        label = 'low_assoc'
    else:
        label = 'outside_assoc_gate'
    return label, best, conf


def table_writer(stack, path, names):
    stream = gzip.open(path, 'wt', encoding='utf-8', newline='') if path.suffix == '.gz' else path.open('w', encoding='utf-8', newline='')
    fh = stack.enter_context(stream)
    writer = csv.writer(fh)
    writer.writerow(names.split())
    return writer


def gap_bin(n):
    if n == 0:
        return '0'
    if n <= 3:
        return '1-3'
    if n <= 10:
        return '4-10'
    if n <= 30:
        return '11-30'
    return '31+'


def size_bin(box):
    side = math.sqrt(max(0., box[2] * box[3]))
    return '<8' if side < 8 else ('8-16' if side < 16 else ('16-32' if side < 32 else '32+'))


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    protocols = {m: load(RUN / m / 'protocol.json') for m in ('rgbt', 'rgb')}
    main = protocols['rgbt']
    configs = main['trackers']['botsort_reid']
    assert [configs[k] for k in ('track_low_thresh','track_high_thresh','new_track_thresh')] == [.1, .5, .6]
    assert main['detector']['conf'] == .05
    metrics = {}
    for v, (mode, tracker) in VARIANTS.items():
        rows = load(RUN / mode / 'metrics/summary.json')['results']
        metrics[v] = {r['sequence']:r for r in rows if r['tracker'] == tracker}
    counts = defaultdict(Counter)
    seq_results = []
    with ExitStack() as stack:
        frameout = table_writer(stack, out/'gt_frames.csv.gz', 'sequence frame gt_id width height size track_id rgb_track_id no_reid_track_id candidate max_det_iou max_aligned_score')
        fpout = table_writer(stack, out/'false_positives.csv.gz', 'sequence frame track_id score max_gt_iou type')
        swaps = table_writer(stack, out/'switches.csv', 'sequence gt_id previous_frame frame previous_track track gap missing_annotations')
        tracks_out = table_writer(stack, out/'gt_trajectories.csv', 'sequence gt_id gt_frames tracked first_gt last_gt first_match last_match distinct_track_ids')
        gaps_out = table_writer(stack, out/'internal_gaps.csv', 'sequence gt_id before after missed_count start_track end_track missing_annotations')
        for seq, frame_paths in main['sequences'].items():
            n = len(frame_paths)
            gt_frames = mot(GT_FOLDER/seq/'gt/gt.txt', n, True)
            outputs = {v:mot(RUN/m/'trackers'/t/'data'/(seq+'.txt'), n) for v,(m,t) in VARIANTS.items()}
            cached = load_detections(RUN/'rgbt/detections', seq)
            assert cached['model_sha256'] == main['weight_sha256'] and len(cached['frames']) == n
            previous = {v:{} for v in VARIANTS}
            last = {v:{} for v in VARIANTS}
            per = {v:Counter() for v in VARIANTS}
            histories = defaultdict(list)
            sr = Counter()
            for f, (gt, d, path) in enumerate(zip(gt_frames, cached['frames'], frame_paths), 1):
                assert d['frame'] == f == path['frame'] and frame_path(d['image']) == frame_path(path['image'])
                det = np.asarray(d['detections'], dtype=float).reshape(-1, 6)
                boxes = det[:, :4].copy()
                boxes[:, 2:] -= boxes[:, :2]
                diou = overlaps(gt[:, 1:5], boxes)
                match = {}
                for v in VARIANTS:
                    preds = outputs[v][f-1]
                    ij, sim = clear_step(gt, preds, previous[v])
                    match[v] = {i:int(preds[j, 0]) for i, j in ij.items()}
                    per[v].update({'TP':len(ij), 'FN':len(gt)-len(ij), 'FP':len(preds)-len(ij)})
                    for i, tid in match[v].items():
                        gid = int(gt[i, 0])
                        if gid in last[v] and last[v][gid] != tid:
                            per[v]['IDSW'] += 1
                        last[v][gid] = tid
                    if v != 'main':
                        continue
                    matched_preds = set(ij.values())
                    for j, pred in enumerate(preds):
                        if j in matched_preds:
                            continue
                        best = float(sim[:, j].max(initial=0))
                        label = ('empty_gt_frame' if not len(gt) else 'overlap_lt10' if best < .1 else
                                 'overlap_10_30' if best < .3 else 'overlap_30_50' if best < .5-EPS else 'overlap_ge50')
                        counts['false_positive'][label] += 1
                        fpout.writerow([seq,f,int(pred[0]),round(pred[5],6),round(best,6),label])
                for i, g in enumerate(gt):
                    gid = int(g[0])
                    tid = match['main'].get(i, -1)
                    label, best, conf = candidate(diou[i], det[:, 4])
                    size = size_bin(g[1:5])
                    counts['size_gt'][size] += 1
                    if tid < 0:
                        counts['fn_by_candidate'][label] += 1
                        counts['size_fn'][size] += 1
                        counts['size_candidate'][size+'|'+label] += 1
                        sr[label] += 1
                    for v in ('rgb', 'no_reid'):
                        state = ('main_tp' if tid >= 0 else 'main_fn')+'|'+('control_tp' if i in match[v] else 'control_fn')
                        counts['paired_'+v][state] += 1
                    histories[gid].append((f,tid,label))
                    frameout.writerow([seq,f,gid,g[3],g[4],size,tid,match['rgb'].get(i,-1),match['no_reid'].get(i,-1),
                                       label,round(best,6),round(conf,6)])
            for gid, h in histories.items():
                hits = [(i,f,t) for i,(f,t,_) in enumerate(h) if t >= 0]
                first = hits[0][0] if hits else None
                tail = hits[-1][0] if hits else None
                tracks_out.writerow([seq,gid,len(h),len(hits),h[0][0],h[-1][0],h[first][0] if hits else '',h[tail][0] if hits else '',len({t for _,_,t in hits})])
                counts['trajectories']['with_match' if hits else 'never_matched'] += 1
                for i,(_,t,label) in enumerate(h):
                    if t >= 0:
                        continue
                    phase = 'never_matched' if first is None else ('before_first' if i < first else 'after_last' if i > tail else 'internal')
                    counts['fn_temporal'][phase] += 1
                    counts['phase_candidate'][phase+'|'+label] += 1
                for (i1,f1,t1),(i2,f2,t2) in zip(hits,hits[1:]):
                    gap = f2-f1-1
                    missed = i2-i1-1
                    absent = gap-missed
                    gbin = gap_bin(gap)
                    counts['transitions'][gbin] += 1
                    if t1 != t2:
                        counts['switches'][gbin] += 1
                        counts['switch_annotation']['gt_interrupted' if absent else 'gt_present_throughout'] += 1
                        swaps.writerow([seq,gid,f1,f2,t1,t2,gap,absent])
                    if missed:
                        gaps_out.writerow([seq,gid,f1,f2,missed,t1,t2,absent])
                        counts['internal_gap_fn'][gbin] += missed
                        if not absent:
                            counts['continuous_internal_gap_fn'][gbin] += missed
            for v in VARIANTS:
                for k in ('FP','FN','IDSW'):
                    assert per[v][k] == metrics[v][seq][k], (v,seq,k,per[v],metrics[v][seq])
                assert per[v]['TP']+per[v]['FN'] == sum(map(len,gt_frames))
                counts['official_'+v].update(per[v])
            seq_results.append(dict(sequence=seq, frames=n, gt=sum(map(len,gt_frames)), counts=dict(per['main']),
                                    fn_candidates=dict(sr), official={v:metrics[v][seq] for v in VARIANTS}))
    for v in VARIANTS:
        for k in ('FN','FP','IDSW'):
            assert counts['official_'+v][k] == metrics[v]['COMBINED_SEQ'][k]
    totals = counts['official_main']
    assert sum(counts['fn_by_candidate'].values()) == sum(counts['fn_temporal'].values()) == totals['FN']
    assert sum(counts['switches'].values()) == totals['IDSW'] and sum(counts['false_positive'].values()) == totals['FP']
    assert sum(counts['internal_gap_fn'].values()) == counts['fn_temporal']['internal']
    assert len(seq_results) == 30 and sum(s['frames'] for s in seq_results) == 12590
    output = dict(source=str(RUN),protocol=main['evaluation'],detector=main['weight_sha256'],reid=main['reid_sha256'],
                  rgb_detector=protocols['rgb']['weight_sha256'],config=configs,counts=dict(counts),sequences=seq_results,
                  metrics={v:metrics[v]['COMBINED_SEQ'] for v in VARIANTS},checks='per-sequence CLEAR FN, FP, IDSW match frozen TrackEval for all three variants')
    (out/'diagnosis.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(counts,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=OUTPUT)
    run(parser.parse_args().out)
