"""Decompose official CLEAR FN deltas into changes on the same annotated target and frame."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np


def load_original_scorer(qr):
    # Import by explicit filename because different experiment folders contain score.py.
    sys.path.insert(0, str(qr))
    spec = importlib.util.spec_from_file_location('archived_qr_scorer', qr/'score.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def emit(path, rows):
    if not rows:
        raise ValueError(f'Unexpected empty cohort: {path}')
    fields = tuple(rows[0])
    path.parent.mkdir(exist_ok=True,parents=True)
    writer = gzip.open if path.suffix == '.gz' else open
    with writer(path,'wt',encoding='utf-8-sig',newline='') as stream:
        records=csv.DictWriter(stream,fieldnames=fields)
        records.writeheader()
        records.writerows(rows)


def candidate(dets, gt_xywh, low, high):
    """All raw detector rows and rows eligible by score in native BoT-SORT."""
    gt=np.asarray(gt_xywh)
    inter=np.maximum(np.minimum(dets[:,2:4],gt[:2]+gt[2:4])-np.maximum(dets[:,:2],gt[:2]),0.).prod(axis=1)
    area=np.prod(dets[:,2:4]-dets[:,:2],axis=1)
    overlap=inter/(area+gt[2:].prod()-inter)
    entries={}
    for name,idx,suffix in [('best',np.arange(len(dets)),'det_'),
                            ('best_eligible',np.flatnonzero((dets[:,4]>low)&(dets[:,4]!=high)),'')]:
        k=int(idx[np.argmax(overlap[idx])]) if len(idx) else None
        entries[name+'_det']=k
        entries[name+'_'+suffix+'score']=None if k is None else float(dets[k,4])
        entries[name+'_'+suffix+'iou']=None if k is None else float(overlap[k])
    return entries


def frame_maps(scorer, data_set, method, seq, run, detections, expected):
    raw,data,masks,assigned,clear=scorer.clear_assignments(data_set,method,seq)
    counts={k:int(clear[src]) for k,src in [('FP','CLR_FP'),('FN','CLR_FN'),('IDSW','IDSW'),('TP','CLR_TP')]}
    if expected and any(counts[k]!=int(expected[k]) for k in ('FP','FN','IDSW')):
        raise ValueError(f'{method}/{seq}: official per-sequence counts differ: {counts} vs {expected}')
    logs=scorer.event_frames(run/'qr_events'/method/(seq+'.jsonl.gz'),len(detections))
    frames=[]
    for i,(mask,matches,ds,entry) in enumerate(zip(masks,assigned,detections,logs,strict=True)):
        tids=np.asarray(raw['tracker_ids'][i],int)
        gids=np.asarray(raw['gt_ids'][i][mask],int)
        gt_boxes=data['gt_dets'][i]
        boxes=data['tracker_dets'][i]
        scores=raw['tracker_confidences'][i]
        ledger={int(t):int(d) for t,d in entry['rows']}
        if (len(ledger)!=len(entry['rows']) or len(set(ledger.values()))!=len(ledger)
                or set(ledger)!=set(map(int,tids))):
            raise ValueError(f'{method}/{seq}/{i+1}: output row/track alignment')
        by_gt,by_det={},{}
        for col,tid in enumerate(tids):
            tid=int(tid)
            det=ledger[tid]
            if det<0 or det>=len(ds) or not np.isclose(scores[col],ds[det,4],rtol=1e-6,atol=1e-8):
                raise ValueError(f'{method}/{seq}/{i+1}: detection row {det}')
            grow=matches.get(col)
            gt=None if grow is None else int(gids[grow])
            by_det[det]=(tid,gt,col)
            if gt is not None:
                by_gt[gt]=(tid,det,float(ds[det,4]),float(data['similarity_scores'][i][grow,col]),*map(float,boxes[col]))
        frames.append((tuple(gids),gt_boxes,by_gt,by_det,data['similarity_scores'][i]))
    if counts['TP']!=sum(len(frame[2]) for frame in frames) or counts['FN']!=sum(len(f[0])-len(f[2]) for f in frames):
        raise ValueError(f'{method}/{seq}: reconstructed FN/TP differs from library CLEAR')
    return frames,counts


def signed_status(native,learned):
    if native:return 'both_hit' if learned else 'lost'
    return 'gained' if learned else 'both_missed'


def sequential_runs(events):
    runs=[]
    for ev in sorted(events,key=lambda e:(e['sequence'],e['gt_id'],e['frame'])):
        if not runs or any(ev[k]!=runs[-1][k] for k in ('sequence','gt_id','status')) or ev['frame']!=runs[-1]['end']+1:
            runs.append(dict(sequence=ev['sequence'],gt_id=ev['gt_id'],status=ev['status'],start=ev['frame'],
                             end=ev['frame'],frames=1))
        else:
            runs[-1]['end']=ev['frame']
            runs[-1]['frames']+=1
    return runs


def gt_summary(events,methods):
    counts=defaultdict(Counter)
    for ev in events:
        c=counts[ev['sequence'],ev['gt_id']]
        c[ev['status']]+=1
        for m in methods[2:]:
            if ev[m+'_tid'] is not None:c[ev['status']+'_'+m+'_hit']+=1
    return [dict(sequence=s,gt_id=g,lost=c['lost'],gained=c['gained'],net_fn=c['lost']-c['gained'],
                 **{key:c[key] for m in methods[2:] for key in ('lost_'+m+'_hit','gained_'+m+'_hit')})
            for (s,g),c in sorted(counts.items())]


def investigate(seq,methods,maps,detections,low,high,all_truth):
    rows,totals=[],Counter()
    for i,ds in enumerate(detections):
        native=maps['native'][i]
        for m in methods[1:]:
            frame=maps[m][i]
            if native[0]!=frame[0] or not np.array_equal(native[1],frame[1]):
                raise ValueError(f'{seq}/{i+1}/{m}: GT rows disagree')
        for index,g in enumerate(native[0]):
            g=int(g)
            hits={m:maps[m][i][2].get(g) for m in methods}
            status=signed_status(hits['native'],hits['learned_r'])
            totals[status]+=1
            for m in methods[2:]:
                own=signed_status(hits['native'],hits[m])
                totals[m+'_'+own]+=1
                if hits[m]:totals[status+'_'+m+'_hit']+=1
            if 'high_r' in hits and status=='lost':
                category=('both' if hits['low_r'] and hits['high_r'] else 'only_low' if hits['low_r']
                          else 'only_high' if hits['high_r'] else 'neither')
                totals['lost_saved_'+category]+=1
            if not all_truth and status not in ('lost','gained'):continue
            gt=native[1][index]
            row=dict(sequence=seq,frame=i+1,gt_id=g,status=status,gt_x=float(gt[0]),gt_y=float(gt[1]),
                     gt_w=float(gt[2]),gt_h=float(gt[3]),**candidate(ds,gt,low,high))
            nd=None if hits['native'] is None else hits['native'][1]
            for m,match in hits.items():
                for key,val in zip(('tid','det','score','iou','x','y','w','h'),match or (None,)*8,strict=True):
                    row[m+'_'+key]=val
                emitted=maps[m][i][3].get(nd)
                row[m+'_same_det_emitted_tid']=None if emitted is None else emitted[0]
                row[m+'_same_det_emitted_gt']=None if emitted is None else emitted[1]
                row[m+'_same_det_iou_to_gt']=None if emitted is None else float(maps[m][i][4][index,emitted[2]])
            rows.append(row)
    return rows,totals


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('all','original','case','full'))
    parser.add_argument('--original-root',type=Path)
    parser.add_argument('--counter-run',type=Path)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    root=Path(os.environ.get('VT_RF_ROOT',Path(__file__).resolve().parent)).resolve()
    qr=args.original_root or Path(os.environ.get('VT_RF_QR',root.parent/'native_neural_qr_v1'))
    original=qr/'full'
    scorer=load_original_scorer(qr)
    scorer.install_evaluation_paths(scorer.environment())
    from botsort_plus.io import read_detection_cache,detection_fingerprint
    from evaluate_vttiny import VTTinyObjects
    from qr_settings import read,sha,write
    stage='original' if args.stage=='all' else args.stage
    counter=args.counter_run or root/('pilot' if stage=='case' else 'full')
    out=(args.output_dir or root/'results')/stage
    methods=['native','learned_r']+({'case':['low_r','high_r'],'full':['low_r']}.get(stage,[]))
    roots={m:(original if m in ('native','learned_r') else counter) for m in methods}
    configs={m:read(roots[m]/'protocols'/(m+'.json')) for m in methods}
    previous=read(qr/'results/score_audit.json')
    if previous['status']!='PASS':raise ValueError('Archived scoring audit failed')
    seqs=['DJI_0309_4'] if stage=='case' else sorted(previous['sequences'])
    reference=read(qr/'results/metrics/summary.json')['results']
    original_metrics={(r['tracker'],r['sequence']):r for r in reference}
    s=read(scorer.SETTINGS)
    threshold=configs['native']['native_config']
    low,high=threshold['track_low_thresh'],threshold['track_high_thresh']
    per_sequence,changes,case_frames=[],[],[]
    for seq in seqs:
        gt_file=Path(s['gt_root'])/seq/'gt/gt.txt'
        if sha(gt_file)!=previous['gt_sha256'][seq]:raise ValueError('Official GT mismatched: '+seq)
        source=Path(s['baseline_root'])/'detections'/(seq+'.json.gz')
        if not source.exists():source=source.with_suffix('')
        dc=read_detection_cache(source)
        fingerprint=detection_fingerprint(dc)
        ds=[np.asarray(f['detections'],dtype=np.float32).reshape(-1,6) for f in dc['frames']]
        maps,stats={},{}
        for m in methods:
            cfg=configs[m]
            if (cfg['status']!='COMPLETE' or cfg['sequences'][seq]!=len(ds) or
                    cfg['detection_fingerprints'][seq]!=fingerprint or
                    sha(roots[m]/'trackers'/m/'data'/(seq+'.txt'))!=cfg['output_prediction_hashes'][seq]):
                raise ValueError(f'{m}/{seq}: input or output protocol inconsistent')
            for k in ('source_protocol_sha256','detector_sha256','rgb_encoder_sha256','native_config','boxmot_source_hashes'):
                if cfg[k]!=configs['native'][k]:raise ValueError(f'{m}/{seq}: changed {k}')
            for k in ('decoded_image_hashes','feature_hashes'):
                if cfg[k][seq]!=configs['native'][k][seq]:raise ValueError(f'{m}/{seq}: changed {k}')
            dataset=VTTinyObjects(dict(GT_FOLDER=s['gt_root'],TRACKERS_FOLDER=str(roots[m]/'trackers'),
                OUTPUT_FOLDER=str(out/'unused'),TRACKERS_TO_EVAL=[m],BENCHMARK='VT-Tiny-MOT',SPLIT_TO_EVAL='test',
                SKIP_SPLIT_FOL=True,SEQ_INFO={seq:len(ds)},DO_PREPROC=False,PRINT_CONFIG=False))
            maps[m],stats[m]=frame_maps(scorer,dataset,m,seq,roots[m],ds,original_metrics.get((m,seq)))
        rows,c=investigate(seq,methods,maps,ds,low,high,stage=='case')
        events=[row for row in rows if row['status'] in ('lost','gained')]
        changes.extend(events)
        if stage=='case':case_frames.extend(rows)
        summary=dict(sequence=seq,frames=len(ds),lost=c['lost'],gained=c['gained'],net_fn=c['lost']-c['gained'],
                     both_hit=c['both_hit'],both_missed=c['both_missed'])
        for m in methods:
            summary.update({m+'_'+k.lower():v for k,v in stats[m].items()})
        if summary['net_fn']!=stats['learned_r']['FN']-stats['native']['FN']:raise ValueError('Pairing identity failed')
        for m in methods[2:]:
            summary.update({key:c[key] for key in (m+'_lost',m+'_gained','lost_'+m+'_hit','gained_'+m+'_hit')})
            if c[m+'_lost']-c[m+'_gained']!=stats[m]['FN']-stats['native']['FN']:raise ValueError('CF pairing failed')
        if 'high_r' in methods:
            summary.update({key:c[key] for key in ['lost_saved_both','lost_saved_only_low','lost_saved_only_high','lost_saved_neither']})
        per_sequence.append(summary)
        print(stage,seq,summary['lost'],summary['gained'],summary['net_fn'],flush=True)
    combined={key:sum(row[key] for row in per_sequence) for key in per_sequence[0] if key!='sequence'}
    expected=original_metrics['learned_r',seqs[0] if stage=='case' else 'COMBINED_SEQ']['FN']-original_metrics['native',seqs[0] if stage=='case' else 'COMBINED_SEQ']['FN']
    if combined['net_fn']!=expected:raise ValueError('Total changes disagree with original reported net')
    by_seq=sorted(per_sequence,key=lambda v:(-v['net_fn'],-v['lost'],v['sequence']))
    emit(out/'by_sequence.csv',by_seq+[{'sequence':'COMBINED_SEQ',**combined}])
    emit(out/'events.csv.gz',changes)
    emit(out/'runs.csv',sequential_runs(changes))
    emit(out/'by_gt.csv',gt_summary(changes,methods))
    if case_frames:emit(out/'gt_frames.csv.gz',case_frames)
    write(out/'audit.json',dict(status='PASS',stage=stage,methods=methods,sequence_count=len(seqs),**combined,
                                historical_audit_sha256=sha(qr/'results/score_audit.json'),
                                source_protocol_sha256={m:sha(p/'protocols'/(m+'.json')) for m,p in roots.items()}))


if __name__=='__main__':main()
