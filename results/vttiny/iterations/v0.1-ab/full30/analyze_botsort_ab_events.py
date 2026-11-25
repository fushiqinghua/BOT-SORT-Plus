"""Post-hoc event attribution on frozen online outputs. Run from the integration environment."""
import argparse
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


METHODS = ('baseline', 'A', 'B', 'AB')
METRICS = ('HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'FP', 'FN', 'IDSW')


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write_table(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rank_frame_gap(g):
    return '0' if g == 0 else '1-3' if g <= 3 else '4-10' if g <= 10 else '11-30' if g <= 30 else '31+'


def events_and_counts(file, expected):
    events = []
    counts = Counter()
    stream = file.open(encoding='utf-8') if file.exists() else gzip.open(str(file)+'.gz', 'rt', encoding='utf-8')
    with stream as fh:
        for line in fh:
            event = json.loads(line)
            name = event['event']
            if name in ('weak_birth', 'recovered'):
                events.append(event)
                counts[name] += 1
    for name in ('weak_birth', 'recovered'):
        assert counts[name] == expected.get(name, 0)
    return events


def replay_clear(gt, pred, metrics, clear_step):
    previous, last_gt_match = {}, {}
    tally = Counter()
    track_history, gt_history = defaultdict(dict), defaultdict(list)
    for frame, (g, p) in enumerate(zip(gt,pred), 1):
        pairs, overlap = clear_step(g, p, previous)
        reverse = {j:i for i,j in pairs.items()}
        for i, row in enumerate(g):
            gid = int(row[0])
            tid = int(p[pairs[i],0]) if i in pairs else None
            gt_history[gid].append((frame, tid))
            if tid is not None:
                if gid in last_gt_match and last_gt_match[gid] != tid:
                    tally['IDSW'] += 1
                last_gt_match[gid] = tid
        tally.update({'FP':len(p)-len(pairs), 'FN':len(g)-len(pairs)})
        for j, row in enumerate(p):
            i = reverse.get(j)
            best = float(overlap[:,j].max(initial=0))
            duplicate = any(overlap[i2,j]>=.5-np.finfo(float).eps for i2 in pairs) if i is None else False
            track_history[int(row[0])][frame] = (int(g[i,0]) if i is not None else None,best,duplicate)
    assert all(tally[k] == metrics[k] for k in ('FP','FN','IDSW')), (tally,metrics)
    return track_history, gt_history


def birth_row(event, track, gt_history):
    frame=event['frame']
    gid, iou, duplicate = track.get(frame,(None,None,False))
    assigned = [h[0] for h in track.values() if h[0] is not None]
    dup_frames = sum(h[2] for h in track.values())
    peak = max((h[1] for h in track.values()), default=None)
    first = None if not assigned else next(k for k,v in track.items() if v[0] is not None)
    if gid is not None:
        seen = any(f < frame and t is not None for f,t in gt_history[gid])
        at_birth = 'previously_seen_gt' if seen else 'newly_covered_gt'
    elif iou is None:
        at_birth = 'no_output'
    elif duplicate:
        at_birth = 'overlaps_other_assigned_gt'
    else:
        at_birth = 'unmatched'
    if assigned:
        life='matched_some_gt'
    elif not track:
        life='no_output'
    elif dup_frames:
        life='never_matched_duplicate_overlap'
    elif peak < .1:
        life='never_near_gt'
    else:
        life='unmatched_near_gt'
    return dict(birth_gt=gid,at_birth=at_birth,birth_max_iou=iou,lifetime=life,output_frames=len(track),
                gt_matched_frames=len(assigned),distinct_gt_ids=len(set(assigned)),first_match_frame=first,
                duplicate_overlap_frames=dup_frames,lifetime_max_iou=peak)


def recovery_row(event, track):
    frame, age=event['frame'],event['age']
    before = frame-age
    prev = track.get(before,(None,None,False))[0]
    now = track.get(frame,(None,None,False))[0]
    verdict = 'undetermined' if prev is None or now is None else 'same_gt' if prev==now else 'different_gt'
    return dict(previous_frame=before,age=age,intervening_frames=age-1,gap_bin=rank_frame_gap(age-1),
                previous_gt=prev,current_gt=now,verdict=verdict,previous_output=before in track,current_output=frame in track)


def main(out, repo, original, gt_folder, low_file):
    sys.path.insert(0,str(repo/'tools'))
    from diagnose_vttiny_botsort_m import mot, clear_step
    dest=out/'event_analysis'
    dest.mkdir(exist_ok=True)
    rows=read(out/'metrics/summary.json')['results']
    metrics={(r['tracker'],r['sequence']):r for r in rows}
    original_rows={r['sequence']:r for r in read(original/'metrics/summary.json')['results'] if r['tracker']=='botsort_reid'}
    protocols={name:read(out/'protocols'/(name+'.json')) for name in METHODS}
    baseline=protocols['baseline']
    assert all(p['feature_hashes']==baseline['feature_hashes'] and p['detection_fingerprints']==baseline['detection_fingerprints'] and not p['subset'] and p['status']=='COMPLETE' for p in protocols.values())
    seqs=baseline['sequences']
    assert len(seqs)==30 and sum(seqs.values())==12590
    low=defaultdict(set)
    with gzip.open(low_file,'rt',encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            if row['candidate']=='low_assoc' and int(row['track_id'])<0:
                low[row['sequence']].add((int(row['gt_id']),int(row['frame'])))
    assert sum(map(len,low.values()))==10795
    births,recoveries,seqrows,gtrows,reference=[],[],[],[],[]
    for seq,n in seqs.items():
        gt=mot(gt_folder/seq/'gt/gt.txt',n,True)
        for name in METHODS:
            predpath=out/'trackers'/name/'data'/(seq+'.txt')
            pred=mot(predpath,n)
            m=metrics[name,seq]
            hist, gh=replay_clear(gt,pred,m,clear_step)
            if name=='baseline':
                frozen=original/'trackers/botsort_reid/data'/(seq+'.txt')
                reference.append(dict(sequence=seq,same_bytes=predpath.read_bytes()==frozen.read_bytes(),
                                      **{'delta_'+k:m[k]-original_rows[seq][k] for k in METRICS}))
            covered=sum((gid,f) in low[seq] and tid is not None for gid,h in gh.items() for f,tid in h)
            timing=read(out/'timing'/name/(seq+'.json'))
            s=dict(method=name,sequence=seq,**{k:m[k] for k in METRICS},never_matched=0,before_first_fn=0,
                   historical_low_fn=len(low[seq]),historical_low_now_matched=covered)
            for gid,h in gh.items():
                idx=next((i for i,(_,t) in enumerate(h) if t is not None),None)
                if idx is None: s['never_matched']+=1
                else: s['before_first_fn']+=idx
                gtrows.append(dict(method=name,sequence=seq,gt_id=gid,annotated_frames=len(h),
                                   first_matched_frame=None if idx is None else h[idx][0],before_first=idx))
            seqrows.append(s)
            for event in events_and_counts(out/'events'/name/(seq+'.jsonl'),timing['event_counts']):
                assert event['sequence']==seq
                tid=event['track_id']
                shared=dict(method=name,sequence=seq,frame=event['frame'],track_id=tid,candidate_id=event['candidate_id'],det_index=event['det_index'])
                if event['event']=='weak_birth':
                    births.append({**shared,**birth_row(event,hist[tid],gh)})
                else:
                    recoveries.append({**shared,**recovery_row(event,hist[tid])})
    for file,data in [('birth_events',births),('recovery_events',recoveries),('per_sequence',seqrows),('gt_first_match',gtrows),('baseline_replay',reference)]:
        write_table(dest/(file+'.csv'),data)
    groups=[]
    for name in ('B','AB'):
        for k in ['all','0','1-3','4-10','11-30','31+']:
            subset=[e for e in recoveries if e['method']==name and (k=='all' or e['gap_bin']==k)]
            c=Counter(e['verdict'] for e in subset)
            judged=c['same_gt']+c['different_gt']
            groups.append(dict(method=name,gap=k,total=len(subset),**{v:c[v] for v in ('same_gt','different_gt','undetermined')},
                               correct_among_judged=(c['same_gt']/judged if judged else None)))
    write_table(dest/'recovery_by_gap.csv',groups)
    first={(r['method'],r['sequence'],r['gt_id']):r['before_first'] for r in gtrows}
    paired={}
    for method in ('A','B','AB'):
        pairs=[(v,first['baseline',s,i]) for (m,s,i),v in first.items() if m==method]
        differences=[v-b for v,b in pairs if v is not None and b is not None]
        paired[method]=dict(common_gt=len(differences),earlier=sum(x<0 for x in differences),
                            later=sum(x>0 for x in differences),equal=differences.count(0),
                            total_delay_difference=sum(differences),
                            newly_matched=sum(v is not None and b is None for v,b in pairs),
                            lost_gt=sum(v is None and b is not None for v,b in pairs))
    (dest/'first_match_paired.json').write_text(json.dumps(paired,indent=2))
    summary={}
    for name in METHODS:
        sr=[r for r in seqrows if r['method']==name]
        br=[r for r in births if r['method']==name]
        valid=[r for r in br if r['lifetime']=='never_near_gt' and r['output_frames']>=3]
        summary[name]=dict(pooled=metrics[name,'COMBINED_SEQ'],never_matched=sum(r['never_matched'] for r in sr),
                           before_first_fn=sum(r['before_first_fn'] for r in sr),
                           historical_low_fn_now_matched=sum(r['historical_low_now_matched'] for r in sr),
                           births=len(br),at_birth=dict(Counter(r['at_birth'] for r in br)),lifetime=dict(Counter(r['lifetime'] for r in br)),
                           persistent_background_proxy=len(valid),recovery=[r for r in groups if r['method']==name])
    summary['baseline_check']=dict(identical_files=sum(r['same_bytes'] for r in reference),total=len(reference),
                                   max_per_sequence_abs_delta={k:max(abs(r['delta_'+k]) for r in reference) for k in METRICS},
                                   pooled_delta={k:metrics['baseline','COMBINED_SEQ'][k]-original_rows['COMBINED_SEQ'][k] for k in METRICS})
    (dest/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('out','repo','original','gt-folder','low-file'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(**vars(parser.parse_args()))
