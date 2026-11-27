"""Offline GT attribution of the high/low R interventions on DJI_0309_4."""
from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import statistics

SEQ = 'DJI_0309_4'
METHODS = ('native','learned_r','low_r','high_r')


def integer(value):
    return None if value in ('',None) else int(value)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def read_trace(path):
    with gzip.open(path,'rt',encoding='utf-8') as file:
        result={record['frame']:record for record in map(json.loads,file)}
    if sorted(result)!=list(range(1,len(result)+1)):
        raise ValueError(f'Incomplete frame order in {path}')
    return result


def take_runs(rows):
    run=[]
    for row in rows:
        if run and (row['status']!='lost' or int(row['frame'])!=int(run[-1]['frame'])+1):
            yield run
            run=[]
        if row['status']=='lost':
            run.append(row)
    if run:yield run


def counterfactual(row):
    low=integer(row['low_r_tid']) is not None
    high=integer(row['high_r_tid']) is not None
    if low and not high: return 'high_alone_reproduces'
    if high and not low: return 'low_alone_reproduces'
    if low and high: return 'both_single_arms_recover'
    return 'both_single_arms_miss'


def iou_xywh(box, truth):
    def corners(a):
        x,y,w,h=a[:4]
        return x-w/2,y-h/2,x+w/2,y+h/2
    l,t,r,b=corners(box)
    L,T,R,B=corners(truth)
    area=max(0.,min(r,R)-max(l,L))*max(0.,min(b,B)-max(t,T))
    return area/max(1.e-12,box[2]*box[3]+truth[2]*truth[3]-area)


def gt_box(row):
    x,y,w,h=(float(row['gt_'+k]) for k in ('x','y','w','h'))
    return x+w/2,y+h/2,w,h


def update_evidence(frame, obs, truth):
    center_error=lambda state:math.hypot(state[0]-truth[0],state[1]-truth[1])
    return dict(frame=frame,tid=obs['tid'],det=obs['det'],stage=obs['stage'],score=obs['score'],applied=obs['applied'],
                native_kx=obs['native_gain'][0],r_kx=obs['r_gain'][0],native_ky=obs['native_gain'][1],r_ky=obs['r_gain'][1],
                native_rx=obs['native_r'][0],learned_rx=obs['learned_r'][0],native_ry=obs['native_r'][1],learned_ry=obs['learned_r'][1],
                prior_err_px=center_error(obs['prior']),det_err_px=center_error(obs['obs']),raw_det_iou_to_gt=iou_xywh(obs['obs'],truth),
                same_prior_native_post_err_px=center_error(obs['native_post']),same_prior_r_post_err_px=center_error(obs['r_post']),
                same_prior_native_post_iou=iou_xywh(obs['native_post'],truth),same_prior_r_post_iou=iou_xywh(obs['r_post'],truth))


def match(record,tid):
    # Compare actual assignments (including unconfirmed tracks) before emission or CLEAR scoring.
    if tid is None:return None
    for event in record['associations']:
        for identity,det in event['matches']:
            if identity==tid:return (det,event['stage'])
    for b in record['births']:
        if b['tid']==tid:return (b['det'],'birth')
    return None


def candidate(record,tid,det,stage):
    if stage is None or stage=='birth':return {'status':'no_native_matching_stage'}
    for a in record['associations']:
        if a['stage']!=stage:continue
        t=tid in a['tids']
        d=det in a['dets']
        if not (t and d):
            return dict(status='missing_track_and_det' if not t and not d else 'track_not_in_stage' if not t else 'det_not_in_stage',
                        track_available=t,det_available=d,threshold=a['threshold'])
        i=a['tids'].index(tid); j=a['dets'].index(det)
        cost=float(a['cost'][i][j]); limit=float(a['threshold'])
        matches={int(tid_):int(det_) for tid_,det_ in a['matches']}
        selected=matches.get(tid)
        owner=next((who for who,what in matches.items() if what==det),None)
        status='selected' if selected==det else 'above_threshold' if cost>limit else 'admissible_unselected'
        return dict(status=status,cost=cost,iou_cost=float(a['iou_cost'][i][j]),threshold=limit,
                    margin=limit-cost,prior_xyxy=a['prior_boxes'][i],assigned_det=selected,det_assigned_tid=owner)
    return {'status':'stage_not_called'}


def event_between(traces, native_tid, r_tid, start, end):
    for t in range(start+1,end+1):
        a=match(traces['native'][t],native_tid)
        b=match(traces['learned_r'][t],r_tid)
        # A changed stage for the same frozen row is worth logging only if it changes the selected row later.
        da=a[0] if a else None
        db=b[0] if b else None
        if da==db:continue
        ref=a if a else b
        return dict(frame=t,native_selected=da,r_selected=db,stage=ref[1],frozen_reference_det=ref[0],
                    native_candidate=candidate(traces['native'][t],native_tid,*ref),
                    r_candidate=candidate(traces['learned_r'][t],r_tid,*ref))
    return None


def emission_status(row):
    if integer(row['learned_r_same_det_emitted_tid']) is None:return 'native_det_not_emitted'
    target=integer(row['learned_r_same_det_emitted_gt'])
    if target is not None and target!=int(row['gt_id']):return 'native_det_matched_other_gt'
    value=float(row['learned_r_same_det_iou_to_gt'])
    return 'native_det_iou_below_half' if value<.5 else 'native_det_ge_half_unassigned'


def analyze(root, output):
    with gzip.open(output/'case/gt_frames.csv.gz','rt',encoding='utf-8-sig',newline='') as f:
        cohort=list(csv.DictReader(f))
    traces={m:read_trace(root/'pilot/trace'/m/(SEQ+'.jsonl.gz')) for m in METHODS}
    native_config=json.loads((root/'pilot/protocols/native.json').read_bytes())['native_config']
    high,birth_threshold=(float(native_config[k]) for k in ('track_high_thresh','new_track_thresh'))
    history=defaultdict(list)
    for row in cohort:
        if row['sequence']!=SEQ:raise ValueError('Unexpected case sequence')
        history[int(row['gt_id'])].append(row)
        t=int(row['frame'])
        for method in METHODS:
            tid=integer(row[method+'_tid']); det=integer(row[method+'_det'])
            if tid is not None and [tid,det] not in traces[method][t]['rows']:
                raise ValueError(f'CLEAR/trace row mismatch {method}/{t}/{tid}/{det}')
    births={m:{b['tid']:(t,b['det']) for t,event in trace.items() for b in event['births']}
            for m,trace in traces.items()}
    r_births={v:k for k,v in births['learned_r'].items()}
    lost_rows=[]
    episodes=[]
    support_rows=[]
    for gid, rows in sorted(history.items()):
        rows.sort(key=lambda r:int(r['frame']))
        by_frame={int(row['frame']):row for row in rows}
        for run in take_runs(rows):
            first=run[0]
            start,end=int(first['frame']),int(run[-1]['frame'])
            labels=Counter(counterfactual(row) for row in run)
            for row in run:
                lost_rows.append(dict(gt_id=gid,frame=row['frame'],run_start=start,classification=counterfactual(row),
                                      emission_status=emission_status(row)))
            # Anchor identity across branches using a common GT and frozen row, or identical birth.
            # Track IDs are scoped to individual replays and cannot be compared numerically across arms.
            before=[r for r in rows if int(r['frame'])<start]
            n=integer(first['native_tid'])
            anchor_row=next((r for r in reversed(before) if integer(r['native_tid'])==n and
                             integer(r['native_det']) is not None and r['native_det']==r['learned_r_det']),None)
            birth=births['native'].get(n)
            rt_birth=r_births.get(birth)
            if anchor_row:
                anchor=int(anchor_row['frame']); rt=integer(anchor_row['learned_r_tid']); typ='same_gt_frozen_row'
            elif rt_birth is not None:
                anchor=birth[0];rt=rt_birth;typ='same_birth_frozen_row'
            else:
                anchor=None;rt=None;typ='unresolved'
            link_near=event_between(traces,n,rt,anchor,start) if anchor is not None else None
            link_ever=(event_between(traces,n,rt_birth,birth[0],start) if rt_birth is not None and rt_birth==rt else None)
            link_in_run=(event_between(traces,n,rt,start-1,end) if anchor is not None else None)
            for link in (link_near,link_ever,link_in_run):
                if link is None or link['frame'] not in by_frame:continue
                f=link['frame']
                truth=gt_box(by_frame[f])
                for arm,tid in (('native',n),('learned_r',rt)):
                    accepted=next((v for v in traces[arm][f]['updates'] if v['tid']==tid),None)
                    if accepted:link[arm+'_matched_update']=update_evidence(f,accepted,truth)
            first_diff=link_near or link_ever
            cut=first_diff['frame'] if first_diff else start
            # Only link a previous observation to this GT if the official rows in both branches
            # confirm the same frozen detection on the very identities being followed.
            recent=[]
            for f in range(cut-1,max(0,cut-31),-1):
                g=by_frame.get(f)
                if not g or integer(g['native_tid'])!=n or rt is None or integer(g['learned_r_tid'])!=rt:continue
                det=integer(g['native_det'])
                if det is None or det!=integer(g['learned_r_det']):continue
                u=next((u for u in traces['learned_r'][f]['updates'] if u['tid']==rt and u['det']==det),None)
                if u:recent.append(update_evidence(f,u,gt_box(g)))
            for ev in recent:
                support_rows.append(dict(gt_id=gid,run_start=start,before_frame=cut,**ev))
            det=integer(first['native_det'])
            current_match=match(traces['native'][start],n)
            stage=current_match[1] if current_match else None
            onset_row=dict(frame=start,det=det,score=float(first['native_score']),
                           emission_status=emission_status(first),native_iou=float(first['native_iou']),
                           r_same_det_iou=(float(first['learned_r_same_det_iou_to_gt']) if first['learned_r_same_det_iou_to_gt'] else None),
                           native_candidate=candidate(traces['native'][start],n,det,stage),
                           r_candidate=candidate(traces['learned_r'][start],rt,det,stage) if rt is not None else None)
            emitted_tid=integer(first['learned_r_same_det_emitted_tid'])
            onset_update=next((v for v in traces['learned_r'][start]['updates']
                               if v['det']==det and v['tid']==emitted_tid),None)
            if onset_update:
                onset_row['same_frozen_row_r_update']=update_evidence(start,onset_update,gt_box(first))
            scores=[float(row['native_score']) for row in run]
            downstream=dict(native_low_score_frames=sum(s<high for s in scores),
                            native_score_below_birth_threshold=sum(s<birth_threshold for s in scores),
                            min_score=min(scores),max_score=max(scores),
                            native_det_emission_in_full_r=dict(Counter(emission_status(row) for row in run)))
            episodes.append(dict(gt_id=gid,start=start,end=end,length=len(run),counterfactual=dict(labels),downstream=downstream,
                                 anchor_type=typ,anchor_frame=anchor,native_tid=n,r_tid=rt,
                                 first_assignment_change_after_anchor=link_near,first_assignment_change_since_common_birth=link_ever,
                                 first_assignment_change_within_run=link_in_run,
                                 onset=onset_row,previous_shared_observations=recent[:4]))
    lost=len(lost_rows)
    gained=sum(row['status']=='gained' for row in cohort)
    if lost-gained!=76:
        raise ValueError(f'Independent CLEAR case totals differ: {lost} - {gained}')
    groups=defaultdict(list)
    for row in cohort:
        for m in METHODS:
            tid=integer(row[m+'_tid'])
            if tid is None:continue
            t=int(row['frame']); det=integer(row[m+'_det'])
            e=next((v for v in traces[m][t]['updates'] if v['tid']==tid and v['det']==det),None)
            if e is not None and m=='learned_r':
                groups[e['stage']].append(update_evidence(t,e,gt_box(row)))
    calibration={}
    for stage, events in groups.items():
        calibration[stage]=dict(n=len(events),center_k_lower_both_axes=sum(e['r_kx']<e['native_kx'] and e['r_ky']<e['native_ky'] for e in events),
                               median_native_center_gain=statistics.median((e['native_kx']+e['native_ky'])/2 for e in events),
                               median_learned_center_gain=statistics.median((e['r_kx']+e['r_ky'])/2 for e in events),
                               mean_same_prior_r_minus_native_center_error_px=statistics.mean(e['same_prior_r_post_err_px']-e['same_prior_native_post_err_px'] for e in events))
    fn={m:sum(integer(row[m+'_tid']) is None for row in cohort) for m in METHODS}
    result=dict(sequence=SEQ,high_score_threshold=high,birth_threshold=birth_threshold,
                official_gt_frames=len(cohort),fn=fn,native_only_tp=lost,r_only_tp=gained,
                interaction_delta_fn=fn['learned_r']-fn['low_r']-fn['high_r']+fn['native'],
                gross_lost_counterfactual=dict(Counter(r['classification'] for r in lost_rows)),
                gross_lost_emission=dict(Counter(r['emission_status'] for r in lost_rows)),
                per_gt_native_only_tp=dict(Counter(r['gt_id'] for r in lost_rows)),
                learned_r_update_conditional_on_official_tp=calibration,
                events=sorted(episodes,key=lambda e:(-e['length'],e['gt_id'],e['start'])))
    target=output/'case'
    write_csv(target/'missed_frame_attribution.csv',lost_rows)
    write_csv(target/'shared_updates_before_divergence.csv',support_rows)
    (target/'association_attribution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='events'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    from rf_settings import ROOT,OUT
    analyze(ROOT,OUT)
