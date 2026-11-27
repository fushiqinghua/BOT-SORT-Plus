"""Trace the old correspondence through both causal native association pipelines."""
import csv
import gzip
import json
from collections import defaultdict
import numpy as np
from rf_settings import CASE,ROOT,OUT,write,table

def integer(v):return None if v=='' else int(v)

def trace(method,identities):
    frames={}
    with gzip.open(ROOT/'pilot/trace'/method/(CASE+'.jsonl.gz'),'rt') as f:
        for line in f:
            raw=json.loads(line);stages=[]
            for a in raw['associations']:
                dets={d:i for i,d in enumerate(a['dets'])}
                candidates={t:(np.asarray(a['cost'][i]),np.asarray(a['iou_cost'][i])) for i,t in enumerate(a['tids']) if t in identities}
                stages.append((a['stage'],float(a['threshold']),dets,candidates,dict(a['matches']),{d:t for t,d in a['matches']}))
            updates={u['tid']:u for u in raw['updates'] if u['tid'] in identities}
            choices={tid:(d,a[0]) for a in stages for tid,d in a[4].items() if tid in identities}
            frames[raw['frame']]=dict(stages=stages,choices=choices,updates=updates,births=raw['births'])
    return frames

def edge(frame,tid,det,stage):
    a=next(v for v in frame['stages'] if v[0]==stage)
    _,cut,columns,rows,assignment,owners=a
    cost=raw=None
    if tid in rows and det in columns:
        i=columns[det];cost=float(rows[tid][0][i]);raw=float(rows[tid][1][i])
    return dict(cut=cut,cost=cost,iou_cost=raw,selected=assignment.get(tid),owner=owners.get(det),
                available_track=tid in rows,available_det=det in columns)

def details(row,traces,history):
    g=int(row['gt']);t0=int(row['frame']);p=int(row['previous'])
    m=row['method'];o='low_r' if m=='native' else 'native'
    old={m:int(row['old_tid']),o:integer(row['other_previous_tid'])}
    if old[o] is None:return None
    # Anchor using the two GT-matched old tracks, never the equality of cross-run ID numbers.
    shared=[]
    for t in range(1,p+1):
        snap=history[g].get(t)
        if not snap:continue
        if all(integer(snap[k+'_tid'])==old[k] for k in old) and snap['native_det']==snap['low_r_det']:
            shared.append(t)
    if not shared:return None
    anchor=shared[-1]
    # The first changed association after the last common GT/detection anchor.
    diff=None
    for t in range(anchor+1,t0+1):
        selected={k:traces[k][t]['choices'].get(old[k]) for k in old}
        if selected['native']==selected['low_r']:continue
        diff=dict(frame=t,selected=selected)
        for k in old:
            if selected[k]:
                d,stage=selected[k]
                diff[k+'_choice_edges']={j:edge(traces[j][t],old[j],d,stage) for j in old}
        break
    stop=t0 if diff is None else diff['frame']
    preceding=[]
    for t in reversed(shared):
        if t>=stop or stop-t>40:continue
        u=traces['low_r'][t]['updates'].get(old['low_r'])
        if not u or not u['applied']:continue
        snap=history[g][t]
        x,y,w,h=[float(snap[f'gt_{v}']) for v in 'xywh'];center=np.asarray([x+w/2,y+h/2])
        err=lambda box:float(np.linalg.norm(np.asarray(box[:2])-center))
        preceding.append(dict(frame=t,det=u['det'],tid=old['low_r'],score=u['score'],r_gain=u['r_gain'][:2],
            native_gain=u['native_gain'][:2],prior_err=err(u['prior']),det_err=err(u['obs']),
            r_post_err=err(u['r_post']),native_post_err=err(u['native_post'])))
        if len(preceding)>=4:break
    return dict(method=m,gt=g,frame=t0,previous=p,status=row['kind'],old_ids=old,anchor=anchor,
                first_assignment_difference=diff,common_previous_low_updates=preceding)

def main():
    for m in ('native','low_r'):
        from rf_settings import RUNS,sha
        assert sha(ROOT/'pilot/trackers'/m/'data'/(CASE+'.txt'))==sha(RUNS[m]/'trackers'/m/'data'/(CASE+'.txt'))
    with (OUT/'switch_ledger.csv').open(encoding='utf-8-sig') as f:ledger=list(csv.DictReader(f))
    with gzip.open(OUT/'official_gt_frames.csv.gz','rt',encoding='utf-8-sig') as f:truth=list(csv.DictReader(f))
    history=defaultdict(dict)
    for r in truth:history[int(r['gt'])][int(r['frame'])]=r
    ids={m:set() for m in ('native','low_r')}
    for r in ledger:
        if r['kind']=='same_frame_switch':continue
        k=r['method'];o=next(j for j in ids if j!=k)
        ids[k].update([int(r['old_tid']),int(r['new_tid'])])
        if r['other_previous_tid']:ids[o].add(int(r['other_previous_tid']))
    traces={m:trace(m,ids[m]) for m in ids}
    notes=[details(r,traces,history) for r in ledger if r['kind']!='same_frame_switch']
    write(OUT/'decision_chains.json',[n for n in notes if n is not None])
    print('Tracked decision chains',sum(n is not None for n in notes),flush=True)

if __name__=='__main__':main()
