"""Audit CLEAR switch endpoints and measured adjacent-frame appearance for C2."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import os
import sys
import numpy as np

job=Path(__file__).resolve().parent
stage=job.parents[1]; project=stage.parents[1]; repo=stage/'BOT-SORT-Plus'
os.environ['TRACKEVAL_ROOT']=str(project/'external/TrackEval')
sys.path.insert(0,str(repo/'tools'))
from diagnose_vttiny_botsort_m import clear_step,mot
full=stage/'experiments/c2_full_v1/full30'; out=job/'evidence'
names=['B_ref','B_C2','native_only','bank_only','release_bank_history']
selected={'DJI_0032_3':[(42,250,286)],'DJI_0327_3':[(14,296,328),(67,125,160),(15,386,392),(80,200,234)]}
events=defaultdict(list)
with (out/'switch_events.csv').open(encoding='utf-8-sig') as f:
    for e in csv.DictReader(f):
        if e['comparison']=='B_C2 vs B_ref' and e['which']!='both':events[e['sequence']].append(e)
def write(name,rows):
    with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
switches=[];timeline=[];appearance=[];stats=Counter()
def quantiles(v):
    return [float(np.quantile(v,x)) for x in (.1,.5,.9)]
for seq,n in json.loads((full/'protocols/B_ref.json').read_bytes())['sequences'].items():
    gt=mot(project/'data/official_gt/GT/GT_00/test'/seq/'gt/gt.txt',n,True)
    npz=stage.parent/'botsort_plus_v0_1_0'/'features_rgb'/(seq+'.npz')
    with np.load(npz) as z:
        fs=z['frame_ids'];offsets=z['offsets'];feats=z['rgb'];valid=z['rgb_valid']
    assert np.array_equal(fs,np.arange(1,n+1))
    present={(i,int(b[0])) for i,g in enumerate(gt,1) for b in g}
    maps={}
    for method in (names if seq in selected else names[:2]):
        root=full if method in names[:2] else job/'controls'
        outputs=mot(root/'trackers'/method/'data'/(seq+'.txt'),n)
        byframe=defaultdict(list); updates={}
        with (root/'events'/method/(seq+'.jsonl')).open() as f:
            for l in f:
                e=json.loads(l); ev=e['event'];fr=e['frame'];tid=e.get('track_id')
                if ev in ('match','recovered','native_birth'):
                    updates[fr,tid]=(e['det_index'],e.get('stage',ev))
                if method=='B_C2' and ev in ('identity_decision','identity_seed'):byframe[fr].append(e)
        prior={};gt_map={};last={};anchor={}
        for fr,(g,pr) in enumerate(zip(gt,outputs),1):
            pairs,_=clear_step(g,pr,prior)
            label={int(pr[j,0]):int(g[i,0]) for i,j in pairs.items()}
            for tid,gid in label.items():
                di,st=updates[fr,tid];gt_map[fr,gid]=(tid,di,st)
            if method!='B_C2':continue
            for e in byframe[fr]:
                tid=e['track_id'];gid=label.get(tid); di=e['det_index']; ix=int(offsets[fr-1])+di
                assert ix<int(offsets[fr])
                if e.get('stage')=='high' and e.get('reason')=='history_conflict':
                    stats['high_history_conflict']+=1
                    old=last.get(tid);anc=anchor.get(tid)
                    if gid is not None and old and (old[0],old[1])==(fr-1,gid) and valid[ix] and valid[old[2]]:
                        v=feats[ix].astype(np.float64);u=feats[old[2]].astype(np.float64)
                        d=float(1-np.dot(v,u)/(np.linalg.norm(v)*np.linalg.norm(u)))
                        assert np.isfinite(d) and anc and e['distance']>.4-1e-7
                        appearance.append((seq,fr,tid,gid,round(d,8),float(e['distance']),fr-anc[0],e['history_updates'],anc[1]==gid,anc[1] is None))
                if e.get('private_anchor_written'):anchor[tid]=(fr,gid)
            for tid,gid in label.items():last[tid]=(fr,gid,int(offsets[fr-1])+updates[fr,tid][0])
        maps[method]=gt_map
    for e in events[seq]:
        m,other,key=('B_ref','B_C2','ref_event') if e['which']=='ref_only' else ('B_C2','B_ref','variant_event')
        pf,old,new,st,di=json.loads(e[key]);fr=int(e['frame']);gid=int(e['gt'])
        a=maps[other].get((pf,gid));b=maps[other].get((fr,gid))
        assert maps[m][pf,gid][0]==old and maps[m][fr,gid][0]==new
        tag=('same_id' if a[0]==b[0] else 'different_ids') if a and b else ('previous_only' if a else 'current_only' if b else 'neither')
        switches.append(dict(sequence=seq,gt=gid,switch_in=m,previous_frame=pf,frame=fr,old_id=old,new_id=new,other_method=other,other_previous_id=a[0] if a else '',other_current_id=b[0] if b else '',other_endpoints=tag))
    for gid,lo,hi in selected.get(seq,[]):
        for fr in range(lo,hi+1):
            row=dict(sequence=seq,gt=gid,frame=fr,gt_present=(fr,gid) in present)
            for m in names:
                x=maps[m].get((fr,gid))
                row.update({m+'_id':x[0] if x else '',m+'_det_index':x[1] if x else '',m+'_stage':x[2] if x else ''})
            timeline.append(row)
    print(seq,flush=True)
write('switch_counterparts.csv',switches);write('case_timelines.csv',timeline)
cols=['sequence','frame','track_id','gt','adjacent_distance','anchor_distance','anchor_age','history_updates','last_write_same_gt','last_write_unknown']
write('contiguous_conflicts.csv',[dict(zip(cols,r)) for r in appearance])
arr=np.array([r[4:8] for r in appearance]); dist,ref,age,cnt=arr.T
stats['same_gt_previous_frame_valid']=len(arr);stats['adjacent_within_040']=int((dist<=.4).sum());stats['adjacent_within_020']=int((dist<=.2).sum());stats['anchor_age_over_30']=int((age>30).sum());stats['history_below_3']=int((cnt<3).sum());stats['last_write_same_gt']=sum(int(r[8]) for r in appearance)
res={'counts':dict(stats),'quantiles_q10_q50_q90':{'adjacent_distance':quantiles(dist),'anchor_distance':quantiles(ref),'anchor_age':quantiles(age)},'switch_endpoints':{m:dict(Counter(x['other_endpoints'] for x in switches if x['switch_in']==m)) for m in names[:2]}}
assert stats['high_history_conflict']==31032
assert sum(res['switch_endpoints']['B_ref'].values())==74 and sum(res['switch_endpoints']['B_C2'].values())==65
(out/'case_summary.json').write_text(json.dumps(res,ensure_ascii=False,indent=2))
print(json.dumps(res),flush=True)
