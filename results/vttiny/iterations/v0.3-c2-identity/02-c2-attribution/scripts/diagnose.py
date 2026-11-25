"""Post-hoc official CLEAR matching, detected-row pairing and per-writer evidence."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import gzip
import hashlib
import json
import os
import sys
import numpy as np

job=Path(__file__).resolve().parent
stage=job.parents[1]; project=stage.parents[1]; repo=stage/'BOT-SORT-Plus'
os.environ['TRACKEVAL_ROOT']=str(project/'external/TrackEval')
sys.path.insert(0,str(repo/'tools'))
from diagnose_vttiny_botsort_m import clear_step,mot
full=stage/'experiments/c2_full_v1/full30'
out=job/'evidence';out.mkdir(exist_ok=True)
methods=['B_ref','B_C2_memory','B_C2_association','B_C2','B_C2_shadow']
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
proto={m:read(full/'protocols'/(m+'.json')) for m in methods}
seqs=proto['B_ref']['sequences']
metric={(v['tracker'],v['sequence']):v for v in read(full/'metrics/summary.json')['results']}
audit=read(full/'metrics/evaluation_audit.json')
F,G,E,REC,SW,GAIN,TARGET=[],[],[],[],[],[],[]
case_writes=[];case_lookup=[];score_counts=Counter();anchor_counts=Counter()
def csvout(name,rows):
    rows=list(rows)
    if not rows:return
    with (out/name).open('w',encoding='utf-8-sig',newline='') as fp:
        wr=csv.DictWriter(fp,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
def trace(path):
    updates={};dec={};seeds={};rec={};checks={};cand={};writes=set();assoc=Counter()
    with path.open() as f:
        for line in f:
            e=json.loads(line);k=e['event'];fr=e['frame']; tid=e.get('track_id');di=e.get('det_index')
            if k in ('match','recovered','native_birth'):
                assert (fr,tid) not in updates
                updates[fr,tid]=(di,e.get('stage',k))
                if k=='recovered':rec[fr,di]=e
            if k=='identity_decision':dec[fr,tid]=e
            if k=='identity_seed':seeds[fr,tid]=e
            if k=='memory_update':writes.add((fr,tid))
            if k in ('candidate_seed','candidate_hit','recovered') and di is not None:cand[fr,di]=e['candidate_id']
            if k=='recovery_check':checks[fr,e['candidate_id'],tid]=e['reason']
            if k=='identity_association':
                for kk in ('pair_count','proposed_changed_pairs','cold_reference_pairs'):assoc[e['stage'],kk]+=e[kk]
    return updates,dec,seeds,rec,checks,cand,writes,assoc
def classify(a,b):
    if a is None:return 'current_unmatched'
    if b is None:return 'previous_unknown'
    return 'same' if a==b else 'different'
for seq,n in seqs.items():
    gt_path=project/'data/official_gt/GT/GT_00/test'/seq/'gt/gt.txt'
    assert sha(gt_path)==audit['ground_truth']['sha256'][seq]
    truth=mot(gt_path,n,True)
    with gzip.open(repo/'results/vttiny/yolo11m_rgb_rgbt/rgbt/detections'/(seq+'.json.gz'),'rt') as fp:
        dets=read_cache=json.load(fp)['frames']
    series={}
    allids={int(v[0]) for g in truth for v in g}
    for m in methods:
        pred=full/'trackers'/m/'data'/(seq+'.txt')
        assert sha(pred)==proto[m]['output_prediction_hashes'][seq]
        frames=mot(pred,n)
        updates,decs,seeds,rec,checks,cand,writes,assoc=trace(full/'events'/m/(seq+'.jsonl'))
        gt_map={};row_map={};id_map={};prior={};last_gt={};last_track={};anchors={};switches={};last_recover={}
        recover_data={};gt_by_id=defaultdict(set)
        for fr,(g,pr,dt) in enumerate(zip(truth,frames,dets),1):
            pairs,ov=clear_step(g,pr,prior)
            labels={int(pr[j,0]):int(g[i,0]) for i,j in pairs.items()}
            emitted={int(t[0]) for t in pr}
            for t in pr:
                tid=int(t[0]);di,st=updates[fr,tid];gid=labels.get(tid);df=dt['detections'][di]
                assert abs(df[4]-t[5])<1e-6 and (fr,di) not in row_map
                if st=='recovered':last_recover[tid]=(fr,di)
                item=(tid,gid,st,di,last_recover.get(tid))
                id_map[fr,tid]=row_map[fr,di]=item
                if gid is not None:
                    gt_map[fr,gid]=item;gt_by_id[gid].add(fr)
                    if gid in last_gt and last_gt[gid][1]!=tid:switches[fr,gid]=(last_gt[gid][0],last_gt[gid][1],tid,st,di)
                    last_gt[gid]=(fr,tid)
            # Both emitted and non-emitted matches/seeds contribute to actual memory.
            # Index frame events once outside this loop via the frame's update IDs.
            if fr==1:
                eventids=defaultdict(list)
                for (ff,tt) in updates:eventids[ff].append(tt)
            for tid in eventids[fr]:
                key=(fr,tid);e=decs.get(key);se=seeds.get(key);gid=labels.get(tid)
                if e:
                    why=e['reason'];st=e['stage'];wr=e['allow_write']
                    old=last_track.get(tid,(None,None))
                    anc=anchors.get(tid,(None,None))
                    prevtag=classify(gid,old[1]);anctag=classify(gid,anc[1])
                    score_counts[m,st,why,wr,prevtag,(key in writes)]+=1
                    anchor_counts[m,st,why,anctag]+=1
                    if m in ['B_C2','B_C2_shadow'] and why=='history_conflict' and st=='high' and prevtag=='same' and anctag=='same' and anc[0] is not None and len(case_writes)<200000:
                        case_writes.append((seq,fr,gid,m,tid,anc[0],fr-anc[0],old[0],round(e['distance'],5),round(e['quality'],5),e['history_updates']))
                    if e['private_anchor_written']:anchors[tid]=(fr,gid)
                if se and se['private_anchor_written']:anchors[tid]=(fr,gid)
                if gid is not None:last_track[tid]=(fr,gid)
        for (fr,di),e in rec.items():
            tid=e['track_id'];pf=fr-e['age'];before=id_map.get((pf,tid));now=id_map.get((fr,tid))
            g1=before[1] if before else None;g2=now[1] if now else None
            recover_data[fr,di]=(tid,pf,before[3] if before else None,g1,g2,'uncertain' if g1 is None or g2 is None else ('same_gt' if g1==g2 else 'different_gt'))
        fp=sum(i[1] is None for i in row_map.values());fn=sum(len(g) for g in truth)-len(gt_map)
        assert [fp,fn,len(switches)]==[metric[m,seq][k] for k in ('FP','FN','IDSW')],(seq,m)
        G.append(dict(sequence=seq,method=m,FP=fp,FN=fn,IDSW=len(switches),matches=len(gt_map),recovered=len(rec),memory_writes=len(writes),**{k+'_'+v:assoc[k,v] for k in ['high','unconfirmed'] for v in ['pair_count','proposed_changed_pairs','cold_reference_pairs']}))
        series[m]=dict(gt=gt_map,rows=row_map,ids=id_map,sw=switches,rec=recover_data,checks=checks,cand=cand,by_id=gt_by_id)
    for m,ref in [('B_C2','B_ref'),('B_C2_memory','B_ref'),('B_C2_association','B_ref'),('B_C2','B_C2_memory')]:
        a=series[ref];b=series[m];comp=m+' vs '+ref
        lost=a['gt'].keys()-b['gt'].keys();gained=b['gt'].keys()-a['gt'].keys()
        diff_rec=a['rec'].keys() ^ b['rec'].keys()
        for direction,from_,to_,ks in [('loss',a,b,lost),('gain',b,a,gained)]:
            for fr,gid in sorted(ks):
                tid,_,st,di,lr=from_['gt'][fr,gid];other=to_['rows'].get((fr,di))
                GAIN.append(dict(comparison=comp,sequence=seq,direction=direction,frame=fr,gt=gid,track_id=tid,stage=st,det_index=di,other_same_row=('absent' if other is None else 'FP' if other[1] is None else 'different_GT'),last_recovery_frame=lr[0] if lr else None,last_recovery_not_shared=bool(lr and lr in diff_rec)))
        for k in sorted(a['sw'].keys() | b['sw'].keys()):
            fr,gid=k;x=a['sw'].get(k);y=b['sw'].get(k)
            SW.append(dict(comparison=comp,sequence=seq,frame=fr,gt=gid,which='both' if x and y else 'ref_only' if x else 'variant_only',ref_event=json.dumps(x),variant_event=json.dumps(y)))
        for direction,from_,to_ in [('ref',a,b),('variant',b,a)]:
            for (fr,di),(tid,pf,prevdi,prevgt,curgt,label) in from_['rec'].items():
                rr=to_['rec'].get((fr,di)); other=to_['rows'].get((fr,di)); otherprior=to_['rows'].get((pf,prevdi))
                cid=to_['cand'].get((fr,di))
                check=to_['checks'].get((fr,cid,otherprior[0])) if otherprior and cid is not None else None
                REC.append(dict(comparison=comp,sequence=seq,side=direction,frame=fr,di=di,tid=tid,previous_frame=pf,previous_di=prevdi,previous_gt=prevgt,current_gt=curgt,endpoint=label,shared=rr is not None,other_stage=other[2] if other else 'not_emitted',other_gt=other[1] if other else None,other_prior_tid=otherprior[0] if otherprior else None,other_candidate=cid,other_check=check))
        for gid in allids:
            aa=a['by_id'][gid];bb=b['by_id'][gid]
            TARGET.append(dict(comparison=comp,sequence=seq,gt=gid,ref_frames=len(aa),var_frames=len(bb),lost=len(aa-bb),gained=len(bb-aa)))
        pairfp=Counter()
        for key in a['rows'].keys()|b['rows'].keys():
            x=a['rows'].get(key);y=b['rows'].get(key)
            tag=lambda z:'absent' if z is None else 'FP' if z[1] is None else 'TP'
            pairfp[tag(x)+'_to_'+tag(y)]+=1
        F.append(dict(comparison=comp,sequence=seq,gained=len(gained),lost=len(lost),**{x+'_to_'+y:pairfp[x+'_to_'+y] for x in ['absent','FP','TP'] for y in ['absent','FP','TP']}))
    print(seq,flush=True)
csvout('matched_gates.csv',(dict(method=m,stage=s,reason=r,gate_allows=a,previous_labeled_track=p,custom_wrote=w,count=n) for (m,s,r,a,p,w),n in sorted(score_counts.items())))
csvout('anchor_gates.csv',(dict(method=m,stage=s,reason=r,last_private_write_gt=p,count=n) for (m,s,r,p),n in sorted(anchor_counts.items())))
for file,rows in [('per_sequence.csv',G),('paired_fp.csv',F),('changed_gt_frames.csv',GAIN),('switch_events.csv',SW),('recoveries.csv',REC),('targets.csv',TARGET)]:csvout(file,rows)
csvout('history_conflict_examples.csv',(dict(sequence=q[0],frame=q[1],gt=q[2],method=q[3],tid=q[4],last_anchor=q[5],anchor_age=q[6],previous_match=q[7],distance=q[8],quality=q[9],updates=q[10]) for q in sorted(case_writes,key=lambda r:-r[6])[:400]))
print('COMPLETE',flush=True)
