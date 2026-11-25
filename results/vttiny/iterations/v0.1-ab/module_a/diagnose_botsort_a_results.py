"""Compare frozen baseline and A at matched GT, prediction, and candidate level."""
import argparse
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_csv(path, records):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def extract_events(path):
    origins, handoff, signals = {}, {}, defaultdict(list)
    birth, update = {}, {}
    source = path.open(encoding='utf-8') if path.exists() else gzip.open(str(path)+'.gz', 'rt', encoding='utf-8')
    with source as stream:
        for line in stream:
            e=json.loads(line)
            kind, f=e['event'],e['frame']
            if kind in ('weak_birth','native_birth'):
                t=e['track_id']
                assert t not in origins
                origins[t]='weak' if kind=='weak_birth' else 'native'
                birth[t]=e
            if kind in ('match','native_birth','weak_birth'):
                key=(f,e['track_id'])
                assert key not in update
                update[key]=e
            if kind=='weak_handoff':
                handoff[e['track_id']]=e
            if kind in ('candidate_seed','candidate_hit'):
                signals[e['candidate_id']].append(e)
    return origins, handoff, signals, birth, update


def bucket(iou):
    if iou < .1: return 'under_10'
    if iou < .3: return '10_30'
    if iou < .5-np.finfo(float).eps: return '30_50'
    return 'ge_50'


def side_rows(values):
    return {int(g[0]): i for i,g in enumerate(values)}


def series_stats(xs):
    if not xs:
        return {}
    return dict(n=len(xs),p25=float(np.percentile(xs,25)),median=float(np.median(xs)),p75=float(np.percentile(xs,75)))


def run(repo, exp, diag, gt_folder):
    sys.path.insert(0,str(repo/'tools'))
    from diagnose_vttiny_botsort_m import mot, clear_step, overlaps
    dest=diag/'analysis'
    dest.mkdir(exist_ok=True)
    seqs=load(exp/'protocols/A.json')['sequences']
    official={(r['tracker'],r['sequence']):r for r in load(exp/'metrics/summary.json')['results']}
    cases, switches, promotions, gt_totals, paired_sequences=[],[],[],[],[]
    totals=defaultdict(Counter)
    seq_detail=defaultdict(dict)
    docontrol=diag/'controls'
    for seq,n in seqs.items():
        path=exp/'trackers/A/data'/(seq+'.txt')
        assert path.read_bytes()==(docontrol/'trackers/audit_A/data'/(seq+'.txt')).read_bytes(),seq
        pred={'A':mot(path,n),'baseline':mot(exp/'trackers/baseline/data'/(seq+'.txt'),n)}
        gt=mot(gt_folder/seq/'gt/gt.txt',n,True)
        with gzip.open(repo/'results/vttiny/yolo11m_rgb_rgbt/rgbt/detections'/(seq+'.json.gz'),'rt') as f:
            cache=json.load(f)['frames']
        origin, handoff, signals, birth, updates=extract_events(docontrol/'events/audit_A'/(seq+'.jsonl'))
        seqcounts=defaultdict(Counter)
        gth=defaultdict(dict)
        th=defaultdict(dict)
        detgeom={}
        previous={k:{} for k in pred}
        last={k:{} for k in pred}
        for f,g in enumerate(gt,1):
            d=np.asarray(cache[f-1]['detections']).reshape(-1,6)
            db=d[:,:4].copy()
            db[:,2:]-=db[:,:2]
            di=overlaps(g[:,1:5],db)
            assert cache[f-1]['frame']==f
            ids=g[:,0].astype(int)
            for j in range(len(d)):
                col=di[:,j]
                detgeom[f,j]=(float(col.max(initial=0)),ids[col>=.5-np.finfo(float).eps].tolist(),float(d[j,4]))
            matches={}
            for name in ('baseline','A'):
                t=pred[name][f-1]
                ij,ti=clear_step(g,t,previous[name])
                reverse={j:i for i,j in ij.items()}
                matches[name]={i:int(t[j,0]) for i,j in ij.items()}
                for j,row in enumerate(t):
                    tid=int(row[0])
                    orig=origin[tid] if name=='A' else 'baseline'
                    ok=j in reverse
                    iou=float(ti[:,j].max(initial=0))
                    seqcounts['outputs'][orig+('|TP' if ok else '|FP')]+=1
                    if not ok:
                        seqcounts['fp_geometry'][orig+'|'+bucket(iou)]+=1
                    idx=None
                    if name=='A':
                        e=updates[f,tid]
                        idx=e['det_index']
                        assert abs(d[idx,4]-row[5])<1e-6
                        raw_iou=detgeom[f,idx][0]
                        if not ok:
                            seqcounts['fp_raw_vs_track'][orig+'|'+bucket(raw_iou)+'|'+bucket(iou)]+=1
                            seqcounts['fp_stage'][orig+'|'+e.get('stage',e['event'])]+=1
                        th[tid][f]=dict(gid=int(ids[reverse[j]]) if ok else None, iou=iou, raw_iou=raw_iou,
                                        stage=e.get('stage',e['event']),idx=idx)
            for i,gid in enumerate(ids):
                gid=int(gid)
                base=matches['baseline'].get(i)
                a=matches['A'].get(i)
                role=('both' if base is not None else 'gained') if a is not None else ('lost' if base is not None else 'neither')
                seqcounts['paired_gt'][role]+=1
                if a is not None:
                    seqcounts['paired_source'][role+'|'+origin[a]]+=1
                old_b=last['baseline'].get(gid)
                b_switch=bool(base is not None and old_b and old_b[0]!=base)
                old_a=last['A'].get(gid)
                a_switch=bool(a is not None and old_a and old_a[0]!=a)
                gth[gid][f]=(base,a)
                if a_switch:
                    prev,prevf=old_a
                    event=birth[a]
                    born=f==event['frame']
                    audit=handoff.get(a,{}) if born else {}
                    rivals={v['id']:v for v in audit.get('lost',[])}
                    old_record=rivals.get(prev)
                    prior_base=gth[gid][prevf][0]
                    base_stable=(prior_base is not None and prior_base==base)
                    switches.append(dict(sequence=seq,gt=gid,previous_frame=prevf,frame=f,previous_id=prev,id=a,
                        gap=f-prevf-1,previous_origin=origin[prev],origin=origin[a],id_age=f-event['frame'],
                        birth_now=born,previous_id_in_lost_at_weak_birth=bool(old_record),
                        previous_lost_iou=None if old_record is None else old_record['iou'],
                        baseline_previous=prior_base,baseline_current=base,baseline_same_id_at_endpoints=base_stable,
                        baseline_switch_same_frame=b_switch))
                    seqcounts['switch_origin'][origin[prev]+'->'+origin[a]]+=1
                if base is not None: last['baseline'][gid]=(base,f)
                if a is not None: last['A'][gid]=(a,f)
                seqcounts['switch_pair'][('base_yes' if b_switch else 'base_no')+'|'+('A_yes' if a_switch else 'A_no')]+=1
        for gid,h in gth.items():
            stats={}
            for i,name in enumerate(('baseline','A')):
                valid=[t[i] for t in h.values() if t[i] is not None]
                stats[name+'_matched']=len(valid)
                stats[name+'_ids']=len(set(valid))
                stats[name+'_switches']=sum(a!=b for a,b in zip(valid,valid[1:]))
            gt_totals.append(dict(sequence=seq,gt=gid,frames=len(h),**stats))
        for tid,e in birth.items():
            if origin[tid]!='weak':
                continue
            f=e['frame']; cid=e['candidate_id']
            obs=[s for s in signals[cid] if f-5<s['frame']<=f]
            assert obs[-1]['det_index']==e['det_index'] and obs[-1]['frame']==f and len(obs)==e['hits']
            geometry=[detgeom[s['frame'],s['det_index']] for s in obs]
            common=set(geometry[0][1]).intersection(*(set(v[1]) for v in geometry[1:]))
            h=th[tid]
            hits=[(fr,row['gid']) for fr,row in h.items() if row['gid'] is not None]
            hf=handoff[tid]
            born_gid=h.get(f,{}).get('gid')
            oldid=oldframe=None
            if born_gid is not None:
                prior=[(fr,arr[1]) for fr,arr in gth[born_gid].items() if fr<f and arr[1] is not None]
                if prior: oldframe,oldid=prior[-1]
            opp={x['id']:x for x in hf['lost']}
            support='same_gt_all' if common else 'no_aligned_hit' if all(not v[1] for v in geometry) else 'mixed'
            peak=max((q['iou'] for q in h.values()),default=-1)
            assert (f in h)==hf['kept'] and all(abs(v)<1e-8 for v in hf['velocity'])
            promotions.append(dict(sequence=seq,frame=f,id=tid,candidate=cid,evidence=e['evidence'],hits=e['hits'],
                span=f-obs[0]['frame']+1,mean_score=np.mean([v[2] for v in geometry]),score=geometry[-1][2],
                max_score=max(v[2] for v in geometry),mean_geometry=np.mean([s.get('geometry_cost',0) for s in obs]),
                gt_support=support,raw_all_iou_lt10=all(v[0]<.1 for v in geometry),confirmed_gt_ids=','.join(map(str,sorted(common))),
                birth_raw_max_iou=geometry[-1][0],birth_gt=born_gid,baseline_at_birth=None if born_gid is None else gth[born_gid][f][0],
                native_kept=hf['kept'],lost_max_iou=max((x['iou'] for x in opp.values()),default=0),lost_count=len(opp),
                smooth_rgb_at_birth=hf['smooth_feature'],prior_A_id=oldid,prior_A_frame=oldframe,prior_id_in_lost=oldid in opp,
                prior_lost_iou=opp[oldid]['iou'] if oldid in opp else None,output_frames=len(h),matched_frames=len(hits),
                fp_frames=len(h)-len(hits),lifetime_max_iou=peak,last_frame=max(h,default=None),unique_gt=len(set(g for _,g in hits))))
        m0,ma=official['baseline',seq],official['A',seq]
        assert sum(v for k,v in seqcounts['switch_origin'].items())==ma['IDSW']
        assert sum(seqcounts['switch_pair'][x] for x in ('base_yes|A_no','base_yes|A_yes'))==m0['IDSW']
        assert seqcounts['outputs']['baseline|FP']==m0['FP']
        assert sum(seqcounts['outputs'][x+'|FP'] for x in ('weak','native'))==ma['FP']
        assert seqcounts['paired_gt']['gained']-seqcounts['paired_gt']['lost']==m0['FN']-ma['FN']
        for k,v in seqcounts.items():
            totals[k].update(v)
            seq_detail[seq][k]=dict(v)
        paired_sequences.append(dict(sequence=seq,frames=n,gt=sum(map(len,gt)),gained=seqcounts['paired_gt']['gained'],
                                    lost=seqcounts['paired_gt']['lost'],weak_fp=seqcounts['outputs']['weak|FP'],
                                    native_fp=seqcounts['outputs']['native|FP'],baseline_fp=m0['FP'],
                                    delta_HOTA=ma['HOTA']-m0['HOTA'],delta_FP=ma['FP']-m0['FP'],delta_IDSW=ma['IDSW']-m0['IDSW']))
        # A few pre-specified GT trajectories can be inspected by frame in the delivery.
        for gid in ([5,3] if seq=='DJI_0030_2' else [1] if seq=='DJI_0129_2' else []):
            for f, (base,a) in gth[gid].items():
                i=side_rows(gt[f-1])[gid]
                candidates=np.asarray(cache[f-1]['detections']).reshape(-1,6)
                boxes=candidates[:,:4].copy();boxes[:,2:]-=boxes[:,:2]
                overlap=overlaps(gt[f-1][i:i+1,1:5],boxes)[0]
                ok=overlap>=.5-np.finfo(float).eps
                cases.append(dict(sequence=seq,gt=gid,frame=f,baseline_id=base,A_id=a,A_origin=None if a is None else origin[a],
                                  best_det_iou=overlap.max(initial=0),aligned_max_score=candidates[ok,4].max(initial=-1)))
    for fname,data in [('promotions',promotions),('switches',switches),('gt_tracks',gt_totals),
                       ('paired_sequences',paired_sequences),('case_frames',cases)]:
        save_csv(dest/(fname+'.csv'),data)
    assert len(promotions)==3393 and len(switches)==1549 and len(gt_totals)==870
    outcomes=defaultdict(Counter)
    for e in promotions:
        outcome=('no_output' if not e['output_frames'] else 'matched' if e['matched_frames'] else
                 'never_near' if e['lifetime_max_iou']<.1 else 'other_unmatched')
        outcomes[e['gt_support']][outcome]+=1
    evidence={}
    for name,subset in [('ever_matched',[x for x in promotions if x['matched_frames']]),
                        ('never_near',[x for x in promotions if 0<=x['lifetime_max_iou']<.1]),
                        ('all',promotions)]:
        evidence[name]={k:series_stats([e[k] for e in subset]) for k in ('evidence','mean_score','mean_geometry','output_frames')}
        evidence[name]['3_hits_in_3_frames']=sum(e['span']==3 and e['hits']==3 for e in subset)
    s=dict(counts=totals,promotion_support=outcomes,evidence=evidence,sequence=seq_detail)
    (dest/'attribution.json').write_text(json.dumps(s,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in s.items() if k!='sequence'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('repo','exp','diag','gt-folder'):
        p.add_argument('--'+name,type=Path,required=True)
    run(**vars(p.parse_args()))
