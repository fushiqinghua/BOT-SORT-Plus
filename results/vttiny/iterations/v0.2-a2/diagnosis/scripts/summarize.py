"""Pair detections and GT frames separately, including direct 0.18 vs 0.20 contrasts."""
import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_evidence import METHODS, csv_file, read


def rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def band(iou):
    if iou < .1: return "0_.1"
    if iou < .3: return ".1_.3"
    if iou < .5-np.finfo(float).eps: return ".3_.5"
    return ".5_1"


def summarize(full,root):
    db=sqlite3.connect(root/"evidence.sqlite");db.row_factory=sqlite3.Row
    official={r["tracker"]:r for r in read(full/"metrics/comparison.json")["results"] if r["tracker"] in METHODS}
    checks=rows(root/"official_checks.csv"); targets=rows(root/"targets.csv")
    switches=rows(root/"switches.csv"); births=rows(root/"weak_births.csv")
    rec=rows(root/"recovery_endpoints.csv")
    totals=defaultdict(Counter)
    pair_data=defaultdict(Counter)
    paired_fp=defaultdict(Counter)
    per_seq=defaultdict(Counter)
    fp_intersection=Counter()
    gt_profiles=defaultdict(Counter)
    track_summaries={}
    born={(q["sequence"],q["method"],int(q["track_id"])):q for q in births}
    sp={(q['sequence'],q['method'],int(q['frame']),int(q['gt'])):q for q in switches}
    comparisons=[("A2B","B_ref"),("A2B_score020","B_ref"),("A2B_score020","A2B")]
    event_features=defaultdict(Counter)
    def add_pair(method, reference, kind, label):
        pair_data[method,reference,kind][label]+=1
    for seq in read(root/"validation.json")['sequences']:
        O={}
        for m in METHODS:
            O[m]={(r["frame"],r['det_idx']):dict(r) for r in db.execute(
                "SELECT frame,det_idx,tid,stage,origin,gid,max_iou,raw_max_iou,score FROM outputs WHERE sequence=? AND method=?",(seq,m))}
            for (frame,idx),o in O[m].items():
                key=seq,m,o["tid"]
                st=track_summaries.setdefault(key,dict(sequence=seq,method=m,track_id=o["tid"],origin=o["origin"],
                    first=frame,last=frame,output_frames=0,tp=0,fp=0,peak_iou=0.,fp_without_B_row=0,
                    fp_with_B_tp=0,fp_with_B_fp=0,fp_without_other_A_row=0,fp_with_other_A_fp=0))
                st["first"]=min(frame,st["first"]);st["last"]=max(frame,st["last"]);st["output_frames"]+=1
                st["peak_iou"]=max(st["peak_iou"],o["max_iou"])
                if o['gid'] is None:
                    st['fp']+=1
                    for name,val in [('fp_origin',o['origin']),('fp_stage',o['stage']),('fp_geom',band(o['max_iou'])),
                                     ('fp_raw_geom',band(o['raw_max_iou'])+' to '+band(o['max_iou']))]:
                        totals[m,name][val]+=1
                else:st['tp']+=1
        for m in METHODS[1:]:
            alt=METHODS[2] if m==METHODS[1] else METHODS[1]
            for key,o in O[m].items():
                if o["gid"] is not None:continue
                st=track_summaries[seq,m,o["tid"]]
                ref=O["B_ref"].get(key);other=O[alt].get(key)
                st["fp_without_B_row" if ref is None else "fp_with_B_fp" if ref["gid"] is None else "fp_with_B_tp"]+=1
                if other is None:st['fp_without_other_A_row']+=1
                elif other['gid'] is None:st['fp_with_other_A_fp']+=1
        for method,ref in comparisons:
            per_seq.setdefault((seq,method,ref),Counter())
            paired_targets={(frame,o["gid"]):o for (frame,_),o in O[method].items() if o["gid"] is not None}
            for key in set(O[method])|set(O[ref]):
                a=O[method].get(key);b=O[ref].get(key)
                u='absent' if a is None else 'FP' if a['gid'] is None else 'TP'
                v='absent' if b is None else 'FP' if b['gid'] is None else 'TP'
                add_pair(method,ref,'detection_row',v+' to '+u)
                if v=='FP' and u=='FP':
                    fp_intersection[seq,method,ref]+=1
                if v==u:continue
                if u=='FP' or v=='FP':
                    which='added' if u=='FP' else 'removed'
                    obj=a if u=='FP' else b
                    stats=track_summaries[seq,method if u=='FP' else ref,obj['tid']]
                    outcome='matched_sometime' if stats['tp'] else 'never_near' if stats['peak_iou']<.1 else 'near_unmatched'
                    li=[('max_gt_iou',band(obj['max_iou'])),('origin',obj['origin']),('stage',obj['stage']),
                        ('raw_to_output',band(obj['raw_max_iou'])+' to '+band(obj['max_iou'])),
                        ('track_lifetime',outcome),('source_and_lifetime',obj['origin']+'|'+outcome)]
                    for fld,cat in li:paired_fp[method,ref,which,fld][cat]+=1
                    per_seq[seq,method,ref][which+'_FP_rows']+=1
            for g in db.execute("SELECT * FROM gt_frames WHERE sequence=?",(seq,)):
                f,gid=g['frame'],g['gt']
                if g[method] is not None and g[ref] is None:
                    per_seq[seq,method,ref]['gained_gt_frames']+=1
                    output=paired_targets[f,gid]
                    assert output['tid']==g[method]
                    add_pair(method,ref,'gained_origin',output['origin'])
                    add_pair(method,ref,'gained_stage',output['stage'])
                    score=g['best_aligned_score']
                    group='no_aligned_raw' if score is None else 'lt0.5' if score < .5 else 'ge0.5'
                    add_pair(method,ref,'gained_aligned_score',group)
                if g[method] is None and g[ref] is not None:
                    per_seq[seq,method,ref]['lost_gt_frames']+=1
                a=sp.get((seq,method,f,gid));b=sp.get((seq,ref,f,gid))
                if a or b:
                    side='shared' if a and b else 'variant_only' if a else 'ref_only'
                    add_pair(method,ref,'switch_pair',side)
                    per_seq[seq,method,ref][side+'_switches']+=1
                    if a and not b:
                        for key,val in [('new_origin_pair',a['old_origin']+'->'+a['new_origin']),
                                        ('stage',a['stage']),('ref_endpoints',a[ref+'_endpoints'])]:
                            add_pair(method,ref,'variant_only_switch_'+key,val)
        for m in METHODS:
            for e in (x for x in switches if x['method']==m and x['sequence']==seq):
                for key,val in [('switch_stage',e['stage']),('switch_pair',e['old_origin']+'->'+e['new_origin']),
                                ('switch_endpoints_B',e['B_ref_endpoints'])]:
                    totals[m,key][val]+=1
                if e['switch_at_weak_birth']=='1':
                    code=e['guard_for_old_tid']
                    if code=='not_in_guard_list':
                        if e['old_tid_output_now']=='1':code='absent_old_emitted_now'
                        elif int(e['frame'])-int(e['old_tid_last_output_before'])>30:code='absent_last_output_over_30'
                        else:code='absent_other_within_30'
                    totals[m,'guard_last_GT_tid_at_switch_birth'][code]+=1
                    totals[m,'recovery_check_last_tid_at_switch_birth'][e['recovery_check_for_old_tid'] or 'no_pair_check']+=1
            for q in (x for x in rec if x['sequence']==seq and x['method']==m):
                totals[m,'recovery_origin_endpoints'][q['track_origin']+'|'+q['endpoint_outcome']]+=1
    for z in track_summaries.values():
        k=z['sequence'],z['method'],z['track_id']
        if k not in born:continue
        b=born[k]; assert z['output_frames']==int(b['output_frames'])
        life='ever_matched' if z['tp'] else 'never_near' if z['peak_iou']<.1 else 'near_unmatched'
        m=z['method'];b['lifetime']=life
        for name in ['tp','fp','peak_iou','first','last','fp_without_B_row','fp_with_B_tp','fp_with_B_fp','fp_with_other_A_fp','fp_without_other_A_row']:
            b[name]=z[name]
        totals[m,'birth_outcome'][life]+=1
        totals[m,'birth_support_x_outcome'][b['gt_support']+'|'+life]+=1
        if life=='never_near':
            event_features[m,'far_continuation']['all_output_frames']+=z['fp']
            event_features[m,'far_continuation']['no_B_row']+=z['fp_without_B_row']
            event_features[m,'far_continuation']['raw_support_all_lt10']+=int(b['support_all_lt10'])
            event_features[m,'far_continuation']['ever_recovered_ids']+=int(int(b['weak_recoveries'])>0)
        event_features[m,'birth_support']['too_low_for_020_at_same_time']+=int(min(float(b['armed_mean']),float(b['verify_mean']))<.20)
    for m in METHODS[1:]:
        seen_stats=[]
        for q in targets:
            for label,val in [('class',q['category']),('size',q['size_bin'])]:
                ctr=gt_profiles[m,label,val]
                ctr['official_gt']+=1;ctr['unseen_B']+=int(int(q['B_ref_matched'])==0)
                ctr['novel_ids']+=int(q[m+'_novel_id']);ctr['gained']+=int(q[m+'_gained'])
                ctr['lost']+=int(q[m+'_lost']);ctr['novel_matched_frames']+=int(q[m+'_new_target_frames'])
            if q[m+'_novel_id']=='1':seen_stats.append(int(q[m+'_matched']))
            for typ in ['new_target_frames','early_frames','internal_frames','late_frames']:
                totals[m,'gain_role'][typ]+=int(q[m+'_'+typ])
        totals[m,'new_ID_quality'].update(dict(n=len(seen_stats),one_frame=seen_stats.count(1),
                    at_most_three=sum(v<=3 for v in seen_stats),at_least_20=sum(v>=20 for v in seen_stats),
                    median_frames=float(np.median(seen_stats))))
    tables = {n:t for n,t in [('track_outcomes',track_summaries.values()),('weak_birth_lifetimes',births)]}
    tables['target_profiles']=[dict(method=m,group=fld,value=val,**c) for (m,fld,val),c in gt_profiles.items()]
    tables['paired_fp_breakdown']=[dict(method=m,reference=b,direction=dr,field=f,label=k,count=v)
              for (m,b,dr,f),c in paired_fp.items() for k,v in sorted(c.items())]
    tables['paired_sequence_results']=[dict(sequence=q,method=m,reference=b,**{
           k:c[k] for k in ['gained_gt_frames','lost_gt_frames','added_FP_rows','removed_FP_rows','shared_switches','variant_only_switches','ref_only_switches']},
           common_FP_rows=fp_intersection[q,m,b]) for (q,m,b),c in per_seq.items()]
    for name,vals in tables.items():csv_file(root/(name+'.csv'),vals)
    stats={}
    for m in METHODS:
        stats[m]={k:dict(v) for (n,k),v in totals.items() if m==n}
        stats[m].update({k:dict(v) for (n,k),v in event_features.items() if m==n})
        assert sum(stats[m]['fp_geom'].values())==official[m]['FP']
        assert sum(stats[m]['switch_stage'].values())==official[m]['IDSW']
        assert sum(int(q['FP']) for q in checks if q['method']==m)==official[m]['FP']
    for method,ref in comparisons:
        key=method+'_vs_'+ref;stats[key]={k:dict(c) for (m,b,k),c in pair_data.items() if (method,ref)==(m,b)}
        fp=stats[key]['detection_row'];sw=stats[key]['switch_pair']
        added=fp.get('absent to FP',0)+fp.get('TP to FP',0)
        removed=fp.get('FP to absent',0)+fp.get('FP to TP',0)
        assert added-removed==official[method]['FP']-official[ref]['FP']
        assert sw['variant_only']-sw['ref_only']==official[method]['IDSW']-official[ref]['IDSW']
        assert added==sum(paired_fp[method,ref,'added','origin'].values())
        stats[key]['fp_breakdown']={dr:{f:dict(v) for (m,b,d,f),v in paired_fp.items() if (m,b,d)==(method,ref,dr)}
                                     for dr in ['added','removed']}
    stats['category_frames']={row[0]:row[1] for row in db.execute("SELECT category,COUNT(*) FROM gt_frames GROUP BY category")}
    tri=db.execute("SELECT B_ref IS NOT NULL,A2B IS NOT NULL,A2B_score020 IS NOT NULL,COUNT(*) FROM gt_frames GROUP BY 1,2,3")
    stats['gt_intersection']={','.join(str(v) for v in row[:3]):row[3] for row in tri}
    stats['official']=official
    (root/"summary.json").write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding="utf-8")
    db.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--run-dir",type=Path,default=Path(__file__).resolve().parents[2]/"full30")
    p.add_argument("analysis",type=Path)
    a=p.parse_args()
    summarize(a.run_dir,a.analysis)
