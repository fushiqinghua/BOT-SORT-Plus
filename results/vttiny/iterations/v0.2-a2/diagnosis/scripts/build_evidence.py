"""Offline attribution on three frozen VT-Tiny-MOT runs. No tracker update is invoked."""
import argparse
import csv
import gzip
import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

METHODS = ("B_ref", "A2B", "A2B_score020")
EPS = np.finfo(float).eps


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_file(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_categories(path):
    source = read(path)
    images = {i["id"]: (i["file_name"].split("/")[0], int(i["frame_id"])+1) for i in source["images"]}
    names = {c["id"]: c["name"] for c in source["categories"]}
    labels = defaultdict(set)
    for ann in source["annotations"]:
        if not ann.get("ignore",0) and not ann.get("iscrowd",0):
            labels[(*images[ann["image_id"]], *ann["bbox"])].add(names[ann["category_id"]])
    return labels


def parse_trace(path):
    updates, births, guarded, signals, rec_checks, lifetime = {}, {}, {}, {}, {}, {}
    source = path.open(encoding="utf-8") if path.exists() else gzip.open(str(path)+".gz","rt",encoding="utf-8")
    with source as src:
        for line in src:
            e = json.loads(line)
            f, k = e["frame"], e["event"]
            if k in ("match", "native_birth", "birth_committed", "recovered"):
                key = f, e["track_id"]
                assert key not in updates
                updates[key] = e
                if k in ("native_birth","birth_committed"):
                    assert e["track_id"] not in births
                    births[e["track_id"]] = e
            elif k == "track_lifetime":
                lifetime[e["track_id"]] = e
            elif k == "birth_identity":
                guarded[f,e["candidate_id"]] = e
            elif k in ("candidate_seed", "candidate_hit"):
                signals[f,e["candidate_id"]] = e["det_index"]
            elif k == "recovery_check":
                rec_checks[f,e["candidate_id"],e["track_id"]] = e["reason"]
    return updates, births, guarded, signals, rec_checks, lifetime


def make_db(path):
    if path.exists():
        raise FileExistsError(path)
    db=sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE gt_frames(
          sequence TEXT,gt INT,frame INT,x REAL,y REAL,w REAL,h REAL,category TEXT,
          B_ref INT,A2B INT,A2B_score020 INT,best_det_iou REAL,best_aligned_score REAL,
          PRIMARY KEY(sequence,gt,frame)) WITHOUT ROWID;
        CREATE TABLE outputs(
          sequence TEXT,method TEXT,frame INT,tid INT,det_idx INT,stage TEXT,origin TEXT,gid INT,
          nearest_gid INT,max_iou REAL,matched_iou REAL,raw_max_iou REAL,score REAL,
          x REAL,y REAL,w REAL,h REAL,rx REAL,ry REAL,rw REAL,rh REAL,
          PRIMARY KEY(sequence,method,frame,tid),
          UNIQUE(sequence,method,frame,det_idx)) WITHOUT ROWID;
    """)
    return db


def insert_values(db, table, data):
    if data:
        marks = ",".join("?" for _ in data[0])
        db.executemany(f"INSERT INTO {table} VALUES ({marks})", data)


def bucket(area):
    side = np.sqrt(area)
    return "lt8" if side < 8 else "8_16" if side < 16 else "16_32" if side < 32 else "ge32"


def ref_ends(before, now):
    if before is None and now is None: return "neither"
    if before is None: return "current_only"
    if now is None: return "previous_only"
    return "same" if before==now else "different"


def analyze(full, result, gt_root, co_path):
    repo=Path(__file__).resolve().parents[6]
    sys.path.insert(0,str(repo/"tools"))
    from diagnose_vttiny_botsort_m import clear_step, mot, overlaps
    result.mkdir(parents=True,exist_ok=True)
    db=make_db(result/"evidence.sqlite")
    protocols={m:read(full/"protocols"/(m+".json")) for m in METHODS}
    p=protocols[METHODS[0]]
    keys=("sequences","detector_sha256","rgb_encoder_sha256","source_protocol_sha256",
          "detection_fingerprints","feature_hashes","native_config","implementation_sha256","input_roots")
    assert all(all(v[k]==p[k] for k in keys) and v["status"]=="COMPLETE" for v in protocols.values())
    audit=read(full/"metrics/evaluation_audit.json")
    actual=read(full/"structure_audit.json")
    assert audit["status"]=="COMPLETE" and actual["status"]=="PASS"
    labels=read_categories(co_path)
    cache_root=repo/"results/vttiny/yolo11m_rgb_rgbt/rgbt/detections"
    official={(v["tracker"],v["sequence"]):v for v in read(full/"metrics/summary.json")["results"]}
    sw_all, births_all, rec_all, gts_all, checks = [], [], [], [], []
    for seq,n in p["sequences"].items():
        truth=gt_root/seq/"gt/gt.txt"
        assert digest(truth)==audit["ground_truth"]["sha256"][seq]
        gt=mot(truth,n,True)
        with gzip.open(cache_root/(seq+".json.gz"),"rt") as src:
            cache=json.load(src)
        assert cache["model_sha256"]==p["detector_sha256"] and len(cache["frames"])==n
        raw_dets, raw_iou = [], []
        for f,(g,c) in enumerate(zip(gt,cache["frames"]),1):
            assert c["frame"]==f
            d=np.asarray(c["detections"],dtype=float).reshape(-1,6).copy()
            d[:,2:4]-=d[:,:2]
            raw_dets.append(d)
            raw_iou.append(overlaps(g[:,1:5],d[:,:4]))
        mates, switches, infos = {}, {}, {}
        all_out = []
        for method in METHODS:
            pred_file=full/"trackers"/method/"data"/(seq+".txt")
            assert digest(pred_file)==protocols[method]["output_prediction_hashes"][seq]
            pred=mot(pred_file,n)
            update,born,guard,signals,rc,life=parse_trace(full/"events"/method/(seq+".jsonl"))
            previous,last_gt,last_on_tid,last_det={}, {}, {}, {}
            h, swh, confirmed, fp_count = {}, {}, [], 0
            emitted=set()
            gt_by_tid={}
            for f,(g,t,d,ri) in enumerate(zip(gt,pred,raw_dets,raw_iou),1):
                matches,iou=clear_step(g,t,previous)
                inverse={j:i for i,j in matches.items()}
                now={int(t[j,0]):int(g[i,0]) for i,j in matches.items()}
                gid_now={gid:tid for tid,gid in now.items()}
                for j,row in enumerate(t):
                    tid=int(row[0])
                    e=update[f,tid]
                    stage_name=e.get("stage",e["event"])
                    idx=int(e["det_index"])
                    assert abs(float(d[idx,4])-row[5])<1e-6
                    origin="weak" if born[tid]["event"]=="birth_committed" else "native"
                    near=int(np.argmax(iou[:,j])) if len(g) else None
                    max_iou=float(iou[:,j].max(initial=0))
                    paired_iou=float(iou[inverse[j],j]) if j in inverse else None
                    all_out.append((seq,method,f,tid,idx,stage_name,origin,now.get(tid),
                                    int(g[near,0]) if max_iou>0 else None,max_iou,paired_iou,
                                    float(ri[:,idx].max(initial=0)),float(d[idx,4]),*row[1:5],*d[idx,:4]))
                    emitted.add((f,tid))
                    fp_count+=int(j not in inverse)
                    if tid in now:gt_by_tid[f,tid]=now[tid]
                for gid,tid in gid_now.items():
                    h[f,gid]=tid
                    old=last_gt.get(gid)
                    if old and old[1]!=tid:
                        pf,prev=old
                        b=born[tid];weak=b["event"]=="birth_committed"
                        on_birth=weak and f==b["frame"]
                        prevg=last_on_tid.get(tid)
                        old_info,rv,rc_reason=None,None,None
                        if on_birth:
                            cid=b["candidate_id"];v=guard[f,cid]
                            old_info=next((o for o in v['diagnostics'] if o['track_id']==prev),None)
                            rv=old_info['reason'] if old_info else 'not_in_guard_list'
                            rc_reason=rc.get((f,cid,prev))
                        swh[f,gid]=dict(sequence=seq,method=method,gt=gid,frame=f,previous_frame=pf,
                            gap=f-pf-1,old_tid=prev,new_tid=tid,
                            old_origin="weak" if born[prev]["event"]=="birth_committed" else "native",
                            new_origin="weak" if weak else "native",stage=update[f,tid].get('stage',update[f,tid]['event']),
                            new_tid_age=f-b['frame'],switch_at_weak_birth=int(on_birth),
                            last_gt_on_arriving_tid=prevg,old_tid_gt_now=now.get(prev),
                            old_tid_output_now=int((f,prev) in emitted),old_tid_last_output_before=last_det.get(prev),
                            guard_for_old_tid=rv,old_pred_age=old_info['age'] if old_info else None,
                            recovery_check_for_old_tid=rc_reason)
                    last_gt[gid]=f,tid
                last_on_tid.update(now)
                for row in t:last_det[int(row[0])]=f
            for (f,tid),e in update.items():
                if e['event']!='recovered':continue
                oldf=f-e['age'];old_gid=gt_by_tid.get((oldf,tid));new_gid=gt_by_tid.get((f,tid))
                rec_all.append(dict(sequence=seq,method=method,frame=f,track_id=tid,prior_frame=oldf,
                    age=e["age"],track_origin=e["track_origin"],prev_gt=old_gid,current_gt=new_gid,
                    endpoint_outcome=("unobserved" if old_gid is None or new_gid is None else
                                      "same_gt" if old_gid==new_gid else "different_gt"),
                    cost=e['cost'],candidate=e["candidate_id"]))
            for tid,b in born.items():
                if b['event']!="birth_committed":continue
                f,cid=b["frame"],b["candidate_id"];sup=b['support'];v=guard[f,cid]
                assert v["status"]=="CLEAR_NEW" and (f,tid) in emitted
                groups=[sup['armed_frames'],sup['verify_frames']]
                assert max(groups[0]) < min(groups[1]) and groups[1][-1]==f
                records=[]; aligned=[]
                for phase,fs in zip(["armed","verified"],groups):
                    scores=[]
                    for fr in fs:
                        idx=signals[fr,cid];col=raw_iou[fr-1][:,idx]
                        ok=gt[fr-1][col>=.5-EPS,0].astype(int).tolist()
                        aligned.append(set(ok))
                        conf=float(raw_dets[fr-1][idx,4]);scores.append(conf)
                        records.append(dict(phase=phase,frame=fr,det_idx=idx,score=round(conf,7),
                                            max_gt_iou=float(col.max(initial=0)),aligned_gt_ids=ok))
                    intended=sup['armed_mean_score' if phase=='armed' else 'verify_mean_score']
                    assert abs(np.mean(scores)-intended)<1e-6
                common=set.intersection(*aligned)
                excluded=Counter(o["reason"] for o in v["diagnostics"])
                assert not any(o['conflict'] for o in v['diagnostics'])
                assert sum(1 for fr,t in emitted if t==tid)==life[tid]['output_frames']
                confirmed.append(dict(sequence=seq,method=method,track_id=tid,frame=f,candidate=cid,
                    det_idx=b["det_index"],first_candidate=b['candidate_first_frame'],delay=b['confirmation_delay_frames'],
                    birth_gt=gt_by_tid.get((f,tid)),armed_mean=sup['armed_mean_score'],verify_mean=sup['verify_mean_score'],
                    gt_support="same_gt_all" if common else "no_aligned_hit" if not any(aligned) else "mixed",
                    support_all_lt10=int(all(rec["max_gt_iou"]<.1 for rec in records)),
                    old_ids_checked=len(v["diagnostics"]),excluded_geometry=excluded["current_geometry_excludes"],
                    excluded_appearance=excluded["current_appearance_excludes"],
                    weak_recoveries=life[tid]['recovery_count'],output_frames=life[tid]['output_frames'],
                    support_records=json.dumps(records,ensure_ascii=False,separators=(',',':'))))
            ref=official[method,seq];ng=sum(map(len,gt))
            assert (fp_count,ng-len(h),len(swh))==(ref["FP"],ref["FN"],ref["IDSW"]), (method,seq)
            assert len(confirmed)==actual["trackers"][method]["sequences"][seq]["births_committed"]
            checks.append(dict(sequence=seq,method=method,frames=n,gt_frames=ng,FP=fp_count,FN=ng-len(h),IDSW=len(swh),
                               prediction_sha256=digest(pred_file),GT_sha256=digest(truth)))
            mates[method]=h;switches[method]=swh;births_all+=confirmed
            print(seq,method,len(h),len(swh),flush=True)
        insert_values(db,"outputs",all_out)
        for method,sv in switches.items():
            for (f,gid),z in sv.items():
                pf=z["previous_frame"]
                for other,h in mates.items():
                    z[other+"_previous"]=h.get((pf,gid))
                    z[other+"_current"]=h.get((f,gid))
                    z[other+"_switch_here"]=int((f,gid) in switches[other])
                    z[other+"_endpoints"]=ref_ends(h.get((pf,gid)),h.get((f,gid)))
                sw_all.append(z)
        gtrows=[]
        for f,(g,ri,d) in enumerate(zip(gt,raw_iou,raw_dets),1):
            for i,row in enumerate(g):
                gid=int(row[0]); box=tuple(map(float,row[1:5]))
                cl=labels.get((seq,f,*box),set())
                category=next(iter(cl)) if len(cl)==1 else "ambiguous" if cl else "unmatched"
                scores=d[ri[i]>=.5-EPS,4]
                best=float(scores.max()) if len(scores) else None
                gtrows.append((seq,gid,f,*box,category,*(mates[m].get((f,gid)) for m in METHODS),
                               float(ri[i].max(initial=0)),best))
                infos.setdefault(gid,[]).append((f,box,category))
        insert_values(db,"gt_frames",gtrows);db.commit()
        for gid,info in infos.items():
            frames=[f for f,_,_ in info]
            known=Counter(c for _,_,c in info if c not in ("ambiguous","unmatched"))
            area=float(np.median([b[2]*b[3] for _,b,_ in info]))
            rr=dict(sequence=seq,gt=gid,category=next(iter(known)) if len(known)==1 else "mixed" if known else "unverified",
                    category_supported_frames=sum(known.values()),categories_observed=json.dumps(known),
                    annotated_frames=len(info),first_gt=frames[0],last_gt=frames[-1],
                    median_sqrt_area_px=round(np.sqrt(area),3),size_bin=bucket(area))
            covered={}
            for m in METHODS:
                h=mates[m];matched=[f for f in frames if (f,gid) in h];covered[m]=set(matched)
                rr.update({m+"_matched":len(matched),m+"_first":min(matched,default=None),
                           m+"_last":max(matched,default=None),m+"_ids":len({h[f,gid] for f in matched}),
                           m+"_switches":sum((f,gid) in switches[m] for f in matched)})
            B=covered["B_ref"]
            for m in METHODS[1:]:
                gain=covered[m]-B
                rr.update({m+"_gained":len(gain),m+"_lost":len(B-covered[m]),m+"_novel_id":int(not B and bool(gain)),
                           m+"_new_target_frames":len(gain) if not B else 0,
                           m+"_early_frames":sum(f<min(B) for f in gain) if B else 0,
                           m+"_late_frames":sum(f>max(B) for f in gain) if B else 0,
                           m+"_internal_frames":sum(min(B)<f<max(B) for f in gain) if B else 0})
            rr["score020_only_frames"]=len(covered[METHODS[2]]-covered[METHODS[1]])
            rr["score018_only_frames"]=len(covered[METHODS[1]]-covered[METHODS[2]])
            gts_all.append(rr)
    db.executescript("CREATE INDEX output_track ON outputs(method,sequence,tid,frame);"
                     "CREATE INDEX gt_frame ON gt_frames(sequence,frame);")
    db.close()
    for fn,rows in [("targets",gts_all),("new_targets",[g for g in gts_all if any(g[m+"_novel_id"] for m in METHODS[1:])]),
                    ("switches",sw_all),("weak_births",births_all),("recovery_endpoints",rec_all),("official_checks",checks)]:
        csv_file(result/(fn+".csv"),rows)
    assert len(gts_all)==870 and sum(r['annotated_frames'] for r in gts_all)==179772 and len(checks)==90
    record=dict(sequences=p["sequences"],detector=p["detector_sha256"],rgb_reid=p["rgb_encoder_sha256"],
                categories_sha256=digest(co_path),input_fields_equal=keys,official_evaluator=audit['evaluator']['commit'],
                trackeval_audit=audit["status"],compared_mot_files=len(checks),initial_tracks=len(gts_all),
                gt_frames=sum(r['annotated_frames'] for r in gts_all),full_run=str(full))
    (result/"validation.json").write_text(json.dumps(record,indent=2),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir",type=Path,default=Path(__file__).resolve().parents[2]/"full30")
    parser.add_argument("--gt-root",type=Path,required=True)
    parser.add_argument("--categories",type=Path,required=True)
    parser.add_argument("result",type=Path)
    args=parser.parse_args()
    analyze(args.run_dir,args.result,args.gt_root,args.categories)
