"""Validate candidate-level equivalence, then calculate official metric contrasts."""
from collections import Counter
import csv
import gzip
import json
import math
from settings_factor import ROOT,PREVIOUS,ARMS,FOCUS,FIELDS,COUNT,load,save


def csv_output(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def records(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def verify_sequence(root,seq,nframes,focus_rows):
    counts=Counter(dict(frames=0,triggers=0,skip_calls=0,lightglue_calls=0,full_calls=0,
        skip_selected=0,lightglue_selected=0,full_skip=0,full_lightglue=0,full_sof=0,
        both_eligible=0,both_choose_skip=0,both_choose_lightglue=0,
        singleton_skip=0,singleton_lightglue=0,trigger_no_accepted=0))
    streams=[records(root/"warps"/m/(seq+".jsonl.gz")) for m in ARMS]
    streams.append(records(PREVIOUS/"full/audit/adaptive_gmc/warps"/(seq+".jsonl.gz")))
    for frame,items in enumerate(zip(*streams,strict=True),1):
        native,skip,lg,full,old=items
        assert all(r["frame"]==frame and r["native"]==native["native"] for r in items),(seq,frame,"original SOF")
        assert all(r["used"]==r["native"] for r in items if r["choice"]=="sof")
        assert native["choice"]=="sof" and not native["model_called"] and not skip["model_called"]
        for k in ("validation","validation_cells","native_error","trigger"):
            assert skip.get(k)==lg.get(k)==full.get(k)==old.get(k),(seq,frame,k)
        assert lg["model_called"]==full["model_called"]==full["trigger"]
        for k in ("choice","used","candidate_errors","selected_error","fit_matches","fit_cells","inliers"):
            assert full.get(k)==old.get(k),(seq,frame,"previous full",k)
        for key in ("fit_matches","fit_cells","inliers"):
            assert lg.get(key)==full.get(key),(seq,frame,key)
        assert skip["choice"] in ("sof","identity") and lg["choice"] in ("sof","lightglue")
        se=skip.get("candidate_errors",{});le=lg.get("candidate_errors",{})
        assert set(se)<= {"identity"} and set(le)<= {"lightglue"}
        assert {**se,**le}==full.get("candidate_errors",{}),(seq,frame,"candidate residuals")
        s=skip["choice"]=="identity";l=lg["choice"]=="lightglue"
        eligible=(["identity"] if s else [])+(["lightglue"] if l else [])
        assert eligible==full.get("eligible",[]),(seq,frame,"eligible")
        if eligible:
            winner=min(eligible,key=lambda c:tuple(full["candidate_errors"][c]))
            assert full["choice"]==winner
            assert full["used"]==(skip if winner=="identity" else lg)["used"]
        else:
            assert full["choice"]=="sof"
        counts.update(dict(frames=1,triggers=int(full["trigger"]),skip_calls=int(skip["model_called"]),
            lightglue_calls=int(lg["model_called"]),full_calls=int(full["model_called"]),
            skip_selected=int(s),lightglue_selected=int(l),full_skip=int(full["choice"]=="identity"),
            full_lightglue=int(full["choice"]=="lightglue"),full_sof=int(full["choice"]=="sof"),
            both_eligible=int(s and l),both_choose_skip=int(s and l and full["choice"]=="identity"),
            both_choose_lightglue=int(s and l and full["choice"]=="lightglue"),
            singleton_skip=int(s and not l),singleton_lightglue=int(l and not s),
            trigger_no_accepted=int(full["trigger"] and not eligible)))
        if seq in FOCUS:
            focus_rows.append(dict(sequence=seq,frame=frame,trigger=int(full["trigger"]),
                skip_choice=skip["choice"],lightglue_choice=lg["choice"],full_choice=full["choice"],
                native_median_px=(full.get("native_error") or [None])[0],
                skip_median_px=(se.get("identity") or [None])[0],
                lightglue_median_px=(le.get("lightglue") or [None])[0]))
    assert counts["frames"]==nframes and counts["full_sof"]+counts["full_skip"]+counts["full_lightglue"]==nframes
    return counts


def contrasts(group):
    base,skip,lg,full=[group[m] for m in ARMS]
    out={name:{} for name in ("skip_alone","lightglue_alone","both","add_skip_to_lightglue",
            "add_lightglue_to_skip","interaction","skip_order_average","lightglue_order_average")}
    for k in FIELDS:
        a,b,c,d=base[k],skip[k],lg[k],full[k]
        values=(b-a,c-a,d-a,d-c,d-b,d-b-c+a,((b-a)+(d-c))/2,((c-a)+(d-b))/2)
        for name,v in zip(out,values):
            out[name][k]=v
        assert math.isclose(values[-2]+values[-1],d-a,abs_tol=1e-9)
    return out


def summarize(root):
    spec=load(root/"protocols/native.json")["sequences"]
    out=root/"analysis"
    out.mkdir(exist_ok=True)
    focus_rows=[]; by_seq={}; total=Counter()
    for seq,n in spec.items():
        c=verify_sequence(root,seq,n,focus_rows)
        by_seq[seq]=dict(c)
        total.update(c)
    by_seq["COMBINED_SEQ"]=dict(total)
    csv_output(out/"camera_choices.csv",[dict(sequence=s,**v) for s,v in by_seq.items()])
    csv_output(out/"focus_frame_choices.csv",focus_rows)
    historic={(r["sequence"],r["tracker"]):r for r in load(PREVIOUS/"full/metrics/summary.json")["results"]}
    raw=load(root/"metrics/summary.json")["results"]
    groups={seq:{} for seq in [*spec,"COMBINED_SEQ"]}
    for r in raw:
        s,m=r["sequence"],r["tracker"]
        groups[s][m]={k:r[k] for k in FIELDS}
        if m in ("native","adaptive_gmc") and (s!="COMBINED_SEQ" or root.name=="full"):
            old=historic[s,m]
            assert all(math.isclose(r[k],old[k],abs_tol=1e-9,rel_tol=0) for k in FIELDS),(m,s,"trackeval")
    changes={s:contrasts(g) for s,g in groups.items()}
    csv_output(out/"metrics_by_sequence.csv",[
        dict(sequence=s,method=m,**g[m],**{"delta_"+k:g[m][k]-g["native"][k] for k in FIELDS})
        for s,g in groups.items() for m in ARMS])
    csv_output(out/"factorial_contrasts.csv",[dict(sequence=s,contrast=c,**vals)
        for s,cs in changes.items() for c,vals in cs.items()])
    timing={}
    for m in ARMS:
        times=[load(root/"timing"/m/(s+".json")) for s in spec]
        timing[m]=sum(t["association_and_gmc_seconds"] for t in times)
    joint=groups["COMBINED_SEQ"]
    summary=dict(status="COMPLETE",stage=root.name,inputs=load(root/"input_parity.json"),
         prior_metrics_and_frame_choices_match=True,
         joint=joint,effects=changes["COMBINED_SEQ"],choices=by_seq["COMBINED_SEQ"],replay_seconds=timing,
         focus={s:dict(metrics=groups[s],effects=changes[s],choices=by_seq[s]) for s in FOCUS})
    save(ROOT/(root.name+"_summary.json"),summary)
    print("Verified",root.name,"native and previous complete matches; frames",total["frames"],flush=True)
