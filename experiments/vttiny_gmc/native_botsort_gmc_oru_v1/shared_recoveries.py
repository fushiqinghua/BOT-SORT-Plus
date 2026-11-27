"""Pair both algorithms at the same physical detections and verified GT endpoints."""
import csv
from paths import METHODS,read,save

HORIZONS=(3,5,10)


def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def write_shared(root):
    out=root/"diagnostic_results"
    events=rows(out/"recovery_events.csv")
    by_method={m:{} for m in ("native","oru")}
    for r in events:
        if r["status"]!="same_gt":
            continue
        key=(r["sequence"],r["anchor_gt"],r["old_frame"],r["frame"],r["old_det"],r["det"])
        lookup=by_method[r["method"]]
        assert key not in lookup
        lookup[key]=r["tid"]
    keys=sorted(set(by_method["native"]) & set(by_method["oru"]))
    own={(r["method"],r["sequence"],r["gt"],r["anchor"],r["frame"],r["track"],int(r["horizon"])):r["outcome"]
         for r in rows(out/"own_windows.csv")}
    selected=[]
    for key in keys:
        seq,gt,old,frame,old_det,det=key
        for n in HORIZONS:
            status={m:own.get((m,seq,gt,old,frame,by_method[m][key],n)) for m in by_method}
            if not status["native"]:
                assert not status["oru"]
                continue
            assert status["oru"]
            selected.append(dict(sequence=seq,gt=gt,anchor=old,frame=frame,old_det=old_det,det=det,
                       horizon=n,native_id=by_method["native"][key],oru_id=by_method["oru"][key],**status))
    with (out/"same_detection_recovery_windows.csv").open("w",encoding="utf-8-sig",newline="") as f:
        fields=["sequence","gt","anchor","frame","old_det","det","horizon","native_id","oru_id","native","oru"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(selected)
    tally={}
    for n in HORIZONS:
        s=[r for r in selected if r["horizon"]==n]
        tally[str(n)]=dict(eligible=len(s),native=sum(r["native"]=="held" for r in s),oru=sum(r["oru"]=="held" for r in s),
            gained=sum(r["native"]!="held" and r["oru"]=="held" for r in s),
            lost=sum(r["native"]=="held" and r["oru"]!="held" for r in s))
    summary_path=root.parent/(root.name+"_summary.json")
    obj=read(summary_path)
    obj["shared_endpoints"]=dict(events=len(keys),windows=tally)
    obj["replay_seconds"]={m:sum(read(p)["association_and_gmc_seconds"] for p in (root/"timing"/m).glob("*.json")) for m in METHODS}
    save(summary_path,obj)
    return obj
