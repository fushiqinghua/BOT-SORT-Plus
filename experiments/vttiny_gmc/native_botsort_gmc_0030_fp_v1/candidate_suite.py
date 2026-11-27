"""Validate on the focus first, then carry both low-score guard variants unchanged to all 30 sequences."""
import shutil
import sys
import time
import subprocess
from common_audit import (ROOT,FACTOR,REPO,SETTINGS,TRACK_EVAL,FIELDS,SKIP,FULL,SEQ,
                          env,read,write,sha,csvout)

CANDIDATES=("dual_low","dual_low_center")


def run(stage):
    root=ROOT/stage;root.mkdir(exist_ok=True)
    arms=(SKIP,FULL,*CANDIDATES,"pixel_margin") if stage=="focus" else (SKIP,FULL,*CANDIDATES)
    historical={m:read(FACTOR/"full/protocols"/(m+".json")) for m in (SKIP,FULL)}
    if stage=="full":
        assert read(ROOT/"focus/live_summary.json")["status"]=="COMPLETE"
    for m in arms:
        protocol=root/"protocols"/(m+".json")
        if stage=="full" and m in historical:
            for key in ("protocols","trackers"):
                origin=FACTOR/"full"/key/(m+(".json" if key=="protocols" else ""))
                dest=root/key/origin.name
                dest.parent.mkdir(exist_ok=True,parents=True)
                if not dest.exists():
                    if origin.is_dir():shutil.copytree(origin,dest)
                    else:shutil.copy2(origin,dest)
            continue
        if protocol.is_file() and read(protocol)["status"]=="COMPLETE":
            continue
        for f in [root/"trackers"/m,protocol,root/"timing"/m,root/"actions"/m]:
            if f.exists():
                dst=root/"previous_attempts"/(str(time.time_ns())+"_"+m)/f.relative_to(root)
                dst.parent.mkdir(exist_ok=True,parents=True);f.rename(dst)
        print("start",m,flush=True)
        with (root/(m+".log")).open("w") as f:
            subprocess.run([sys.executable,"-Bu",str(ROOT/"live_arm.py"),m,stage],env=env(m!=SKIP),
                           stdout=f,stderr=subprocess.STDOUT,check=True)
    for m in arms:
        p=read(root/"protocols"/(m+".json"));base=historical[FULL]
        assert p["status"]=="COMPLETE"
        for k in ("native_config","detector_sha256","rgb_encoder_sha256","encoder_contracts","boxmot_source_hashes","source_protocol_sha256"):
            assert p[k]==base[k],(m,k)
        for s,v in p["sequences"].items():
            for k in ("decoded_image_hashes","feature_hashes","detection_fingerprints","sequences"):
                assert p[k][s]==base[k][s],(m,k,s)
            if m in historical:
                assert p["output_prediction_hashes"][s]==historical[m]["output_prediction_hashes"][s]
        if stage=="focus" and m!="dual_low_center":
            assert p["output_prediction_hashes"][SEQ]==sha(ROOT/"trace/trackers"/m/"data"/(SEQ+".txt")),(m,"offline vs live")
        if stage=="full":
            assert p["output_prediction_hashes"][SEQ]==read(ROOT/"focus/protocols"/(m+".json"))["output_prediction_hashes"][SEQ]
    with (root/"trackeval.log").open("w") as log:
        subprocess.run([sys.executable,"-Bu",str(REPO/"tools/evaluate_botsort_plus.py"),"--run-dir",str(root),
            "--trackers",*arms,"--reference-tracker",FULL,"--gt-root",read(SETTINGS)["gt_root"],
            "--trackeval-root",str(TRACK_EVAL),"--output-dir",str(root/"metrics")],env=env(),
            stdout=log,stderr=subprocess.STDOUT,check=True)
    subj=read(root/"metrics/summary.json")["results"]
    lookup={(r["tracker"],r["sequence"]):r for r in subj}
    hist={(r["tracker"],r["sequence"]):r for r in read(FACTOR/"full/metrics/summary.json")["results"]}
    for (m,s),r in lookup.items():
        if m in historical and (stage=="full" or s!="COMBINED_SEQ"):
            assert all(abs(r[k]-hist[m,s][k])<1e-9 for k in FIELDS)
    rows=[dict(**v,**{"delta_vs_full_"+k:v[k]-lookup[FULL,s][k] for k in FIELDS},
                     **{"delta_vs_skip_"+k:v[k]-lookup[SKIP,s][k] for k in FIELDS}) for (m,s),v in lookup.items()]
    csvout(root/"compared.csv",rows)
    if stage=="focus":
        from clear_audit import evaluate_rows,source,compare
        ds=source()
        control=evaluate_rows(root/"trackers",FULL,ds)
        skip=evaluate_rows(root/"trackers",SKIP,ds)
        for m in CANDIDATES:
            c=evaluate_rows(root/"trackers",m,ds)
            compare(control,c,FULL,m,root/"audit"/(m+"_vs_full"))
            compare(skip,c,SKIP,m,root/"audit"/(m+"_vs_skip"))
    seqs=read(root/"protocols"/(FULL+".json"))["sequences"]
    write(root/"live_summary.json",dict(status="COMPLETE",sequences=len(seqs),frames=sum(seqs.values()),inputs_verified=True,
        controls_match_previous=True,focus_matches_frozen_warps=True,
        joint={m:lookup[m,"COMBINED_SEQ"] for m in arms},target={m:lookup[m,SEQ] for m in arms}))
    print(stage,"COMPLETE",flush=True)


if __name__=="__main__":
    run(sys.argv[1])
