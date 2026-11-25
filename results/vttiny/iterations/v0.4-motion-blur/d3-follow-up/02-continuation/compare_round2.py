"""Verify shared inputs, then evaluate D3 continuation ablations with pinned TrackEval."""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys
import subprocess

JOB = Path(__file__).resolve().parent
sys.path.insert(0,str(JOB.parent))
from run_round import FULL, GT, PROJECT, REPO, env, read, write

BASE = "B_early_moderate"
PINNED_INPUT_FIELDS = ("sequences","native_config","detector_sha256","rgb_encoder_sha256",
                       "source_protocol_sha256","detection_fingerprints","feature_hashes",
                       "decoded_image_hashes","encoder_contracts","boxmot_source_hashes")
METRICS = ("HOTA","DetA","AssA","IDF1","MOTA","FP","FN","IDSW")


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def table(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def assemble(entries):
    assembly=JOB/"comparison_input"
    assembly.mkdir(exist_ok=False)
    (assembly/"protocols").mkdir()
    (assembly/"trackers").mkdir()
    base=read(FULL/"observed/protocols"/(BASE+".json"))
    evidence={}
    for name,run in entries.items():
        p=run/"protocols"/(name+".json")
        data=read(p)
        assert data["status"]=="COMPLETE" and not data["subset"]
        assert data["causal"] and not data["interpolation"] and not data["backfilled_frames"]
        for key in PINNED_INPUT_FIELDS:
            assert data[key]==base[key], (name,key)
        assert data["config"]["name"]==name
        config={**data["config"],"name":BASE}
        assert config==base["config"], (name,"existing effective configuration")
        assert len(data["sequences"])==30 and sum(data["sequences"].values())==12590
        (assembly/"protocols"/p.name).symlink_to(p)
        (assembly/"trackers"/name).symlink_to(run/"trackers"/name,target_is_directory=True)
        sizes={}
        for seq,sha in data["output_prediction_hashes"].items():
            assert digest(run/"trackers"/name/"data"/(seq+".txt"))==sha
            sizes[seq]=sha == base["output_prediction_hashes"][seq]
        events_dir=run/"events"/name
        assert all((events_dir/(seq+".jsonl")).is_file() for seq in data["sequences"])
        evidence[name] = dict(root=str(run),protocol_sha256=digest(p),compare_to_baseline_hash=sizes,
                              identical_to_baseline_sequences=sum(sizes.values()))
    write(JOB/"same_input_verification.json",dict(fields=list(PINNED_INPUT_FIELDS),run_evidence=evidence))
    return assembly, base["sequences"]


def metrics(full, minus_one, names):
    combined=[]
    seqs=[]
    for label, raw in (("full30",full),("other29",minus_one)):
        base={r["sequence"]:r for r in raw if r["tracker"]==BASE}
        for row in raw:
            name,seq=row["tracker"],row["sequence"]
            result=dict(scope=label,method=name,sequence=seq,**{k:row[k] for k in METRICS},
                        **{"delta_"+k:row[k]-base[seq][k] for k in METRICS})
            if seq=="COMBINED_SEQ":combined.append(result)
            elif label=="full30":seqs.append(result)
    assert len(seqs)==30*len(names) and len(combined)==2*len(names)
    table(JOB/"combined_metrics.csv",combined)
    table(JOB/"sequence_metrics.csv",seqs)
    for r in combined:
        print(r)


def main():
    args=sys.argv[1:]
    assert len(args)==4
    entries={BASE:FULL/"observed",args[0]:Path(args[1]),args[2]:Path(args[3])}
    assert len(entries)==3
    data,sequences=assemble(entries)
    pinned=PROJECT/"external/TrackEval"
    env_eval=env()
    env_eval["TRACKEVAL_ROOT"]=str(pinned)
    subprocess.run([sys.executable,"-B","-u",str(REPO/"tools/evaluate_botsort_plus.py"),
        "--run-dir",str(data),"--trackers",*entries,"--gt-root",str(GT),"--trackeval-root",
        str(pinned),"--reference-tracker",BASE,"--output-dir",str(JOB/"official30")],
        env=env_eval,cwd=REPO,check=True)
    official=read(JOB/"official30/summary.json")["results"]
    audit=read(JOB/"official30/evaluation_audit.json")
    assert audit["status"]=="COMPLETE"
    assert audit["summary_sha256"]==digest(JOB/"official30/summary.json")
    os.environ["TRACKEVAL_ROOT"]=str(pinned)
    sys.path[:0]=[str(REPO/"tools"),str(REPO)]
    from evaluate_vttiny import evaluate
    del sequences["DJI_0327_3"]
    subset=evaluate("rgb","test",data/"trackers",list(entries),JOB/"official29",
                    seqs=sequences,gt_folder=GT)
    metrics(official,subset["results"],entries)
    print("EVALUATION_COMPLETE")


if __name__=="__main__":
    main()
