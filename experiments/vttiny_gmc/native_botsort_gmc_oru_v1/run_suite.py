import json
from pathlib import Path
import subprocess
import sys
import time
from paths import ROOT,REPO,PROJECT,SETTINGS,HISTORY,METHODS,FOCUS,read,save,environment


def validate(stage):
    root=ROOT/stage
    history=read(HISTORY)
    runs={m:read(root/"protocols"/(m+".json")) for m in METHODS}
    ref=runs["native"]
    for method,p in runs.items():
        assert p["status"]=="COMPLETE"
        for field in ("decoded_image_hashes","native_config","feature_hashes","detection_fingerprints",
                      "detector_sha256","rgb_encoder_sha256","encoder_contracts","boxmot_source_hashes"):
            assert p[field]==ref[field],(method,field)
        if stage=="full":
            smoke=read(ROOT/"smoke/protocols"/(method+".json"))
            for seq in FOCUS:
                assert smoke["output_prediction_hashes"][seq]==p["output_prediction_hashes"][seq],(method,seq)
    for field in ("native_config","detector_sha256","rgb_encoder_sha256","encoder_contracts"):
        assert ref[field]==history[field],field
    for seq in ref["sequences"]:
        for field in ("output_prediction_hashes","feature_hashes","detection_fingerprints","sequences"):
            assert ref[field][seq]==history[field][seq],(seq,field)
    audit=dict(sequences=len(ref["sequences"]),frames=sum(ref["sequences"].values()),
               identical_to_historical_native=list(ref["sequences"]),same_cache_and_decoded_images=True,
               smoke_full_output_identity=stage=="full")
    save(root/"parity.json",audit)
    return audit


def run(stage):
    root=ROOT/stage
    root.mkdir(exist_ok=True)
    if stage=="full":
        assert (ROOT/"smoke_summary.json").is_file()
    handles=[]
    for method in METHODS:
        protocol=root/"protocols"/(method+".json")
        if protocol.exists() and read(protocol).get("status")=="COMPLETE":
            continue
        leftovers=[root/"trackers"/method,root/"timing"/method,root/"events"/method,
                   root/"protocols"/(method+".json"),root/"audit"/method,root/"warps"/method,
                   root/"recoveries"/method,root/"prestate"/method,root/(method+".log")]
        if any(p.exists() for p in leftovers):
            bak=root/"failed_attempts"/(method+"-"+str(time.time_ns()))
            for p in leftovers:
                if p.exists():
                    dest=bak/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);p.rename(dest)
        log=(root/(method+".log")).open("w")
        handle=subprocess.Popen([sys.executable,"-Bu",str(ROOT/"replay_arm.py"),method,stage],
                                env=environment(method),stdout=log,stderr=subprocess.STDOUT)
        handles.append((method,handle,log))
    codes={}
    for name,proc,log in handles:
        codes[name]=proc.wait();log.close()
    if any(codes.values()):
        raise RuntimeError(codes)
    print("parity",validate(stage),flush=True)
    args=["--run-dir",str(root),"--trackers",*METHODS,"--reference-tracker","native",
          "--gt-root",read(SETTINGS)["gt_root"],"--trackeval-root",str(PROJECT/"external/TrackEval"),
          "--output-dir",str(root/"metrics")]
    with (root/"trackeval.log").open("w") as log:
        subprocess.run([sys.executable,"-Bu",str(REPO/"tools/evaluate_botsort_plus.py"),*args],
                       stdout=log,stderr=subprocess.STDOUT,env=environment("evaluation"),check=True)
    subprocess.run([sys.executable,"-Bu",str(ROOT/"outcomes.py"),stage],env=environment("evaluation"),check=True)
    print(stage,"COMPLETE",flush=True)


if __name__=="__main__":
    stage=sys.argv[1]
    if stage in ("smoke","full"):
        run(stage)
    elif stage=="analyze":
        subprocess.run([sys.executable,"-Bu",str(ROOT/"package_results.py")],check=True)
