"""Run four independent native BoT-SORT replays and score their outputs in separate processes."""
import subprocess
import sys
import time
from settings_factor import (ROOT,PREVIOUS,REPO,VENDOR,SETTINGS,TRACKEVAL,ARMS,FOCUS,
                             load,save,sha,environment)


def start_arm(root,method):
    protocol=root/"protocols"/(method+".json")
    if protocol.exists() and load(protocol)["status"]=="COMPLETE":
        return
    partial=[root/"trackers"/method,root/"timing"/method,root/"warps"/method,protocol,root/(method+".log")]
    backup=root/"failed_attempts"/(method+"-"+str(time.time_ns()))
    for path in partial:
        if path.exists():
            dest=backup/path.relative_to(root)
            dest.parent.mkdir(parents=True,exist_ok=True)
            path.rename(dest)
    print("start",method,flush=True)
    with (root/(method+".log")).open("w") as log:
        subprocess.run([sys.executable,"-Bu",str(ROOT/"replay_factor.py"),method,root.name],
                       stdout=log,stderr=subprocess.STDOUT,env=environment(method),check=True)
    print("finished",method,flush=True)


def verify_inputs(root):
    runs={m:load(root/"protocols"/(m+".json")) for m in ARMS}
    expected={m:load(PREVIOUS/"full/protocols"/(m+".json")) for m in ("native","adaptive_gmc")}
    base=expected["native"]
    seqs=(list(FOCUS) if root.name=="smoke" else list(base["sequences"]))
    for method,p in runs.items():
        assert p["status"]=="COMPLETE" and set(p["sequences"])==set(seqs)
        assert p["subset"]==(root.name=="smoke")
        for field in ("native_config","detector_sha256","rgb_encoder_sha256","encoder_contracts",
                      "source_protocol_sha256","boxmot_source_hashes","implementation_sha256"):
            assert p[field]==base[field],(method,field)
        for seq in seqs:
            for field in ("decoded_image_hashes","feature_hashes","detection_fingerprints","sequences"):
                assert p[field][seq]==base[field][seq],(method,seq,field)
            if method in expected:
                assert p["output_prediction_hashes"][seq]==expected[method]["output_prediction_hashes"][seq],(method,seq)
        if root.name=="full":
            smoke=load(ROOT/"smoke/protocols"/(method+".json"))
            for seq in FOCUS:
                assert p["output_prediction_hashes"][seq]==smoke["output_prediction_hashes"][seq],(method,seq,"smoke")
        assert p["gmc_factorial"]["kalman"]=="original_boxmot"
    return dict(sequences=len(seqs),frames=sum(base["sequences"][s] for s in seqs),
                cache_image_and_config_exact=True,previous_native_and_full_predictions_exact=True,
                focus_outputs_equal_smoke=root.name=="full",
                detector=base["detector_sha256"],reid=base["rgb_encoder_sha256"])


def main(stage):
    if stage=="analyze":
        from bundle import package
        package()
        return
    assert stage in ("smoke","full")
    if stage=="full":
        assert load(ROOT/"smoke_summary.json")["status"]=="COMPLETE"
    root=ROOT/stage
    root.mkdir(exist_ok=True)
    weights=load(VENDOR/"provenance.json")["weights"]
    assert all(sha(VENDOR/"torch/hub/checkpoints"/k)==v for k,v in weights.items())
    for method in ARMS:
        start_arm(root,method)
    save(root/"input_parity.json",verify_inputs(root))
    with (root/"trackeval.log").open("w") as log:
        subprocess.run([sys.executable,"-Bu",str(REPO/"tools/evaluate_botsort_plus.py"),
                        "--run-dir",str(root),"--trackers",*ARMS,"--reference-tracker","native",
                        "--gt-root",load(SETTINGS)["gt_root"],"--trackeval-root",str(TRACKEVAL),
                        "--output-dir",str(root/"metrics")],stdout=log,stderr=subprocess.STDOUT,
                       env=environment("evaluation"),check=True)
    from analyze_factor import summarize
    summarize(root)
    print(stage,"COMPLETE",flush=True)


if __name__=="__main__":
    main(sys.argv[1])
