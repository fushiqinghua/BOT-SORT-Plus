"""Run frozen-smoke parity then an isolated 30-sequence weak continuation replay."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

JOB = Path(__file__).resolve().parent
SUITE = JOB.parents[1]
REPO = SUITE / "BOT-SORT-Plus"
PROJECT = SUITE.parents[1]
V01 = SUITE.with_name("botsort_plus_v0_1_0")
RUNTIME = PROJECT.parent / "M3OT_MOT_Baseline_v0_1"
BASE = REPO / "outputs/diagnostics_v0.4_full30_round1/observed"
CODE = JOB / "overlay"
BASELINE = "B_early_moderate"
VARIANT = "B_postlow_v1"
SMOKE = ["DJI_0051_2", "DJI_0327_3"]


def read(path):
    return json.loads(path.read_bytes())


def put(path, document):
    path.write_text(json.dumps(document,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def env():
    return {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
            "PYTHONUSERBASE": str(V01/"python_user"), "TRACKING_RUNTIME_ROOT": str(RUNTIME),
            "YOLO_CONFIG_DIR": str(JOB/"yolo_settings")}


def launch(name, output, select=None):
    command = [sys.executable,"-B","-X","utf8",str(CODE/"tools/run_botsort_plus.py"),
               "--baseline-root",str(REPO/"results/vttiny/yolo11m_rgb_rgbt/rgbt"),
               "--features-dir",str(V01/"features_rgb"),
               "--images-root",str(PROJECT/"data/raw/VT-Tiny-MOT"),
               "--config",str(JOB/(name+".json")),"--runtime-root",str(RUNTIME),
               "--output-dir",str(output),"--trace"]
    if select:
        command += ["--sequences",*select]
    with (JOB/(output.name+".log")).open("x",encoding="utf-8") as log:
        subprocess.run(command,cwd=CODE,env=env(),stdout=log,stderr=subprocess.STDOUT,check=True)
    variant=read(output/"protocols"/(name+".json"))
    frozen=read(BASE/"protocols"/(BASELINE+".json"))
    assert variant["status"]=="COMPLETE"
    for k in ("feature_hashes","decoded_image_hashes","detection_fingerprints","sequences"):
        a,b=variant[k],frozen[k]
        assert (a==b if select is None else a=={seq:b[seq] for seq in select}), k
    for k in ("native_config","detector_sha256","rgb_encoder_sha256","source_protocol_sha256"):
        assert variant[k]==frozen[k], k
    return variant, frozen


def weak_checks(directory, sequence_names):
    counts={}
    for seq in sequence_names:
        path=directory/"events"/VARIANT/(seq+".jsonl")
        frames={}
        total=0
        with path.open("r",encoding="utf-8") as file:
            for line in file:
                event=json.loads(line)
                if event.get("event")=="recovered":
                    frames[int(event["track_id"])]=int(event["frame"])
                if event.get("stage")!="post_B_weak":
                    continue
                recovery=int(event["b_recovery_frame"])
                assert frames[event["track_id"]]==recovery
                assert 1<=int(event["frame"])-recovery<=3
                assert .05<=event["score"]<.1 and event["cost"]<=.5
                assert event["weak_updates_in_window"]<=2
                total+=1
        counts[seq]=total
    return counts


def main():
    CODE.mkdir()
    for name in ("tools","botsort_plus"):
        shutil.copytree(REPO/name,CODE/name,ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(JOB/"short_weak.py",CODE/"botsort_plus/short_weak.py")
    adapter=CODE/"botsort_plus/adapter_v2.py"
    source=adapter.read_text(encoding="utf-8")
    mark="    tracker._update_track_states(removed)\n"
    assert source.count(mark)==1
    adapter.write_text(source.replace(mark,
        '    if plus.name == "B_postlow_v1":\n'
        '        from .short_weak import post_recovery_weak\n'
        '        lost = post_recovery_weak(tracker, frame, all_dets, pool, lost, activated, refound)\n'
        +mark),encoding="utf-8")
    cfg=read(REPO/"configs/botsort_plus"/(BASELINE+".json"))
    put(JOB/(BASELINE+".json"),cfg)
    put(JOB/(VARIANT+".json"),{**cfg,"name":VARIANT})
    put(JOB/"policy.json",{
        "experiment":VARIANT,"score_low_inclusive":.05,"score_high_exclusive":.10,
        "latest_original_recovery_max_age":3,"max_weak_per_recovery_window":2,
        "pred_iou_min":.50,"row_and_track_margin":.10,"occupied_iou_reject":.50,
        "applied_after":"original high/low/unconfirmed, B recovery and new high birth",
        "inputs":"current predicted boxes, current detections, prior B event ages",
        "contract":"past-only, already active at frame entry, no new ids, no appearance writes",
        "code_sha256":hashlib.sha256((JOB/"short_weak.py").read_bytes()).hexdigest()})
    print("STAGE disabled smoke",flush=True)
    off,frozen=launch(BASELINE,JOB/"disabled_smoke",SMOKE)
    for seq in SMOKE:
        assert off["output_prediction_hashes"][seq]==frozen["output_prediction_hashes"][seq], seq
        def events(root):
            return (root/"events"/BASELINE/(seq+".jsonl")).read_bytes()
        assert events(JOB/"disabled_smoke")==events(BASE), seq
    put(JOB/"disabled_parity.json",{"status":"PASS","sequences":SMOKE,"byte_equal_mot_and_events":True})
    print("STAGE enabled smoke",flush=True)
    launch(VARIANT,JOB/"enabled_smoke",SMOKE)
    put(JOB/"enabled_smoke_checks.json",weak_checks(JOB/"enabled_smoke",SMOKE))
    print("STAGE full",flush=True)
    variant,_=launch(VARIANT,JOB/"variant_full")
    counts=weak_checks(JOB/"variant_full",variant["sequences"])
    assert len(counts)==30 and sum(variant["sequences"].values())==12590
    put(JOB/"full_checks.json",{"status":"PASS","sequences":len(counts),"frames":12590,
                                "counts":counts,"accepted_total":sum(counts.values())})
    print("FULL_COMPLETE",sum(counts.values()),flush=True)


if __name__=="__main__":
    main()
