"""Run v0.4's documented preflight, smoke and full diagnostic suite in order."""
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

JOB = Path(__file__).resolve().parent
INTEGRATION = JOB.parent
REPO = INTEGRATION / "BOT-SORT-Plus"
PROJECT = INTEGRATION.parents[1]
V01 = INTEGRATION.with_name("botsort_plus_v0_1_0")
RUNTIME = PROJECT.parent / "M3OT_MOT_Baseline_v0_1"
RAW = PROJECT / "data/raw/VT-Tiny-MOT"
GT = PROJECT / "data/official_gt/GT/GT_00/test"
SETTINGS = REPO / "configs/diagnostics/motion_blur_v0.4.local.json"
ENTRY = REPO / "tools/run_botsort_diagnostics.py"
SMOKE = REPO / "outputs/diagnostics_v0.4_smoke_DJI_0030_2"
FULL = REPO / "outputs/diagnostics_v0.4_full30_round1"


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    pending = path.with_suffix(path.suffix + ".part")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def env():
    return {
        **os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "PYTHONUSERBASE": str(V01/"python_user"),
        "TRACKING_RUNTIME_ROOT": str(RUNTIME), "YOLO_CONFIG_DIR": str(JOB/"yolo_settings"),
    }


def invoke(stage, subcommand, *args):
    command = [sys.executable, "-X", "utf8", "-u", str(ENTRY), subcommand, "--settings", str(SETTINGS), *map(str, args)]
    with (JOB/(stage+".log")).open("x", encoding="utf-8") as log:
        subprocess.run(command, cwd=REPO, env=env(), stdout=log, stderr=subprocess.STDOUT, check=True)


def verify_complete(root, minimum):
    summary = read(root / "experiment.json")
    parity = read(root / "verification.json")
    assert summary["status"] == "COMPLETE", summary["status"]
    assert parity["status"] == "PASS" and len(parity["sequences"]) == minimum
    assert all(r["byte_identical"] and r["original_events_byte_identical"] for r in parity["sequences"].values())
    assert summary["stages"]["D1_D2_D3"] == "COMPLETE"
    assert summary["stages"]["D4_quality_proxies"] == "COMPLETE"
    assert (root.parent/(root.name+"_evidence.zip")).is_file()
    return {"sequences": minimum, "frames": sum(r["frames"] for r in parity["sequences"].values()),
            "analysis_complete": True, "manual_review": summary["stages"]["D4_natural_blur_labels"]}


def main():
    status = JOB/"progress.json"
    assert not status.exists() and not SETTINGS.exists() and not FULL.exists() and not SMOKE.exists()
    write(SETTINGS, {
        "schema": "botsort_diagnostics_settings_v1",
        "baseline_root": "results/vttiny/yolo11m_rgb_rgbt/rgbt",
        "features_dir": str(V01/"features_rgb"),
        "images_root": str(RAW), "gt_root": str(GT),
        "output_root": str(FULL), "runtime_root": str(RUNTIME), "sequences": None, "max_review": 120,
    })
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
    record = {"status":"RUNNING", "commit":commit, "stage":"preflight", "stages":{},
              "settings":str(SETTINGS), "smoke_root":str(SMOKE), "full_root":str(FULL)}
    write(status, record)
    try:
        invoke("preflight", "preflight")
        text = (JOB/"preflight.log").read_text(encoding="utf-8")
        readiness = json.loads(text[text.rfind("\n{")+1:])
        inputs = readiness["prediction_preflight"]
        assert readiness["status"] == "READY" and len(inputs["sequences"]) == 30 and not inputs["subset"]
        write(JOB/"preflight.json",readiness)
        record["stages"]["preflight"] = {"sequences":30, "frames":sum(inputs["sequences"].values()),"status":"READY"}
        record["stage"] = "smoke"
        write(status,record)
        invoke("smoke", "run", "--sequences", "DJI_0030_2", "--output-root", SMOKE)
        record["stages"]["smoke"] = verify_complete(SMOKE, 1)
        record["stage"] = "full"
        write(status,record)
        invoke("full", "run")
        record["stages"]["full"] = verify_complete(FULL, 30)
        record.update(status="COMPLETE", stage="analysis")
    except Exception:
        record.update(status="FAILED", error=traceback.format_exc())
        raise
    finally:
        write(status, record)


if __name__ == "__main__":
    main()
