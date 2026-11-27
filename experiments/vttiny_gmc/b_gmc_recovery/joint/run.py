"""Four-arm frozen-input evaluation. Start in a fresh directory containing repo/."""
from pathlib import Path
import json, os, subprocess, sys, shutil

root = Path(__file__).resolve().parent
repo = root / "repo"
integration = root.parent
suite = integration / "botsort_plus_v0_4_motion_blur"
original = suite / "BOT-SORT-Plus-followup-c3e142e6"
settings = suite / "v04_followup_round2/P0/motion_blur_v0.4.local.json"
cfg = json.loads(settings.read_bytes())
vendor = integration / "native_botsort_gmc_oru_v1/vendor"
full = root / "full"
b = "B_early_moderate"
combo = b + "_dual_low"
names = ["native", b, "dual_low", combo]
env = dict(os.environ, PYTHONUSERBASE=str(integration/"botsort_plus_v0_1_0/python_user"),
           PYTHONPATH=os.pathsep.join(map(str,[repo,repo/"tools",Path(cfg["runtime_root"])/"tracking_runtime"])),
           CUDA_VISIBLE_DEVICES="5", CUDA_DEVICE_ORDER="PCI_BUS_ID", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
           OMP_NUM_THREADS="1", YOLO_CONFIG_DIR=str(root/"runtime_cfg"),PYTHONDONTWRITEBYTECODE="1",
           PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ["PATH"])
def run(label, script, args):
    print(label,flush=True)
    with (root/(label+".log")).open("w") as f:
        subprocess.run([sys.executable,"-Bu",str(script),*map(str,args)],env=env,cwd=repo,stdout=f,stderr=subprocess.STDOUT,check=True)
def reuse(method, src):
    for sub, file in (("trackers",method),("protocols",method+".json")):
        dest = full/sub/file
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.symlink_to(src/sub/file)

run("combo",repo/"tools/run_vttiny_gmc.py",["--settings",settings,"--vendor",vendor,"--output-dir",full])
run("B",repo/"tools/run_botsort_plus.py",["--config",repo/"configs/botsort_plus/B_early_moderate.json",
    "--baseline-root",cfg["baseline_root"],"--features-dir",cfg["features_dir"],"--runtime-root",cfg["runtime_root"],
    "--images-root",cfg["images_root"],"--output-dir",full])
reuse("native",integration/"native_botsort_gmc_factorial_v1/full")
reuse("dual_low",integration/"native_botsort_gmc_0030_fp_v1/full")
protocols=[json.loads((full/"protocols"/(m+".json")).read_bytes()) for m in names]
for p in protocols:
    for k in ("native_config","sequences","detector_sha256","rgb_encoder_sha256","feature_hashes","decoded_image_hashes","detection_fingerprints"):
        assert p[k]==protocols[0][k], k
assert protocols[1]["config"]["recovery"]==protocols[3]["config"]["recovery"]
assert protocols[1]["output_prediction_hashes"]==json.loads((repo/"B_reference.json").read_bytes())["prediction_sha256"]
run("eval",original/"tools/evaluate_botsort_plus.py",["--run-dir",full,"--trackers",*names,"--reference-tracker","native",
    "--gt-root",cfg["gt_root"],"--trackeval-root",integration.parent/"external/TrackEval","--output-dir",full/"metrics"])
rows=json.loads((full/"metrics/summary.json").read_bytes())["results"]
actions=[json.loads(f.read_bytes()) for f in (full/"actions/dual_low").glob("*.json")]
assert len(actions)==30
stats={k:sum(a[k] for a in actions) for k in actions[0]}
assert stats["blocked_native_edges"]>0 and stats["center_outputs"]==0
report=dict(metrics=[r for r in rows if r["sequence"]=="COMBINED_SEQ"],actions=stats,B_reference_parity=30,
    sequences=30,frames=sum(protocols[0]["sequences"].values()))
(root/"joint_result.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
