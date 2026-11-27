"""Evaluate the two locked policies using live GMC, reusing the verified four control runs."""
import json,os,shutil,subprocess,sys
from pathlib import Path
from common import ROOT,REPO,JOINT,INT,SETTINGS,EVAL,B,COMBO,OUT,SEQS,read,save

deploy=ROOT/"live_repo"
if not deploy.exists():
    shutil.copytree(REPO,deploy)
case=deploy/"experiments/vttiny_gmc/native_botsort_gmc_0030_fp_v1"
shutil.copy2(ROOT/"trial_live.py",case/"live_arm.py")
shutil.copy2(ROOT/"lost_camera_margin.py",case/"lost_camera_margin.py")
settings=read(SETTINGS)
vendor=INT/"native_botsort_gmc_oru_v1/vendor"
original=INT/"botsort_plus_v0_4_motion_blur/BOT-SORT-Plus-followup-c3e142e6"
env=dict(os.environ,PYTHONUSERBASE=str(INT/"botsort_plus_v0_1_0/python_user"),CUDA_VISIBLE_DEVICES="5",
         PYTHONPATH=os.pathsep.join(map(str,[deploy,deploy/"tools",Path(settings["runtime_root"])/"tracking_runtime"])),
         OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",
         PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ["PATH"],
         PYTHONDONTWRITEBYTECODE="1",YOLO_CONFIG_DIR=str(ROOT/"runtime_cfg"))
joint=ROOT/"full_evaluation"
controls=["native","dual_low",B,COMBO]
names=["margin_both","evidence_both"]
reference=read(JOINT/"full/protocols"/(COMBO+".json"))
def attach(name,src):
    for sub,file in (("trackers",name),("protocols",name+".json")):
        dest=joint/sub/file
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): dest.symlink_to(src/sub/file)
for name in controls: attach(name,JOINT/"full")
for name in names:
    work=ROOT/"full_runs"/name
    print("running",name,flush=True)
    if not (work/"protocols"/(name+".json")).exists():
        env["BG_LOST_POLICY"]=name
        subprocess.run([sys.executable,"-Bu",str(deploy/"tools/run_vttiny_gmc.py"),"--settings",str(SETTINGS),
                        "--vendor",str(vendor),"--output-dir",str(work)],env=env,check=True)
    p=read(work/"protocols"/(name+".json"))
    assert p["status"]=="COMPLETE" and p["lost_camera_policy"]["variant"]==name
    for k in ("native_config","sequences","feature_hashes","decoded_image_hashes","detection_fingerprints","detector_sha256","rgb_encoder_sha256","boxmot_source_hashes"):
        assert p[k]==reference[k],(name,k)
    for s in SEQS:
        stage="evidence" if name=="evidence_both" else "trial"
        h=next(x["hash"] for x in read(OUT/s/(stage+"_replays.json"))["runs"] if x["name"]==name)
        assert h==p["output_prediction_hashes"][s],(name,s,"live/frozen-camera mismatch")
    attach(name,work)
subprocess.run([sys.executable,"-Bu",str(original/"tools/evaluate_botsort_plus.py"),"--run-dir",str(joint),
                "--trackers",*controls,*names,"--reference-tracker",COMBO,"--gt-root",settings["gt_root"],
                "--trackeval-root",str(EVAL),"--output-dir",str(OUT/"full")],env=env,check=True)
base=read(JOINT/"full/metrics/summary.json")["results"]
rows=read(OUT/"full/summary.json")["results"]
assert len(rows)==186
for r in base:
    current=next(v for v in rows if v["tracker"]==r["tracker"] and v["sequence"]==r["sequence"])
    assert all(abs(current[k]-r[k])<1e-9 for k in ("FP","FN","IDSW","HOTA","IDF1","MOTA","DetA","AssA"))
save(OUT/"full/parity.json",dict(sequences=30,frames=sum(reference["sequences"].values()),controls=controls,
    cached_inputs_verified=True,focus_mot_byte_parity={n:SEQS for n in names}))
print("COMPLETE live",flush=True)
