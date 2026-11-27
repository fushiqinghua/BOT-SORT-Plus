import os, subprocess, sys
from common import ROOT, INT, REPO, PRIOR, FACTOR, SETTINGS, EVAL, read, SEQS

runtime = read(SETTINGS)["runtime_root"]
env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str,[ROOT,REPO,REPO/"tools",PRIOR,FACTOR,runtime+"/tracking_runtime",EVAL])),
           PYTHONUSERBASE=str(INT/"botsort_plus_v0_1_0/python_user"), CUDA_VISIBLE_DEVICES="",
           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
           TRACKEVAL_ROOT=str(EVAL), YOLO_CONFIG_DIR=str(ROOT/"runtime_cfg"), PYTHONDONTWRITEBYTECODE="1")
stage = sys.argv[1]
if stage=="panels":
    subprocess.run([sys.executable,"-Bu",str(ROOT/"panels.py")],env=env,check=True)
    sys.exit(0)
if stage=="live":
    subprocess.run([sys.executable,"-Bu",str(ROOT/"full_live.py")],env=env,check=True)
    sys.exit(0)
sequences = SEQS
if stage == "full":
    sequences = sorted(read(INT/"b_gmc_joint_r1/full/protocols/B_early_moderate.json")["sequences"])
for script in (["initial","replay","score"] if stage=="initial" else ["replay","score"]):
    for seq in sequences:
        args = [sys.executable,"-Bu",str(ROOT/(script+".py")),seq,stage]
        print(script,seq,stage,flush=True)
        subprocess.run(args,env=env,check=True)
print("COMPLETE",stage,flush=True)
