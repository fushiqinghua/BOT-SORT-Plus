from pathlib import Path
import json
import os
import sys

ROOT=Path(__file__).resolve().parent
INTEGRATIONS=ROOT.parent
PROJECT=INTEGRATIONS.parent
SUITE=INTEGRATIONS/"botsort_plus_v0_4_motion_blur"
REPO=SUITE/"BOT-SORT-Plus-followup-c3e142e6"
SETTINGS=SUITE/"v04_followup_round2/P0/motion_blur_v0.4.local.json"
HISTORY=INTEGRATIONS/"botsort_plus_C2_v0_3_0/experiments/kf_measurement_round1/full/protocols/kf_original.json"
METHODS=["native","adaptive_gmc","oru"]
FOCUS=["DJI_0173_2","DJI_0277_2"]


def read(p):
    return json.loads(Path(p).read_bytes())


def save(p,obj):
    p=Path(p)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)+"\n")


def environment(method):
    settings=read(SETTINGS)
    vendor=ROOT/"vendor"
    pin=(vendor/"lightglue_commit.txt").read_text().strip()
    paths=[ROOT,REPO,REPO/"tools",Path(settings["runtime_root"])/"tracking_runtime",
           vendor/"dependencies",vendor/("LightGlue-"+pin)]
    return dict(os.environ, PYTHONUSERBASE=str(INTEGRATIONS/"botsort_plus_v0_1_0/python_user"),
        PYTHONPATH=os.pathsep.join(map(str,paths)),PATH=str(Path(sys.executable).parent)+":"+os.environ["PATH"],
        YOLO_CONFIG_DIR=str(ROOT/"runtime_cfg"),TORCH_HOME=str(vendor/"torch"),
        OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES="5" if method=="adaptive_gmc" else "",CUBLAS_WORKSPACE_CONFIG=":4096:8")
