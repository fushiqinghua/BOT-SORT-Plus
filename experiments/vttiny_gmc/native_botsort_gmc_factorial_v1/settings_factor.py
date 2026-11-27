import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
INTEGRATIONS = ROOT.parent
PREVIOUS = INTEGRATIONS/"native_botsort_gmc_oru_v1"
SUITE = INTEGRATIONS/"botsort_plus_v0_4_motion_blur"
REPO = SUITE/"BOT-SORT-Plus-followup-c3e142e6"
SETTINGS = SUITE/"v04_followup_round2/P0/motion_blur_v0.4.local.json"
TRACKEVAL = INTEGRATIONS.parent/"external/TrackEval"
VENDOR = PREVIOUS/"vendor"
FOCUS = ("DJI_0109_1","DJI_0173_2")
ARMS = {"native":(False,False), "skip_only":(True,False),
        "lightglue_only":(False,True), "adaptive_gmc":(True,True)}
FIELDS = ("HOTA","DetA","AssA","IDF1","MOTA","FP","FN","IDSW")
COUNT = {"FP","FN","IDSW"}


def load(path):
    return json.loads(path.read_bytes())


def save(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".writing")
    temp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf-8")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment(method):
    settings=load(SETTINGS)
    commit=(VENDOR/"lightglue_commit.txt").read_text().strip()
    paths=[ROOT,REPO,REPO/"tools",Path(settings["runtime_root"])/"tracking_runtime",
           VENDOR/"dependencies",VENDOR/("LightGlue-"+commit)]
    return dict(os.environ, PYTHONUSERBASE=str(INTEGRATIONS/"botsort_plus_v0_1_0/python_user"),
        PYTHONPATH=os.pathsep.join(map(str,paths)), PATH=str(Path(sys.executable).parent)+":"+os.environ["PATH"],
        YOLO_CONFIG_DIR=str(ROOT/"runtime_cfg"),TORCH_HOME=str(VENDOR/"torch"),
        OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES="5" if method in ARMS and ARMS[method][1] else "",
        CUBLAS_WORKSPACE_CONFIG=":4096:8")
