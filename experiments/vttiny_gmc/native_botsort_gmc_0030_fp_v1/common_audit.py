import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
INTEGRATIONS=ROOT.parent
FACTOR=INTEGRATIONS/"native_botsort_gmc_factorial_v1"
SUITE=INTEGRATIONS/"botsort_plus_v0_4_motion_blur"
REPO=SUITE/"BOT-SORT-Plus-followup-c3e142e6"
SETTINGS=SUITE/"v04_followup_round2/P0/motion_blur_v0.4.local.json"
TRACK_EVAL=INTEGRATIONS.parent/"external/TrackEval"
SEQ="DJI_0030_2"
SKIP="skip_only"
FULL="adaptive_gmc"
FIELDS=("HOTA","DetA","AssA","IDF1","MOTA","FP","FN","IDSW")


def read(p):
    return json.loads(p.read_bytes())


def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+".writing")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf8")
    tmp.replace(p)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def csvout(p,rows,columns=None):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8-sig",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=columns or list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def csvin(p):
    with p.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def motions(method,seq):
    with gzip.open(FACTOR/"full/warps"/method/(seq+".jsonl.gz"),"rt") as f:
        return [json.loads(line) for line in f]


def env(gpu=False):
    cfg=read(SETTINGS)
    vendor=INTEGRATIONS/"native_botsort_gmc_oru_v1/vendor"
    paths=[ROOT,FACTOR,REPO,REPO/"tools",Path(cfg["runtime_root"])/"tracking_runtime",TRACK_EVAL,
           vendor/"dependencies",vendor/("LightGlue-"+(vendor/"lightglue_commit.txt").read_text().strip())]
    return dict(os.environ,PYTHONUSERBASE=str(INTEGRATIONS/"botsort_plus_v0_1_0/python_user"),
        PYTHONPATH=os.pathsep.join(map(str,paths)),PATH=str(Path(sys.executable).parent)+":"+os.environ["PATH"],
        YOLO_CONFIG_DIR=str(ROOT/"runtime_cfg"),TRACKEVAL_ROOT=str(TRACK_EVAL),CUDA_VISIBLE_DEVICES="5" if gpu else "",
        CUDA_DEVICE_ORDER="PCI_BUS_ID",CUBLAS_WORKSPACE_CONFIG=":4096:8",TORCH_HOME=str(vendor/"torch"),
        PYTHONDONTWRITEBYTECODE="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1")
