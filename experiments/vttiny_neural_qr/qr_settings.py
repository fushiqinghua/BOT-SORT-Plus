"""Shared archive locations, frozen configuration and a deterministic CPU environment."""
from pathlib import Path
import csv
import hashlib
import json
import os

SRC = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('VT_QR_ROOT', SRC)).resolve()
INT = Path(os.environ.get('VT_INTEGRATIONS', ROOT.parent))
PROJECT = INT.parent
SUITE = INT/'botsort_plus_v0_4_motion_blur'
ORIG = SUITE/'BOT-SORT-Plus-followup-c3e142e6'
SETTINGS = SUITE/'v04_followup_round2/P0/motion_blur_v0.4.local.json'
MOTION = INT/'native_motion_345_v1'
KF = INT/'botsort_plus_C2_v0_3_0/experiments/kf_measurement_round1'
TRAIN = KF.parent/'ir_motion_localization_v1'
EVAL = PROJECT/'external/TrackEval'
OUT = ROOT/'results'

def read(path):
    return json.loads(Path(path).read_bytes())

def write(path, obj):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)

def table(path, rows):
    path.parent.mkdir(exist_ok=True, parents=True)
    rows = list(rows)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def environment():
    runtime = Path(read(SETTINGS)['runtime_root'])/'tracking_runtime'
    paths = (SRC, ORIG, ORIG/'tools',runtime,EVAL,MOTION)
    return dict(os.environ, PYTHONPATH=os.pathsep.join(map(str,paths)),
                PYTHONUSERBASE=str(INT/'botsort_plus_v0_1_0/python_user'), CUDA_VISIBLE_DEVICES='',
                OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                PYTHONDONTWRITEBYTECODE='1', YOLO_CONFIG_DIR=str(ROOT/'runtime-config'), TRACKEVAL_ROOT=str(EVAL))
