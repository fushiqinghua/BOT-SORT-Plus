"""Frozen tracker and evaluation dependencies for the VT-Tiny-MOT 2 x 2 replay."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('VT_COMBO_ROOT', SOURCE)).resolve()
INTEGRATIONS = Path(os.environ.get('VT_INTEGRATIONS', ROOT.parent)).resolve()
MOTION = INTEGRATIONS / 'native_motion_345_v1'
FORENSICS = INTEGRATIONS / 'b_gmc_forensics_r1'
FIX_REPO = FORENSICS / 'live_repo'
SUITE = INTEGRATIONS / 'botsort_plus_v0_4_motion_blur'
BASE_REPO = SUITE / 'BOT-SORT-Plus-followup-c3e142e6'
CONFIG = SUITE / 'v04_followup_round2/P0/motion_blur_v0.4.local.json'
VENDOR = INTEGRATIONS / 'native_botsort_gmc_oru_v1/vendor'
EVALUATOR = INTEGRATIONS.parent / 'external/TrackEval'
OUT = ROOT / 'results'
NATIVE = 'native'
OVERLAP = 'bam_overlap'
FIX = 'evidence_both'
COMBINED = 'evidence_both_bam_overlap'
METHODS = (NATIVE, OVERLAP, FIX, COMBINED)
MEASURES = ('HOTA','DetA','AssA','IDF1','MOTA','FP','FN','IDSW')
COUNTS = ('FP','FN','IDSW')


def read(path):
    return json.loads(path.read_bytes())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.part')
    temporary.write_bytes((json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode())
    temporary.replace(path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def run_dir(method):
    return ROOT/'runs'/method/'full'


def history(method):
    if method in (NATIVE,OVERLAP):
        return MOTION/'full'
    return FORENSICS/'full_runs'/FIX


def protocol(directory, method):
    return read(directory/'protocols'/(method+'.json'))


def environment(method, gpu):
    repo=FIX_REPO if method in (FIX,COMBINED) else BASE_REPO
    paths=(SOURCE,repo,repo/'tools',MOTION,Path(read(CONFIG)['runtime_root'])/'tracking_runtime',EVALUATOR)
    return dict(os.environ, PYTHONPATH=os.pathsep.join(map(str,paths)), VT_INTEGRATIONS=str(INTEGRATIONS),
                VT345_ROOT=str(run_dir(method).parent), BG_LOST_POLICY=FIX,
                PYTHONUSERBASE=str(INTEGRATIONS/'botsort_plus_v0_1_0/python_user'),
                CUDA_VISIBLE_DEVICES=str(gpu) if method in (FIX,COMBINED) else '',
                PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''),
                OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
                PYTHONDONTWRITEBYTECODE='1',YOLO_CONFIG_DIR=str(ROOT/'runtime_config'/method),
                TRACKEVAL_ROOT=str(EVALUATOR))
