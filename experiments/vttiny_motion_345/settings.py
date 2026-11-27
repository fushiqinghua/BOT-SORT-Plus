"""Layout of the frozen VT-Tiny-MOT research suite. VT345_ROOT may point to a separate run directory."""
import csv
import hashlib
import json
import os
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('VT345_ROOT', SRC)).resolve()
INT = Path(os.environ.get('VT_INTEGRATIONS', ROOT.parent)).resolve()
PROJECT = INT.parent
SUITE = INT/'botsort_plus_v0_4_motion_blur'
REPO = SUITE/'BOT-SORT-Plus-followup-c3e142e6'
SETTINGS = SUITE/'v04_followup_round2/P0/motion_blur_v0.4.local.json'
KF = INT/'botsort_plus_C2_v0_3_0/experiments/kf_measurement_round1'
TRAIN = KF.parent/'ir_motion_localization_v1'
HISTORY = INT/'native_botsort_gmc_factorial_v1/full/protocols/native.json'
EVAL = PROJECT/'external/TrackEval'
RESULT = ROOT/'results'
PILOT = ('DJI_0173_2','DJI_0277_2','DJI_0169_3')
CAMERA = ('affine_forward','homography_forward','homography_bidir')
BAM = ('bam_iou','bam_overlap','bam_bottom')
QARMS = ('q_constant','q_motion')


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, obj):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix+'.part')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def table(path, rows):
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def environment():
    runtime = Path(read(SETTINGS)['runtime_root'])/'tracking_runtime'
    return dict(os.environ, PYTHONPATH=os.pathsep.join(map(str,(SRC, REPO, REPO/'tools',runtime, EVAL))),
                PYTHONUSERBASE=str(INT/'botsort_plus_v0_1_0/python_user'),CUDA_VISIBLE_DEVICES='',
                OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
                PYTHONDONTWRITEBYTECODE='1',YOLO_CONFIG_DIR=str(ROOT/'runtime-config'),TRACKEVAL_ROOT=str(EVAL))
