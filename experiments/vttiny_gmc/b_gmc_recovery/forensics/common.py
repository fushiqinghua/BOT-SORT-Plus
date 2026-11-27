import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INT = ROOT.parent
JOINT = INT / "b_gmc_joint_r1"
REPO = JOINT / "repo"
PRIOR = INT / "native_botsort_gmc_0030_fp_v1"
FACTOR = INT / "native_botsort_gmc_factorial_v1"
SETTINGS = INT / "botsort_plus_v0_4_motion_blur/v04_followup_round2/P0/motion_blur_v0.4.local.json"
EVAL = INT.parent / "external/TrackEval"
B = "B_early_moderate"
COMBO = B + "_dual_low"
SEQS = ["DJI_0030_2", "DJI_0169_3"]
OUT = ROOT / "results"


def read(path):
    return json.loads(path.read_bytes())


def save(path, obj):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
