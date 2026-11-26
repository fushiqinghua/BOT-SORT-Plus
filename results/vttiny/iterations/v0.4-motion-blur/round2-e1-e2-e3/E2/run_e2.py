"""Run the fixed four-sequence check or one E2 full experiment."""
import json
from pathlib import Path
import subprocess
import sys

stage, = sys.argv[1:]
assert stage in ("smoke", "full"), stage
job = Path(__file__).resolve().parent
suite = job.parents[1]
project = suite.parents[1]
repo = suite / "BOT-SORT-Plus-followup-c3e142e6"
frozen = suite / "BOT-SORT-Plus/outputs/diagnostics_v0.4_full30_round1"
raw = suite / "round1/d3_validate_output_only_v1/variant_full"
previous = suite / "round1/d3_output_only_comparison_v1/full/trackeval"
gt = project / "data/official_gt/GT/GT_00/test"
trackeval = project / "external/TrackEval"
detections = repo / "results/vttiny/yolo11m_rgb_rgbt/rgbt/detections"
p0 = job.parent / "P0/verification.json"
output = job / stage
if stage == "full":
    smoke = json.loads((job / "smoke/E2_evaluation_summary.json").read_bytes())
    assert smoke["status"] == "COMPLETE" and smoke["smoke"]
    assert smoke["clear"]["target"]["frame"] == 58
for script, *options in [
    ("e2_boxes.py", repo, frozen, raw, p0, output, *(["--smoke"] if stage == "smoke" else [])),
    ("e2_evaluate.py", repo, frozen, output, gt, trackeval, detections, previous),
]:
    command = [sys.executable, "-B", "-u", str(job / script), *map(str, options)]
    print(command, flush=True)
    subprocess.run(command, check=True)
