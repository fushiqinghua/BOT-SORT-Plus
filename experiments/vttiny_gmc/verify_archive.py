"""Verify the three published experimental archives. Requires only Python 3.11+."""
import csv
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2] / "results/vttiny/iterations/v0.4-motion-blur/gmc"
ARMS = {
    "01-gmc-oru": ("native", "adaptive_gmc", "oru"),
    "02-factorial": ("native", "skip_only", "lightglue_only", "adaptive_gmc"),
    "03-dji0030": ("skip_only", "adaptive_gmc", "dual_low", "dual_low_center"),
}
METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "FP", "FN", "IDSW")


def require(condition, label):
    if not condition:
        raise ValueError(label)


def digest(data):
    return sha256(data).hexdigest()


def verify():
    predictions = {}
    metrics = {}
    shared_inputs = None
    count = 0
    for name, arms in ARMS.items():
        directory = ROOT / name
        manifest = json.loads((directory / "MANIFEST.json").read_bytes())
        bundle = directory / "evidence.zip"
        require(digest(bundle.read_bytes()) == manifest["evidence_zip_sha256"], bundle)
        with ZipFile(bundle) as evidence:
            require(set(evidence.namelist()) == set(manifest["members"]), name)
            for path, expected in manifest["members"].items():
                require(digest(evidence.read(path)) == expected, path)
            for path, expected in manifest["files"].items():
                require(digest((directory / path).read_bytes()) == expected == manifest["members"][path], path)
            report = json.loads(evidence.read("full/metrics/summary.json"))
            require(report["status"] == "PASS" and report["split"] == "test", name)
            require({row["tracker"] for row in report["results"]} == set(arms), name)
            for arm in arms:
                protocol = json.loads(evidence.read(f"full/protocols/{arm}.json"))
                seqs = protocol["sequences"]
                require(protocol["status"] == "COMPLETE" and len(seqs) == 30 and sum(seqs.values()) == 12590, arm)
                inputs = {key: protocol[key] for key in ("sequences", "detection_fingerprints", "feature_hashes",
                          "decoded_image_hashes", "detector_sha256", "rgb_encoder_sha256")}
                if shared_inputs is None:
                    shared_inputs = inputs
                require(inputs == shared_inputs, f"Different frozen inputs: {name}/{arm}")
                hashes = protocol["output_prediction_hashes"]
                require(set(hashes) == set(seqs), arm)
                require(predictions.setdefault(arm, hashes) == hashes, f"Control predictions differ: {arm}")
                for seq, expected in hashes.items():
                    path = f"full/trackers/{arm}/data/{seq}.txt"
                    require(manifest["members"][path] == expected, path)
                rows = [r for r in report["results"] if r["tracker"] == arm]
                require(len(rows) == 31 and {r["sequence"] for r in rows} == set(seqs) | {"COMBINED_SEQ"}, arm)
                values = {r["sequence"]: [r[k] for k in METRICS] for r in rows}
                require(metrics.setdefault(arm, values) == values, f"Control metrics differ: {arm}")
                count += len(hashes)
        print(f"{name}: {len(arms)} methods x 30 sequences")
    case = ROOT / "03-dji0030"
    replay = json.loads((ROOT / "entry_parity.json").read_bytes())
    sequence = replay["sequence"]
    require(sequence == "DJI_0030_2" and replay["frames"] == 590, "Entry point replay sequence")
    checks = replay["byte_identical_to_frozen_full_outputs"]
    require(set(checks) == {"dual_low", "dual_low_center"}, "Entry point replay methods")
    with ZipFile(case / "evidence.zip") as evidence:
        for arm, check in checks.items():
            data = evidence.read(f"full/trackers/{arm}/data/{sequence}.txt")
            require(check["sha256"] == digest(data) and check["rows"] == len(data.splitlines()), arm)
    for name, expected in (("diagnosis/frames.csv", 590), ("diagnosis/changed_fp.csv", 15),
                           ("counterfactual/metrics_rows.csv", 20)):
        with (case/name).open(encoding="utf-8-sig", newline="") as stream:
            require(len(list(csv.DictReader(stream))) == expected, name)
    print(f"Verified {count} full-sequence MOT files; {len(metrics)} distinct methods; 590 case frames; 20 interventions.")


if __name__ == "__main__":
    verify()
