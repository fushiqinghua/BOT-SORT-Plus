"""Opt-in integration with the exact baseline TrackEval source commit.

Set VTTINY_TEST_TRACKEVAL_ROOT to the extracted pinned source directory. When
running in an overlay that has not been applied yet, also set
VTTINY_TEST_BASELINE_ROOT to the uploaded baseline repository. The fixtures are
explicitly synthetic software diagnostics, never reported as model performance.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@unittest.skipUnless(os.environ.get("VTTINY_TEST_TRACKEVAL_ROOT"),
                     "Set VTTINY_TEST_TRACKEVAL_ROOT to execute the pinned TrackEval integration")
class PinnedTrackEvalIntegrationTests(unittest.TestCase):
    def test_synthetic_perfect_and_one_miss_one_fp_through_real_cli(self):
        trackeval = Path(os.environ["VTTINY_TEST_TRACKEVAL_ROOT"]).resolve()
        baseline = Path(os.environ.get("VTTINY_TEST_BASELINE_ROOT", str(REPO))).resolve()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "test_repository"
            (project / "tools").mkdir(parents=True)
            (project / "provenance").mkdir()
            shutil.copy2(REPO / "tools" / "evaluate_botsort_plus.py", project / "tools")
            for name in ("evaluate_vttiny.py", "vttiny_tracking_common.py"):
                shutil.copy2(baseline / "tools" / name, project / "tools")
            shutil.copy2(baseline / "provenance" / "trackeval.json", project / "provenance")
            sequences = {"synthetic_object": 5, "synthetic_empty": 3}
            gt_root = root / "synthetic_rgb_gt"
            gt_object = "".join(f"{frame},1,10,10,4,4,1,1,1\n" for frame in range(1, 6))
            for sequence, count in sequences.items():
                target = gt_root / sequence / "gt" / "gt.txt"
                target.parent.mkdir(parents=True)
                target.write_text(gt_object if sequence == "synthetic_object" else "")
                (target.parent.parent / "seqinfo.ini").write_text(
                    f"[Sequence]\nname={sequence}\nimDir=img1\nframeRate=15\n"
                    f"seqLength={count}\nimWidth=640\nimHeight=512\nimExt=.jpg\n")
            run = root / "synthetic_run"
            for name in ("baseline", "one_miss_one_fp"):
                directory = run / "trackers" / name / "data"
                directory.mkdir(parents=True)
                perfect = [f"{frame},1,10,10,4,4,1,-1,-1,-1\n" for frame in range(1, 6)]
                if name == "one_miss_one_fp":
                    perfect.pop(2)
                (directory / "synthetic_object.txt").write_text("".join(perfect))
                (directory / "synthetic_empty.txt").write_text(
                    "1,2,100,100,4,4,1,-1,-1,-1\n" if name == "one_miss_one_fp" else "")
                protocol = {
                    "schema": "botsort_plus_run_v1", "status": "COMPLETE",
                    "purpose": "SYNTHETIC_SOFTWARE_DIAGNOSTIC_ONLY_NOT_MODEL_RESULTS",
                    "config": {"name": name}, "native_config": {}, "sequences": sequences,
                    "subset": True, "detector_sha256": "a" * 64, "rgb_encoder_sha256": "b" * 64,
                    "source_protocol_sha256": "c" * 64,
                    "detection_fingerprints": {seq: "d" * 64 for seq in sequences},
                    "feature_hashes": {seq: {"features": "e" * 64, "metadata": "f" * 64} for seq in sequences},
                    "encoder_contracts": {"rgb": {"sha256": "b" * 64, "profile": "synthetic_diagnostic"}},
                    "output_prediction_hashes": {seq: digest(directory / f"{seq}.txt") for seq in sequences},
                }
                path = run / "protocols" / f"{name}.json"
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps(protocol))
            env = dict(os.environ)
            env["VTTINY_ROOT"] = str(project)
            env["MPLBACKEND"] = "Agg"
            result = subprocess.run(
                [sys.executable, str(project / "tools" / "evaluate_botsort_plus.py"),
                 "--run-dir", str(run), "--trackers", "baseline", "one_miss_one_fp",
                 "--gt-root", str(gt_root), "--trackeval-root", str(trackeval)],
                cwd=project, env=env, text=True, capture_output=True, timeout=90)
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            summary = json.loads((run / "metrics" / "summary.json").read_text())
            combined = {row["tracker"]: row for row in summary["results"] if row["sequence"] == "COMBINED_SEQ"}
            for metric in ("HOTA", "IDF1", "MOTA"):
                self.assertAlmostEqual(combined["baseline"][metric], 100.0, places=6)
            self.assertEqual([combined["baseline"][key] for key in ("FP", "FN", "IDSW")], [0, 0, 0])
            self.assertEqual(combined["one_miss_one_fp"]["FP"], 1)
            self.assertEqual(combined["one_miss_one_fp"]["FN"], 1)
            self.assertEqual(combined["one_miss_one_fp"]["IDSW"], 0)
            self.assertLess(combined["one_miss_one_fp"]["HOTA"], 100.0)
            self.assertAlmostEqual(combined["one_miss_one_fp"]["MOTA"], 60.0)
            comparison = json.loads((run / "metrics" / "comparison.json").read_text())
            self.assertEqual(comparison["reference"]["kind"], "same_run_replay")
            self.assertEqual(comparison["results"][1]["delta"]["FP"], 1)
            audit = json.loads((run / "metrics" / "evaluation_audit.json").read_text())
            self.assertEqual(audit["evaluator"]["commit"], "12c8791b303e0a0b50f753af204249e622d0281a")
            self.assertGreater(audit["evaluator"]["verified_source_files"], 30)
            # SEQ_INFO supplied explicit counts; the old project seqmap does not
            # exist in this isolated repository, proving it is not required.
            self.assertFalse((project / "data" / "processed_v1").exists())


if __name__ == "__main__":
    unittest.main()
