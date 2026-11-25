"""Opt-in CLI integration on generated data, never an algorithm result.

M2_TEST_BOXMOT=1 enables this test in the reviewed BoxMOT runtime. Set
TRACKEVAL_ROOT as well to exercise the pinned evaluator. Every CLI operation
uses its own process because prediction deliberately installs a GT-read guard.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
ENABLED = os.environ.get("M2_TEST_BOXMOT") == "1"
VARIANTS = ("native_iou", "m2_nll_baseline_r", "m2_nll_floor", "m2_nll_calibrated")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@unittest.skipUnless(ENABLED, "Set M2_TEST_BOXMOT=1 with the reviewed runtime for CLI integration")
class CLIIntegrationTests(unittest.TestCase):
    def _command(self, *arguments, expected=0):
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        command = [sys.executable, "-m", "botsort_research.cli", *map(str, arguments)]
        completed = subprocess.run(command, cwd=REPOSITORY, env=env, text=True,
                                   capture_output=True, timeout=90)
        self.assertEqual(completed.returncode, expected,
                         f"Command: {command}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
        return completed

    def _fixture(self, root):
        import cv2
        cache = root / "cache"
        images = root / "images"
        gt_root = root / "gt"
        features_root = root / "features"
        cache.mkdir()
        features_root.mkdir()
        names = ["DJI_9001_1", "DJI_9002_1", "DJI_9003_1"]
        rng = np.random.default_rng(21)
        background = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        # Strongly textured, deterministic images make the optical-flow stage
        # observable without depending on a real scene or any model weights.
        for x in range(10, 310, 20):
            for y in range(10, 230, 20):
                cv2.circle(background, (x, y), 3, (255, 255, 255), -1)
        for sequence in names:
            frames, gt_rows, vectors, indices, offsets = [], [], [], [], [0]
            for frame in range(1, 25):
                detections = []
                truth = []
                for identity in (1, 2):
                    x = (45 if identity == 1 else 90) + .4 * frame
                    y = 70 if identity == 1 else 80
                    w = 9.0 + identity + .2 * (frame % 3)
                    h = 10.0 + identity + .1 * (frame % 4)
                    truth.append((identity, x, y, w, h))
                    if frame not in {1, 12, 13}:
                        score = .25 if identity == 1 and frame in {6, 7, 8} else .9
                        jitter = .15 * ((frame % 5) - 2)
                        detections.append([x+jitter, y-jitter, x+w+jitter, y+h-jitter, score, identity+1])
                rgb = Path("test2017") / sequence / "00" / f"{frame-1:05d}.png"
                thermal = Path("test2017") / sequence / "01" / f"{frame-1:05d}.png"
                image = np.roll(background, shift=frame-1, axis=1)
                for relative in (rgb, thermal):
                    destination = images / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    self.assertTrue(cv2.imwrite(str(destination), image))
                frames.append({"frame": frame, "image": rgb.as_posix(), "thermal": thermal.as_posix(), "detections": detections})
                # The third evaluation sequence intentionally has empty GT.
                # It must remain in evaluation and contribute false positives.
                if sequence != names[2]:
                    gt_rows.extend(f"{frame},{identity},{x},{y},{w},{h},1,{identity},1\n"
                                   for identity, x, y, w, h in truth)
                for index, detection in enumerate(detections):
                    if detection[4] > .5:
                        vector = np.zeros(4, dtype=np.float32)
                        if not (frame == 4 and index == 0):
                            vector[index] = 1.0
                        vectors.append(vector)
                        indices.append(index)
                offsets.append(len(vectors))
            detection_path = cache / f"{sequence}.json"
            detection_path.write_text(json.dumps({"model_sha256": "a" * 64,
                "fixture": "SYNTHETIC_SOFTWARE_VALIDATION_ONLY", "frames": frames}), encoding="utf-8")
            gt_path = gt_root / sequence / "gt" / "gt.txt"
            gt_path.parent.mkdir(parents=True)
            gt_path.write_text("".join(gt_rows), encoding="utf-8")
            feature_path = features_root / f"{sequence}.npz"
            np.savez_compressed(feature_path, features=np.asarray(vectors, dtype=np.float32).reshape(-1, 4),
                                indices=np.asarray(indices, dtype=np.int64), offsets=np.asarray(offsets, dtype=np.int64),
                                frame_ids=np.arange(1, 25, dtype=np.int64))
            feature_path.with_suffix(".json").write_text(json.dumps({
                "detection_sha256": _hash(detection_path), "features_sha256": _hash(feature_path),
                "reid_sha256": "b" * 64, "purpose": "Synthetic one-hot vectors, not learned ReID features"}), encoding="utf-8")
        split = root / "split.json"
        split.write_text(json.dumps({"schema_version": 1, "interpretation": "retrospective_development_only",
                                    "calibration": names[:1], "evaluation": names[1:]}), encoding="utf-8")
        config = json.loads((REPOSITORY / "configs/research/m2_default.json").read_text())
        config["expected_image_shape"] = [240, 320]
        config["status"] = "synthetic_software_validation_only"
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return cache, images, gt_root, features_root, split, names

    def test_full_cli_pipeline_and_provenance_rejection(self):
        with tempfile.TemporaryDirectory(prefix="botsort_cli_fixture_") as temp:
            root = Path(temp)
            cache, images, gt_root, features, split, names = self._fixture(root)
            doctor = root / "doctor.json"
            self._command("doctor", "--cache-dir", cache, "--image-root", images,
                          "--check-runtime", "--output", doctor)
            check = json.loads(doctor.read_text())
            self.assertEqual(check["frames"], 72)
            self.assertEqual(check["missing_image_count"], 0)
            self.assertEqual(check["empty_detection_frames"], 9)
            self._command("make-split", "--cache-dir", cache, "--seed", 3,
                          "--calibration-fraction", .34, "--output", root / "automatic-split.json")
            auto = json.loads((root / "automatic-split.json").read_text())
            self.assertFalse(set(auto["calibration"]) & set(auto["evaluation"]))
            self.assertEqual(auto["interpretation"], "retrospective_development_only")
            noise = root / "noise.json"
            self._command("calibrate", "--cache-dir", cache, "--gt-root", gt_root,
                          "--split", split, "--min-samples", 20, "--output", noise)
            calibrated = json.loads(noise.read_text())
            self.assertGreaterEqual(calibrated["diagnostics"]["samples"], 20)
            self.assertEqual(calibrated["provenance"]["calibration_sequences"], names[:1])

            run = root / "run"
            self._command("run", "--cache-dir", cache, "--image-root", images,
                          "--config", root / "config.json",
                          "--split", split, "--noise-model", noise, "--cmc", "none", "--trace",
                          "--variants", *VARIANTS, "--output", run)
            manifest = json.loads((run / "experiment.json").read_text())
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertEqual(manifest["sequences"], names[1:])
            self.assertEqual(len(manifest["timing"]), 8)
            self.assertEqual(manifest["runtime"]["max_time_lost"], 30)
            self.assertEqual(manifest["noise_model_sha256"], _hash(noise))
            for variant in VARIANTS:
                for sequence in names[1:]:
                    path = run / "trackers" / variant / "data" / f"{sequence}.txt"
                    rows = np.loadtxt(path, delimiter=",", ndmin=2)
                    self.assertEqual(rows.shape[1], 10)
                    self.assertGreater(len(rows), 10)
                    self.assertTrue(np.isfinite(rows).all())
                    self.assertTrue(np.all((rows[:, 0] >= 1) & (rows[:, 0] <= 24)))
                    self.assertFalse(np.isin(rows[:, 0], [1, 12, 13]).any(), "No retrospective or prediction-only gap outputs")
                    self.assertTrue(np.all(rows[:, 4:6] > 0))
                    if variant != "native_iou":
                        trace_path = run / "traces" / variant / f"{sequence}.jsonl"
                        trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
                        self.assertEqual(len(trace), 48)
                        self.assertTrue(any(event["matches"] for event in trace))
                        self.assertEqual({event["stage"] for event in trace}, {"first", "second"})

            motion = root / "m1"
            self._command("diagnose-m1", "--cache-dir", cache, "--image-root", images,
                          "--split", split, "--max-pairs", 5, "--output", motion)
            motion_manifest = json.loads((motion / "manifest.json").read_text())
            self.assertEqual(len(motion_manifest["sequences"]), 2)
            self.assertTrue(all(item["status"] == "COMPLETE" for item in motion_manifest["sequences"]))
            self.assertTrue(all(item["estimated_pairs"] > 0 for item in motion_manifest["sequences"]))

            appearance = root / "m3"
            self._command("diagnose-m3", "--cache-dir", cache, "--gt-root", gt_root,
                          "--split", split, "--embeddings-dir", features,
                          "--predictions-dir", run / "trackers/native_iou/data", "--output", appearance)
            report = json.loads((appearance / "summary.json").read_text())
            self.assertEqual(report["status"], "COMPLETE")
            self.assertTrue(all(item["features"]["status"] == "available" for item in report["sequences"]))
            self.assertTrue(all(item["features"]["zero_vectors_excluded"] == 1 for item in report["sequences"]))
            self.assertGreater(len((appearance / "appearance_pairs.csv").read_text().splitlines()), 1)
            self.assertGreater(len((appearance / "prediction_gaps.csv").read_text().splitlines()), 1)

            corrupted = copy.deepcopy(calibrated)
            corrupted["provenance"]["inputs"][names[0]]["detections_sha256"] = "0" * 64
            wrong_noise = root / "wrong-noise.json"
            wrong_noise.write_text(json.dumps(corrupted), encoding="utf-8")
            failed = self._command("run", "--cache-dir", cache, "--image-root", images,
                                   "--config", root / "config.json",
                                   "--split", split, "--noise-model", wrong_noise,
                                   "--variants", "m2_nll_calibrated", "--output", root / "bad-run", expected=2)
            self.assertIn("Calibration cache changed", failed.stderr)
            self.assertFalse((root / "bad-run").exists())
            leaked = self._command("run", "--cache-dir", cache, "--image-root", images,
                                   "--config", root / "config.json",
                                   "--split", split, "--subset", "calibration", "--noise-model", noise,
                                   "--variants", "m2_nll_calibrated", "--output", root / "leaked-run", expected=2)
            self.assertIn("evaluation partition", leaked.stderr)
            self.assertFalse((root / "leaked-run").exists())

            evaluator = os.environ.get("TRACKEVAL_ROOT")
            if evaluator:
                self._command("evaluate", "--run-dir", run, "--gt-root", gt_root,
                              "--trackeval-root", evaluator)
                result = json.loads((run / "metrics/summary.json").read_text())
                combined = [row for row in result["results"] if row["sequence"] == "COMBINED_SEQ"]
                self.assertEqual({row["tracker"] for row in combined}, set(VARIANTS))
                self.assertTrue(all(np.isfinite(row[k]) for row in combined for k in ("HOTA", "AssA", "IDF1", "MOTA")))
                empty_gt = [row for row in result["results"] if row["sequence"] == names[2]]
                self.assertEqual(len(empty_gt), 4)
                self.assertTrue(all(row["FP"] > 0 for row in empty_gt))
                comparison = root / "comparison"
                self._command("summarize", "--runs", run, "--output", comparison)
                summary = json.loads((comparison / "comparison.json").read_text())
                self.assertEqual(len(summary["results"]), 4)
                self.assertTrue(all(row["delta_HOTA"] is None for row in summary["results"]),
                                "CMC-off fixture must not invent an SOF baseline difference")
                self.assertTrue((comparison / "comparison.csv").is_file())
                self.assertTrue((comparison / "comparison.md").is_file())


if __name__ == "__main__":
    unittest.main()
