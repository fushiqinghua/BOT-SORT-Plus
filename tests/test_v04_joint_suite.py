"""Experiment-scope, immutable-output and GT-boundary software contracts."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from experiments.vttiny_gmc.round3_joint import common, suite


class SuiteContracts(unittest.TestCase):
    def test_method_matrix_has_only_one_declared_factor_per_control(self):
        self.assertEqual(common.METHODS["B_skip_only"],
                         {"camera": "skip_only", "dual_low": False, "evidence_both": False})
        self.assertEqual(common.METHODS["B_adaptive_no_dual_low"],
                         {"camera": "adaptive_gmc", "dual_low": False, "evidence_both": False})
        a = dict(common.frozen_config("B_skip_only"))
        b = dict(common.frozen_config("B_adaptive_no_dual_low"))
        a.pop("name"); b.pop("name")
        self.assertEqual(a, b)

    def test_prediction_process_gets_no_truth_or_future_references(self):
        actual = common.prediction_settings({"schema": common.SCHEMA, "baseline_root": "/det",
            "images_root": "/rgb", "gt_root": "/gt", "manual_labels": "/labels", "references": {"B": "future"},
            "original_video_groups": {"clip": "video"}, "sequences": ["special_case"], "gmc_mode": "cached"})
        self.assertEqual(set(actual), {"schema", "baseline_root", "images_root", "gmc_mode"})

    def test_init_preserves_old_settings_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, dest = root / "old.json", root / "new.json"
            old.write_text(json.dumps({"features_dir": "真实缓存", "runtime_root": None}), encoding="utf-8-sig")
            original = old.read_bytes()
            suite.init_settings(old, dest, joint_run=root / "joint", factorial_run=root / "factor")
            new = common.read_json(dest)
            self.assertEqual(new["features_dir"], "真实缓存")
            self.assertEqual(new["gmc_mode"], "cached")
            self.assertEqual(set(new["warp_sources"]), {"native", "skip_only", "adaptive_gmc"})
            with self.assertRaises(FileExistsError):
                suite.init_settings(old, dest)
            self.assertEqual(old.read_bytes(), original)

    def test_analyze_explicitly_passes_selected_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "analysis"
            with patch.object(suite, "select_sequences", return_value={"clip": 5}), \
                 patch.object(suite, "run_child") as child, patch.object(suite, "read_json", return_value={"status": "PASS"}):
                suite.analyze({"sequences": ["clip"]}, output)
            args = child.call_args.args[1]
            self.assertEqual(args[-2:], ["--sequences", "clip"])

    def test_small_smoke_cannot_unlock_unchecked_full_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common.write_json(root / "smoke/verification.json", {"status": "PASS", "contract_sha256": "same",
                                                                "sequences": {common.SMOKE[0]: 1}})
            with patch.object(suite, "contract", return_value={}), patch.object(suite, "json_digest", return_value="same"), \
                 patch.object(suite, "select_sequences", return_value={s: 1 for s in common.SMOKE}):
                with self.assertRaisesRegex(ValueError, "scope is insufficient"):
                    suite.require_smoke({"output_root": str(root)})

    def test_smoke_contract_change_fails_before_any_tracking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common.write_json(root / "smoke/verification.json", {"status": "PASS", "contract_sha256": "old"})
            with patch.object(suite, "contract", return_value={}), patch.object(suite, "json_digest", return_value="new"):
                with self.assertRaisesRegex(ValueError, "these inputs"):
                    suite.require_smoke({"output_root": str(root)})

    def test_archive_excludes_raw_inputs_and_optional_large_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            for path in ("report.md", "probes/B/clip.jsonl.gz", "warps/B/clip.jsonl.gz", "gt.txt",
                         "model.pt", "features.npz", "partial.tmp", "prior.zip", "evaluation_input/protocols/B.json"):
                dest = root / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"software fixture")
            result = suite.package(root)
            with zipfile.ZipFile(result["path"]) as z:
                self.assertEqual(z.namelist(), ["report.md"])
            with self.assertRaises(FileExistsError):
                suite.package(root)
            result = suite.package(root, include_traces=True, archive_path=Path(tmp)/"with_traces.zip")
            with zipfile.ZipFile(result["path"]) as z:
                self.assertEqual(set(z.namelist()), {"report.md", "probes/B/clip.jsonl.gz", "warps/B/clip.jsonl.gz"})


if __name__ == "__main__":
    unittest.main()
