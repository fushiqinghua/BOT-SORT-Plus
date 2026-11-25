"""Round2 CLI contract on synthetic data; never reports real algorithm gains."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from botsort_research.io import dump_json, load_json, sha256


ENABLED = os.environ.get("M2_TEST_BOXMOT") == "1" and bool(os.environ.get("TRACKEVAL_ROOT"))


def _fixture_helper():
    # Import a module, not its TestCase into our module globals; unittest must
    # not discover and rerun the first-round TestCase through this helper.
    path = Path(__file__).with_name("test_cli_integration.py")
    spec = importlib.util.spec_from_file_location("_round1_fixture_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CLIIntegrationTests()


@unittest.skipUnless(ENABLED, "Set M2_TEST_BOXMOT=1 and TRACKEVAL_ROOT for the real runtime/evaluator")
class Round2CLIIntegrationTests(unittest.TestCase):
    def test_all_m2_and_rgb_variants_replay_evaluate_and_attribute(self):
        from botsort_research.adapter import SUPPORTED_VARIANTS
        from botsort_research.appearance_adapter import APPEARANCE_VARIANTS
        helper = _fixture_helper()
        with tempfile.TemporaryDirectory(prefix="round2_cli_fixture_") as tmp:
            root = Path(tmp)
            cache, images, gt_root, features, split, names = helper._fixture(root)
            # The first-round diagnostic deliberately had one zero feature. M3
            # online replay correctly refuses invalid vectors, so this distinct
            # synthetic fixture provides valid per-identity one-hot features.
            for sequence in names:
                feature_path = features / f"{sequence}.npz"
                with np.load(feature_path, allow_pickle=False) as archive:
                    arrays = {k: archive[k].copy() for k in archive.files}
                zero = np.linalg.norm(arrays["features"], axis=1) == 0
                arrays["features"][zero, 0] = 1
                np.savez_compressed(feature_path, **arrays)
                metadata = load_json(feature_path.with_suffix(".json"))
                metadata["features_sha256"] = sha256(feature_path)
                dump_json(feature_path.with_suffix(".json"), metadata)
                # Hard negative pairs require distinct IDs of the same class.
                gt_path = gt_root / sequence / "gt/gt.txt"
                rows = [line.split(",") for line in gt_path.read_text().splitlines()]
                for row in rows:
                    row[7] = "1"
                gt_path.write_text("".join(",".join(row) + "\n" for row in rows))
            config_path = root / "config.json"
            config = load_json(config_path)
            config["tracker"].update(appearance_thresh=.4, proximity_thresh=.5)
            config["variant_overrides"] = {"iou_relaxed": {"tracker": {"match_thresh": .9}}}
            dump_json(config_path, config)
            helper._command("calibrate-appearance", "--cache-dir", cache, "--gt-root", gt_root,
                            "--embeddings-dir", features, "--split", split,
                            "--min-same", 2, "--min-different", 2, "--output", root / "appearance")
            model_path = root / "appearance/model.json"
            model = load_json(model_path)
            self.assertEqual(model["provenance"]["sequences"], names[:1])
            self.assertGreater(model["global"]["same_pairs"], 2)
            self.assertGreater(model["global"]["different_pairs"], 2)
            variants = [v for v in SUPPORTED_VARIANTS if v != "m2_nll_calibrated"] + list(APPEARANCE_VARIANTS)
            run = root / "run"
            helper._command("run", "--cache-dir", cache, "--image-root", images,
                            "--config", config_path, "--split", split, "--cmc", "none",
                            "--embeddings-dir", features, "--appearance-model", model_path,
                            "--variants", *variants, "--trace", "--output-trace", "--output", run)
            manifest = load_json(run / "experiment.json")
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertEqual(len(manifest["timing"]), 2 * len(variants))
            self.assertEqual(manifest["effective_variants"]["iou_relaxed"]["tracker"]["match_thresh"], .9)
            for variant in ("reid_recovery_fixed", "reid_recovery"):
                events = [json.loads(line) for line in (run / "traces" / variant / f"{names[1]}.jsonl").read_text().splitlines()]
                self.assertTrue(any(e["stage"] == "appearance_recovery" and e["matches"] for e in events))
            helper._command("evaluate", "--run-dir", run, "--gt-root", gt_root,
                            "--trackeval-root", os.environ["TRACKEVAL_ROOT"])
            helper._command("analyze-run", "--run-dir", run, "--gt-root", gt_root,
                            "--output", root / "attribution")
            attribution = load_json(root / "attribution/summary.json")
            self.assertEqual(attribution["status"], "COMPLETE")
            self.assertEqual(len(attribution["sequences"]), len(variants) * 2)
            self.assertTrue(all(r["verified"] for r in attribution["sequences"]))
            empty_gt = [r for r in attribution["sequences"] if r["sequence"] == names[2]]
            self.assertTrue(all(r["FP"] > 0 and r["GT"] == 0 for r in empty_gt))
            helper._command("summarize", "--runs", run, "--output", root / "comparison")
            self.assertEqual(len(load_json(root / "comparison/comparison.json")["results"]), len(variants))


if __name__ == "__main__":
    unittest.main()
