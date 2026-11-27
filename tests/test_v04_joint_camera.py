"""Frozen-camera stream contracts; these tests do not run a tracker or a model."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from experiments.vttiny_gmc.round3_joint import camera


class CameraCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.protocol = {
            "status": "COMPLETE", "native_config": {"cmc_method": "sof"},
            "source_protocol_sha256": "source", "detector_sha256": "detector",
            "rgb_encoder_sha256": "encoder", "encoder_contracts": {"rgb": {"id": "same"}},
            "sequences": {"seq_a": 2}, "detection_fingerprints": {"seq_a": "detections"},
            "feature_hashes": {"seq_a": {"features": "features", "metadata": "meta"}},
            "decoded_image_hashes": {"seq_a": "pixels"},
        }
        self.native = [[1., 0., 5.], [0., 1., 3.]]
        self.identity = camera.IDENTITY.tolist()

    def tearDown(self):
        self.temporary.cleanup()

    def rows(self, policy="skip_only"):
        return [{"frame": f, "native": self.native,
                 "used": self.native if policy == "native" else self.identity,
                 "choice": "sof" if policy == "native" else "identity",
                 "eligible": [] if policy == "native" else ["identity"],
                 "trigger": policy != "native", "model_called": False} for f in (1, 2)]

    def make_cache(self, policy="skip_only", rows=None, flags=None):
        root = self.root / ("cache_" + policy)
        (root / "protocols").mkdir(parents=True, exist_ok=True)
        (root / "warps" / policy).mkdir(parents=True, exist_ok=True)
        source = copy.deepcopy(self.protocol)
        spec = json.loads((camera.REPO / "experiments/vttiny_gmc/reference.json").read_text())["gmc_factorial"]
        spec.update(skip=policy != "native", lightglue=policy == "adaptive_gmc")
        if policy != "adaptive_gmc":
            spec["model"] = None
        if flags:
            spec.update(flags)
        source["gmc_factorial"] = spec
        (root / "protocols" / (policy + ".json")).write_text(json.dumps(source))
        with gzip.open(root / "warps" / policy / "seq_a.jsonl.gz", "wt", encoding="utf-8") as stream:
            for row in self.rows(policy) if rows is None else rows:
                stream.write(json.dumps(row) + "\n")
        return {"gmc_mode": "cached", "warp_sources": {policy: {"run_dir": str(root), "method": policy}}}

    def stream(self, settings, policy="skip_only"):
        def never_called(*_):
            raise AssertionError("Cached replay must not execute native GMC")
        return camera.WarpCamera(settings, policy, "seq_a", self.protocol, never_called, None,
                                 self.root / "generated.jsonl.gz")

    def test_native_cache_preserves_native_matrix_and_never_calls_native(self):
        item = self.stream(self.make_cache("native"), "native")
        for f in (1, 2):
            native, used, backup, record = item.apply(None, None, f)
            np.testing.assert_array_equal(native, self.native)
            np.testing.assert_array_equal(used, native)
            np.testing.assert_array_equal(backup, native)
            self.assertEqual(record["choice"], "sof")
        manifest = item.close()
        self.assertEqual(manifest["frames"], 2)
        self.assertEqual(manifest["choices"], {"sof": 2})

    def test_identity_cache_and_fallback_are_exact_identity(self):
        item = self.stream(self.make_cache())
        for f in (1, 2):
            native, used, backup, _ = item.apply(None, None, f)
            np.testing.assert_array_equal(native, self.native)
            np.testing.assert_array_equal(used, self.identity)
            np.testing.assert_array_equal(backup, self.identity)
        item.close()

    def test_wrong_candidate_set_cache_is_rejected(self):
        settings = self.make_cache(flags={"skip": True, "lightglue": True})
        with self.assertRaisesRegex(ValueError, "candidate-set"):
            self.stream(settings)

    def test_changed_gmc_thresholds_are_rejected_even_with_correct_flags(self):
        spec = json.loads((camera.REPO / "experiments/vttiny_gmc/reference.json").read_text())["gmc_factorial"]
        spec["policy"]["median_trigger"] = 10.0
        settings = self.make_cache(flags={"policy": spec["policy"]})
        with self.assertRaisesRegex(ValueError, "thresholds"):
            self.stream(settings)

    def test_changed_selector_source_is_rejected(self):
        settings = self.make_cache(flags={"source_sha256": {"frozen_gmc.py": "bad", "gmc_variants.py": "bad"}})
        with self.assertRaisesRegex(ValueError, "source provenance"):
            self.stream(settings)

    def test_changed_lightglue_model_is_rejected(self):
        spec = json.loads((camera.REPO / "experiments/vttiny_gmc/reference.json").read_text())["gmc_factorial"]
        spec["model"]["commit"] = "another_model"
        settings = self.make_cache("adaptive_gmc", flags={"model": spec["model"]})
        with self.assertRaisesRegex(ValueError, "model provenance"):
            self.stream(settings, "adaptive_gmc")

    def test_missing_policy_source_is_not_substituted(self):
        settings = self.make_cache("adaptive_gmc")
        with self.assertRaisesRegex(ValueError, "Missing warp_sources.skip_only"):
            self.stream(settings)

    def test_truncated_stream_is_rejected(self):
        item = self.stream(self.make_cache(rows=self.rows()[:1]))
        try:
            item.apply(None, None, 1)
            with self.assertRaisesRegex(ValueError, "Truncated"):
                item.apply(None, None, 2)
        finally:
            item.abort()

    def test_extra_frame_is_rejected_at_close(self):
        rows = self.rows()
        rows.append({**rows[-1], "frame": 3})
        item = self.stream(self.make_cache(rows=rows))
        item.apply(None, None, 1)
        item.apply(None, None, 2)
        with self.assertRaisesRegex(ValueError, "Extra frame"):
            item.close()

    def test_nonfinite_matrix_is_rejected(self):
        rows = self.rows()
        rows[0]["used"] = [[1., 0., float("nan")], [0., 1., 0.]]
        item = self.stream(self.make_cache(rows=rows))
        try:
            with self.assertRaisesRegex(ValueError, "Invalid used"):
                item.apply(None, None, 1)
        finally:
            item.abort()

    def test_call_order_and_stored_frame_order_are_checked(self):
        rows = self.rows()
        rows[0]["frame"] = 2
        item = self.stream(self.make_cache(rows=rows))
        try:
            with self.assertRaisesRegex(ValueError, "order violation"):
                item.apply(None, None, 2)
            with self.assertRaisesRegex(ValueError, "cache frame mismatch"):
                item.apply(None, None, 1)
        finally:
            item.abort()

    def test_identity_and_sof_choices_must_agree_with_matrix(self):
        rows = self.rows()
        rows[0]["used"] = self.native
        item = self.stream(self.make_cache(rows=rows))
        try:
            with self.assertRaisesRegex(ValueError, "non-identity"):
                item.apply(None, None, 1)
        finally:
            item.abort()

    def test_selected_alternative_must_be_eligible(self):
        rows = self.rows()
        rows[0]["eligible"] = []
        item = self.stream(self.make_cache(rows=rows))
        try:
            with self.assertRaisesRegex(ValueError, "absent from eligible"):
                item.apply(None, None, 1)
        finally:
            item.abort()

    def test_cached_backup_cannot_change_guard_semantics(self):
        rows = self.rows()
        rows[0]["backup"] = self.native
        item = self.stream(self.make_cache(rows=rows))
        try:
            with self.assertRaisesRegex(ValueError, "Recorded backup"):
                item.apply(None, None, 1)
        finally:
            item.abort()

    def test_skip_only_cannot_claim_lightglue_execution(self):
        rows = self.rows()
        rows[0]["model_called"] = True
        item = self.stream(self.make_cache(rows=rows))
        try:
            with self.assertRaisesRegex(ValueError, "Non-LightGlue"):
                item.apply(None, None, 1)
        finally:
            item.abort()

    def test_cache_detects_recorded_file_hash_mismatch(self):
        settings = self.make_cache()
        settings["warp_sources"]["skip_only"]["sha256"] = {"seq_a": "0" * 64}
        with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
            self.stream(settings)

    def test_live_skip_selector_receives_no_matcher(self):
        calls = []

        class FakeSelector:
            def __init__(self, allow_skip, matcher):
                calls.append((allow_skip, matcher))

            def apply(self, image, dets, native):
                return camera.IDENTITY, {"choice": "identity", "eligible": ["identity"],
                                         "trigger": True, "model_called": False}

        fake = types.SimpleNamespace(Selector=FakeSelector, __file__=str(camera.FACTOR / "gmc_variants.py"))
        with patch.dict("sys.modules", {"gmc_variants": fake}):
            item = camera.WarpCamera({"gmc_mode": "live"}, "skip_only", "seq_a", self.protocol,
                                     lambda *_: self.native, object(), self.root / "live.jsonl.gz")
            self.assertEqual(calls, [(True, None)])
            for f in (1, 2):
                item.apply(None, None, f)
            self.assertEqual(item.close()["mode"], "live")

    def test_live_adaptive_requires_local_vendor_before_model_import(self):
        with self.assertRaises(FileNotFoundError):
            camera.setup_matcher({"gmc_mode": "live"}, "adaptive_gmc")

    def test_live_adaptive_cannot_silently_degrade_to_skip_only(self):
        with self.assertRaisesRegex(ValueError, "no skip-only fallback"):
            camera.WarpCamera({"gmc_mode": "live"}, "adaptive_gmc", "seq_a", self.protocol,
                              lambda *_: self.native, None, self.root / "bad_live.jsonl.gz")

    def test_preloaded_selector_from_another_project_is_rejected(self):
        fake = types.SimpleNamespace(__file__=str(self.root / "other/gmc_variants.py"))
        with patch.dict("sys.modules", {"gmc_variants": fake}):
            with self.assertRaisesRegex(RuntimeError, "outside the pinned"):
                camera.setup_matcher({"gmc_mode": "live"}, "skip_only")

    def test_verified_vendor_is_actual_hub_cache_and_download_fallback_is_blocked(self):
        vendor = self.root / "vendor"
        weights = vendor / "torch/hub/checkpoints/weights.pth"
        source = vendor / "LightGlue-fixture/lightglue/__init__.py"
        weights.parent.mkdir(parents=True)
        source.parent.mkdir(parents=True)
        weights.write_bytes(b"fixture-weights")
        source.write_text("# fixture source\n")
        reference = {"model": {"commit": "fixture", "weights": {"weights.pth": camera.sha256(weights)},
                               "source_sha256": {"__init__.py": camera.sha256(source)}}}
        observed = []
        hub = types.SimpleNamespace(set_dir=lambda path: observed.append(path))
        def network(*_args, **_kwargs):
            self.fail("Constructor must never reach a network fallback")
        hub.download_url_to_file = network
        class FakeMatcher:
            def __init__(self):
                hub.download_url_to_file("unexpected-weight-url", "unexpected-file")
        modules = {
            "frozen_gmc": types.SimpleNamespace(Matcher=FakeMatcher, __file__=str(camera.FACTOR / "frozen_gmc.py")),
            "lightglue": types.SimpleNamespace(__file__=str(source)),
            "torch": types.SimpleNamespace(hub=hub),
        }
        with patch.object(camera, "_verify_sources", return_value=(reference, {})), patch.dict("sys.modules", modules):
            with self.assertRaisesRegex(RuntimeError, "network download"):
                camera.setup_matcher({"gmc_mode": "live", "vendor_dir": str(vendor)}, "adaptive_gmc")
        self.assertEqual(observed, [str(vendor / "torch/hub")])
        self.assertIs(hub.download_url_to_file, network)


if __name__ == "__main__":
    unittest.main()
