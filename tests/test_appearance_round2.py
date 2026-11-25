"""ReID calibration/replay contracts and causal recovery lifecycle checks."""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from botsort_research.appearance import (APPEARANCE_MODEL_KIND, appearance_threshold,
    calibrate_appearance, dense_frame_features, fit_threshold, load_replay_features,
    validate_appearance_model)
from botsort_research.appearance_adapter import conservative_recovery_matches
from botsort_research.io import dump_json, sha256


def model(global_threshold=.3):
    return {"kind": APPEARANCE_MODEL_KIND, "global": {"threshold": global_threshold},
            "by_gap": {label: {"threshold": threshold} for label, threshold in
                       (("0", .2), ("1-3", .4), ("4-10", .5), ("11-30", .6))}}


class AppearanceNumericalTests(unittest.TestCase):
    def test_empirical_far_uses_ties_and_sparse_bins_require_backoff(self):
        result = fit_threshold([.1, .2, .3, .4], [.15, .15, .8, .9],
                               max_empirical_far=.25, min_same=1, min_different=1)
        self.assertEqual(result["threshold"], .1)
        self.assertEqual(result["different_accepted"], 0)
        self.assertEqual(result["same_accepted"], 1)
        self.assertGreater(result["descriptive_wilson95_far_upper"], 0)
        sparse = fit_threshold([.1], [.9], min_same=2, min_different=2)
        self.assertEqual(sparse["status"], "insufficient_pairs")
        self.assertIsNone(sparse["threshold"])
        disabled = fit_threshold([.5], [.1], min_same=1, min_different=1)
        self.assertIsNone(disabled["threshold"])

    def test_gap_threshold_and_explicit_disable(self):
        self.assertEqual([appearance_threshold(model(), gap) for gap in (0, 1, 3, 4, 10, 11, 30)],
                         [.2, .4, .4, .5, .5, .6, .6])
        self.assertEqual(appearance_threshold(model(), 5, conditioned=False), .3)
        self.assertIsNone(appearance_threshold(model(), 31))
        with self.assertRaises(ValueError):
            appearance_threshold(model(), -1)

    def test_recovery_requires_mutual_uniqueness_not_just_threshold(self):
        dist = np.array([[.1, .11], [.12, .4]])
        self.assertEqual(len(conservative_recovery_matches(dist, np.ones_like(dist, bool), [.5, .5], .05)), 0)
        dist = np.array([[.1, .8], [.7, .2]])
        np.testing.assert_array_equal(conservative_recovery_matches(dist, np.ones_like(dist, bool), [.5, .5]), [[0, 0], [1, 1]])
        np.testing.assert_array_equal(conservative_recovery_matches(dist, np.ones_like(dist, bool), [None, .5]), [[1, 1]])
        # A geometrically valid, threshold-rejected competitor still blocks an
        # ambiguous claim, rather than disappearing from the uniqueness test.
        dist = np.array([[.1], [.11]])
        self.assertEqual(len(conservative_recovery_matches(dist, np.ones_like(dist, bool), [.5, .01])), 0)

    def test_dense_rows_are_copied_and_low_score_is_not_evidence(self):
        frame = {"frame": 1, "detections": [[0, 0, 4, 4, .9, 0], [5, 0, 9, 4, .4, 0], [10, 0, 14, 4, .5, 0]]}
        features = {1: {0: np.array([3, 4], np.float32)}}
        dense = dense_frame_features(frame, features, 2)
        np.testing.assert_array_equal(dense, [[3, 4], [0, 0], [0, 0]])
        dense[0, 0] = 999
        self.assertEqual(features[1][0][0], 3)
        with self.assertRaisesRegex(ValueError, "missing or misaligned"):
            dense_frame_features(frame, {}, 2)

    def test_score_boundary_uses_runtime_float32_rounding(self):
        frame = {"frame": 1, "detections": [[0, 0, 4, 4, .50000001, 0]]}
        # This score rounds to exactly .5, which belongs to neither native
        # association stage; it must not require or fabricate a high feature.
        np.testing.assert_array_equal(dense_frame_features(frame, {1: {}}, 2), [[0, 0]])


class AppearanceFileTests(unittest.TestCase):
    def _fixture(self, root):
        cache, embeddings, gt = root / "cache", root / "embeddings", root / "GT"
        cache.mkdir(); embeddings.mkdir()
        for sequence in ("DJI_0001_1", "DJI_0002_1"):
            frames = [{"frame": f, "detections": [[0, 0, 4, 4, .9, 0], [10, 0, 14, 4, .9, 0], [20, 0, 24, 4, .9, 0]]}
                      for f in range(1, 5)]
            path = cache / (sequence + ".json")
            dump_json(path, {"model_sha256": "d"*64, "frames": frames})
            feat = embeddings / (sequence + ".npz")
            np.savez_compressed(feat, features=np.tile(np.eye(3, dtype=np.float32), (4, 1)),
                                indices=np.tile(np.arange(3), 4), offsets=np.arange(0, 13, 3), frame_ids=np.arange(1, 5))
            dump_json(feat.with_suffix(".json"), {"detection_sha256": sha256(path), "features_sha256": sha256(feat), "reid_sha256": "a"*64})
            truth = gt / sequence / "gt/gt.txt"
            truth.parent.mkdir(parents=True)
            # Third identity belongs to a different class and must not become a
            # trivially easy negative for either of the first two identities.
            truth.write_text("".join(f"{f},{i},{x},0,4,4,1,{cls},1\n" for f in range(1, 5)
                                     for i, x, cls in ((1, 0, 1), (2, 10, 1), (3, 20, 2))))
        split = {"schema_version": 1, "interpretation": "retrospective_development_only",
                 "calibration": ["DJI_0001_1"], "evaluation": ["DJI_0002_1"]}
        return cache, embeddings, gt, split

    def test_exact_feature_hash_dtype_and_index_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, embeddings, _, _ = self._fixture(root)
            path = cache / "DJI_0001_1.json"
            frames = json.loads(path.read_text())["frames"]
            features, report = load_replay_features(path, frames, embeddings, "DJI_0001_1")
            self.assertEqual(report["dimension"], 3)
            self.assertEqual(features[1][0].dtype, np.float32)
            metadata = embeddings / "DJI_0001_1.json"
            value = json.loads(metadata.read_text()); value["detection_sha256"] = "f"*64
            dump_json(metadata, value)
            with self.assertRaisesRegex(ValueError, "unavailable"):
                load_replay_features(path, frames, embeddings, "DJI_0001_1")

    def test_calibration_has_temporal_same_class_pairs_and_validated_holdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, embeddings, gt, split = self._fixture(root)
            result = calibrate_appearance(cache, gt, embeddings, split, root / "fitted", min_same=2, min_different=2)
            self.assertEqual(result["global"]["different_pairs"], 6)
            self.assertEqual(result["global"]["same_pairs"], 9)
            self.assertEqual(result["by_gap"]["1-3"]["fallback"], "global_insufficient_bin_samples")
            self.assertEqual(result["by_gap"]["1-3"]["threshold"], result["global"]["threshold"])
            self.assertEqual(validate_appearance_model(result, split, cache, embeddings, split["evaluation"])["status"], "PASS")
            with self.assertRaisesRegex(ValueError, "disjoint"):
                validate_appearance_model(result, split, cache, embeddings, split["calibration"])
            metadata = embeddings / "DJI_0002_1.json"
            value = json.loads(metadata.read_text()); value["reid_sha256"] = "b"*64
            dump_json(metadata, value)
            with self.assertRaisesRegex(ValueError, "Replay encoder"):
                validate_appearance_model(result, split, cache, embeddings, split["evaluation"])


@unittest.skipUnless(os.environ.get("M2_TEST_BOXMOT") == "1", "Set M2_TEST_BOXMOT=1 with the frozen BoxMOT runtime")
class AppearanceTrackerTests(unittest.TestCase):
    CONFIG = dict(det_thresh=.5, track_high_thresh=.5, track_low_thresh=.1, new_track_thresh=.6,
                  match_thresh=.8, track_buffer=60, frame_rate=15, with_reid=False, cmc_method="sof",
                  proximity_thresh=.5, appearance_thresh=.4, fuse_first_associate=False)
    IMAGE = np.zeros((100, 100, 3), np.uint8)

    @classmethod
    def setUpClass(cls):
        from botsort_research.runner import configure_runtime
        configure_runtime()

    def tracker(self, variant="reid_recovery", **overrides):
        from botsort_research.appearance_adapter import build_appearance_tracker
        tracker = build_appearance_tracker(variant, {**self.CONFIG, **overrides}, calibration=model())
        class Identity:
            def apply(self, image, dets):
                return np.eye(2, 3)
        tracker.cmc = Identity()
        return tracker

    def update(self, tracker, rows=(), features=()):
        return tracker.update(np.asarray(rows, np.float32).reshape(-1, 6), self.IMAGE,
                              embs=np.asarray(features, np.float32).reshape(-1, 2))

    def test_native_cached_wrapper_preserves_exact_native_outputs(self):
        import torch
        from boxmot.trackers.botsort.botsort import BotSort
        detections = [[], [[10, 10, 18, 18, .9, 0]], [[10.2, 10, 18.2, 18, .9, 0]],
                      [[10.4, 10, 18.4, 18, .4, 0]], [], [[10.8, 10, 18.8, 18, .9, 0]]]
        features = [[], [[.8, .6]], [[.9, .4]], [[0, 0]], [], [[1, 0]]]
        native = BotSort(reid_weights=Path("unused.pt"), device=torch.device("cpu"), half=False, **self.CONFIG)
        native.with_reid = True
        native.cmc = self.tracker("native_reid").cmc
        expected = [self.update(native, det, emb).copy() for det, emb in zip(detections, features)]
        wrapped = self.tracker("native_reid")
        actual = [self.update(wrapped, det, emb).copy() for det, emb in zip(detections, features)]
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)

    def test_lost_recovery_keeps_id_without_iou_and_freezes_its_evidence(self):
        tracker = self.tracker()
        a = self.update(tracker, [[10, 10, 18, 18, .9, 0]], [[1, 0]])
        self.update(tracker)
        feature = [.95, np.sqrt(1-.95**2)]
        recovered = self.update(tracker, [[30, 10, 38, 18, .9, 0]], [feature])
        self.assertEqual(int(recovered[0, 4]), int(a[0, 4]))
        self.assertEqual(len(tracker.active_tracks), 1)
        np.testing.assert_array_equal(tracker.active_tracks[0].trusted_feature, [1, 0])
        self.assertEqual(tracker.active_tracks[0].trusted_feature_frame, 1)

    def test_low_score_does_not_birth_or_recover_lost_track(self):
        tracker = self.tracker()
        self.assertEqual(self.update(tracker, [[10, 10, 18, 18, .4, 0]], [[0, 0]]).size, 0)
        self.update(tracker, [[10, 10, 18, 18, .9, 0]], [[1, 0]])
        self.update(tracker, [[10, 10, 18, 18, .9, 0]], [[1, 0]])
        self.update(tracker)
        self.assertEqual(self.update(tracker, [[10, 10, 18, 18, .4, 0]], [[0, 0]]).size, 0)
        self.assertEqual(len(tracker.lost_stracks), 1)

    def test_active_and_unconfirmed_observations_are_not_stolen_by_lost(self):
        tracker = self.tracker()
        self.update(tracker, [[10, 10, 18, 18, .9, 0], [50, 10, 58, 18, .9, 0]], [[1, 0], [0, 1]])
        self.update(tracker, [[10, 10, 18, 18, .9, 0]], [[1, 0]])
        result = self.update(tracker, [[10, 10, 18, 18, .9, 0]], [[0, 1]])
        self.assertEqual(result[:, 4].tolist(), [1])
        self.assertEqual([t.id for t in tracker.lost_stracks], [2])
        tracker = self.tracker()
        self.update(tracker, [[10, 10, 18, 18, .9, 0]], [[1, 0]])
        self.update(tracker)
        self.assertEqual(self.update(tracker, [[40, 10, 48, 18, .9, 0]], [[0, 1]]).size, 0)
        result = self.update(tracker, [[40, 10, 48, 18, .9, 0]], [[1, 0]])
        self.assertEqual(result[:, 4].tolist(), [2])

    def test_expired_track_and_ambiguous_candidates_are_not_recovered(self):
        tracker = self.tracker(track_buffer=4)
        self.update(tracker, [[10, 10, 18, 18, .9, 0]], [[1, 0]])
        self.update(tracker); self.update(tracker)
        result = self.update(tracker, [[30, 10, 38, 18, .9, 0]], [[1, 0]])
        self.assertEqual(result.size, 0)
        self.assertTrue(any(t.id == 1 for t in tracker.removed_stracks))
        tracker = self.tracker()
        self.update(tracker, [[10, 10, 18, 18, .9, 0]], [[1, 0]])
        self.update(tracker)
        self.assertEqual(self.update(tracker, [[30, 10, 38, 18, .9, 0], [50, 10, 58, 18, .9, 0]], [[1, 0], [1, 0]]).size, 0)
        self.assertEqual([t.id for t in tracker.lost_stracks], [1])

    def test_missing_high_feature_fails_before_state_advance(self):
        tracker = self.tracker()
        with self.assertRaisesRegex(ValueError, "nonzero"):
            self.update(tracker, [[10, 10, 18, 18, .9, 0]], [[0, 0]])
        self.assertEqual(tracker.frame_count, 0)
        with self.assertRaisesRegex(ValueError, "embedding array"):
            tracker.update(np.empty((0, 6), np.float32), self.IMAGE)


if __name__ == "__main__":
    unittest.main()
