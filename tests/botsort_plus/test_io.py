"""Integrity, alignment, modality and no-GT boundaries of cached replay."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from botsort_plus.io import (
    FeatureCache, detection_fingerprint, forbid_ground_truth, read_detection_cache,
    resolve_image, sha256, write_json,
)


class CacheIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "sample.npz"
        self.cache = {
            "model_sha256": "d" * 64,
            "frames": [
                {"frame": 1, "image": "test2017/s/00/1.jpg", "detections": []},
                {"frame": 2, "image": "test2017/s/00/2.jpg", "detections": [
                    [1, 1, 9, 9, .9, 0], [2, 2, 10, 10, .3, 1]]},
                {"frame": 3, "image": "test2017/s/00/3.jpg", "detections": [
                    [3, 3, 11, 11, .2, 0]]},
            ],
        }
        self.arrays = {"frame_ids": np.array([1, 2, 3]), "offsets": np.array([0, 0, 2, 3]),
                       "rgb": np.array([[1., 0], [0, 1.], [0, 0]], np.float32),
                       "rgb_valid": np.array([True, True, False]),
                       "rgb_quality": np.array([.8, .4, 0], np.float32)}
        self.meta = {"schema": "botsort_plus_features_v1", "encoders": {"rgb": {"sha256": "a" * 64}},
                     "spatial_mapping": None}

    def save(self):
        np.savez_compressed(self.path, **self.arrays)
        write_json(self.path.with_suffix(".json"), {**self.meta, "features_sha256": sha256(self.path),
                   "detection_fingerprint": detection_fingerprint(self.cache)})

    def add_thermal(self):
        self.arrays.update(thermal=np.array([[1., 0, 0], [0, 0, 1.], [0, 0, 0]], np.float32),
                           thermal_valid=np.array([True, True, False]),
                           thermal_quality=np.array([.9, .7, 0], np.float32))
        self.meta["encoders"]["thermal"] = {"sha256": "b" * 64}
        self.meta["spatial_mapping"] = {"mode": "aligned", "declared_by_user": True}

    def test_frame_packets_follow_exact_offsets_including_empty_frame(self):
        self.save()
        reader = FeatureCache(self.path, self.cache, expected_rgb_sha="a" * 64)
        self.assertEqual(reader.frame(0), [])
        frame = reader.frame(1)
        np.testing.assert_array_equal(frame[0]["rgb"].vector, [1, 0])
        np.testing.assert_array_equal(frame[1]["rgb"].vector, [0, 1])
        self.assertEqual(reader.frame(2), [{}])

    def test_checksum_tampering_rejected(self):
        self.save()
        with self.path.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            FeatureCache(self.path, self.cache)

    def test_detector_row_permutation_rejected(self):
        self.save()
        permuted = copy.deepcopy(self.cache)
        permuted["frames"][1]["detections"].reverse()
        with self.assertRaisesRegex(ValueError, "ordered detector"):
            FeatureCache(self.path, permuted)

    def test_checkpoint_mismatch_rejected(self):
        self.save()
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            FeatureCache(self.path, self.cache, expected_rgb_sha="c" * 64)

    def test_wrong_offsets_and_frames_rejected(self):
        self.arrays["offsets"] = np.array([0, 1, 2, 3])
        self.save()
        with self.assertRaisesRegex(ValueError, "offsets"):
            FeatureCache(self.path, self.cache)
        self.arrays["offsets"] = np.array([0, 0, 2, 3])
        self.arrays["frame_ids"] = np.array([2, 3, 4])
        self.save()
        with self.assertRaisesRegex(ValueError, "offsets"):
            FeatureCache(self.path, self.cache)

    def test_nan_feature_and_quality_rejected(self):
        self.arrays["rgb"][0, 0] = np.nan
        self.save()
        with self.assertRaisesRegex(ValueError, "feature matrix"):
            FeatureCache(self.path, self.cache)
        self.arrays["rgb"][0, 0] = 1
        self.arrays["rgb_quality"][0] = np.nan
        self.save()
        with self.assertRaisesRegex(ValueError, "quality"):
            FeatureCache(self.path, self.cache)

    def test_zero_placeholder_marked_valid_rejected(self):
        self.arrays["rgb_valid"][2] = True
        self.save()
        with self.assertRaisesRegex(ValueError, "Zero placeholder"):
            FeatureCache(self.path, self.cache)

    def test_nonunit_or_huge_finite_features_are_rejected(self):
        for value in (2., 1e30):
            self.arrays["rgb"][0] = [value, 0]
            self.save()
            with self.assertRaisesRegex(ValueError, "unit normalized"):
                FeatureCache(self.path, self.cache)

    def test_declared_embedding_dimension_is_checked(self):
        self.meta["encoders"]["rgb"]["output_dim"] = 1280
        self.save()
        with self.assertRaisesRegex(ValueError, "dimension"):
            FeatureCache(self.path, self.cache)

    def test_c_requires_real_thermal_and_explicit_mapping(self):
        self.save()
        with self.assertRaisesRegex(ValueError, "real thermal"):
            FeatureCache(self.path, self.cache, require_thermal=True)
        self.add_thermal()
        self.save()
        reader = FeatureCache(self.path, self.cache, require_thermal=True)
        self.assertEqual(set(reader.frame(1)[0]), {"rgb", "thermal"})
        self.meta["spatial_mapping"] = None
        self.save()
        with self.assertRaises(ValueError):
            FeatureCache(self.path, self.cache, require_thermal=True)

    def test_c_rejects_all_invalid_thermal_and_missing_provenance(self):
        self.add_thermal()
        self.arrays["thermal_valid"][:] = False
        self.save()
        with self.assertRaisesRegex(ValueError, "real thermal"):
            FeatureCache(self.path, self.cache, require_thermal=True)
        self.arrays["thermal_valid"][0] = True
        self.meta["encoders"].pop("thermal")
        self.save()
        with self.assertRaisesRegex(ValueError, "provenance"):
            FeatureCache(self.path, self.cache, require_thermal=True)

    def test_json_and_gzip_share_content_fingerprint(self):
        path = self.root / "detections.json"
        path.write_text(json.dumps(self.cache))
        compressed = self.root / "detections.json.gz"
        with gzip.open(compressed, "wt") as stream:
            json.dump(self.cache, stream)
        self.assertEqual(detection_fingerprint(read_detection_cache(path)),
                         detection_fingerprint(read_detection_cache(compressed)))
        self.assertNotEqual(sha256(path), sha256(compressed))

    def test_changed_detector_or_class_changes_fingerprint(self):
        changed = copy.deepcopy(self.cache)
        changed["model_sha256"] = "e" * 64
        self.assertNotEqual(detection_fingerprint(changed), detection_fingerprint(self.cache))
        changed = copy.deepcopy(self.cache)
        changed["frames"][1]["detections"][1][5] = 0
        self.assertNotEqual(detection_fingerprint(changed), detection_fingerprint(self.cache))

    def test_missing_image_has_no_synthetic_fallback(self):
        with self.assertRaises(FileNotFoundError):
            resolve_image("test2017/missing.jpg", self.root)

    def test_ground_truth_guard_blocks_strings_bytes_and_paths(self):
        for path in ("/data/s/gt/gt.txt", b"/data/s/gt/gt.txt", Path("/data/annotations/anything.json"),
                     "/data/instances_test2017.json"):
            with self.subTest(path=path), self.assertRaises(PermissionError):
                forbid_ground_truth("open", (path, "r", 0))
        forbid_ground_truth("open", ("/data/test2017/s/00/1.jpg", "r", 0))


if __name__ == "__main__":
    unittest.main()
