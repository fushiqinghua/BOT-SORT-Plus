"""Synthetic exporters/encoder fixtures verify mechanisms, not trained accuracy."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from botsort_plus.feature_export import (
    export_sequence, map_boxes_to_thermal, validate_spatial_mapping, TorchScriptCropEncoder,
)


ALIGNED = {"mode": "aligned", "declared_by_user": True}


class FakeEncoder:
    """Test-only deterministic vector generator; never used by the CLI."""
    output_dim = 3

    def __init__(self, zero=False):
        self.boxes = []
        self.zero = zero

    def get_features(self, boxes, image):
        self.boxes.extend(np.asarray(boxes).tolist())
        if self.zero:
            return np.zeros((len(boxes), 3), np.float32)
        return np.column_stack((np.ones(len(boxes)), boxes[:, 0] + 1, boxes[:, 1] + 2))


class SpatialMappingTests(unittest.TestCase):
    def test_alignment_must_be_declared(self):
        for value in (None, {}, {"mode": "aligned"}):
            with self.assertRaises(ValueError):
                validate_spatial_mapping(value)

    def test_four_corner_homography_and_clipping(self):
        mapping = {"mode": "homography", "declared_by_user": True,
                   "rgb_to_thermal": [[0, -1, 20], [1, 0, 0], [0, 0, 1]]}
        mapped, valid = map_boxes_to_thermal([[1, 2, 5, 8], [50, 50, 60, 60]], (32, 32), mapping)
        np.testing.assert_allclose(mapped[0], [12, 1, 18, 5])
        np.testing.assert_array_equal(valid, [True, False])
        np.testing.assert_array_equal(mapped[1], [0, 0, 0, 0])

    def test_horizon_crossing_and_singular_matrix_rejected(self):
        mapping = {"mode": "homography", "declared_by_user": True,
                   "rgb_to_thermal": [[1, 0, 0], [0, 1, 0], [1, 0, -3]]}
        _, valid = map_boxes_to_thermal([[1, 1, 5, 5]], (32, 32), mapping)
        self.assertFalse(valid[0])
        mapping["rgb_to_thermal"] = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
        with self.assertRaises(ValueError):
            validate_spatial_mapping(mapping)

    def test_empty_boxes(self):
        boxes, valid = map_boxes_to_thermal(np.empty((0, 4)), (32, 32), ALIGNED)
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(valid.shape, (0,))

    def test_mapped_clipping_retains_quality_penalty(self):
        boxes, valid, fraction = map_boxes_to_thermal(
            [[-4, 0, 4, 8]], (32, 32), ALIGNED, return_visibility=True)
        np.testing.assert_array_equal(boxes[0], [0, 0, 4, 8])
        self.assertTrue(valid[0])
        self.assertEqual(fraction[0], 0.5)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "features"
        self.cache = self.root / "sequence.json"
        self.image = ((np.indices((32, 32)).sum(axis=0) % 2) * 255).astype(np.uint8)
        self.rgb = np.repeat(self.image[..., None], 3, axis=2)
        (self.root / "rgb.png").touch()
        (self.root / "thermal.png").touch()
        self.frames = [
            {"frame": 1, "image": "rgb.png", "thermal": "thermal.png", "detections": []},
            {"frame": 2, "image": "rgb.png", "thermal": "thermal.png", "detections": [
                [1, 1, 9, 9, 0.05, 0], [2, 2, 10, 10, 0.3, 0],
                [3, 3, 11, 11, 0.9, 0], [4, 4, 12, 12, 0.5, 0],
                [5, 5, 13, 13, 0.1, 0]]},
        ]
        self.metadata = {"sha256": "a" * 64, "output_dim": 3, "test_fixture": True}
        self.write_cache()

    def write_cache(self):
        self.cache.write_text(json.dumps({"model_sha256": "d" * 64,
                                         "inference_wall_seconds": 0, "frames": self.frames}))

    def loader(self, path, modality):
        return self.rgb.copy() if modality == "rgb" else self.image.copy()

    def export(self, encoder=None, **kwargs):
        return export_sequence(self.cache, self.out, encoder or FakeEncoder(), self.metadata,
                               self.root, image_loader=self.loader, **kwargs)

    def test_alignment_low_score_coverage_and_checksum(self):
        encoder = FakeEncoder()
        metadata = self.export(encoder, batch_size=2)
        self.assertEqual(len(encoder.boxes), 3)  # .3, .9, and exact .5, not .05/.1.
        with np.load(self.out / "sequence.npz") as arrays:
            np.testing.assert_array_equal(arrays["offsets"], [0, 0, 5])
            np.testing.assert_array_equal(arrays["frame_ids"], [1, 2])
            np.testing.assert_array_equal(arrays["rgb_valid"], [False, True, True, True, False])
            np.testing.assert_array_equal(arrays["rgb"][[0, 4]], 0)
            np.testing.assert_allclose(np.linalg.norm(arrays["rgb"][[1, 2, 3]], axis=1), 1, atol=1e-6)
        digest = hashlib.sha256((self.out / "sequence.npz").read_bytes()).hexdigest()
        self.assertEqual(metadata["features_sha256"], digest)
        self.assertEqual(metadata["schema"], "botsort_plus_features_v1")

    def test_overwrite_requires_explicit_switch(self):
        self.export()
        with self.assertRaises(FileExistsError):
            self.export()
        self.export(overwrite=True)

    def test_zero_high_score_feature_fails_without_publishing(self):
        with self.assertRaisesRegex(ValueError, "high-score"):
            self.export(FakeEncoder(zero=True))
        self.assertFalse((self.out / "sequence.npz").exists())

    def test_zero_low_score_output_remains_invalid(self):
        self.frames[1]["detections"] = [[1, 1, 9, 9, 0.3, 0]]
        self.write_cache()
        self.export(FakeEncoder(zero=True))
        with np.load(self.out / "sequence.npz") as arrays:
            self.assertFalse(arrays["rgb_valid"][0])
            self.assertEqual(arrays["rgb_quality"][0], 0)

    def test_real_but_flat_crop_is_valid_with_zero_quality(self):
        self.rgb[:] = 0
        self.export()
        with np.load(self.out / "sequence.npz") as arrays:
            self.assertTrue(arrays["rgb_valid"][2])
            self.assertEqual(arrays["rgb_quality"][2], 0)

    def test_rgb_rounded_degenerate_high_box_fails(self):
        self.frames[1]["detections"] = [[1.1, 1.1, 1.2, 1.2, 0.9, 0]]
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "invalid RGB crop"):
            self.export()

    def test_thermal_requires_mapping_and_retains_separate_arrays(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            self.export(thermal_encoder=FakeEncoder(), thermal_metadata=self.metadata)
        result = self.export(thermal_encoder=FakeEncoder(), thermal_metadata=self.metadata, spatial_mapping=ALIGNED)
        self.assertEqual(result["spatial_mapping"], ALIGNED)
        with np.load(self.out / "sequence.npz") as arrays:
            self.assertEqual(arrays["thermal"].shape, (5, 3))
            np.testing.assert_array_equal(arrays["thermal_valid"], arrays["rgb_valid"])

    def test_empty_sequence_has_aligned_empty_features(self):
        self.frames = [self.frames[0]]
        self.write_cache()
        self.export()
        with np.load(self.out / "sequence.npz") as arrays:
            self.assertEqual(arrays["rgb"].shape, (0, 3))
            np.testing.assert_array_equal(arrays["offsets"], [0, 0])


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("cv2"), "Requires torch and OpenCV")
class ThermalTorchScriptTests(unittest.TestCase):
    def test_real_torchscript_execution_and_explicit_preprocessing(self):
        import torch
        class SyntheticEncoder(torch.nn.Module):
            def forward(self, x):
                return torch.stack((x.mean(dim=(1, 2, 3)), x.square().mean(dim=(1, 2, 3)) + 1), dim=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic_thermal_test.pt"
            torch.jit.trace(SyntheticEncoder(), torch.rand(1, 1, 8, 8)).save(str(path))
            encoder = TorchScriptCropEncoder(path, (8, 8), 1, [0.0], [1.0], "cpu")
            image = np.zeros((16, 32), np.uint8)
            image[:, 16:] = 255
            features = encoder.get_features(np.array([[0, 0, 16, 16], [16, 0, 32, 16]]), image)
            self.assertEqual(features.shape, (2, 2))
            np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1, atol=1e-6)
            self.assertGreater(np.linalg.norm(features[0] - features[1]), 0.1)
            self.assertEqual(encoder.metadata["mean"], [0.0])
            with self.assertRaises(ValueError):
                encoder.get_features(np.array([[0, 0, 16, 16]]), image.astype(float) + 300)


if __name__ == "__main__":
    unittest.main()
