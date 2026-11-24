"""Mechanism tests for C; these are not VT-Tiny-MOT benefit measurements."""
import unittest

import numpy as np

from botsort_plus.appearance import (
    AppearanceConfig, MemoryBank, appearance_details, appearance_distance, estimate_quality,
)
from botsort_plus.types import Feature


def feature(vector, quality=1.0):
    return Feature(np.array(vector, dtype=float), quality)


class AppearanceTests(unittest.TestCase):
    def setUp(self):
        self.config = AppearanceConfig()
        self.memory = MemoryBank(self.config)
        # Modalities intentionally have different dimensions.
        self.memory.update({"rgb": feature([1, 0]), "thermal": feature([1, 0, 0])})

    def test_separate_spaces_support_different_dimensions(self):
        details = appearance_details(self.memory, {"rgb": feature([1, 0]), "thermal": feature([0, 1, 0])})
        self.assertAlmostEqual(details["distance"], 0.5)
        self.assertEqual(details["used_modalities"], ["rgb", "thermal"])
        self.assertAlmostEqual(details["modalities"]["thermal"]["raw_distance"], 1)

    def test_quality_shift_changes_match_preference(self):
        rgb_reliable = {"rgb": feature([1, 0], 1), "thermal": feature([0, 1, 0], 0.2)}
        thermal_reliable = {"rgb": feature([1, 0], 0.2), "thermal": feature([0, 1, 0], 1)}
        self.assertLess(appearance_distance(self.memory, rgb_reliable),
                        appearance_distance(self.memory, thermal_reliable))
        self.assertAlmostEqual(appearance_distance(self.memory, rgb_reliable), 1 / 6)

    def test_fixed_fusion_ablation_only_changes_weights(self):
        features = {"rgb": feature([1, 0], 1), "thermal": feature([0, 1, 0], 0.2)}
        fixed = AppearanceConfig(weighting="fixed")
        self.assertAlmostEqual(appearance_distance(self.memory, features, fixed), 0.5)
        self.assertAlmostEqual(appearance_distance(self.memory, features, self.config), 1 / 6)
        # The same low-quality modality is unavailable in both variants.
        features["thermal"].quality = 0.05
        self.assertAlmostEqual(appearance_distance(self.memory, features, fixed), 0)
        self.assertAlmostEqual(appearance_distance(self.memory, features, self.config), 0)

    def test_rgb_only_quality_control_does_not_require_thermal(self):
        cfg = AppearanceConfig(enabled=True, weight_thermal=0)
        self.assertAlmostEqual(appearance_distance(self.memory, {"rgb": feature([0, 1])}, cfg), 1)

    def test_missing_thermal_falls_back_to_rgb(self):
        self.assertAlmostEqual(appearance_distance(self.memory, {"rgb": feature([0, 1])}), 1)

    def test_zero_vectors_are_missing_not_perfect_similarity(self):
        d = appearance_details(self.memory, {"rgb": feature([0, 0]), "thermal": feature([0, 0, 0])})
        self.assertIsNone(d["distance"])
        self.assertFalse(d["within_gate"])
        self.assertEqual(d["modalities"]["rgb"]["reason"], "zero_placeholder")
        fresh = MemoryBank()
        self.assertEqual(fresh.update({"rgb": feature([0, 0])}), [])
        self.assertEqual(fresh.prototypes, {})

    def test_zero_quality_even_with_identical_vectors_is_unavailable(self):
        self.assertIsNone(appearance_distance(self.memory, {"rgb": feature([1, 0], 0),
                                                           "thermal": feature([1, 0, 0], 0)}))

    def test_low_quality_observation_does_not_poison_prototype(self):
        before = self.memory.as_features()
        self.assertEqual(self.memory.update({"rgb": feature([0, 1], 0.2)}), [])
        np.testing.assert_array_equal(self.memory.prototypes["rgb"], before["rgb"].vector)
        self.assertEqual(self.memory.qualities["rgb"], before["rgb"].quality)

    def test_read_before_write_association_is_pure(self):
        proposal = {"rgb": feature([0, 1])}
        before = self.memory.prototypes["rgb"].copy()
        before_updates = self.memory.update_counts.copy()
        first = appearance_distance(self.memory, proposal)
        second = appearance_distance(self.memory, proposal)
        self.assertEqual(first, second)
        np.testing.assert_array_equal(self.memory.prototypes["rgb"], before)
        self.assertEqual(self.memory.update_counts, before_updates)
        # Only this explicit acceptance changes memory.
        self.memory.update(proposal)
        self.assertLess(appearance_distance(self.memory, proposal), first)

    def test_dimension_change_is_not_cross_modal_comparison(self):
        with self.assertRaisesRegex(ValueError, "rgb embedding dimension changed"):
            appearance_distance(self.memory, {"rgb": feature([1, 0, 0])})
        with self.assertRaisesRegex(ValueError, "thermal embedding dimension changed"):
            self.memory.update({"thermal": feature([1, 0])})

    def test_invalid_packet_does_not_partially_update_memory(self):
        before = self.memory.prototypes["rgb"].copy()
        with self.assertRaises(ValueError):
            self.memory.update({"rgb": feature([0, 1]), "thermal": feature([1, float("nan"), 0])})
        np.testing.assert_array_equal(self.memory.prototypes["rgb"], before)
        with self.assertRaises(ValueError):
            appearance_distance(self.memory, {"rgb": feature([[1, 0]])})
        with self.assertRaises(ValueError):
            appearance_distance(self.memory, {"rgb": feature([1, 0], 1.1)})

    def test_historical_quality_also_limits_weight(self):
        memory = MemoryBank()
        memory.update({"rgb": feature([1, 0], 0.4), "thermal": feature([1, 0, 0], 1)})
        d = appearance_details(memory, {"rgb": feature([0, 1], 1), "thermal": feature([1, 0, 0], 1)})
        self.assertAlmostEqual(d["modalities"]["rgb"]["weight"], 0.4)
        self.assertAlmostEqual(d["distance"], 0.4 / 1.4)

    def test_scale_is_explicit_and_bad_match_is_not_missing(self):
        config = AppearanceConfig(rgb_distance_scale=2, max_distance=0.3)
        d = appearance_details(self.memory, {"rgb": feature([0, 1])}, config)
        self.assertAlmostEqual(d["distance"], 0.5)
        self.assertTrue(d["available"])
        self.assertFalse(d["within_gate"])

    def test_input_cache_is_not_normalized_in_place(self):
        vector = np.array([3.0, 4.0])
        memory = MemoryBank()
        memory.update({"rgb": Feature(vector)})
        appearance_distance(memory, {"rgb": Feature(vector)})
        np.testing.assert_array_equal(vector, [3, 4])

    def test_antipodal_ema_does_not_create_zero_memory(self):
        memory = MemoryBank(AppearanceConfig(ema=0.5))
        memory.update({"rgb": feature([1, 0])})
        self.assertEqual(memory.update({"rgb": feature([-1, 0])}), [])
        np.testing.assert_array_equal(memory.prototypes["rgb"], [1, 0])

    def test_configuration_validation(self):
        for kwargs in ({"ema": 1}, {"weighting": "unknown"}, {"rgb_distance_scale": 0}, {"max_distance": float("nan")},
                       {"weight_rgb": 0, "weight_thermal": 0}, {"min_update_quality": 0.1}):
            with self.assertRaises(ValueError):
                AppearanceConfig(**kwargs)


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.gray = ((np.indices((32, 32)).sum(axis=0) % 2) * 255).astype(np.uint8)
        self.image = np.repeat(self.gray[..., None], 3, axis=2)

    def test_native_tiny_crop_has_less_information_than_large_crop(self):
        small = estimate_quality(self.image, [0, 0, 4, 4])
        large = estimate_quality(self.image, [0, 0, 32, 32])
        self.assertGreater(large, small)
        self.assertLessEqual(small, 0.25)
        self.assertEqual(large, 1.0)

    def test_flat_and_outside_images_have_no_appearance_quality(self):
        self.assertEqual(estimate_quality(np.ones((32, 32, 3), np.uint8) * 127, [0, 0, 32, 32]), 0)
        self.assertEqual(estimate_quality(self.image, [40, 40, 50, 50]), 0)
        self.assertEqual(estimate_quality(self.image, [0, 0, 1, 1]), 0)

    def test_clipped_crop_penalized(self):
        self.assertLess(estimate_quality(self.image, [-16, 0, 16, 32]),
                        estimate_quality(self.image, [0, 0, 16, 32]))

    def test_thermal_uint16_and_normalized_float_are_supported(self):
        thermal = self.gray.astype(np.uint16) * 257
        self.assertEqual(estimate_quality(thermal, [0, 0, 32, 32], "thermal"), 1)
        self.assertEqual(estimate_quality(self.gray.astype(float) / 255, [0, 0, 32, 32], "thermal"), 1)

    def test_bad_intensity_and_box_are_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            estimate_quality(self.image.astype(float) * float("nan"), [0, 0, 32, 32])
        with self.assertRaises(ValueError):
            estimate_quality(self.image, [0, 0, 0, 10])
        with self.assertRaises(ValueError):
            estimate_quality(np.ones((32, 32)) * 300, [0, 0, 32, 32])
        with self.assertRaises(ValueError):
            estimate_quality(self.gray.astype(float), [0, 0, 32, 32])


if __name__ == "__main__":
    unittest.main()
