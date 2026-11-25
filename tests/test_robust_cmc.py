"""Known image motions and adversarial support validate M1's meaning."""
import importlib.util
import json
import unittest

import numpy as np

from botsort_research.robust_cmc import (RobustCMC, SUPPORTED_CMC_MODES,
                                        build_cmc, scaled_to_native)


class CoordinateTests(unittest.TestCase):
    def test_actual_resize_dimensions_and_pixel_centers(self):
        native_shape, small_shape = (513, 641), (77, 96)
        native = np.array([[0., 0.], [640., 512.], [230.25, 119.75]])
        scale = np.array([96 / 641, 77 / 513])
        small = (native + .5) * scale - .5
        np.testing.assert_allclose(scaled_to_native(small, native_shape, small_shape), native, atol=1e-12)


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required for M1")
class PointReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([(x, y) for y in np.linspace(35, 450, 6)
                                for x in np.linspace(35, 590, 6)])

    def test_true_rotation_scale_and_large_translation_are_supported(self):
        angle = np.deg2rad(3.)
        matrix = 1.03 * np.array([[np.cos(angle), -np.sin(angle)],
                                  [np.sin(angle), np.cos(angle)]])
        following = self.points @ matrix.T + [28., -12.]
        for mode in SUPPORTED_CMC_MODES:
            with self.subTest(mode=mode):
                cmc = build_cmc(mode)
                warp, report = cmc.estimate_points(self.points, following, (512, 640))
                self.assertTrue(report["estimated"])
                np.testing.assert_allclose(warp[:, :2], matrix, atol=1e-6)
                np.testing.assert_allclose(warp[:, 2], [28., -12.], atol=2e-5)
                self.assertEqual(cmc.frame, 0)  # geometry QA cannot mutate replay history

    def test_reliability_does_not_validate_on_its_own_fit_points(self):
        following = self.points + [3., -2.]
        following[::5] += [20., 25.]  # all held-out correspondences contradict fit
        warp, report = RobustCMC().estimate_points(self.points, following, (512, 640))
        self.assertFalse(report["estimated"])
        self.assertEqual(report["status"], "unreliable_identity_fallback")
        self.assertIn("heldout_residual_failed", report["affine_rejection_reasons"])
        self.assertIn("heldout_residual_failed", report["translation_rejection_reasons"])
        np.testing.assert_array_equal(warp, np.eye(2, 3))

    def test_collinear_support_never_claims_reliable_camera_motion(self):
        points = np.column_stack([np.linspace(50, 590, 50), np.full(50, 100.)])
        warp, report = RobustCMC().estimate_points(points, points + [3., 2.], (512, 640))
        self.assertFalse(report["estimated"])
        self.assertIn("ill_conditioned_spatial_support", report["reasons"])
        np.testing.assert_array_equal(warp, np.eye(2, 3))
        json.dumps(report, allow_nan=False)

    def test_supported_translation_fallback_has_independent_checks(self):
        # The configured scale bound deliberately excludes this tiny dilation;
        # translation is still an adequate fit within the 3 native pixel gate.
        cmc = RobustCMC(config={"min_affine_scale": 1., "max_affine_scale": 1.})
        following = self.points * 1.003 + [3., -2.]
        warp, report = cmc.estimate_points(self.points, following, (512, 640))
        self.assertEqual(report["model"], "translation")
        self.assertTrue(report["estimated"])
        self.assertIn("implausible_scale", report["affine_rejection_reasons"])
        np.testing.assert_array_equal(warp[:, :2], np.eye(2))
        self.assertLess(report["holdout_median_native_px"], 3.)

    def test_insufficient_holdout_is_explicit(self):
        points = self.points[:10]
        _, report = RobustCMC().estimate_points(points, points + [1., 1.], (512, 640))
        self.assertFalse(report["estimated"])
        self.assertIn("insufficient_holdout_support", report["reasons"])

    def test_parameters_and_points_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "Unknown M1"):
            RobustCMC(config={"unknown": 1})
        with self.assertRaises(ValueError):
            RobustCMC(config={"scale": 0})
        with self.assertRaises(ValueError):
            RobustCMC().estimate_points([[np.nan, 2]], [[1, 2]], (512, 640))


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required for M1")
class ImageMotionTests(unittest.TestCase):
    def setUp(self):
        import cv2
        self.cv2 = cv2
        self.image = np.random.default_rng(71).integers(0, 256, (513, 641, 3), dtype=np.uint8)

    def shifted(self, image, x, y):
        return self.cv2.warpAffine(image, np.array([[1., 0., x], [0., 1., y]]),
                                   (image.shape[1], image.shape[0]), borderMode=self.cv2.BORDER_REFLECT)

    def test_translation_native_units_for_all_staged_modes(self):
        following = self.shifted(self.image, 4., -3.)
        for mode in SUPPORTED_CMC_MODES:
            with self.subTest(mode=mode):
                events = []
                cmc = build_cmc(mode, {"scale": .5}, events.append)
                np.testing.assert_array_equal(cmc.apply(self.image), np.eye(2, 3))
                warp = cmc.apply(following, np.empty((0, 6)))
                self.assertTrue(cmc.last_report["estimated"])
                np.testing.assert_allclose(warp[:, 2], [4., -3.], atol=.5)
                self.assertEqual([e["frame"] for e in events], [1, 2])
                self.assertEqual(events[0]["status"], "initialized_no_pair_estimate")
                self.assertEqual(cmc.summary()["frames"], 2)
                json.dumps(events, allow_nan=False)

    def test_rotation_from_images_maps_previous_to_current(self):
        matrix = self.cv2.getRotationMatrix2D((320., 256.), 1.5, 1.)
        following = self.cv2.warpAffine(self.image, matrix, (641, 513), borderMode=self.cv2.BORDER_REFLECT)
        cmc = RobustCMC(config={"scale": .5})
        cmc.apply(self.image)
        actual = cmc.apply(following)
        self.assertTrue(cmc.last_report["estimated"])
        sample = np.array([[100., 100.], [540., 400.], [320., 256.]])
        np.testing.assert_allclose(sample @ actual[:, :2].T + actual[:, 2],
                                   sample @ matrix[:, :2].T + matrix[:, 2], atol=.7)

    def test_foreground_mask_recovers_background_with_independent_moving_object(self):
        rng = np.random.default_rng(92)
        background = rng.integers(0, 120, (512, 640, 3), dtype=np.uint8)
        foreground = rng.integers(0, 256, (340, 320, 3), dtype=np.uint8)
        previous = background.copy()
        previous[90:430, 160:480] = foreground
        following = self.shifted(background, 4., -3.)
        following[100:440, 180:500] = foreground
        cmc = RobustCMC(config={"scale": 1.})
        cmc.apply(previous, np.array([[160., 90., 480., 430., .9, 0]]))
        points_native = scaled_to_native(cmc.points, (512, 640), cmc.previous.shape)
        inside_object = ((points_native[:, 0] >= 157) & (points_native[:, 0] < 483) &
                         (points_native[:, 1] >= 87) & (points_native[:, 1] < 433))
        self.assertFalse(inside_object.any())
        actual = cmc.apply(following, np.array([[180., 100., 500., 440., .9, 0]]))
        self.assertTrue(cmc.last_report["estimated"])
        np.testing.assert_allclose(actual[:, 2], [4., -3.], atol=.8)

    def test_blank_frames_and_native_shape_changes_have_flagged_identity(self):
        cmc = RobustCMC()
        for _ in range(2):
            actual = cmc.apply(np.zeros((512, 640, 3), dtype=np.uint8))
            np.testing.assert_array_equal(actual, np.eye(2, 3))
            self.assertFalse(cmc.last_report["estimated"])
            self.assertEqual(cmc.last_report["status"], "insufficient_initial_features")
        # Both native shapes round to the same .15 resized dimensions. Native
        # shape still must reset the pair because its coordinate mapping changed.
        cmc.apply(self.image)
        self.assertEqual(cmc.last_report["status"], "reset_image_shape_changed")
        self.assertFalse(cmc.last_report["estimated"])

    def test_default_scale_translation_and_empty_detections(self):
        cmc = RobustCMC()
        cmc.apply(self.image, np.empty((0, 7)))
        actual = cmc.apply(self.shifted(self.image, 6., -6.), np.empty((0, 7)))
        self.assertTrue(cmc.last_report["estimated"])
        np.testing.assert_allclose(actual[:, 2], [6., -6.], atol=1.5)

    def test_fully_masked_frame_does_not_initialize_foreground_as_background(self):
        cmc = RobustCMC()
        actual = cmc.apply(self.image, np.array([[0, 0, 641, 513, .95, 0]]))
        np.testing.assert_array_equal(actual, np.eye(2, 3))
        self.assertFalse(cmc.last_report["estimated"])
        self.assertEqual(cmc.last_report["keypoints"], 0)


if __name__ == "__main__":
    unittest.main()
