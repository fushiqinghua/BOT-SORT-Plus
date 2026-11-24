"""Counterexamples for geometry search gates and assignment rejection."""
import unittest

import numpy as np

from botsort_plus.assignment import assign_with_rejection
from botsort_plus.geometry import GeometryConfig, box_iou, geometry_cost, warp_box


class GeometryCounterexamples(unittest.TestCase):
    def setUp(self):
        self.cfg = GeometryConfig()
        self.a = np.array([10.0, 10.0, 14.0, 14.0])

    def test_tiny_one_pixel_diagonal_shift_has_low_iou_but_finite_cost(self):
        b = self.a + np.array([1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(box_iou(self.a, b), 9.0 / 23.0)
        cost = geometry_cost(self.a, b, self.cfg)
        self.assertTrue(np.isfinite(cost))
        self.assertLess(cost, 1.0 - box_iou(self.a, b))

    def test_more_uncertainty_cannot_make_an_existing_match_cheaper(self):
        b = self.a + np.array([2.0, 1.0, 2.0, 1.0])
        costs = [geometry_cost(self.a, b, self.cfg, center_cov=np.eye(2) * v)
                 for v in (0.0, 0.1, 10.0, 1e6)]
        np.testing.assert_array_equal(costs, np.repeat(costs[0], 4))

    def test_uncertainty_can_expand_search_but_only_up_to_pixel_cap(self):
        outside_base = self.a + np.array([9.0, 0.0, 9.0, 0.0])
        self.assertTrue(np.isinf(geometry_cost(self.a, outside_base, self.cfg)))
        self.assertTrue(np.isfinite(geometry_cost(self.a, outside_base, self.cfg,
                                                 center_cov=np.eye(2) * 25.0)))
        outside_cap = self.a + np.array([49.0, 0.0, 49.0, 0.0])
        self.assertTrue(np.isinf(geometry_cost(self.a, outside_cap, self.cfg,
                                              center_cov=np.eye(2) * 1e12)))

    def test_size_gate_and_pixel_floor_are_explicit(self):
        oversized = np.array([6.0, 10.0, 18.0, 14.0])
        self.assertTrue(np.isinf(geometry_cost(self.a, oversized, self.cfg)))
        tiny = np.array([10.0, 10.0, 11.0, 11.0])
        displaced = tiny + np.array([1.0, 0.0, 1.0, 0.0])
        self.assertTrue(np.isfinite(geometry_cost(tiny, displaced, self.cfg)))

    def test_covariance_contract_rejects_non_psd_and_nonfinite(self):
        for cov in (np.diag([-1.0, 1.0]), np.eye(3), np.array([[np.nan, 0.0], [0.0, 1.0]])):
            with self.assertRaises(ValueError):
                geometry_cost(self.a, self.a, self.cfg, center_cov=cov)

    def test_rotation_uses_all_box_corners(self):
        warp = np.array([[0.0, -1.0, 100.0], [1.0, 0.0, 0.0]])
        np.testing.assert_allclose(warp_box(np.array([10.0, 20.0, 14.0, 26.0]), warp),
                                   [74.0, 10.0, 80.0, 14.0])


class AssignmentCounterexamples(unittest.TestCase):
    def test_reject_is_optimized_not_post_hoc_filtering(self):
        # Forced 2x2 Hungarian would use (row 0,col 1), then drop its 0.4
        # over-threshold edge, leaving the worse 0.2 pair. Explicit dummy
        # rejection chooses the legitimate 0.1 pair from the beginning.
        pairs, rows, cols = assign_with_rejection([[0.1, 0.4], [0.2, 0.7]], 0.3)
        self.assertEqual(pairs, [(0, 0)])
        self.assertEqual(rows, [1])
        self.assertEqual(cols, [1])

    def test_global_assignment_can_preserve_two_nonconflicting_matches(self):
        pairs, rows, cols = assign_with_rejection([[0.11, 0.2], [0.12, np.inf]], 0.8)
        self.assertEqual(set(pairs), {(0, 1), (1, 0)})
        self.assertEqual(rows, [])
        self.assertEqual(cols, [])

    def test_mutual_margin_rejects_row_tie_and_column_tie(self):
        pairs, _, _ = assign_with_rejection([[0.1, 0.1]], 0.5, margin=0.01)
        self.assertEqual(pairs, [])
        pairs, _, _ = assign_with_rejection([[0.1], [0.1]], 0.5, margin=0.01)
        self.assertEqual(pairs, [])

    def test_margin_is_measured_before_rival_deletion(self):
        pairs, _, _ = assign_with_rejection([[0.10, 0.12], [0.11, 0.8]], 0.9, margin=0.05)
        self.assertEqual(pairs, [])

    def test_all_forbidden_rectangular_and_empty_inputs(self):
        self.assertEqual(assign_with_rejection(np.full((2, 3), np.inf), 0.8),
                         ([], [0, 1], [0, 1, 2]))
        self.assertEqual(assign_with_rejection(np.empty((0, 3)), 0.8), ([], [], [0, 1, 2]))
        self.assertEqual(assign_with_rejection(np.empty((2, 0)), 0.8), ([], [0, 1], []))

    def test_exact_acceptance_boundary_is_rejected(self):
        self.assertEqual(assign_with_rejection([[0.3]], 0.3), ([], [0], [0]))

    def test_invalid_cost_or_margin_rejected(self):
        for cost in ([[np.nan]], [[-np.inf]], [[-0.1]], [0.1]):
            with self.assertRaises(ValueError):
                assign_with_rejection(cost, 0.3)
        for margin in (-0.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                assign_with_rejection([[0.1]], 0.3, margin)


if __name__ == "__main__":
    unittest.main()
