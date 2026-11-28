"""Mechanism and historical compatibility tests for the W R3.1 diagnostics."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import random
import sys
import unittest

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.research.wem_r31.engine import METHODS, DEFAULT_PROTOCOL, edge_key, evaluate_frame


def row(track=1, det=0, *, box=None, **updates):
    value = dict(sequence="clip", frame=10, track_id=track, det_index=det,
                 raw_box_xyxy=[det * 20., 0., det * 20. + 10., 10.] if box is None else box,
                 raw_score=0.08, normalized_score=0.6, geometry_group_rank=1,
                 native_iou=0.8, motion=0.8, cosine=0.9, history_score=0.7875,
                 row_unknown=False, column_unknown=False, row_margin=None, column_margin=None,
                 challengers=[])
    value.update(updates)
    return value


def selected(decisions):
    return {edge_key(item) for item in decisions if item["selected"]}


def historical_functions():
    """Execute only the original pure score/assign AST, without importing its
    GPU feature extractor or any local-server orchestration modules.
    """
    source = ROOT / "tools/research/vttiny_wem_r3/admission.py"
    module = ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
    pure = ast.Module(body=[node for node in module.body
                            if isinstance(node, ast.FunctionDef) and node.name in ("score", "assign")],
                      type_ignores=[])
    namespace = {"np": np, "linear_sum_assignment": linear_sum_assignment}
    exec(compile(pure, str(source), "exec"), namespace)
    return namespace["assign"]


class WMechanismTests(unittest.TestCase):
    def test_low_cosine_has_no_absolute_veto_in_w2(self):
        item = row(native_iou=0.78, motion=0.92, cosine=0.55)
        soft = evaluate_frame([item], "W2")[0]
        old = evaluate_frame([item], "W_old")[0]
        self.assertTrue(soft["selected"])
        self.assertAlmostEqual(soft["utility"], 0.835)
        self.assertFalse(old["selected"])
        self.assertIn("old_absolute_cosine", old["reasons"])

    def test_w1_w2_have_identical_eligibility_for_all_cosines(self):
        for cosine in (-1., -0.5, 0., 0.55, 0.99, None):
            for motion in (None, 0.24, 0.25, 0.8):
                with self.subTest(cosine=cosine, motion=motion):
                    item = row(native_iou=0.5, motion=motion, cosine=cosine)
                    a = evaluate_frame([item], "W1")[0]
                    b = evaluate_frame([item], "W2")[0]
                    self.assertEqual(a["eligible"], b["eligible"])
                    if cosine is None and motion is not None:
                        self.assertEqual(a["utility"], b["utility"])
                        self.assertEqual(b["appearance_status"], "missing_fallback")
                    if motion is not None and motion >= 0.25:
                        self.assertTrue(b["selected"])

    def test_w1_does_not_need_rgb_or_old_joint_score(self):
        item = row(cosine=None, history_score=None)
        self.assertTrue(evaluate_frame([item], "W1")[0]["selected"])
        self.assertFalse(evaluate_frame([item], "W_old")[0]["selected"])

    def test_motion_missing_or_below_floor_is_explicit(self):
        for motion, expected in [(None, "motion_unavailable"), (0.24, "motion_floor")]:
            item = row(motion=motion)
            self.assertTrue(evaluate_frame([item], "W0")[0]["selected"])
            for method in ("W1", "W2", "W3"):
                out = evaluate_frame([item], method)[0]
                self.assertFalse(out["eligible"])
                self.assertIn(expected, out["reasons"])

    def test_soft_appearance_can_change_choice_without_removing_edges(self):
        items = [row(1, 0, cosine=-0.8),
                 row(1, 1, native_iou=0.79, motion=0.79, cosine=0.9, geometry_group_rank=2)]
        geometry_motion = evaluate_frame(items, "W1")
        soft = evaluate_frame(items, "W2")
        self.assertEqual([d["eligible"] for d in geometry_motion], [True, True])
        self.assertEqual([d["eligible"] for d in soft], [True, True])
        self.assertEqual(selected(geometry_motion), {edge_key(items[0])})
        self.assertEqual(selected(soft), {edge_key(items[1])})

    def test_w3_already_matched_and_unmatched_neighbors_both_compete(self):
        for status in ("high", "start_tracked_or_recent"):
            item = row(cosine=0.55, challengers=[dict(track_id=2, cosine=0.53, source=status)])
            out = evaluate_frame([item], "W3")[0]
            self.assertFalse(out["selected"])
            self.assertEqual(out["column_competitor_identities"], 1)
            self.assertEqual(out["column_competition_status"], "available")
            self.assertAlmostEqual(out["column_margin"], 0.02)
            self.assertIn("column_relative_margin", out["reasons"])

    def test_w3_relative_advantage_can_pass_below_old_absolute_gate(self):
        item = row(cosine=0.55, challengers=[dict(track_id=2, cosine=0.23), dict(track_id=3, cosine=0.18)])
        out = evaluate_frame([item], "W3")[0]
        self.assertTrue(out["selected"])
        self.assertAlmostEqual(out["column_margin"], 0.32)
        self.assertFalse(evaluate_frame([item], "W_old")[0]["selected"])

    def test_w3_outside_top_k_and_motion_missing_row_still_competes(self):
        items = [row(1, 0, cosine=0.70),
                 row(1, 1, cosine=0.80, motion=None, geometry_group_rank=6)]
        soft = evaluate_frame(items, "W2")
        constrained = evaluate_frame(items, "W3")
        self.assertEqual(selected(soft), {edge_key(items[0])})
        self.assertFalse(selected(constrained))
        self.assertIn("row_relative_margin", constrained[0]["reasons"])
        self.assertEqual(constrained[0]["row_competitor_groups"], 1)
        self.assertAlmostEqual(constrained[0]["row_margin"], -0.1)
        self.assertIn("top_k", constrained[1]["reasons"])

    def test_w3_no_competition_is_not_applicable_even_without_appearance(self):
        item = row(cosine=None)
        out = evaluate_frame([item], "W3")[0]
        self.assertTrue(out["selected"])
        self.assertEqual(out["appearance_status"], "missing_fallback")
        for axis in ("row", "column"):
            self.assertEqual(out[axis + "_competition_status"], "not_applicable")
            self.assertIsNone(out[axis + "_margin"])

    def test_w3_missing_rival_or_own_appearance_is_unavailable(self):
        for own, rival in ((0.8, None), (None, 0.8)):
            item = row(cosine=own, challengers=[dict(track_id=2, cosine=rival)])
            out = evaluate_frame([item], "W3")[0]
            self.assertFalse(out["selected"])
            self.assertEqual(out["column_competition_status"], "unavailable")
            self.assertIsNone(out["column_margin"])
            self.assertIn("column_appearance_unavailable", out["reasons"])

    def test_w3_missing_alternative_group_appearance_is_unavailable(self):
        items = [row(1, 0), row(1, 1, cosine=None, geometry_group_rank=2)]
        out = evaluate_frame(items, "W3")
        self.assertFalse(selected(out))
        self.assertTrue(all(d["row_competition_status"] == "unavailable" for d in out))
        self.assertTrue(all(d["row_margin"] is None for d in out))

    def test_duplicate_coordinates_are_one_group_not_mutual_rivals(self):
        box = [0., 0., 10., 10.]
        items = [row(1, 0, box=box), row(1, 1, box=box, raw_score=0.09, normalized_score=0.8)]
        out = evaluate_frame(items, "W3")
        self.assertEqual(selected(out), {edge_key(items[1])})
        self.assertTrue(all(d["row_competitor_groups"] == 0 for d in out))

    def test_assignment_is_global_and_one_to_one(self):
        items = [row(1, 0, native_iou=0.90), row(1, 1, native_iou=0.80, geometry_group_rank=2),
                 row(2, 0, native_iou=0.89), row(2, 1, native_iou=0.50, geometry_group_rank=2)]
        out = evaluate_frame(items, "W0")
        self.assertEqual(selected(out), {edge_key(items[1]), edge_key(items[2])})

    def test_unmatched_tracks_are_not_given_dummy_measurements(self):
        items = [row(1), row(2), row(3)]
        out = evaluate_frame(items, "W0")
        self.assertEqual(sum(d["selected"] for d in out), 1)
        self.assertEqual(sum(d["decision"] == "unassigned" for d in out), 2)
        self.assertTrue(all(d["eligible"] for d in out))
        rejected = evaluate_frame([row(native_iou=0.49)], "W0")[0]
        self.assertFalse(rejected["selected"])
        self.assertIn("native_geometry", rejected["reasons"])

    def test_no_gt_access_or_input_mutation(self):
        class Forbidden:
            def __getattribute__(self, name):
                raise AssertionError("Offline label evidence must not be touched")
        clean = [row(), row(1, 1, cosine=0.2, geometry_group_rank=2)]
        originals = copy.deepcopy(clean)
        tainted = [dict(r, label=Forbidden(), gt_id=Forbidden(), source_video=Forbidden()) for r in clean]
        for method in METHODS:
            self.assertEqual(evaluate_frame(clean, method), evaluate_frame(tainted, method))
        self.assertEqual(clean, originals)

    def test_w_old_is_fixed_even_if_new_parameters_change(self):
        item = row(native_iou=0.6, motion=0.8, cosine=0.9, history_score=0.7, geometry_group_rank=2)
        changed = {**DEFAULT_PROTOCOL, "budget": 1, "native_iou_floor": 0.9,
                   "motion_floor": 0.9, "relative_margin": 0.9, "appearance_weight": 0.0}
        self.assertEqual(evaluate_frame([item], "W_old"), evaluate_frame([item], "W_old", changed))

    def test_empty_frame_and_input_order(self):
        for method in METHODS:
            self.assertEqual(evaluate_frame([], method), [])
        items = [row(2), row(1, 1)]
        self.assertEqual([edge_key(x) for x in evaluate_frame(items, "W1")], list(map(edge_key, items)))
        out = evaluate_frame(items, "W1")
        self.assertTrue(all(d["row_competitor_groups"] is None and d["column_competitor_identities"] is None for d in out))

    def test_malformed_or_mixed_frames_fail(self):
        cases = [
            [row(), row(2, frame=11)],
            [row(), row(2, sequence="other")],
            [row(), row()],
            [row(native_iou=float("nan"))],
            [row(motion=float("inf"))],
            [row(cosine=True)],
            [row(raw_box_xyxy=[0, 0, 0, 10])],
            [row(), row(2, raw_score=0.09)],
        ]
        for items in cases:
            with self.subTest(items=items), self.assertRaises(ValueError):
                evaluate_frame(items, "W2")
        with self.assertRaises(ValueError):
            evaluate_frame([row()], "W5")
        with self.assertRaises(ValueError):
            evaluate_frame([row()], "W3", {"budget": True})
        with self.assertRaises(ValueError):
            evaluate_frame([row()], "W3", {"unknown_threshold": 0.5})


class HistoricalCompatibilityTests(unittest.TestCase):
    def test_w0_and_old_match_original_r3_assignment(self):
        original_assign = historical_functions()
        rng = random.Random(28011)
        for iteration in range(80):
            items = []
            scores = {det: rng.uniform(0.05, 0.10) for det in range(8)}
            for track in range(1, 5):
                for det in rng.sample(range(8), rng.randint(1, 8)):
                    cosine = rng.choice([None, rng.uniform(-1.0, 1.0), 0.90])
                    motion = rng.choice([None, rng.uniform(0.0, 1.0), 0.8])
                    iou = rng.uniform(0.5, 1.0)
                    norm = (scores[det] - 0.05) / 0.05
                    utility = None if cosine is None or motion is None else (iou + norm + (cosine + 1.0) / 2.0 + motion) / 4.0
                    items.append(row(track, det, native_iou=iou, raw_score=scores[det], normalized_score=norm,
                                     cosine=cosine, motion=motion, history_score=utility,
                                     geometry_group_rank=rng.randint(1, 8),
                                     row_unknown=rng.random() < 0.1, column_unknown=rng.random() < 0.1,
                                     row_margin=rng.choice([None, -0.03, 0.04, 0.06, 0.2]),
                                     column_margin=rng.choice([None, -0.03, 0.04, 0.06, 0.2])))
            for method, setting in [("W0", {"method": "geometry", "floor": 0.5}),
                                    ("W_old", {"method": "history_competition", "cosine_floor": 0.85, "margin": 0.05})]:
                old = original_assign([x for x in items if x["geometry_group_rank"] <= 5], setting)
                with self.subTest(iteration=iteration, method=method):
                    self.assertEqual({edge_key(x) for x in old}, selected(evaluate_frame(items, method)))

    def test_legacy_duplicate_slot_ties_match(self):
        original_assign = historical_functions()
        items = [row(1, 2, box=[0., 0., 10., 10.]),
                 row(1, 1, box=[0., 0., 10., 10.]),
                 row(2, 3, box=[0., 0., 10., 10.])]
        for method, setting in [("W0", {"method": "geometry", "floor": 0.5}),
                                ("W_old", {"method": "history_competition", "cosine_floor": 0.85, "margin": 0.05})]:
            self.assertEqual(selected(evaluate_frame(items, method)), {edge_key(x) for x in original_assign(items, setting)})


if __name__ == "__main__":
    unittest.main()
