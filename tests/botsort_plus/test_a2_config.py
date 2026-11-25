"""Experiment-matrix contracts: explicit E/I ablations and old-config loading."""
from dataclasses import asdict
import json
from pathlib import Path
import unittest

from botsort_plus.config import PlusConfig


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs" / "botsort_plus"


class A2ConfigurationTests(unittest.TestCase):
    def config(self, filename):
        return PlusConfig.load(CONFIGS / filename)

    def test_twelve_original_configs_still_choose_legacy_adapter(self):
        files = sorted(p for p in CONFIGS.glob("*.json") if int(p.name.split("_", 1)[0]) < 12)
        self.assertEqual(len(files), 12)
        for path in files:
            original = json.loads(path.read_text(encoding="utf-8"))
            cfg = PlusConfig.load(path)
            with self.subTest(config=path.name):
                self.assertEqual(cfg.adapter_mode, "legacy")
                self.assertEqual(cfg.name, original["name"])
                for key in ("birth", "recovery", "appearance"):
                    self.assertEqual(getattr(cfg, key).enabled,
                                     original.get(key, {}).get("enabled", False))
                    for option, value in original.get(key, {}).items():
                        self.assertEqual(asdict(getattr(cfg, key))[option], value)
                self.assertEqual(cfg.native_overrides, original.get("native_overrides", {}))

    def test_all_fifteen_new_configs_roundtrip_without_silent_defaults(self):
        files = sorted(p for p in CONFIGS.glob("*.json") if 12 <= int(p.name.split("_", 1)[0]) <= 26)
        self.assertEqual(len(files), 15)
        names = set()
        for path in files:
            raw = json.loads(path.read_text(encoding="utf-8"))
            cfg = PlusConfig.load(path)
            with self.subTest(config=path.name):
                self.assertEqual(cfg.adapter_mode, "v2")
                self.assertFalse(cfg.appearance.enabled)
                self.assertIn("mode", raw["birth_policy"])
                self.assertIn("enabled", raw["birth_identity"])
                self.assertEqual(PlusConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))).to_dict(),
                                 cfg.to_dict())
                self.assertNotIn(cfg.name, names)
                names.add(cfg.name)

    def test_reference_configs_do_not_enable_a_or_c(self):
        baseline = self.config("12_baseline_ref.json")
        b = self.config("13_B_ref.json")
        self.assertFalse(baseline.enabled)
        self.assertFalse(baseline.birth.enabled)
        self.assertFalse(b.birth.enabled)
        self.assertTrue(b.recovery.enabled)
        self.assertFalse(b.birth_identity.enabled)
        legacy_b = self.config("05_B.json")
        self.assertEqual(asdict(b.recovery), asdict(legacy_b.recovery))
        self.assertEqual(asdict(b.birth), asdict(legacy_b.birth))
        self.assertEqual(b.native_overrides, legacy_b.native_overrides)

    def test_two_by_two_ablation_changes_e_and_i_explicitly(self):
        expected = {
            "20_A1_ref.json": (False, "legacy", False),
            "14_A2_E.json": (False, "two_stage", False),
            "15_A2_I.json": (False, "legacy", True),
            "16_A2.json": (False, "two_stage", True),
            "21_A1B_ref.json": (True, "legacy", False),
            "17_A2B_E.json": (True, "two_stage", False),
            "18_A2B_I.json": (True, "legacy", True),
            "19_A2B.json": (True, "two_stage", True),
        }
        for filename, values in expected.items():
            cfg = self.config(filename)
            with self.subTest(config=filename):
                self.assertTrue(cfg.birth.enabled)
                self.assertEqual((cfg.recovery.enabled, cfg.birth_policy.mode,
                                  cfg.birth_identity.enabled), values)
                self.assertFalse(cfg.birth_policy.observe_only)

    def test_delay_and_score_controls_change_only_the_named_policy_option(self):
        default = self.config("19_A2B.json").to_dict()
        cases = (("22_A2B_score_window.json", "mode", "score_window"),
                 ("23_A2B_fixed_window.json", "mode", "fixed_window"),
                 ("24_A2B_score020.json", "mean_score", .2))
        for filename, field, value in cases:
            expected = json.loads(json.dumps(default))
            actual = self.config(filename).to_dict()
            expected["name"] = actual["name"]
            expected["birth_policy"][field] = value
            with self.subTest(config=filename):
                self.assertEqual(actual, expected)

    def test_shadow_changes_only_observation_flag_relative_to_a2(self):
        for base, shadow in (("16_A2.json", "25_A2_shadow.json"),
                             ("19_A2B.json", "26_A2B_shadow.json")):
            expected = self.config(base).to_dict()
            actual = self.config(shadow).to_dict()
            expected["name"] = actual["name"]
            expected["birth_policy"]["observe_only"] = True
            self.assertEqual(actual, expected)

    def test_unknown_top_level_or_nested_keys_fail(self):
        invalid = ({"adapter_mod": "v2"},
                   {"adapter_mode": "v2", "birth_policy": {"mean_confidence": .2}},
                   {"adapter_mode": "v2", "birth_identity": {"wait_frame": 5}},
                   {"recovery": {"unknown": 1}},
                   {"birth": {"confirm_after": 3}})
        for data in invalid:
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, "Unknown"):
                PlusConfig.from_dict(data)

    def test_invalid_modes_windows_scores_and_identity_flags_fail(self):
        invalid = ({"adapter_mode": "v3"},
                   {"adapter_mode": "v2", "birth_policy": {"mode": "two-stage"}},
                   {"adapter_mode": "v2", "birth_policy": {"collect_hits": 6}},
                   {"adapter_mode": "v2", "birth_policy": {"verify_hits": 4}},
                   {"adapter_mode": "v2", "birth_policy": {"fixed_hits": 9}},
                   {"adapter_mode": "v2", "birth_policy": {"collect_window": True}},
                   {"adapter_mode": "v2", "birth_policy": {"mean_score": float("nan")}},
                   {"adapter_mode": "v2", "birth_policy": {"mean_score": float("inf")}},
                   {"adapter_mode": "v2", "birth_policy": {"mean_score": -.1}},
                   {"adapter_mode": "v2", "birth_policy": {"mean_score": 1.1}},
                   {"adapter_mode": "v2", "birth_policy": {"observe_only": 1}},
                   {"adapter_mode": "v2", "birth_identity": {"enabled": "true"}},
                   {"adapter_mode": "v2", "birth_identity": {"wait_frames": 0}},
                   {"adapter_mode": "v2", "birth_identity": {"max_age": False}})
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                PlusConfig.from_dict(data)

    def test_nonobject_json_is_not_silently_loaded_as_baseline(self):
        for data in ([], (), None, "baseline", 1):
            with self.subTest(data=data), self.assertRaises(ValueError):
                PlusConfig.from_dict(data)
        for section in ("birth", "recovery", "appearance", "geometry",
                        "birth_policy", "birth_identity"):
            for value in (None, [], "enabled", 1):
                data = {"adapter_mode": "v2", section: value}
                with self.subTest(data=data), self.assertRaises(ValueError):
                    PlusConfig.from_dict(data)

    def test_custom_a2_settings_require_v2_adapter(self):
        for section, options in (("birth_policy", {"mean_score": .2}),
                                 ("birth_identity", {"wait_frames": 7})):
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, "adapter_mode=v2"):
                PlusConfig.from_dict({section: options})

    def test_candidate_capacity_covers_requested_evidence_window(self):
        data = {"adapter_mode": "v2", "birth": {"enabled": True, "max_age": 5},
                "birth_policy": {"mode": "fixed_window", "fixed_window": 8, "fixed_hits": 5}}
        with self.assertRaisesRegex(ValueError, "max_age"):
            PlusConfig.from_dict(data)


if __name__ == "__main__":
    unittest.main()
