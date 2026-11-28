"""Versioned experimental constants, separate from local machine paths."""

BASE_COMMIT = "b734690e7af42f67be49eb3eec603a2b3d9a3683"
PARTITIONS = ("fit", "calibration", "holdout")
LABELS = ("same_identity", "other_identity", "confirmed_background", "unknown", "ignored")
REFERENCE = "B_early_moderate"
PINS = {
    "split": "43329496f50105b87a9175d312f70b65b1dd3e050307697b20a0656709398c04",
    "encoder": "27f110f2fa8ae3c619941c126699486771a42d252b4e675d1c077bb91e0bd2d1",
    "reference_protocol": "ddb7f476ad97d870aae3acf8aa1c071ef190846cd62fcb267f8105014b0aa69a",
    "full_feature_manifest": "e3c28c4b312d575bbba2631f3bc5bdd893be4f0489849c06195833e8092edad2",
    "full_label_receipt": "16ad7ccc7fe3e75072defcb66fae1e0b585bf345ea2d0744ffa1627ff4e47f61",
}
ENGINE = {"budget": 5, "native_iou_floor": 0.5, "motion_floor": 0.25,
          "appearance_weight": 0.2, "relative_margin": 0.05}
PROTOCOL = {
    "schema": "wem_r31_protocol_v1",
    "base_commit": BASE_COMMIT,
    "reference": REFERENCE,
    "candidate_source": "S0 frozen R3 scores; original geometry/GMC/ownership",
    "engine": ENGINE,
    "methods": {
        "W0": "i; no motion or appearance requirement",
        "W1": "(i+m)/2; require m >= 0.25",
        "W2": "0.4*i+0.4*m+0.2*((c+1)/2); same eligibility as W1; missing c falls back to W1",
        "W3": "W2 plus pre-top-k row/identity cosine margins >= 0.05 when competitors exist",
        "W_old": "historical history_competition: c>=0.85, m>=0.25, history_score>=0.55, margins>=0.05",
    },
    "soft_appearance_role": "ranking_only_no_absolute_cosine_or_joint_utility_cutoff",
    "competition": {
        "row": "all geometry/ownership-eligible distinct raw-box alternatives, before top-k",
        "column": "all recorded other-identity challengers including already matched neighbors",
        "no_competitor": "not_applicable; retain W2 decision eligibility",
        "missing_comparison": "reject as unavailable when a competitor exists",
    },
    "assignment": "maximum utility, exact-coordinate candidate groups, one-to-one, zero-utility dummy rejection",
    "score_tie_break": "original normalized detection score then original row index; inherited from R3",
    "source_pins": dict(PINS),
    "threshold_search": False,
    "operating_point_selection": "none",
    "enabled_online": False,
    "holdout_status": "previously inspected; developmental diagnostic, not a new blind test",
    "unknown_policy": "report separately; neither false positives nor true positives",
    "history_control": "W_old is a historical reference, not a one-factor W2 ablation",
}

SETTINGS = {
    "schema": "wem_r31_settings",
    "scope": "full",
    "paths": {
        "features_dir": None,
        "labels_dir": None,
        "split_file": None,
        "output_dir": "runs/wem_r3_1/full",
    },
}
