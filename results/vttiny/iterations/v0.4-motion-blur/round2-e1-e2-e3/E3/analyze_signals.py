"""Grouped held-out diagnostic separability using GT-free causal features."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from audit_native_fp import csv_output, fingerprint, original_video, read_json


FEATURES = (
    "detection_score", "output_score", "output_short_side", "past15_observations",
    "past15_score_median", "past15_high_observations", "past15_recoveries",
    "frames_since_previous_measured", "gmc_speed_diag_per_frame", "rgb_quality",
    "rgb_cosine_to_previous_real_feature",
)


def as_number(value):
    return float(value) if value else np.nan


def key(row):
    return row["sequence"], int(row["frame"]), int(row["track_id"])


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def weighted_auc(y, score, weights):
    order = np.argsort(score, kind="stable")
    yy, ww, ss = y[order], weights[order], score[order]
    index = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]
    negative = np.add.reduceat(ww * (yy == 0), index)
    positive = np.add.reduceat(ww * (yy == 1), index)
    denom = negative.sum() * positive.sum()
    if denom == 0:
        return np.nan
    cdf = np.cumsum(negative) - negative * .5
    return float(np.dot(positive, cdf) / denom)


def upper_quantile(values, weights, q):
    order = np.argsort(values, kind="stable")
    rank = np.searchsorted(np.cumsum(weights[order]), q * weights.sum(), side="left")
    return float(values[order[min(rank, len(values)-1)]])


def unit_track_weights(keys):
    counts = Counter(keys)
    return np.array([1. / counts[k] for k in keys])


def merge(a_root):
    causal = {key(r): r for r in read_rows(a_root / "E3_causal_signal_rows.csv")}
    result = []
    for row in read_rows(a_root / "E3_all_rows_CLEAR.csv"):
        indices = key(row)
        features = causal[indices]
        if row["original_video"] != original_video(indices[0]) or features["original_video"] != row["original_video"]:
            raise ValueError("Original-video grouping mismatch")
        if row["matched_gt"]:
            label = 0
        elif row["offline_category"] == "persistent_no_gt_support":
            label = 1
        else:
            continue
        result.append((indices, row["original_video"], label,
                       [as_number(features[name]) for name in FEATURES]))
    if not result:
        raise ValueError("No anchored positive or persistent negative diagnostics")
    return result


def diagnostic(cohort, pairs):
    outcomes = []
    groups = np.array([r[1] for r in pairs])
    y = np.array([r[2] for r in pairs])
    inputs = np.array([r[3] for r in pairs])
    for index, name in enumerate(FEATURES):
        signal = inputs[:, index]
        finite = np.isfinite(signal)
        finite_pairs = [r for r, ok in zip(pairs, finite) if ok]
        weights = np.zeros(len(pairs))
        weights[finite] = unit_track_weights([(r[0][0], r[0][2]) for r in finite_pairs])
        for group in np.unique(groups):
            train = finite & (groups != group)
            test = finite & (groups == group)
            all_test = groups == group
            if len(np.unique(y[train])) < 2 or not test.any():
                continue
            a = weighted_auc(y[train], signal[train], weights[train])
            direction = 1 if a >= .5 else -1
            calibrated = direction * signal
            true_train = train & (y == 0)
            cutoff = upper_quantile(calibrated[true_train], weights[true_train], .9)
            risk = test & (calibrated > cutoff)
            score = weighted_auc(y[test], calibrated[test], weights[test])
            outcomes.append({
                "cohort": cohort, "feature": name, "heldout_original_video": group,
                "train_original_videos": len(set(groups[train])),
                "train_labels_used_only_for_direction_and_quantile": True,
                "holdout_candidate_rows": int((all_test & (y == 1)).sum()),
                "holdout_candidate_feature_rows": int((test & (y == 1)).sum()),
                "holdout_candidate_rejected_rows": int((risk & (y == 1)).sum()),
                "holdout_true_rows": int((all_test & (y == 0)).sum()),
                "holdout_true_feature_rows": int((test & (y == 0)).sum()),
                "holdout_true_rejected_rows": int((risk & (y == 0)).sum()),
                "holdout_true_weight": float(weights[test & (y == 0)].sum()),
                "holdout_candidate_weight": float(weights[test & (y == 1)].sum()),
                "holdout_true_rejected_weight": float(weights[risk & (y == 0)].sum()),
                "holdout_candidate_rejected_weight": float(weights[risk & (y == 1)].sum()),
                "holdout_auc_tracks_equalized": float(score) if np.isfinite(score) else None,
                "direction_gt_free_at_application": direction, "cutoff_gt_free_at_application": cutoff,
            })
    return outcomes


def aggregate(rows):
    result = []
    groups = sorted({(r["cohort"], r["feature"]) for r in rows})
    for cohort, name in groups:
        folds = [r for r in rows if r["cohort"] == cohort and r["feature"] == name]
        auc = [r["holdout_auc_tracks_equalized"] for r in folds if r["holdout_auc_tracks_equalized"] is not None]
        def total(k):
            return sum(r[k] for r in folds)
        def ratio(x, y):
            return total(x)/total(y) if total(y) else None
        result.append({
            "cohort": cohort, "feature": name, "folds_tested": len(folds), "folds_with_both_classes": len(auc),
            "macro_holdout_auc": float(np.mean(auc)) if auc else None,
            "min_fold_auc": min(auc) if auc else None, "max_fold_auc": max(auc) if auc else None,
            "heldout_candidate_rows": total("holdout_candidate_rows"),
            "heldout_true_rows": total("holdout_true_rows"),
            "heldout_candidate_feature_coverage": ratio("holdout_candidate_feature_rows", "holdout_candidate_rows"),
            "heldout_true_feature_coverage": ratio("holdout_true_feature_rows", "holdout_true_rows"),
            "candidate_rejection_fraction_tracks_equalized": ratio("holdout_candidate_rejected_weight", "holdout_candidate_weight"),
            "true_rejection_fraction_tracks_equalized": ratio("holdout_true_rejected_weight", "holdout_true_weight"),
            "candidate_rejected_rows": total("holdout_candidate_rejected_rows"),
            "true_rejected_rows": total("holdout_true_rejected_rows"),
        })
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("e3a", "e3b", "output"):
        p.add_argument("--" + field, type=Path, required=True)
    args = p.parse_args()
    audit = read_json(args.e3a / "E3_A_evaluation_audit.json")
    assert audit["status"] == "PASS"
    for name in ("E3_all_rows_CLEAR.csv", "E3_causal_signal_rows.csv"):
        assert fingerprint(args.e3a / name) == audit["outputs_sha256"][name]
    reviews = read_json(args.e3b / "review_contract.json")
    assert reviews["a_digest"] == fingerprint(args.e3a / "E3_A_evaluation_audit.json")
    pairs = merge(args.e3a)
    indices = FEATURES.index("output_short_side"), FEATURES.index("gmc_speed_diag_per_frame")
    small = [r for r in pairs if r[3][indices[0]] <= 12 and r[3][indices[1]] <= .12]
    data = {"all_confirmed_CLEAR_vs_no_GT_proxy": pairs, "small_near_static_matched_niche": small}
    args.output.mkdir(parents=True)
    results = [r for cohort, members in data.items() for r in diagnostic(cohort, members)]
    summary = aggregate(results)
    csv_output(args.output, "E3_signal_analysis.csv", summary)
    csv_output(args.output, "E3_signal_heldout_folds.csv", results)
    coverage = {}
    for name, members in data.items():
        coverage[name] = {
            "source_videos": len({r[1] for r in members}),
            "confirmed_CLEAR_rows": sum(r[2] == 0 for r in members),
            "persistent_no_GT_proxy_rows": sum(r[2] == 1 for r in members),
            "candidate_source_videos": len({r[1] for r in members if r[2] == 1}),
        }
    report = {
        "status": "INCONCLUSIVE_MANUAL_CONFIRMATION_REQUIRED",
        "operating_point_proxy_only": "Direction plus 90th percentile of real-object score selected on other original videos. No hyperparameter search.",
        "features_sha256": audit["outputs_sha256"]["E3_causal_signal_rows.csv"],
        "feature_schema": FEATURES,
        "cohorts": coverage,
        "small_static_definition": "Both classes have native output short side <=12 px and current compensated displacement <=.12 prior-box diagonals per elapsed frame.",
        "candidate_interpretation": "Offline lifetime non-overlap is a diagnostic proxy, not visual confirmation of background.",
        "manual_label_count": 0,
        "aggregation": "Each track's finite rows for a feature have total weight 1, macro AUC across original-video folds with two classes.",
        "output_sha256": {path.name:fingerprint(path) for path in sorted(args.output.glob("*.csv"))},
    }
    with (args.output / "E3_C_signal_protocol.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print("E3_C_PROXY_COMPLETE", coverage, flush=True)


if __name__ == "__main__":
    main()
