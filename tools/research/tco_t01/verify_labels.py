"""Check official counts and extract every bounded crossing with its exact source indices."""
from argparse import ArgumentParser
from collections import Counter
from itertools import chain
from pathlib import Path
import csv
import gzip
import json


def read(path):
    return json.loads(path.read_bytes())


def records(path):
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def case_key(row, detected):
    return (row["run_fingerprint"], row["sequence"], row["frame"], row["track_id"], row[detected])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("prior_metrics", type=Path)
    parser.add_argument("official_input_manifest", type=Path)
    args = parser.parse_args()
    folder = args.labels
    report = read(folder/"routing_summary.json")
    prior = {(r["tracker"], r["sequence"]): r for r in read(args.prior_metrics)["results"]}
    assert len(report["sequence_scores"]) == 60
    for row in report["sequence_scores"]:
        expected = prior[row["tracker"], row["sequence"]]
        assert all(row[k] == expected[k] for k in ("TP","FP","FN","IDSW") if k in expected)
    inputs = read(args.official_input_manifest)
    assert report["gt_sha256"] == inputs["ground_truth"]["sha256"]
    assert report["trackeval_commit"] == inputs["trackeval"]["commit"]
    by_key = {}
    counts = Counter()
    fields = ("method","sequence","source_video","frame","track_id","assigned_det_index","gt_id",
              "gt_short_side_bin","identity_status","reference_low_output_status","anchor_covered",
              "raw_iou","feasible_iou","nearest_other_iou","gt_exact_reachable","raw_xyxy","gt_xyxy")
    for row in records(folder/"episodes.jsonl.gz"):
        linked = row["bounded_cross_050"]
        if not linked and not row["geometry_only_bounded_cross_050"]:
            continue
        key = case_key(row,"assigned_det_index")
        assert key not in by_key
        archive = {k:row[k] for k in fields}
        archive["case_status"] = ("same_line_already_official_tp" if linked else
                                  "identity_unverified_already_covered" if row["anchor_covered"] else
                                  "identity_unverified_official_miss")
        archive["human_review"] = "PENDING"
        by_key[key] = archive
        counts[row["method"], archive["case_status"]] += 1
    for row in records(folder/"pairs.labels.jsonl.gz"):
        key = case_key(row,"det_index")
        if key in by_key:
            assert row["finite_candidate_crosses_050"]
            assert row["feasible_xyxy"] and 0 < row["raw_iou"] < .5 <= row["feasible_iou"]
            by_key[key]["feasible_xyxy"] = row["feasible_xyxy"]
    assert all("feasible_xyxy" in row for row in by_key.values())
    for method, totals in report["methods"].items():
        assert counts[method,"same_line_already_official_tp"] == totals.get("linked_legal_candidate_crosses_050",0)
        assert counts[method,"identity_unverified_already_covered"] == totals.get("geometry_only_bounded_cross_050_anchor_covered",0)
        assert counts[method,"identity_unverified_official_miss"] == totals.get("geometry_only_bounded_cross_050_anchor_uncovered",0)
    rows = sorted(by_key.values(), key=lambda r:(r["case_status"],r["method"],r["sequence"],r["frame"],r["track_id"]))
    with (folder/"feasible_cases.csv").open("x",encoding="utf-8-sig",newline="") as file:
        writer = csv.DictWriter(file,fieldnames=[*fields,"case_status","human_review","feasible_xyxy"])
        writer.writeheader()
        writer.writerows(rows)
    audit = {"status":"PASS","sequence_scores_match_historical":60,"case_rows":len(rows),
             "producer_trackeval_sha":report["trackeval_commit"],
             "matched_gt_hashes":len(report["gt_sha256"]),
             "case_counts":[dict(method=m,case_status=s,count=n) for (m,s),n in sorted(counts.items())]}
    (folder/"offline_verification.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    print("Offline verification",audit,flush=True)


if __name__ == "__main__":
    main()
