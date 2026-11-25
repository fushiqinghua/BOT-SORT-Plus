"""Explore conservative conjunctions of pre-screened native trajectory signals."""
import argparse
import csv
import json
from pathlib import Path
from zipfile import ZipFile
from causal_screen import METHOD, csv_data, evaluate, row_key, row_key_dict, write_csv
from causal_screen import hash_member


def trace(row):
    return dict(
        small=float(row["min_side_px"]) < 8,
        static=int(row["native_rows_15"]) >= 5 and float(row["raw_center_span_15"]) <= .5,
        repeated=int(row["restores_15"]) >= 2,
        low_support=row["high_30"] == "False",
    )


def masks(rows):
    rules = {
        "veto_tiny_and_static": ("small", "static"),
        "veto_static_and_repeated": ("static", "repeated"),
        "veto_tiny_static_and_repeated": ("small", "static", "repeated"),
        "veto_tiny_static_and_no_high": ("small", "static", "low_support"),
        "veto_tiny_and_no_high": ("small", "low_support"),
        "veto_static_and_no_high": ("static", "low_support"),
    }
    result = []
    for row in rows:
        signals = trace(row)
        for rule, operands in rules.items():
            result.append({**row_key_dict(row), "rule": rule,
                           "keep": int(not all(signals[key] for key in operands))})
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("features", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    rows = csv_data(args.features.read_bytes())
    filters = masks(rows)
    args.output.mkdir(exist_ok=False)
    write_csv(args.output / "masks_all.csv", filters)
    with ZipFile(args.archive) as archive:
        hashes = json.loads(archive.read("EVIDENCE_SHA256.json"))
        labels = csv_data(hash_member(archive, hashes, "parity/full/supplemental_rows.csv"))
        labels = {row_key(row): row for row in labels}
        fixed = csv_data(hash_member(archive, hashes, "identity/full/paired_events.csv"))
        d3 = {(row["sequence"], int(row["original_failure_frame"]),
               int(row["run_verified_anchor_id"]))
              for row in fixed if row["run"] == METHOD
              and row["original_failure_true_output_only_rescue"] == "True"}
    result = evaluate(filters, labels, d3)
    (args.output / "screen.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result:
        if row["scope"] in ("all", "DJI_0327_3", "DJI_0339_2"):
            print(json.dumps(row))


if __name__ == "__main__":
    main()
