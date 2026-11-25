"""Join fixed filter decisions to GT diagnostic categories and per-track evidence."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


KEYS = ("sequence", "frame", "track_id")


def read_rows(path):
    with path.open(encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def index(row):
    return tuple(row[k] for k in KEYS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("results", "masks"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    lineage = read_rows(args.results / "lineage_rows.csv")
    by_key = {index(row): row for row in lineage}
    per_rule = defaultdict(dict)
    for row in read_rows(args.masks):
        rule = per_rule[row["rule"]]
        key = index(row)
        assert key not in rule and key in by_key and row["keep"] in {"0", "1"}
        rule[key] = int(row["keep"])
    assert all(r.keys() == by_key.keys() for r in per_rule.values())
    joined = [{**row, **{f"keep__{rule}": keep[index(row)] for rule, keep in per_rule.items()}}
              for row in lineage]
    with (args.results / "filter_lineage_join.csv").open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)
    summary = {}
    for rule, keep in per_rule.items():
        totals = {"kept": Counter(), "dropped": Counter()}
        for key, row in by_key.items():
            totals["kept" if keep[key] else "dropped"][row["diagnostic_category"]] += 1
        summary[rule] = dict(totals)
    with (args.results / "filter_lineage_summary.json").open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
