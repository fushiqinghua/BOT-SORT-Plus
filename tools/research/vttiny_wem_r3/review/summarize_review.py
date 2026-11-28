"""Build split-specific W R3 review indexes without interpreting model votes as labels."""

import argparse
from collections import Counter, defaultdict
import csv
from html import escape
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PUBLISHED_REVIEW = REPO / "results/vttiny/wem_v0.1/r3/review"
PARTS = ("calibration", "holdout")
PART_TEXT = {"calibration": "校准组", "holdout": "留出组"}
POOL_TEXT = {"background": "背景候选", "identity": "身份候选"}
VOTE_TEXT = {
    "likely_background": "疑似背景",
    "plausible_other_identity": "疑似其他目标",
    "same_identity_plausible": "可能为原目标",
    "visible_target_unresolved_identity": "可见目标，身份未定",
    "uncertain": "图像证据不足",
}
CONF_TEXT = {"low": "低", "medium": "中", "high": "高"}
CASE_FIELDS = ("case_id", "partition", "source_video", "sequence", "review_pool", "frame", "track_id",
               "det_index", "raw_score", "best_gt_iou", "gt_id", "best_candidate_gt_id", "anchor_status")
VOTE_FIELDS = ("model_assisted_label", "confidence", "model_assisted_note")


def read_votes(file, cases):
    with file.open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    votes = {item["case_id"]: item for item in records}
    selected = {case["case_id"]: case for case in cases}
    if len(votes) != len(records) or len(selected) != len(cases) or set(votes) != set(selected):
        raise ValueError(f"Missing or repeated model judgments in {file.name}")
    for row in records:
        if (row["model_assisted_label"] not in VOTE_TEXT or row["confidence"] not in CONF_TEXT
                or not row["model_assisted_note"].strip()):
            raise ValueError(f"Unexpected model judgment in {file.name}")
        case = selected[row["case_id"]]
        for key in ("partition", "source_video", "sequence", "review_pool", "frame", "track_id", "det_index"):
            if row[key] != str(case[key]):
                raise ValueError(f"Published review index disagrees with frozen candidate {case['case_id']} on {key}")
    return votes


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Empty review rows for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def review_html(part, cases, votes):
    sections = []
    for source in sorted({c["source_video"] for c in cases}):
        source_cases = [case for case in cases if case["source_video"] == source]
        rows = []
        for c in source_cases:
            case_id = c["case_id"]
            vote = votes[case_id]
            path = f"{c['review_pool']}/{case_id}.png"
            rows.append(f'''<tr><td><a href="{path}">{case_id}</a><br><small>{c['sequence']} / 帧 {c['frame']} / ID {c['track_id']}</small></td>
<td>{POOL_TEXT[c['review_pool']]}</td>
<td><strong>{VOTE_TEXT[vote['model_assisted_label']]}</strong> · {CONF_TEXT[vote['confidence']]}<br>{escape(vote['model_assisted_note'])}</td></tr>''')
        summary = Counter(c["review_pool"] for c in source_cases)
        sections.append(f'''<section><h2>{escape(source)} · {len(source_cases)} 例</h2>
<p>背景候选 {summary['background']} 例，身份候选 {summary['identity']} 例。</p>
<table><thead><tr><th scope="col">图卡</th><th scope="col">抽样类型</th><th scope="col">模型辅助判断</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></section>''')
    return f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>W R3 逐帧复核 · {PART_TEXT[part]}</title>
<style>
body{{font:16px/1.6 system-ui,"Microsoft YaHei",sans-serif;margin:36px auto;max-width:1040px;padding:0 18px;background:#f8fafc;color:#152439}}
h1{{line-height:1.3}} h2{{font-size:1.2rem;margin-bottom:4px}} section{{margin-top:28px}}
table{{width:100%;border-collapse:collapse;background:white}} th,td{{text-align:left;vertical-align:top;padding:10px 12px;border-bottom:1px solid #dce5ee}}
th{{background:#e8eff7}} td:first-child{{min-width:150px}} td:nth-child(2){{min-width:82px}}
a{{color:#004a81}} small{{color:#465b71}}
</style>
<h1>W R3 逐帧复核 · {PART_TEXT[part]}</h1>
<p>共 {len(cases)} 例模型辅助判断。人工状态单独记录在复核表，初始均为待复核。点击案例编号打开完整图卡。</p>
<p><a href="model_assisted.csv">模型辅助表</a> · <a href="human_review_pending.csv">人工复核表</a> · <a href="group_summary.csv">按来源组统计</a></p>
{''.join(sections)}
</html>
'''


def compile_partition(destination, part, cases, votes):
    model, human = [], []
    groups = defaultdict(Counter)
    for case in cases:
        case_id = case["case_id"]
        rel = f"{case['review_pool']}/{case_id}.png"
        if not (destination / rel).is_file():
            raise ValueError(f"Review card missing for {case_id}")
        fields = {k: case[k] for k in CASE_FIELDS}
        vote = {key: votes[case_id][key] for key in VOTE_FIELDS}
        model.append(dict(**fields, card=rel, **vote))
        human.append(dict(**fields, card=rel, human_review_status="pending", human_decision="", reviewer="", human_note=""))
        counter = groups[case["source_video"], case["review_pool"]]
        counter["cases"] += 1
        counter[vote["model_assisted_label"]] += 1
    aggregate = []
    for (group, pool), counts in sorted(groups.items()):
        aggregate.append(dict(partition=part, source_video=group, review_pool=pool, cases=counts["cases"],
                              **{key: counts[key] for key in VOTE_TEXT}, human_pending_at_delivery=counts["cases"]))
    write_csv(destination / "model_assisted.csv", model)
    human_path = destination / "human_review_pending.csv"
    if not human_path.exists():
        write_csv(human_path, human)
    write_csv(destination / "group_summary.csv", aggregate)
    (destination / "index.html").write_text(review_html(part, cases, votes), encoding="utf-8")
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=HERE / "inputs/candidates.json")
    parser.add_argument("--rendered", type=Path, default=HERE / "rendered/full")
    parser.add_argument("--votes-root", type=Path, default=PUBLISHED_REVIEW,
                        help="Directory containing partition-specific published model_assisted.csv files")
    args = parser.parse_args()
    content = json.loads(args.candidates.read_bytes())
    source = content["candidates"]
    report = {}
    for part in PARTS:
        cases = [c for c in source if c["partition"] == part]
        votes = read_votes(args.votes_root / part / "model_assisted.csv", cases)
        groups = compile_partition(args.rendered / part, part, cases, votes)
        report[part] = dict(cases=len(cases), proxy_groups=len({c["source_video"] for c in cases}),
                            model_assisted=dict(Counter(v["model_assisted_label"] for v in votes.values())),
                            human_pending_at_delivery=len(cases), grouping=groups)
    output = dict(status="MODEL_ASSISTED_VISUAL_REVIEW", formal_labels_changed=False, r4_run=False, **report)
    (args.rendered / "review_summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                                                       encoding="utf-8")
    print({part: {k: v for k, v in row.items() if k != "grouping"} for part, row in report.items()})


if __name__ == "__main__":
    main()
