"""Assemble the paired R4 results and check the official and event ledgers agree."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


NAMES = {
    "B_off": "B 参考", "BG_off": "B＋GMC 参考", "B_lower_low": "B＋原生低阈值 0.05",
    "B_plain": "B＋旁路几何补接", "B_coast": "B＋Coast-only", "B_w_coast": "B＋W-coast",
    "B_w_full": "B＋完整 W", "BG_w_full": "B＋GMC＋完整 W",
}
PERCENT = {"HOTA": "HOTA___AUC", "DetA": "DetA___AUC", "AssA": "AssA___AUC", "IDF1": "IDF1", "MOTA": "MOTA"}
COUNTS = {"FP": "CLR_FP", "FN": "CLR_FN", "IDSW": "IDSW", "Frag": "Frag"}
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def metrics(path):
    return {r["seq"]: dict(**{k:float(r[col])*100 for k,col in PERCENT.items()},
                           **{k:int(float(r[col])) for k,col in COUNTS.items()}) for r in csv_rows(path)}


def fcell(key, value):
    return f"{value:.3f}" if key in PERCENT else f"{value:,d}"


def dcell(key, value):
    return f"{value:+.3f}" if key in PERCENT else f"{value:+,d}"


def report(scope):
    root = HERE / "results" / scope
    completed = read(root / "complete.json")
    checks_path = root / "verified.json"
    checked = read(checks_path)
    comparison = read(root / "metrics/comparison.json")
    events = read(root / "attribution/summary.json")["variants"]
    commits = csv_rows(root / "attribution/committed_clear.csv")
    adopted = Counter(row["arm"] for row in commits)
    all_metrics = {arm: metrics(root / "metrics" / arm / "object_detailed.csv") for arm in NAMES}
    if (completed["status"] != "PASS" or completed["checks"] != hashlib.sha256(checks_path.read_bytes()).hexdigest()
            or set(comparison["sequences"]) != set(checked["scope"])):
        raise ValueError("R4 frozen settings and evaluation coverage differ")
    fields = tuple(PERCENT) + tuple(COUNTS)
    data, notes = [], []
    table = ["| 对照组 | " + " | ".join(fields) + " |",
             "| --- | " + " | ".join(["---:"]*len(fields)) + " |"]
    deltas = table.copy()
    official = {row["tracker"]: row for row in comparison["results"]}
    for arm, desc in NAMES.items():
        paired = "BG_off" if arm.startswith("BG_") else "B_off"
        summary = all_metrics[arm]["COMBINED"]
        baseline = all_metrics[paired]["COMBINED"]
        if any(abs(summary[k]-official[arm][k]) > 1e-7 for k in fields if k != "Frag"):
            raise ValueError(f"Official comparison and detailed metrics differ for {arm}")
        if not set(all_metrics[arm]) == {"COMBINED", *checked["scope"]}:
            raise ValueError(f"Per-sequence scope differs for {arm}")
        table.append("| " + desc + " | " + " | ".join(fcell(key, summary[key]) for key in fields) + " |")
        if arm != paired:
            delta = {k:summary[k]-baseline[k] for k in fields}
            event = events[arm]
            if (event["gained"]-event["lost"] != -delta["FN"] or event["fp_delta"] != delta["FP"]
                    or event["idsw_delta"] != delta["IDSW"]
                    or checked["arms"][arm]["totals"].get("committed",0) != adopted[arm]):
                raise ValueError(f"CLEAR and native event ledgers differ for {arm}")
            deltas.append("| " + desc + " | " + " | ".join(dcell(key, delta[key]) for key in fields) + " |")
        for seq in checked["scope"]:
            current, old = all_metrics[arm][seq], all_metrics[paired][seq]
            data.append(dict(arm=arm, reference=paired, sequence=seq, **current,
                             **{f"delta_{k}":current[k]-old[k] for k in fields}))
        if adopted[arm]:
            notes.append(f"| {desc} | {adopted[arm]} | {event['newly_covered']} | {event['unmatched_FP']} | "
                         f"{event['TP_same_old_identity']} | {event['TP_other_identity']} | {event['gained']} | {event['lost']} | "
                         f"{event['B_reference']} → {event['B_actual']} | {event['reference_B_coverage_lost']} |")
    with (root / "paired_per_sequence.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=data[0].keys())
        w.writeheader()
        w.writerows(data)
    parts = ["# VT-Tiny-MOT W R4 " + ("完整 30 段" if scope == "full" else "三段试跑"), "",
             f"覆盖 {len(checked['scope'])} 段、{comparison['frames']:,} 帧。指标采用固定版 TrackEval 的 COMBINED 聚合。",
             "变体与各自的冻结 B 或 B＋GMC 参考配对。HOTA、DetA、AssA、IDF1、MOTA 以百分制呈现，差值为百分点。", "",
             *table, "", "## 配对变化", "", *deltas, "", "## 提交事件逐帧核对", "",
             "| 对照组 | 提交 | 当帧补回 | 当帧未匹配 FP | 有历史支持的原目标 | 有历史支持的其他目标 | 全程新增覆盖 | 全程失去覆盖 | B 恢复次数 | 失去原 B 覆盖 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |", *notes, "",
             "当帧身份来自官方 CLEAR 匹配和此前五帧内至少两次一致的历史匹配。全程覆盖比较各组整段轨迹，包含分叉后的间接影响。原 B 覆盖减少使用实际事件键与真值逐帧比较。", "",
             "## W 的序列分布", "",
             "| 对照组 | HOTA 提升段数 | 持平段数 | 下降段数 | 主要贡献段与 HOTA 变化 |", "| --- | ---: | ---: | ---: | --- |"]
    for arm in ("B_w_coast", "B_w_full", "BG_w_full"):
        sequence = [(r["sequence"], r["delta_HOTA"]) for r in data if r["arm"] == arm]
        pos = sum(d > 1e-8 for _, d in sequence)
        neg = sum(d < -1e-8 for _, d in sequence)
        top = sorted(((s,d) for s,d in sequence if abs(d)>1e-8), key=lambda item: -abs(item[1]))[:3]
        highlights = "、".join(f"{s} {d:+.3f}" for s,d in top) or "无"
        parts.append(f"| {NAMES[arm]} | {pos} | {len(sequence)-pos-neg} | {neg} | {highlights} |")
    before, after = ({tuple(r[k] for k in ("sequence", "frame", "track_id", "det_index"))
                      for r in commits if r["arm"] == arm} for arm in ("B_w_coast","B_w_full"))
    parts.extend(["", f"W-coast 和完整 W 共有 {len(before & after)} 个相同的提交键，两组分别另有 {len(before-after)} 和 {len(after-before)} 个提交键。", "",
             "[逐序列配对指标](paired_per_sequence.csv) · [逐帧提交与真值](attribution/committed_clear.csv) · [变化的 B 恢复事件](attribution/changed_B_events.csv) · [输入与状态核验](verified.json)", ""]
    )
    (root / "报告.md").write_text("\n".join(parts), encoding="utf-8")
    print("\n".join([*table, "", *deltas, "", *notes]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scope", choices=("pilot", "full"))
    report(p.parse_args().scope)
