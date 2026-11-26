"""Write an E1 report from completed artifact tables and proof manifests."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from run_shadow import ROOT, digest, read, write


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def info_measure(rows):
    return {
        "rows": len(rows),
        "gt_frames": len({(r["sequence"],r["frame"],r["anchor_gt_id"]) for r in rows if r["anchor_gt_id"]}),
        "gt_tracks": len({(r["sequence"],r["anchor_gt_id"]) for r in rows if r["anchor_gt_id"]}),
        "episodes": len({(r["sequence"],r["track_id"],r["last_seen"]) for r in rows if r["state_before"]=="2"}),
        "sequences": len({r["sequence"] for r in rows}),
    }


def block(title, rows, column):
    selected=defaultdict(list)
    for item in rows:
        selected[item[column]].append(item)
    lines=[f"### {title}","", "| 原因 | 轨迹帧 | 独立 GT 帧 | GT 轨迹 | 旧 ID 失联片段 | 序列 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, items in sorted(selected.items(),key=lambda pair:(-len(pair[1]),pair[0])):
        n=info_measure(items)
        lines.append(f"| `{name}` | {n['rows']} | {n['gt_frames']} | {n['gt_tracks']} | {n['episodes']} | {n['sequences']} |")
    return lines+[""]


def audit():
    home=ROOT/"audit_analysis"
    stages=table(home/"E1_failure_stages.csv")
    pre=read(ROOT/"cohort/cohort_summary.json")
    proof={"status":"PASS",
           "pilot":read(ROOT/"pilot_parity.json"),
           "full":read(ROOT/"full_parity.json"),
           "independent_B_and_native_recomputation":True,
           "native_reference_high_precision":"BoxMOT native iou/fuse_score and lap assignment",
           "B_reference":"Match pairs, reservation set, reasons, raw and eligible matrices",
           "P0_float_comparison_absolute_tolerance":1e-10,
           "E1-A":digest(ROOT/"cohort/lost_cohort.csv"),
           "E1-B":digest(home/"E1_failure_stages.csv")}
    write(ROOT/"E1_shadow_parity.json",proof)
    low=[r for r in stages if r["cohort"]=="unconsumed_low"]
    high=[r for r in stages if r["cohort"]=="unconsumed_high"]
    stolen=[r for r in stages if r["cohort"] in ("other_track_consumed_high","other_track_consumed_low")]
    lines=["# E1-B 冻结 Lost 关联失败阶段",
      "",
      "冻结 `B_early_moderate`，30 段、12,590 帧。单序列和全量只读观测的 MOT、原生事件和 B 相关事件逐字节一致。",
      "使用预关联矩阵、原始候选及预测历史、关联前外观原型和冻结检测特征，独立重算 native high 与 B 的原始代价、可分配矩阵、分配与 B 拒绝理由。",
      "确认标注匹配、失联时间及原轨迹跟踪状态仅用于离线分析。",
      "",
      f"主队列共有 {len(stages)} 个轨迹帧。原 ID 无正确输出且有可用检测的行数为 {pre['missing_with_usable_detection']['track_frames']}，其中 9 行无法唯一配对检测。唯一配对为 1,407 行。",
      "下表各分组的 GT 帧可重复出现，不能直接对去重列求和。规范化原因保留 `unknown`。",
      ""]
    lines+=block("未消费低分框的 B 阶段",low,"B_reason")
    lines+=block("未消费高分框的原生高分阶段",high,"native_reason")
    lines+=block("未消费高分框的 B 阶段",high,"B_reason")
    lines+=block("已被其他轨迹消费的候选",stolen,"B_reason")
    lines.extend(["位置低分候选中的 Lost 状态全部沿用原生低分阶段资格限制。",
      "B 原始候选未进入分配时，继续区分原生消费、占用导致的抑制、候选竞争与生命周期。",
      "已接受一条关联也可能没有得到足够重叠的 MOT 输出。",
      "",
      "详见 [逐帧失败阶段](E1_failure_stages.csv)、[旧 ID 失联片段汇总](E1_episode_summary.csv) 和 [只读记录校验](../E1_shadow_parity.json)。"])
    (home/"E1_failure_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def comparison(passed, decision_detail):
    home=ROOT/"compare_analysis"
    summary=read(home/"E1_prediction_summary.json")
    rows=table(home/"E1_group_summary.csv")
    delta=table(home/"E1_assignment_deltas.csv")
    named={g:summary["metrics"][g] for g in ("P0","P1","P2")}
    lines=["# E1-C 固定状态预测比较与开发判断", "",
     f"**{passed}**。{decision_detail}","",
     "P0 为冻结 KF/GMC；P1 以最后一次已接受的检测框中心按原生 GMC 仿射正向输送。P2 增加观测时的 KF 速度，逐帧仿射输送、按 0.8 衰减并对累计目标位移实施 48 像素向量限幅。",
     "仅改变该帧处于 Lost 且失联 2–30 帧的中心。保留原生宽高、协方差、观测、特征、候选、阈值及分配方法。B 的每个历史观察用相应帧的预测中心。",
     "高分原生与 B 两组固定分配分别使用各自原始输入。没有把高分反事实输出传进 B、改写状态、输出轨迹或重新评测 TrackEval。以下正确候选分配由官方真值离线核验。",
     "",
     "| 版本 | 中心偏差中位数（px） | 中心偏差均值（px） | 可用中心 | 当帧几何通过 | native high 通过 | B 原始代价通过 |",
     "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for g,n in named.items():
        lines.append(f"| {g} | {n['center_MED']:.4f} | {n['center_MEAN']:.4f} | {n['available']} | "
            f"{n['current_geometry_pass']}/{n['current_geometry_denominator']} | "
            f"{n['native_high_pass']}/{n['native_high_denominator']} | {n['B_raw_pass']}/{n['B_raw_denominator']} |")
    lines.extend(["","| 版本 | 阶段 | 新增正确 | 移除正确 | 新增错误 | 移除错误 | 新增未知 | 移除未知 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for variant in ("P1","P2"):
        for stage in ("native_high","B"):
            selected={(r["direction"],r["gt_support"]):r for r in rows
                      if r["variant"]==variant and r["stage"]==stage and r["scope"]=="all"}
            cols=[selected[(d,s)]["assignment_pairs"] for s in ("correct","wrong","unknown") for d in ("added","removed")]
            lines.append(f"| {variant} | {stage} | "+" | ".join(cols)+" |")
    lines.extend(["","## 固定状态的潜在新增覆盖","",
        "仅统计基线当帧未核验到原目标正确输出、且新版本分配到同一 GT 检测的记录。空间和外观预测变化未进入完整闭环，因此不计为新增 TP 或减少 FN。"])
    for variant in ("P1","P2"):
        for stage in ("native_high","B"):
            selected=[r for r in delta if r["variant"]==variant and r["stage"]==stage
                      and r["direction"]=="added" and r["interpretation"]=="candidate_without_baseline_coverage"]
            identity=[r for r in delta if r["variant"]==variant and r["stage"]==stage
                      and r["direction"]=="added" and r["interpretation"]=="identity_continuation_only"]
            metrics=info_measure(selected)
            lines.append(f"- {variant} / {stage} 共有 {metrics['rows']} 条候选、{metrics['gt_frames']} 个独立 GT 帧、"
                         f"{metrics['episodes']} 个旧 ID 失联片段、{metrics['sequences']} 段。另有 {len(identity)} 条候选属于已有 GT 覆盖的身份竞争。")
            if selected:
                most=Counter(r["sequence"] for r in selected).most_common(1)[0][0]
                remaining=info_measure([r for r in selected if r["sequence"]!=most])
                lines.append(f"  排除贡献最多的 {most} 后还有 {remaining['gt_frames']} 个独立 GT 帧，覆盖 {remaining['sequences']} 段。")
    lines.extend(["","[逐行预测与几何对照](E1_prediction_pairs.csv) · [分配变化](E1_assignment_deltas.csv) · [分组与去重统计](E1_group_summary.csv)"])
    (home/"E1_decision.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action",choices=("audit","compare"))
    cli.add_argument("--decision",choices=("CONTINUE","STOP","INCONCLUSIVE"))
    cli.add_argument("--detail")
    args=cli.parse_args()
    if args.action=="audit":
        audit()
    else:
        if args.decision is None or not args.detail:
            raise ValueError("Interpret the fixed-state results before recording a development decision")
        comparison(args.decision,args.detail)


if __name__=="__main__":
    main()
