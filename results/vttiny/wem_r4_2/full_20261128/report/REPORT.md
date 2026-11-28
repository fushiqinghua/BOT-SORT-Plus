# W R4.2 定型实验结果

状态：`FULL_EVALUATION_COMPLETE`。覆盖 30 段、每组 12,590 帧、5 个实验组。

总体指标采用 TrackEval COMBINED_SEQ；百分制指标差值为百分点，不取逐序列百分数平均。
BG 是冻结的 evidence_both 复合底座；本批不把 BG−B 解释为纯 GMC 贡献。

## 总体结果

所有组均相对 BG_off。W0 为既有完整受约束续接；lower_low 只降低原生低分阈值；relaxed_anchor 放宽独立锚点支持；emit_only 仅输出弱观测，保留原生未来状态。

| 组别 | HOTA | IDF1 | MOTA | ΔHOTA | ΔIDF1 | ΔMOTA | ΔFP | ΔFN | ΔIDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BG_off | 38.8596 | 44.0828 | 25.7220 | +0.0000 | +0.0000 | +0.0000 | +0 | +0 | +0 |
| BG_W0 | 39.0339 | 44.2372 | 25.6842 | +0.1743 | +0.1544 | -0.0378 | +578 | -510 | +0 |
| BG_lower_low | 39.2564 | 44.7332 | 25.4322 | +0.3967 | +0.6504 | -0.2898 | +2173 | -1655 | +3 |
| BG_W0_relaxed_anchor | 39.0235 | 44.1730 | 25.4823 | +0.1639 | +0.0903 | -0.2397 | +1027 | -605 | +9 |
| BG_W0_emit_only | 38.9502 | 44.1805 | 25.7037 | +0.0905 | +0.0977 | -0.0184 | +367 | -334 | +0 |

## 定型对照

差值方向固定如下。每个组独立运行，后续候选和关联可随状态分叉；W0−emit_only 衡量状态反馈带来的完整影响，不单独证明某一 KF 公式的贡献。

| 对比 | ΔHOTA | ΔIDF1 | ΔMOTA | ΔFP | ΔFN | ΔIDSW | 轨迹发生变化的序列 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BG_W0_minus_BG_off | +0.1743 | +0.1544 | -0.0378 | +578 | -510 | +0 | 25 |
| BG_W0_minus_BG_lower_low | -0.2224 | -0.4960 | +0.2520 | -1595 | +1145 | -3 | 28 |
| BG_W0_relaxed_anchor_minus_BG_W0 | -0.0104 | -0.0641 | -0.2019 | +449 | -95 | +9 | 22 |
| BG_W0_minus_BG_W0_emit_only | +0.0838 | +0.0567 | -0.0195 | +211 | -176 | +0 | 25 |

## 弱输出与状态更新

旧事件名称 committed 在 emit_only 中表示弱观测输出已接纳，不表示未来跟踪状态被更新。下表把两者明确分开。

| 组别 | 弱观测输出数 | 弱观测写入未来状态数 | 计数依据 |
| --- | ---: | ---: | --- |
| BG_off | 0 | 0 | derived_from_committed_count_and_fixed_arm_semantics |
| BG_W0 | 730 | 730 | derived_from_committed_count_and_fixed_arm_semantics |
| BG_lower_low | 0 | 0 | derived_from_committed_count_and_fixed_arm_semantics |
| BG_W0_relaxed_anchor | 946 | 946 | recorded_and_checked |
| BG_W0_emit_only | 701 | 0 | recorded_and_checked |

## 放宽锚点后是否真的改变行为

该消融移除 ready 条件、锚点与正式预测的一致性门、弱框与锚点的一致性门；保留最近常规观测、预算、所有权及其他安全约束。三个被移除条件作为一组解释，不将其效果分别认领。

| 诊断计数 | 数量 |
| --- | ---: |
| exposure_anchor_not_ready_bypassed_tracks | 525 |
| exposure_anchor_prediction_bypassed_tracks | 131 |
| exposure_anchor_box_bypassed_edges | 31 |
| exposure_anchor_support_bypassed_edges | 196 |
| anchor_support_bypassed_selected | 191 |
| anchor_support_bypassed_committed | 191 |
| exposure_anchor_guard_added_selected | 191 |
| exposure_anchor_guard_removed_selected | 0 |
| exposure_anchor_guard_changed_track_frames | 191 |
| exposure_anchor_guard_added_eligible | 196 |
| exposure_anchor_guard_removed_eligible | 0 |

anchor_guard 对照在 relaxed_anchor 当前状态下重算严格锚点策略，属于同状态反事实；它与独立运行的 W0 全程序列对比有不同含义。
若放宽条件没有引入额外提交且轨迹相同，应记录本批未激活该差异；若轨迹改变而指标持平，应检查变化的收益与代价是否相抵。零差异不自动等于约束无用，也不能作为有效贡献。

## 结果如何使用

- W0 与 lower_low 需要同时看覆盖、FP、IDSW 和主指标，保留更高召回与更少误报之间的真实取舍。
- relaxed_anchor 劣于 W0 支持整组锚点约束的价值；若优于 W0，应报告现有约束过严的证据，不用阈值搜索掩盖结果。
- W0 优于 emit_only 支持未来状态反馈的价值；若相同，不能把当帧补框的收益全部解释成运动状态改善。
- 指标下降不会阻止整批报告生成。方向提示是描述性辅助，没有统计显著或自动科研准入含义。

## 复用与统计边界

- BG_off、BG_W0 可以来自经过输入、代码和文件核验的 R4.1 完整运行；来源保存在 summary.json 的 reference_sources。复用不是新的重复实验。
- emit_only 的 states 文件以原生输出计算，专用于核对未来状态与 BG_off 一致；不能把该哈希称为实际增强输出的状态哈希。
- 导入参考与新运行的耗时条件可能不同，本报告不据此比较 FPS；部署速度应另行关闭诊断后测量。
- 官方 30 段已经用于开发。本批没有新增独立盲测证据。CLEAR 未匹配 FP 不等价于人工确认背景。

[总体指标](comparison.csv) · [逐序列配对](paired_per_sequence.csv) · [组件差值](component_deltas.csv) · [组件逐序列差值](component_per_sequence.csv) · [候选诊断](exposure.csv) · [归因](attribution.csv) · [核验摘要](summary.json)
