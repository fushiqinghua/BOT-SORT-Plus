# M3OT 检测覆盖与诊断记录

M3OT 的本轮诊断使用官方 RGB Val 的 `1-08`、`2-08`，各 600 帧、10 FPS，车辆标注合计 13,781 个目标帧。下文的检测库存、候选质量与冻结回退核验使用各自的统计单位。经过闭环回放或正式输出评分的 D3、E2、TCO P2/P3 在跟踪指标总表中另有记录。

## 强弱检测输入的覆盖

弱档和强档分别使用同一次 YOLO11-S RGB-T 训练的 `epoch0.pt` 与 `best.pt`，记录的验证 AP50 为 0.51216 和 0.79987。两档使用相同的 Val 图像与导出参数，最终缓存的置信度下限为 0.05。覆盖核验在每帧先最大化 IoU ≥ 0.50 的一对一匹配数，再最大化匹配 IoU 之和。高分为 `score > 0.50`，低分为 `0.10 < score < 0.50`。“可关联”取两档分数的并集，仅表示检测进入该分数范围。“仅有低分候选”按真值框查询符合 IoU 的候选，不执行一对一指派。

| 数据范围 | 真值目标帧 | 全缓存召回 | 可关联召回 | 高分召回 | 仅有低分候选 |
| --- | ---: | ---: | ---: | ---: | ---: |
| M3OT 弱检测 | 13,781 | 85.12% | 81.58% | 43.47% | 38.10% |
| M3OT 强检测 | 13,781 | 89.65% | 87.96% | 74.40% | 13.56% |
| VT-Tiny-MOT 历史 RGB-T，30 段 | 179,772 | 42.59% | 40.48% | 31.22% | 9.30% |

在同一 M3OT Val 上，达到起轨门槛 0.60 的一对一召回从 31.68% 增至 68.54%。可关联分数范围中未被一对一匹配的检测从 37,482 个降至 5,847 个。在有匹配观测夹住的内部缺口中，超过一秒的区间从 23 个减至 17 个。强弱检查点同时改变了覆盖、分数与背景候选分布。VT-Tiny-MOT 行提供历史覆盖参照，其七类合并口径、目标大小和场景与 M3OT 车辆组不同。M3OT Val 曾用于检测器选模，本轮按开发验证解读。[分层覆盖与输入一致性](evidence/coverage/comparison.json)

## W 弱证据候选库存

固定 M3OT 强检测器，按原研究代码的分数区间扫描两段完整缓存。S0 定义为 `score ≤ 0.10` 的缓存检测。两个序列在 `score = 0.50` 的分界处均无遗漏行。

| 序列 | 原缓存检测行 | 原生高分 | 原生低分 | S0 检测行 | 含 S0 的帧 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1-08` | 9,226 | 5,755 | 2,316 | 1,155 | 378 |
| `2-08` | 11,212 | 6,567 | 3,331 | 1,314 | 472 |
| 合计 | 20,438 | 12,322 | 5,647 | 2,469 | 850 |

按现有 RGB ReID 导出策略，S0 行中符合导出门槛的数量为 0。将 S0 纳入离线逐帧一对一匹配，使强档匹配名额从 12,122 增至 12,355，增加 233 个。这个差值衡量候选几何供给。W 的原轨迹资格、候选所有权、外观补充以及后续状态影响尚待闭环验证。该阶段的产物为候选库存和覆盖检查。[S0 库存汇总](evidence/wem/summary.json) · [两段明细](evidence/wem/per_sequence.csv) · [输入指纹](evidence/wem/input_verification.json)

## TCO 冻结回退与候选质量

以下两套底座分别是 `B_early_moderate` 和含 B、自适应 GMC 与长间隔保护的 `evidence_both`。P1、Q2 absolute 和 Q2 paired 沿用 VT-Tiny-MOT 归档的回退配置。每种模式均完成两段回放。分别有资格评估 455／989 个低分匹配和 456／992 个低分匹配。六组回放均采用 0 次修正，输出轨迹及原有事件在两段均与各自关闭组逐字节一致。

离线候选核对按同一帧、同一轨迹和同一检测行与关闭组对应，并要求关闭组已取得官方 CLEAR 匹配。两个 Q2 质量头在同一事件上的候选框逐元素相同，使用不同的候选评分。表中的向上和向下是候选框替换原检测框后与同一真值的 IoU 穿过 0.50 的次数。每个底座独立统计，以免把同一场景重复当作独立证据。

| 冻结模式 | 底座 | 可对齐真值的候选数 | IoU 向上 | IoU 向下 | 实际采用 |
| --- | --- | ---: | ---: | ---: | ---: |
| P1 | B | 1,054 | 1 | 7 | 0 |
| P1 | B＋GMC | 1,053 | 1 | 7 | 0 |
| Q2 absolute | B | 1,054 | 9 | 1 | 0 |
| Q2 paired | B | 1,054 | 9 | 1 | 0 |
| Q2 absolute | B＋GMC | 1,053 | 9 | 1 | 0 |
| Q2 paired | B＋GMC | 1,053 | 9 | 1 | 0 |

这些记录提供了回退行为和候选几何的核验。Q2 开启后的全程跟踪增量仍待有效门槛与闭环评估。P2/P3 使用 VT 训练侧已冻结的模型及门槛，在两个底座上分别实际替换了 11 次和 9 次量测。其历史验收未通过，本次作为探索性迁移计分。两段合并时 FP、FN、IDSW 与对应关闭组相同，HOTA 的绝对差不超过 0.002 个百分点。[候选质量与采用记录](evidence/tco/tco_proposal_diagnostics.csv) · [冻结回退凭据](evidence/tco/)

## C 系列与 D/E 状态

| 研究 | 本轮 M3OT 取得的证据与适用范围 |
| --- | --- |
| RGB 质量记忆 C、C2 | RGB 质量记忆 C 已按归档重跑，C2 已完成第二批及后续变体评估，在跟踪指标中计入对应配置。 |
| 原始双模态 C、ABC、固定双模态特征策略 | 当前冻结 ReID 特征来自 RGB CLIP-VeRi。完整实现依赖独立热红外 ReID 编码器和相应坐标标定，这些资产尚缺。早期 VT-Tiny-MOT 归档也未完成三项的全量验证。在强 RGB-T 检测或红外位置诊断上使用图像属于各自的已验证实验。 |
| D1、D2、D4 | 原 VT-Tiny-MOT 资料分别提供预测与观测误差、同状态 SOF 补偿分析，以及图块质量和外观分布。从这些只读分析派生出的可移植策略另列跟踪评分。D4 的正式可比人工标签仍待补充。 |
| D3 | 相邻几何、恢复后弱更新、只补帧和十二种筛选规则均有 M3OT 配对回放。筛选组删除全部补出行后，与原生 MOT 输出逐字节一致。 |
| E1 | 原研究在 VT-Tiny-MOT 已完成 Lost 归因和固定状态预测对照。该部分属于状态审计。为特定视频帧或轨迹设计的干预保留在原证据范围。 |
| E2 | M3OT 已复现 B 的影子轨迹，并在与 D3 只补帧版相同的 25 个提名上离线替换框坐标。其输出单独计分，相对 D3 只补帧版 HOTA −0.0019、IDF1 +0.0074 个百分点，FP −1、FN −1、IDSW 持平。 |
| E3 | 原 VT-Tiny-MOT 数据上的原生误报审计、模型辅助核对及按母视频留出的代理检验已完成。正式人工复核仍待提交。M3OT 部分仅归档研究状态。 |

[E2 影子与输入凭据](evidence/de/e2_receipt.json)

## 证据来源

下列路径对应实验归档的原始来源。公开附件使用同一份数值、哈希和事件计数。

| 证据 | 原实验中的相对路径 |
| --- | --- |
| 检测覆盖、尺寸分层与缺口统计 | `m3ot_first_batch/coverage/results/comparison.json`，及其下一级 `m3ot_lower/`、`m3ot_higher/`、`vt_tiny_historical/` 中的 `summary.json`、`size_bins.csv`、`gap_summary.csv` |
| W 库存与分段输入核验 | `m3ot_branch_replay/wem/results/{summary.json,per_sequence.csv,input_verification.json}` |
| TCO 候选离线核对 | `m3ot_branch_replay/learned/results/tco_proposal_diagnostics.csv` |
| TCO 回退逐序列校验 | `m3ot_branch_replay/learned/results/tco_receipts/` 中 `tco_p1_*`、`tco_q2_absolute_*`、`tco_q2_paired_*` 的 JSON |
| D3 与 E2 的 M3OT 回放和指标 | `m3ot_branch_replay/rules/README.md`、`m3ot_branch_replay/rules/results/receipts/{d3_shadow,e2_shadow}.json`、`m3ot_branch_replay/scores/7280433ecb6163c9/metrics.csv` |
| VT-Tiny-MOT 的 D1、D2、D4 统计与复核状态 | 原分支 `results/vttiny/iterations/v0.4-motion-blur/` 下的 `D1_paired_by_group.csv`、`D2_frame_effect.csv`、`D4_quality_strata.csv`、`D4_visual_review.md` |
| VT-Tiny-MOT 的 E1、E3 审计与复核状态 | 同目录 `round2-e1-e2-e3/` 下的 `E1/compare_analysis/E1_decision.md`、`E3/E3_report.md`、`E3/B/E3_machine_assisted_review.csv`、`E3/B/E3_manual_review.csv` |
| C 与双模态依赖 | `m3ot_branch_replay/birth/README.md` |
