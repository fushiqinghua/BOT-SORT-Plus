# v0.4 第二批实验归档

实验依据为 [执行计划](../../../../../docs/botsort_plus/diagnostics_v0.4/NEXT_ROUND.md) 的 `c3e142e6` 版本。30 段、12,590 帧均沿用冻结 `B_early_moderate`，按开发集口径解读。P0 复现历史指标，E1 和 E2 的方案判定 **STOP**。E3 完成了全量误报归因和按原视频留出的代理检验，最终判定 **INCONCLUSIVE**。

| 阶段 | 关键结果 | 结果文件 |
| --- | --- | --- |
| P0 冻结核验 | 30 段的 MOT、原有事件、完整状态流、实际解码像素、GT 与评测版本均核对。复现 HOTA 38.8450、IDF1 44.0596、MOTA 25.6942。 | [完整性证明](P0/verification.json) · [逐序列及汇总](P0/trackeval/summary.json) |
| E1-A/B Lost 队列与归因 | 主队列 7,792 个轨迹帧。1,407 个有可用检测且候选唯一的缺失项中，1,098 个对应未消费低分框，其中 596 个卡在 B 历史不足。旁路重放通过原生轨迹和事件的逐字节核验。 | [队列统计](E1/cohort/cohort_summary.json) · [E1-B 错因](E1/audit_analysis/E1_failure_report.md) · [旁路等价性](E1/E1_shadow_parity.json) |
| E1-C 固定状态对照 | 原生、GMC 运送、GMC 加衰减速度的中心误差中位数为 2.611／5.800／3.960 px。P2 高分阶段正确分配新增／移除为 39／78，B 为 23／33。 | [E1 决策](E1/compare_analysis/E1_decision.md) · [分配变化](E1/compare_analysis/E1_assignment_deltas.csv) |
| E2 固定补帧坐标 | 固定 191 个提名，仅修改补框坐标。TP 72→73，原 ID 首次中断补回 49→50，MOTA 25.6681，冻结 B 为 25.6942。 | [E2 报告](E2/E2_report.md) · [全量指标](E2/full/E2_metrics.csv) · [逐框对照](E2/full/E2_box_pairs.csv) |
| E3 原生误报审计 | 精确复现 15,943 个 FP。6,563 个同目标定位不过线，3,158 个来自 41 条持续缺少 GT 支持的轨迹。 | [E3 报告](E3/E3_report.md) · [CLEAR 核验](E3/A/E3_A_evaluation_audit.json) · [逐行误报](E3/A/E3_native_fp_rows.csv.gz) |

## E3 样例核对状态

| 环节 | 状态 | 文件 |
| --- | --- | --- |
| 模型辅助 RGB 目视核对 | **已完成**。22 个样例均有独立的建议标签和视觉理由。10 个持续缺少 GT 支持样例为 6 个背景特征、1 个疑似漏标、3 个不确定。 | [模型辅助表](E3/B/E3_machine_assisted_review.csv) |
| 独立人工复核 | **待完成**。22 个正式人工标签均为 `pending`。样例编号、帧和图卡路径保留在表格中。 | [人工标注表](E3/B/E3_manual_review.csv) · [样例协议](E3/B/review_contract.json) |
| 原视频留出 | **代理检验已完成，正式可分性 INCONCLUSIVE**。基于持续缺少 GT 支持代理，结果仍受可能漏标与不确定案例影响。 | [信号统计](E3/C/E3_signal_analysis.csv) · [原视频分折](E3/C/E3_signal_heldout_folds.csv) · [状态](E3/review_status.json) |

[汇总报告](report.md) · [任务验收](task_status.json) · [复现说明](REPRODUCE.md) · [文件与哈希清单](EVIDENCE_MANIFEST.json)

本归档保存实验结果、关键逐行分析和复现脚本。完整输入、轨迹状态、原始 RGB、人工复核图卡及全量中间重放随原实验环境或本地证据包保存。E3 图卡按原方案可由归档脚本和原始输入重新生成。
