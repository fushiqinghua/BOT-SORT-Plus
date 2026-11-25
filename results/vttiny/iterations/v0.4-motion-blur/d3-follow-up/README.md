# v0.4 D3 恢复后续轨核验

本页接续 [v0.4 首轮诊断](../README.md)。所有跟踪比较使用冻结参考 `B_early_moderate`、同一 YOLO11-M RGB–T 检测与 RGB ReID 缓存，以及 VT-Tiny-MOT 官方测试 30 段、12,590 帧。固定 D3 口径包含 287 个恢复窗口和其中 210 个首次中断。官方指标使用 TrackEval `12c8791b303e0a0b50f753af204249e622d0281a`。以下按同一测试集上的回顾性消融解释。

## 四组证据

| 阶段 | 结果 | 可复核文件 |
| --- | --- | --- |
| 两组首次中断 | 89 例存在低于原分数线的同目标候选。22 例已有原生分数段的可用框，其中 15 例可定位到关联门控，4 例关联后输出未跨过 IoU 0.50。 | [报告](01-interruption-audit/report.md) · [89 例逐帧证据](01-interruption-audit/outside_score_89/event_frames.csv) · [22 例候选判定](01-interruption-audit/usable_detections_22/candidate_decisions.csv) |
| 有状态低分续接和紧邻帧门控 | 低分续接补回 44 个原 ID 首次中断，全量 FP 增加 406。紧邻帧门控补回 3 个，连续性未改善。 | [报告](02-continuation/report.md) · [全量与 29 段指标](02-continuation/combined_metrics.csv) · [固定事件对照](02-continuation/event_audit/results/paired_events.csv) |
| 只补当前帧 | 相比有状态低分版少 287 个 FP，原 ID 首次中断补回 49 个。移除补帧后，30 段原生轨迹和事件与冻结 B 逐字节一致。 | [报告](03-output-only/report.md) · [全量指标](03-output-only/full/combined_metrics.csv) · [固定事件](03-output-only/identity/full/paired_events.csv) · [191 个补框的真值复核](03-output-only/parity/full/supplemental_rows.csv) |
| 背景误报溯源与输出筛选 | 119 个新增 FP 中，45 个可追到既有背景轨迹。小框且近静止的筛选少 35 个 FP，保住 66/72 个正确框和 46/49 个原 ID 首次中断补回。 | [报告](04-background-filter/report.md) · [191 框错因](04-background-filter/lineage/lineage_rows.csv) · [筛选对照](04-background-filter/lineage/filter_lineage_join.csv) · [全量指标](04-background-filter/metrics/combined_metrics.csv) |

## 复现与归档范围

- [有状态低分续接](02-continuation/implementation/low/short_weak.py)、[紧邻帧门控](02-continuation/implementation/gate/adjacent_gate.py)、[只补帧逻辑](03-output-only/implementation/weak_output.py)和[输出筛选规则](04-background-filter/rules/causal_screen.py)保留实验用实现。各轮报告同目录保留运行、重评及核验脚本。脚本沿用首轮的冻结回放、检测缓存、特征缓存、图像及 GT 目录布局，实际位置由本地环境提供。
- 公开文件保存实验报告、指标、病例核验、策略和实现快照。原始图像、模型和全量原始轨迹事件随原实验环境或本地完整证据包保管。路径前缀以 `${TRACKING_ROOT}` 和 `${REDETR_ENV}` 表示。
- [证据清单](EVIDENCE_MANIFEST.json)记录源压缩包、所选源文件和公开版本的 SHA-256。第二轮报告中 `DJI_0327_3` 轨迹 27 的第 182、183 帧分数已按原始事件更正为 0.095828、0.063791，与第三轮报告一致。
- 正式比较仍以冻结 `B_early_moderate` 为对照。输出筛选是在同一测试集上查看单条件表现后形成的探索性规则。
