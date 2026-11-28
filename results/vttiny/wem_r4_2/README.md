# W R4.2 定型验证结果

**五组已完成。** 冻结的 BG = `evidence_both` 底座复用了 R4.1 的 `BG_off` 和 `BG_W0`，另外回放了降低原生低分阈值、放宽锚点约束、仅输出弱框三组。五组均覆盖官方开发评测的 30 段、12,590 帧。

W0 保留原锚点约束和正式状态反馈，作为定型候选。相对关闭组，W0 的 HOTA 提升 0.1743 个百分点，少漏 510 个，同时多报 578 个，MOTA 下降 0.0378 个百分点。原生阈值降至 0.05 后，HOTA 和 IDF1 更高，误报和换号也更多，作为召回优先的强对照保留。

[完整结果、定型依据与复现](full_20261128/README.md) · [五组指标报告](full_20261128/report/REPORT.md) · [逐帧案例](full_20261128/cases/ANALYSIS.md)

[实验设计](../../../docs/botsort_plus/wem_r4_2/EXPERIMENT_PLAN.md) · [运行手册](../../../docs/botsort_plus/wem_r4_2/LOCAL_RUNBOOK.md) · [软件核验](../../../docs/botsort_plus/wem_r4_2/VALIDATION.md) · [R4.1 结果](../wem_r4_1/full_20261128/README.md)
