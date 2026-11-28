# W R4.2 定型验证结果

**真实数据状态：`NOT_RUN`。** 本目录是下一批实验的归档入口，目前没有新增实测指标。

本批承接 [W R4.1 完整结果](../wem_r4_1/full_20261128/README.md)，在冻结的 BG = `evidence_both` 底座上检验三件事：与原生降阈值的取舍、锚点约束包的作用、弱观测反馈正式轨迹状态的价值。

| 组别 | 运行安排 |
| --- | --- |
| BG_off | 核验后复用本地完整 R4.1 参考，或显式重跑 |
| BG_W0 | 核验后复用本地完整 R4.1 参考，或显式重跑 |
| BG_lower_low | 新增完整 30 段 |
| BG_W0_relaxed_anchor | 新增完整 30 段 |
| BG_W0_emit_only | 新增完整 30 段 |

每组覆盖 30 段、12,590 输入帧。既有官方测试划分在本项目中沿用开发评测口径。

- [实验设计与后续任务](../../../docs/botsort_plus/wem_r4_2/EXPERIMENT_PLAN.md)
- [本地执行手册](../../../docs/botsort_plus/wem_r4_2/LOCAL_RUNBOOK.md)
- [结果记录模板](../../../docs/botsort_plus/wem_r4_2/RESULTS_TEMPLATE.md)
- [代码与合成验证说明](../../../docs/botsort_plus/wem_r4_2/VALIDATION.md)

完成后在本目录新建带实际运行日期的归档子目录，保留五组完整指标、逐序列对照、组件差值、候选曝光、提交归因和运行来源。复用结果必须保留来源凭据，不能把复制的指标表描述为新运行的轨迹。
