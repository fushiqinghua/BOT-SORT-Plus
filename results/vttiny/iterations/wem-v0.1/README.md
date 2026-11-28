# W v0.1 结果归档

**R3.1 的 74 段离线对照已完成。785 个未知候选组待人工复核，新 R4 暂缓。** [本次结果](../../wem_r3_1/full_20261128/README.md)

[当前 R3.1 实验安排](../../../../docs/botsort_plus/wem_r3_1/EXPERIMENT_PLAN.md) · [运行手册](../../../../docs/botsort_plus/wem_r3_1/LOCAL_RUNBOOK.md) · [新结果入口](../../wem_r3_1/README.md)

分支初始状态为 `NOT_RUN`。以下保留当时的 P0/R1a 回传说明，后续 R3 与 R4 归档见文末。

第一批执行后，建议在 r1a_s0/ 子目录归档：

- manifest.json：输入摘要、方法参数、代码与执行范围；
- summary.json：仅候选库存统计；
- per_sequence.csv：逐序列结果；
- 必要的 S0 候选明细或可追溯压缩包说明；
- 对应的解释和下一步决定。

不要把库存数量填作可恢复 FN，不要创建假的 HOTA/IDF1 数值。保留原始本地输出；公开清单中的本机路径可别名化并注明转换。

后续结果按 r0/、r1b/、r2/、r3/、r4/ 分批归档，每批附输入/代码/配置指纹。失败或零采用也归档，不覆写旧结果。

[结果模板](../../../../docs/botsort_plus/wem_v0.1/RESULTS_TEMPLATE.md) · [实验计划](../../../../docs/botsort_plus/wem_v0.1/EXPERIMENT_PLAN.md) · [运行手册](../../../../docs/botsort_plus/wem_v0.1/LOCAL_RUNBOOK.md)

## R3 已归档结果

训练侧校准和 113 例模型辅助逐帧复核已归档，人工确认仍为 0/113。校准与留出分开记录，正式准入继续拒绝全部弱框；后续 R4 探索性闭环另行归档。[R3 结果与复现](../../wem_v0.1/r3/README.md)

## R4 已归档结果

完成三段开关核验、三段试跑和完整 30 段、八组探索性闭环。评测包含逐序列指标和逐帧归因。R3 冻结决策继续生效。[R4 结果与复现](../../wem_v0.1/r4/README.md)

