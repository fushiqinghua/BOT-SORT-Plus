# W R3.1 离线对照结果

**完整运行已归档。** `research/vttiny-weak-evidence` 上的 W0、W1、W2、W3 和 W_old 使用同一份原 R3 冻结评分与标签，覆盖 74 段、28,931 帧。W0 和 W_old 的六项历史结果核验通过。785 个实际采用过的未知候选组待人工复核。本轮暂缓新的 R4。

| 归档 | 内容 |
| --- | --- |
| [完整运行](full_20261128/README.md) | 五组方法、分区计数和结论 |
| [逐序列](full_20261128/per_sequence.csv) · [逐来源组](full_20261128/per_source.csv) | 各序列和来源组的覆盖及采用 |
| [错配分析](full_20261128/错配分析.md) | 已知错配、误拒和改选案例 |
| [待人工复核队列](full_20261128/review_queue.csv) | 785 个未知候选及其采用方法和人工状态 |
| [复现记录](full_20261128/复现记录.md) | 冻结输入、源码、输出摘要和重跑命令 |

W_old 复现 R3 的余弦 0.85 工作点。此前采用余弦 0.75 的 R4 探索性闭环继续作为单独历史结果。原 R3 `REJECT_ALL_FALLBACK` 保持生效。[实验协议](../../../docs/botsort_plus/wem_r3_1/EXPERIMENT_PLAN.md) · [R3 结果](../wem_v0.1/r3/README.md) · [此前 R4](../wem_v0.1/r4/README.md)
