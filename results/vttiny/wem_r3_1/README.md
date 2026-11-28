# W R3.1 四组离线对照结果

**本批 R3.1 真实数据状态：NOT_RUN。** 本目录为 W0–W3 四组实验及 W_old 历史参照预留归档入口。代码测试和构造数据验证不能作为本目录的真实实验结果。

本批在现有 `research/vttiny-weak-evidence` 分支中，复用原 R3 的冻结特征评分与离线标签，比较几何、历史运动、外观软排序和相对竞争。固定参数及实际命令见：

- [实验计划](../../../docs/botsort_plus/wem_r3_1/EXPERIMENT_PLAN.md)
- [本地运行手册](../../../docs/botsort_plus/wem_r3_1/LOCAL_RUNBOOK.md)
- [结果回传模板](../../../docs/botsort_plus/wem_r3_1/RESULTS_TEMPLATE.md)
- [原 R3 冻结结果](../wem_v0.1/r3/README.md)
- [已完成的 R4 探索性闭环](../wem_v0.1/r4/README.md)

R4 已归档余弦 0.75 探索点的八组回放，本批 W_old 重算原 R3 的余弦 0.85 对照。R3.1 新增 W0–W3 固定状态消融，研究外观使用方式，不重复执行旧 R4 八组。

## 归档内容

运行后在本目录下建立独立子目录，保存：

| 文件 | 内容 |
| --- | --- |
| `manifest.json`、`summary.json` | 来源、配置、执行范围与整体摘要 |
| `comparison.csv` | 各分区 W0、W1、W2、W3、W_old 的固定方法结果 |
| `per_sequence.csv`、`per_source.csv` | 逐序列和来源组结果 |
| `method_deltas.csv` | 逐项新增机制带来的变化 |
| `review_queue.csv` | 实际采用未知等需复核案例及关联方法 |
| `decisions.jsonl.gz` | 逐候选评分、拒绝和指派明细；体积较大时保留原件与摘要说明 |
| 本次结果说明 | 按模板记录事实、未知项与研究决定 |

本批不覆盖原 R3 的 `REJECT_ALL_FALLBACK`，不自动选出线上启用点，不包含在线 W 更新或新的 HOTA／IDF1。旧 holdout 已用于分析，本批按开发诊断解释。未知候选与人工确认标签继续分开记录。
