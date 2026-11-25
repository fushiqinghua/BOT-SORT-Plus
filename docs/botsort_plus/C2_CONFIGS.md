# C2 配置与消融矩阵

## 配置对应关系

以下配置均使用冻结的检测、RGB ReID 特征和原生参数。新配置显式选择 `adapter_mode="v2"`，关闭 A、A2 身份准入和旧 C。B_ref 继续作为主要对照。

| 文件 | 运行名称 | B | 关联约束 | 写入门控 | 观察模式 | 用途 |
|---|---|---:|---:|---:|---:|---|
| `12_baseline_ref.json` | `baseline_ref` | 关 | 关 | 关 | 否 | 固定原生基线 |
| `13_B_ref.json` | `B_ref` | 开 | 关 | 关 | 否 | 已有 B 的同基座对照 |
| `27_B_C2_shadow.json` | `B_C2_shadow` | 开 | 拟议 | 拟议 | 是 | 与 B_ref 逐文件一致，记录拟议门控 |
| `28_B_C2_memory.json` | `B_C2_memory` | 开 | 关 | 开 | 否 | 单独检验写入资格 |
| `29_B_C2_association.json` | `B_C2_association` | 开 | 开 | 关 | 否 | 单独检验外观加成约束 |
| `30_B_C2.json` | `B_C2` | 开 | 开 | 开 | 否 | 主方案 |
| `31_C2.json` | `C2` | 关 | 开 | 开 | 否 | 检验 C2 对 B 的依赖与互补性 |

最小主实验为 B_ref、B_C2_memory、B_C2_association、B_C2。另加 baseline_ref 和 C2 可以观察 B 与 C2 的互补关系。B_C2_shadow 是接线/归因验收，不是新增性能对照方法。

## identity 配置段

完整 C2 的开关写法：

```json
{
  "identity": {
    "enabled": true,
    "observe_only": false,
    "association_enabled": true,
    "memory_enabled": true
  }
}
```

该片段只展示新段；完整运行应选择上述配置文件。`PlusConfig` 严格拒绝未知拼写。所有解析后的默认值会写入运行 `protocols/<name>.json`，可据此复核实际参数。

| 参数 | 默认值 | 作用与解释 |
|---|---:|---|
| `min_quality` | 0.35 | 使用当前 RGB 身份证据的最低 crop quality；不是身份概率门槛 |
| `min_history` | 3 | 锚点成为成熟历史所需有效写入数，包含出生时合格的种子 |
| `history_max_distance` | 0.40 | 当前特征与可靠锚点的余弦距离一致性尺度 |
| `appearance_margin` | 0.10 | 双向外观竞争间隔尺度；不以“无竞争者”伪造大间隔 |
| `geometry_margin` | 0.10 | 几何竞争间隔要求 |
| `bootstrap_max_iou_cost` | 0.30 | 冷启动写入所需强几何证据，即 IoU 不低于 0.70 |
| `write_max_iou_cost` | 0.50 | 常规写入几何代价上限，即 IoU 不低于 0.50 |
| `ema` | 0.95 | C2 私有可靠历史锚点的 EMA 历史权重；不替代所有原生 EMA 参数 |

这些值用于可复核的初版实现，尚未由新模块真实实验验证为最优。初轮应固定这组参数运行完整消融，不同时更改检测阈值、B 恢复参数或 ReID 权重。

成熟身份没有可行竞争者时，写入也要求 `bootstrap_max_iou_cost=0.30` 的强几何条件；`write_max_iou_cost=0.50` 并不是所有场景中唯一的写入限制。有竞争者时需要已知且有正间隔的身份证据。具体条件见设计文档和事件 `reason`。

## 读结果时的归因

“仅关联”仍维护 C2 私有历史，但不门控原生及 B 的外观记忆；“仅记忆”保持参考关联公式，却会改变之后被关联和 B 读取的外观模板。因此两项消融不是两条互不影响的物理通道，其共同作用需要用完整方案验证。

建议先检查 shadow 完全一致，然后比较相同输入下的 HOTA、AssA、IDF1、IDSW，并保留 DetA、FP、FN、MOTA。检查各序列同向改善情况，尤其避免把单个高 IDSW 序列的变化视为全部场景的规律。记录新增/丢失的 GT 覆盖和身份断裂变化时，按 `(sequence, GT_ID, frame)` 对齐，不能直接跨运行比较预测 ID。

不把更低的写入数量本身视为成功。需结合可靠写入覆盖率、错误身份写入、拒写后正确续轨、冷启动持续时间，以及 B 恢复前后的模板状态判断机制是否工作。GT 仅用于离线分析，不能进入 runtime。
