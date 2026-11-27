# J1 四组收益交集与事件账本

覆盖 30 个序列、12590 帧。匹配来自固定 TrackEval CLEAR；逐帧计数与 CLEAR 返回值一致。

G 代表自适应 GMC 与 dual_low 的现有组合。覆盖交集只解释 GT 帧命中，不分解 HOTA 或 IDF1。

## 原始基线漏检集合中的覆盖变化

| 项目 | GT 帧数 |
|---|---:|
| B_repairs_native_miss | 2657 |
| G_repairs_native_miss | 72 |
| both_repair_native_miss | 19 |
| union_repair_preserved | 2688 |
| union_repair_lost | 22 |
| joint_only_repair | 16 |

## 证据范围

缺少事件文件的运行/序列数：120。这些位置不重建关联阶段或检测索引。

association_events 含已记录的原生关联与 B 恢复及后 3/5/10 帧结果；recovery_proposals 保留全部已记录的 B 候选判定。

原生高分的 output_gap_only 只说明输出中断后重连，不能替代内部 Lost 状态证据。

output_rows_CLEAR 的重连来源只是同一运行内的身份沿袭，不能据此宣称某事件因果导致全部后续误报。

后续窗口以事件之后的帧计数；序列末端不补齐，GT 不在场与 FN 分开。未知身份不强制归为错误。

按 original_video_groups 显式分组；现有数据仍属于开发性评估。下一步由 J2/J3 受控回放确定路径。
