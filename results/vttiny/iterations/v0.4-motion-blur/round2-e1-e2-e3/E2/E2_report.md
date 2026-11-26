# E2 固定提名的输出坐标对照

本轮决策 **STOP**。全量固定 191 条补帧提名，正确补帧由 72 增至 73。E2-kf 的 MOTA 相对 E2-raw 变化 +0.0011 个百分点，相对冻结 B 变化 -0.0261 个百分点。作为 D3 补帧方向的收尾，冻结 B 继续用于正式对照。

## 实验约束

保留原生 MOT 行和补帧帧号、轨迹 ID、检测索引、分数、总数与顺序。每个提名单独复制首轮实测状态流中 after-GMC 处的 8 维均值和 8×8 协方差，按同帧原检测中心 XYWH 调用一次固定 BoxMOT 16.0.11 的 KalmanFilterXYWH.update，只将后验前四维转换为 MOT TLWH。预测脚本无真值参数，真值只用于随后的独立 TrackEval 与诊断配对。

## 试跑

四段核验覆盖 DJI_0303_2、DJI_0327_3、DJI_0339_2、DJI_0141_2。在 DJI_0303_2 第 58 帧轨迹 2／检测 5，和 GT 1 的 IoU 从 0.473510 变为 0.529298，CLEAR 分配由 未命中变为 命中 GT 1。
首次试跑已成功生成坐标并复现 TrackEval，逐框核验因 IoU 接口输入形状有误中断。保留首次日志和坐标，以原样的坐标重做真值复核，再启动唯一的全量试验。

## 官方 30 段指标

| 方法 | HOTA | IDF1 | MOTA | FP | FN | IDSW |
| --- | --- | --- | --- | --- | --- | --- |
| 冻结 B | 38.8450 | 44.0596 | 25.6942 | 15943 | 117264 | 374 |
| E2-raw 原始检测框 | 38.8608 | 44.0742 | 25.6670 | 16062 | 117192 | 376 |
| E2-kf 复制状态更新 | 38.8611 | 44.0742 | 25.6681 | 16061 | 117191 | 376 |

两组历史基准的每序列和汇总结果均复现，TrackEval 版本和官方真值逐文件哈希一致。比较 raw 与 kf，191 行发生 FP→TP 2 次、TP→FP 1 次、仍命中但 GT 改变 0 次。原生 MOT 行逐字节一致，其中 CLEAR 匹配身份实际变化 0 行。

| 原误差类型 | 提名数 | raw TP | kf TP | FP→TP | TP→FP |
| --- | --- | --- | --- | --- | --- |
| fp_competing_assignment | 1 | 0 | 0 | 0 | 0 |
| fp_displaced_or_different_gt | 18 | 0 | 0 | 0 | 0 |
| fp_localization_same_anchor | 44 | 0 | 2 | 2 | 0 |
| fp_persistent_background | 45 | 0 | 0 | 0 | 0 |
| fp_unconfirmed_no_overlap | 3 | 0 | 0 | 0 | 0 |
| fp_weak_unconfirmed_overlap | 8 | 0 | 0 | 0 | 0 |
| tp_different_from_recent_clear_gt | 2 | 2 | 2 | 0 | 0 |
| tp_same_gt_longer_history_only | 9 | 9 | 9 | 0 | 0 |
| tp_same_id_double_witness | 61 | 61 | 60 | 0 | 1 |

三个跨越 CLEAR 匹配线的原始候选如下。

| 序列 | 帧 | 轨迹 ID | GT | raw IoU | kf IoU | 变化 |
| --- | --- | --- | --- | --- | --- | --- |
| DJI_0061_2 | 52 | 95 | 30 | 0.462348 | 0.504294 | FP→TP |
| DJI_0073_3 | 78 | 53 | 45 | 0.511367 | 0.486345 | TP→FP |
| DJI_0303_2 | 58 | 2 | 1 | 0.473510 | 0.529298 | FP→TP |

## 287 个冻结 D3 窗口

| 方法 | 原 ID 首次中断补回 | 连续 3 帧 | 连续 5 帧 | 连续 10 帧 |
| --- | --- | --- | --- | --- |
| 冻结 B | 0 | 152 | 122 | 77 |
| E2-raw 原始检测框 | 49 | 164 | 124 | 78 |
| E2-kf 复制状态更新 | 50 | 164 | 124 | 78 |

在保留 191 行、原生 CLEAR 分配与冻结 B 一致，且 IDSW 增量为本实验观测值的条件下，补框至少需要 97 个 TP 才能使 MOTA 不低于冻结 B。本轮只做一次坐标变体，30 段为开发评估。

主证据为 `full/E2_row_integrity.json`、`full/E2_box_pairs.csv`、`full/E2_clear_deltas.csv`、`full/E2_fixed_windows.csv`、`full/E2_metrics.csv` 和 `E2_evidence_manifest.json`。
