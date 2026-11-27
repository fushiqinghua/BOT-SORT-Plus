# J3 只读诊断解读

方法 `B_adaptive_no_dual_low`，30 段、12590 帧。GT 仅在本离线分析中读取。

## 条件路径变化

| 路径 | 可比较帧 | 分配/成员变化帧 |
|---|---:|---:|
| C_B_assignment | 12590 | 4 |
| C_observation_members | 12590 | 123 |
| N_B_assignment | 12590 | 3 |
| N_eligible_lost_ids | 12590 | 36 |
| N_high | 12590 | 24 |
| N_low | 12590 | 16 |
| N_residual_indices | 12590 | 37 |
| N_unconfirmed | 12590 | 3 |
| P_B_assignment | 12590 | 11 |

P/N/C 是同一参考历史上的条件诊断，不是三个独立全序列算法；变化数不能相加为净收益。

## 需要查看的证据

- recovery_signals：全部实际 B 提名/拒绝及接受结果，含过去 GT 锚点和精确检测行的离线支持。
- motion_disagreement：逐身份的相机假设分歧；正确和错误恢复都应检查。
- identity_competition：Lost 与活跃身份的诊断支持。仅外观覆盖次数与历史长度相同时报告 lost_minus_best_active；正值表示该活跃身份得分更低。coverage_mismatch_or_unknown 不作强弱判断，未校正差值另列。
- accepted_signal_summary：按原视频和结果分组描述分歧，不训练阈值，不宣称统计显著。

- native_high_reconnections：依据实际 state_before 区分 Lost=2 与其他状态，并联结此前 GT 锚点和相机分歧。
- guard_to_B：按精确 (frame, det_index) 检查被 dual_low 拒绝的边是否进入后来 B 候选历史；保留原 guard 身份、提名/接回身份、后续实测数量和外观支持。
- output_indices：正式输出对应的原始检测行；有 decisions 日志时与 probe 输出逐帧核对。

## 标签与验收边界

same_GT_identity / different_GT_identity 依赖恢复前已确认的 GT 锚点。无旧锚点、未输出及定位/身份混合误差单列。
raw_detection_status 是当前原始检测对官方 GT 的唯一 IoU 支持，不能替代实际分配或物理背景判断。
guard_to_B 只描述已记录历史成员关系；代价下降不代表新增身份支持，未出现成员关系也不等于永远没有后续影响。缺失旧版 decisions 日志明确记 UNKNOWN。
parity.csv 记录内部影子一致性和 trace/MOT 对应；开关前后 MOT 字节一致性仍以 batch parity 产物为准。
进入 J4 前需核对正例保留、错误解释和主要设计视频外的重复证据。本报告不自动选择门限或宣称新模块有效。
