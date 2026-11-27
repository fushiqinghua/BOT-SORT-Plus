# VT-Tiny-MOT 的透视关联、遮挡更新与学习过程噪声

这一组实验在原始 BoT-SORT 上分别检验三项改动。全部对照共用 YOLO11-M RGB–T 检测、RGB ReID 缓存和原始 RGB 帧，在 VT-Tiny-MOT 官方测试集 30 段、12,590 帧上计算全类别合并的 TrackEval 指标。原始 BoT-SORT 的 30 份轨迹逐字节复现历史结果。官方测试集曾用于模型和门限选择，本文按开发性对照解读。

后续另将④加入长间隔重连修正版，检验两种机制的交互。学习 R 和神经网络 Q 的后续训练与此处的首轮低维 Q 分开记录。原始实验的脚本、指标表和核验报告均可从下表进入。

| 实验 | 代码与复现说明 | 真实数据结果 |
|---|---|---|
| ③ 透视投影关联、④ 遮挡降权、⑤ 学习 Q 首轮消融 | [单模块实现与复现](../../../../experiments/vttiny_motion_345/README.md) | [首轮报告](../../../../experiments/vttiny_motion_345/results/实验报告.md) · [全量指标](../../../../experiments/vttiny_motion_345/results/tracking_combined.csv) · [输入校验](../../../../experiments/vttiny_motion_345/results/input_audit.json) |
| ④ 与长间隔重连修正版组合 | [组合实现与复现](../../../../experiments/vttiny_evidence_overlap/README.md) | [组合报告](../../../../experiments/vttiny_evidence_overlap/results/实验报告.md) · [全量指标](../../../../experiments/vttiny_evidence_overlap/results/tracking_combined.csv) |
| ④ 组合新增误报的逐段干预与原生速度对照 | [诊断实现与复现](../../../../experiments/vttiny_overlap_forensics/README.md) | [三段归因与全量复核](../../../../experiments/vttiny_overlap_forensics/results/实验报告.md) · [全量逐序列指标](../../../../experiments/vttiny_overlap_forensics/results/full_velocity/tracking.csv) |
| ⑤ 后续神经网络 Q 和 R，以及高低分 R 归因 | [后续研究索引及复现](neural-qr/README.md) | [神经网络 Q/R 首轮](../../../../experiments/vttiny_neural_qr/results/实验报告.md) |

## ③ 透视变化下的双向投影匹配

HomView-MOT 对前后帧目标框进行正向投影和逆向投影，取两个 IoU 的平均。论文表 III 中，保留 FHE 和 VCIL 后增加 HMF，VisDrone 的 IDF1 从 70.9 升到 75.1，换号从 1,322 降至 1,073。论文的快速估计使用后续关键帧，本实验只从当前帧和过去帧提取背景对应点，并让原生 SOF 负责卡尔曼状态传输。先比较仿射与单应候选，再固定同组矩阵比较正向和双向 IoU。[HomView-MOT 表 III、公式 14 与算法 1](https://arxiv.org/html/2403.10830)

12,590 帧中有 142 帧选择单应、938 帧选择替代仿射、11,510 帧沿用原生 SOF。选择单应的帧中，125 帧至少两个象限呈现相反方向的参考残差。544 个有效帧—象限样本的残差向量长度中位数从 0.629 降至 0.264 像素。全量 HOTA 从仅仿射的 37.732 升至加入单应候选的 37.783，原始 BoT-SORT 为 37.861。

使用同组矩阵时，双向 IoU 改变了部分候选代价。138,099 个候选对中，9,335 个代价变化超过 0.000001，16 个超过 0.01，有重叠候选行的几何最优列变化为 0，最终 30 份输出与正向组逐字节一致。[配准逐段统计](../../../../experiments/vttiny_motion_345/results/perspective_by_sequence.csv) · [双向代价核验](../../../../experiments/vttiny_motion_345/results/bidirectional_cost_audit.json)

## ④ 遮挡时减弱低分观测对状态的拉动

OA-SORT 的 BAM 使用上一观测的遮挡程度及当前预测与检测的 IoU 调节低分观测更新。论文表 4 的 DanceTrack-val 上，从遮挡感知关联进一步加入 BAM，HOTA 从 59.9 升到 60.5，IDF1 从 60.9 升到 62.2。本实验只对已匹配的低分观测缩放原生均值修正，分别比较当前 IoU、前帧重叠面积，以及前帧底边筛选。官方数据缺少实例深度标注，底边线索用下一帧偏框误差间接检查。[OA-SORT 第 4 节与表 4](https://arxiv.org/html/2603.06034v1)

上帧重叠面积版相对原始 BoT-SORT 的 HOTA 变化为 **+0.0072** 个百分点，IDF1 **+0.0868** 个百分点，FP **−8**，FN **−46**，IDSW **−18**。在同一批正确匹配的低分事件中，原始检测比先验偏得更远时，更新后平均中心误差从 1.431 降至 1.344 像素。原始检测至少与先验一样近时，从 1.267 升至 1.357 像素。训练侧留出的底边代理检验只有 52.8% 的配对符合预期方向。[低分事件分层](../../../../experiments/vttiny_motion_345/results/crowded_low_bias_strata.csv) · [底边代理检验](../../../../experiments/vttiny_motion_345/results/bottom_order_training_proxy.csv)

在长间隔重连修正版 `evidence_both` 上加入同一更新，换号 **352→340**、漏检 **117,234→117,213**、误报 **15,945→16,016**，HOTA 从 38.8596 变为 38.8577。三段典型序列合计净增 60 个误报，其中直接属于 B 接回连续片段的净增为 61，直接属于原生关联连续片段的净减为 1。结合定点撤回和阻断证据可分别看见低分状态更新的上游影响，以及 B 对误报的延续。只对位置与尺寸降权、速度沿用原生更新的方案，在焦点三段少 20 个误报，但完整 30 段比原组合多 4 个误报、2 个漏检、1 次换号，HOTA 和 IDF1 均下降。[组合对照](../../../../experiments/vttiny_evidence_overlap/results/实验报告.md) · [干预和全量复核](../../../../experiments/vttiny_overlap_forensics/results/实验报告.md)

## ⑤ 按运动状态拟合过程噪声

ACNTrack 使用神经网络动态估计 Q 和 R，并报告 VT-MOT 上的多模态整体结果。VT-Tiny-MOT 的首轮实验只对 Lost 状态下的平面加速度过程噪声做低维拟合，由官方训练来源序列按母视频划分拟合和留出组。状态转移、R 和关联门限沿用同一原生对照。早期 D1 的 11–30 帧失联覆盖率约 59.7%，其基线是 B_early_moderate，样本配对与下列 67.28% 参考点不同。[ACNTrack 原文](https://doi.org/10.1016/j.neucom.2025.130811) · [出版方摘要](https://www.sciencedirect.com/science/article/abs/pii/S0925231225014833)

在 4,352 个三组共同的真实失联 11–30 帧预测上，运动相关 Q 将标称 95% 位置椭圆覆盖从 **67.28% 提至 85.50%**，椭圆面积中位数从 **525.9 增至 2,408.1 px²**，平均 NLL 从 15.426 降至 7.257。常数 Q 的覆盖达到 100%，椭圆中位面积达到 680,143.8 px²。运动 Q 在共同找回事件的 +3、+5、+10 帧，持续保号变化分别为 **0、−2、+1**，全量 HOTA 相较原生下降约 0.045 个百分点，IDF1 下降约 0.069 个百分点。覆盖率、所需面积和身份收益需结合判断。[真实失联覆盖](../../../../experiments/vttiny_motion_345/results/official_gap_calibration.csv) · [找回后原 ID](../../../../experiments/vttiny_motion_345/results/recovery_identity.csv) · [训练侧留出](../../../../experiments/vttiny_motion_345/results/train_gap_calibration.csv)

## 复现边界

各实验 README 列出运行顺序和输出位置。下面的路径相对于研究项目的 `${VT_INTEGRATIONS}`，用于恢复与服务器一致的目录布局。发布的核验副本用 `${VT_PROJECT}` 表示项目根目录，保留数据与代码指纹、逐序列指标、汇总报告和诊断性统计。

| 实验入口 | 部署脚本的历史目录名 | 完整复现依赖 |
|---|---|---|
| `vttiny_motion_345/workflow.py` | `native_motion_345_v1` | `botsort_plus_v0_4_motion_blur/BOT-SORT-Plus-followup-c3e142e6` 的冻结跟踪源码与同套件 `v04_followup_round2/P0/motion_blur_v0.4.local.json`；`native_botsort_gmc_factorial_v1/full/protocols/native.json` 的参考轨迹哈希。 |
| 首轮训练 Q | 同上 | `botsort_plus_C2_v0_3_0/experiments/kf_measurement_round1` 的 `pairs.pkl`、`truth.pkl`、`inputs.json`；同级 `ir_motion_localization_v1` 的 `window_plan.json`、`training_inputs/detections` 及计划指向的训练 RGB 图像；`${VT_PROJECT}/data/processed_v1/splits.json` 的官方划分。 |
| `vttiny_evidence_overlap/experiment.py` | `b_gmc_evidence_overlap_v1` | `native_motion_345_v1` 的代码与已完成结果；`b_gmc_forensics_r1/live_repo` 的修正版跟踪源码及 `full_runs/evidence_both` 的历史对照；`native_botsort_gmc_oru_v1/vendor` 中固定的 LightGlue 代码与权重。 |
| `vttiny_overlap_forensics/workflow.py` | `b_gmc_evidence_overlap_v1_forensics` | 相邻 `b_gmc_evidence_overlap_v1` 的代码与全量回放结果，以及上述修正版和冻结输入。 |

在原研究环境中运行这些入口需要项目持有者提供冻结检测、RGB ReID、原始帧、官方 GT、`${VT_PROJECT}/external/TrackEval` 和 `botsort_plus_v0_1_0/python_user` 的依赖。冻结跟踪源码的起点和长间隔重连的部署脚本分别见 [c3e142e6](https://github.com/fushiqinghua/BOT-SORT-Plus/tree/c3e142e6c11df0f2a7e8a84bede8556c7dabec1e) 与 [历史重连实验](../../../../experiments/vttiny_gmc/b_gmc_recovery/forensics/full_live.py)。
