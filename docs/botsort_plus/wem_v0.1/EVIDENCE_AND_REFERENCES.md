# 设计依据与文献

## 已有项目证据

设计基点：176f2a81613c34d16a860765ea7be0845116b534。以下归档由本分支继承，W 本身没有新的真实数据结果。

- [TCO 后续总览](../../../results/vttiny/iterations/tco-v0.1/followup/README.md)：P2/P3 留出净修复 −3/−2；Q2 两组校准后采用均为 0；没有开启 TCO 的完整 30 段指标。
- [T2/T3a 报告](../../../results/vttiny/iterations/tco-v0.1/followup/tco_t23/results/T2_T3a_exploratory_report.md)：训练划分、有限困难样本、工作点与量测效果。
- [Q2 冻结质量头归因](../../../results/vttiny/iterations/tco-v0.1/followup/tco_q2/diagnosis/gain_gate/成对监督跌线放行归因.md)：区分单个增量门控诊断与真实联合策略；不能将它改写为 W 的实验结果。
- [B/GMC 第三轮](../../../results/vttiny/iterations/v0.4-motion-blur/round3-b-gmc/README.md)：继承两组参考和额外相机对照。
- [T1 机会说明](../../../results/vttiny/iterations/tco-v0.1/T1/T1_DECISION.md)：几何可达不等于身份已知，也不等于真实新增覆盖。

## 已核对的代码事实

- [adapter_v2.py](../../../botsort_plus/adapter_v2.py)：原生 low 后即 mark_lost，B 可以恢复本帧新 Lost；B 和原生起轨先于 W 设计接入点。
- [feature_export.py](../../../botsort_plus/feature_export.py)：float32 score > 0.1 的导出覆盖与原生精度边界需要区分。
- [RGB-T 检测导出](../../../tools/evaluate_vttiny_m_pair.py)：使用 YOLOMM 后处理结果，继承冻结检测协议；官方开发 30 段的使用事实已披露。
- [Round 3 方法定义](../../../experiments/vttiny_gmc/round3_joint/common.py)：evidence_both 含 adaptive_gmc、dual_low 和证据规则，不是纯 GMC 因子。
- [Round 3 保护条件](../../../experiments/vttiny_gmc/round3_joint/worker.py)：原生几何分歧保护不能被 W 旁路绕过。

## 一手文献

| 工作 | 相关机制与对照意义 | 来源 |
| --- | --- | --- |
| ByteTrack，ECCV 2022 | 轨迹相似性利用低分检测；比较范围需超出原生已使用候选 | [论文](https://arxiv.org/abs/2110.06864) |
| TrackTrack，CVPR 2025 | 高分、低分和 NMS 抑制候选；轨迹视角关联、轨迹感知起轨 | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html) · [作者代码](https://github.com/kamkyu94/TrackTrack) |
| BoostTrack++，2024 arXiv/2025 修订 | 形状、Mahalanobis、soft BIoU 与轨迹条件置信度提升 | [论文](https://arxiv.org/abs/2408.13003) · [作者代码](https://github.com/vukasin-stanojevic/BoostTrack) |
| HGTMT/HGT-Track，TMM 2024 | RGB-T 小目标 ReDet，历史特征与跨模态当前证据 | [论文](https://arxiv.org/html/2412.10861v1) · [作者代码](https://github.com/xuqingyu26/HGTMT) |
| ProgTrack，CVPR 2026 | 小目标上下文特征、全局运动及空间关系 | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_ProgTrack_A_Multi-Object_Tracking_Algorithm_with_Progressive_Matching_Strategy_CVPR_2026_paper.html) |

这些文献是设计依据，不保证本项目上的收益。低分利用、轨迹引导和 NMS 恢复不是独占概念。本研究可验证的增量是限定剩余观测缺口、当前身份竞争和弱更新隔离后的实际效果。

W 与 W-coast 相当不能否定全部弱证据；需同时比较不查询当前弱候选的 Coast-only。若只移植相关论文的一个组件，实验名称标明组件参考，不冒充完整论文复现。

