# TCMO v0.1 结果与论文图表回传模板

状态字段：DESIGN_READY / BLOCKED_ASSET / INPUTS_VERIFIED / SOFTWARE_TESTED / SMOKE_PASS / R0_COMPLETE / R1_FULL_COMPLETE / FAILED_TECHNICAL。单测通过不等于科学收益；目前 NOT_RUN。
公开目录 results/vttiny/tcmo_v0.1/<actual_date>/。README 写实际 commit、输入 scope、哪些组重跑/复用、权重与来源、真实运行失败、配置变更和完整结论。

## R0 必交
inventory.json、aux_manifest.json、split_receipt.json、calibration_lock.json、mapping_receipt.json、off_parity.json、r0/summary.json、all_fn_opportunities.csv、eligible_fn_opportunities.csv、per_sequence.csv、per_source_group.csv、size_strata.csv、mapping_quality.csv、identity_and_rejection.csv、case_index.csv。区分分母(全部 FN vs 合格有锚轨迹 FN)、原检测分带、RGB/IR native/映射/proxy/posterior、已知身份/unknown/ignore、实际一对一覆盖、IR 原生有 GT 但 RGB 无 GT。
## R1 必交
comparison.csv（M0–M3/C1/C2 和历史 BG_lower_low 单列）、component_deltas.csv（7 个固定比较）、paired_per_sequence.csv（180 个科学组序列组合）、trackeval/summary.json 和每组 object_detailed.csv、exposure.csv、attribution.csv、source_usage.csv、coverage_overlap.csv、events_by_sequence.csv、cases/ANALYSIS.md、manifest.json/run_provenance.json/TECHNICAL_VALIDATION.md。统计 HOTA, DetA, AssA, IDF1, MOTA, FP, FN, IDSW；同 GT 帧新增、丢失、保持，FN_ref−FN_arm=新增−丢失；记录 B 恢复事件、1/3/5/10帧同身份保持、未匹配输出延续、来源预算、候选缺失、R 反馈。跨组轨迹 ID 不可直接相等比较，按 GT key/事件对应；只用 CLEAR 不能断言海浪背景。逐序列正负零、留一敏感性用正确官方聚合；同状态反事实和独立回放分列。C1 附计算/候选增量、新生/去重，C2 附越界和共同有效样本，避免把 raw 模态收益误归给结构。
## 论文图表规划（数值未产生，不填虚构数据）
| ID | 图表 | 来源 |
|---|---|---|
| M-F1 | TCMO 总框架及 W/TCMO 共同事务 | 冻结真实实现，结构图非实验证据 |
| M-F2 | F/V/T 独有及共有补充机会 UpSet，全部 FN/合格 FN 两面板 | R0 去重 GT keys |
| M-F3 | 跨模态中心误差分布、尺寸分层、IoU=.5 跨线 | R0 A0；R2 A1/A2 后补 |
| M-F4 | 固定与自适应噪声中心更新示意及实测校准 | R2，区分示意与真实分布 |
| M-T1 | M0–M3、C1、C2 完整指标表；IR 条件增量 | R1 |
| M-F5 | ΔHOTA/ΔIDF1 与 FP/FN 权衡，逐序列贡献 | R1 |
| M-F6 | 同步 RGB/IR 正反例，主检测缺失、单模态观测、映射、最终轨迹和后续帧 | R0/R1 真实 GT 与输出，本机生成 |
| M-T2 | M3/M4/M5/M6 机制消融 | R2 待定 |
保留真实原始坐标、帧号、原始置信度、源标识、GT、匹配 ID/阶段、映射前后、实际弱提交与下一帧轨迹，图卡使用相同视野/尺寸/缩放，IR 未配准不得直接叠 RGB GT。示意与实测严格分开。PDF/SVG 矢量、PNG 300–600dpi、source csv、绘图脚本、figure manifest、数据/图 hash。原图、GT、MOT、权重、cache、私有路径留本地；上传获得授权的派生记录，公开脱敏说明。不能从预期流程生成虚构真实图。
