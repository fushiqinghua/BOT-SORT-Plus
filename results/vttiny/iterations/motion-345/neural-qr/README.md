# ⑤的后续实验　学习 Q／R 与低分观测归因

这一组实验沿用 ③–⑤ 的原生 BoT-SORT 对照，冻结 YOLO11-M RGB–T 检测、RGB ReID 缓存以及相机补偿。学习 Q 用失联前的连续观测估计附加过程噪声。学习 R 用当前框置信度、几何与同步 RGB/T 图块质量估计测量协方差。训练与内部留出按母视频拆分，在线回放共用此前用于选模及研发的官方测试 30 段、12,590 帧。

## 学习过程噪声与测量噪声

下表为统一 TrackEval 的 `COMBINED_SEQ`，HOTA、IDF1、MOTA 以百分数记。

| 方法 | HOTA | IDF1 | MOTA | FP | FN | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| 原始 BoT-SORT | 37.8608 | 42.4507 | 25.7593 | 13,397 | 119,645 | 422 |
| 学习 Q | 37.7913 | 42.3629 | 25.7510 | 13,415 | 119,648 | 416 |
| 全分数学习 R | 37.7810 | 42.3862 | 25.7281 | 13,261 | 119,836 | 423 |
| 学习 Q＋R | 37.7675 | 42.3891 | 25.7265 | 13,253 | 119,845 | 425 |

- **Q 的校准取舍。** 内部留出集的 20 帧模拟缺口中，中心覆盖率从 86.3% 升至 95.7%，平均椭圆面积从 2,356 扩至 19,648 px²。学习 Q 达到 250 次更新上限时优化器未报告收敛。全量跟踪换号少 6 次，HOTA 与 IDF1 均小幅下降。[校准数据](../../../../../experiments/vttiny_neural_qr/results/q_holdout.csv) · [正式指标](../../../../../experiments/vttiny_neural_qr/results/tracking_combined.csv)
- **R 的输入增量。** 在 20 个留出母视频的 7,880 个低分观测中，完整输入 AUC 为 0.6596，比去掉图像质量特征高 0.0091。按母视频配对重采样的 95% 区间为 [−0.0227, +0.0191]。四组在线回放与历史原始轨迹进行逐字节核校。[完整实验报告](../../../../../experiments/vttiny_neural_qr/results/实验报告.md) · [配对重采样](../../../../../experiments/vttiny_neural_qr/results/r_discrimination_uncertainty.json)

## 学习 R 的分数门控与逐帧取证

R 的附加对照为全分数、仅低分 `0.10 < score < 0.50`、仅高分 `score > 0.50`，保持原生出生机制，共用同一组训练参数和质量缓存。

- **高分降权造成的漏检。** 全分数 R 共出现 656 个新增漏检和 465 个补回目标帧，净增 191 个 FN。DJI_0309_4 中，第 380 帧相同先验改用原生更新时输出 IoU 为 0.582，学习 R 的输出 IoU 为 0.461。第 381 帧全分数 R 对低分候选的关联代价跨过 0.5 门限，目标到第 394 帧以新 ID 找回。[漏检归因](../../../../../experiments/vttiny_r_forensics/results/漏检归因报告.md)
- **仅低分 R 在完整测试的得失。** 相比原始 BoT-SORT，HOTA 从 37.8608 变为 37.8222，IDF1 从 42.4507 变为 42.5147，FP 减 97、FN 减 12、换号减 21。21 次净减中的 20 次来自 DJI_0061_2。独立合并其余 29 段时，换号仅少 1 次、漏检增加 25 个，误报仍减少 47 个。[四组全量与逐段指标](../../../../../experiments/vttiny_r_forensics/results/four_way_metrics.csv) · [剔除该段后的独立汇总](../../../../../experiments/vttiny_r_0061/results/pool_metrics.csv)
- **少换号与长期身份损失并存。** DJI_0061_2 中，指定撤销真值 82 对应轨迹的四次低分观测更新后，第 228 帧由仅低分 R 原本保住的 ID 240 改为 ID 258。与此同时，真值 139 与 142 的轨迹先后互换，整段 AssA 下降 0.6267 个百分点。HOTA 阈值分解给出检测项 +0.0165、关联项 −0.2459 个百分点。[换号归因与时间线](../../../../../experiments/vttiny_r_0061/results/DJI_0061_2_换号归因.md) · [定点撤销结果](../../../../../experiments/vttiny_r_0061/results/targeted_interventions.csv)

## 历史复现入口

三套脚本以相同的检测、RGB ReID、RGB/T 原图、官方 GT 和固定 TrackEval 为输入。预测进程与读取真值的评测进程独立。用拥有这些冻结资产的环境，按脚本中的实验根目录映射准备独立运行目录。

| 阶段 | 实现与运行顺序 |
|---|---|
| Q/R 训练与全量四组 | [`vttiny_neural_qr/reproduce.py`](../../../../../experiments/vttiny_neural_qr/reproduce.py) 依次拟合 R 与 Q、缓存 R 在线特征、运行四组跟踪、评分与配对重采样。用 `VT_INTEGRATIONS` 指定历史实验父目录，参数为一个空输出目录。 |
| R 分数门控 | [`vttiny_r_forensics/branch_replay.py`](../../../../../experiments/vttiny_r_forensics/branch_replay.py) 用 `pilot` 回放 `native`、`learned_r`、`low_r`、`high_r`，用 `full` 回放 `low_r`、`high_r`。随后用 `truth_events.py all / case / full` 分别核验三类集合，执行 `analyze_case.py`、`formal_score.py`、`report.py`。 |
| DJI_0061_2 专项 | [`vttiny_r_0061/capture.py`](../../../../../experiments/vttiny_r_0061/capture.py) 分别回放 `native` 和 `low_r`，继续运行 `measure.py`、`events.py`、`mechanism.py`、`probes.py`、`report.py`。 |

- 原部署中三套代码目录名依次为 `native_neural_qr_v1`、`native_r_forensics_v1`、`low_r_0061_audit_v1`，三个目录同级。相应设置模块中的 `environment()` 提供子进程所需的本地依赖搜索路径。R 分数门控可用 `VT_RF_ROOT` 与 `VT_RF_QR` 指定运行目录和前一轮结果。
- 训练输入由 [目录映射](../../../../../experiments/vttiny_neural_qr/qr_settings.py) 中的 `KF`、`TRAIN` 和 `MOTION` 读取，包含历史训练窗口计划、训练配对、同步双模态图像、训练检测缓存，以及前一轮 Q 校准模块。原跟踪入口来自 `c3e142e6` 的历史检出。完整重放还要求原始回放协议和缓存检验通过。
- 已发布的评分和诊断数据为真实回放记录。表格和策略参数保留原文件字节，发布版报告的链接已指向随附材料。两个评分审计 JSON 的绝对路径统一替换为 `${VT_DATASET_ROOT}`。报告中涉及原图、官方真值框及逐框数据的核验需要从冻结资产重新生成。这些发布版本的文件哈希见 [SHA-256 清单](files_sha256.json)。
