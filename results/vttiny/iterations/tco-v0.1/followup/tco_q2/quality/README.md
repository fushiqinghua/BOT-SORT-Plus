# TCO 成对质量监督

使用同一批官方训练来源样本从头拟合 P3，按已登记轮数训练。冻结 P3 的编码器、支持分数和框修正网络，再从同一基础检查点复制两个质量头。绝对质量对照使用原版两个绝对 IoU 损失，成对版本同时监督同一检测框修正前后的真实 IoU 差值。质量损失只更新质量头，两组的候选、修正框和支持分数逐项核对。

成对版本保留原绝对 IoU Smooth L1 锚点，加上带符号的增量拟合。增量 Smooth L1 的转折点为 0.02，跨过 IoU 0.50 的修复和损伤样本各使用 4 倍权重。真实变化幅度达到 0.005 时增加方向损失，存在正负两种方向的批次将两种方向等权平均。损失目标与修正框坐标断开梯度。身份未知的样本不作为质量监督，有明确错误身份的候选继续以两个零绝对质量为目标。

两组在同一个已注册的校准网格内选择各自的检查点和门槛。沿用原正确观测损失率上限 0.5%、已知负例采用率上限 1%。训练和校准阶段只读取相应两个来源分区。锁定检查点和门槛后评估一个来源组完全不重叠的验收分区。

```bash
python fit_quality.py fit --base-source /local/tco_t23/src --corpus /local/new_corpus --output /local/q2_run --device cuda:0
python fit_quality.py accept --base-source /local/tco_t23/src --corpus /local/new_corpus --output /local/q2_run --device cuda:0
```

输入为旧采样器兼容的已锁定 `crop_manifest.json` 与 `fit`、`calibration`、`holdout` 三个分区。命令 `fit` 生成 `calibration_lock.json` 和两个质量头的检查点与校准曲线。命令 `accept` 验证锁定文件，输出 `acceptance.json` 和逐候选表。只在净修复高于零且优于绝对质量对照、平均 IoU 有改善、各项损伤预算合格、收益跨至少两组且去掉贡献最大组后净修复仍为正时，生成线上接入使用的 `selection.json` 和 `t2_gate.json`。
