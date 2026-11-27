# TCO 训练、质量监督与跌线诊断

这批实验检验已匹配低分检测框的定位修正。所有增量均以同一观测的官方真值 IoU 计算。完整 30 段上已验证的是 T0 只读采样一致性。两轮训练侧验收未达到开启修正的条件，当前没有 TCO 开启后的 30 段 HOTA、IDF1、MOTA、FP、FN 或 IDSW 变化。

| 实验 | 划分与操作 | 结果 |
| --- | --- | --- |
| T2 探索版，seed 41 | 官方训练侧按代理来源分组，拟合 51 段、校准 11 段、留出 12 段 | 留出 P1、P2、P3 净修复分别为 0、−3、−2。T3a 在两种参考配置上完成六段关闭、记录、回退一致性检查 |
| Q2 质量监督复验 | 在此前排除的 11 段中，4 段加入拟合，7 段整组用于新验收。拟合 55 段、校准 11 段，旧留出 12 段用于诊断 | 在 1,126 个可度量方向的验收样本中，绝对监督判对 601 个，成对监督判对 582 个。两者校准后实际采用均为 0 |
| Q2 冻结质量头归因 | 对齐校准集 3,057 个合法正例，核对 14 个升线、72 个跌线提案，再按同一初始化和批次重放损失 | 单看增量门槛 0.005，跌线放行从 17 增至 42。新增 29、撤回 4。去掉连续增量回归后跌线放行降至 8 个，跨线权重改为 1 倍仍为 42 个 |

训练数据按 DJI 文件名和重复帧审计构建代理来源组，官方归档缺少可核验的母视频映射。Q2 允许新增的训练片段与测试分组代理重合。低 IoU 困难样本的补充身份标签来自模型辅助复核，人工状态待复核。图卡文件、训练权重、全量源图与运行缓存保留在原实验环境，归档提供数值、索引和校验值。

- [T2 留出结果、训练划分和 T3a 接入核验](tco_t23/results/T2_T3a_exploratory_report.md)
- [Q2 重训、校准与新验收](tco_q2/results/Q2_实验报告.md)
- [旧质量头四帧错判](tco_q2/diagnosis/四帧质量头复核.md)
- [17→42 的逐项归因](tco_q2/diagnosis/gain_gate/成对监督跌线放行归因.md) · [86 个跨线提案明细](tco_q2/diagnosis/gain_gate/crossing_evidence.csv)
- [本地图卡索引和 SHA-256](local_card_index.csv) · [发布文件校验清单](manifest.json)

## 复核与复现

只复算已发布的跨线结果时，在安装 NumPy 的 Python 环境中进入 `tco_q2/diagnosis/gain_gate`，运行 `python verify_findings.py`。脚本按原始行号连接候选框与损失诊断，重建 `crossing_evidence.csv` 和 `attribution_summary.json`。两份 `inputs/*.csv.gz` 保存完整校准打分，`inputs/row_keys.jsonl.gz` 提供逐行身份键。

复现模型探针时，在带原有检查点、训练图像和裁剪语料的实验主机上使用原 PyTorch 环境，将 `TCO_RUN_ROOT` 设为包含 `t01_v1`、`t23_v1`、`q2_v1` 的运行目录。依次运行同目录的 `fetch_calibration.py`、`compare_pairs.py`、`run_probe.py` 和 `verify_findings.py`。图卡可用 `render_cases.py --fetch` 在本地重建。中文字体可用环境变量 `TCO_CJK_FONT` 指定。

首轮训练按[冻结输入准备](tco_t23/assets/README.md)、[原生低分匹配采样](tco_t23/online/README.md)和 `tco_t23` 中的划分、标记、裁剪、训练脚本执行。Q2 依次使用 `tco_q2/split`、`prepare_data.py` 及[质量头训练与验收](tco_q2/quality/README.md)。原始计算源码与归档权重摘要可从 [Q2 校准锁定文件](tco_q2/results/quality_run/calibration_lock.json)对照。运行时按所用脚本将 `tco_t23`、`tco_q2/quality` 和已有的 [T0/T1 采样与标记代码](../../../../../tools/research/tco_t01/README.md)加入 Python 模块搜索路径。
