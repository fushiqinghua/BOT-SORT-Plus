# BoT-SORT-Plus · 轨迹引导的弱检测证据挖掘

本分支 **research/vttiny-weak-evidence** 为毕业设计目标跟踪章节准备 W 模块及下一批实验，从 research/vttiny-track-conditioned-observation 的 **176f2a81613c34d16a860765ea7be0845116b534** 建立，继承 B、GMC、TCO/Q2 及先前实验归档。

**当前阶段已完成 R3 训练侧冻结准入校准和 113 例逐帧复核。18 个有限工作点均未达到准入条件，保留拒绝全部弱框的策略，R4 未运行。** [R3 结果与图卡索引](results/vttiny/wem_v0.1/r3/README.md)

## 从这里开始

| 入口 | 内容 |
| --- | --- |
| [本地运行手册](docs/botsort_plus/wem_v0.1/LOCAL_RUNBOOK.md) | 拉取、填路径、检查、运行首批审计、回传结果 |
| [W 完整设计](docs/botsort_plus/wem_v0.1/MODULE_DESIGN.md) | 候选来源、历史、身份竞争、可信锚与 B/GMC 衔接 |
| [分批实验安排](docs/botsort_plus/wem_v0.1/EXPERIMENT_PLAN.md) | R0、R1a/R1b、R2、R3、R4 的输入、对照、验收与停止条件 |
| [后续本地实现任务](docs/botsort_plus/wem_v0.1/IMPLEMENTATION_TASKS.md) | 给本地 Codex 的逐任务交接，标明尚需实现的采样器和机制 |
| [数据合同](docs/botsort_plus/wem_v0.1/DATA_CONTRACT.md) | 原始行号、S0 掩码、状态快照、标签与所有权 |
| [结果回传模板](docs/botsort_plus/wem_v0.1/RESULTS_TEMPLATE.md) | 必交产物、结果解释和继续/停止结论 |
| [历史证据与文献](docs/botsort_plus/wem_v0.1/EVIDENCE_AND_REFERENCES.md) | TCO/Q2 结果、相关工作和可验证增量 |
| [本次软件核验](docs/botsort_plus/wem_v0.1/VALIDATION.md) | 新工具测试、适用范围和未验证内容 |

机器配置：[本地模板](configs/research/wem_v0.1.example.json) · [研究协议](configs/research/wem_v0.1_protocol.json)。

## 先运行这一批

在仓库根目录执行。首次创建本地分支：

~~~bash
git fetch origin
git switch --track origin/research/vttiny-weak-evidence
python -X utf8 tools/manage_wem_batch.py init --output configs/research/wem_v0.1.local.json
~~~

已存在本地分支时，使用 git switch research/vttiny-weak-evidence，再 git pull --ff-only origin research/vttiny-weak-evidence。

先填写本地配置中的 paths.detections_dir，使其指向已有逐序列检测 JSON 的目录；核对 thresholds.low/high 与真实参考运行一致。其余资产可以随后补齐。

~~~bash
python -X utf8 tools/manage_wem_batch.py check --settings configs/research/wem_v0.1.local.json --strict
python -X utf8 tools/manage_wem_batch.py prepare --settings configs/research/wem_v0.1.local.json
python -X utf8 tools/manage_wem_batch.py status --settings configs/research/wem_v0.1.local.json
~~~

prepare 会打印采用实际路径的审计命令；请执行该命令。其形式为：

~~~bash
python -X utf8 tools/audit_wem_s0.py --detections-dir "<实际检测缓存目录>" --output-dir "runs/wem_v0.1/r1a_s0" --low-thresh 0.1 --high-thresh 0.5
~~~

上面的尖括号必须替换。若修改 output_root，采用 prepare 打印的实际输出路径。

首批仅需 CPU 和 Python 标准库，不读取 GT，不运行检测器、ReID 或跟踪器，不改变原缓存。重复审计请指定新输出目录。

## 这一批能得到什么

审计生成 manifest.json、summary.json、per_sequence.csv、s0_candidates.jsonl，统计：

- 当前缓存中确实存在的 S0 弱候选；
- 原生高低分集合和阈值边界行；
- 各序列候选分布与输入摘要；
- 按现有 ReID 导出策略，哪些行通常没有特征覆盖。

结果状态为 **INVENTORY_ONLY_NOT_W_OPPORTUNITY**。它不判断轨迹当时是否合格、不判断身份、不等于可补回 FN 的数量。下一步需按[实现任务](docs/botsort_plus/wem_v0.1/IMPLEMENTATION_TASKS.md)接入状态只读采样和离线标记。

## 阶段状态

| 阶段 | 当前状态 |
| --- | --- |
| P0：路径配置与计划快照 | 工具可运行；不是 R0 科研验收 |
| R1a：S0 缓存库存 | 工具可运行；真实数据结果待本地执行 |
| R0：参考复现与 W 新采样等价性 | R3 训练侧 74 段只读回放通过 |
| R1b：轨迹资格、所有权、身份机会 | 已用于 R3 训练侧候选和真值离线核验 |
| R2：冻结状态 raw/prior/posterior 比较 | 前期离线实验为 R3 提供动机 |
| R3：身份/背景区分与校准留出 | [已完成](results/vttiny/wem_v0.1/r3/README.md)，113 例人工复核待完成 |
| R4：W 闭环及组合消融 | 准入验收未通过，未运行 |

W 首版只帮助帧首已确认活跃旧身份，使用一个当前真实弱框与过去常规观测验证，不新建 ID，不将弱外观写入长期模板，保留原生/B/起轨优先权。S1 更低导出阈值和 S2 NMS 抑制候选分开研究，默认关闭。

闭环安排中保留 **Coast-only → W-coast → 完整 W** 三组，分别区分预测延续、当前弱证据准入和坐标量测更新的贡献。

## 最新历史结论

TCO T2 已完成：P2、P3 的留出净修复分别为 −3、−2。Q2 两组校准后实际采用均为 0。归档包含质量增量误判和冻结质量头归因。当前没有开启 TCO 后的完整 30 段跟踪指标。此前 TCO 的旧 README 状态以[最新 followup 归档](results/vttiny/iterations/tco-v0.1/followup/README.md)为准。

参考 B 为 B_early_moderate。这里的 BG 是 evidence_both，包含自适应 GMC、dual_low 及证据规则，不能当作纯 GMC 单因子。需要相机因素归因时另列 B_early_moderate_dual_low。

## 历史实证入口

- [TCO T2/T3a、Q2 及误差归因](results/vttiny/iterations/tco-v0.1/followup/README.md)
- [TCO T0/T1 与独立图像复核](results/vttiny/iterations/tco-v0.1/README.md)
- [第三轮 B+GMC 七组完整结果](results/vttiny/iterations/v0.4-motion-blur/round3-b-gmc/README.md)
- [③④⑤与神经 Q/R](results/vttiny/iterations/motion-345/README.md)
- [B/C2 与前期改进](results/vttiny/iterations/v0.3-c2-identity/README.md)
- [W 结果回传目录](results/vttiny/iterations/wem-v0.1/README.md)

官方 30 段已经多轮用于开发与选择，后续仍按开发评估集描述。新门槛按训练侧来源分组校准；母视频映射无法核验时保留限制。初始化、准备工具和单元测试通过，都不表示真实实验已通过。

