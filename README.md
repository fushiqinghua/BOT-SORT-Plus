# BoT-SORT-Plus · 弱检测证据挖掘与关联消融

本分支 **research/vttiny-weak-evidence** 为毕业设计目标跟踪章节准备 W 模块及下一批实验，从 research/vttiny-track-conditioned-observation 的 **176f2a81613c34d16a860765ea7be0845116b534** 建立，继承 B、GMC、TCO/Q2 及先前实验归档。

**当前批次：R3.1 的 W0–W3 四组缓存对照已实现，真实数据结果待本地执行。** 本批复用 R3 已有特征和离线标签，不需要重新运行检测器或 ReID，不启用在线 W。另重算 W_old 作为历史对照。[当前运行手册](docs/botsort_plus/wem_r3_1/LOCAL_RUNBOOK.md) · [实验协议](docs/botsort_plus/wem_r3_1/EXPERIMENT_PLAN.md)

历史 R3 已完成训练侧冻结校准和 **113 例模型辅助图卡复核，人工确认 0/113**。18 个工作点未达到原准入条件，冻结策略继续为 `REJECT_ALL_FALLBACK`。[历史 R3 结果](results/vttiny/wem_v0.1/r3/README.md)

**已归档 W R4 探索性闭环的八组全量评测。完整 W 少漏 59 帧，同时增加约 145 个误报，换号持平。R3 的默认拒绝全部策略继续有效。** [R4 结果与复现](results/vttiny/wem_v0.1/r4/README.md)

本次 R3.1 对照复用训练侧 R3 缓存，研究外观作为软关联证据的作用；既有 R4 使用 0.75 外观门的探索点，另行保留，不重复其八组回放。

## 从这里开始

| 入口 | 内容 |
| --- | --- |
| [当前 R3.1 运行手册](docs/botsort_plus/wem_r3_1/LOCAL_RUNBOOK.md) | 复用现有缓存，一次运行 W0、W1、W2、W3 与 W_old |
| [当前 R3.1 实验安排](docs/botsort_plus/wem_r3_1/EXPERIMENT_PLAN.md) | 固定机制、单因素比较、适用范围与停止条件 |
| [当前 R3.1 回传模板](docs/botsort_plus/wem_r3_1/RESULTS_TEMPLATE.md) | 条件覆盖、未知采用、相对竞争与来源分层 |
| [前期本地运行手册](docs/botsort_plus/wem_v0.1/LOCAL_RUNBOOK.md) | 拉取、填路径、检查、运行首批审计、回传结果 |
| [W 完整设计](docs/botsort_plus/wem_v0.1/MODULE_DESIGN.md) | 候选来源、历史、身份竞争、可信锚与 B/GMC 衔接 |
| [分批实验安排](docs/botsort_plus/wem_v0.1/EXPERIMENT_PLAN.md) | R0、R1a/R1b、R2、R3、R4 的输入、对照、验收与停止条件 |
| [后续本地实现任务](docs/botsort_plus/wem_v0.1/IMPLEMENTATION_TASKS.md) | 给本地 Codex 的逐任务交接，标明尚需实现的采样器和机制 |
| [数据合同](docs/botsort_plus/wem_v0.1/DATA_CONTRACT.md) | 原始行号、S0 掩码、状态快照、标签与所有权 |
| [结果回传模板](docs/botsort_plus/wem_v0.1/RESULTS_TEMPLATE.md) | 必交产物、结果解释和继续/停止结论 |
| [历史证据与文献](docs/botsort_plus/wem_v0.1/EVIDENCE_AND_REFERENCES.md) | TCO/Q2 结果、相关工作和可验证增量 |
| [本次软件核验](docs/botsort_plus/wem_v0.1/VALIDATION.md) | 新工具测试、适用范围和未验证内容 |

当前批次配置：[R3.1 本地模板](configs/research/wem_r3_1.example.json) · [R3.1 固定协议](configs/research/wem_r3_1_protocol.json)。前期配置与设计继续保留在 wem_v0.1 目录。

## 当前批次：R3.1 四组关联对照

继续使用当前分支：

~~~bash
git fetch origin
git switch research/vttiny-weak-evidence
git pull --ff-only origin research/vttiny-weak-evidence
python -X utf8 tools/run_wem_r31_ablation.py init --settings configs/research/wem_r3_1.local.json
~~~

填写配置中的 `paths.features_dir`（R3 分数缓存目录）、`paths.labels_dir`（R3 独立标签目录）和 `paths.split_file`（原冻结 training_split.json）；`output_dir` 使用新的目录，保留默认值即可。这里的 features_dir 包含 manifest.json 和逐序列分数 jsonl.gz，不是更早的 ReID npz 目录。

~~~bash
python -X utf8 tools/run_wem_r31_ablation.py check --settings configs/research/wem_r3_1.local.json
python -X utf8 tools/run_wem_r31_ablation.py run --settings configs/research/wem_r3_1.local.json
~~~

| 实验 | 固定机制 |
| --- | --- |
| W0 | 原生几何与原检测分数的确定性平局处理 |
| W1 | 几何＋历史运动，效用 (i+m)/2 |
| W2 | 与 W1 相同可行边，外观仅软排序；缺外观时回退 W1 |
| W3 | W2＋裁剪前的双向相对外观竞争，包含已匹配邻居 |
| W_old | 原 R3 严格外观门历史对照，参数保持原值 |

W2 不设绝对余弦或联合效用阈值。W3 将无竞争、竞争证据缺失和相对间隔不足分开记录。所有方法保留原生几何、GMC、所有权和每轨迹5个候选组预算。完整运行自动核验 W0/W_old 与历史 R3 固定结果一致，并生成真实被采用 unknown/ignored 的复核清单；不会把未知计为误检或正确。运行通过只表示离线比较完成，不代表科研准入通过。

主要输出为 comparison.csv、method_deltas.csv、per_sequence.csv、per_source.csv、review_queue.csv、summary.json、decisions.jsonl.gz、manifest.json 和协议快照。[结果归档目录](results/vttiny/wem_r3_1/README.md) · [软件核验](docs/botsort_plus/wem_r3_1/VALIDATION.md)

本批只需要已有 NumPy/SciPy 环境。旧 holdout 已用于诊断，继续按开发分析报告，不作为新的盲测。本轮没有阈值搜索或线上工作点选择。

## 前期 P0/R1a 入口

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

## 前期 R1a 库存审计的输出

审计生成 manifest.json、summary.json、per_sequence.csv、s0_candidates.jsonl，统计：

- 当前缓存中确实存在的 S0 弱候选；
- 原生高低分集合和阈值边界行；
- 各序列候选分布与输入摘要；
- 按现有 ReID 导出策略，哪些行通常没有特征覆盖。

结果状态为 **INVENTORY_ONLY_NOT_W_OPPORTUNITY**。它不判断轨迹当时是否合格、不判断身份、不等于可补回 FN 的数量。后续状态只读采样和离线标记已用于本次归档的 R3；当前四组比较请使用上面的 R3.1 入口。

## 阶段状态

| 阶段 | 当前状态 |
| --- | --- |
| P0：路径配置与计划快照 | 工具可运行；不是 R0 科研验收 |
| R1a：S0 缓存库存 | 工具可运行；真实数据结果待本地执行 |
| R0：参考复现与 W 新采样等价性 | R3 训练侧 74 段只读回放通过 |
| R1b：轨迹资格、所有权、身份机会 | 已用于 R3 训练侧候选和真值离线核验 |
| R2：冻结状态 raw/prior/posterior 比较 | 前期离线实验为 R3 提供动机 |
| R3：身份/背景区分与校准留出 | [已完成](results/vttiny/wem_v0.1/r3/README.md)，113 例人工复核待完成 |
| R3.1：W0–W3 与 W_old 缓存对照 | 工具与固定协议已就绪，真实数据 NOT_RUN |
| R4：W 探索性闭环及组合消融 | [八组全量结果已归档](results/vttiny/wem_v0.1/r4/README.md)，默认策略仍关闭；本次 R3.1 不运行新闭环 |

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
- [W R4 探索性八组全量结果](results/vttiny/wem_v0.1/r4/README.md)
- [W 结果回传目录](results/vttiny/iterations/wem-v0.1/README.md)

官方 30 段已经多轮用于开发与选择，后续仍按开发评估集描述。新门槛按训练侧来源分组校准；母视频映射无法核验时保留限制。初始化、准备工具和单元测试通过，都不表示真实实验已通过。


