# BoT-SORT-Plus · W0 弱观测续接与定型实验

本分支 **research/vttiny-weak-evidence** 为毕业设计目标跟踪章节准备 W 模块及下一批实验，从 research/vttiny-track-conditioned-observation 的 **176f2a81613c34d16a860765ea7be0845116b534** 建立，继承 B、GMC、TCO/Q2 及先前实验归档。

**W R4.2 已完成五组官方 30 段开发评测。保留带原锚点约束和正式状态反馈的 W0 作为定型候选，并将原生低阈值保留为召回优先的强对照。** [当前运行手册](docs/botsort_plus/wem_r4_2/LOCAL_RUNBOOK.md) · [实验设计](docs/botsort_plus/wem_r4_2/EXPERIMENT_PLAN.md) · [软件核验](docs/botsort_plus/wem_r4_2/VALIDATION.md) · [结果入口](results/vttiny/wem_r4_2/README.md)

R4.1 已完成 B 和 BG 的十组闭环评测。W0 相对两参考的 HOTA 分别 +0.1724、+0.1743 个百分点，FN 分别 −508、−510，伴随 FP +554、+578 和 MOTA 小幅下降。W1/W2 与 W0 轨迹一致，W3 相对 W2 多出 57 FP、27 FN。本轮收敛公共的受约束弱续接机制，不继续搜索运动或外观评分。[R4.1 完整结果](results/vttiny/wem_r4_1/full_20261128/README.md)

R3.1 已完成 W0–W3 与 W_old 五组冻结离线对照。785 个未知候选组仍可随后定向复核；这不构成本批完整闭环的前置条件。[R3.1 结果与复现](results/vttiny/wem_r3_1/full_20261128/README.md)

历史 R3 已完成训练侧冻结校准和 **113 例模型辅助图卡复核，人工确认 0/113**。18 个工作点未达到原准入条件，冻结策略继续为 `REJECT_ALL_FALLBACK`。[历史 R3 结果](results/vttiny/wem_v0.1/r3/README.md)

已归档旧 W R4 探索性闭环的八组全量评测。其严格外观门完整 W 少漏 59 帧，同时增加约 145 个误报，换号持平；当时的默认拒绝全部结论按历史记录保留。旧 B_plain 与当前 B_W0 的全套 TrackEval 明细一致；旧 lower_low 的 HOTA/IDF1 更高而误报、换号代价更大，这一对照继续保留。[R4 结果与复现](results/vttiny/wem_v0.1/r4/README.md)

R4.1 固定策略沿用 R3.1 的 W0–W3 定义；旧 R4 的 0.75 外观门结果单独保留。当前实验采用真实状态演化，最终以完整 TrackEval 指标及逐帧归因判断继续、修改或停止；不按旧 R3 reject-all、人工复核状态或指标正负阻断运行。

## 从这里开始

| 入口 | 内容 |
| --- | --- |
| [当前 R4.2 运行手册](docs/botsort_plus/wem_r4_2/LOCAL_RUNBOOK.md) | 继承上一轮本地配置、复用参考、全量执行和回传 |
| [当前 R4.2 实验设计](docs/botsort_plus/wem_r4_2/EXPERIMENT_PLAN.md) | BG 下低阈值、锚点支持与正式状态反馈三项验证 |
| [当前 R4.2 回传模板](docs/botsort_plus/wem_r4_2/RESULTS_TEMPLATE.md) | 固定五组、四个对比方向及实际干预证据 |
| [当前 R4.2 软件核验](docs/botsort_plus/wem_r4_2/VALIDATION.md) | 合成状态、真实源码接入、来源复用和报告检查 |
| [已完成 R4.1 运行手册](docs/botsort_plus/wem_r4_1/LOCAL_RUNBOOK.md) | 原十组完整闭环的复现入口 |
| [已完成 R4.1 结果](results/vttiny/wem_r4_1/full_20261128/README.md) | 完整指标、组件增量、组合增量与逐帧归因 |
| [已完成 R3.1 运行手册](docs/botsort_plus/wem_r3_1/LOCAL_RUNBOOK.md) | 复用冻结缓存的历史离线比较 |
| [已完成 R3.1 实验安排](docs/botsort_plus/wem_r3_1/EXPERIMENT_PLAN.md) | 固定关联机制与条件覆盖诊断 |
| [前期本地运行手册](docs/botsort_plus/wem_v0.1/LOCAL_RUNBOOK.md) | 拉取、填路径、检查、运行首批审计、回传结果 |
| [W 完整设计](docs/botsort_plus/wem_v0.1/MODULE_DESIGN.md) | 候选来源、历史、身份竞争、可信锚与 B/GMC 衔接 |
| [分批实验安排](docs/botsort_plus/wem_v0.1/EXPERIMENT_PLAN.md) | R0、R1a/R1b、R2、R3、R4 的输入、对照、验收与停止条件 |
| [后续本地实现任务](docs/botsort_plus/wem_v0.1/IMPLEMENTATION_TASKS.md) | 给本地 Codex 的逐任务交接，标明尚需实现的采样器和机制 |
| [数据合同](docs/botsort_plus/wem_v0.1/DATA_CONTRACT.md) | 原始行号、S0 掩码、状态快照、标签与所有权 |
| [结果回传模板](docs/botsort_plus/wem_v0.1/RESULTS_TEMPLATE.md) | 必交产物、结果解释和继续/停止结论 |
| [历史证据与文献](docs/botsort_plus/wem_v0.1/EVIDENCE_AND_REFERENCES.md) | TCO/Q2 结果、相关工作和可验证增量 |
| [本次软件核验](docs/botsort_plus/wem_v0.1/VALIDATION.md) | 新工具测试、适用范围和未验证内容 |

当前批次配置：[R4.2 本地模板](configs/research/wem_r4_2.example.json) · [R4.2 固定协议](configs/research/wem_r4_2_protocol.json)。此前 R4.1、R3.1 与 wem_v0.1 的配置、结果和设计继续保留。

## 当前批次：R4.2 三组新增全量实验

使用上一轮成功运行 R4.1 的环境，在仓库根目录执行：

~~~bash
python -X utf8 tools/run_wem_r42.py init --from-r41-settings configs/research/wem_r4_1.local.json --settings configs/research/wem_r4_2.local.json
python -X utf8 tools/run_wem_r42.py check --settings configs/research/wem_r4_2.local.json
python -X utf8 tools/run_wem_r42.py full --settings configs/research/wem_r4_2.local.json
~~~

默认从旧配置的 `paths.output_root/full` 读取完整原始运行目录，验证后复制 BG_off/BG_W0 到新批次。公开 CSV 结果归档不能替代完整本地运行目录。若其已搬迁，初始化时加 `--r41-batch <原full目录>`；如果完整旧目录不可用，可在初始化时显式加 `--reference-mode rerun`，重新运行五组。新输出默认写入 `runs/wem_r4_2`。

| 组别 | 执行方式 | 目的 |
| --- | --- | --- |
| BG_off | 默认复用 R4.1 | 已有 BG 参考 |
| BG_W0 | 默认复用 R4.1 | 冻结的受约束弱观测续接 |
| BG_lower_low | 新跑全部 30 段 | 原生低阈值改为 0.05，W 关闭 |
| BG_W0_relaxed_anchor | 新跑全部 30 段 | 去掉三项锚点支持条件，保留上一帧常规观测及其余保护 |
| BG_W0_emit_only | 新跑全部 30 段 | 发布私有测量预演的弱框，但不反馈正式轨迹状态 |

`full` 自动完成三段 × 五策略技术 smoke、固定五组完整预测（默认仅三组新跟踪）、TrackEval、CLEAR 归因和报告。emit_only 在原生更新及清理结束后才发布弱输出，正式状态链及原生事件必须与 BG_off 相同；真实发布输出另行审计。指标下降、零变化或零提交不会被当作软件失败。

[已完成的全量评测、逐帧复核和复现记录](results/vttiny/wem_r4_2/full_20261128/README.md)。本批用于完成 W0 的机制解释与代价对照，不增加阈值搜索，不把复用参考视为新的独立重复，也不将诊断耗时用作部署 FPS。

## 已完成批次：R4.1 十组完整闭环

在已成功运行旧 R4 的环境中，从仓库根目录执行：

~~~bash
python -X utf8 tools/run_wem_r41.py init --settings configs/research/wem_r4_1.local.json
~~~

按 [运行手册](docs/botsort_plus/wem_r4_1/LOCAL_RUNBOOK.md) 填入已有 R0 预测设置、一致性凭据、参考与评测清单、微调 RGB 权重和本机路径。然后执行：

~~~bash
python -X utf8 tools/run_wem_r41.py check --settings configs/research/wem_r4_1.local.json
python -X utf8 tools/run_wem_r41.py full --settings configs/research/wem_r4_1.local.json
~~~

`full` 自动完成技术 smoke、十组全部 30 段跟踪、TrackEval、CLEAR 归因与汇总。默认顺序运行，已完成且摘要一致的实验臂可复用。核心输出为 `runs/wem_r4_1/full/report/REPORT.md`。

| 策略 | B 参考 | BG 参考 |
| --- | --- | --- |
| 关闭 W | B_off | BG_off |
| 原生几何 W0 | B_W0 | BG_W0 |
| 几何＋运动 W1 | B_W1 | BG_W1 |
| 软外观 W2 | B_W2 | BG_W2 |
| 相对竞争 W3 | B_W3 | BG_W3 |

BG 沿用 `evidence_both`，包含既有自适应 GMC、低分双几何和关联证据配置；各 BG 变体与自身 BG_off 比较。接纳弱框后真实更新 KF 状态并进入后续帧，原 ID 保持、弱框不写外观记忆，原生关联及 B 仍优先。W_H 只在当前在线状态上记录反事实诊断，不增加重复完整实验臂。

软件核验不代表新模块已有效。完整评测按已查看的官方开发评测集解释，以 HOTA、IDF1、FP/FN/IDSW、逐序列分布及 B 恢复事件变化共同决定后续去留。

## 历史批次：R3.1 四组关联对照

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
| R3.1：W0–W3 与 W_old 缓存对照 | [74 段离线结果已归档](results/vttiny/wem_r3_1/full_20261128/README.md)，785 例待人工复核 |
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


