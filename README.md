# BoT-SORT-Plus · 轨迹条件化时序观测增强研究

本分支为毕业设计目标跟踪章节准备 **TCO（Track-Conditioned Observation）模块设计与下一批实验协议**，接续 `research/vttiny-gmc-kf-345` 的 `235fe4097066b5023cc84ca8cd6edd5757fd658c`，完整保留此前 B、GMC、③④⑤及神经 Q/R 的实现和结果。

**当前状态**　T0 只读采样通过六段与全量 30 段的逐字节一致性核验。T1 完成官方开发集的离线几何诊断和逐帧模型辅助复核。拟合、校准、独立人工复核和闭环评估仍待进行。

## 从这里开始

| 文档 | 内容 |
|---|---|
| [T0/T1 实验与复核归档](results/vttiny/iterations/tco-v0.1/README.md) | 6 段与 30 段只读校验、离线几何统计、原抽样与补样、图卡索引 |
| [T0/T1 复现入口](tools/research/tco_t01/README.md) | 冻结输入、采样回放、独立标记、图卡生成和状态核对 |
| [模块完整设计](docs/botsort_plus/tco_v0.1/MODULE_DESIGN.md) | 小型空间编码器、历史条件、支持／质量／定位输出、损失、原生低分阶段接入 |
| [实验协议与逐阶段任务](docs/botsort_plus/tco_v0.1/EXPERIMENT_PLAN.md) | T0–T4、最小对照、机会量核验、继续与停止条件、回传清单 |
| [本地开始与 Codex 交接](docs/botsort_plus/tco_v0.1/LOCAL_RUNBOOK.md) | 本轮可运行命令、需要准备的资产、首个本地实现任务 |
| [数据与实现接口](docs/botsort_plus/tco_v0.1/DATA_CONTRACT.md) | 精确检测索引、历史和标签隔离、坐标、可达性、私有状态与产物 |
| [证据与文献](docs/botsort_plus/tco_v0.1/EVIDENCE_AND_REFERENCES.md) | 为什么选择此方向，以及与 A、C2、旧位置修正、Q/R 的区别 |
| [设计复核记录](docs/botsort_plus/tco_v0.1/DESIGN_REVIEW.md) | 本次检查了什么、准备工具的软件验证与尚未验证的内容 |

机器可读材料：[本地设置模板](configs/research/tco_v0.1.example.json) · [设计协议](configs/research/tco_v0.1_protocol.json) · [任务台账模板](configs/research/tco_v0.1_tasks.json)。

## 本轮研究重点

TCO 学习“这条已确认轨迹在当前局部图像中的目标支持与位置”，为原生低分观测提供有条件的细化。首版使用 RGB，先在固定候选上验证图像与历史信息，再测试闭环影响。

| 阶段 | 目标 | 当前实现状态 |
|---|---|---|
| T0 | 复现 B 与 `evidence_both`，核验只读记录和输入来源 | 六段与全量 30 段均已通过只读等价性核验 |
| T1 | 量化首版能触及的定位／身份错误，建立母视频隔离的数据集 | 官方开发集的离线标记及模型辅助复核已归档，人工复核与训练来源划分待完成 |
| T2 | 比较原框、几何对照、当前 RGB、历史条件 RGB | 待实现，先 seed 41 |
| T3a | 原生低分匹配后，仅细化本帧已接受的测量 | 待实现，T2 有证据才推进 |
| T3b | 原低分可行边内，在分配前做逐边细化 | 条件阶段，保留原保护和一对一约束 |
| T4 | 成功后单独研究 IR、扩边或新观测 | 需独立版本与消融 |

高分匹配、起轨规则、原低分分数区间及 B 身份恢复职责在首版协议中冻结。阈值外检测和无框重检测属于后置扩展；旧 D3 的 188/210 次缺少可用观测，不能全部记为首版的可改进样本。

参考别名 `BG` 在本批指 **`evidence_both`**：其中包含完整自适应 GMC、`dual_low` 和长间隔证据规则，因此不能把它与 B 的差值解读为单独 GMC 因子的贡献。需要纯因子比较时，另加已归档的 `B_early_moderate_dual_low`。

## 当前可执行的准备步骤

在仓库根目录执行，以下命令可用于 Bash 或 PowerShell：

```bash
git fetch origin
git switch --track origin/research/vttiny-track-conditioned-observation
python -X utf8 tools/manage_tco_batch.py init --output-config configs/research/tco_v0.1.local.json
```

若已在该本地分支，使用 `git switch research/vttiny-track-conditioned-observation` 与 `git pull --ff-only origin research/vttiny-track-conditioned-observation`。

填写新本地配置中的资产路径后：

```bash
python -X utf8 tools/manage_tco_batch.py check --settings configs/research/tco_v0.1.local.json
python -X utf8 tools/manage_tco_batch.py prepare --settings configs/research/tco_v0.1.local.json
python -X utf8 tools/manage_tco_batch.py status --settings configs/research/tco_v0.1.local.json
```

`prepare` 只创建独立批次目录、复制小型协议与任务清单，结果为 **`PREPARED_NOT_RUN`**。它不训练模型、不运行跟踪、不认证 T0 已通过。缺少资产时返回 3 并列出缺项；不会用合成样本冒充真实实验。完整路径映射和本地实施顺序见[运行说明](docs/botsort_plus/tco_v0.1/LOCAL_RUNBOOK.md)。

## 历史实证入口

- [第三轮 B＋GMC：七组 30 段结果与 J0–J3](results/vttiny/iterations/v0.4-motion-blur/round3-b-gmc/README.md)
- [③④⑤及后续神经 Q/R](results/vttiny/iterations/motion-345/README.md)
- [C2、B、原始 IR 与位置修正](results/vttiny/iterations/v0.3-c2-identity/README.md)
- [D1–D4 运动与观测诊断](results/vttiny/iterations/v0.4-motion-blur/README.md)
- [E1–E3 第二轮归档](results/vttiny/iterations/v0.4-motion-blur/round2-e1-e2-e3/README.md)

历史 30 段已多次用于开发和选模。本批沿用该事实标签，并为新头单独建立按原视频划分的拟合、校准与内部留出集合。E3 的 22 例正式人审作为独立短支线继续收尾。
