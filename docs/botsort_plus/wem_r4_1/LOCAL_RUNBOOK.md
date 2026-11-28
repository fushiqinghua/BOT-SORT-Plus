# W R4.1 本地运行手册

本批把 W0–W3 接入真实跟踪更新，默认在 B 和 BG 上完成 **10 组 × 全部 30 段**，随后自动执行 TrackEval、逐帧归因和报告汇总。人工未知候选复核与原 R3 的 reject-all 结论不会阻断本批实验。

[实验设计](EXPERIMENT_PLAN.md) · [结果记录模板](RESULTS_TEMPLATE.md) · [归档入口](../../../results/vttiny/wem_r4_1/README.md)

## 1. 拉取分支并使用已有实验环境

在现有仓库中执行：

```bash
git fetch origin
git switch research/vttiny-weak-evidence
git pull --ff-only origin research/vttiny-weak-evidence
```

使用此前成功运行 R4 的 Python 环境，例如已有的 `redetr` 环境；在仓库根目录运行后续命令。新入口直接在当前机器读写实验文件，不需要沿用旧 `transfer.py` 的 SSH 桥接或固定服务器 GPU 编号。

本批复用已冻结的 RGB-T 检测缓存、常规 RGB ReID 特征、相机补偿和参考协议。W2/W3 对门内当前 S0 框提取 RGB 特征，使用同一份微调 ReID 权重。无需重新训练检测器或 ReID，也不需要重新导出原检测结果。运行环境仍需具备此前跟踪与评测所用的 Torch、BoxMOT、OpenCV、NumPy、SciPy 和 TrackEval 等依赖。

## 2. 生成本机配置

```bash
python -X utf8 tools/run_wem_r41.py init --settings configs/research/wem_r4_1.local.json
```

如果本机保留了旧项目的 `integrations` 目录布局，可使用其上一级项目目录辅助填路径：

```bash
python -X utf8 tools/run_wem_r41.py init --settings configs/research/wem_r4_1.local.json --legacy-project-root /path/to/project
```

`/path/to/project` 是本机实际目录，不能直接照抄。Windows 可使用带引号的完整路径。辅助初始化只填写旧布局中的资产位置，仍需检查路径、运行环境和 GPU 是否符合当前机器。

### 2.1 必需输入

| 配置字段 | 内容 | 旧布局中的对应资产 |
| --- | --- | --- |
| `paths.prediction_settings` | 已冻结的预测设置，不含 GT 路径 | `integrations/vttiny_wem_r01_v1/observer/prediction_settings.json` |
| `paths.r0_parity` | 30 段参考输出、事件和状态链一致性凭据 | `integrations/vttiny_wem_r01_v1/observer/parity/full.json` |
| `paths.reference_manifest` | B、BG 原始参考运行位置和协议 | `integrations/botsort_plus_v0_4_motion_blur/tco_53faeccb/t01_v1/T0/reference_manifest.json` |
| `paths.evaluation_manifest` | 冻结 GT 与 TrackEval 清单 | 同一 `T0` 下的 `input_manifest.json` |
| `paths.rgb_weights` | 与原特征摘要一致的微调 RGB ReID 权重 | 旧项目使用的 `clip_veri.pt` |
| `paths.output_root` | 本批独立输出目录 | 建议 `runs/wem_r4_1` |

完整序列清单来自冻结凭据，不能自行删掉困难序列。配置中的相对路径按仓库根目录解释。输出目录应与输入及历史结果分开。

### 2.2 本机路径迁移

若配置文件来自旧服务器，其中可能包含已失效的绝对路径。使用显式覆盖字段指向同一批本地资产，不必修改原冻结清单：

- `prediction_overrides`：可覆盖 `baseline_root`、`features_dir`、`images_root`、`runtime_root`、`vendor_dir`、`warp_sources`、`python_user_base` 等原预测设置中的位置字段。
- `evaluation_overrides`：可覆盖 `gt_root`、`trackeval_root`。
- `reference_overrides`：通过 `B.run_dir`、`BG.run_dir` 指向本机保存的参考运行目录。

覆盖只用于迁移位置。原检测、特征、相机输入、参考输出与代码摘要仍按凭据检查，不能通过改路径换成另一批数据。

### 2.3 运行设置

`runtime` 的关键字段为：

| 字段 | 建议初值 | 说明 |
| --- | --- | --- |
| `python` | `null` | 使用当前 Python；也可指定已有跟踪环境的解释器 |
| `cuda_visible_devices` | `"0"` | 分配给 RGB ReID 的物理 GPU，可改成本机空闲卡 |
| `rgb_device` | `"cuda:0"` | 子进程看到的逻辑设备；设置可见物理卡后通常仍用逻辑 0 |
| `workers` | `1` | 默认顺序执行不同实验臂，减少显存竞争 |

例如物理卡 4 可设为 `cuda_visible_devices: "4"`，而 `rgb_device` 保持 `"cuda:0"`。不要把旧脚本默认使用 GPU 4 误认为本批必须占用该卡。增加并发前，应先确认单组显存和机器资源。

## 3. 检查输入

```bash
python -X utf8 tools/run_wem_r41.py check --settings configs/research/wem_r4_1.local.json
```

检查验证解释器位置、路径、冻结资产与参考覆盖是否一致；模型依赖能否实际加载由后续 smoke 验证。若检查失败，按报错修复实际路径或缺失资产，再执行；不要删去摘要检查来勉强运行。此命令不产生模块性能结论。

## 4. 执行完整实验

推荐直接执行：

```bash
python -X utf8 tools/run_wem_r41.py full --settings configs/research/wem_r4_1.local.json
```

完整命令将执行或复用技术 smoke，随后完成十组 30 段跟踪，再自动运行评测、归因和汇总。**无需先获得新的人工复核结果，也无需证明 W1/W2/W3 离线有收益。**

smoke 用固定短序列核对关闭、影子及强制拒绝状态与 B/BG 参考的一致性。它只检验接入是否可靠，不按 HOTA 正负决定是否允许完整实验。若需要单独先查看技术结果：

```bash
python -X utf8 tools/run_wem_r41.py smoke --settings configs/research/wem_r4_1.local.json
```

本批完整臂固定为：

```text
B_off   B_W0   B_W1   B_W2   B_W3
BG_off  BG_W0  BG_W1  BG_W2  BG_W3
```

W_H 在闭环内作为反事实诊断记录，不额外运行一套状态更新。其作用是核验“历史可计算性”是否已被共同锚点条件覆盖。

## 5. 中断、续跑与重新汇总

同一配置重新执行 `full` 会检查已完成实验臂的输入、代码和输出凭据；一致的完成臂可以复用。没有完成标记的失败或不完整臂不会覆盖原记录，而是移入 `superseded` 后重新执行，相关操作会打印日志。已有完成标记但文件摘要、输入或代码不一致时会停止，要求保留原结果并使用新的输出目录；不会自动把这类变化解释为普通中断。

一次只允许一个批次进程使用同一输出根目录。若异常退出后残留运行锁，先确认原进程确实结束，再按错误提示处理残留锁，避免两个进程同时写同一批结果。

如果跟踪已完成，只需要重新执行评测或生成汇总，可使用：

```bash
python -X utf8 tools/run_wem_r41.py evaluate --settings configs/research/wem_r4_1.local.json
python -X utf8 tools/run_wem_r41.py summarize --settings configs/research/wem_r4_1.local.json
```

不要因为某组完整指标下降就将其删掉后只保留较好的臂。负结果也是本批最终去留判断的一部分。缺失序列或进程错误属于未完成，不能按零分计入汇总。

## 6. 先看哪些结果

输出位于 `paths.output_root`，其下分开保存 `smoke` 和 `full`。首先打开：

```text
full/report/REPORT.md
```

同目录的主要机器可读结果为：

| 文件 | 内容 |
| --- | --- |
| `comparison.csv` | 十组原始指标，以及相对各自 off 参考的差值 |
| `paired_per_sequence.csv` | 每组每段与自身参考的配对变化 |
| `component_deltas.csv` | B/BG 内 W1−W0、W2−W1、W3−W2 的完整聚合增量 |
| `component_per_sequence.csv` | 上述组件比较的逐序列增量 |
| `interactions.csv` | 四个策略在 BG 和 B 上的增量之差 |
| `exposure.csv` | 各组逐序列与汇总的资格、评分和决策计数 |
| `attribution.csv` | 提交与整段变化的归因摘要 |
| `summary.json`、`manifest.json` | 汇总状态、检查结果和报告文件摘要 |

详细逐帧归因另保存在 `full/attribution/`，官方 TrackEval 结果在 `full/metrics/`。报告中的方向提示只描述本轮观测，不等于统计显著性，也不会据此阻断实验。

按以下顺序阅读：

1. 技术核验、实际序列数和十组完整性；
2. TrackEval COMBINED 指标；
3. 每个 W 变体相对自身 B_off 或 BG_off 的变化；
4. W1−W0、W2−W1、W3−W2 的增量；
5. 逐序列分布与 B/BG 组合增量；
6. motion 缺失／低分、外观缺失、竞争拒绝、W_H 同帧差异、提议与正式提交；
7. 当帧补回和整段 FP/FN/IDSW、B 恢复事件的变化。

总体 HOTA 直接使用 TrackEval COMBINED，不能把逐序列 HOTA 求平均代替。HOTA、IDF1、MOTA 以百分制呈现，差值为百分点。BG_Wk 的主差值必须相对 BG_off。

新增日志中保存原始候选信息和实际决定。主进程启动前只核验 GT 文件摘要，预测子进程不接收 GT；GT 匹配结果只在离线评测、归因阶段加入。反事实策略是在当前变体的在线状态上计算，不应把它当成另一个独立变体的实际未来状态。

本批耗时包含诊断日志和反事实计算，先用于比较本批执行成本；不要把它直接写成最终部署模块的额外延迟。

## 7. 回传建议

将输入与代码凭据、smoke 技术结果、完整运行状态、官方聚合与逐序列指标、生成报告、决策统计和归因摘要放入新的日期子目录，例如：

```text
results/vttiny/wem_r4_1/full_YYYYMMDD/
```

填写 [结果记录模板](RESULTS_TEMPLATE.md)，说明实际运行提交、环境、重跑和未完成项。图像、GT、模型权重、大型特征缓存与带本机路径的配置保留在实验环境；公开衍生副本若删除字段，分别保存原件与副本摘要。

本轮优先交付完整指标与可解释的运行记录。人工候选复核可以随后针对实际造成指标变化的事件开展，不需要先完成全部历史 unknown 队列。
