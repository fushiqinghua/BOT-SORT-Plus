# TCO 本地准备与实施交接

## 1. 这次交付的准确范围

分支：`research/vttiny-track-conditioned-observation`。基础提交：`235fe4097066b5023cc84ca8cd6edd5757fd658c`。

本次已经提供设计、实验协议、数据接口和准备工具。**TCO 神经网络、采样器、训练器和跟踪适配器属于后续本地实施工作，本分支当前没有可直接运行的 TCO 训练／全量实验命令。** 既有 B/GMC 和 ③④⑤代码随基础分支保留。

本地分两步使用：先登记资产和冻结本批设计，再让本地 Codex 按实验协议 I0–I6 实现并验证。模型机制与真实收益分别验收，准备成功不会自动把实验状态改成 PASS。

## 2. 拉取新分支

首次获取：

```bash
git fetch origin
git switch --track origin/research/vttiny-track-conditioned-observation
```

已经有本地分支时：

```bash
git switch research/vttiny-track-conditioned-observation
git pull --ff-only origin research/vttiny-track-conditioned-observation
```

沿用已跑通 B/GMC 的环境。准备工具只用 Python 标准库；后续训练环境需在 T0 根据本机原跟踪环境记录并验证，不以本次工具测试环境替代正式实验环境。

## 3. 创建本地配置

```bash
python -X utf8 tools/manage_tco_batch.py init --output-config configs/research/tco_v0.1.local.json
```

已有同名配置会被拒绝覆盖。这个本地文件已加入 Git 忽略规则。首次模板含 `null` 资产路径，需要填写实际位置；JSON 字符串支持 Windows 的 `D:/...` 写法和 UTF-8 BOM。所有相对路径相对于仓库根目录，父目录资产请填写绝对路径，不使用 `../`。

| 配置位置 | 填写的已有资产 | 后续谁核验内容 |
|---|---|---|
| `paths.baseline_root` | 冻结检测根目录，含检测文件和原输入协议 | T0 精确输入摘要与逐帧行映射 |
| `paths.features_dir` | 原微调 RGB ReID 缓存 | T0 检测索引与特征对齐 |
| `paths.images_root` | 开发评估原始 RGB 图像 | T0 解码像素与帧映射 |
| `paths.gt_root` | 官方 RGB 评估 GT | 独立离线评分／标记进程 |
| `paths.trackeval_root` | 历史固定 TrackEval 源码 | T0 摘要及官方口径 |
| `paths.train_images_root` | 官方训练来源 RGB 图像 | T1 训练采样与像素映射 |
| `paths.train_gt_root` | 官方训练来源 RGB 轨迹 GT | T1 离线监督构建 |
| `paths.split_manifest` | 已有数据来源／划分 JSON；T1 据此构造新头母视频划分 | T1 必须补充并验证真正的原视频映射 |
| `references.B.run_dir` | 内含 `protocols/B_early_moderate.json` 的原 B 运行根目录 | T0 复现与逐字节对照 |
| `references.BG.run_dir` | 内含 `protocols/evidence_both.json` 的运行根目录 | T0 对照修正版完整来源 |
| `output_root` | 新批次输出目录，默认 `outputs/tco_v0.1_pilot` | 准备工具拒绝覆盖 |

`split_manifest` 初次可以指向现有官方来源划分清单；目录存在或 JSON 可解析，**不表示已经满足母视频隔离**。在 T1 形成新的 fit/calibration/holdout 映射，列出同母视频排除项并冻结。若历史 detector/ReID 曾见过这些训练图像，必须记录；新头留出不自动等同整个系统从未见过的测试集。

新协议的模型、历史和训练预算在 `configs/research/tco_v0.1_protocol.json` 中。它们是本次前瞻设计默认值，未被实验选优。支持、定位质量和质量增量的三个阈值为 `null`，必须在校准集合拟合并封存后才能进入开启模型的闭环回放。

训练来源同样需要冻结检测与 ReID 的真实缓存。若现有缓存只覆盖开发 30 段，本地应先用相同冻结权重对训练来源生成完整逐帧检测／特征，再构造 T1 运行与样本；不能把开发缓存当作训练输入。准备工具目前只登记资产位置，后续导出器应在输入清单中分别列出各划分的检测、特征与来源摘要。

## 4. 当前可运行命令

### 检查配置与资产位置

```bash
python -X utf8 tools/manage_tco_batch.py check --settings configs/research/tco_v0.1.local.json
```

此命令只读，不创建输出目录。它检查配置约束、目录、小型 JSON 协议和任务引用，并记录小型元数据摘要。**不会扫描图像内容、验证 GT 身份、执行 TrackEval 或证明两个运行完全等价。**

退出码：0 表示本命令的元数据检查完成；3 表示资产未配置／未准备；2 表示输入、协议或路径有错误。脚本化流程可使用 `check --strict`，此时缺资产也返回 2。

### 准备独立批次

```bash
python -X utf8 tools/manage_tco_batch.py prepare --settings configs/research/tco_v0.1.local.json
```

只有资产位置和小型元数据检查通过后，才创建全新 `output_root`。它保存：

- `batch_manifest.json`：来源路径、摘要、准备状态；
- `preparation_check.json`：本次检查与明确未检查的项目；
- `task_ledger.json`：全部阶段初始为 `PLANNED`；
- `frozen/`：本地设置、设计协议、任务表、划分来源和参考协议副本。

准备状态为 **`PREPARED_NOT_RUN`**。已有输出目录不会覆盖。失败后保留必要日志，修改本地配置指向新的输出目录重试，不能改一个状态字符串代替真实复核。

### 查看准备记录

```bash
python -X utf8 tools/manage_tco_batch.py status --settings configs/research/tco_v0.1.local.json
```

该工具核对准备快照及准备工具源码摘要。快照、本地设置或工具源码变化会报告 `SNAPSHOT_MISMATCH`，退出码 2；工具更新后应为新批次重新准备，不能把旧快照冒充新版本。任务台账后续被更新时，会展示 `LEDGER_CHANGED_UNVERIFIED` 和记录内容；这是它对职责范围的说明，并非实验失败。实验是否通过由各阶段真实输出及核验报告确定。

## 5. 给本地 Codex 的首个任务

可以将以下任务直接交给本地工作区中的 Codex：

> 阅读 `docs/botsort_plus/tco_v0.1/`、设计协议和任务表，接续本分支。先完成实验协议的 I0–I1：核验两参考运行与实际部署源码，按数据接口实现只读候选／历史导出及独立离线标记，形成 T0 等价性结果和 T1 可达性／划分报告。采样必须来自真实预测状态；保留未知身份、原检测行、原 guard 允许边和所有资格失败原因。此阶段不训练网络、不修改参考跟踪决策。通过源代码核验和软件小样本后，在指定真实训练／开发资产上执行。若参考来源不一致，停止下游阶段并报告具体差异。产物就绪后，根据实验协议的继续条件选择下一工作包，不用占位结果替代尚未完成的实现。

T0/T1 就绪后，再执行 I2–I6。每个工作包结束应留下：实际修改、运行命令、数据与代码摘要、结果、限制、下一步状态。技术上的继续条件不要求额外的形式化批准；满足已经约定的范围和验收条件即可推进。超出 v0.1 范围的机制应使用独立协议版本，保留此前结果。

## 6. 实验运行器的本地实现要求

本地实现完成后，应提供真实支持 `--help` 的命令入口，并把可执行命令回填到本文档；在那之前不要复制并执行推测的 `train_tco.py` 或 `run_tco.py` 命令。

运行器至少区分：只读采样、离线标记、训练、校准、关闭／只记录一致性、T3a 闭环、条件 T3b 和正式评分。预测侧配置采用明确白名单，不能将 GT 根目录、离线标签表或未来信息传入在线模块。

`evidence_both` 不能仅通过普通 B 配置改名复现。其实际包装器包含相机候选选择、低分双重确认和长间隔证据规则。应复用并核对 [round3 worker](../../../experiments/vttiny_gmc/round3_joint/worker.py) 及部署来源，而后在新增适配器中接入 TCO。

## 7. 回传与版本管理

优先回传每阶段报告、协议、输入／源码摘要、精确逐事件表、逐序列及联合指标、失败日志；训练阶段额外回传校准参数、种子和最优检查点选择依据。完整原图、GT、权重与大型缓存保留在实验环境，不提交进仓库。

新结果建议归档到 `results/vttiny/iterations/tco-v0.1/`（当前尚不存在）。每次结果提交注明真实执行范围、官方 30 段的开发性质及未完成项目。E3 人审继续使用既有 `tools/run_v04_joint_batch.py e3-prepare/e3-check/e3-report` 的运行手册，与 TCO 训练材料分开。
