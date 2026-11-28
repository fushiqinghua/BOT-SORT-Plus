# W R3.1 本地执行手册

本批在原分支 **`research/vttiny-weak-evidence`** 执行 W0–W3 四组固定对照，附带 W_old 历史参照。使用已经生成的 R3 分数与标签缓存，运行不需要 GPU、图像、ReID 权重或原始 GT 文件。

[已有 R4 探索性闭环](../../../results/vttiny/wem_v0.1/r4/README.md) 已完成八组回放；其中使用余弦 0.75 的探索点，本批 W_old 重算的是 R3 余弦 0.85 历史对照。两者分别保留，本批不重复旧八组，新增的是 W0–W3 的缓存关联消融。

[实验定义](EXPERIMENT_PLAN.md) · [结果模板](RESULTS_TEMPLATE.md) · [归档目录](../../../results/vttiny/wem_r3_1/README.md)

## 1. 更新原分支

在本地仓库根目录执行：

```bash
git fetch origin
git switch research/vttiny-weak-evidence
git pull --ff-only origin research/vttiny-weak-evidence
```

沿用现有 Python 3.11 环境，需 NumPy 和 SciPy。若只检查依赖，可执行：

```bash
python -c "import sys, numpy, scipy; print(sys.version); print(numpy.__version__, scipy.__version__)"
```

本批工具读取冻结缓存后离线重新评分，不调用旧的远程回放入口。

## 2. 找到本地已有的三类输入

| 配置路径 | 应指向什么 | 内容识别方式 |
| --- | --- | --- |
| `features_dir` | 原 R3 特征评分输出目录 | 包含 `manifest.json` 和逐序列 `*.jsonl.gz`；这些 JSONL 保存 IoU、运动、余弦与竞争信息 |
| `labels_dir` | 原 R3 离线候选标签根目录 | 包含 `summary.json`，以及 `fit`、`calibration`、`holdout` 三个子目录中的 `candidate_edges.jsonl.gz` |
| `split_file` | 原始冻结训练划分文件 | 即原 R3 使用的 `training_split.json` |
| `output_dir` | 本次新输出目录 | 与上述输入及旧 R3 结果目录分开 |

**仓库里公开的 `calibration_grid.csv`、`calibration_lock.json` 和 `source_receipts.json` 是结果摘要，不能替代逐候选输入。** 本地已有的 `human_review_pending.csv` 也不能代替 `candidate_edges.jsonl.gz`。

需要的是原 R3 的评分 JSONL，不是最早的检测 JSON 缓存，也不是原始常规框 ReID 的 NPZ 目录。逐序列分数文件应保留完整竞争字段；只有总体统计时无法重建 W2/W3 的关联选择。

若原 R3 在服务器执行，本次可以直接在拥有上述缓存的环境中运行，或者将其完整复制到本地。沿用已有授权与下载方式；本批不提供新的服务器账户或固定机器路径。原缓存恢复说明见 [R3 复现入口](../../../tools/research/vttiny_wem_r3/README.md)。

## 3. 初始化本机配置

```bash
python -X utf8 tools/run_wem_r31_ablation.py init --settings configs/research/wem_r3_1.local.json
```

打开生成的配置，填写 `paths` 中的四项路径。以下只是格式示意，需替换为自己的真实目录：

```json
{
  "schema": "wem_r31_settings",
  "scope": "full",
  "paths": {
    "features_dir": "D:/tracking/r3/features_full",
    "labels_dir": "D:/tracking/r3/labels_full",
    "split_file": "D:/tracking/frozen/training_split.json",
    "output_dir": "runs/wem_r3_1/full"
  }
}
```

保留工具生成的其他字段。Windows JSON 路径建议使用 `/`；若使用反斜杠，应写成 `\\`。中文和带空格目录可直接作为 JSON 字符串填写。

`scope=full` 对应原 R3 完整训练侧缓存。只有当输入确实是原 R3 的 smoke 输出时才使用 `scope=smoke`。改变配置中的 scope 不会自动把完整缓存抽成小样本。

本批参数固定在实现与实验协议中。本机配置用于指定输入、执行范围和输出，不用于逐次搜索权重或阈值。

## 4. 先检查，再运行

```bash
python -X utf8 tools/run_wem_r31_ablation.py check --settings configs/research/wem_r3_1.local.json
```

检查应确认输入可读、来源凭据对应、序列覆盖一致、候选和标签能够按原始键对齐。完整运行绑定本次已归档 R3 的原始特征与标签清单哈希；如已另行更新人工标签，请保留原版作为本轮输入，并为标签更新建立新的实验版本。若失败，先根据错误定位缺失文件或不一致输入，不要改写原始凭据中的 SHA 来绕过校验。

检查通过后执行：

```bash
python -X utf8 tools/run_wem_r31_ablation.py run --settings configs/research/wem_r3_1.local.json
```

这一条命令按固定定义运行 W0、W1、W2、W3 和 W_old，不需要手动连续改五次配置。重复运行时使用新的输出目录，保留上一轮原始产物。

## 5. 如何阅读输出

| 文件 | 优先查看的内容 |
| --- | --- |
| `manifest.json` | 输入摘要、配置、实现和执行范围，确认这次比较来自同一批缓存 |
| `summary.json` | 整体状态、执行范围、计数和解释限制 |
| `protocol.json` | 本次固定实验协议快照 |
| `comparison.csv` | 各分区五组方法的主要计数；先查看 W0、W_old 对照，再比较 W1–W3 |
| `per_sequence.csv` | 是否主要由某个片段贡献或损伤 |
| `per_source.csv` | 来源组层面的覆盖与变化 |
| `method_deltas.csv` | W1−W0、W2−W1、W3−W2 等方法差异 |
| `decisions.jsonl.gz` | 逐候选评分、拒绝原因和最终指派，用于核查具体选择 |
| `review_queue.csv` | 待复核候选及对应方法，优先处理实际采用的未知候选 |

完整运行会自动核验 W0 与 W_old 是否重现原 R3 的 6 个分区对照；不一致时不会发布输出目录。`scope=smoke` 的小缓存不与完整 74 段汇总强行比较。

`check` 成功状态为 `INPUTS_VERIFIED_NOT_RUN`；`run` 完成状态为 `OFFLINE_COMPARISON_COMPLETE`，科研准入状态仍为 `NOT_EVALUATED`。错误退出码为 2。

随后依次看：

1. W1 是否在保留正确机会的同时减少已知错误或未知采用。
2. W2 是否真正改变了竞争场景中的候选选择；没有竞争时与 W1 相同属于预期可能结果。
3. W3 改变了哪些案例，因真实外观竞争拒绝还是因比较信息缺失拒绝。
4. 结果是否跨来源成立，以及未知标签是否仍然阻塞结论。

不要仅用“采用数量更多”判断有效。未知采用需要进一步裁决，脚本执行完成不等于线上工作点通过。

## 6. 回传材料

建议在 [结果归档目录](../../../results/vttiny/wem_r3_1/README.md) 下新建本次运行子目录，优先提交：

- `manifest.json`、`summary.json`；
- `comparison.csv`、`per_sequence.csv`、`per_source.csv`、`method_deltas.csv`；
- `review_queue.csv`；
- 按 [RESULTS_TEMPLATE.md](RESULTS_TEMPLATE.md) 填写的运行与解释记录。

`decisions.jsonl.gz` 保留完整本地原件，体积合适时一并归档；否则给出哈希、保存位置说明和需要复核的摘录。运行配置中涉及本机绝对路径时，可另给脱敏副本并注明原文件哈希，不修改原始产物冒充其原字节。

人工复核意见独立保存，明确 reviewer、状态和证据。模型辅助建议不自动变成人工标签；此轮不要手工覆盖原 R3 冻结标签后仍宣称输入未变。

## 7. 常见情况

| 情况 | 处理 |
| --- | --- |
| 找不到 `manifest.json` 或逐序列分数文件 | 当前目录不是 R3 特征评分缓存；按第 2 节重新定位 |
| 只有仓库公开摘要，没有完整缓存 | 从自己的原 R3 环境恢复逐候选输出；摘要不能重建关联图 |
| 输入指纹、帧或行号不一致 | 核对是否混用了 smoke/full、不同回放或重新导出的候选；保留错误信息 |
| 输出目录已有结果 | 配置一个新输出目录，保留旧结果 |
| W2 与 W1 完全相同 | 查看是否只有单一可行候选或冲突很少，检查实际改变指派的数量 |
| W3 大量因比较不可用拒绝 | 区分缺失证据与真实竞争失败，不把缺失当成高置信度拒绝 |
| 未知采用增加 | 从复核队列定位，不能把它们直接当作新增误报或新增正确目标 |

本批 R3.1 不运行 R4、不修改跟踪器、不产生新 MOT 结果。已有 R4 指标见上方归档入口；本批新关联规则若需评测 HOTA、IDF1 或在线 KF 更新作用，应另行通过后续闭环验证。
