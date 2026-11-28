# W R4.2 本地执行手册

本批默认复用上一轮本地完整运行的 `BG_off`、`BG_W0`，新增三组完整跟踪，再统一评测五组。复用只读取并复制旧产物，不修改 R4.1 运行目录。

[实验定义](EXPERIMENT_PLAN.md) · [结果模板](RESULTS_TEMPLATE.md) · [结果入口](../../../results/vttiny/wem_r4_2/README.md)

## 1. 拉取代码

在已有 BOT-SORT-Plus 本地仓库根目录运行，继续使用原分支：

```bash
git fetch origin
git switch research/vttiny-weak-evidence
git pull --ff-only origin research/vttiny-weak-evidence
```

使用上一轮成功执行 R4.1 的 Python 环境。输入仍为同一份冻结检测缓存、常规 ReID 特征、相机补偿、GT 和 TrackEval。该批不训练检测器或 ReID，也不重新估计相机补偿。

## 2. 从上一轮配置初始化

假设上一轮本地配置为 `configs/research/wem_r4_1.local.json`：

```bash
python -X utf8 tools/run_wem_r42.py init --from-r41-settings configs/research/wem_r4_1.local.json --settings configs/research/wem_r4_2.local.json
```

默认从旧配置的 `paths.output_root/full` 定位完整运行目录，并采用 `reuse` 模式。若旧批次后来搬到了其他目录，初始化时明确传入：

```bash
python -X utf8 tools/run_wem_r42.py init --from-r41-settings configs/research/wem_r4_1.local.json --r41-batch /path/to/wem_r4_1/full --settings configs/research/wem_r4_2.local.json
```

将 `/path/to/wem_r4_1/full` 换成实际目录。Windows 可使用带引号的正斜杠路径，如 `"E:/tracking_runs/wem_r4_1/full"`。这里应指向含原始预测、事件、状态、协议和完成凭据的本地批次，不能指向 GitHub 上只包含派生报告的 `results/vttiny/wem_r4_1/full_20261128`。

生成的配置结构为：

```json
{
  "schema": "wem_r42_settings",
  "r41_settings": "configs/research/wem_r4_1.local.json",
  "r41_batch": "runs/wem_r4_1/full",
  "reference_mode": "reuse",
  "output_root": "runs/wem_r4_2",
  "runtime": {}
}
```

实际初始化结果可能已经写为解析后的绝对路径，以生成文件为准。相对路径按仓库根目录解析。`runtime` 留空时继承上一轮配置；需要降低并发时，可明确写 `"runtime": {"workers": 1}`。路径变化通过配置处理，不编辑冻结实验协议。

### 没有完整的旧运行目录

可以显式选择重跑两个参考：

```bash
python -X utf8 tools/run_wem_r42.py init --from-r41-settings configs/research/wem_r4_1.local.json --reference-mode rerun --settings configs/research/wem_r4_2.rerun.local.json
```

此时全量阶段执行五组跟踪。若之前已经使用默认输出目录执行了另一套配置，为本次设置一个新的 `output_root`。不要将旧批次的完成凭据复制到新配置，也不要为通过检查手动改哈希。

`rerun` 只解决缺少可复用参考的问题，不绕过冻结输入、原始参考或评测器的一致性校验。

## 3. 先检查输入与复用来源

```bash
python -X utf8 tools/run_wem_r42.py check --settings configs/research/wem_r4_2.local.json
```

此步骤检查冻结资产、完整序列范围、代码与协议，以及待复用参考的完整性。通过后表示输入可供执行，不代表 GPU 跟踪或数据集实验已经完成。

如果检查失败，按报错定位具体文件或范围差异。只有旧完整运行目录缺失时才考虑 `rerun`；若是检测缓存、相机补偿或 GT 指纹变化，应恢复本批所规定的资产，或另建明确的新协议，不能混用数据后沿用既有结论。

## 4. 一条命令完成预定批次

```bash
python -X utf8 tools/run_wem_r42.py full --settings configs/research/wem_r4_2.local.json
```

`full` 按依赖顺序执行：

1. 三段技术 smoke：`DJI_0030_2`、`DJI_0061_2`、`DJI_0327_3`，覆盖本批五种策略；
2. 导入核验后的 BG_off、BG_W0，或在 `rerun` 模式下重新运行参考；
3. 执行 BG_lower_low、BG_W0_relaxed_anchor、BG_W0_emit_only 的全部 30 段；
4. 对五组完整预测进行同版 TrackEval 评测；
5. 离线生成覆盖、弱观测和 B 恢复归因；
6. 生成最终比较报告。

技术 smoke 检查参考一致性和干预边界，不按指标好坏选择是否继续。全量阶段不设置正收益、人工标注数量或候选数量门槛。发生真实技术错误时保留输出与日志，修复原因后重用同一命令继续。

如需先单独运行技术检查：

```bash
python -X utf8 tools/run_wem_r42.py smoke --settings configs/research/wem_r4_2.local.json
```

之后仍执行 `full`。已完成且摘要匹配的工作可以复用，不能手工修改已完成目录中的文件后继续把它作为同一次运行。

## 5. 中断、恢复与只重建报告

中断后先确认上一进程已经停止，再以原命令、原配置恢复。保留完整日志；不要同时让两个进程写入同一个输出根目录。

已完成组须通过完成标记和文件摘要核验后复用；不完整产物会保留在备份位置后重跑。输入、代码或设置发生变化时，应选择新的输出根目录，不能覆盖原实验。

预测已经完成，单独执行后处理：

```bash
python -X utf8 tools/run_wem_r42.py evaluate --settings configs/research/wem_r4_2.local.json
python -X utf8 tools/run_wem_r42.py summarize --settings configs/research/wem_r4_2.local.json
```

后处理仍检查完成产物与输入来源，不把缺失组或缺失序列当作零分。

## 6. 读结果的顺序

默认新输出根目录为 `runs/wem_r4_2`，其中 `smoke/` 保存技术验证，`full/` 保存完整五组、评测和报告。

首先阅读 `full/report/REPORT.md`，然后核对：

| 内容 | 应确认的问题 |
| --- | --- |
| 五组完整指标及逐序列配对 | 是否全部覆盖 30 段，参考是否统一为 BG_off |
| W0 与 lower_low 的差值 | HOTA/IDF1、召回与 FP/IDSW 的交换是否清楚 |
| relaxed_anchor 与 W0 的差值及曝光计数 | 放宽是否真实改变资格/提交，改变后有何收益与代价 |
| W0 与 emit_only 的差值 | 正式状态反馈是否改善完整轨迹，而不只改变当帧输出 |
| 弱观测和 B 恢复归因 | 全程新增/丢失覆盖是否与 FN 差值一致 |
| 运行与参考来源记录 | 哪些组来自旧批次，哪些组为本轮新运行 |

`emit_only` 的正式状态链和原生事件应与 BG_off 一致，但最终 MOT 可以多出弱观测输出。状态摘要采用原生输出，实际发布预测包含附加输出，两者在协议中分别说明。该检查只能证明正式状态没有接受弱观测反馈，不能据此宣称实际输出与关闭组相同。

报告中的 `weak_outputs_emitted` 是弱框输出数，`weak_state_updates` 才是正式弱状态更新数。emit_only 后者应为零；离线逐提交表将该组标为 `emitted_measurement_no_state_feedback` 并记录 `state_committed=0`。

本轮计时包含诊断、缓存和运行环境影响，不作为最终端到端 FPS。不同日期复用的参考耗时不应直接用于速度优劣结论。

## 7. 回传要求

依据 [结果模板](RESULTS_TEMPLATE.md) 整理结果，至少保留：

- 五组完整指标、所有逐序列配对与三项主比较；
- 候选曝光和约束变化计数；
- 弱观测提交、全程覆盖与 B 恢复归因；
- 代码提交、冻结输入指纹、参考复用凭据、技术检查和完整运行状态；
- 中断、重试、无提交或零变化情况；
- 两到三个能够解释主要收益或失败的具体序列事件。

将可公开的结果归档到 `results/vttiny/wem_r4_2/` 下带实际日期的子目录，并更新结果入口。预测所需的完整 MOT、缓存和本机配置保留在本地供进一步复核。公开文件经过本机路径清理或换行归一时，记录原始与发布文件的摘要关系。

结果负向或完全相同都应回传。下一次讨论依据整个预定矩阵决定最终 W 规则，不在实验途中修改门限后只保留有利版本。
