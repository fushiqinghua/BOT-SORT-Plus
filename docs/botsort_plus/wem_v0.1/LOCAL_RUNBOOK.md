# W v0.1 本地执行与交接手册

## 1. 这次可以立即运行什么

本分支提供两个真实命令入口：

- tools/manage_wem_batch.py：初始化配置、检查路径、保存计划快照、查看状态。
- tools/audit_wem_s0.py：遍历现有原生检测缓存，完成精确分数掩码及 S0 库存审计。

它们只依赖 Python 标准库。后续 R1b 轨迹状态采样、R2/R3 验证和 W 在线模块仍是[本地实现任务](IMPLEMENTATION_TASKS.md)，不能把目录和计划的存在理解为已经具备完整 W 运行器。

## 2. 拉取新分支

在现有 BOT-SORT-Plus 仓库内，首次执行：

~~~bash
git fetch origin
git switch --track origin/research/vttiny-weak-evidence
~~~

如果本地已有该分支：

~~~bash
git switch research/vttiny-weak-evidence
git pull --ff-only origin research/vttiny-weak-evidence
~~~

分支基点是 176f2a81613c34d16a860765ea7be0845116b534，包含最新 TCO/Q2 归档。本轮没有修改主跟踪代码和历史结果。

## 3. 初始化本地配置

在仓库根目录运行：

~~~bash
python -X utf8 tools/manage_wem_batch.py init --output configs/research/wem_v0.1.local.json
~~~

命令拒绝覆盖已有本地配置。已有配置直接编辑并检查即可。

路径可以用本机绝对路径；配置中的相对路径按仓库根目录解析。Windows JSON 建议写正斜杠，例如 E:/research/vttiny/detections。不要把 Windows 盘符路径原样传给 Linux。脚本不猜测历史实验文件所在位置。

### 第一批必填

- paths.detections_dir：包含逐序列检测缓存的目录，每个 JSON 为 frames 数组，各帧含 frame 与 detections。
- thresholds.low / high：核对真实参考运行，当前设计基线为 0.1 / 0.5。
- output_root：建议独立 runs/wem_v0.1 或仓库之外的新目录。

### 后续阶段补齐

| 字段 | 内容 | 用途 |
| --- | --- | --- |
| paths.baseline_root | 含检测协议与冻结资产说明的参考根目录 | R0 来源验证 |
| paths.features_dir | 与原检测行对齐的 RGB ReID 缓存 | R0/R1b；S0 行通常另需按需提取 |
| paths.images_root | 原 RGB 图像 | 弱候选特征与人工复核 |
| paths.gt_root | 官方 RGB GT | 仅离线标记与评价 |
| paths.trackeval_root | 已冻结版本的 TrackEval | 离线 CLEAR 与完整指标 |
| paths.source_group_manifest | 已核验或明确标注为代理的来源分组 JSON | 校准/留出隔离 |
| references.B.run_dir | B_early_moderate 的冻结运行 | 参考复现 |
| references.BG.run_dir | evidence_both 的冻结运行 | 联合参考复现 |

以上字段初始为 null，R1a 不需要补齐 GT 或图像。不能把缺少后续资产误写为第一批工具无法执行。

## 4. 检查并准备

~~~bash
python -X utf8 tools/manage_wem_batch.py check --settings configs/research/wem_v0.1.local.json --strict
python -X utf8 tools/manage_wem_batch.py prepare --settings configs/research/wem_v0.1.local.json
python -X utf8 tools/manage_wem_batch.py status --settings configs/research/wem_v0.1.local.json
~~~

check 核对本地配置和路径，strict 只将第一批必要输入作为执行条件。prepare 保存设置、研究协议及来源摘要，状态为 PREPARED_NOT_RUN。

这些命令没有解码图像、核对真实 ReID、运行参考跟踪或认证 R0 已通过。后续资产是否存在会单独显示。

准备后改变设置或研究协议应使用新的 output_root，避免不同批次混在同一目录。状态命令不会自动将候选库存晋升为实验通过。

## 5. 执行 R1a

优先复制 prepare 打印的完整命令，它已代入本地资产和输出位置。格式为：

~~~bash
python -X utf8 tools/audit_wem_s0.py --detections-dir "<实际检测目录>" --output-dir "runs/wem_v0.1/r1a_s0" --low-thresh 0.1 --high-thresh 0.5
~~~

工具验证输入后才发布结果，不修改输入。已有非空输出目录不得覆盖；重跑用 r1a_s0_rerun01 等新目录，并保留上次 manifest。

第一次可用一个实际存在的序列检查格式：

~~~bash
python -X utf8 tools/audit_wem_s0.py --detections-dir "<实际检测目录>" --output-dir "runs/wem_v0.1/r1a_one_sequence" --low-thresh 0.1 --high-thresh 0.5 --sequence "<实际序列名>"
~~~

序列名为缓存文件名去掉 .json，可重复传入 --sequence。检查无误后去掉 --sequence 执行全部缓存。单序列只是格式检查，不能当作完整覆盖诊断。

输出：

| 文件 | 含义 |
| --- | --- |
| manifest.json | 输入文件摘要、参数、版本和范围 |
| summary.json | 总体库存与解释边界 |
| per_sequence.csv | 逐序列分布 |
| s0_candidates.jsonl | 原始行索引、分数、框、类别和来源标记 |

结果仅代表库存。ReID 导出策略标记是按代码规则推断的覆盖条件，不是读取 npz 后验证实际特征有效性。

## 6. 第一批结果怎样回传

优先回传：

1. manifest.json；
2. summary.json；
3. per_sequence.csv；
4. 原生参考 low/high 与 detector export 协议的摘要；
5. 是否全量、是否官方开发 30 段、是否存在缺帧/缺序列；
6. 若文件体积可控，再回传 s0_candidates.jsonl；否则保留本地全量，提供压缩文件或摘要及可追溯抽样。

发布前将本机用户名/绝对路径替换为统一资产别名，保留原始本地 manifest 和内容哈希，记录路径重写事实。不要改写实验数值。原图、模型权重和大缓存留在本地，提交目录见[结果模板](RESULTS_TEMPLATE.md)。

原生分数边界和 ReID 策略分歧是诊断信息，不代表已经发现可用身份；很多弱候选可能是背景或重复响应。

## 7. 第一批之后，本地 Codex 应做什么

请依次阅读 MODULE_DESIGN.md、DATA_CONTRACT.md、EXPERIMENT_PLAN.md 和 IMPLEMENTATION_TASKS.md。

先实现 R0/R1b 的只读采样和离线标记，再做 R2 的冻结状态比较；确认机会及信息增量后才做 R3 校准。R4 需要明确的验收文件，不通过时保持 W 关闭。

可以参考已有 tools/research/tco_t01 的源码和历史复现方法，但它针对原生低分已配对观测，并有固定提交/输入检查，不能直接当成 W 采样器。特别需要新记录帧首确认状态、B 后所有权和未采用 S0 行。

禁止为适配新分支而删除旧工具的来源校验，或把 TCO 结果目录改名后冒充 W 结果。历史核验用于继承证据；新采样器仍须独立验证。

## 8. 执行环境

首批工具需要 Python 3.10 及以上，建议沿用项目的 Python 3.11；无需安装新包。执行时统一加 -X utf8。Windows 打印的下一条命令供 PowerShell 使用，POSIX 主机打印相应 shell 命令。

后续跟踪采样继续使用此前已核验的 BoxMOT 16.0.11、冻结检测缓存、RGB ReID 和相机缓存。不要为 W 准备工作升级整个跟踪环境。

## 9. 常见解释错误

- 有 S0 框，不等于有正确目标。
- 有 GT 几何重叠，不等于已证实该旧轨迹的身份。
- 有库存，不等于轨迹当时符合 W 条件。
- 有 R2 受约束机会，不等于完整 HOTA 上限。
- W 与 W-coast 相当，只能说明坐标更新未显示额外收益；还要看 W-coast 是否优于 Coast-only。
- 历史 30 段及旧 Q2 留出已经被多轮使用，不能重命名为新的盲测集。

