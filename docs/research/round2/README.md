# 第二批实验：M2 错因修正、M1 稳健补偿与 M3 RGB 恢复

本批继续使用 **`research/m2-association-m1-m3-diagnostics`** 分支，承接结果提交 `39f6f676cbb9ba26ef1913a81fc163607ad36995`。第一批结果保存在原目录；本批增加实现、预声明配置和可重复执行的任务计划。

**先运行原生基线并逐帧核对存档，再分别运行 M2、M1、M3，最后统一汇总。** 本批实现已做软件验证；真实 VT-Tiny-MOT 的第二批结果由本地运行产生，仓库没有预填指标或改进结论。

## 1. 本批检验的假设

第一批 [MOTA 错因分析](../results/round1_2026-11-25/mota_attribution/README.md) 表明，floor R 组出现了较多输出框定位损失。改变当帧更新噪声和改变输出坐标，均能在已选案例或离线对照中缓解该现象，但这些观察还不能代替全流程重放。因此本批首先把关联、更新和输出三个因素拆开。

| 实验组 | 本批实际实现 | 主要对照与问题 |
|---|---|---|
| M2 | 关联 R / 更新 R 分离；KF / 当前检测框输出；第一或第二阶段单独 NLL | 更新过弱、低分延续和关联选择各自贡献了多少错误？ |
| M2 简单对照 | 第一阶段 IoU 阈值 `.8/.85/.9`；NWD 原图尺度常数 `8/12.8/20` | 更简单的方法能否达到同等收益？仅在 9 段 calibration 选择参数 |
| M1 | 原图像素一致的拟合、前景遮罩与均匀选点、前后向光流筛选、独立留出点验证与模型回退 | 能否降低不可靠补偿的损失，同时保留有效相机运动补偿？ |
| M3 | 原生 RGB ReID；全局 / 时间间隔门限；使用可信特征的显式 Lost 恢复 | 长间隔外观证据和独立恢复阶段能否改善身份连续性？ |
| 检测覆盖 | 多个 IoU 门槛、四个候选池的逐帧一对一最大匹配 | 现有检测候选本身覆盖了多少 GT，哪些尺寸最受限？ |

M1、M2、M3 分别改变基线的一部分，本批不自动选择三者组合。M3 使用原先已有的高分 RGB 特征；低分特征补提取、热红外 ReID 训练、扩大轨迹保留期和补帧不在本批执行范围内。

## 2. 拉取与运行环境

在已有仓库内执行：

```bash
git fetch origin
git switch research/m2-association-m1-m3-diagnostics
git pull --ff-only origin research/m2-association-m1-m3-diagnostics
```

继续使用第一批通过复现验收的 Python / BoxMOT / TrackEval 环境。运行器校验被审查的 BoxMOT `v16.0.11` 核心源码以及 TrackEval 提交 `12c8791b303e0a0b50f753af204249e622d0281a`；包内旧 `__version__` 字符串不能代替源码校验。无需为本批重新训练检测器或 ReID。

可先运行软件测试，`TRACKEVAL_ROOT` 指向已有固定版本目录：

```bash
M2_TEST_BOXMOT=1 TRACKEVAL_ROOT=/path/to/TrackEval \
  python -m unittest discover -s tests -v
```

这些是软件约束与合成数据测试；真实任务收益仍由后面的数据集实验决定。详细验证范围见 [VALIDATION.md](VALIDATION.md)。

## 3. 一次填写本地路径，生成冻结计划

下面的路径替换成第一批使用的真实目录。推荐给整个批次使用**与 RGB 特征原始配套的检测缓存**，避免路径规范化后的 JSON 字节变化造成特征元数据哈希不一致。

```bash
python -m botsort_research.round2 plan \
  --cache-dir /path/to/original/rgbt/detections \
  --image-root /path/to/VT-Tiny-MOT \
  --gt-root /path/to/GT/GT_00/test \
  --embeddings-dir /path/to/original/rgbt/embeddings \
  --trackeval-root /path/to/TrackEval \
  --output outputs/round2
```

路径含义：

| 参数 | 目录应包含 |
|---|---|
| `--cache-dir` | `<sequence>.json` 或 `.json.gz`，一个序列只保留一种形式 |
| `--image-root` | `test2017/<sequence>/00/<image>`；原始 RGB 图像用于原 SOF 或 M1 |
| `--gt-root` | `<sequence>/gt/gt.txt`，与第一批官方 RGB 坐标真值一致 |
| `--embeddings-dir` | 每个序列原始 `.npz` 和同名 `.json` 元数据，不能只给权重文件 |
| `--trackeval-root` | 固定版本的 `trackeval/` 包所在目录 |

若 BoxMOT 放在原始独立运行时内，另加 `--runtime-root /path/to/runtime`，其中应有 `tracking_runtime/boxmot/`。已在当前 Python 中安装正确运行时则不需要这个参数。

没有特征缓存时，可以不写 `--embeddings-dir`，先完成 M2、M1 和覆盖率诊断。生成计划会明确标记 M3 未启用；之后不能把这种计划的 `all` 理解为已完成 M3。找到配套特征后，建立另一个输出目录的计划并运行 `--phase m3`。

`plan` 生成以下文件，并不会开始重放：

- `plan.json`：任务依赖、绝对路径、输入指纹、源码指纹和分组。
- `config/`：本批全部配置以及 9/21 划分的快照。
- `commands.md`：每个实际命令，便于审阅和单独复查。

既有输出目录不会被覆盖；要尝试修改过的配置，请创建新批次目录。计划生成后修改源码、输入或配置快照，执行器会拒绝把两种条件混到同一批结果中。

## 4. 推荐执行顺序

### 4.1 先通过预检和原生基线复现

```bash
python -m botsort_research.round2 run \
  --plan outputs/round2/plan.json --phase preflight --resume
```

预检核对检测、图像路径、运行时；提供特征时也核对特征哈希、逐帧索引、维度、编码器和有效向量。随后在完整 21 段 evaluation 上重放 `native_iou` 与 `native_iou_det`，并把 `native_iou` 与归档 `botsort_no_reid` 轨迹逐帧比较。只有通过原生复现后才继续解释其他实验。

每个 replay、离线评估和离线诊断都在独立进程执行，避免预测进程的 GT 访问限制被评估流程影响。检测模型、候选顺序、输入帧和 GT 都有指纹记录。

### 4.2 优先完成 M2

```bash
python -m botsort_research.round2 run \
  --plan outputs/round2/plan.json --phase m2 --resume
```

核心组一次运行 8 个变体；结合预检中的两种基线输出，可分别观察关联、更新、输出和第二阶段的作用。随后执行 IoU 和 NWD 的小规模 calibration 对照。每种简单方法按预先固定的顺序选择：**HOTA 较高 → IDF1 较高 → FP+FN+IDSW 较少 → 配置列表中靠前**。选定参数与完整候选分数写入 `selected_iou/`、`selected_nwd/`，再各自重放 21 段。

不得从多个 21 段结果中再挑一个参数，称作 calibration 选出的结果。NLL 使用第一轮承接的参数，并未获得额外的 evaluation 网格搜索预算。完整变体矩阵和公式见 [M2.md](M2.md)。

### 4.3 M1 分阶段消融

```bash
python -m botsort_research.round2 run \
  --plan outputs/round2/plan.json --phase m1 --resume
```

运行原生 SOF、关闭 CMC，以及四个逐步增加机制的 M1 变体。各 M1 设置先在 calibration 子集运行，再运行同一设置的 evaluation；脚本不会根据 evaluation 指标重选可靠性门限。calibration 对照自动汇总到 `m1_calibration_comparison/`。

所有 M1 组使用原生 IoU 和原生 KF。逐帧光流质量、变换选择和回退原因保存在 trace，逐序列计数在 `experiment.json` 的 `cmc_summary` 中。低残差、高内点比例均只是几何代理量，不能当作真实相机运动 GT。详细机制见 [M1.md](M1.md)。

### 4.4 M3 高分 RGB 外观与显式恢复

```bash
python -m botsort_research.round2 run \
  --plan outputs/round2/plan.json --phase m3 --resume
```

先重放 `native_reid` 并与归档 `botsort_reid` 对齐，再使用 9 段 calibration 配对拟合门限，然后运行 4 个实验变体。全局固定门限与按间隔门限有成对对照；两种显式恢复方案也有固定 / 按间隔对照。

默认校准只使用同类不同身份的邻近负样本，约束所选配对的经验误接受率；稀疏时间组回退到全局门限。这个经验预算不等于实际在线误接率，必须同时查看恢复事件与 CLEAR 指标。恢复阶段的日志记录实际门限、候选距离、歧义程度以及拒绝原因。详情见 [M3.md](M3.md)。

### 4.5 候选覆盖率

```bash
python -m botsort_research.round2 run \
  --plan outputs/round2/plan.json --phase coverage --resume
```

分别对 calibration 与 evaluation 计算 IoU `.3/.5/.75` 的一对一最大匹配覆盖，候选池包括全部缓存框、高分框、两级关联可用框和出生可用框。同一个检测不能重复覆盖多个 GT。

该覆盖值是“只选择现有原始检测框”的条件上界，**不是 KF 定位、预测框、HOTA、MOTA 或身份恢复的上界**。尺寸分层来自同一次全局匹配，不能解释成每个尺寸单独优化得到的上界。

如果希望连续运行全部已启用阶段，可以用：

```bash
python -m botsort_research.round2 run \
  --plan outputs/round2/plan.json --phase all --resume
```

每次进入一个阶段会自动包含必要的基线依赖，已完成任务只有在输出哈希复核通过后才跳过。

## 5. 错误、中断与继续运行

执行器实时显示当前任务和日志路径，完整命令及输出位于 `outputs/round2/logs/`，状态位于 `state.json`。遇到错误会停在当前任务，不把失败任务当作完成。

如果只是进程中断、磁盘空间不足等执行问题，检查日志并解决后可运行：

```bash
python -m botsort_research.round2 run \
  --plan outputs/round2/plan.json --phase all --resume --retry-failed
```

`--retry-failed` 将未完成输出移入 `outputs/round2/failed/`，保留原字节，再重新执行该任务。它不会删除原输出，也不会绕过源码或输入发生变化的检查。若需要修改实验代码、换缓存或换参数，应生成新计划。

特征哈希不匹配时，请找到原本配套的检测与特征缓存，或明确另建缓存版本；不要只改元数据的 SHA 来让检查通过。原生轨迹复现失败时先定位运行时、输入或参数差异，再解释改进结果。

## 6. 输出与回传

每个实验目录 `outputs/round2/runs/<实验名>/` 包含：

| 文件 | 用途 |
|---|---|
| `experiment.json` | 源码、配置、检测 / 图像 / 特征指纹、实际开关、时间与状态 |
| `trackers/<variant>/data/*.txt` | 完整 MOT 输出 |
| `traces/<variant>/*.jsonl` | M2 关联、M1 补偿或 M3 外观 / 恢复事件 |
| `outputs/<variant>/*.jsonl` | 每条输出的 ID、准确源检测索引、原始框、KF 框和实际输出框 |
| `metrics/summary.json` | 固定 TrackEval 的正式合并与逐序列指标 |
| `analysis/clear_verification.csv` | 逐序列及合计的 FP/FN/IDSW 诊断计数与正式指标核对 |
| `analysis/frame_errors.csv` | 逐帧 FP/FN/IDSW，便于找退化区段 |
| `analysis/size_summary.csv`、`gap_summary.csv` | 尺寸、间隔和诊断身份变化 |
| `analysis/fp_cases.csv.gz`、`gt_frames.csv.gz` | 完整 FP 与每个 GT 目标帧的诊断明细 |
| `analysis/case_samples.csv` | 每个变体 / 序列 / FP 类型按时间取前 3 个定位案例；不是统计抽样 |
| `analysis/recovery_cases.csv`、`recovery_summary.csv` | 显式恢复前最近可验证 GT 与当前 CLEAR 归属，区分同身份接回、不同身份接回及不可验证事件 |

框错因类别描述的是可观察到的定位、候选覆盖和竞争现象，不能单凭类别自动认定因果机制。诊断会逐项核对正式 CLEAR 结果；不一致时标记失败，避免拿另一种匹配口径冒充正式误差统计。

恢复归因只比较该轨迹最近一次、严格早于恢复帧的 CLEAR GT 归属。当前未定位、竞争中未分配、无历史和没有实际输出均分别列出；它们不会被强行算成“接对”或“接错”。只在前后均可验证的事件上给出同身份接回比例，这个条件比例不代表所有恢复的成功率，也不能证明轨迹历史从未被污染。

完成后导出一个适合提交到当前分支的精简结果目录：

```bash
python -m botsort_research.round2 collect \
  --plan outputs/round2/plan.json \
  --output docs/research/results/round2_2026-11-25
```

目录名称可换成你的实际实验日期；已存在时请用新的名称。精简包包含合并指标、逐序列差值、各组配置与指纹、诊断汇总、代表帧号、控制参数选择、外观校准和覆盖率。没有完成全部计划时，默认拒绝声称完成；若需先反馈 M2，可显式加 `--allow-partial`，输出会记录未完成任务。

保留完整证据包，后续具体错因复查会用到：

```bash
python -m botsort_research.round2 pack \
  --plan outputs/round2/plan.json \
  --output outputs/round2_full_evidence.zip
```

ZIP 旁边会生成文件大小和 SHA-256 清单。完整包包含已运行任务的轨迹、源检测、日志和帧级诊断，不复制原始图像与模型，不包含 `failed/` 中的历史失败尝试。未完成任务仍能从 `state.json` 中识别。ZIP 不自动加入 Git；可以作为附件保留或回传。

精简目录检查后，按平时的流程提交到**同一个实验分支**即可：

```bash
git add docs/research/results/round2_2026-11-25
git commit -m "Document round-two tracking experiments and error attribution"
git push origin research/m2-association-m1-m3-diagnostics
```

建议在生成目录的 README 中补充：运行环境、是否修改参数、失败及重试情况、最有代表性的变好 / 变坏序列和帧号。完整的建议判读顺序见 [RESULT_REVIEW.md](RESULT_REVIEW.md)。

## 7. 结果解释范围

本批继续合并 VT-Tiny-MOT 七类目标，原 official test 及其 9/21 子集已经参与历史选模与第一批分析，均标记为**回顾性开发实验**。选择有效模块后，仍需冻结独立的海上落水目标评估数据，才能支持毕业设计场景的泛化结论。

所有跟踪速度都是缓存重放下的 association/update 时间；检测、RGB 特征提取、图像解码、日志与指纹计算的成本不能由该数字代表。当前阶段优先确认“哪个改变实际解决了哪类错误”，再决定是否组合模块、补提取低分特征或进入检测器改造。
