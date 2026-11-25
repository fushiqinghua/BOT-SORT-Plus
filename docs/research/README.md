# M2 实现与 M1 / M3 诊断实验手册

> **第二批已就绪：** 继续使用当前实验分支，运行入口、M2 分离对照、M1 稳健补偿、M3 RGB 恢复和结果回传见 [第二批手册](round2/README.md)。本页保留第一批的协议、命令与推进背景。

本分支从冻结基线 `70f28b112ba34014b851152fbeedf007dd7cbb52` 创建。第一轮检验微小目标的概率几何关联、原生相机运动补偿的影响，以及已有 RGB 特征在中断恢复中的身份区分能力。

首轮真实数据实验已完成。30 段原生轨迹通过逐帧基线复现，按固定划分用 9 段校准、21 段对照。正式指标、误报来源、M1 / M3 诊断及复现指纹见 [首轮实验报告](results/round1_2026-11-25/README.md)。

当前缓存属于 VT-Tiny-MOT 七类目标合并评估，沿用冻结基线口径。它用于本轮实现与机制验证；海上落水人员的专门结论还需要明确对应视频、类别和标注口径，不能直接用七类汇总指标替代。

## 第一批交付范围

| 内容 | 当前状态 | 入口 / 说明 |
|---|---|---|
| 原生 BoT-SORT 无 ReID 回放 | 已实现，保留原生实例 | `run --variants native_iou` |
| M2 高斯概率关联、显式拒配、自适应 R | 已实现，可分组运行 | [M2 设计与公式](M2_DESIGN.md) |
| M2 噪声离线校准、序列隔离与指纹检查 | 已实现 | [校准协议](CALIBRATION_PROTOCOL.md) |
| M1 SOF 几何质量诊断 | 已实现，需原始图像 | `diagnose-m1` |
| M1 原生 SOF / 关闭 CMC 对照 | 已实现 | `run --cmc sof / none` |
| M3 候选覆盖、低分连续片段、输出中断诊断 | 已实现，需官方 GT | `diagnose-m3` |
| M3 RGB 同 ID / 不同 ID 特征分布 | 已实现，另需原始特征缓存 | `diagnose-m3 --embeddings-dir ...` |
| M1 新的鲁棒运动补偿 / 不确定性传播 | 后续设计阶段 | 本文的推进条件与 [诊断说明](M1_M3_DIAGNOSTICS.md) |
| M3 RGB–IR 双分支、质量门控、长短期记忆 | 后续设计阶段 | 本文的推进条件与 [诊断说明](M1_M3_DIAGNOSTICS.md) |

所有命令从仓库根目录执行，原始运行产物写到被 Git 忽略的 `outputs/research/`。每次运行使用新的输出路径，程序会检查输出目录。图像、GT、权重和特征缓存在原实验环境中保存。整理后的报告、统计表和复现指纹见 [首轮结果](results/round1_2026-11-25/README.md)。

## 一、准备本机环境

优先激活原基线的 Python 环境。原实验使用 Python 3.11、PyTorch 2.8.0、BoxMOT 16.0.11、OpenCV 5.0.0.93。新命令从冻结检测缓存开始，无需重新运行 YOLOMM，也无需检测或 ReID 权重。M2 第一轮不提取外观特征。

```bash
# 仅轻量计算依赖；已有环境满足时无需重新安装。
python -m pip install -r requirements-research.txt

# 使用外置的冻结 BoxMOT 时，路径须包含 tracking_runtime/boxmot。
# 若当前 Python 已安装匹配的 BoxMOT，则不设置这一变量。
export TRACKING_RUNTIME_ROOT=/path/to/M3OT_MOT_Baseline_v0_1

# 以下两个路径分别用于图像和标注；二者不是同一目录。
export VTTINY_IMAGES=/path/to/VT-Tiny-MOT
export VTTINY_GT=/path/to/GT/GT_00/test
```

图像路径应能组成 `$VTTINY_IMAGES/test2017/DJI_0028_3/00/00000.jpg`。GT 路径应能组成 `$VTTINY_GT/DJI_0028_3/gt/gt.txt`。保留所有官方序列，包括空 GT 的 `DJI_0325_2`。缓存中的框与图像均使用原始 `512×640` 坐标，运行器拒绝缩放后的图像。

如果当前机器没有 BoxMOT，需要先配置单独实验环境或恢复原运行时。不要在原环境直接升级到其他版本。`doctor --check-runtime` 会检查实际导入的核心文件，不依赖单一版本字符串：官方 v16.0.11 的包内 `__version__` 遗留为 16.0.10，因此这是已知情况而非版本检查失效。

评估继续采用归档的 TrackEval：

```bash
# 已有固定版本时直接设置 TRACKEVAL_ROOT。
git clone https://github.com/JonathonLuiten/TrackEval.git external/TrackEval
git -C external/TrackEval checkout 12c8791b303e0a0b50f753af204249e622d0281a
export TRACKEVAL_ROOT="$PWD/external/TrackEval"
```

评估会比对全部归档源码指纹；Linux LF 与原 Windows CRLF 的纯换行差异可通过，其余源码变化会报错。实际文件指纹记录到结果中。

### 1.1 预检查与软件测试

```bash
# 这条只检查仓库自带缓存，不需要数据集和重型运行库。
python -m botsort_research.cli doctor

# 数据和原运行时准备好后执行。
python -m botsort_research.cli doctor \
  --image-root "$VTTINY_IMAGES" --check-runtime \
  --output outputs/research/preflight.json

python -m unittest discover -s tests -v
M2_TEST_BOXMOT=1 python -m unittest discover -s tests -v
```

第一条测试命令可在轻量环境运行；真实 BoxMOT 集成测试默认跳过。第二条启用真实跟踪器和完整 CLI 合成测试；若同时设置了 `TRACKEVAL_ROOT`，还会验证合成数据的 TrackEval 命令链。纯软件测试不会下载数据或模型，也不能替代真实数据实验。Windows PowerShell 可用 `$env:M2_TEST_BOXMOT='1'` 后执行相同 Python 命令。

## 二、先核对原生回放是否复现基线

```bash
python -m botsort_research.cli run \
  --image-root "$VTTINY_IMAGES" \
  --variants native_iou \
  --output outputs/research/b0_native

python -m botsort_research.cli compare-baseline \
  --run-dir outputs/research/b0_native
```

默认使用仓库 `results/vttiny/yolo11m_rgb_rgbt/rgbt/detections/`，并逐帧比较原生无 ReID 回放与归档 `botsort_no_reid` 轨迹。所有 30 段均参与，空检测帧也推进跟踪器。比较使用绝对误差 `1e-6`，不只是比较汇总指标。

**通过后再解释 M2 的收益。** 如果失败，查看 `baseline_comparison.json`，先核对图像、BoxMOT 实际源码、OpenCV、参数和输出差异，不要把复现偏差当成模块改进。原生主基线含 RGB ReID；本轮 M2 的直接因果对照是其冻结的无 ReID 变体。二者不能混为同一对照组。

## 三、固定校准与评估划分

本分支附带 `configs/research/split_retrospective.json`：9 段用于噪声校准、21 段用于本轮比较，按 DJI 录像前缀分组。该文件可直接使用。需要重新制定划分时，可先生成候选，再人工核查同航次、同地点、同对象的关联：

```bash
python -m botsort_research.cli make-split \
  --seed 0 --calibration-fraction 0.3 \
  --output outputs/research/split_candidate.json
```

`DJI_数字` 只是一种录像分组代理，不等于已经证明数据独立。旧 official test 已用于历史检测器 / ReID 选模与门限校准；重新划分不会将它变成独立测试集。当前结论应称为“回顾性开发实验”。后续论文的泛化验证需要另外固定未参与选择的数据。

若继续手工调门限，需在校准集合内部另设调参子集或引入新的开发数据。不要反复看这 21 段的成绩再调整参数，同时仍声称其为独立验证。

## 四、运行 M2 第一轮实验

### 4.1 离线拟合测量噪声

```bash
python -m botsort_research.cli calibrate \
  --split configs/research/split_retrospective.json \
  --gt-root "$VTTINY_GT" \
  --output outputs/research/m2_noise.json
```

该命令只读取校准序列，输出方差模型、样本量、尺度 / 分数分层、输入指纹及退化提示。它不把 GT 输入跟踪器。默认至少 100 个可用残差对；样本不足时应检查数据与选对规则，不要自动借用评估序列。详细采样偏差与统计解释见 [校准协议](CALIBRATION_PROTOCOL.md)。

### 4.2 在同一评估集合回放四组

```bash
python -m botsort_research.cli run \
  --image-root "$VTTINY_IMAGES" \
  --split configs/research/split_retrospective.json \
  --subset evaluation \
  --noise-model outputs/research/m2_noise.json \
  --variants native_iou m2_nll_baseline_r m2_nll_floor m2_nll_calibrated \
  --trace \
  --output outputs/research/m2_eval

python -m botsort_research.cli evaluate \
  --run-dir outputs/research/m2_eval --gt-root "$VTTINY_GT"
```

| 实验组 | 唯一主要变化 | 回答的问题 |
|---|---|---|
| `native_iou` | 原生 IoU 流程 | 同一输入 / 子集上的直接基准 |
| `m2_nll_baseline_r` | 原生 R 下使用 NLL 与拒配 | 概率关联本身是否有效 |
| `m2_nll_floor` | NLL + 人工噪声下限、尺寸与分数项 | 微小目标像素噪声假设是否值得继续 |
| `m2_nll_calibrated` | NLL + 离线拟合的 R | 数据校准能否优于人工初值 |

默认参数位于 `configs/research/m2_default.json`，是本轮回放前冻结的起始值。低分检测用于更新已有活动轨迹。高分 / 低分分界、建轨门限、30 帧有效 buffer、初始化 P0、运动噪声 Q、原生 SOF、未确认轨迹的关联与当前帧输出策略均按冻结基线执行。

`--trace` 对 M2 写出两个关联阶段的匹配、轨迹 ID、检测索引、NLL、马氏距离、logdet 和拒配原因计数。原生实例没有替换内部关联，因此其 trace 文件为空是预期现象。

### 4.3 读结果

先看 `metrics/comparison.md` 的 HOTA、AssA、IDF1、FP、FN、IDSW，再用逐序列 `metrics/summary.json` 和 trace 查看变化。改善某一指标同时显著增加 FP，不能仅挑更好的数字。噪声下限让非重叠小框有机会匹配，也可能扩大错误匹配范围。

所有四组必须在同一 21 段上比较；不能直接从这里的数值减去 README 中 30 段的归档 HOTA。M2 开始有效后，再加“仅放宽原生门限”“中心距离 / NWD”等对照，以区分新代价的效果与宽松门控的效果。新增控制组在独立输出路径和配置中记录，不改写冻结基线。

## 五、运行 M1 诊断

### 5.1 SOF 几何质量

```bash
python -m botsort_research.cli diagnose-m1 \
  --image-root "$VTTINY_IMAGES" \
  --split configs/research/split_retrospective.json --subset calibration \
  --output outputs/research/m1_geometry
```

每段输出 `pairs.csv` 与 `summary.json`，记录特征点、内点比例、空间覆盖、原图像素残差、前后向光流闭合误差以及仿射参数。初始化或失败状态单独标记。先用 `--sequences DJI_0032_3 --max-pairs 100` 检查接口也可以；这只是截取前 100 对连续帧的调试结果，不能当作全序列结论。

### 5.2 原生 CMC 开 / 关对照

第四节中的 `native_iou` 已提供 SOF 开启对照。在完全相同评估集合上关闭 CMC：

```bash
python -m botsort_research.cli run \
  --image-root "$VTTINY_IMAGES" \
  --split configs/research/split_retrospective.json \
  --variants native_iou --cmc none \
  --output outputs/research/m1_cmc_off

python -m botsort_research.cli evaluate \
  --run-dir outputs/research/m1_cmc_off --gt-root "$VTTINY_GT"

python -m botsort_research.cli summarize \
  --runs outputs/research/m2_eval outputs/research/m1_cmc_off \
  --output outputs/research/comparison_round1
```

汇总命令检查缓存、序列、帧数、GT、图像清单和指标来源一致，生成 CSV / JSON / Markdown。关联 FPS 只计 `tracker.update`，不包含检测、ReID、图像解码和文件校验，不能解释为端到端实时速度。

M1 是否继续实现，要综合 CMC 开关对 HOTA / AssA 的影响、异常仿射与坏轨迹是否时间对应，以及图像中是否存在可靠背景。高内点比例和小拟合残差可能来自一致运动的海浪，并不能证明相机运动估计正确。优先考虑网格均衡背景特征、检测区域排除、前后向一致性与估计可信度；可信度传播到运动预测 P 时，不再重复加到检测 R。

## 六、运行 M3 诊断

先在校准集合检查基线主方法的输出中断与检测机会：

```bash
python -m botsort_research.cli diagnose-m3 \
  --split configs/research/split_retrospective.json --subset calibration \
  --gt-root "$VTTINY_GT" \
  --predictions-dir results/vttiny/yolo11m_rgb_rgbt/rgbt/trackers/botsort_reid/data \
  --output outputs/research/m3_baseline
```

主要输出为 `gt_coverage.csv`、`low_score_runs.csv`、`prediction_gaps.csv` 和 `summary.json`。原生框按 `<8 / 8–16 / 16–32 / ≥32` 像素等效边长分组，中断按 0、1–3、4–10、11–30、31+ 帧分组。诊断身份变化不是 TrackEval 的官方 IDSW；最终指标仍由 `evaluate` 计算。

具备原外观缓存时，补充同 / 异 ID 距离分析：

```bash
python -m botsort_research.cli diagnose-m3 \
  --cache-dir /path/to/original/rgbt/detections \
  --embeddings-dir /path/to/original/rgbt/embeddings \
  --split configs/research/split_retrospective.json --subset calibration \
  --gt-root "$VTTINY_GT" \
  --predictions-dir results/vttiny/yolo11m_rgb_rgbt/rgbt/trackers/botsort_reid/data \
  --output outputs/research/m3_rgb_features
```

每段特征需要原 `.npz` 和对应 `.json` 校验信息。归档时图像路径经过改写，仓库 `.json.gz` 的解压字节不一定与原缓存一致；不能绕过 SHA 检查，此时应指定生成特征时的原检测缓存。特征校验不通过会显式跳过该段外观分析，其余 GT / 轨迹诊断继续。

现有特征只覆盖 `score > 0.5` 的检测。低分无特征、零向量与无效向量均单独计数，不把它们当作“外观不相似”。`appearance_pairs.csv` 提供时间与几何候选范围内的采样距离，并按尺寸与间隔汇总。原生小裁剪放大到 256×256 并不增加真实身份信息。

需要比较 M2 的输出中断时，再执行同一诊断，令 `--subset evaluation`，分别将 `--predictions-dir` 指向 `m2_eval/trackers/<variant>/data`，每组使用新的输出目录。这样可检查总体改善是否也体现在微小目标与短缺口上。

M3 后续实现条件：几何候选仍存在且身份混淆可由可靠外观区分，再考虑 RGB / IR 独立编码、模态有效性和裁剪质量门控、短期与受保护长期记忆、高分候选建轨前的恢复。热红外编码器与身份特征缓存尚未提供，本分支不伪造其结果。先量化配准偏差与原生裁剪质量，再做双模态消融；两套未经对齐的特征空间不能直接做交叉余弦相似度。

## 七、实验推进与返回材料

| 阶段 | 可检查的产物 | 继续条件 |
|---|---|---|
| B0 基线回放 | `preflight.json`、`baseline_comparison.json` | 原生回放通过；差异已有定位 |
| M2 初轮 | 噪声拟合 JSON、四组轨迹、指标、trace | 改善可复核，FP / 身份误连代价明确 |
| M1 定位 | 几何质量 CSV、SOF 开关指标 | 证据支持 CMC 是可改进瓶颈 |
| M3 定位 | 覆盖 / 缺口 / 有效特征 / 距离分布 | 对可恢复样本存在可用身份信号 |
| 下一轮设计 | 固定问题、单一改动、对照和接受标准 | 再实现 M1 / M3 正式模块与联合消融 |

完成后优先带回：`baseline_comparison.json`、`m2_noise.json`、各组 `experiment.json` 与 `metrics/summary.json`、M1 / M3 的 `summary.json`，以及几段明显变好或变坏的序列名。无需先上传全部图像或权重。出现错误时保留命令、报错与 `experiment.json` 的 FAILED 状态，不手工将它改成 COMPLETE。

实现细节见 [M2_DESIGN.md](M2_DESIGN.md)、[CALIBRATION_PROTOCOL.md](CALIBRATION_PROTOCOL.md)、[M1_M3_DIAGNOSTICS.md](M1_M3_DIAGNOSTICS.md)。本分支的软件验证范围见 [VALIDATION.md](VALIDATION.md)。
