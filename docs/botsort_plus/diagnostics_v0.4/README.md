# v0.4：运动模型、GMC、B 恢复与模糊诊断实验

这套实验回答一个具体问题：在当前有效的 B 基座上，剩余错误主要来自预测、观测、全局运动补偿，还是身份恢复后的续轨？据此决定下一项模块应改哪里。

本分支基于 `iteration/v0.3-c2-identity` 的 `667eb40a6aa624c35fb86eb73656bb334b107a11`。主入口固定使用现有 `configs/botsort_plus/B_early_moderate.json`，保持原检测、原 RGB ReID 特征与原生 BoT-SORT 参数；A、旧 C、C2、raw-IR、motion2 和位置修正均关闭。本轮不训练模型、不修改跟踪决策，不预设任何候选改进一定有效。

## 1. 四组实验和输出

| 组别 | 要检验的问题 | 主要输出 | 如何支持后续决策 |
|---|---|---|---|
| D1 预测与观测误差 | 有正确检测时，KF/GMC 的预测是否明显偏离？误差与目标尺度、失联时间、协方差是否匹配？ | `analysis/track_error.csv`、`detection_audit.csv` | 区分运动预测误差、检测定位误差、缺少检测及关联问题 |
| D2 GMC 可靠性 | 原生 GMC 的补偿是否改善同一帧同一状态的预测？低质量光流是否伴随共同残差？ | `analysis/frame_error.csv` | 判断是否值得研究可靠性控制、补偿退化处理或不确定性传播 |
| D3 B 恢复后稳定性 | B 是否恢复到原身份？恢复之后 3、5、10 帧中为什么再次中断？ | `analysis/recovery_events.csv`、`recovery_windows.csv`、`recovery_checks.csv` | 区分恢复端点误配、恢复后预测不稳和后续没有可用检测 |
| D4 观测质量与真实模糊 | 定位与 ReID 退化是否关联模糊，而不只是目标更小、分数更低、对比度更差？ | `blur/observation_quality.csv`、`quality_strata.csv`、人工复核表与 PNG | 判断是否有证据支持质量感知观测、模糊鲁棒特征或其他设计 |

**软件验证不等于真实数据诊断结论。** 本仓库新增测试验证代码、数据对齐、计算规则及诊断开关等价性。真实结论需在你的原数据与冻结缓存上执行后产生。[机制、统计口径与限制](MECHANISM.md) 解释如何读这些结果；[软件验证记录](VALIDATION.md) 记录发布前实际检查。

## 2. 获取代码和准备环境

在现有仓库中执行：

```bash
git fetch origin
git switch --track origin/diagnostics/v0.4-motion-blur
```

继续使用之前已经跑通 B 消融的 Python 环境。主入口不会安装、升级或替换 BoxMOT；原有上游源码哈希检查继续生效。诊断还核对原生 SOF 与 BaseCMC 源码，避免在未知实现上记录含义不同的量。已有环境中的 NumPy、SciPy、OpenCV、PyTorch 和 BoxMOT 即可，不需要新的模型或权重下载。

如果已有本地同名分支，使用 `git switch diagnostics/v0.4-motion-blur`。未提交的本地修改请先按平时工作方式保存，再切换分支。

## 3. 只需填写一个配置文件

复制 `configs/diagnostics/motion_blur_v0.4.example.json` 为同目录的 `motion_blur_v0.4.local.json`，填写实际路径。`*.local.json` 已忽略，不会把你的机器路径提交进仓库。

```json
{
  "schema": "botsort_diagnostics_settings_v1",
  "baseline_root": "results/vttiny/yolo11m_rgb_rgbt/rgbt",
  "features_dir": "D:/tracking/features_rgb_plus",
  "images_root": "D:/datasets/VT-Tiny-MOT",
  "gt_root": "D:/datasets/VT-Tiny-MOT_RGB_GT",
  "output_root": "outputs/diagnostics_v0.4_full30",
  "runtime_root": null,
  "sequences": null,
  "max_review": 120
}
```

| 字段 | 要求 |
|---|---|
| `baseline_root` | 冻结的 RGB–T 检测实验根目录，内含 `protocol.json` 和 `detections/`；默认值指向仓库已有基线 |
| `features_dir` | 之前 B 回放使用的真实 RGB ReID 缓存，逐序列 `.npz` 与配套 `.json`；特征行必须与冻结检测完全对齐 |
| `images_root` | 原始 RGB 图像根目录，能解析检测缓存中的 `image` 路径；不能使用可视化画框图或重缩放图 |
| `gt_root` | **官方 RGB GT** 根目录，布局必须为 `<gt_root>/<sequence>/gt/gt.txt`，序列名与帧编号一致 |
| `output_root` | 本次实验的新目录；必须位于输入数据、特征和 GT 目录之外；已存在时拒绝覆盖 |
| `runtime_root` | 平时直接导入已安装 BoxMOT 时填 `null`；沿用旧运行器的外置运行时布局时填其根目录，下面应有 `tracking_runtime/` |
| `sequences` | 全量用 `null`；小规模检查可写序列名数组。也可用命令行 `--sequences` 临时覆盖 |
| `max_review` | 最多生成多少个人工复核候选，默认 120；这是复核样本上限，不是诊断帧数上限 |

JSON 中的相对路径以**本仓库根目录**为基准。Windows 路径建议使用 `/`，例如 `D:/datasets/...`，避免反斜杠转义。配置支持 UTF-8 与 UTF-8 BOM。不要把红外 GT 或另一份重新编号的 GT 当作 RGB GT 使用。

## 4. 建议的执行顺序

### 4.1 先做输入预检

```bash
python -X utf8 tools/run_botsort_diagnostics.py preflight --settings configs/diagnostics/motion_blur_v0.4.local.json
```

预检检查冻结检测、特征元数据及校验和、图像路径、原生运行时兼容性和 GT 文件的数值格式、重复身份与帧范围。GT 检查发生在离线侧；跟踪子进程既不接收 GT 参数，也不读 GT。返回 `READY` 后再回放。

`READY` 表示输入结构可用，不保证每一幅图像都可解码，也不能自动证明 GT 标注语义正确。回放会检查实际解码图像，离线分析会再次检查 GT；不一致时停止并保留失败记录。

### 4.2 先跑一个序列的小规模检查

下面使用仓库已有序列名；也可以替换成你想检查的序列：

```bash
python -X utf8 tools/run_botsort_diagnostics.py run --settings configs/diagnostics/motion_blur_v0.4.local.json --sequences DJI_0030_2 --output-root outputs/diagnostics_v0.4_smoke
```

成功后确认 `experiment.json` 为 `COMPLETE`，`verification.json` 为 `PASS`，再运行全量。小规模结果会保留 `subset=true`，不能作为 30 段全量结果。

### 4.3 执行完整实验

保持配置中 `sequences=null`，并使用尚不存在的输出目录：

```bash
python -X utf8 tools/run_botsort_diagnostics.py run --settings configs/diagnostics/motion_blur_v0.4.local.json
```

程序依次完成：

1. 输入与运行时预检。
2. 独立子进程 `reference`：关闭诊断的 B 回放。
3. 独立子进程 `observed`：开启诊断的同配置回放。
4. 比较检测、特征、源码、参数、实际解码图像指纹；逐序列比较 MOT 输出与原有事件日志；检查诊断文件和清单校验和。
5. 输入和输出一致后，执行 D1、D2、D3 离线分析。
6. 执行 D4 全量质量统计，生成分层复核样本与无损 PNG。
7. 自动生成证据压缩包，默认放在实验根目录旁边。

自动阶段 `COMPLETE` 时，真实模糊归因仍显示 `PENDING_MANUAL_REVIEW`。这表示人工环节待完成；不能把自动清晰度代理当作真实模糊标签。

诊断需要两次跟踪回放及离线扫描，逐帧日志会保存轨迹状态和协方差，空间和时间均高于普通回放。先跑单序列可估计本机资源需求。诊断开启后的计时包含额外 LK、复制、哈希与序列化开销，不能用作算法 FPS 论文结果。

## 5. 结果在哪里、先读什么

默认实验根目录为 `outputs/diagnostics_v0.4_full30`。

| 相对位置 | 用途 |
|---|---|
| `experiment.json`、`preflight.json` | 配置、来源、阶段状态、异常信息 |
| `verification.json` | 同输入、同配置、同输出和原有事件的验证结果 |
| `reference/` | 关闭诊断的参考回放，含 protocol、MOT、事件和计时 |
| `observed/` | 开启诊断的回放，额外包含 `diagnostics/B_early_moderate/*.jsonl.gz` 和配套清单 |
| `analysis/REPORT.md` | D1–D3 的中文入口报告 |
| `analysis/summary.json` | D1–D3 汇总、样本数、限制与来源校验和 |
| `analysis/*.csv` | 检测、轨迹、帧、恢复事件与未来窗口明细 |
| `blur/summary.json` | D4 输入身份、完整性和人工标签状态 |
| `blur/quality_strata.csv` | 按尺度、分数、对比度等分层的统计；每项指标单独给有效样本数 |
| `blur/manual_review.csv`、`blur/contact_sheets/*.png` | 需人工补充的真实模糊证据 |
| `logs/reference.txt`、`logs/observed.txt` | 两次回放的完整控制台记录 |

优先读 `verification.json`、`analysis/REPORT.md`，再结合具体 CSV 定位事件。所有缺失数值保留为空或 `null`，不要在表格工具中批量改成 0。

## 6. 完成人工模糊复核

复制 `blur/manual_review.csv` 为 `blur/manual_review_completed.csv`，结合对应 PNG 和原始相邻帧填写。PNG 给出原像素裁块、最近邻放大及上下文；放大不会恢复原始细节。样本是分层抽取的描述性样本，不能直接估计全数据真实模糊比例。

只修改以下列：

| 列 | 可填值 / 含义 |
|---|---|
| `label` | `clear`、`motion_blur`、`defocus`、`occlusion`、`low_contrast`、`too_small`、`uncertain`；暂未复核可保留 `unreviewed` |
| `occlusion` | `yes`、`no`、`uncertain`，单独记录遮挡因素 |
| `reviewer` | 可选，复核人标识 |
| `note` | 可选，判断依据或仍存疑之处 |

`label` 表示本次复核的主要现象；多种退化共存时可在 `note` 说明，不能确定运动模糊时用 `uncertain`。保留 `run_protocol_sha256`、`sequence`、`frame`、`det_index` 等键不变。不要仅因拉普拉斯方差低就填写 `motion_blur`。

导入已填表，写入新目录：

```bash
python -X utf8 tools/run_botsort_diagnostics.py blur --settings configs/diagnostics/motion_blur_v0.4.local.json --manual-labels outputs/diagnostics_v0.4_full30/blur/manual_review_completed.csv --output outputs/diagnostics_v0.4_full30/blur_reviewed
```

读取 `blur_reviewed/manual_label_summary.json` 和 `manual_label_comparisons.csv`。比较按序列、尺寸、分数、对比度、GT 类别、可见度和人工遮挡层控制；只有相同层内同时存在 `clear` 与 `motion_blur` 才有比较依据。若可比样本不足，状态为 `NEEDS_MORE_COMPARABLE_MANUAL_LABELS`；不要据此宣布“模糊没有影响”。

这一步仍是描述性关联检验。它不会训练新 ReID、去模糊网络或把 GT 注入跟踪器；完整漏检造成的模糊影响也不能仅靠已有检测行判断。

## 7. 重新分析与打包回传

可单独复核回放完整性：

```bash
python -X utf8 tools/run_botsort_diagnostics.py verify --settings configs/diagnostics/motion_blur_v0.4.local.json
```

如果回放已完成、离线分析失败，可保留原结果并指定新分析目录，不必重跑跟踪：

```bash
python -X utf8 tools/run_botsort_diagnostics.py analyze --settings configs/diagnostics/motion_blur_v0.4.local.json --output outputs/diagnostics_v0.4_full30/analysis_retry
python -X utf8 tools/run_botsort_diagnostics.py blur --settings configs/diagnostics/motion_blur_v0.4.local.json --audit-dir outputs/diagnostics_v0.4_full30/analysis_retry --output outputs/diagnostics_v0.4_full30/blur_retry
```

单独重做离线步骤会在新目录产生自己的状态，不会将最初 `experiment.json` 的失败记录改写成成功。若跟踪本身未完成或对照输出不一致，需要排查原因并用新根目录重跑。

若仅打包失败，状态为 `ANALYSIS_COMPLETE_PACKAGING_FAILED`，已完成的分析仍保留。修复磁盘空间或输出路径后，单独运行下面的 `package` 命令即可。

自动包名为 `outputs/diagnostics_v0.4_full30_evidence.zip`。完成人工复核后重新打包到新文件：

```bash
python -X utf8 tools/run_botsort_diagnostics.py package --settings configs/diagnostics/motion_blur_v0.4.local.json --output outputs/diagnostics_v0.4_reviewed_evidence.zip
```

默认包包含报告、CSV、PNG、协议、MOT、事件和日志，省略体积大的逐帧诊断压缩流。需要进一步复核 KF 原始状态时可额外打包：

```bash
python -X utf8 tools/run_botsort_diagnostics.py package --settings configs/diagnostics/motion_blur_v0.4.local.json --include-traces --output outputs/diagnostics_v0.4_with_traces.zip
```

包内 `EVIDENCE_MANIFEST.json` 给出每个文件的 SHA256。打包器只扫描实验输出目录，不遍历原数据、GT、权重或特征输入目录；人工复核 PNG 含少量原图区域。不要把原始数据另行放入诊断输出根目录。

若要打包小规模实验，所有命令追加其对应的 `--output-root outputs/diagnostics_v0.4_smoke`，确保操作的是同一实验。

## 8. 拿到证据后怎么选方向

| 观察到的证据组合 | 优先考虑的设计 | 尚需排除的解释 |
|---|---|---|
| 同 GT、同帧有较准检测，预测误差仍大；随失联时间增长，且 GMC 质量较好 | 运动模型、恢复后的速度重估、状态与过程不确定性建模 | 旧身份锚点误配、标注误差、检测配对选择偏差 |
| 多目标共同偏移与 GMC 代理变差同时出现；同状态补偿后误差反而更大 | GMC 可靠性控制、失效回退或补偿不确定性处理 | 目标共同运动、纹理不足、实际镜头运动本身 |
| B 端点身份正确，后续有检测却丢失；预测偏差同步变大 | 恢复后状态校正或续轨机制 | 原生关联门控、外观歧义、检测竞争 |
| B 恢复后多数断裂发生在无可用检测的帧 | 检测观测、短时缺测处理或检测与跟踪接口 | 已有检测只是低于阈值，或 GT 可见性不确定 |
| 人工确认运动模糊后，在可比层内仍有定位或外观间隔退化 | 质量感知观测更新、模糊鲁棒外观或记忆处理 | 遮挡、上一帧外观质量、样本量与标注主观性 |
| 问题仅集中于极少数片段，或分层后现象消失 | 暂缓该方向，检查其他占比更高的错因 | 小样本与片段相关性 |

不预设固定“达标改善百分比”，也不把相关性直接当作因果。后续若实施新机制，应冻结诊断得到的假设，再做单模块与 B 组合消融，并按原始视频分组控制开发和验证划分。
