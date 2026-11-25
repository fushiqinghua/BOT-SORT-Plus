# BoT-SORT-Plus · v0.4 运动与模糊诊断

面向无人机海上搜救与微小目标多模态跟踪，在 **YOLO11-M RGB–T + BoT-SORT + 微调 RGB ReID** 基线上进行模块设计、真实数据消融与错因分析。

**当前分支：`diagnostics/v0.4-motion-blur`。当前任务：以已验证有效的 B 为参考，运行四组诊断，确定下一步应优先改进运动预测、相机运动补偿、恢复后续轨，还是模糊相关观测与外观。**

[完整运行说明](docs/botsort_plus/diagnostics_v0.4/README.md) · [实现机制与统计口径](docs/botsort_plus/diagnostics_v0.4/MECHANISM.md) · [软件验证记录](docs/botsort_plus/diagnostics_v0.4/VALIDATION.md) · [v0.4 首轮真实实验](results/vttiny/iterations/v0.4-motion-blur/README.md) · [v0.3 归档](results/vttiny/iterations/v0.3-c2-identity/README.md)

## 当前进展与实验基座

| 阶段 | 已有工作与本轮安排 |
|---|---|
| 冻结基线 | 已归档 VT-Tiny-MOT 上的 RGB/RGB–T 检测、BoT-SORT 与微调 ReID 对照、轨迹和评估结果 |
| A / A2 弱观测起轨 | 已完成真实数据消融与错因分析，历史收益和误报代价保留在[实验归档](results/vttiny/iterations/v0.2-a2/README.md)；本轮关闭 |
| B 短轨迹身份恢复 | 真实数据已验证收益并完成多组消融；本轮固定使用 [B_early_moderate](configs/botsort_plus/B_early_moderate.json) |
| C2、短时外推及位置修正 | 已有消融、错因分析和逐序列证据，见[迭代索引](results/vttiny/iterations/v0.3-c2-identity/README.md)；本轮关闭以便归因 |
| v0.4 四组诊断 | 30 段、12,590 帧的自动诊断已归档，轨迹和原有事件逐字节一致。D4 的真实运动模糊仍待同层人工复核。 [首轮实验](results/vttiny/iterations/v0.4-motion-blur/README.md) |

B 利用原生关联后未消费的候选形成短轨迹，对仍在 Lost 池中的旧轨迹进行时序验证和一对一身份恢复。本轮记录原生 KF/GMC 和 B 的实际过程，保持原检测、原 RGB 特征及跟踪决策。

当前分支从 `iteration/v0.3-c2-identity` 的 `667eb40a6aa624c35fb86eb73656bb334b107a11` 接续。主入口锁定完整有效的 B 配置，关闭 A、旧 C、C2、raw-IR、motion2 和位置修正。历史组合配置的最高分仍可在已有实验中查阅；本轮选择 B 参考是为了分析新增误差来源。

## 四组诊断

| 组别 | 实验内容 | 主要证据 |
|---|---|---|
| **D1 预测与观测误差** | 对照 KF/GMC 更新前预测、同 GT 检测与更新后状态，分析尺度、失联时间和协方差覆盖 | `track_error.csv`、`detection_audit.csv` |
| **D2 GMC 可靠性** | 记录实际补偿矩阵、光流质量代理和共同残差，比较同一状态补偿前后误差 | `frame_error.csv` |
| **D3 B 恢复后稳定性** | 检查恢复端点身份，并分析后续 3、5、10 帧的延续、中断和检测可用性 | `recovery_events.csv`、`recovery_windows.csv`、`recovery_checks.csv` |
| **D4 观测质量与真实模糊** | 分析原生裁块质量、真实相邻帧 ReID 距离及负样本间隔，导出人工模糊复核样本 | `observation_quality.csv`、`quality_strata.csv`、`manual_review.csv`、无损 PNG |

主入口先运行关闭诊断的 `reference` 和开启诊断的 `observed`，核对输入与源码身份，并要求 **MOT 输出和原有跟踪事件逐字节一致**，通过后才开始离线分析。GT 仅用于独立预检与离线分析。

## 快速开始

所有命令从仓库根目录执行。继续使用已经跑通 B 消融的 Python 环境，以及同一份检测、真实 RGB ReID 缓存和原始 RGB 图像。

### 1. 获取当前分支

```bash
git fetch origin refs/heads/diagnostics/v0.4-motion-blur:refs/remotes/origin/diagnostics/v0.4-motion-blur
git switch --track origin/diagnostics/v0.4-motion-blur
```

如果已经有本地同名分支：

```bash
git switch diagnostics/v0.4-motion-blur
git pull --ff-only origin diagnostics/v0.4-motion-blur
```

### 2. 填写本地路径

复制 [configs/diagnostics/motion_blur_v0.4.example.json](configs/diagnostics/motion_blur_v0.4.example.json) 为同目录的 `motion_blur_v0.4.local.json`，填写路径。该本地配置文件已被 Git 忽略。

| 字段 | 内容 |
|---|---|
| `baseline_root` | 含 `protocol.json` 和 `detections/` 的冻结检测目录；默认指向仓库现有 RGB–T 基线 |
| `features_dir` | B 实验使用的真实 RGB ReID 缓存，逐序列 `.npz` 及配套 `.json` |
| `images_root` | 能解析缓存图像路径的原始 RGB 图像根目录 |
| `gt_root` | 官方 RGB GT 根目录，布局为 `<gt_root>/<sequence>/gt/gt.txt` |
| `output_root` | 本次实验的新输出目录，例如 `outputs/diagnostics_v0.4_full30` |
| `sequences` | 全量用 `null`；小规模检查可通过命令行选择序列 |
| `runtime_root` | 普通已安装环境用 `null`；外置运行时沿用旧配置 |

相对路径以仓库根目录为基准。Windows 路径建议写成 `D:/datasets/...`；配置支持 UTF-8/BOM。输入布局、完整示例和环境要求见[详细说明](docs/botsort_plus/diagnostics_v0.4/README.md)。

### 3. 预检、单序列检查、全量实验

预检：

```bash
python -X utf8 tools/run_botsort_diagnostics.py preflight --settings configs/diagnostics/motion_blur_v0.4.local.json
```

先在一个已有序列上走完回放、诊断和打包流程：

```bash
python -X utf8 tools/run_botsort_diagnostics.py run --settings configs/diagnostics/motion_blur_v0.4.local.json --sequences DJI_0030_2 --output-root outputs/diagnostics_v0.4_smoke
```

通过后，保持本地配置 `sequences=null`，运行全量：

```bash
python -X utf8 tools/run_botsort_diagnostics.py run --settings configs/diagnostics/motion_blur_v0.4.local.json
```

成功时 `experiment.json` 的自动阶段为 `COMPLETE`，`verification.json` 为 `PASS`。已有输出目录会被拒绝覆盖。诊断包含两次回放、状态日志和图像扫描，可先用单序列估计本机时间及磁盘需求。

## 读取结果与回传证据

默认结果根目录为 `outputs/diagnostics_v0.4_full30`。

| 相对路径 | 用途 |
|---|---|
| `verification.json` | 检查诊断开关的一致性与日志完整性 |
| `analysis/REPORT.md`、`analysis/summary.json` | D1–D3 的中文汇总、样本数与统计口径 |
| `analysis/*.csv` | 检测、轨迹、GMC 和 B 恢复事件明细 |
| `blur/summary.json`、`blur/quality_strata.csv` | D4 质量与外观统计 |
| `blur/manual_review.csv`、`blur/contact_sheets/*.png` | 真实模糊人工复核表及无损样本图 |
| `reference/`、`observed/` | 两次回放协议、预测、事件及逐帧诊断清单 |
| `logs/` | 回放控制台日志 |

全量成功后自动生成 `outputs/diagnostics_v0.4_full30_evidence.zip`。默认包含报告、CSV、PNG、协议、MOT、事件与日志，省略较大的逐帧诊断压缩流。先回传该证据包即可进行下一步错因判断。

**D4 的真实运动模糊仍需人工确认。** 自动阶段完成时状态为 `PENDING_MANUAL_REVIEW`；按说明填写标签后，可向新目录重做 D4，并重新打包。人工标签导入、失败恢复、单独重分析和 `--include-traces` 用法见[完整运行说明](docs/botsort_plus/diagnostics_v0.4/README.md)。

## 验证状态与结果解释

最终新增 **34 项诊断测试全部通过**；既有回归 **289 项通过、1 项可选 TrackEval 集成跳过**。测试使用明确标注的合成软件数据和真实、源码锁定的 BoxMOT/SOF，覆盖只读等价性、因果身份锚点、窗口删失、缺失特征、输入来源、GT ID=0 和完整 CLI。具体执行批次、跳过原因与源码哈希见[软件验证](docs/botsort_plus/diagnostics_v0.4/VALIDATION.md)及[机器可读记录](docs/botsort_plus/diagnostics_v0.4/validation_report.json)。

诊断配对标签用于错因分析；HOTA、IDF1、MOTA 继续使用原有 TrackEval 口径。GMC 内点和图像清晰度是质量代理，需结合人工复核和具体事件解释；缺 GT、身份歧义与视频尾部截断保留为未知。诊断开启的计时含采集开销，不能作为算法 FPS 结果。

历史官方测试集参与过选模、门限校准和错因分析，因此既有指标和本轮分析按开发实验解读。真实数据诊断结论需等待本地执行，当前软件测试不代表新模块的性能收益。

## 历史实验与复现入口

| 内容 | 入口 |
|---|---|
| B、C2、运动外推、红外与位置修正 | [v0.3 真实数据迭代索引](results/vttiny/iterations/v0.3-c2-identity/README.md)、[模块设计与复算](docs/botsort_plus/FOLLOWUP_DESIGN.md) |
| A / A2 消融及错因 | [v0.1 A/B](results/vttiny/iterations/v0.1-ab/README.md)、[v0.2 A2](results/vttiny/iterations/v0.2-a2/README.md) |
| 冻结检测与跟踪基线 | [RGB / RGB–T 跟踪对照](results/vttiny/yolo11m_rgb_rgbt/comparison.md)、[原始 BoT-SORT 错因](results/vttiny/botsort_error_analysis/BoT-SORT基线错因分析.md) |
| RGB ReID 微调 | [微调前后结果](results/vttiny/rgb_reid_finetune/comparison.md)、[门限校准](results/vttiny/rgb_reid_calibration/comparison.md) |
| C2 实现与历史运行配置 | [模块文档入口](README_BOTSORT_PLUS.md)、[原版设计](docs/botsort_plus/C2_DESIGN_AND_REVIEW.md) |
| 历史基线复算、环境与完整首页 | [v0.3 首页归档](docs/botsort_plus/history/README_v0.3_snapshot.md) |

原始图像、官方 GT、权重和真实外观特征需从已有实验环境提供。冻结检测、轨迹、指标及来源记录可在仓库查阅。数据与上游实现：[VT-Tiny-MOT](https://github.com/xuqingyu26/HGTMT)、[BoxMOT](https://github.com/mikel-brostrom/boxmot)、[TrackEval](https://github.com/JonathonLuiten/TrackEval)、[DeepSORT](https://github.com/nwojke/deep_sort)。第三方代码与补丁沿用各自上游许可。
