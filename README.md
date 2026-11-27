# BoT-SORT-Plus · GMC 与卡尔曼滤波研究

面向无人机海上搜救与微小目标多模态跟踪，在 **YOLO11-M RGB–T + BoT-SORT + 微调 RGB ReID** 基线上进行模块设计、真实数据消融与错因分析。

本研究分支归档 ③ 双向投影匹配、④ 遮挡时的低分观测更新、⑤ 运动相关过程噪声 Q，以及后续的 Q/R 网络和逐帧归因。[研究汇总、全量指标与复现入口](results/vttiny/iterations/motion-345/README.md)

**Round 3 已完成 VT-Tiny-MOT 官方测试集 30 段、12,590 帧的 J0–J3 联合诊断。** 两组新增对照为 `B_skip_only` 和 `B_adaptive_no_dual_low`。E3 已准备 22 例图卡，人工标注待完成。J4 等待复核诊断证据后决定。

[Round 3 真实数据结果、逐序列指标和校验记录](results/vttiny/iterations/v0.4-motion-blur/round3-b-gmc/README.md)

[本批完整执行说明](docs/botsort_plus/diagnostics_v0.4/NEXT_JOINT_ROUND.md) · [统一运行入口](tools/run_v04_joint_batch.py) · [本地配置模板](configs/diagnostics/joint_v0.4_round3.example.json) · [实现代码](experiments/vttiny_gmc/round3_joint/) · [本批软件核验](docs/botsort_plus/diagnostics_v0.4/VALIDATION_ROUND3.md)

历史单 GMC 参考为完整自适应 GMC＋低分匹配双重确认（`dual_low`），保留原版 BoT-SORT KF。已有 B 组合以 `evidence_both` 为重要对照；`pending_guarded` 和 `dual_low_center` 留作历史消融，本批关闭。D1–D4、P0、E1–E3 使用各自的历史实验基线。

[主版本与三轮全量结果](results/vttiny/iterations/v0.4-motion-blur/gmc/README.md) · [运行与复现](experiments/vttiny_gmc/README.md) · [DJI_0030_2 逐帧分析](results/vttiny/iterations/v0.4-motion-blur/gmc/03-dji0030/实验报告.md)

[B 与 GMC 联合消融、两段错因和原 ID 接回](results/vttiny/iterations/v0.4-motion-blur/gmc/04-b-gmc-recovery/README.md)

[完整运行说明](docs/botsort_plus/diagnostics_v0.4/README.md) · [实现机制与统计口径](docs/botsort_plus/diagnostics_v0.4/MECHANISM.md) · [软件验证记录](docs/botsort_plus/diagnostics_v0.4/VALIDATION.md) · [v0.4 首轮真实实验](results/vttiny/iterations/v0.4-motion-blur/README.md) · [v0.3 归档](results/vttiny/iterations/v0.3-c2-identity/README.md)

## 当前批次快速开始

从仓库根目录执行，沿用已经跑通 B/GMC 的环境。下例 `LOCAL/...` 替换为实际历史运行目录，配置中继续填写原始图像、特征、GT 和 TrackEval 路径。优先复用三种 factorial warp 缓存：

```bash
git switch diagnostics/v0.4-motion-blur
git pull --ff-only origin diagnostics/v0.4-motion-blur
python -X utf8 tools/run_v04_joint_batch.py init --from-settings configs/diagnostics/motion_blur_v0.4.local.json --joint-run "LOCAL/b_gmc_joint_r1/full" --factorial-run "LOCAL/native_botsort_gmc_factorial_v1/full" --output-config configs/diagnostics/joint_v0.4_round3.local.json
python -X utf8 tools/run_v04_joint_batch.py run --settings configs/diagnostics/joint_v0.4_round3.local.json
```

初始化时可增加 `--evidence-run "LOCAL/b_gmc_forensics_r1/full_runs/evidence_both"`，把已有修正纳入正式比较。无缓存时使用 `gmc_mode=live` 和本地固定 LightGlue vendor；GPU 由当前环境或 `cuda_visible_devices` 指定，不固定设备编号，不自动下载权重。

`run` 要求全新 `output_root`，默认 `outputs/v0.4_joint_round3`。它复用旧四组输出，新增两组全量对照及一次 BG 只读回放，另做六段的记录关／开试跑。**不自动续跑、不覆盖已有结果。** 逐阶段命令、失败后新目录重试、E3 标签填写和回传清单见[完整执行说明](docs/botsort_plus/diagnostics_v0.4/NEXT_JOINT_ROUND.md)。整批自动阶段结束不代表 E3 人审或 J4 新模块已经完成。

## 当前进展与实验基座

| 阶段 | 已有工作与本轮安排 |
|---|---|
| 冻结基线 | 已归档 VT-Tiny-MOT 上的 RGB/RGB–T 检测、BoT-SORT 与微调 ReID 对照、轨迹和评估结果 |
| A / A2 弱观测起轨 | 已完成真实数据消融与错因分析，历史收益和误报代价保留在[实验归档](results/vttiny/iterations/v0.2-a2/README.md)；本轮关闭 |
| B 短轨迹身份恢复 | 已完成多组消融，早期运动与模糊诊断固定使用 [B_early_moderate](configs/botsort_plus/B_early_moderate.json) |
| C2、短时外推及位置修正 | 已有消融、错因分析和逐序列证据，见[迭代索引](results/vttiny/iterations/v0.3-c2-identity/README.md)；本轮关闭以便归因 |
| v0.4 四组诊断 | 30 段、12,590 帧的自动诊断已归档，轨迹和原有事件逐字节一致。D4 的真实运动模糊仍待同层人工复核。 [首轮实验](results/vttiny/iterations/v0.4-motion-blur/README.md) |
| E1 / E2 / E3 | E1 停止 P1/P2、保留 E1-A/E1-B 诊断资产；E2 固定 191 条补框实验结案；E3 的 22 例正式人审待完成，未宣判整个背景方向无效 |
| B＋自适应 GMC | 联合消融、两段错误链、`evidence_both` 与暂存消融已归档；是本批的实证起点 |
| round 3 J0–J3 | 预检、六段一致性试跑、两组全量对照、官方 TrackEval 和只读路径诊断均已完成。[实验归档](results/vttiny/iterations/v0.4-motion-blur/round3-b-gmc/README.md) |

B 利用原生关联后未消费的候选形成短轨迹，对仍在 Lost 池中的旧轨迹进行时序验证和一对一身份恢复。本批固定 B、检测、RGB 特征与 KF；J2 明确改变相机候选集／`dual_low` 因子，J3 影子记录不改变所属组的正式决策。

v0.4 首轮从 `iteration/v0.3-c2-identity` 的 `667eb40a6aa624c35fb86eb73656bb334b107a11` 接续；本批接续已有结果提交 `7d549bc8bbddfefe99fbb8119ea13d40a4b5bd6d`。A、旧 C、C2、raw-IR、motion2 和位置修正均不纳入本批主线。

## 第二批实验结果（v0.4 round 2）

[P0、E1、E2、E3 结果与复现](results/vttiny/iterations/v0.4-motion-blur/round2-e1-e2-e3/README.md) 已归档。P0 复现冻结 B，E1 的替代中心与 E2 的补框坐标对照均判定 STOP。E3 全量审计与模型辅助核对已完成，22 个正式人工标签待确认，留出代理检验结论为 INCONCLUSIVE。

[执行计划](docs/botsort_plus/diagnostics_v0.4/NEXT_ROUND.md) · [执行前任务清单](configs/diagnostics/v0.4_followup_tasks.json) · [本轮验收状态](results/vttiny/iterations/v0.4-motion-blur/round2-e1-e2-e3/task_status.json)

## 历史：四组诊断（D1–D4）

以下保留首轮复现说明；执行当前 J0–J3 请使用上方新入口，不必重跑全部历史跟踪。

| 组别 | 实验内容 | 主要证据 |
|---|---|---|
| **D1 预测与观测误差** | 对照 KF/GMC 更新前预测、同 GT 检测与更新后状态，分析尺度、失联时间和协方差覆盖 | `track_error.csv`、`detection_audit.csv` |
| **D2 GMC 可靠性** | 记录实际补偿矩阵、光流质量代理和共同残差，比较同一状态补偿前后误差 | `frame_error.csv` |
| **D3 B 恢复后稳定性** | 检查恢复端点身份，并分析后续 3、5、10 帧的延续、中断和检测可用性 | `recovery_events.csv`、`recovery_windows.csv`、`recovery_checks.csv` |
| **D4 观测质量与真实模糊** | 分析原生裁块质量、真实相邻帧 ReID 距离及负样本间隔，导出人工模糊复核样本 | `observation_quality.csv`、`quality_strata.csv`、`manual_review.csv`、无损 PNG |

主入口先运行关闭诊断的 `reference` 和开启诊断的 `observed`，核对输入与源码身份，并要求 **MOT 输出和原有跟踪事件逐字节一致**，通过后才开始离线分析。GT 仅用于独立预检与离线分析。

## 历史 D1–D4 复现命令

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

## 历史 D1–D4 结果与回传证据

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

## 历史 D1–D4 软件验证与结果解释

最终新增 **34 项诊断测试全部通过**；既有回归 **289 项通过、1 项可选 TrackEval 集成跳过**。测试使用明确标注的合成软件数据和真实、源码锁定的 BoxMOT/SOF，覆盖只读等价性、因果身份锚点、窗口删失、缺失特征、输入来源、GT ID=0 和完整 CLI。具体执行批次、跳过原因与源码哈希见[软件验证](docs/botsort_plus/diagnostics_v0.4/VALIDATION.md)及[机器可读记录](docs/botsort_plus/diagnostics_v0.4/validation_report.json)。

诊断配对标签用于错因分析；HOTA、IDF1、MOTA 继续使用原有 TrackEval 口径。GMC 内点和图像清晰度是质量代理，需结合人工复核和具体事件解释；缺 GT、身份歧义与视频尾部截断保留为未知。诊断开启的计时含采集开销，不能作为算法 FPS 结果。

历史官方测试集参与过选模、门限校准和错因分析。首轮真实诊断、D3 后续验证及第二批 E1–E3 已归档，这批诊断使用的历史参考为 `B_early_moderate`。后续 GMC 的对照从原始 BoT-SORT 起算。

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
