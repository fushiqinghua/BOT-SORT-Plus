# v0.4 round 3：B 与自适应 GMC 联合诊断执行说明

本批接续 `7d549bc8bbddfefe99fbb8119ea13d40a4b5bd6d` 的真实实验。目标是区分收益重叠、相机变换影响和关联保护影响，再依据证据决定下一项协同修改。**本次提交提供可运行实验，不包含新的真实数据集性能结论；J4 新策略等待本批诊断结果。**

[本批软件核验](VALIDATION_ROUND3.md) · [机器可读核验记录](validation_round3.json)

## 1. 本批范围与旧任务状态

| 任务 | 本批执行内容 |
|---|---|
| J0 | 核验输入、历史轨迹、原视频分组、固定评测源码和运行环境 |
| J1 | 复用 native／B／G／BG 四组，计算 GT 帧覆盖交集、身份事件及后 3/5/10 帧结果 |
| J2-S | 新跑 `B_skip_only`：原生／恒等变换，关闭 `dual_low` |
| J2-F | 新跑 `B_adaptive_no_dual_low`：原生／恒等／LightGlue，关闭 `dual_low`；同时采集 J3 |
| J3-BG | 对已有 `B_early_moderate_dual_low` 补一次只读回放，并核对历史 MOT |
| E3 | 准备并核验现有 22 个人工样例，输出样例级人机对照 |
| J4 | `AWAITING_DIAGNOSTIC_DECISION`；不提前加入新门控、不搜索阈值 |

G 指已有自适应 GMC＋`dual_low`，不能直接解释为纯变换。B、原版 KF、检测、ReID 及已有门限保持冻结；A、C2、中心输出修正和 `pending_guarded` 不进入本批主线。

E1-A/B 已完成，保留队列和失败阶段分析；停止 E1-C 的 P1/P2。E2 固定 191 条补框坐标实验结案。E3 尚未完成正式人审，保留为短支线。[旧任务结果](../../../results/vttiny/iterations/v0.4-motion-blur/round2-e1-e2-e3/README.md) · [B–GMC 组合结果](../../../results/vttiny/iterations/v0.4-motion-blur/gmc/04-b-gmc-recovery/README.md)

## 2. 拉取与准备本地配置

所有命令从仓库根目录执行，沿用已经跑通 B/GMC 的 Python 环境。命令采用单行，适用于 Bash 或 PowerShell。先拉取同一分支：

```bash
git switch diagnostics/v0.4-motion-blur
git pull --ff-only origin diagnostics/v0.4-motion-blur
```

优先使用既有 factorial warp 缓存。将下例 `LOCAL/...` 换成真实路径：

```bash
python -X utf8 tools/run_v04_joint_batch.py init --from-settings configs/diagnostics/motion_blur_v0.4.local.json --joint-run "LOCAL/b_gmc_joint_r1/full" --factorial-run "LOCAL/native_botsort_gmc_factorial_v1/full" --evidence-run "LOCAL/b_gmc_forensics_r1/full_runs/evidence_both" --output-config configs/diagnostics/joint_v0.4_round3.local.json
```

`--evidence-run` 可省略，但建议配置，以便正式评测同时比较 `evidence_both`。若历史四组不在同一目录，初始化后分别修改 `references.native/B/G/BG.run_dir`。

| 本地输入 | 所需内容 |
|---|---|
| `baseline_root` | 冻结 `protocol.json` 与 `detections/` |
| `features_dir`、`images_root` | 原有 RGB ReID `.npz/.json` 缓存及真实 RGB 图像；cached 模式仍核验实际图像像素 |
| `references.*.run_dir` | `protocols/<tracker>.json`、`trackers/<tracker>/data/<sequence>.txt`；事件及已有 metrics 一并保留 |
| `warp_sources` | factorial 的 native、skip_only、adaptive_gmc 三种缓存，各自完整协议与 `warps/<method>/<sequence>.jsonl.gz` |
| `gt_root`、`trackeval_root` | 官方 RGB GT 与固定 TrackEval 源码；只用于独立预检和离线评估 |
| `runtime_root`、`python_user_base` | 外置环境沿用旧值，普通已安装环境可为 `null` |
| `output_root` | 新目录，默认 `outputs/v0.4_joint_round3` |

相对路径以仓库根目录为基准；Windows 可写 `D:/datasets/...`。`sequences=null` 代表全量。`init` 不覆盖已有配置，不下载数据或权重。

### 实图估计模式

没有完整 warp 缓存时，将 `gmc_mode` 改为 `live`，填写 `vendor_dir`；也可初始化时省略 `--factorial-run`，增加 `--vendor "LOCAL/native_botsort_gmc_oru_v1/vendor"`。LightGlue 需要本地固定源码、权重及可用 GPU，缺失时直接报错，不联网补装、不退化为 skip-only。`cuda_visible_devices` 默认继承当前环境，可自行填写 `"0"` 等设备，不再固定 GPU 5。

缓存模式会核验候选集、GMC policy、源码／模型来源、帧顺序和图像摘要。不能把完整自适应方案最终选中的矩阵当作 skip-only 缓存。模型与输入要求见[已有 GMC 运行说明](../../../experiments/vttiny_gmc/README.md)。

## 3. 执行方式 A：新目录中运行整批

检查配置后执行：

```bash
python -X utf8 tools/run_v04_joint_batch.py run --settings configs/diagnostics/joint_v0.4_round3.local.json
```

该命令依次执行 J0、J1、六段试跑、J2、正式评测、J3-BG、两组 J3 汇总和 E3 人审表准备，最后打包。自动阶段完成时为 `COMPLETE_EXECUTABLE_STAGES`，E3 人审和 J4 决策仍待完成。

**整批入口不自动续跑，不覆盖已存在目录。** 若已经按下一节完成部分阶段，不要在同一目录再运行 `run`。重新整批运行应选择新的 `output_root`，或传入新的 `--output-dir`。

计算预算：全量部分是 **2 个新对照＋1 次 BG 只读回放**；J3-F 已并入 J2-F。另有六段试跑，包含 B 一次及 BG、J2-S、J2-F 各自记录关／开两次，共 7 次组别回放。历史四组全量轨迹直接复用。缓存回放和诊断计时都不能当作算法端到端 FPS。

## 4. 执行方式 B：逐阶段执行与检查

### J0、J1 与试跑

```bash
python -X utf8 tools/run_v04_joint_batch.py preflight --settings configs/diagnostics/joint_v0.4_round3.local.json
python -X utf8 tools/run_v04_joint_batch.py analyze --settings configs/diagnostics/joint_v0.4_round3.local.json
python -X utf8 tools/run_v04_joint_batch.py smoke --settings configs/diagnostics/joint_v0.4_round3.local.json
```

试跑默认覆盖 `DJI_0030_2`、`DJI_0169_3`、`DJI_0109_1`、`DJI_0173_2`、`DJI_0221_2`、`DJI_0028_3`。`smoke/verification.json` 必须为 PASS：历史 B/BG MOT 一致，三组记录关／开的 MOT 和原有事件逐字节一致。全量任务不会接受缺少必要试跑序列的验证记录。

J1 的 `missing_events` 会明确保留缺失阶段或检测索引；不会通过分数相等、最近框或跨运行数字 ID 猜配对。GT 帧覆盖交集不能当作 HOTA/IDF1 的可加分解。

### J2：两组全量及正式评测

```bash
python -X utf8 tools/run_v04_joint_batch.py run-controls --settings configs/diagnostics/joint_v0.4_round3.local.json
python -X utf8 tools/run_v04_joint_batch.py evaluate --settings configs/diagnostics/joint_v0.4_round3.local.json
```

`j2/manifest.json` 的 `COMPLETE_TRACKING_PENDING_EVALUATION` 表示跟踪完成，评测尚需下一条命令。评测会复制原始 MOT 建立明确的同范围评测视图，不平均逐序列 HOTA 代替 `COMBINED_SEQ`。

### J3：BG 回放及两组离线解释

```bash
python -X utf8 tools/run_v04_joint_batch.py probe --settings configs/diagnostics/joint_v0.4_round3.local.json --method B_early_moderate_dual_low
python -X utf8 tools/run_v04_joint_batch.py probe-summary --settings configs/diagnostics/joint_v0.4_round3.local.json --method B_adaptive_no_dual_low --run-dir outputs/v0.4_joint_round3/j2 --output-dir outputs/v0.4_joint_round3/j3_f_analysis
python -X utf8 tools/run_v04_joint_batch.py probe-summary --settings configs/diagnostics/joint_v0.4_round3.local.json --method B_early_moderate_dual_low
```

若修改了 `output_root`，同步替换命令中的显式路径。P 是共同实际锚点之后的预测历史对照；N 比较同一帧前状态下的原生关联与残差；C 比较相同残余输入下的候选链接。它们是条件诊断，变化数不能相加为新算法收益。活跃身份支持使用冻结记忆，仅在历史与外观覆盖可比时报告强弱；缺失保持未知。正式 tracker 独占 ID、KF、记忆和 reservation，影子分支不写回。

重点检查 `native_high_reconnections.csv` 的实际 Lost 高分重连，以及 `guard_to_B.csv` 中被 `dual_low` 拒绝的检测是否又成为 B 候选历史。`guard_membership_summary.csv` 明确日志覆盖范围，`output_indices.csv` 保存输出与原始检测行的精确对应。后续成员关系不等于因果证明，也不能把后面代价下降解释为新增身份支持。表内 GT、CLEAR 结果和未来窗口仅由独立离线分析加入，不返回预测进程。

## 5. E3 人工复核收尾

整批 `run` 已准备此目录；逐阶段执行时运行一次：

```bash
python -X utf8 tools/run_v04_joint_batch.py e3-prepare --output-dir outputs/v0.4_joint_round3/e3_review
```

工具复制 22 例可编辑表并生成 `review_index.csv`。它引用已有原图证据，不下载或重新生成图卡；查看 `missing_cards`，缺图时先恢复本地历史 E3 证据包。独立查看原图、上下文和邻帧后，只填写 `human_label`、`human_reviewer`、`human_reason`。标签为 confirmed_background／possible_unlabelled_object／true_object／uncertain；未审阅保留 pending。

```bash
python -X utf8 tools/run_v04_joint_batch.py e3-check --input outputs/v0.4_joint_round3/e3_review/E3_manual_review.csv
python -X utf8 tools/run_v04_joint_batch.py e3-report --input outputs/v0.4_joint_round3/e3_review/E3_manual_review.csv --output-dir outputs/v0.4_joint_round3/e3_report
```

退出码 3 表示尚有 pending，2 表示输入／表格错误，0 表示本命令完成。uncertain 属于完成复核，不能并入背景负例。可选 `--causal-signals` 仅联结原始 CSV/CSV.GZ 中精确的样例中心帧，不向整条轨迹传播标签。

旧 E3-C 脚本不读取人审标签，`manual_label_count` 固定为 0。本工具独立读取人审并形成样例级对照；不把旧代理 AUC 改称人审验证，不训练过滤器，不外推总体背景比例。

## 6. 失败重试、查看状态与回传

单阶段失败保留原目录和日志，用新的 `--output-dir` 重跑该阶段。例如 J2 失败后：

```bash
python -X utf8 tools/run_v04_joint_batch.py run-controls --settings configs/diagnostics/joint_v0.4_round3.local.json --output-dir outputs/v0.4_joint_round3/j2_retry1
python -X utf8 tools/run_v04_joint_batch.py evaluate --settings configs/diagnostics/joint_v0.4_round3.local.json --run-dir outputs/v0.4_joint_round3/j2_retry1 --output-dir outputs/v0.4_joint_round3/j2_retry1_evaluation
```

J3-F 汇总也应指向重试目录。若试跑使用了自定义输出路径，在配置的 `smoke_manifest` 填入该目录的 `verification.json`；源码、预测设置或历史参考发生变化后，需要重新试跑。不得将 FAILED 改写为 PASS。分阶段运行可通过下列命令查看已有清单并打包：

```bash
python -X utf8 tools/run_v04_joint_batch.py status --settings configs/diagnostics/joint_v0.4_round3.local.json
python -X utf8 tools/run_v04_joint_batch.py package --settings configs/diagnostics/joint_v0.4_round3.local.json
```

默认压缩包为 `<output_root>_evidence.zip`，包含报告、逐行表、协议、MOT、事件和日志；省略 `probes/`、`warps/` 大体积流。需要原始流时增加 `--include-traces`。已有同名压缩包不会覆盖；完成 E3 人审后可指定新的文件名重打包：

```bash
python -X utf8 tools/run_v04_joint_batch.py package --settings configs/diagnostics/joint_v0.4_round3.local.json --archive outputs/v0.4_joint_round3_after_review.zip
```

| 重点产物 | 回传后回答的问题 |
|---|---|
| `j1/report.md`、`coverage_overlap.csv`、事件表 | 修复重叠、组合新增／丢失收益、身份后续情况 |
| `smoke/verification.json` | 记录开关与历史轨迹是否一致 |
| `j2_evaluation/metrics/` | 两个关键对照相对 B、BG、可选 evidence 的正式指标 |
| `j3_f_analysis/`、`j3_bg_analysis/` | P/N/C 变化、相机分歧、活跃身份竞争及原视频分组 |
| `e3_report/` | 人工标签完整性、逐样例对照和有限范围结论 |
| `batch_status.json` 或各阶段 manifest | 已完成、失败、待人审和待决策的真实状态 |

现有 30 段仍属于开发性评估。J4 的进入条件是：错误机制有可重复证据、正确恢复能够保留、主要设计视频之外也有支持、部署信号只使用当帧及过去。信号不可分时保留本批诊断结果，停止增加规则。
