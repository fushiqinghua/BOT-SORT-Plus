# TCO T0/T1 复现

本目录复现两组冻结参考上的只读采样、逐字节一致性、官方 CLEAR 离线标记、分层抽样和本地图卡。历史结果与结论见[实验归档](../../../results/vttiny/iterations/tco-v0.1/README.md)。本阶段使用官方 RGB 测试 30 段作为开发诊断集，未训练或校准 TCO。

## 本地资产

| 参数或配置 | 所需内容 |
| --- | --- |
| `--repo` | 独立检出执行提交 `53faeccba267d8988174c18e1a538f759fa6dd07`。采样器还核对 Round 3 worker 的 Git blob。 |
| `--round3-job` | 历史 Round 3 批次根目录，包含 `settings.json`、`outputs/j0/manifest.json`、`outputs/j2_evaluation/metrics/summary.json`。保留原 J0 `PASS` 及冻结输入指纹。 |
| Round 3 `settings.json` | 使用既有 `v04_joint_batch_v1` 格式。`baseline_root` 含相同检测缓存与 `protocol.json`，`features_dir` 含对齐的逐序列 RGB ReID `.npz`，`images_root` 含原 RGB 图像。沿用 `runtime_root`、`python_user_base` 等已验证环境设定。 |
| GMC 与分组 | 设置 `gmc_mode` 为 `cached`，沿用 `warp_sources.native`、`skip_only`、`adaptive_gmc` 的原协议与逐序列 warp 缓存。不现场拟合或用其他 GMC 策略替代。分组使用历史 E3 `E3_sequence_summary.csv` 的 `sequence` 和 `original_video` 列。 |
| `--tco-config` | 按[分支手册](../../../docs/botsort_plus/tco_v0.1/LOCAL_RUNBOOK.md)在固定提交内创建本地配置，`paths.*` 和两组 `references.*.run_dir` 填写绝对路径。`protocol_file` 保留指向固定提交的 TCO 协议。 |
| 参考产物 | `references.B` 指向 `B_early_moderate`，`references.BG` 指向 `evidence_both`。每组保留原始 `protocols/<method>.json` 以及 30 个 `trackers/<method>/data/<sequence>.txt`。协议原文件与轨迹摘要须吻合 J0 记录。 |
| GT 与历史划分 | `paths.gt_root` 指向官方 RGB `GT_00/test`（各序列下含 `gt/gt.txt`），`paths.trackeval_root` 指向提交 `12c8791b303e0a0b50f753af204249e622d0281a` 的 TrackEval。`paths.split_manifest` 含历史 `splits.{train,val,test}`，RGB 数据根目录须有 `annotations/instances_00_train2017.json` 与 `instances_00_test2017.json`。训练图像目录按官方序列登记，仅用于资产核对。 |

沿用冻结跟踪环境中的 Python 3.11、BoxMOT 16.0.11、PyTorch、OpenCV、NumPy、SciPy 及历史原生适配器。图卡使用 Pillow 12.3.0，已执行渲染环境用 OpenCV 5.0.0。预测 subprocess 使用 CPU 和冻结检测、RGB ReID 与相机变换缓存。图卡计算会核对当前帧与每个历史观测帧的解码像素摘要。历史 J0/J2 的本地文件、原图、缓存和权重需在执行环境中已有。

## 运行 T0/T1

在当前发布分支仓库根目录运行下列 Bash 命令。把 `/local/...` 改为本地资产路径，并在同一个 `WORK` 目录按顺序完成四个阶段。示例把产物放在仓库之外。

```bash
REPRO="$PWD/tools/research/tco_t01"
PINNED=/local/TCO-base-53faeccb
ROUND3=/local/round3-archive
TCO_CONFIG=/local/tco_v0.1.local.json
WORK=/local/tco_t01_reproduction

git worktree add --detach "$PINNED" 53faeccba267d8988174c18e1a538f759fa6dd07
python -X utf8 "$PINNED/tools/manage_tco_batch.py" init --output-config "$TCO_CONFIG"
```

填入上表的本地配置后继续。已有同名配置可直接填写和使用。

```bash
python -X utf8 "$PINNED/tools/manage_tco_batch.py" check --strict --settings "$TCO_CONFIG"
python -X utf8 "$PINNED/tools/manage_tco_batch.py" prepare --settings "$TCO_CONFIG"

for stage in inventory smoke full labels; do
  python -X utf8 "$REPRO/workflow.py" "$stage" \
    --repo "$PINNED" --round3-job "$ROUND3" \
    --tco-config "$TCO_CONFIG" --output-root "$WORK" || exit 1
done
```

- `inventory` 核对官方 30 段、12,590 帧、两组历史轨迹指纹、官方 GT、TrackEval 与开发分组。生成 `T0/input_manifest.json`、`T0/reference_manifest.json` 和 `T1/split_manifest.json`。
- `smoke` 在预注册的六段上运行两组参考的采样开关对照，核对 MOT、原事件和逐帧内部状态摘要。`full` 检查烟测通过、采样器源码与预测配置未变，再运行 30 段并生成 `T0/T0_parity.json`。原协议的引用文件可以保留历史绝对路径字段，资产位置在本地配置中指定。摘要按历史内容核验。
- `labels` 读取全量通过的采样输出及官方 GT，生成离线标记、可行框、可达性统计、种子为 41 的原抽样和 `T1/offline_verification.json`。与历史 J2 每段 TP、FP、FN、IDSW 共 60 条结果核对。实时采样进程只接收 `prediction_settings.json` 的白名单字段；GT 路径进入独立离线进程。

单次运行有四个并行的 CPU 回放进程，每个进程限制线程数。完整捕获保存在 `full/on/{B,BG}/tco_capture/`，观察器关闭组不写捕获。源数据与输出目录迁移可能改变路径相关的配置摘要，gzip 头时间戳也可能影响压缩流摘要。一致性门限比较同一批次的开关输出，并将 MOT 与冻结参考逐字节对照。已中断的目录保留日志，重跑时指定新的 `WORK`。

## 本地图卡与复核表

```bash
CARDS=/local/tco_t01_review_cards
python -X utf8 "$REPRO/build_review.py" "$WORK" "$CARDS" --focus-per-sequence 12
python -X utf8 "$REPRO/finalize_review.py" --results-dir "$CARDS/bundle" --skip-archive
```

`build_review.py` 读取离线标记和真实当前、历史 RGB，生成原抽样 160 例与两个重点序列各 12 例补样，并验证两类样本的轨迹、检测行和 GT 锚点。`finalize_review.py` 构建 `bundle/index.html`，默认打开盲审图，模型结论需要主动展开。模型辅助表和人工复核表分别保存，人工初始状态为 `PENDING_HUMAN_REVIEW`。

要复用此次发布的既有模型判断，把[模型辅助复核表](../../../results/vttiny/iterations/tco-v0.1/review/model_assisted_review.csv)复制到刚生成的 `bundle/`，再运行一次 `finalize_review.py`。脚本按案例 ID 和状态读入既有条目。若另有模型盲审工作表，可用 `--model-dir` 指向含 `case_id,model_identity,model_localization,model_note` 的 CSV 目录。复现图卡和汇总使用记录中的判断，脚本自身不调用模型复核或代填人审。人工复核可直接填写 `bundle/human_review.csv` 后重新生成图册。图卡和含图片的压缩包保留在本地。

`SOURCE_PROVENANCE.json` 记录发布源码摘要及其与历史运行快照的对应关系。发布版 `workflow.py` 只将安装位置和本地输入改为命令行参数，并校验缓存模式及路径。历史 T0 运行清单在 T1 标记脚本定稿前生成，因此该清单中的旧标记代码摘要与最终归档源文件分别保留。
