# 运行主版本与复现三轮消融

主版本的运行入口默认选用 `dual_low`。指定 `--method dual_low_center` 可启用高分检测中心修正。运行一个方法会在输出目录写入标准 MOT 轨迹、跟踪协议和逐序列动作统计。方法选择见 [versions.json](versions.json)，网络版本、权重、源码校验值及 GMC 参数见 [reference.json](reference.json)。[完整实验结果](../../results/vttiny/iterations/v0.4-motion-blur/gmc/README.md)

## 从冻结输入生成轨迹

在仓库根目录使用原实验的 Linux `redetr` 环境。跟踪依赖原项目的 BoxMOT 运行时与微调 RGB 特征。设置 `SETTINGS` 指向原 P0 的 `motion_blur_v0.4.local.json`，或者提供一个包含下列四个绝对目录的本地 JSON。

| 字段 | 内容 |
|---|---|
| `baseline_root` | 冻结 RGB–T 基线的目录，内有 `protocol.json` 和 `detections/` |
| `features_dir` | 与冻结检测逐行对齐的 RGB ReID 特征与编码器元数据 |
| `images_root` | 原始 VT-Tiny-MOT 图像根目录，使用 RGB 帧估计 GMC |
| `runtime_root` | 含 `tracking_runtime/` 的跟踪运行时根目录 |

`VENDOR` 指向原实验中含 `dependencies/`、固定提交的 LightGlue 源码和 `torch/hub/checkpoints/` 的目录。若需重新准备，运行下面的第一条命令，随后将 `VENDOR` 设为该脚本旁的 `vendor` 目录。复用已有环境时，沿用其 `PYTHONUSERBASE`；历史部署值为 `$VT_ROOT/integrations/botsort_plus_v0_1_0/python_user`。通过 `CUDA_VISIBLE_DEVICES` 选择可用 GPU。

```bash
python experiments/vttiny_gmc/native_botsort_gmc_oru_v1/setup_lightglue.py
python tools/run_vttiny_gmc.py --settings "$SETTINGS" --vendor "$VENDOR" --output-dir runs/gmc-main
python tools/run_vttiny_gmc.py --settings "$SETTINGS" --vendor "$VENDOR" --method dual_low_center --output-dir runs/gmc-center
```

只复核个案时，在两条跟踪命令后分别加 `--sequences DJI_0030_2` 并使用新的输出目录。省略序列参数会处理冻结协议中的全部序列。预测与官方 GT 评测在独立进程运行。将 `GT_ROOT` 设为含 `<序列>/gt/gt.txt` 的官方 RGB 测试真值目录，`TRACKEVAL_ROOT` 设为原项目固定版本的 TrackEval。以下命令对主版本评分，可替换目录和方法名来评估可选版。

```bash
python tools/evaluate_botsort_plus.py --run-dir runs/gmc-main --trackers dual_low \
  --reference-tracker dual_low --gt-root "$GT_ROOT" --trackeval-root "$TRACKEVAL_ROOT"
```

## 按原顺序复现消融和反事实

完整实验采用原项目的 `integrations/` 布局。为保留既有结果，可在新的工作目录中沿用相同布局并链接现有只读输入。将本目录的三个 `native_botsort_*` 子目录放入该布局的 `integrations/`。三个历史入口通过相对路径读取同级的 `botsort_plus_v0_4_motion_blur/BOT-SORT-Plus-followup-c3e142e6`、`botsort_plus_v0_4_motion_blur/v04_followup_round2/P0/motion_blur_v0.4.local.json`、`botsort_plus_v0_1_0/python_user` 和项目下的 `external/TrackEval`。第一轮还读取 `botsort_plus_C2_v0_3_0/experiments/kf_measurement_round1/full/protocols/kf_original.json` 核对原生轨迹。P0 设置需包含 `gt_root`。历史环境构造函数对使用 LightGlue 的阶段指定物理 GPU 5。原实验基座见[冻结的代码提交](https://github.com/fushiqinghua/BOT-SORT-Plus/tree/c3e142e6c11df0f2a7e8a84bede8556c7dabec1e)。

在上述布局的 `integrations/` 下依次运行。第二轮读取第一轮输出，第三轮读取第二轮的逐帧变换、完整轨迹和检测协议。历史脚本会复用标有完成状态的结果，独立复现应采用空的实验输出目录。

```bash
set -e
python native_botsort_gmc_oru_v1/setup_lightglue.py
for stage in smoke full analyze; do python native_botsort_gmc_oru_v1/run_suite.py "$stage"; done
for stage in smoke full analyze; do python native_botsort_gmc_factorial_v1/run_suite.py "$stage"; done
for stage in inspect trace counterfactual focus full package; do python native_botsort_gmc_0030_fp_v1/run_stage.py "$stage"; done
```

第一轮预检使用 DJI_0173_2、DJI_0277_2，第二轮使用 DJI_0109_1、DJI_0173_2。第三轮对 DJI_0030_2 运行 CLEAR 分配、按检测行对齐和 20 次单帧干预。最后将两个稳定候选应用到全量 30 段。每轮服务器实际使用的脚本快照同相应 `evidence.zip` 一起保存。仓库中的 `live_arm.py` 额外支持显式输入、输出和序列参数，供公开运行入口调用。

## 校验已有结果

```bash
python experiments/vttiny_gmc/verify_archive.py
```

校验仅需 Python 3.11 或更新版本。脚本核验每个公开文件和压缩成员的 SHA-256，并对照原运行协议检查 330 份全序列轨迹、冻结检测／特征／解码图像的指纹、跨轮对照的逐序列指标以及个案数据行数。
