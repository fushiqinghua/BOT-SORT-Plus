# 复现 B 与自适应 GMC 的联合消融和长间隔重连

本目录记录实验实际使用的脚本。[指标汇总、逐帧错因和接回明细](../../../results/vttiny/iterations/v0.4-motion-blur/gmc/04-b-gmc-recovery/README.md)

## 输入与目录

沿用 [前三轮 GMC 的环境与运行说明](../README.md) 中的固定 BoxMOT 运行时、LightGlue、TrackEval、RGB 帧、YOLO11-M RGB–T 已缓存检测和 RGB ReID 特征。评测需要官方 RGB 测试集真值。`motion_blur_v0.4.local.json` 需要包含 `baseline_root`、`features_dir`、`runtime_root`、`images_root`、`gt_root`。跟踪子进程按原实验选择物理 GPU 5。复现布局中的固定目录名如下，输入可链接到已核验的本地副本。

| 相对项目根目录 | 内容 |
|---|---|
| `external/TrackEval` | 固定提交 `12c8791b303e0a0b50f753af204249e622d0281a` |
| `integrations/botsort_plus_v0_4_motion_blur/BOT-SORT-Plus-followup-c3e142e6` | 固定代码 `c3e142e6c11df0f2a7e8a84bede8556c7dabec1e`，提供历史评测入口 |
| `integrations/botsort_plus_v0_4_motion_blur/v04_followup_round2/P0/motion_blur_v0.4.local.json` | 本地冻结输入设置 |
| `integrations/botsort_plus_v0_1_0/python_user` | 原实验 BoxMOT 等 Python 依赖 |
| `integrations/native_botsort_gmc_oru_v1/vendor` | 前三轮共用的 LightGlue 代码和权重 |
| `integrations/native_botsort_gmc_factorial_v1` | 前轮 `full/` 评测控制组及 `full/warps/` 冻结相机变换 |
| `integrations/native_botsort_gmc_0030_fp_v1` | 前轮单 GMC 结果及 `clear_audit.py`、`common_audit.py`、`guard.py`、`replay_case.py` 等原始脚本 |

## 按顺序回放

令 `ARCHIVE` 指向当前仓库中的 `experiments/vttiny_gmc/b_gmc_recovery`，从全新或隔离的 `integrations/` 目录执行。读取既有共享输入，实验输出位于本轮的三个目录。`git archive` 用前一轮提交生成联合实验的代码基座，随后放入联合实验对应的精确快照。

```bash
set -e
mkdir -p b_gmc_joint_r1/repo b_gmc_forensics_r1 b_gmc_pending_id_r1
git -C "$ARCHIVE/../../.." archive 0a7121ebb2e1dc6b37c1a53b6aaa423169d7a3e9 \
  | tar -xf - -C b_gmc_joint_r1/repo
cp "$ARCHIVE/joint/run.py" b_gmc_joint_r1/
cp "$ARCHIVE/joint/adapter_v2.py" b_gmc_joint_r1/repo/botsort_plus/
cp "$ARCHIVE/joint/live_arm.py" b_gmc_joint_r1/repo/experiments/vttiny_gmc/native_botsort_gmc_0030_fp_v1/
cp "$ARCHIVE/joint/B_reference.json" b_gmc_joint_r1/repo/
python -Bu b_gmc_joint_r1/run.py

cp "$ARCHIVE"/forensics/*.py "$ARCHIVE"/forensics/plan_*.json b_gmc_forensics_r1/
for phase in initial cause path trial evidence live; do
  python -Bu b_gmc_forensics_r1/driver.py "$phase"
done
python -Bu b_gmc_forensics_r1/local_tables.py

cp "$ARCHIVE"/pending/*.py "$ARCHIVE"/pending/plan_*.json b_gmc_pending_id_r1/
for phase in focus focus2 full; do
  python -Bu b_gmc_pending_id_r1/driver.py "$phase"
done
(cd b_gmc_pending_id_r1 && python -Bu -c "from driver import ROOT, run; run([ROOT/'release_context.py'])")
python -Bu b_gmc_pending_id_r1/verify_tables.py
```

联合跟踪脚本回放单 B 和组合，并复用已经校验的原生和单 GMC 轨迹。冻结变换的单帧干预定位第 409、174、108 帧的路径，再用实图运行修正方案。原 ID 暂存实验从 `b_gmc_forensics_r1/live_repo` 和 `b_gmc_forensics_r1/full_runs/evidence_both` 继承实现及基准。科研实验通过隔离副本进入跟踪关联，原 ID 暂存只在选定消融中开启。

`panels.py` 可使用已授权的本地图像生成诊断图与逐真值分配。`case_timeline.py` 使用 `--gt`、`--detections` 和 `--previous-flags` 指定官方单序列真值、冻结检测 JSON.GZ 以及上轮的 `evidence_both_affected.csv`。随后运行 `plot_focus.py` 可生成低分观测图。报告中的得分、身份与事件表可以直接读取。

## 校验归档

```bash
python experiments/vttiny_gmc/b_gmc_recovery/verify_archive.py
```

校验已归档结果文件的 SHA-256，验证控制组的轨迹哈希、全量输入指纹、关键源码字节和接回计数。公开的运行协议使用 `${VT_ROOT}` 代替原实验根目录，保留原有校验值。完整回放需要上表对应的数据和环境。
