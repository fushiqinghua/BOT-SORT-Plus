# A v0.2 接入、复跑与实验步骤

## 1. 接入前确认

使用补丁根目录 `install_patch.py --repo ... --check`，通过后 `--apply`。已有 v0.1 文件如果被你修改过，安装器会拒绝覆盖，按升级 diff 人工合并。保留旧结果、旧配置及备份，不将新实验写到旧运行目录。

本版支持上传 baseline 的固定 BoxMOT 16.0.11 源码。v2 除原来的五个核心源码外，额外核验 `botsort_utils.py`，因为事务预检需要与实际原生去重完全一致。若源码不同，应检查差异并适配，不要删除 hash guard。

现有 baseline 环境已经具备 NumPy/SciPy、PyTorch、OpenCV、BoxMOT 时，通常无需另建环境或更新这些依赖。安装器为标准库 Python；业务代码使用现代类型语法，建议沿用已验证的 Python 3.10+ 环境。

## 2. 复用上轮输入

本版直接复用上轮 A/B 已验证的：

- 冻结 YOLO11-M RGB–T 检测缓存及 `rgbt/protocol.json`；
- 原始 640×512 RGB 帧，用于 SOF GMC；
- 微调 RGB ReID 的完整特征缓存（`botsort_plus_features_v1`，每个序列 `.npz + .json`，对齐所有检测行）；
- 官方 RGB GT 和固定 TrackEval，仅在独立评估步骤读取。

若上轮缓存已经符合该格式，不需重新导出，不需重新训练检测器或 ReID。A2 不要求 thermal ReID；新配置均未启用 C。真实高分 RGB 特征缺失、缓存指纹不一致或原图缺失会明确失败，不能用合成图/零向量代替后汇报真实结果。

`--images-root` 应直接包含检测缓存路径引用的 `test2017/`。如果 BoxMOT 由外部运行包提供，继续使用上轮 `--runtime-root`；当前环境能正确导入固定版本时可以省略。

## 3. 先执行软件验证

在应用补丁后的仓库根目录：

```bash
python -m unittest discover -s tests/botsort_plus -v
```

完整评估器集成验证需设置 `VTTINY_TEST_TRACKEVAL_ROOT` 为已冻结 TrackEval 目录。没提供时该项会明确 skip；不应将 skip 描述成已通过。补丁交付记录中的完整运行使用了该目录，结果见 `validation_v0.2/`。

## 4. 输入预检

以下 `/path/to/...` 全部替换为实际路径，命令可写成一行在 Linux 或 PowerShell 执行。

```bash
python tools/run_botsort_plus.py --config configs/botsort_plus/19_A2B.json --images-root /path/to/VT-Tiny-MOT --features-dir /path/to/features_rgb_plus --output-dir outputs/botsort_plus_v2 --preflight
```

预检不运行跟踪，只确认同一套检测、图像、特征和协议；缺失输入返回退出码 2。按报告修正路径。不要为了通过预检改动被冻结的模型哈希或输入指纹。

## 5. 第一轮先验证基座和观察旁路

按相同命令分别选择以下配置，去掉 `--preflight`，加 `--trace`：

| 配置 | 运行名称 |
|---|---|
| `12_baseline_ref.json` | baseline_ref |
| `13_B_ref.json` | B_ref |
| `25_A2_shadow.json` | A2_shadow |
| `26_A2B_shadow.json` | A2B_shadow |

示例：

```bash
python tools/run_botsort_plus.py --config configs/botsort_plus/13_B_ref.json --images-root /path/to/VT-Tiny-MOT --features-dir /path/to/features_rgb_plus --output-dir outputs/botsort_plus_v2 --trace
```

可先用 `--sequences DJI_0030_2 DJI_0129_2 DJI_0327_3` 检查接线，协议会标为子集。正式比较必须让同一批运行使用相同序列集合；全部30段回放时删除 `--sequences`，不要覆盖已有子集同名运行。

然后执行：

```bash
python tools/audit_botsort_plus_a2.py --run-dir outputs/botsort_plus_v2 --trackers baseline_ref B_ref A2_shadow A2B_shadow --compare-pair baseline_ref A2_shadow --compare-pair B_ref A2B_shadow --output outputs/botsort_plus_v2/shadow_audit.json
```

要求两个 compare-pair 的预测文件逐字节一致。该检查同时核验输入协议，不能用不同特征、图像协议或原生配置的运行假装旁路一致。

审计通过退出码 0；结构或对照失败退出码 2，报告保存具体原因。`--output` 必须是新文件。工具不计算性能指标。

`baseline_ref` 全关时直接使用原生更新。B_ref 使用 v2 基座，单独保留原 `05_B.json` 的历史/重放结果。若历史 B 和 B_ref 有差异，先解释基座变化，再比较 A2。不得用 A2 对旧 B 的差值代替同基座消融。

## 6. 运行主方案与消融

主方案：

```bash
python tools/run_botsort_plus.py --config configs/botsort_plus/16_A2.json --images-root /path/to/VT-Tiny-MOT --features-dir /path/to/features_rgb_plus --output-dir outputs/botsort_plus_v2 --trace
python tools/run_botsort_plus.py --config configs/botsort_plus/19_A2B.json --images-root /path/to/VT-Tiny-MOT --features-dir /path/to/features_rgb_plus --output-dir outputs/botsort_plus_v2 --trace
```

最小二因素比较：B_ref、A1B_ref、A2B_E、A2B_I、A2B。配置分别为 13、21、17、18、19。A1B_ref 是 v2 基座上的旧确认规则，其原生出生优先和事务预检与其它 v2 组相同，不等同历史 AB。

延迟与分数对照：22（3/5 分数窗口）、23（5/8 分数窗口）、24（完整 A2B，均分门槛改0.20）。首先固定其它条件；不要把所有开关同时调到指标最好后失去归因。

原生 duplicate 预检是全部 v2 A 组共同步骤，即使 I 关闭也可能拒绝重叠出生，不能把这部分效果归为 I 单独贡献。

## 7. 官方评估

```bash
python tools/evaluate_botsort_plus.py --run-dir outputs/botsort_plus_v2 --trackers baseline_ref B_ref A1B_ref A2B_E A2B_I A2B A2 --gt-root /path/to/official_RGB_test_GT --trackeval-root /path/to/TrackEval
```

按实际已经完成的运行调整 `--trackers`，全部名字对应配置中的 name。GT 根目录直接包含 `<sequence>/gt/gt.txt`。工具复用原 `evaluate_vttiny.py`、官方 GT、七类合并与 COMBINED_SEQ，不用逐序列百分数平均替代。

输出 comparison.csv/json/md、summary.json 和 evaluation_audit.json。优先读取 A2B 相对 B_ref 的 HOTA、IDF1、AssA、FP/FN/IDSW，同时保留 MOTA；只降低 FN 不足以证明收益。

## 8. 导出新增日志统计

```bash
python tools/audit_botsort_plus_a2.py --run-dir outputs/botsort_plus_v2 --trackers baseline_ref B_ref A2 A2B --output outputs/botsort_plus_v2/a2_structure_audit.json
```

报告包括 E 超时、I 决策和拒绝原因、正式弱出生、首次输出、弱来源 B 恢复、每个弱 ID 的输出帧数及确认延迟。弱出生以 birth_committed 为准；weak_birth 是兼容别名，不相加。

这个工具不读取 GT。因此，新增覆盖、错误恢复、FP 寿命仍应使用上轮的官方 GT 对齐诊断流程计算。比较集合元素必须是 `(sequence, GT_ID, frame)`，不跨运行直接比较预测 ID。重点区分 A2B 新增覆盖、丢失 B 原覆盖、弱轨替代 B 原有覆盖，以及 B 原本连续的身份是否被新方法切碎。

当前30个测试序列已参与过选模和诊断，应继续作为开发实验解释；独立泛化需要未参与选择的原始视频或航次。

## 9. 建议回传材料

保留所有配置、protocols、metrics、timing、events、trackers/data、两份审计 JSON 以及实际安装后的代码版本。先回传 comparison.csv、逐序列指标和审计报告即可定位总体变化；进一步错因需要原事件和预测文件。

本版运行协议自动记录实现版本及各 Python 模块 SHA-256，便于核对结果究竟来自哪份补丁。软件验证和真实指标验证分别记录，不把测试夹具的结果填入论文性能表。
