# C2 接入、回放与真实实验

## 1. 接入来源

本版基于 `iteration/v0.2-a2` 的 `6d9c258556a143036cebde891c93079a772b9945`。代码随独立开发分支提交，保留已有基线、A/B/A2 源码与实验归档。切换分支前先妥善保存本地未提交修改；本版无需重新训练检测器或 RGB ReID。

在已有仓库中首次取得新分支：

```bash
git fetch origin
git switch --track origin/iteration/v0.3-c2-identity
```

若本地已经有该分支，直接 `git switch iteration/v0.3-c2-identity`。新模块已接入跟踪主流程，无需另行复制补丁文件。

完整机制见 [C2_DESIGN_AND_REVIEW.md](C2_DESIGN_AND_REVIEW.md)，配置矩阵见 [C2_CONFIGS.md](C2_CONFIGS.md)。首次接入先运行 B_ref 与 C2 shadow，再开始改变预测的消融组。

## 2. 沿用已验证环境和输入

继续使用上轮通过校验的 BoxMOT 16.0.11 源码、SOF GMC、冻结 YOLO11-M RGB–T 检测缓存、RGB ReID 全量特征缓存及原始 RGB 图像。代码保留源码哈希核验；不要通过删掉哈希检查来适配不同 BoxMOT。

- `--baseline-root` 指向包含 `protocol.json` 和 `detections/` 的冻结 RGB–T 检测结果目录；仓库默认是 `results/vttiny/yolo11m_rgb_rgbt/rgbt`。
- `--features-dir` 指向上轮使用的 `.npz + .json` 特征缓存目录，需对齐全部检测行。
- `--images-root` 是可解析缓存中图像相对路径的根目录，通常直接包含 `test2017/`。
- 若固定环境通过 `tracking_runtime` 提供，沿用 `--runtime-root /path/to/runtime_bundle`；环境本身能导入已验证 BoxMOT 时可省略。
- 本版使用 RGB 身份特征，不要求 thermal ReID 缓存。`appearance.enabled` 保持 false。

## 3. 软件验证

在仓库根目录运行：

```bash
python -m unittest discover -s tests/botsort_plus -v
```

原生集成测试需要真实固定版本 BoxMOT、PyTorch 和 OpenCV。合成图像/向量只用于软件边界和状态验证，不能产生 VT-Tiny-MOT 的性能结论。TrackEval 的集成测试按既有约定使用 `VTTINY_TEST_TRACKEVAL_ROOT`；缺少依赖导致的 skip 必须明确记录。

交付时的软件执行结果、可选测试的跳过说明和旧版本对照见 [validation_v0.3/VALIDATION.md](validation_v0.3/VALIDATION.md)。也可在保留原提交对象的 Git 仓库内复核旧配置兼容性：

```bash
python tools/check_botsort_plus_legacy.py --output outputs/c2_legacy_compatibility.json
```

该工具从固定基准提交临时提取代码，在独立进程回放原有 27 个配置，比较预测及事件哈希，不覆盖当前工作目录。合成回放只验证软件兼容性。

## 4. 输入预检

将以下占位路径替换为现有实际路径。在 Linux 或 PowerShell 中均可将命令写成一行。

```bash
python tools/run_botsort_plus.py --baseline-root /path/to/frozen_rgbt --config configs/botsort_plus/30_B_C2.json --images-root /path/to/VT-Tiny-MOT --features-dir /path/to/features_rgb_plus --output-dir outputs/botsort_plus_c2_full30 --preflight
```

预检只检查输入契约，不运行跟踪。`READY` 后再正式回放。原图缺失、检测/特征指纹不一致、高分检测缺少真实 RGB embedding 时应修正输入，不能用占位内容补齐后报告真实结果。

## 5. 先检查 shadow

```bash
python tools/run_botsort_plus.py --baseline-root /path/to/frozen_rgbt --config configs/botsort_plus/13_B_ref.json --images-root /path/to/VT-Tiny-MOT --features-dir /path/to/features_rgb_plus --output-dir outputs/botsort_plus_c2_full30 --trace
python tools/run_botsort_plus.py --baseline-root /path/to/frozen_rgbt --config configs/botsort_plus/27_B_C2_shadow.json --images-root /path/to/VT-Tiny-MOT --features-dir /path/to/features_rgb_plus --output-dir outputs/botsort_plus_c2_full30 --trace
```

可以先对两条命令同时加 `--sequences DJI_0030_2 DJI_0061_2 DJI_0181_3` 做接线检查，此时使用另一个输出目录，例如 `outputs/botsort_plus_c2_smoke`。全量实验删除 `--sequences`，不得混用不同序列集合的运行。

已有 A2 审计器可复用其输入核验和文件一致性功能：

```bash
python tools/audit_botsort_plus_a2.py --run-dir outputs/botsort_plus_c2_full30 --trackers B_ref B_C2_shadow --compare-pair B_ref B_C2_shadow --output outputs/botsort_plus_c2_full30/c2_shadow_parity.json
```

要求所有序列 MOT 文件逐字节一致，且输入协议一致。该工具沿用 A2 名称，只用于此处的基础输出完整性及 shadow 对照；它不替代对 C2 可信度、双记忆冻结等新机制的专门测试，也不计算准确率。

## 6. 主方案和消融

沿用第 5 节的 runner 命令及相同输出根目录，依次替换配置为：

1. `28_B_C2_memory.json`：仅写入门控。
2. `29_B_C2_association.json`：仅关联约束。
3. `30_B_C2.json`：完整方案。
4. 可选补充 `12_baseline_ref.json` 与 `31_C2.json`：检验与 B 的互补性。

完整方案示例：

```bash
python tools/run_botsort_plus.py --baseline-root /path/to/frozen_rgbt --config configs/botsort_plus/30_B_C2.json --images-root /path/to/VT-Tiny-MOT --features-dir /path/to/features_rgb_plus --output-dir outputs/botsort_plus_c2_full30 --trace
```

同一运行名称已存在时 runner 会拒绝覆盖。修改参数或需要重跑时选新输出根目录，不能在已有目录中静默续跑。协议会记录完整配置、代码版本与模块哈希、检测/特征指纹和实际原生参数。

## 7. C2 日志审计与门控分布

完成所有需要比较的回放后执行：

```bash
python tools/audit_botsort_plus_identity.py --run-dir outputs/botsort_plus_c2_full30 --trackers B_ref B_C2_shadow B_C2_memory B_C2_association B_C2 --compare-pair B_ref B_C2_shadow --output outputs/botsort_plus_c2_full30/c2_identity_audit.json
```

该工具核验输入协议、预测哈希、shadow 逐文件一致性，以及 C2 日志中关联/写入规则是否自洽；按序列和阶段统计 `allow_write`、拒写原因、冷启动参考回退、可信度分布和实际写入数。它要求完整 `--trace`；缺日志会明确失败，不用零计数冒充正常结果。输出文件已存在时拒绝覆盖。

尤其检查 high/unconfirmed 中 `reliability.below_0_1_fraction`、`reference_fallback_fraction`，以及 `history_conflict`、`competition_unknown`、`singleton_geometry_weak` 等原因是否占据大多数。如果低可信度普遍存在，关联支路可能主要依靠几何；如果写入长期饥饿，外观适应可能不足。拒写数量下降或上升本身不等于准确率提升。

`PASS` 表示协议、输出完整性与日志自洽，不代表日志中的身份一定正确。可信度分布统计的是被接受的匹配，并非所有候选对；真实误写率需要另行离线对齐 GT。软件测试对实际对象状态的断言在验证记录中列出。

## 8. 固定评估口径

```bash
python tools/evaluate_botsort_plus.py --run-dir outputs/botsort_plus_c2_full30 --trackers B_ref B_C2_memory B_C2_association B_C2 --gt-root /path/to/official_RGB_test_GT --trackeval-root /path/to/TrackEval
```

若还跑了补充两组，可在 `--trackers` 中加入 `baseline_ref C2`。GT 根目录直接包含各序列的 `gt/gt.txt`。沿用原生七类合并、COMBINED_SEQ 和冻结 TrackEval，不用逐序列百分数均值替代全量指标。

主要看完整方案相对 B_ref 的 HOTA、AssA、IDF1 和 IDSW，同时检查 FP、FN、DetA、MOTA 与逐序列结果。需要检查新增错误是否集中于冷启动、low 更新、B 恢复后的视角变化或某个拥挤序列。已有 30 个官方测试序列已经参与设计分析，应作为开发实验解释；独立泛化需要另行保留未参与选择的视频或航次。

## 9. 建议保留和回传的材料

保留 `protocols/`、`trackers/`、`events/`、`timing/`、评估生成的 `comparison.csv/json/md`、逐序列指标以及 shadow 一致性报告和 `c2_identity_audit.json`。首次可先回传总体比较、逐序列表及 C2 事件摘要；进一步定位需要完整日志和预测文件。

`timing` 中回放速度包含缓存特征的关联和 GMC 开销，不能解释为检测器 + ReID + 跟踪的端到端 FPS。新模块在真实数据上是否有收益、额外开销多少，均留待本轮冻结输入回放确定。
