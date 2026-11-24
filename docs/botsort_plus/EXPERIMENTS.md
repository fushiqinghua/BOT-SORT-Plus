# 实验运行与判读

## 1. 环境与输入

优先复用已经跑通 baseline 的 Python 环境与 `TRACKING_RUNTIME_ROOT`。本补丁不重新训练或调用 YOLO 检测器，不要求安装另一套 Ultralytics；直接读取已冻结的 `.json.gz` 检测缓存。

需要：

- 原始 RGB 图像，根目录下有 `test2017/<sequence>/00/`。
- baseline 已微调 `clip_veri.pt`，SHA 与 `rgbt/protocol.json` 一致。
- 同一 baseline BoxMOT 16.0.11 源码环境；默认仍为 15 FPS / SOF / ReID。
- 评估时需要官方 RGB GT 与对应固定版本 TrackEval；GT 不传给预测进程。
- C 另需配对 thermal 图像、真实 thermal TorchScript ReID 权重、训练一致的预处理与空间映射。

当前上传压缩包没有原图、权重或特征缓存，不能在只有压缩包内容的条件下完成同口径的完整追踪。`--preflight` 会报告缺少的具体输入。实际运行不提供黑图替代 GMC 或伪造特征的选项。

## 2. 软件测试

在补丁应用后的仓库根目录：

```bash
python -m unittest discover -s tests/botsort_plus -v
```

若要同时执行固定版本 TrackEval 的真实进程测试，明确提供其目录：

```bash
VTTINY_TEST_TRACKEVAL_ROOT=/path/to/TrackEval \
python -m unittest discover -s tests/botsort_plus -v
```

不设置该变量时，仅这一项评估器集成测试会跳过；本补丁所附 138 项通过的记录已经设置该变量，实际执行且没有跳过。应用到真实仓库后不需要设置 `VTTINY_TEST_BASELINE_ROOT`。

全部原生测试需要既有 BoxMOT/PyTorch/OpenCV 环境。独立核心可直接运行，例如：

```bash
python -m unittest discover -s tests/botsort_plus -p test_candidates.py -v
python -m unittest discover -s tests/botsort_plus -p test_recovery.py -v
python -m unittest discover -s tests/botsort_plus -p test_appearance.py -v
```

## 3. 导出 A/B 所需真实 RGB 特征

```bash
python tools/export_botsort_plus_features.py \
  --detections-dir results/vttiny/yolo11m_rgb_rgbt/rgbt/detections \
  --images-root /path/to/VT-Tiny-MOT \
  --rgb-weights /path/to/clip_veri.pt \
  --runtime-root /path/to/M3OT_MOT_Baseline_v0_1 \
  --output-dir /path/to/features_rgb_plus \
  --device cuda:0
```

`--runtime-root` 可省略（运行环境已经能直接导入正确 BoxMOT 时）。保留权重文件名 `clip_veri.pt`，BoxMOT 会根据文件名选择架构和裁剪规格。脚本会逐 tensor 校验加载后的 RGB 视觉模型。

新格式保存每个原始检测的位置、有效位与质量。低分检测会实际提取特征；全零无效行不能当身份特征使用。导出默认拒绝覆盖，修改编码器应写新目录。

**先复跑 B0。** 新缓存的编码 batch 含低分裁剪，可能与原来仅高分的批次产生微小浮点差异。因此：先用同一份新特征跑 baseline，与旧冻结结果比较；之后 A/B/C 都和本次重放 baseline 比较。如果 B0 出现实质变化，先排查权重、图像、GMC、缓存、运行库和参数，再解释改进结果。

## 4. 回放

先预检：

```bash
python tools/run_botsort_plus.py \
  --config configs/botsort_plus/00_baseline.json \
  --images-root /path/to/VT-Tiny-MOT \
  --features-dir /path/to/features_rgb_plus \
  --runtime-root /path/to/M3OT_MOT_Baseline_v0_1 \
  --preflight
```

正式 baseline：

```bash
python tools/run_botsort_plus.py \
  --config configs/botsort_plus/00_baseline.json \
  --images-root /path/to/VT-Tiny-MOT \
  --features-dir /path/to/features_rgb_plus \
  --runtime-root /path/to/M3OT_MOT_Baseline_v0_1 \
  --output-dir outputs/botsort_plus_v1
```

A+B：

```bash
python tools/run_botsort_plus.py \
  --config configs/botsort_plus/06_AB.json \
  --images-root /path/to/VT-Tiny-MOT \
  --features-dir /path/to/features_rgb_plus \
  --runtime-root /path/to/M3OT_MOT_Baseline_v0_1 \
  --output-dir outputs/botsort_plus_v1 \
  --trace
```

替换 `--config` 运行对应消融。可用 `--sequences DJI_0030_2 DJI_0061_2` 做接入排错；协议会标为子集。正式评估保留全部 30 段，包括空 GT 序列。

已有同名运行拒绝覆盖。改参数时修改配置名或输出根目录，避免把不同参数的轨迹混到一起。失败运行不会标为 `COMPLETE`，评估入口也会拒绝它。

## 5. 配置与消融顺序

|配置文件|运行名称|回答的问题|
|---|---|---|
|00_baseline.json|baseline|新缓存、同运行环境能否复现主基线|
|01_instant_weak_birth.json|instant_weak_birth|不积累时序证据，单帧弱观测起轨的代价是什么|
|02_fixed_window_birth.json|fixed_window_birth|固定 N/W 确认是否已足够|
|03_A.json|A|证据强度和漏观测惩罚是否有增益|
|04_long_buffer.json|long_buffer|只延长原生 lost buffer 能获得多少变化|
|05_B.json|B|原始 30 帧保留范围内，补充恢复是否有效|
|06_AB.json|AB|起轨与恢复的组合是否优于单项|
|07_C.json|C|只改变质量感知外观是否有效|
|08_ABC.json|ABC|三模块整体是否有效|
|09_AB_long_buffer.json|AB_long_buffer|扩展到实际 60 帧保留时，是否超出仅延长 buffer 的效果|
|10_C_rgb_quality.json|C_rgb_quality|仅 RGB 质量记忆的作用|
|11_C_fixed_fusion.json|C_fixed_fusion|保持质量门与记忆机制，仅移除动态融合权重的影响|

`instant_weak_birth` 仍共享候选重复过滤和原生先关联流程，和 A 的主要差别是取消多帧确认。单独把原生 `new_track_thresh` 降到 .5 以下并不会让原生低分段进入初始化，所以不把这种无效控制误称为“低分起轨”。

`fixed_window_birth` 与 A 共享尺度几何、重复过滤和窗口命中；两者比较主要隔离证据累积条件。若后续要单独证明尺度几何的贡献，应增加对应几何替换对照，不应从这个消融推断全部 A 子组件分别有效。

## 6. 评估与比较

```bash
python tools/evaluate_botsort_plus.py \
  --run-dir outputs/botsort_plus_v1 \
  --trackers baseline A B AB \
  --gt-root /path/to/official_RGB_test_GT \
  --trackeval-root /path/to/TrackEval
```

GT 根目录直接包含 `<sequence>/gt/gt.txt`。这里复用仓库原 `evaluate_vttiny.py`：RGB 坐标、全部类别合并、官方 GT 不修改、TrackEval `COMBINED_SEQ` 聚合。跟踪阶段不能读取 GT，评估是独立进程。

输出 `summary.json`、`comparison.json/.csv/.md` 和 `evaluation_audit.json`。同时检查输入身份、序列帧数、预测 SHA、MOT 格式、每帧 ID 唯一和固定 TrackEval 源码。历史版本的 CRLF 与同源码 LF 换行差异会明确记录，不能借此放行其它代码差异。

若需要对照仓库中历史冻结指标：

```bash
python tools/evaluate_botsort_plus.py \
  --run-dir outputs/botsort_plus_v1 \
  --trackers baseline A B AB \
  --gt-root /path/to/official_RGB_test_GT \
  --trackeval-root /path/to/TrackEval \
  --reference-summary results/vttiny/yolo11m_rgb_rgbt/rgbt/metrics/summary.json \
  --reference-tracker botsort_reid \
  --reference-context "历史冻结主基线；原运行环境及原特征批次"
```

优先使用本次重放 baseline 做因果比较；历史结果用于排查是否偏离主基线，不替代共同输入的消融。

## 7. C 的真实双模态缓存

额外导出命令示例中的 `H W`、`C`、`MEAN...`、`STD...` 必须替换为 thermal 模型训练时的真实预处理；不能照抄一个随意的尺寸和归一化：

```text
python tools/export_botsort_plus_features.py
  --detections-dir results/vttiny/yolo11m_rgb_rgbt/rgbt/detections
  --images-root /path/to/VT-Tiny-MOT
  --rgb-weights /path/to/clip_veri.pt
  --output-dir /path/to/features_rgbt_plus
  --thermal-torchscript /path/to/thermal_reid.ts
  --thermal-input-size H W
  --thermal-channels C
  --thermal-mean MEAN...
  --thermal-std STD...
  --thermal-homography /path/to/rgb_to_thermal_3x3.json
```

上面为参数清单，实际 shell 命令需写在一行或添加续行符。单应矩阵 JSON 是 3×3 数组。仅在输入确实已对齐时，才使用 `--thermal-aligned` 代替 homography。热图 uint8/uint16 分别按 255/65535 归一化；float 输入须已在 [0,1]。模型必须返回 `N×D` Tensor；不是分类 logits 接口。

随后 C / ABC 的回放将 `--features-dir` 指向新 RGB–T 缓存。最好将同一双模态缓存也用于该组 baseline / AB 重放，关闭 C 时 thermal 不参与关联。若只是 `C_rgb_quality`，仍可使用纯 RGB 缓存。

本版不负责热红外 ReID 的训练。若尚无合适 thermal 模型，应先完成 A/B，依据真实失败例决定训练 thermal 编码器或轻量投影；不能将接口已打通等同于 C 已获得性能收益。

## 8. 结果判读与下一轮调参

- A：观察 FN / DetA 的变化，同时看 FP 和新 ID 数量；重复海面背景也可能持续出现，多帧条件并不天然保证真目标。
- B：重点看 AssA / IDF1 / IDSW，检查 `recovered`、`recovery_check` 和 `deferred_birth` 日志，人工/GT 事后核对正确恢复与错误恢复。
- C：先检查 `feature_coverage`、`memory_update` 和匹配日志中的各模态有效性。若大多数微小裁剪达不到写入门限，先调整质量标定，再决定扩大模型。
- 使用逐序列结果检查收益是否仅集中于一个大序列；小于 8、8–16、16–32、≥32 像素的目标尺度分组也应保留。
- 计时中 association+GMC、缓存回放、特征导出分别记录。不能把缓存回放 FPS 当检测+双编码器的端到端实时速度。
- 本次默认参数未在真实数据上优化。保持相同调参预算，先单模块，再组合；开发集调参之后再冻结到真正独立数据验证。
