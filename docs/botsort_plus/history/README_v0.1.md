# BoT-SORT-Plus：A / B / C 增量实现 v0.1

本补丁针对上传的 `BOT-SORT-Plus-baseline.zip`。保留原检测缓存、已完成结果、原训练与评估脚本；新增可独立开关的跟踪扩展和运行入口。

**已实现的是可执行机制和实验通路，尚没有新的 VT-Tiny-MOT HOTA / IDF1 收益结论。** 所有新增阈值都是初始实验设置。软件测试中的图像、向量和 TorchScript 模型是测试夹具，不是训练模型或真实数据集结果。

## 模块

|模块|实现|作用|
|---|---|---|
|A 时序弱观测起轨|`botsort_plus/candidates.py`|尺度感知几何、短窗口命中、证据积累、歧义拒绝、重复抑制|
|B 短轨迹身份恢复|`botsort_plus/recovery.py`|同帧历史 KF 验证、有限不确定性搜索、可信外观、身份预留、一对一恢复|
|C 质量感知 RGB–T 外观|`botsort_plus/appearance.py`|分模态可信记忆、质量加权距离、先判定后更新、缺失模态回退|
|真正接入 BoT-SORT|`botsort_plus/native_adapter.py`|原生高分→低分→未确认匹配后，先 B，再 A / 原生起轨|
|真实特征导出|`tools/export_botsort_plus_features.py`|微调 RGB ReID + 显式热红外 TorchScript 编码器，包含低分检测|
|回放 / 评估|`tools/run_botsort_plus.py`、`tools/evaluate_botsort_plus.py`|冻结检测、SOF GMC、日志、完整性校验、独立 TrackEval 评估|

完整机制和理论复核见 [设计与复核](docs/botsort_plus/DESIGN_AND_REVIEW.md)；完整命令、消融和输入要求见 [实验指南](docs/botsort_plus/EXPERIMENTS.md)。来源及许可见 [来源说明](docs/botsort_plus/PROVENANCE.md)。本次验证记录位于 `docs/botsort_plus/validation/`。

## 最短使用顺序

1. 应用补丁后，在**现有 baseline 的 Python 环境**中运行 `python -m unittest discover -s tests/botsort_plus -v`。纯核心只依赖 NumPy/SciPy；原生集成测试需要 BoxMOT、PyTorch、OpenCV。
2. 用原始 RGB 图像和已微调 `clip_veri.pt` 导出新特征缓存。旧压缩包未包含这些图像或权重；旧高分缓存也不能冒充低分特征。
3. 先跑 `00_baseline.json`，再跑 A、B、A+B。它们必须共用同一份新特征缓存和冻结检测。
4. 通过独立评估入口比较 HOTA、DetA、AssA、IDF1、MOTA、FP、FN、IDSW。
5. 准备真实 thermal 编码器、明确其预处理与 RGB→thermal 空间映射后，再跑 C 与 A+B+C。

```bash
python tools/run_botsort_plus.py \
  --config configs/botsort_plus/06_AB.json \
  --images-root /path/to/VT-Tiny-MOT \
  --features-dir /path/to/features_rgb_plus \
  --preflight
```

`--images-root` 应直接包含缓存路径里的 `test2017/`。`--preflight` 缺输入时返回退出码 2；它不会用黑图代替真实 SOF 输入。正式运行去掉 `--preflight`，增加 `--trace` 可保留逐帧决策日志。默认输出到 `outputs/botsort_plus_v1/`，同名实验拒绝覆盖。

## 本版的重要边界

- **B 是残余恢复。** 原生已经接受的匹配不会被撤销。原生能够直接恢复的目标仍走原流程，B 的补充路径默认需要 3 次观测。
- 默认 `track_buffer=60, frame_rate=15`，实际原生保留窗口为 30 帧；B 不会暗中延长该窗口。
- A、B 都是在线因果实现。确认之前不输出候选；确认之后只输出当前帧，不回填、插值或改写旧 ID。
- C 的融合与记忆已经实现；补丁不含训练好的 thermal ReID 权重。不能把 RGB 编码器未经验证地套用热红外后称为已完成热红外适配。
- 此前官方 test 已用于选模、校准和错因诊断，本轮同一划分上的结果属于开发实验。独立泛化结论仍需真正未使用的数据。
