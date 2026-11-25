# v0.4 软件验证记录

日期：2026-11-25。基准提交：`667eb40a6aa624c35fb86eb73656bb334b107a11`。

**本页记录软件验证，不是 VT-Tiny-MOT 真实数据的诊断结果，也没有报告新模块的 HOTA/IDF1 收益。** JPEG、检测、GT 和单位特征向量均来自明确标注的合成软件 fixture；跟踪回放使用真实、源码锁定的 BoxMOT/SOF。

机器可读结果与受测源码校验和见 [validation_report.json](validation_report.json)。

## 1. 运行环境

| 项目 | 本次验证环境 |
|---|---|
| 系统 | Linux x86_64 |
| Python | 3.12.14 |
| BoxMOT | 16.0.11，继续执行仓库原有源码 SHA256 校验 |
| NumPy | 2.3.5 |
| SciPy | 1.17.0 |
| OpenCV | 5.0.0 |
| PyTorch | 2.8.0+cpu |

这张表描述验证机器，不是要求用户升级到这些版本。请继续使用已跑通原 B 实验、且通过源码校验的运行环境。未在 Windows 上执行本次回归；文档提供了适合 Windows 的 UTF-8、路径和单行命令写法。

## 2. 最终新增测试：34 项全部通过

```bash
python -m unittest discover -s tests/botsort_plus -p "test_diagnostic*.py" -v
```

最终执行结果：`Ran 34 tests in 25.774s`，`OK`。

| 受测部分 | 数量 | 覆盖重点 |
|---|---:|---|
| 原生只读采集 | 4 | 开关前后轨迹、KF 均值/协方差、身份、记忆、候选、预测历史、事件、CMC 内部状态、输入与 OpenCV RNG 完全一致；每帧 SOF 仅一次，RANSAC 次数不变 |
| D1–D3 统计口径 | 12 | 因果历史身份锚点、拒配/歧义、困难样本保留、S 不重复加 R、未消费检测 NIS、GT 类别合并、缺标注/尾部删失、空分母、来源和帧完整性 |
| D4 图像与特征诊断 | 12 | 全检测覆盖、实际相邻帧、同锚点负样本、缺特征、原生像素代理、PNG 与人工标签、输入篡改拒绝、GT ID=0 |
| 统一实验主入口 | 6 | 实际独立回放→核对→D1–D4→打包，冻结阈值、错误 GT 预检、图像指纹不一致、缺 trace、禁止覆盖及包内容范围 |

主入口集成测试实际调用 `tools/run_botsort_diagnostics.py run`，不是用 mock 绕过跟踪器。微型序列包含空检测帧、低分候选、B 身份恢复和未来窗口右删失，最终输出保持原 ID，人工模糊标签初始仍为未复核。

## 3. 原有回归检查

```bash
python -m unittest discover -s tests/botsort_plus -p "test_*.py" -v
```

本轮全量回归执行 **323 项：322 通过、1 跳过**，耗时 100.403 秒。其中包含当时新增诊断的 33 项；既有测试为 289 通过、1 跳过。

跳过项为 `PinnedTrackEvalIntegrationTests.test_synthetic_perfect_and_one_miss_one_fp_through_real_cli`，原因是没有配置该可选集成测试要求的 `VTTINY_TEST_TRACKEVAL_ROOT`。本次没有更改官方评估器，也不把诊断配对指标称为官方 TrackEval 结果。

随后独立复核发现 D4 对 GT ID=0 的下界不一致，仅修正该解析并增加一项测试；重新运行了最终 34 项诊断测试，全部通过。没有将“预计 324 项”写成实际执行过的整套回归数量。

## 4. 与基准代码的差异复核

下列文件相对基准提交没有修改：

- `botsort_plus/native_adapter.py`
- `botsort_plus/recovery.py`
- `botsort_plus/candidates.py`
- `botsort_plus/appearance.py`
- `botsort_plus/geometry.py`
- `botsort_plus/config.py`
- `configs/botsort_plus/B_early_moderate.json`

`adapter_v2.py` 仅新增 9 行只读 observer 调用，没有删改原有决策代码。`run_botsort_plus.py` 增加显式诊断开关、递归源码身份、解码图像指纹和日志清单；关闭诊断时沿用原有跟踪行为。诊断包的源码哈希和原有输出行为是分别检查的，不能用文件名相同代替行为验证。

完整有效 B 配置（含默认值）的规范 JSON SHA256 锁定为：

```text
ac9816cad232ed814f4d810b377e9f146daefc65338f75a46f282e698e6a333d
```

该比较不受 JSON 排版或 Windows 换行影响；改变恢复阈值等有效参数会被拒绝。

## 5. 用户真实数据运行后还需确认

软件验证不能回答真实数据中哪种错因占比更高。请先跑单序列，再跑原冻结全量数据，并保留 `verification.json`、D1–D3 报告和 D4 人工复核结果。

GMC 光流统计是代理；协方差覆盖存在标注和样本选择偏差；真实模糊仍需人工确认；无可比人工样本、无身份锚点或视频尾部不足时保留未知。待这些证据齐备后再选择新模块并做真正的干预消融。
