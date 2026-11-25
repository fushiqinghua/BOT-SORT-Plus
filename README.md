# BoT-SORT 基线 · VT-Tiny-MOT

> 当前研究分支增加了 **M2 概率几何关联实现、M1 / M3 诊断命令与分阶段实验计划**。从 [实验手册](docs/research/README.md) 开始；M2 默认参数尚待本地数据验证。以下为继承的冻结基线归档，原有结果保留。

本分支归档多模态跟踪的冻结基线，供后续跟踪模块改进和消融比较使用。主配置为 **YOLO11-M RGB–红外 P3/P4/P5 特征级融合（Mosaic 开启，第 36 轮）+ BoT-SORT + 微调 RGB ReID**。RGB 单模态采用第 26 轮快照作为输入对照。

| 检测和跟踪配置 | HOTA ↑ | IDF1 ↑ | MOTA ↑ | IDSW ↓ |
|---|---:|---:|---:|---:|
| RGB + BoT-SORT + 微调 ReID | 35.54 | 37.65 | 21.32 | 303 |
| RGB–红外 + BoT-SORT，关闭 ReID | 37.81 | 42.36 | 25.75 | 425 |
| **RGB–红外 + BoT-SORT + 微调 ReID** | **37.86** | **42.45** | **25.76** | **422** |

官方测试集包含 30 段、12,590 组配对帧，按 RGB 坐标汇总七类目标。HOTA、IDF1、MOTA 为 TrackEval 的 COMBINED_SEQ 百分制结果。官方测试集还用于检测器与 ReID 选模、跟踪门限校准和错因分析，所有结果使用这一口径。

## 实验索引

| 内容 | 记录 |
|---|---|
| YOLO11-M RGB 与 RGB–红外 | [8 个跟踪配置的对照表](results/vttiny/yolo11m_rgb_rgbt/comparison.md)、[RGB](results/vttiny/yolo11m_rgb_rgbt/rgb)与 [RGB–红外](results/vttiny/yolo11m_rgb_rgbt/rgbt)的检测缓存、480 份轨迹和逐序列指标 |
| BoT-SORT 错因分析 | [中文报告](results/vttiny/botsort_error_analysis/BoT-SORT基线错因分析.md)、[案例帧索引](results/vttiny/botsort_error_analysis/case_index.json)、[逐帧明细](results/vttiny/botsort_error_analysis) |
| YOLO11-S RGB 早期基线 | [七种跟踪器结果](results/vttiny/rgb_yolo11s/baseline_results.md) |
| RGB ReID 微调 | [微调前后结果](results/vttiny/rgb_reid_finetune/comparison.md)、[下降原因分析](results/vttiny/rgb_reid_diagnosis/diagnosis.md) |
| 跟踪门限校准 | [四种外观方法](results/vttiny/rgb_reid_calibration/comparison.md)、[BoT-SORT ReID 对照](results/vttiny/rgb_botsort_reid/comparison.md)、各自试验台账 |
| 检测器训练 | [网络、数据与训练配置](configs/detectors)、[三组 YOLO11-M 训练曲线原始记录](results/vttiny/detector_training_snapshot)。这组训练日志截取于归档时，追踪评测所用轮次以各自 `checkpoint.json` 为准 |
| M3OT 早期实验 | [历史阶段结果及多机汇总](results/m3ot_history) |

[跟踪器参数](configs/trackers)来自冻结评测协议。ReID 的输入为检测框在 RGB 帧上的裁剪。BoT-SORT 的高分门限为 0.50、低分门限为 0.10、新轨门限为 0.60。15 FPS 下配置为 60 的缓冲经实现缩放后为 30 帧。微调 ReID 的外观距离门限为 0.40，相机运动补偿使用稀疏光流。

## 用已归档轨迹复算指标

准备官方 VT-Tiny-MOT RGB 标注目录，格式为 `GT/GT_00/test/<序列>/gt/gt.txt`，包含空标注的 DJI_0325_2。所需序列名和长度由冻结协议提供。该步骤仅需 CPU、官方标注及仓库中的预测轨迹。

```bash
python -m pip install -r requirements-eval.txt
git clone https://github.com/JonathonLuiten/TrackEval.git external/TrackEval
git -C external/TrackEval checkout 12c8791b303e0a0b50f753af204249e622d0281a
python tools/verify_baseline.py --gt-folder /path/to/GT/GT_00/test --output outputs/recomputed_metrics
```

`verify_baseline.py` 会逐项比对两套检测输入下全部 496 行记录和八项指标。使用外部 TrackEval 路径时可设置 `TRACKEVAL_ROOT`。原实验的 TrackEval 源文件指纹见 [provenance/trackeval.json](provenance/trackeval.json)。

## 从检测缓存复算错因

仓库内两组检测缓存均为按序列压缩的 JSON，保留框坐标、类别、置信度、帧号及检测权重指纹。帧路径相对于官方数据目录。报告可在 CPU 上用冻结轨迹与官方标注复算。

```bash
export VTTINY_BASELINE_RESULTS="$PWD/results/vttiny/yolo11m_rgb_rgbt"
export VTTINY_GT_FOLDER=/path/to/GT/GT_00/test
export VTTINY_DIAGNOSIS_OUTPUT="$PWD/outputs/error_analysis"
python tools/diagnose_vttiny_botsort_m.py --out "$VTTINY_DIAGNOSIS_OUTPUT"

# 可选。用有权访问的官方图像在本机重绘案例
export VTTINY_FRAME_ROOT=/path/to/VT-Tiny-MOT
python tools/inspect_vttiny_botsort_cases.py
```

数据入口见 [VT-Tiny-MOT 官方仓库](https://github.com/xuqingyu26/HGTMT)。仓库提供预测框、轨迹、评测指标、错因明细和案例帧索引。原始数据、原画面可视化、权重、外观嵌入及 YOLOMM 框架保存在原实验环境。原权重和输入缓存可通过所附 SHA-256 验证。归档中的路径为相对路径或环境变量占位符。检测框和轨迹数值沿用原实验。

## 在原实验环境回放基线

`tools/evaluate_vttiny_m_pair.py` 是两套检测与八种跟踪配置的原实验入口。部署路径及依赖如下。

- `VTTINY_ROOT` 指向含 `data/processed_v1` 和 `outputs` 的 VT-Tiny-MOT 实验根目录。完整运行依赖已准备的配对清单、早期校准协议、两组冻结检测权重及微调 ReID 权重。
- `TRACKING_RUNTIME_ROOT` 指向含 `tracking_runtime` 与 `tracking_extensions` 的实验运行目录。原环境为 Python 3.11、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、OpenCV 5.0.0.93、BoxMOT 16.0.11。其余跟踪器依赖和补丁记录见 [provenance](provenance)。
- `VTTINY_DETECTOR_FRAMEWORK` 指向有 `YOLOMM` 及融合模块的 Ultralytics 实验框架。两组推理使用的 GPU 可由 `VTTINY_RGB_GPU`、`VTTINY_RGBT_GPU` 指定。
- 使用专用实验输出副本运行 `python tools/evaluate_vttiny_m_pair.py`，以保留已有成果。原始运行入口保存在结果目录的 `source/` 中，用于追溯；`tools/` 中的入口支持环境变量配置。

上游代码与许可见 [BoxMOT](https://github.com/mikel-brostrom/boxmot)、[TrackEval](https://github.com/JonathonLuiten/TrackEval) 和 [DeepSORT](https://github.com/nwojke/deep_sort)。实验归档中的第三方片段和兼容性补丁沿用其各自的上游许可。
