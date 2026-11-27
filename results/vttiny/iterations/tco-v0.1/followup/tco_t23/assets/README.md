# TCO 官方训练序列资产

`frozen_inputs.py` 仅从官方训练图像索引生成冻结检测和 RGB ReID。它核对上次 T0 的两个权重摘要，沿用原 RGB–T 检测参数、RGB ReID 模型和检测行顺序。训练缓存不参与复用官方测试 30 段的跟踪结果。处理范围包括完整的 85 段和 34,111 对 RGB–IR 图像。原始 ZIP 中的 `train2017/DJI_0061_2` 图像没有官方训练帧索引，不纳入计划。

`offline_assets.py audit` 重新计算全部已索引训练和开发 RGB 帧的原始 JPEG 字节 SHA-256，对照历史入库记录，并报告跨划分和跨片段的完全重复。没有重复字节仍不能证明不同帧来自不同原始录像。`offline_assets.py gt` 单独把训练 COCO 转为逐段 MOT 标注，保留坐标、可信身份、原始类别和 1 基帧号。有冲突的身份及已有忽略／crowd 标志写为 `mark=0`，每个影响判断的来源记录可按标注 ID 追溯。

在 T0/T1 服务器工作目录下使用同一 `redetr` 环境。设置 `TCO` 为 `tco_53faeccb` 根目录，保留 `t01_v1` 配置及相邻的固定代码和 Round 3 配置。对于每张 GPU 的缓存进程设置与该物理卡一致的 `CUDA_VISIBLE_DEVICES`，沿用 Round 3 `PYTHONUSERBASE` 与 `TRACKING_RUNTIME_ROOT`。缓存生成进程不读取 GT。

```bash
ASSETS="$TCO/t23_v1/assets"
INPUTS="$TCO/t23_v1/training_inputs"
python "$ASSETS/frozen_inputs.py" plan --tco-root "$TCO" --output "$INPUTS" --gpus 5 6 7
python "$ASSETS/offline_assets.py" audit --tco-root "$TCO"
python "$ASSETS/offline_assets.py" gt --tco-root "$TCO"
CUDA_VISIBLE_DEVICES=5 python "$ASSETS/frozen_inputs.py" shard --tco-root "$TCO" --output "$INPUTS" --gpu 5
CUDA_VISIBLE_DEVICES=6 python "$ASSETS/frozen_inputs.py" shard --tco-root "$TCO" --output "$INPUTS" --gpu 6
CUDA_VISIBLE_DEVICES=7 python "$ASSETS/frozen_inputs.py" shard --tco-root "$TCO" --output "$INPUTS" --gpu 7
python "$ASSETS/frozen_inputs.py" verify --tco-root "$TCO" --output "$INPUTS"
```

三个 `shard` 进程可并行，已完成的缓存会在再次调用时重新核对。`verified.json` 验证每段特征行与检测行顺序、帧内偏移、模型摘要和文件字节。预测输入只指向 `training_inputs`。`training_gt` 和冲突清单供离线监督构建和评测使用。
