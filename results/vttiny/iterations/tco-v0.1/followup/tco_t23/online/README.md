# TCO 原生低分匹配后修正

`replay.py` 在固定提交的原生低分求解之后调用 TCO。通过校准阈值的候选仅替换本次 KF 量测坐标，原始检测行索引、置信度、类别、特征与外观记忆更新保留。历史由同一运行已经接受的原始检测行形成。跟踪进程只接收冻结输入与模型采用参数，评测在独立进程读取真值。

使用与 T0 一致的服务端环境运行下列命令。变量指向现有资产和本次新输出目录。默认 TCO 推理使用 CPU，可在已分配设备上用 `--cuda-devices` 和 `--device` 指定 GPU。

```bash
ONLINE="$T23/src/online"
COMMON=(--repo "$PIN" --collector "$T0/src/collector.py" --t0 "$T0")
TRAIN=(--train-baseline "$T23/training_inputs" --train-features "$T23/training_inputs/features"
       --images-root "$RAW" --split "$T23/split/training_split.json")

python -Bu "$ONLINE/batch.py" capture "${COMMON[@]}" "${TRAIN[@]}" \
  --output "$T23/training_runs" --jobs 4 --streaming-capture
```

训练捕获使用 B 的原生 SOF。采样分片覆盖不同序列，验证每片后原样合并文件，并记录各自协议摘要。完成凭据为 `training_runs/capture_manifest.json`，离线训练标签生成时读取 `training_runs/B/`。

```bash
python -Bu "$ONLINE/batch.py" parity "${COMMON[@]}" --selection "$SELECTION" --output "$T23/parity"
python -Bu "$ONLINE/batch.py" holdout "${COMMON[@]}" "${TRAIN[@]}" \
  --selection "$SELECTION" --gt "$TRAIN_GT" --trackeval "$TRACKEVAL" \
  --parity "$T23/parity/TCO_runtime_parity.json" --output "$T23/holdout"
python -Bu "$ONLINE/batch.py" official "${COMMON[@]}" --split "$T23/split/training_split.json" \
  --selection "$SELECTION" --gt "$OFFICIAL_GT" --trackeval "$TRACKEVAL" \
  --parity "$T23/parity/TCO_runtime_parity.json" --holdout "$T23/holdout/comparison.json" \
  --t2-gate "$T2_GATE" --output "$T23/T3a"
```

`parity` 对两个参考在同一六段样本验证关闭、只记录和强制回退。三种模式的轨迹、原事件和状态链均须与 T0 一致。训练侧留出默认为 B。可配置准确的历史 LightGlue 资产和 GPU 后，用 `--include-bg` 评测留出侧自适应 GMC。完整 30 段始终沿用冻结的相机补偿缓存，两组采用同一模型和阈值。正式运行入口检验 T2 闸门、训练侧留出凭据、模型文件、源码和输入摘要。`T2_GATE` 由训练侧验收写明 `allow_t3a` 及 `selection_sha256`。

`comparison.csv` 列出原版、开启 TCO 和两者差值。采用 TrackEval `COMBINED_SEQ` 汇总七类合并的 HOTA、DetA、AssA、IDF1 及 MOTA 百分数，误报、漏检和换号为计数。`changed_events.csv` 以 GT 身份或原检测行连接闭环差异，`refinement_effects.csv` 区分量测、KF 后状态与实际输出的定位变化，`followup_5_10.csv` 记录后续关联及覆盖，缺失锚点与尾部截断单独保留。
