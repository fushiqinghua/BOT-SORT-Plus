# VT-Tiny-MOT：RGB 七种跟踪算法测试结果

已完成 200 轮训练的 RGB YOLO11s best.pt；官方测试集 30 段序列、12,590 帧。采用官方 RGB GT，所有目标类别联合评估；结果为 TrackEval COMBINED_SEQ，不是逐序列百分比平均。

|方法|HOTA↑|DetA↑|AssA↑|IDF1↑|MOTA↑|FP↓|FN↓|IDSW↓|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|BoT-SORT (no ReID)|36.36|23.39|57.58|42.25|25.54|11305|122310|240|
|ByteTrack|34.96|23.24|53.75|39.96|24.21|13541|122275|439|
|DeepSORT|32.90|21.10|52.38|37.53|23.55|8554|128428|447|
|ImprAsso|32.64|23.12|47.40|34.51|24.73|11067|122971|1268|
|Deep OC-SORT|31.72|19.30|53.11|35.47|22.62|5404|133372|330|
|OC-SORT|31.43|19.08|52.75|35.09|22.34|5352|133890|377|
|StrongSORT (online)|30.58|18.07|52.68|33.82|21.52|4132|136569|383|

## 实验口径

- 所有方法共用固定权重和检测缓存，检测阈值 0.05，NMS IoU 0.7，max_det 300，FP32，imgsz 640。原始七类检测保留，关联阶段合并为 object；不额外执行跨类别 NMS。
- 沿用已有跟踪参数，未在测试集调参。15 FPS 下缓存参数 60 对应最大丢失 30 帧。空检测帧仍推进跟踪器；空 GT 序列不排除。
- Deep OC-SORT、ImprAsso、DeepSORT、StrongSORT 共享 CLIP-VeRi 外观特征；车辆预训练，未在本数据集七类航拍小目标上微调。
- BoT-SORT 为无 ReID 版本；StrongSORT 为在线版本，不包含 AFLink/GSI。结果应以这些具体配置列入论文，不能称为原论文的完整复现。
- 未接入 CenterTrack / PFTrack，本轮共七种跟踪方法。

权重 SHA256：`713581a800f7e3b5d9233bbc4ff149a39d05a06cdc8fa324ea993bedfe02591e`

可复现信息见 protocol.json / source / implementation_provenance；逐序列与汇总指标见 metrics/summary.json；完整 MOT 轨迹见 trackers/<method>/data/。

GPU1 用于检测和外观特征；关联阶段多进程并行，当前计时不能作为严格的单任务端到端实时速度对比。