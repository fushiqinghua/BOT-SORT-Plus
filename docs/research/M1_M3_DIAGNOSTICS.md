# M1 / M3：先诊断，再决定下一步实现

本分支实现 **M2 最小原型**；M1 和 M3 的交付范围是可运行的诊断工具、受控实验安排和后续模块接口。这里没有把尚未训练的红外 ReID、未标定的相机运动不确定性或未运行的实验写成已完成的算法改进。

## 1. 公共实验口径

- 采用冻结的检测缓存，不重新训练检测器，不改变检测阈值、NMS 或类别池化方式。
- 跟踪运行阶段不读 GT；本文件中的 M3 诊断是独立的离线分析进程，可以读 GT。
- M1 的原生 SOF / 关闭 SOF 对照，保留同一套关联、轨迹出生、删除和输出规则。
- 所有官方指标以 TrackEval 的 `COMBINED_SEQ` 为准。诊断中的匹配、候选覆盖和身份变化不冒充官方 FN、IDSW 或 ReID 准确率。
- 原官方测试序列已参与历史模型选择、参数比较或诊断。本轮按录制序列分组划分后的结果仍属于回顾性开发实验；独立泛化结论需要未参与开发的新留出集。

下面命令在仓库根目录执行。`--image-root` 指向含 `test2017/` 的图像根目录，`--gt-root` 指向含 `<sequence>/gt/gt.txt` 的 RGB GT 根目录。所有输出目录必须尚不存在，以免混用两次实验。

## 2. M1：运动补偿是否确实存在可改善的问题

### 2.1 已实现的逐帧诊断

```bash
python -m botsort_research.cli diagnose-m1 \
  --image-root /path/to/VT-Tiny-MOT \
  --split runs/research/split.json --subset evaluation \
  --output runs/research/m1_quality
```

第一次检查可加 `--sequences DJI_0159_2 --max-pairs 100`，前提是该序列属于指定划分。`--max-pairs` 取前 N 个连续帧对，不能将其结果写成全序列诊断。每个序列生成 `pairs.csv` 和 `summary.json`，总目录另有选择范围清单。

`SOFProbe` 使用 BoxMOT v16.0.11 SOF 的基本设置：缩放 0.15、最多 1,000 个 Shi–Tomasi 角点、LK 窗口 21×21、3 层金字塔、部分仿射 RANSAC。它不读取 GT、不使用检测框遮罩，也不把反向光流结果加入估计筛选。

|输出|含义|判断时需要注意|
|---|---|---|
|`status` / `estimated`|初始化、特征不足、光流失败、仿射失败或成功估计|失败时输出的单位矩阵不是“相机静止”的证据|
|`tracked` / `inliers` / `inlier_fraction`|成功跟踪点、RANSAC 内点及其比例|海浪自身一致运动也可能产生高内点率|
|`inlier_grid_coverage`|4×4 网格中有内点的格子比例|集中在局部水纹上的大量内点不等于覆盖全图|
|`residual_*_native_px`|内点仿射重投影残差，换算至原图像素|是模型拟合残差，不能当作真实相机补偿误差|
|`fb_*_native_px`|光流前向再反向的闭合误差|仅观测，不改变当前仿射变换|
|`tx` / `ty` / `rotation_degrees` / `affine_scale`|前一帧到当前帧的变换参数|检查异常跳变，同时回看对应原图|

RANSAC 的默认阈值是 **缩小后图像上的 3 像素**；0.15 倍缩放时相当于原图约 20 像素。日志已经明确记录两个单位。这个换算本身不证明阈值应该改小，仍需用跟踪结果检验。

该工具是独立的 SOF 风格诊断回放，并非跟踪器内部插桩。因此日志用于定位值得回看的帧，不能声称每一项内部值与原生 SOF 的一次实际运行完全相同。正式跟踪对照仍使用原 BoxMOT 实现。

### 2.2 必须配套的因果对照

```bash
python -m botsort_research.cli run \
  --image-root /path/to/VT-Tiny-MOT \
  --split runs/research/split.json --subset evaluation \
  --variants native_iou --cmc sof --output runs/research/m1_sof

python -m botsort_research.cli run \
  --image-root /path/to/VT-Tiny-MOT \
  --split runs/research/split.json --subset evaluation \
  --variants native_iou --cmc none --output runs/research/m1_none

python -m botsort_research.cli evaluate \
  --run-dir runs/research/m1_sof --gt-root /path/to/mot/rgb/test \
  --trackeval-root /path/to/TrackEval

python -m botsort_research.cli evaluate \
  --run-dir runs/research/m1_none --gt-root /path/to/mot/rgb/test \
  --trackeval-root /path/to/TrackEval
```

先看总体 HOTA、AssA、IDF1 和 FN/FP/IDSW，再看逐序列差异，并把差异较大的序列与 `pairs.csv` 和视频一起检查。只有质量代理变差且 SOF 对照也显示性能问题时，才有证据优先开展鲁棒背景点选择或补偿不确定性建模。不能仅凭几帧残差大就宣布 M1 有效。

### 2.3 后续模块设计与边界

后续可将检测区域屏蔽、空间均匀采样、前后向一致性筛选作为独立消融，再研究相机变换参数协方差及其对轨迹预测协方差的传播。若相机不确定性已加入预测协方差 P，就不能在检测测量噪声 R 中重复计算。没有静态参照、IMU 或相机标定时，水面整体运动与相机运动可能不可辨识，需保留退化状态和回退策略。

## 3. M3：当前外观特征在哪些条件下有用

### 3.1 无需外观缓存即可运行的诊断

```bash
python -m botsort_research.cli diagnose-m3 \
  --gt-root /path/to/mot/rgb/test \
  --split runs/research/split.json --subset evaluation \
  --predictions-dir results/vttiny/yolo11m_rgb_rgbt/rgbt/trackers/botsort_reid/data \
  --output runs/research/m3_coverage
```

这一步分析冻结检测中的候选覆盖、原生目标尺寸、连续低分检测片段，以及提供轨迹后的输出中断和身份变化。没有特征时，外观部分会明确记为 `embeddings_not_supplied`，其余诊断正常完成。

|文件|用途|
|---|---|
|`gt_coverage.csv`|每个 GT 的尺寸分组、候选 IoU、可用置信度类别和诊断匹配轨迹 ID|
|`low_score_runs.csv`|连续存在可对齐低分检测的片段，帮助判断低分关联问题是否值得继续研究|
|`prediction_gaps.csv`|两次成功输出匹配之间的间隔、是否有标注缺口以及是否保持原 ID|
|`appearance_pairs.csv`|有有效特征时的同 ID / 不同 ID 余弦距离样本|
|`summary.json`|分组计数、有效与缺失特征覆盖、距离分布、输入哈希和明确的跳过原因|

尺寸按原图框的等效边长 `sqrt(width * height)` 分为 `<8`、`8–16`、`16–32` 和 `32+` 像素。把 7 像素小框放大到 256×256 不会恢复其原始身份信息，因此分别统计检测裁剪尺寸和有效特征裁剪尺寸。

候选匹配使用 IoU≥0.5 的离线定义，用来描述检测中存在怎样的机会；它没有重现某个时刻的预测协方差和关联矩阵，不能直接断言跟踪器“错误拒绝了这个检测”。置信度仍遵循冻结实现中的严格比较：`.1 < score < .5` 属于低分关联；`score > .5` 属于高分；恰好 .1/.5 的边界单列为关联门限之外。若同时有 .5 和 .4 的候选，诊断仍识别可用的 .4 低分检测。

### 3.2 有原始 RGB 特征缓存时补充外观分析

```bash
python -m botsort_research.cli diagnose-m3 \
  --gt-root /path/to/mot/rgb/test \
  --split runs/research/split.json --subset evaluation \
  --embeddings-dir /path/to/original/rgbt/embeddings \
  --predictions-dir results/vttiny/yolo11m_rgb_rgbt/rgbt/trackers/botsort_reid/data \
  --max-pairs-per-sequence 10000 --seed 0 \
  --output runs/research/m3_rgb_features
```

每段序列需要 `<sequence>.npz` 和同名 `.json`。NPZ 使用原基线格式：`features`、`indices`、`offsets`、`frame_ids`；JSON 至少包含 `detection_sha256` 和 `features_sha256`。压缩检测缓存允许用**解压后原始字节**与原检测哈希一致来验证；不会通过重新排序或重新编码 JSON 来规避哈希不一致。若仓库归档改变了缓存中的图像路径，应通过 `--cache-dir` 提供生成特征时对应的原缓存，或重新提取特征。

帧号、索引边界、重复索引和偏移数组均会检查。零向量与非有限向量按无效特征排除；缺失低分检测特征不会被当作“低余弦距离”。哈希不一致会跳过该序列的外观分析并记录原因，不能将它解释为外观效果差。

外观样本的具体定义：先用检测与 GT 的一对一匹配获得身份标签，对 GT 归属有歧义的检测排除；使用该身份上一条有效特征构成时间记忆。正、负样本均需满足最大 30 帧的观测间隔和 64 像素中心距离门限。负样本最多取最近的五个其他身份；不把同帧新特征先写进图库。每序列、每种正负类别最多保留指定数量的固定随机种子蓄水池样本，并报告实际见过的样本总数。

这些是局部几何候选范围内的外观混淆诊断，不是完整 ReID 检索评测，也不代表原跟踪器的全部门控条件。先比较小目标、长间隔等分组下正负距离是否分离，再决定特征质量门控和记忆更新。日志中的 `distance > 0.4` 比例仅对照原第一阶段阈值，不表示 0.4 是新模块的最优阈值。

### 3.3 红外分支与时序记忆的后续实现条件

当前仅使用用户已有的 RGB 特征，未实现红外身份网络。图像尺寸相同也不保证 RGB/IR 像素严格对齐。进入双模态外观实验前，需要分别确认配对帧、原生裁剪和配准偏差，并将 ±1、±2 原图像素扰动作为敏感性实验。没有这些验证，不应直接把 RGB 框从 IR 图像裁出后视为可靠身份信息。

后续缓存应显式记录两路独立的特征、有效掩码、原始检测索引、模型哈希和裁剪/配准参数。两个编码器的特征不一定处于同一向量空间，不能直接互算余弦距离；先在各自空间校准距离和质量，再设计融合。

后续时序记忆建议分别保护长期身份样本和近期适应样本：低质量观测可参与当帧匹配，但不要立即覆盖长期模板；身份恢复置于新轨迹出生之前，并保留当前 30 帧删除上限作为第一组控制条件。延长缓存、重新激活已删除轨迹、历史补帧应分别消融，不能与特征融合一次性合并后归因于 M3。

## 4. 本分支的验证与结论边界

纯数值测试覆盖空集合、IoU 匹配、置信度边界、特征缓存哈希、重复索引拒绝、零向量排除、低分连续片段、输出中断，以及可分离的合成正负身份特征。安装 OpenCV 后，还验证纹理不足时不会把单位矩阵当作有效估计，以及已知平移的像素单位转换。

这些测试证明诊断代码的关键机制符合定义，不证明真实海上场景的跟踪精度提升。真实数据的总体与分组指标、速度和失败案例需要在本地数据与原运行环境中实际执行后填写。

实现参考：[BoxMOT v16.0.11 SOF](https://github.com/mikel-brostrom/boxmot/blob/v16.0.11/boxmot/motion/cmc/sof.py)、[BaseCMC 图像预处理](https://github.com/mikel-brostrom/boxmot/blob/v16.0.11/boxmot/motion/cmc/base_cmc.py)。
