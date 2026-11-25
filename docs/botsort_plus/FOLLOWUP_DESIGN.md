# B 恢复、红外位置修正与 C2 记忆的设计

各配置复用冻结 YOLO11-M RGB–红外检测、微调 RGB ReID 和 SOF 相机运动补偿。原生 BoT-SORT 高分、低分和新轨检测门槛分别为 0.50、0.10 和 0.60。以下机制由配置或运行参数启用。指标、逐序列差值及典型失败证据见 [迭代实验归档](../../results/vttiny/iterations/v0.3-c2-identity/README.md)。

## B 的短时恢复与预测输出

- **两次观测提前接回旧 ID。** 丢失轨迹与候选的最近两次观测必须相邻，两次都通过已有检查。`moderate` 要求最差几何代价 ≤0.42、最差 RGB 外观代价 ≤0.55。双向竞争、总成本门限和一对一分配继续约束恢复。证据不足的候选继续积累原三次观测。 [恢复实现](../../botsort_plus/recovery.py) · [配置](../../configs/botsort_plus/B_early_moderate.json)
- **最多两帧运动外推。** 最近三次实测输出连续，检测最高分 ≥0.55、均分 ≥0.35，且后两帧的预测与测量 IoU ≥0.35。断测后使用 Kalman 与 SOF 路径的预测框。图像边界与已输出框的重叠检查限制外推。预测是当前帧的一条输出，后续匹配仍使用实测历史。 [外推实现](../../botsort_plus/prediction_output.py) · [配置](../../configs/botsort_plus/B_early_motion2.json)

## 跟踪阶段的原始红外信息

- **存在性与关联消融。** 从当前红外图块和两次历史实测图块提取局部对比度、极性及空间模板证据。在原有空间门限内，对代价接近的高分匹配或 B 恢复候选重排。外推否决、早确认否决、只记录和错位红外各有单独配置。区域配准或图块证据不足时，沿用原决定。 [红外观测](../../botsort_plus/thermal_observer.py) · [存在性判定](../../botsort_plus/spatial_presence.py) · [配置与结果](../../results/vttiny/iterations/v0.3-c2-identity/04-raw-infrared/README.md)
- **位置修正。** 从最近两次连续实测建立图块模板，在当前外推框附近搜索相对位移。合格位移须通过相似度、峰值间隔、两段历史一致性、对比度和邻近实测目标检查。将估计位移乘以 0.5，限制在 1.75 像素及按短边计算的限幅内，只平移外推框中心。主规则在按原视频划分的训练窗口上锁定。追加的 RGB 优先配置对两路运行同一规则，RGB 有可信位移时采用 RGB，否则采用可信的红外位移，两侧均缺乏支持时保留原框。 [在线接入](../../botsort_plus/online_position.py) · [匹配与位移](../../botsort_plus/position_localizer.py) · [锁定参数](../../configs/position/locked_policy.json)

## C2 的确认记忆与连续低分预热

**初版 C2** 在高分和待确认关联中依据历史外观一致性与竞争间隔校准原有的外观加成。接受跟踪关联与允许写入身份记忆分别判定。独立消融显示，B 恢复库持续拒绝正确的近期观测后，恢复候选常被陈旧的 RGB 原型判为外观冲突。 [初版设计](C2_DESIGN_AND_REVIEW.md) · [控制变量和逐帧归因](../../results/vttiny/iterations/v0.3-c2-identity/02-c2-attribution/README.md)

**确认记忆** 让原生 BoT-SORT 外观模板按正常逻辑更新，C2 私有长期锚点采用严格写入。对 B 恢复外观库，已确认观测照常写入。被判为历史冲突的观测也可通过近期一致性确认，条件是同轨相邻两次均为高分匹配、图块质量合格、两帧几何代价 ≤0.50，且有竞争时几何优势 ≥0.10，两个 RGB 特征的余弦距离 ≤0.40。重激活打断连续支持。 [记忆接入](../../botsort_plus/identity_adapter.py) · [配合提前恢复的配置](../../configs/botsort_plus/B_confirmed_early.json)

**连续低分预热** 在 `identity.stable_low_warmup` 开启时运行。私有锚点少于 3 次有效更新、当前低分续跟被判为 `warmup_requires_high` 时，检查同一轨迹是否连续两帧满足以下条件。

1. 预测与检测的 IoU ≥0.70。有可行竞争配对时，当前配对比同轨迹或同检测的最佳替代配对至少高 0.10 IoU。
2. 两帧均有有效 RGB 特征，图块质量达到 C2 门槛和 B 外观库门槛，两者在本配置中均为 0.35。
3. 从第二个连续合格帧起，只把当前 RGB 特征写入 B 恢复库。高分匹配、重新激活、几何或质量不合格以及断帧都会中断连续条件。

`tools/audit_botsort_plus_identity.py` 按阶段、帧连续性、几何、质量、写入模态和私有锚点写入声明核对日志。生命周期的实测输出计数按 `motion_output` 事件剔除预测框。 [低分预热配置](../../configs/botsort_plus/confirmed_stable_low.json) · [单段及全量结果](../../results/vttiny/iterations/v0.3-c2-identity/07-low-score-warmup/README.md)

## 前向回放

使用原实验的 VT-Tiny-MOT 双模态帧目录、冻结检测缓存、逐检测 RGB 特征缓存和 BoT-SORT 运行环境。环境路径以参数传入，检测缓存默认从仓库中的 RGB–红外基线读取。位置修正使用同一序列中配对的 `00` 和 `01` 图像，策略与配准文件的 SHA-256 由运行器核对。

```bash
for config in B_early_motion2 final_c2_confirmed confirmed_stable_low; do
  python tools/run_botsort_plus.py \
    --config "configs/botsort_plus/$config.json" \
    --images-root "$VTTINY_FRAMES" --features-dir "$RGB_FEATURE_CACHE" \
    --runtime-root "$TRACKING_RUNTIME_ROOT" --output-dir outputs/replay_c2 \
    --position-policy configs/position/locked_policy.json \
    --position-registration configs/position/registration.json --position-rgb-first --trace
done
```

归档中的 `rgb_then_ir` 对应上述 `B_early_motion2` 配置加位置参数。`final_c2.json` 用于初版完整 C2 的同基座对照。单段回放追加 `--sequences DJI_0327_3` 并使用新的输出路径。运动外推本身通过 `recovery.output_prediction_frames` 控制。只做红外位置修正时省去 `--position-rgb-first`。免训练红外的 `ir_*.json` 配置使用各自的 `--ir-tiles` 和 `--ir-registration`。

深度归因目录中的控制脚本沿用初版 C2 的实验布局，基准代码为 `ba1276aba66797de9d65fb27fe33020c5ace444e`。脚本与说明中的 `$EXPERIMENT_ROOT` 对应原 `experiments` 根目录，`$VTTINY_ROOT` 对应 VT-Tiny-MOT 项目根目录。

## 用已归档轨迹复算三组对照

准备 [官方 RGB 真值](https://github.com/xuqingyu26/HGTMT) 的 `GT/GT_00/test/<序列>/gt/gt.txt`，以及提交为 `12c8791b303e0a0b50f753af204249e622d0281a` 的 TrackEval。环境依赖见 [requirements-eval.txt](../../requirements-eval.txt)。

```bash
RUN=results/vttiny/iterations/v0.3-c2-identity/07-low-score-warmup/full
python tools/evaluate_botsort_plus.py --run-dir "$RUN" \
  --trackers rgb_then_ir final_c2_confirmed confirmed_stable_low \
  --reference-tracker rgb_then_ir --gt-root "$VTTINY_TEST_RGB_GT" \
  --trackeval-root "$TRACKEVAL_ROOT" --output-dir outputs/recomputed_c2_full
```

单段复算将 `full` 换成 `pilot`，三组名称改为 `off both confirmed_stable_low`，参考用 `off`。结果可按 `tracker,sequence` 和归档中的 `metrics/summary.json` 对齐。测试集参与检测器和 ReID 选模、规则迭代与变体比较，当前结果按官方测试集开发实验记录。 [重新评分与日志审计](../../results/vttiny/iterations/v0.3-c2-identity/07-low-score-warmup/publication_validation.json)
