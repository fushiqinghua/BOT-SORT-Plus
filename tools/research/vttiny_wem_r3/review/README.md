# W R3 逐帧复核复现

从冻结的训练侧校准和留出结果重建 113 张图卡。校准组包含 29 张背景候选和 35 张身份候选，留出组分别包含 28 张和 21 张。背景和身份是抽样类型。图卡按传感器显示官方真值。红外面板上的弱框、预测框和历史观测框沿用 RGB 坐标供对照，可能出现模态偏移。

在上一级 `tools/research/vttiny_wem_r3` 目录运行下列命令，需要 Python 3.11 或更新版本和 Pillow。按上一级说明设置 `VT_WEM_SSH_TARGET`、`VT_WEM_REMOTE_PYTHON` 和 `VT_WEM_REVIEW_PROJECT_ROOT`。最后一项也可在取图时用 `--project-root` 指定。

```sh
python observer/transfer.py fetch full
python labels/transfer.py fetch full --detail
python transfer.py fetch full
python review/select_candidates.py
python review/collect_frames.py
python review/build_cards.py
python review/summarize_review.py
```

前三步取回冻结的观察器证明、离线标签、特征清单及校准锁。选择脚本比对校准锁中的标签文件摘要，再按序列、帧、旧 ID 和原始坐标去重。取图脚本核对冻结观测、候选原始行、官方标注索引、RGB–IR 配对及原图摘要。它读取训练序列的当前帧和最多三条过去实际观测。若需先核对流程，可为取图和制图步骤加 `--smoke`，生成四张核验图卡。

索引脚本读取发布结果中校准和留出两组的 `model_assisted.csv`，校验案例编号和原始观测键。默认在 `review/rendered/full/{calibration,holdout}/index.html` 建立本地图卡索引。每组另输出模型辅助表、按来源组统计和独立人工复核表。已有人工表中的记录会保留。所有复核结果的发布状态以结果目录说明为准。 [发布的复核结果](../../../../results/vttiny/wem_v0.1/r3/review/)

下载的原图、完整逐帧标注及图卡只写到本地生成目录 `review/inputs`、`review/assets` 和 `review/rendered`。这些目录由 `.gitignore` 排除。
