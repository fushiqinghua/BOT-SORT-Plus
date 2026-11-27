# 文本归档与校验

`T0/` 保存冻结输入、运行版本、参考轨迹以及抽查和全量一致性证据。`T1/` 保存离线标记、几何可达性和原始 160 例抽样。`review/` 保存图卡索引、逐帧结论、模型辅助判断、待人工复核表和统计。逐帧复核以 `case_id` 对齐，`sample_scope` 标明 `original` 原抽样或 `focus` 两段重点序列的补样。`T1/review_sample.json` 中的 `SAMPLE_DRAWN_NOT_REVIEWED` 是抽样时的历史状态；最新结果以 `review/review_summary.json` 和两个复核表为准。

`review/case_index.csv` 的 `annotated_card` 和 `blind_card` 均相对于本地压缩包的根目录。`current_source` 相对于数据集根目录。GitHub 仅保存文本记录，图卡 PNG、源图、盲审拼图和整包 ZIP 留在本地。文档中用案例 ID 查找原本地图片，编号和源像素校验值用于跨环境核对。

归档把原运行环境的绝对路径改成 `${VT_TINY_PROJECT_ROOT}` 或 `${PYTHON_EXECUTABLE}`。所有数据、模型、图像和运行哈希保留原值。`published_data_manifest.json` 中 `source_sha256` 来自原始归档字节，`published_sha256` 对应 GitHub 归档字节，`transformation` 描述差异。T0/T1 的原始服务端传输哈希已在整理前复核。图卡的本地文件哈希清单和 ZIP 传输校验值保留在本地。`T0/runtime_manifest.json` 和 `review/build_verification.json` 的脚本哈希描述生产这次归档的实际执行源文件，可供和参数化发布脚本区分。

`MODEL_REVIEWED` 是模型辅助完成，`NO_NATIVE_ASSIGNMENT` 是原生低分阶段无已分配候选。`PENDING_HUMAN_REVIEW` 表示人工复核待进行。两批图卡各自计数，补样不提供总体估计。几何可达性和图像复核只用于离线诊断，本轮未训练网络，也未运行修正框接管轨迹的在线评估。
