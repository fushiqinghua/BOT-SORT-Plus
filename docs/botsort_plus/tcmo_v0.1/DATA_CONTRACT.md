# TCMO 数据合同 v0.1（实现前规范）

## 预测输入
`prediction_bundle.json`：schema、base commit、protocol sha、primary detector/reid/source protocol SHA、代码 pin、序列长度、V/T model weight SHA、训练与校准 receipt SHA、所有 image/cache/mapping sha、路径、device、runtime、source flags。严格 allowlist；禁止 GT、GT ID、评测器和 reference prediction 路径进入 child。校准模型参数可进入 bundle，离线带 GT 的拟合样本不可进入。评测 bundle 单独保存。旧 R4.1 prediction bundle 的校验保留，不能给历史 bundle 伪造新字段。

`aux_cache/{V,T}/<sequence>.jsonl.gz`：一行/输入帧（包含空 frames）；frame 1..N，frame_id、sequence、modality、source_image_relpath/decoded_sha256、native_shape=[height,width]、raw detections 列表。每 row={row_id:0..n−1, xyxy:[x1,y1,x2,y2], score, class_id, q(optional only after lock)}；有限数、正面积、明确 float32/float64 与 score endpoints，源坐标为逆 letterbox 后原始像素，保留全部导出行、顺序不变。主 F cache 不改。model/config/NMS/input-size/score-floor/normalization/thermal input format/weights/paired frame mapping 全写 manifest；路径在私有环境，公开摘要只用占位符。q 可由冻结模型 runtime 计算或另存 sidecar，以原 row key 对齐，不得覆盖 raw score。
`mapping.json`：source=T,target=V, direction=IR_TO_RGB，model type/3x3 matrix or per-frame transforms、native shapes、同坐标粗配准声明和证据、适用范围、有效性；T 坐标点齐次投影、有限/分母/边界检查。A0 保留原框、mapped_corners_envelope、mapped_center、clip_fraction；中心映射不能简单从透视框 envelope 推定；中心 proxy 用当前轨迹预测宽高，独立记录。C2 加在 mapped center/box 之后，仅变更空间，另记 shifted true。缺模态/配对失败不得将缺失视作背景负样本；完整实验要求所有帧具备输入或明确不可用原因且区分缺文件与无检测。

## R0 快照（GT-free）
`risk/<seq>.jsonl.gz`：frame、start confirmed active id/state/last_frame、W 后剩余 lost id、normal accepted(stage,primary_det_index)、consumed/reserved source rows、W proposed/committed/rejected、anchor mean/P/observed/ready/spent、selected/backup mean/P/xyxy 与 gmc guard、raw history（max3、因果常规）、protected boxes、所有 auxiliary raw/mapped候选 key/分组/可用性、每条边 geometry/q/cost/拒绝码/实际一对一选择；M0 最终输出与正式状态 fingerprint。GT 不在此文件。
`labels/<partition>/<seq>.jsonl.gz` 离线追加：严格历史 anchor GT id + support、current valid RGB/IR GT keys/boxes、reference covered、同身份 native overlap/other overlap、mapped raw/proxy/posterior IoU、identity_label(same/other/unknown/ignored)、GT 忽略原因、human review pending/confirmed、opportunity 独立字段。跨模态相同 ID 须先验核验同步与标注定义。unknown 不自动升格为负例；IR-only visible no RGB label 单独计，不并入 RGB FN。一个 GT key=(seq,frame,gt_id) 去重；一条边=(seq,frame,track_id,source_key)，与一对一采用、事件及 episode 数量分开。

## 闭环事件
`tcmo_events/<seq>.jsonl`：proposed/committed/rejected_finalization 等，frame、track_id、source_key（V/T 原 row）、group_key/members、mapping/score/q、几何（selected/anchor/backup/proxy）、prior mean/P 摘要、measurement z/R、posterior mean/P 摘要、old/new state、output_box、spent before/after、rejection reason、source_observed=true、state_committed bool。TCMO 不伪装原生 match/regular history，primary_det_index 仅 F 用。辅助 det_ind 的 sentinel 语义须明确测试，不允许隐式 Python 负索引。actual output 使用相同 MOT 10 列格式/同原生导出规范；独立日志追踪关联来源，原生 events/状态摘要与增强输出区分。
C1 新增 row 单独保存 `combined_row_index→source_key`、source score 与 native input score、真实 RGB feature validity；F 索引逐字节保留。
完成文件 manifest 保存所有相对路径、大小、sha256、代码/输入指纹、阶段 status。公开产物脱敏后另记发布 hash，保留原始 hash 与字段说明。
