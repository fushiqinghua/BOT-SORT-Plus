# TCMO v0.1 方法设计（待实现、待实验）

## 1. 总体结构与可证伪假设
主检测 D_F=f_F(RGB,IR) 保持冻结，辅助 D_V=f_V(RGB)、D_T=f_T(IR) 分别读取自身模态。三个子模型可能误差相关，不做独立性假设。先执行原生关联/G/B/正常起轨、W0，TCMO 仅处理剩余有可靠历史的 confirmed-active 轨迹；最后统一输出整理。COM=Complementary Observation Mining；TRA=Trajectory-conditioned Residual Alignment；RMA=Reliability-aware Modality Association；UMU=Uncertainty-aware Measurement Update。首版 COM+基础 RMA+固定噪声中心量测；TRA/自适应 UMU 后续消融。科学假设是 IR 或 RGB 在融合检测缺少有效观测时提供**额外的时空证据**，而非所有红外都可靠。

## 2. COM：来源与轨迹条件
t 帧首活跃已确认集合 A_t，正常流程+W 已经获得观测集合 M_t；TCMO 集合 C_t=A_t\M_t，并且仍在本次 lost 列表、W anchor.ready、unspent、observed=t−1、anchor vs selected prior IoU≥.5。只处理当前刚失去常规观测的轨迹，不新建 ID。局部搜索可用中心协方差椭圆：(p−c_hat)^T(P_c+Σ0)^−1(p−c_hat)≤γ；R1 先用与 W 一致的有界几何搜索及下述 proxy IoU，不额外改变门。后续可预登记 Mahalanobis 门，避免协方差膨胀放任远候选。每轨迹最多5个去重候选组，固定稳定排序。保护普通匹配/B 预留、W 采用和所有正式输出；候选来源分别命名，不使用主检测原行号。跨模态重复组先在共同坐标中构造，歧义或与已占目标近重复的组拒绝；跨不同轨迹的 T 观测不得因轨迹依赖映射而复制物理候选键。

## 3. TRA：坐标与残差
所有几何与最终 MOT 在 RGB 像素域。粗映射 c_TV=π(H_TV[t] * [c_T,1]^T)，输入已粗配准且坐标声明一致时 H=I。对正常实测支持且双模态一对一明确的同轨迹，计算 δ_k=c_TV,k−c_V,k。A0=粗映射；A1=训练侧固定位置/尺度分组估计偏差 b(s,p)；A2=A1+近期残差中位数（残差定义 δ−b，不能重复减偏置）。仅过去可靠配对，建议初始窗口最近5帧至少3个有效配对，视实测分布在训练侧冻结；不使用 GT/未来/当前待补观测；过期、目标尺度剧变、视角改变和冲突时回退 A1 或 A0（同组明确记录），不假定稳定。若使用历史 GMC 转移残差，应使用位移的线性/局部雅可比而非平移项，并作为独立验证；GMC 不是跨相机标定。对配准偏差波动用 MAD 分量 1.4826*median(abs(δ−median δ))，方差加入训练侧噪声下限；样本少时 shrinkage/回退总体协方差。离线看同 ID IR native 与映射 RGB 的误差，不能仅按两框 IoU 判断物理身份。

## 4. RMA：共同的首版关联规则
三项证据分开：q=各模态校准的检测质量；几何=轨迹对应；R=坐标误差。校准标签基于各自 native GT，q 不是“该轨迹身份正确”的概率。R1 设计起点（不是已验证超参）q≥.25，映射中心与轨迹预测宽高构造 proxy box（所有 V/T 同样只使用中心），selected、anchor 和适用 backup 与 proxy IoU 各≥.5，和任何已拥有或保留候选的近重复 IoU≥.9 拒绝。跨源去重同时看真实映射框及中心接近性，保留原始 key 和所有候选成员；歧义保护不得把密集邻居连成一个虚假大目标。第一版用确定性一对一分组和保守冲突拒绝；具体分组阈值/并列处理要在训练侧锁定，不用 dev 选。代价 C=.5(1−IoU_proxy)+.5(1−q)，每行不匹配 dummy cost=.5，严格 C<dummy 才可分配，平局拒绝，Hungarian/linear_sum_assignment 一对一；不强制唯一候选被采用，不再设 ReID 硬门。top5 对应候选组而非每模态各5；同一 group/原始 row 不可多轨迹使用。C1 不使用轨迹条件，为直接合并基线，不冒充同一可行边集合。
后续 RMA 可用创新 ν=z−H_c x−，S=H_c P− H_c^T+R，d=ν^T S^−1ν；统一像素域的 NLL=.5d+.5logdet(S)，另限制绝对搜索范围与 R 最大值（高不确定性不等于可安全接受）。R1 不同时改评分、映射、R；先看行为变化。

## 5. UMU：中心量测与正式轨迹更新
确认 BoxMOT 固定源码的状态顺序 x=[cx,cy,w,h,vx,vy,vw,vh]^T，H_c=[I2,0]；正式先验已由原 KF/GMC 推进，不重复 predict。z 为真实辅助检测映射中心，R1 取原 KF 量测噪声中中心 2x2 block（从冻结 projected observation noise 源码提取，不直接用含 P 的创新协方差当 R），V/T 同一规则；若实现不提供单独 R，推导并对照原代码，禁止凭空猜测。K=P H_c^T(H_c P H_c^T+R)^−1；x+=x−+K(z−H_c x−)；Joseph P+=(I−KH_c)P(I−KH_c)^T+KRK^T；线性求解不显式求逆，检查对称/有限/PSD、正宽高和边界。中心量测不直接用红外宽高，但交叉协方差可间接改变尺度/速度；严格冻结宽高另作受约束增益实验，不在当前混用。
R2 自适应 R_T=J R_det^T J^T+Σ_align（近似不相关、同坐标误差时），更稳健可直接估计 mapped IR center−RGB GT center 的总残差协方差，不重复加总。R_V 来自同样训练侧中心误差。根据目标尺寸和配准可靠性拟合并留出检验，防止单一序列占主导。R 影响关联与更新的范围须标明；同一次预测只接纳一个模态真实量测，避免将高度相关的 F/V/T 重复更新。拒绝时保留原生状态。

## 6. 共享预算、事务与输出
W0 是已验证机制，TCMO 必须在同一帧的共同事务中保持 W 优先、只补 W 未成功提交且合格的轨迹。现有 W transact 内部会调用 native_preview 并赋值 active/lost/removed；不能先执行完整 W finalizer 再另一次 finalizer 或裸改 active。重构为保持 W 选择、逐案预演和 commit 语义的共同事务，先得到真实 W 成功集合，再处理剩余 TCMO；对每项 TCMO 在私有 track copy 上执行中心量测+原 reactivation 生命周期等价操作，核对其他轨迹的输出/列表/状态、旧 ID、ReID/记忆不变、正常事件/所有权不被覆盖；最终原生整理逻辑只正式提交一次，并原样保留后续输出记录/cleanup。失败 rollback 不能消耗预算、不能留下内存或时间戳副作用。不能直接以原 W 的 STrack.re_activate 全4维量测实现中心更新；需按 pinned BoxMOT 的 re_activate 语义复制生命周期动作并独立验证，禁止重复4维+2维更新。辅助键不可当作 all_dets/feature 缓存索引，必须单独追踪来源；任何输出 det_ind sentinel 都要审查后续消费者，不能用负索引读取末行。附加日志不加入 current_assignments 的 regular 事件集合。共用 W anchor.spent，一次弱更新/缺失过程；重新连续正常支持才重建，辅助量测不写 regular history、B 长期记忆、RGB ReID、TRA 可信残差。真实更新、状态时间、是否已输出和候选使用权原子化提交。off、shadow、force_reject 必须与原 M0 相同。
