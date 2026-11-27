"""Generate the Chinese experiment report from the completed audits and pooled TrackEval results."""
import csv
import math
import statistics
from collections import Counter
from settings import BAM, CAMERA, QARMS, RESULT, read


def rows(name):
    with (RESULT/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def one(table, **where):
    select=[r for r in table if all(r[k]==v for k,v in where.items())]
    assert len(select)==1,where
    return select[0]


def mdtable(head,values):
    line=lambda fields:'| '+' | '.join(map(str,fields))+' |'
    return '\n'.join([line(head),line(['---']+['---:']*(len(head)-1)),*(line(v) for v in values)])


def fraction(r,h):
    return f"{r[f'retain_{h}_yes']} / {r[f'retain_{h}_n']}"


def main():
    inp=read(RESULT/'input_audit.json'); checks=read(RESULT/'selfcheck.json')
    assert inp['history_native_exact']==30 and checks['status']=='PASS'
    pool=rows('metrics/comparison.csv'); by=rows('tracking_by_sequence.csv')
    cam=rows('perspective_by_sequence.csv'); quad=rows('perspective_quadrants.csv')
    cost=read(RESULT/'bidirectional_cost_audit.json')
    lows=rows('crowded_low_followup.csv'); strata=rows('crowded_low_bias_strata.csv')
    bot=one(rows('bottom_order_training_proxy.csv'),split='check',group='all')
    gaps=rows('official_gap_calibration.csv'); recovery=rows('recovery_identity.csv')
    params=read(RESULT/'q_policy.json')
    names={'native':'原始 BoT-SORT','affine_forward':'③ 仿射候选，正向',
        'homography_forward':'③ 加单应候选，正向','homography_bidir':'③ 同矩阵，双向',
        'bam_iou':'④ 仅按当前 IoU 降权','bam_overlap':'④ 上帧重叠面积降权',
        'bam_bottom':'④ 底边筛选后降权','q_constant':'⑤ 训练常数 Q','q_motion':'⑤ 训练运动相关 Q'}
    methods=['native',*CAMERA,*BAM,*QARMS]
    assert {r['tracker'] for r in pool}==set(methods)
    metrics=[]
    for m in methods:
        r=one(pool,tracker=m)
        metrics.append([names[m],*(f"{float(r[k]):.3f}" for k in ('HOTA','AssA','IDF1','MOTA')),
            *(r[k] for k in ('FP','FN','IDSW'))])
    hrows=[r for r in cam if r['method']==CAMERA[1]]
    sums=lambda k:sum(int(r[k]) for r in hrows)
    f=one(pool,tracker=CAMERA[0]); h=one(pool,tracker=CAMERA[1])
    delta=lambda a,b,k:float(a[k])-float(b[k])
    quadrant=[statistics.median(math.hypot(float(r[k+'_dx']),float(r[k+'_dy'])) for r in quad)
              for k in ('reference','homography')]
    counts=cost['totals']
    native={r['sequence']:r for r in by if r['tracker']=='native'}
    changes={r['sequence']:int(r['IDSW'])-int(native[r['sequence']]['IDSW']) for r in by
             if r['tracker']=='bam_overlap' and r['sequence']!='COMBINED_SEQ'}
    distribution=Counter((d>0)-(d<0) for d in changes.values())
    base=one(lows,cohort='common',method='native')
    area=one(lows,cohort='common',method='bam_overlap')
    bias_rows=[]
    for group,label in [('det_farther_than_native_prior','检测比先验偏得更远'),('det_at_least_as_close','检测至少与先验一样近')]:
        for m in ('native','bam_overlap'):
            r=one(strata,native_observation=group,method=m)
            bias_rows.append([label,names[m],r['post_error_n'],f"{float(r['det_error_mean']):.3f}",
               f"{float(r['post_error_mean']):.3f}",fraction(r,3),
               f"{float(r['shared_error_3_mean']):.3f}（{r['shared_error_3_n']}）"])
    qrows=[]
    for m in ('native',*QARMS):
        g=one(gaps,cohort='common',age='11-30',method=m)
        r=one(recovery,cohort='common',method=m)
        qrows.append([names[m],f"{100*float(g['covered_mean']):.2f}%",f"{float(g['area_median']):,.1f}",
          f"{float(g['nll_mean']):.3f}",f"{float(r['post_error_mean']):.3f}",*(fraction(r,h) for h in (3,5,10))])
    gap=one(gaps,cohort='common',age='11-30',method='native')
    rec=one(recovery,cohort='common',method='native')
    rm=one(recovery,cohort='common',method='q_motion')
    audited=rows('event_denominators.csv')
    correct_recoveries=lambda m:sum(int(r['same_gt'] or 0) for r in audited if r['method']==m)
    keep_delta=[int(rm[f'retain_{h}_yes'])-int(rec[f'retain_{h}_yes']) for h in (3,5,10)]
    mb=one(pool,tracker='bam_overlap'); mq=one(pool,tracker='q_motion')
    a17=one(hrows,sequence='DJI_0173_2'); a27=one(hrows,sequence='DJI_0277_2')
    s27=one(by,tracker=CAMERA[1],sequence='DJI_0277_2')
    index=lambda name,group:one(strata,method=name,native_observation=group)
    off='det_farther_than_native_prior'; close='det_at_least_as_close'
    offset_native,offset_bam=index('native',off),index('bam_overlap',off)
    near_native,near_bam=index('native',close),index('bam_overlap',close)
    text=f"""# VT-Tiny-MOT 上三个原生组件扩展的首轮消融

上帧重叠面积降权取得本轮最高的 HOTA 和 IDF1。单应矩阵在部分帧改善背景配准，双向 IoU 在当前门限下没有带来额外跟踪收益。学习运动相关的 Q 明显扩大了失联覆盖范围，原 ID 的长期保持收益很小。

## 完整 30 个测试序列

使用 {inp['frames']:,} 帧、冻结的 YOLO11-M RGBT 检测与 RGB ReID 缓存。按官方 RGB 坐标的真值合并所有类别，采用 TrackEval COMBINED_SEQ。基线的 30 份轨迹逐字节复现历史结果。每组的 3 个先导序列均复现了该组全量运行的相应输出。百分比指标按 0–100 计。

{mdtable(['方法','HOTA','AssA','IDF1','MOTA','FP','FN','IDSW'],metrics)}

官方测试集曾参与此前的模型及门限选择，这些数值用于同一开发数据上的对照。全量逐序列结果见 [tracking_by_sequence.csv](tracking_by_sequence.csv)，相对原生的各项差值见 [comparison.csv](metrics/comparison.csv)。

## ③ 背景配准与双向关联

全量 {inp['frames']:,} 帧中，单应候选组在 {sums('homography')} 帧选择单应、{sums('affine')} 帧选择替代仿射、{sums('native'):,} 帧保留原生。单应被选的帧中，{sums('h_frames_opposite')} 帧至少有两个象限的参考残差方向相反。在 {len(quad)} 个有效帧—象限样本上，象限残差向量长度的中位数从 {quadrant[0]:.3f} 降至 {quadrant[1]:.3f} 像素。该统计描述通过筛选的背景点，无法据此识别飞机的具体姿态。

增加单应候选相对仅仿射使 HOTA 变化 {delta(h,f,'HOTA'):+.3f}、IDF1 {delta(h,f,'IDF1'):+.3f}、换号 {int(delta(h,f,'IDSW')):+d}。DJI_0173_2 有 {a17['native']} / {a17['frames']} 帧保留了原生几何。DJI_0277_2 有 {a27['homography']} 帧选中单应，相对原生 HOTA 变化 {delta(s27,native['DJI_0277_2'],'HOTA'):+.3f}，换号变化 {int(delta(s27,native['DJI_0277_2'],'IDSW')):+d}。全量 HOTA 仍低于原生。

双向代价核验重放了全部 {cost['probe_exact_output_and_geometry']} 段。启用替代矩阵的关联调用中，有 {counts['pairs']:,} 对候选，{counts['pairs_delta_gt_1e_6']:,} 对的几何代价改变超过 0.000001，{counts['pairs_delta_gt_0_01']:,} 对超过 0.01，最大绝对差 {cost['max_abs_cost_delta']:.4f}。有重叠的候选行中，几何最优列变化 {counts['row_best_candidate_changed']} 次。正向、双向的最终轨迹在 {len(cost['forward_bidir_identical_sequences'])}/30 段逐字节一致。原生高分阶段还使用外观与门限，几何代价的变化不一定改变输出关联。[配准分序列明细](perspective_by_sequence.csv) · [双向代价核验](bidirectional_cost_audit.json)

## ④ 拥挤时的低分观测更新

训练侧留出组有 {int(bot['n']):,} 对上一帧重叠且底边相差超过 5 像素、下一帧双方仍有配对的样本。底边较高一方的下一帧归一化中心误差更大的比例是 {100*float(bot['fraction_back_worse']):.1f}%。这些数据没有逐实例的遮挡前后标注，测试中因此分开检验重叠覆盖与底边筛选。重叠面积版相对原生的误报、漏检、换号变化分别是 {int(mb['delta_FP']):+d}、{int(mb['delta_FN']):+d}、{int(mb['delta_IDSW']):+d}。{distribution[-1]} 段换号下降，{distribution[0]} 段持平，{distribution[1]} 段上升，其中 DJI_0061_2 少了 {-changes['DJI_0061_2']} 次。全量 HOTA 增量为 {float(mb['delta_HOTA']):+.4f} 个百分点。

四种方法共同正确匹配的 {base['post_error_n']} 次拥挤低分事件中，原 ID 连续保持至第 +1 帧从 {fraction(base,1)} 变为 {fraction(area,1)}，保持至第 +3 帧从 {fraction(base,3)} 变为 {fraction(area,3)}。下面进一步根据原始低分观测是否比原生预测更偏，比较同一事件。中心误差单位为原图像素。最右列要求四种方法在窗口内均保住原 ID，括号给出共同样本数。

{mdtable(['原生事件分层','方法','事件','原始框中心误差','更新后中心误差','持续至 +3 帧','+3 帧中心误差'],bias_rows)}

在检测中心比原生先验更偏的 {offset_native['post_error_n']} 次事件中，降权将更新后平均中心误差从 {float(offset_native['post_error_mean']):.3f} 降到 {float(offset_bam['post_error_mean']):.3f} 像素，第 +3 帧多保住 {int(offset_bam['retain_3_yes'])-int(offset_native['retain_3_yes'])} 次原 ID。在检测中心至少与先验一样近的 {near_native['post_error_n']} 次中，当帧平均误差从 {float(near_native['post_error_mean']):.3f} 上升至 {float(near_bam['post_error_mean']):.3f} 像素。低分降权在这两类观测上的定位影响方向相反。同一目标可以产生多次相邻事件，原 ID 事件差与全量换号数分别计数。[低分偏框明细](crowded_low_bias_strata.csv) · [全部事件分母](event_denominators.csv)

## ⑤ 运动相关过程噪声

在 {len(params['sequences']['fit'])} 段官方训练来源序列的 {int(params['samples']['fit']):,} 个终点样本上拟合。训练侧留出 {len(params['sequences']['check'])} 段、{int(params['samples']['check']):,} 个终点。筛选后两组分别来自 {len(params['groups']['fit'])} 和 {len(params['groups']['check'])} 个母视频。学习模型只输出平面加速度方差。常数模型的截距达到优化上界，运动模型的速度与尺寸系数均接近上界，表中另列椭圆面积以反映扩大覆盖的代价。

下表的覆盖取真实失联 11–30 帧的 {int(gap['covered_n']):,} 个共同预测帧，面积为 95% 椭圆中位数。找回部分以 {rec['post_error_n']} 个共同正确接回原 GT、使用相同断前和找回检测的事件为起点，按 GT 能否持续标注到对应未来帧确定分母。

{mdtable(['方法','95% 椭圆覆盖','面积 px²','平均 NLL','找回后中心误差 px','原 ID +3 帧','原 ID +5 帧','原 ID +10 帧'],qrows)}

运动 Q 改善了不确定性校准，原 ID 在三个时间点的净差分别为 {keep_delta[0]:+d}、{keep_delta[1]:+d}、{keep_delta[2]:+d}。原生可核实的同 GT 找回共有 {correct_recoveries('native')} 次，运动 Q 为 {correct_recoveries('q_motion')} 次。扩大协方差会改变再观测时的卡尔曼增益。原生 BoT-SORT 依靠 IoU 和外观进行关联，扩大椭圆本身不会增加可配对候选。全量运动 Q 的换号变化 {int(mq['delta_IDSW']):+d}，漏检变化 {int(mq['delta_FN']):+d}，HOTA 与 IDF1 均略降。历史 D1 的 59.7% 来自 B_early_moderate 及当时的配对规则。本轮的原始 BoT-SORT 参考值由上表的共同事件重新计算。[官方失联覆盖](official_gap_calibration.csv) · [找回后的原 ID](recovery_identity.csv) · [拟合参数](q_policy.json)

## 文件与复现

方法、论文对应关系、训练数据筛选和运行命令见[实现说明](../README.md)。输入一致性见 [input_audit.json](input_audit.json)，数值和时序校验见 [selfcheck.json](selfcheck.json)。服务器实验根目录是 `${VT_PROJECT}/integrations/native_motion_345_v1`，各方法完整轨迹在 `full/trackers/<方法>/data/`，压缩逐帧记录在 `full/diagnostics345/<方法>/`。
"""
    (RESULT/'实验报告.md').write_bytes(text.encode('utf-8'))
    print('Report written')


if __name__=='__main__':
    main()
