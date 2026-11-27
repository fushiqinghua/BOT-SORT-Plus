"""Write a human-readable factorial comparison from official scores and the event audit."""
import csv
from joint_settings import COMBINED, COUNTS, FIX, MEASURES, METHODS, NATIVE, OVERLAP, read

LABELS={NATIVE:'原始 BoT-SORT',OVERLAP:'单独④ 上帧重叠面积降权',FIX:'长间隔重连修正版',COMBINED:'修正版＋④'}


def records(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def table(headers,rows):
    def line(vals):
        return '| '+' | '.join(map(str,vals))+' |'
    return '\n'.join([line(headers),line(['---']+['---:']*(len(headers)-1)),*(line(r) for r in rows)])


def display(value,metric,signed=False):
    if metric in COUNTS:
        return format(int(float(value)),'+,d' if signed else ',d')
    return format(float(value),'+.4f' if signed else '.4f')


def generate(directory):
    absolute={r['tracker']:r for r in records(directory/'tracking_combined.csv')}
    diff={r['contrast']:r for r in records(directory/'factorial_contrasts.csv')}
    per=records(directory/'sequence_contrasts.csv')
    by={r['sequence']:r for r in per}
    audit=read(directory/'verification.json')
    events=read(directory/'event_totals.json')
    hashes={r['sequence']:r for r in records(directory/'trajectory_hashes.csv')}
    assert audit['status']=='PASS'
    klist=('HOTA','IDF1','MOTA','FP','FN','IDSW')
    rows=[[LABELS[m],*(display(absolute[m][k],k) for k in klist)] for m in METHODS]
    contrast_labels={'overlap_on_native':'单独④ − 原始','fix_on_native':'修正版 − 原始',
                     'overlap_on_fix':'修正版＋④ − 修正版','combined_on_native':'修正版＋④ − 原始',
                     'interaction':'交互差值'}
    changes=[[label,*(display(diff[arm][k],k,True) for k in klist)] for arm,label in contrast_labels.items()]
    expect=int(diff['additive_expectation']['IDSW'])
    observed=int(absolute[COMBINED]['IDSW'])
    delta=diff['overlap_on_fix']
    def count_action(m):
        v=int(delta[m])
        return ('增加 '+str(v)) if v>0 else ('减少 '+str(-v)) if v<0 else '持平'
    if observed==expect:
        stacking='换号的净减量与两项单独收益的算术和相同。'
    elif observed<expect:
        stacking=f'组合比两项收益直接相加的预期又少 {expect-observed} 次换号。'
    elif int(delta['IDSW'])<0:
        stacking=f'④在修正版上仍减少换号，比完全叠加的预期少兑现 {observed-expect} 次。'
    else:
        stacking=f'在修正版上，④未带来额外的换号净减少。实际比直接相加的预期多 {observed-expect} 次。'
    def sequence_value(r,k):
        return int(r['overlap_on_fix_'+k])
    seqnames=['DJI_0030_2','DJI_0169_3','DJI_0061_2']
    for k in ('IDSW','FP','FN'):
        asc=sorted(per,key=lambda r:(sequence_value(r,k),r['sequence']))
        for r in [asc[0],asc[-1]]:
            if sequence_value(r,k) and r['sequence'] not in seqnames:seqnames.append(r['sequence'])
    focus=[[s,*[display(by[s]['overlap_on_fix_'+k],k,True) for k in ('HOTA','IDF1','FP','FN','IDSW')],
            '相同' if int(hashes[s]['fix_combo_byte_equal']) else '有变化'] for s in seqnames]
    seq_counts=[[k,sum(sequence_value(r,k)<0 for r in per),sum(sequence_value(r,k)==0 for r in per),
        sum(sequence_value(r,k)>0 for r in per)] for k in ('IDSW','FP','FN')]
    n=absolute[NATIVE]; f=absolute[FIX]; o=absolute[OVERLAP]
    text=f"""# 低分观测降权与 B＋自适应 GMC 长间隔重连修正版的组合消融

原始 BoT-SORT 出现 {n['IDSW']} 次换号，单独④为 {o['IDSW']} 次，修正版为 {f['IDSW']} 次，组合为 {observed} 次。{stacking} 相对修正版，组合的误报{count_action('FP')}，漏检{count_action('FN')}。

## 完整 30 段结果

每一组都重新回放官方测试集 {audit['frames']:,} 帧。固定 YOLO11-M RGBT 检测、RGB ReID 缓存和解码后的 RGB 图像，使用官方 RGB 坐标真值、所有类别合并的 TrackEval COMBINED_SEQ。HOTA、IDF1、MOTA 采用 0–100 标度。指标差值的单位为百分点，FP、FN、IDSW 的单位为个数。

{table(['方法','HOTA ↑','IDF1 ↑','MOTA ↑','FP ↓','FN ↓','IDSW ↓'],rows)}

{table(['比较','ΔHOTA','ΔIDF1','ΔMOTA','ΔFP','ΔFN','ΔIDSW'],changes)}

交互差值为“组合相对修正版的变化”减去“单独④相对原始的变化”。换号完全叠加时应有 {f['IDSW']} + {o['IDSW']} − {n['IDSW']} = {expect} 次，实际为 {observed} 次。HOTA 和 IDF1 为非线性汇总指标，其交互差值仅描述本轮对照。[四组完整指标](tracking_combined.csv) · [所有指标的交互差值](factorial_contrasts.csv)

## 换号、漏检与误报如何变化

按官方 CLEAR 匹配，将“同一序列、同一真值 ID、同一帧”视作同一换号位置。单独④使原始结果中的 {events['native_switches_removed']} 个换号位置消失，另外出现 {events['native_switches_added']} 个。在修正版上加④后，{events['fixed_switches_removed']} 个原换号位置消失，出现 {events['fixed_switches_added']} 个。两次添加④时消失的位置有 {events['same_removed_switch_frame_gt']} 个相同。单独④消除的位置中，有 {events['standalone_removals_already_absent_in_fix']} 个在加入④之前已不出现在修正版中。换号若改在另一帧发生，这一统计会各计一次消失和出现。[逐个换号位置与身份](switch_events.csv)

组合补回了修正版遗漏的 {events['fixed_misses_recovered']} 个真值目标帧，另有 {events['combined_new_misses']} 个真值目标帧变为漏检。按帧比较误报数量，误报减少的帧合计少 {events['fewer_FP_on_frames']} 个，增加的帧合计多 {events['extra_FP_on_frames']} 个。此处的误报按帧计数，同一帧内多个背景框的身份没有跨方法配对。[各序列事件明细](events_by_sequence.csv)

{table(['组合相对修正版','下降序列数','持平序列数','上升序列数'],seq_counts)}

## 典型序列

下表包含历史重连错案 DJI_0030_2、DJI_0169_3，上一轮单独④换号降幅最大的 DJI_0061_2，以及此次三项计数变化最明显的序列。各差值均相对修正版。

{table(['序列','ΔHOTA','ΔIDF1','ΔFP','ΔFN','ΔIDSW','轨迹文件'],focus)}

全量共有 {audit['fix_combo_MOT_files_exact']} 段修正版与组合版的轨迹文件逐字节相同。[30 段差值](sequence_contrasts.csv) · [各方法逐序列原始指标](tracking_by_sequence.csv)

## 实现和复核

修正版采用归档主版本 evidence_both，即 B_early_moderate＋自适应 GMC＋低分双重确认，并对长间隔的 B 恢复及原生高分找回执行观测与代价裕量检查。④复用前一轮 bam_overlap 的实现。仅在当前低分匹配通过关联、上帧观测确有重叠时，用“当前 IoU ×（1 − 上帧覆盖比例）”缩放卡尔曼均值的更新幅度。

组合共有 {audit['accepted_low_combo']:,} 次被接受的低分匹配，逐帧核实其中 {audit['weighted_updates']:,} 次应用了④，分布在 {audit['weighted_frames']:,} 个视频帧。两个修正版在全部 {audit['fixed_combo_camera_exact_frames']:,} 帧上使用的 GMC 矩阵逐字节一致。原始、单独④及修正版各 30 份轨迹与对应历史文件逐字节一致，重新计算的逐序列及总体指标也与历史结果一致。官方测试集曾用于前序选模和调参，本轮冻结规则，结果按开发性对照解读。[输入与复现核验](verification.json) · [配准及低分触发统计](camera_and_updates.csv) · [复现说明](../README.md)
"""
    (directory/'实验报告.md').write_bytes(text.encode('utf-8'))
