"""Render a Chinese experiment report from completed Q/R result files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


METHODS = ('native', 'learned_q', 'learned_r', 'learned_qr')
LABELS = {'native': '原始 BoT-SORT', 'learned_q': '学习 Q', 'learned_r': '学习 R', 'learned_qr': '学习 Q＋R'}
R_LABELS = {'scale_proxy': '目标尺寸基准', 'score': '置信度与统一尺寸基准',
            'no_image': '置信度、几何和同帧检测环境', 'full': '再加入 RGB/IR 图像特征'}
Q_LABELS = {'native': '原始 Q', 'q_constant': '常量附加 Q', 'q_mlp': '学习附加 Q'}
STRATA = {'all': '全部', 'low': '低分'}


def read_csv(directory, name):
    with (directory/name).open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def read_json(directory, name):
    return json.loads((directory/name).read_bytes())


def index(rows, *fields):
    result = {tuple(row[k] for k in fields): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate result rows for {fields}')
    return result


def decimal(value, places=4):
    return '—' if value in (None, '') else f'{float(value):.{places}f}'


def signed(value, places=4):
    if value in (None,''):
        return '—'
    return f'{float(value):+.{places}f}' if float(value) else f'{0:.{places}f}'


def percent(value, places=1):
    return '—' if value in (None, '') else f'{100*float(value):.{places}f}%'


def count(value):
    return f'{int(value):,}'


def delta_count(value):
    value = int(float(value))
    return f'{value:+,}' if value else '0'


def md_table(columns, rows):
    return ['| '+' | '.join(columns)+' |', '|'+ '|'.join(['---']+['---:']*(len(columns)-1))+'|',
            *['| '+' | '.join(map(str, row))+' |' for row in rows], '']


def tracking_sections(directory):
    overall = index(read_csv(directory,'tracking_combined.csv'), 'tracker')
    sequences = read_csv(directory,'tracking_by_sequence.csv')
    fields = ('HOTA','DetA','AssA','IDF1','MOTA','FP','FN','IDSW')
    rows = []
    for method in METHODS:
        result = overall[method,]
        scores = [decimal(result[k]) for k in fields[:5]]
        events = [count(result[k]) for k in fields[5:]]
        rows.append([LABELS[method], *scores, *events])
    deltas = [[LABELS[m], *(signed(overall[m,]['delta_'+k]) for k in fields[:5]),
               *(delta_count(overall[m,]['delta_'+k]) for k in fields[5:])] for m in METHODS[1:]]
    lines = ['## 完整测试集跟踪结果', '',
             'TrackEval 按 `COMBINED_SEQ` 汇总。HOTA、DetA、AssA、IDF1、MOTA 为百分数，差值为百分点。', '',
             *md_table(['方法',*fields], rows),
             '相对原始 BoT-SORT 的差值。', '', *md_table(['方法',*['Δ'+k for k in fields]], deltas)]
    for method in METHODS[1:]:
        row = overall[method,]
        selected = [r for r in sequences if r['tracker']==method and r['sequence']!='COMBINED_SEQ']
        if len(selected) != 30:
            raise ValueError(f'{method} is missing a full sequence-level comparison')
        directions = [float(r['delta_HOTA']) for r in selected]
        losses = sum(v< -1e-9 for v in directions)
        gains = sum(v>1e-9 for v in directions)
        lines.extend([f'- **{LABELS[method]}**。HOTA 有 {gains} 段提高、{losses} 段下降、{len(selected)-gains-losses} 段持平。'
                      f'全量 FP {delta_count(row["delta_FP"])}、FN {delta_count(row["delta_FN"])}、换号 {delta_count(row["delta_IDSW"])}。'])
    lines.extend(['', '逐序列报告 HOTA 和 MOTA 降幅最大的序列，以及 FP、FN 增量最大的序列。', ''])
    worst_rows = []
    for method in METHODS[1:]:
        selected = [r for r in sequences if r['tracker']==method and r['sequence']!='COMBINED_SEQ']
        for metric, sign in (('HOTA',1),('MOTA',1),('FP',-1),('FN',-1)):
            key = 'delta_'+metric
            outcomes = sorted((r for r in selected if sign*float(r[key]) < -1e-9),
                              key=lambda r: (sign*float(r[key]),r['sequence']))
            if outcomes:
                row = outcomes[0]
                effect = signed(row[key]) if metric in fields[:5] else delta_count(row[key])
                worst_rows.append([LABELS[method],metric,row['sequence'],effect,
                                   *[delta_count(row['delta_'+k]) for k in ('FP','FN','IDSW')]])
            else:
                worst_rows.append([LABELS[method],metric,'无','—','—','—','—'])
    lines.extend(md_table(['方法','指标','序列','该项差值','ΔFP','ΔFN','ΔIDSW'], worst_rows))
    lines.append('[全量指标](tracking_combined.csv) · [逐序列指标](tracking_by_sequence.csv) · [评分与输入审计](score_audit.json)')
    lines.append('')
    return lines


def localization_section(directory, audit):
    rows = index(read_csv(directory,'localization_combined.csv'), 'cohort','method','score_stratum')
    metrics = ('center_error_px','width_abs_error_px','height_abs_error_px','iou')
    lines = ['## 定位误差与低分检测', '',
             f'全体 TP 口径包含每组自身被 CLEAR 正确匹配的框。四组配对口径固定同一序列、帧、真值目标和冻结检测行，共 {count(audit["common_rows"])} 个匹配。'
             f'低分段为 {audit["low_score_threshold"]:g} < 分数 ≤ {audit["high_score_threshold"]:g}。', '',
             '中心误差采用欧氏距离，宽高使用绝对误差，单位为像素。以下为平均值。', '']
    for cohort,title in (('all_official_clear_tp','各组全部官方 TP'),('four_way_same_detection','四组同检测行的官方 TP')):
        lines.extend([f'### {title}', ''])
        values = []
        for score_band,label in STRATA.items():
            for method in METHODS:
                row = rows[cohort,method,score_band]
                measures = [decimal(row[k+'_mean'],5) for k in metrics]
                if cohort=='four_way_same_detection' and method!='native':
                    measures = [f'{value}（{signed(row[k+"_mean_delta_native"],5)}）' for value,k in zip(measures,metrics,strict=True)]
                values.append([label,LABELS[method],count(row['n']),*measures])
        lines.extend(md_table(['样本','方法','匹配数','中心误差 ↓','宽度误差 ↓','高度误差 ↓','IoU ↑'], values))
    links = ['[全量定位统计及细分置信度](localization_combined.csv)', '[逐序列定位](localization_by_sequence.csv)']
    if (directory/'localization_paired_rows.csv.gz').is_file():
        links.append('[共同检测行明细 CSV 压缩包](localization_paired_rows.csv.gz)')
    lines.extend(['括号为同一批匹配相对原生组的配对差值。四组交集用于比较定位质量，评估 FP、FN 和身份保持仍使用全量结果。', '',
                  ' · '.join(links), ''])
    return lines


def r_section(directory):
    rows = [r for r in read_csv(directory,'r_low_bias.csv') if r['split']=='check']
    calibration = index(read_csv(directory,'r_holdout_calibration.csv'), 'method','split','score')
    summary = index(rows,'method','size')
    metadata = read_json(directory,'r_policy.json')
    full = summary['full','all']
    q75 = decimal(full['bias_cut_fit_norm_center'],4)
    all_row = calibration['full','check','low']
    constrained = all_row['parents']
    lines = ['## R 的低分偏框风险', '',
             f'内部留出使用训练集中的 {constrained} 个独立母视频，对低分检测的归一化中心误差做排序。'
             f'大偏框由拟合集低分样本的第 75 百分位定义，阈值 {q75}。'
             f'留出低分共 {count(full["n"])} 个检测，其中 {count(full["large_bias"])} 个超过阈值。', '']
    data = []
    for method,label in R_LABELS.items():
        outcome = calibration[method,'check','low']
        sizes = [summary[method,size] for size in ('all','short_lt8','short_ge8')]
        if any(s['n']!=summary['full',s['size']]['n'] or s['large_bias']!=summary['full',s['size']]['large_bias'] for s in sizes):
            raise ValueError('R arms are compared on different internal detection cohorts')
        data.append([label,*[decimal(s['roc_auc']) for s in sizes],decimal(outcome['center_nll']),
                     percent(outcome['center_95_coverage']), decimal(outcome['center_95_area_median'],2)])
    lines.extend(md_table(['R 输入','低分 AUC ↑','短边 <8 px AUC ↑','短边 ≥8 px AUC ↑',
                           '中心 NLL ↓','95% 中心覆盖率','95% 椭圆中位面积 px²'], data))
    base = summary['no_image','all']['roc_auc']
    difference = signed(float(full['roc_auc'])-float(base)) if full['roc_auc'] and base else '—'
    tiny = summary['full','short_lt8']
    lines.extend([f'加入图像特征后，整体低分 AUC 相对无图像对照变化 {difference}。'
                  f'训练使用 {count(metadata["fit_rows"])} 个检测匹配，内部留出使用 {count(metadata["check_rows"])} 个。'
                  f'短边小于 8 像素的检测占低分留出 {count(tiny["n"])} / {count(full["n"])}。'
                  '样本需通过训练阶段的匹配质量筛选，因此这一 AUC 描述已配对检测的偏框风险。', '',
                  *r_uncertainty_section(directory, summary),
                  '[偏框分层 AUC 与样本数](r_low_bias.csv) · [R 校准明细](r_holdout_calibration.csv)', ''])
    return lines


def r_uncertainty_section(directory, summary):
    file = directory/'r_discrimination_uncertainty.json'
    if not file.is_file():
        return []
    report = read_json(directory,file.name)
    originals = {'model_sha256':'r_policy.json','input_audit_sha256':'r_training_inputs.json',
                 'frozen_discrimination_sha256':'r_low_bias.csv'}
    for field,name in originals.items():
        digest = hashlib.sha256((directory/name).read_bytes()).hexdigest()
        if report[field] != digest:
            raise ValueError(f'Paired AUC CI does not refer to the supplied {name}')
    comparisons = []
    for mode in ('full','no_image','score'):
        original = summary[mode,'all']
        value = report['pooled_methods'][mode]['auc']['value']
        if (report['sample_count'] != int(original['n']) or report['positive_count'] != int(original['large_bias'])
                or abs(value-float(original['roc_auc'])) > 1e-10):
            raise ValueError(f'Paired AUC CI and frozen discrimination disagree for {mode}')
    for mode in ('no_image','score'):
        entry = report['paired_contrasts']['full_minus_'+mode]['auc']
        low, high = entry['ci95']
        interval = f'[{signed(low)}, {signed(high)}]'
        note = '覆盖 0' if low <= 0 <= high else '区间全为正' if low > 0 else '区间全为负'
        comparisons.append([f'完整模型减{R_LABELS[mode]}', signed(entry['difference']), interval,note])
    n_parents = len(report['parents'])
    replicates = count(report['bootstrap']['replicates'])
    return [f'在固定模型参数下，以 {n_parents} 个留出母视频为单位进行 {replicates} 次配对自助重采样。'
            '每次为各模型选取同一批母视频。', '',
            *md_table(['同一批低分中的对照','ΔAUC','95% 置信区间','区间与零点关系'], comparisons),
            '[重采样与配对区间明细](r_discrimination_uncertainty.json)', '']


def q_section(directory):
    report = read_csv(directory,'q_holdout.csv')
    policy = read_json(directory,'q_policy.json')
    center = index([r for r in report if r['split']=='check' and r['block']=='center'], 'h','method')
    lines = ['## Q 的内部留出校准', '',
             f'Q 使用训练集母视频内的模拟观测缺口拟合。拟合窗 {count(policy["samples"]["fit"])} 个，留出窗 '
             f'{count(policy["samples"]["check"])} 个。下表比较中心预测的 95% 椭圆覆盖、面积和 NLL。', '']
    rows = []
    for horizon in sorted({int(n) for n,_ in center}):
        for method,label in Q_LABELS.items():
            r = center[str(horizon),method]
            rows.append([horizon,label,count(r['n']),r['sequences'],percent(r['coverage95']),
                         decimal(r['mean_area95'],2),decimal(r['nll'],3)])
    lines.extend(md_table(['缺口帧数','Q','留出窗','序列数','中心覆盖率','平均椭圆面积 px²','中心 NLL ↓'],rows))
    fit_summaries = []
    any_incomplete = False
    for method in ('q_constant','q_mlp'):
        r = policy['optimization'][method]
        converged = r['converged']
        any_incomplete |= not converged
        status = '达到收敛条件' if converged else '优化器未报告收敛'
        fit_summaries.append(f'- {Q_LABELS[method]}迭代 {r["iterations"]} 次，目标值 {decimal(r["objective"],5)}，{status}。')
    lines.extend(fit_summaries)
    lines.extend(['', '覆盖率应与 95% 标称值及椭圆面积一起看。同检测行的输出框误差衡量最终定位精度。'])
    if any_incomplete:
        lines.append('发生未收敛时，本轮在线结果对应运行时采用的实际有限迭代参数。')
    lines.extend(['', '[中心、宽高及联合校准](q_holdout.csv) · [优化参数与分组](q_policy.json)', ''])
    return lines


def recovery_section(directory):
    file = directory/'recovery_retention_four_way.csv'
    if not file.is_file():
        return []
    rows = index(read_csv(directory,file.name),'method','policy','horizon')
    lines = ['## 找回后原身份保持', '',
             '四组共用队列要求两端的匹配真值、帧号和冻结检测行一致。分母限于相应窗口内可核验的事件。', '']
    result = []
    for horizon in ('3','5','10'):
        for policy,label in (('endpoint','指定帧'),('continuous','全程')):
            subset = [rows[(method,policy,horizon)] for method in METHODS]
            if len({r['eligible'] for r in subset})!=1:
                raise ValueError('Unpaired recovery window denominators')
            result.append([horizon,label,count(subset[0]['eligible']),*[count(r['held'])+'（'+percent(r['retention_rate'])+'）' for r in subset]])
    lines.extend(md_table(['找回后帧数','判定','共同事件',*[LABELS[m] for m in METHODS]],result))
    lines.extend(['[各组自身接回事件](recovery_retention.csv) · [共同接回事件](recovery_retention_four_way.csv)', ''])
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=Path(__file__).resolve().parent/'results')
    args = parser.parse_args()
    directory = args.results_dir.resolve()
    audit = read_json(directory,'score_audit.json')
    if audit['status']!='PASS' or len(audit['sequences'])!=30 or audit['native_exact']!=30 or not audit['native_metric_parity']:
        raise ValueError('Expected verified full official test scoring and historical native parity')
    lines = ['# VT-Tiny-MOT 学习过程噪声 Q 与观测噪声 R', '',
             f'在官方测试集 30 段、{count(audit["frames"])} 帧上比较原始 BoT-SORT、仅学习 Q、仅学习 R 和同时学习 Q/R。'
             '四组共用检测、RGB ReID 和视频输入。原生轨迹与历史参考的 30 个文件逐字节一致。', '',
             *tracking_sections(directory),*localization_section(directory,audit),*r_section(directory),
             *q_section(directory),*recovery_section(directory), '## 样本与解释边界', '',
             'Q 和 R 用官方训练母视频划分拟合集与内部留出集。历史检测和 ReID 权重曾使用官方测试集进行选模与门限开发，'
             '因此 30 段全量结果属于当前冻结实验链路上的开发证据。模拟轨迹缺口、已匹配低分检测和四组共同 TP 分别回答不同的误差问题。', '']
    path = directory/'实验报告.md'
    path.write_text('\n'.join(lines),encoding='utf-8')
    print(path)


if __name__=='__main__':
    main()
