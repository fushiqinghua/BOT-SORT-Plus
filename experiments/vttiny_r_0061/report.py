"""Summarize the official and counterfactual evidence, including the 29-clip control."""
import csv
import gzip
from collections import Counter,defaultdict
from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from rf_settings import OUT,read,write,table

def rows(name):
    path=OUT/name
    op=gzip.open if path.suffix=='.gz' else open
    with op(path,'rt',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def timing(history,ledger):
    result=[]
    for near in rows('nearby_switches.csv'):
        if int(near['radius'])!=5:continue
        g,n,l=map(int,(near['gt'],near['native_frame'],near['low_frame']))
        pair={'native':ledger[g,'native',n],'low_r':ledger[g,'low_r',l]}
        hist=history[g]
        def endpoint(t,side):
            row=hist[t]
            return row['native_det'] and row['native_det']==row['low_r_det'] and all(
                row[m+'_tid']==pair[m][side+'_tid'] for m in pair)
        before=next((t for t in sorted(hist,reverse=True) if t<min(n,l) and endpoint(t,'old')),None)
        after=next((t for t in sorted(hist) if t>=max(n,l) and endpoint(t,'new')),None)
        counts={m:sum(int(hist[t][m+'_switch']) for t in hist if before is not None and after is not None and before<t<=after) for m in pair}
        result.append(dict(**near,common_before=before,common_after=after,
            one_transition_per_arm=bool(before and after and all(c==1 for c in counts.values()))))
    table(OUT/'transition_time_checks.csv',result)
    return result

def identity_figure(hota):
    methods=('native','low_r')
    tids={'native':{156:'#195784',172:'#8a3c67',248:'#b77027'},'low_r':{154:'#007970',167:'#b64828'}}
    fig,axs=plt.subplots(2,1,figsize=(9.4,4.45),sharex=True)
    for ax,m,title in zip(axs,methods,('Original KF','Low-score R')):
        for y,g in enumerate((142,139)):
            group=[r for r in hota if r['method']==m and int(r['gt'])==g]
            for r in group:
                ax.broken_barh([(int(r['frame'])-.5,1)],(y-.21,.42),
                    facecolors=tids[m].get(int(r['tid']),'#afb4bd'),linewidth=0)
        ax.set(yticks=[0,1],yticklabels=['GT 142','GT 139'],ylim=(-.5,1.6),xlim=(64,241))
        ax.set_title(title,loc='left',fontsize=11,fontweight='bold')
        ax.legend(handles=[Patch(color=c,label=f'ID {t}') for t,c in tids[m].items()]+[Patch(color='#afb4bd',label='Other ID')],
                  loc='upper right',ncol=4,fontsize=8,frameon=False,bbox_to_anchor=(1,1.26))
        ax.grid(axis='x',alpha=.16)
        for k in ['top','right','left']:ax.spines[k].set_visible(False)
    axs[-1].set_xlabel('Frame')
    fig.suptitle('DJI_0061_2  |  Matched identities at HOTA IoU 0.50',fontweight='bold')
    fig.text(.08,.012,'Each mark represents a matched target-frame. Gaps include unmatched frames and frames outside the annotation.',fontsize=8)
    fig.subplots_adjust(left=.1,right=.98,top=.83,bottom=.17,hspace=.54)
    fig.savefig(OUT/'identity_timeline.png',dpi=160);plt.close(fig)

def main():
    aud=read(OUT/'metric_audit.json');ev=read(OUT/'event_audit.json')
    assert aud['status']==ev['status']=='PASS'
    data={(r['scope'],r['method']):r for r in rows('pool_metrics.csv')}
    ledger=rows('switch_ledger.csv')
    hist=defaultdict(dict)
    for r in rows('official_gt_frames.csv.gz'):hist[int(r['gt'])][int(r['frame'])]=r
    times=timing(hist,{(int(r['gt']),r['method'],int(r['frame'])):r for r in ledger})
    gtvals=defaultdict(list)
    for r in rows('gt_association_thresholds.csv'):gtvals[int(r['gt']),r['method']].append(float(r['assoc_contribution_pp']))
    importance=[]
    for g in sorted(hist):
        a,b=(mean(gtvals[g,m]) for m in ('native','low_r'))
        importance.append(dict(gt=g,native_assA_contribution_pp=a,low_assA_contribution_pp=b,delta=b-a))
    importance.sort(key=lambda r:r['delta']);table(OUT/'gt_association_contributions.csv',importance)
    assert abs(sum(r['delta'] for r in importance)-(float(data['case','low_r']['AssA'])-float(data['case','native']['AssA'])))<1e-8
    identity_figure(rows('hota_matches_alpha050.csv'))
    fields=('HOTA','DetA','AssA','IDF1','MOTA','FP','FN','IDSW')
    lines=['# DJI_0061_2 的保号与关联误差','',
           '原始 KF 和仅低分 R 共用检测、观测质量、模型参数、配置与视频输入。两条取证回放的轨迹均与前次正式结果逐字节一致。'
           '所有目标和帧号均按官方可见光真值；身份延续在各自回放内部判定，跨方法对齐依靠真值目标、帧和冻结检测行。','',
           '## 分段与剔除该段后的独立评测','',
           '| 范围 | 方法 | '+' | '.join(fields)+' |','|---|---|'+('---:|'*len(fields))]
    for s,label in [('case','DJI_0061_2 · 240 帧'),('other29','其余 29 段 · 12,350 帧'),('all30','完整 30 段 · 12,590 帧')]:
        for m,ml in [('native','原始 KF'),('low_r','仅低分 R')]:
            r=data[s,m];lines.append(f'| {label} | {ml} | '+' | '.join(f'{float(r[k]):.4f}' if k in fields[:5] else f'{int(r[k]):,}' for k in fields)+' |')
    lines+=['','每个范围都先用冻结版 TrackEval 对相应序列计算，再调用每项指标自己的跨序列合并函数。30 段、单段以及每段结果均与先前档案核对。','',
            '## 换号逐项核销','',
            'CLEAR 对每个真值比较当前匹配与上一次可匹配到的轨迹 ID，允许中间有漏检。原始 KF 有 158 次换号，低分 R 有 138 次；108 次发生在同一真值同一帧。下面只列两组各自独有的变化。','',
            '| 对照方法在原方法换号处的表现 | 原始 KF 独有 50 次 | 低分 R 独有 30 次 |','|---|---:|---:|']
    for k,label in [('retained_across_endpoints','前后两端保持同一 ID，区间内无换号'),('other_misses_current','当前真值未匹配'),
                    ('other_misses_previous_anchor','上个参照帧未匹配，当前已匹配'),('other_transition_in_interval','两端均命中，区间中已有身份变化')]:
        lines.append(f"| {label} | {ev['classification']['native'][k]} | {ev['classification']['low_r'][k]} |")
    stable=[r for r in ledger if r['method']=='native' and r['kind']=='retained_across_endpoints']
    denom=sum(r['continuous_at_10']!='unobservable' for r in stable)
    held=sum(r['continuous_at_10']=='same_id' for r in stable)
    confirmed=[r for r in times if r['one_transition_per_arm']]
    lines+=['',f'在 24 次端点保号中，{sum(r["same_detector_both_endpoints"]=="True" for r in stable)} 次连两端使用的检测行也完全相同。'
            f'在后续连续 10 帧有真值的 {denom} 次中，{held} 次全程以同一 ID 命中。'
            f'另按同真值、前后 5 帧内的一对一最近换号复核，得到 {len(times)} 对时间邻近事件，其中 {len(confirmed)} 对可在同一检测行界定的前后锚点内验证两组各发生一次身份变化。该复核和上表有重叠。','',
            '真值 10 在原始 KF 中经历 4 次换号，低分 R 为 0。第 102、149、166 帧低分 R 均保有 ID 11；第 198 帧该方法漏检，不能计作成功保号。','',
            '## 逐帧机制与定点撤销','',
            '真值 31 的第 104 帧，两组使用同一低分检测，得分 0.436，检测中心与真值相距 2.60 px。'
            '在低分 R 当时的同一先验下，原始更新后误差为 2.30 px，学习 R 为 1.69 px。第 105 帧 ID 63 选择冻结检测行 71 或 49；对应行 71 的代价在原始 KF 中为 0.446，在低分 R 中为 0.521，越过门限 0.5。第 109 帧真值由原始 KF 的 ID 169 接手，低分 R 仍为 63。','',
            '真值 82 在第 208 帧之后有一段观测间断。第 223 帧，两组对同一高分候选的代价分别为 0.847 与 0.686，门限 0.8。第 228 帧原始 KF 从 ID 255 改由 272 匹配，低分 R 仍沿用自己的 ID 240。','',
            '为区分单次更新与更长的历史，以下在低分 R 回放中只将指定轨迹的一次或连续数次低分更新恢复成原生 KF。其余输入和跟踪参数一致，首次干预之前的输出逐字节一致。选取的均为这段测试视频上事后核查的事件。','',
            '| 真值 | 实际撤销帧 | 核查帧 | 原 KF 的 ID | 低分 R 的 ID | 撤销后 ID | 是否沿用原低分 R 的旧 ID |','|---:|---|---:|---:|---:|---:|---|']
    probes=[]
    for name in ('gt31_last','gt31_window','gt82_window','gt101_last'):
        p=read(OUT/(name+'.json'));observed=next(r for r in p['frames'] if r['frame']==p['event'])
        pframes=','.join(str(i['frame']) for i in p['interventions'])
        probes.append(dict(name=name,gt=p['gt'],frames=pframes,**observed))
        fmt=lambda v:'漏检' if v is None else str(v)
        lines.append(f"| {p['gt']} | {pframes} | {p['event']} | {fmt(observed['native_id'])} | {fmt(observed['low_id'])} | {fmt(observed['probe_id'])} | "+('是' if observed['probe_id']==observed['old_r_track'] else '否')+' |')
    table(OUT/'targeted_interventions.csv',probes)
    lines+=['','## 为什么 HOTA 下降','',
        'HOTA 在 19 个 IoU 阈值上分别综合检测准确度和整条轨迹的关联准确度。该段 DetA 变化 +0.0168、AssA 变化 −0.6267 个百分点。'
        '逐阈值分别替换 DetA 和 AssA，并平均两种替换顺序，检测项对 HOTA 的贡献为 +0.0165、关联项为 −0.2459，合计 −0.2294 个百分点。','',
        '真值 139 的 AssA 贡献下降最大。以 IoU 0.50 为例，原始 KF 的 ID 156 对应真值 139 达 157 帧。低分 R 的 ID 154 与 167 分别对应它 98、54 帧，还分别对应相邻真值 142 共 43、84 帧。'
        '这些帧共同降低了同一真值与单一轨迹的长期对应程度。下图使用 HOTA 本身的逐帧匹配，空白表示该阈值下未对应或真值尚未出现。','',
        '![两个真值的 HOTA 轨迹身份](identity_timeline.png)','',
        '## 结论','',
        '完整 30 段净少的 21 次换号中，20 次来自本段；剔除后 29 段净少 1 次。全量 FN 净减 12，其中该段净减 37，其余序列合计反增 25。'
        'FP 改善在本段和其余序列分别为 50 与 47；其余 29 段的 IDF1 仍从 43.3187 提高到 43.3808。该段主导了换号与净漏检收益，不能由换号计数推断 HOTA 或每一段的身份改善。','',
        '[逐换号清单](switch_ledger.csv) · [逐目标逐帧真值分配](official_gt_frames.csv.gz) · [实际关联与 KF 更新](decision_chains.json) · [29 段汇总](pool_metrics.csv) · [各真值的 AssA 分解](gt_association_contributions.csv)']
    (OUT/'DJI_0061_2_换号归因.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Report verified',len(ledger),'switch records',len(confirmed),'paired timing changes',flush=True)
if __name__=='__main__':main()
