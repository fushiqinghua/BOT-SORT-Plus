"""Render verified frame crops and a Chinese report using the completed forensic data."""
import csv
import gzip
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from rf_settings import CASE, QR, ROOT, OUT, SETTINGS, read
from botsort_plus.io import read_detection_cache, resolve_image

METHODS=('native','learned_r','low_r','high_r')
LABELS=('原始 KF','全分数 R','仅低分 R','仅高分 R')

def load(path):
    op=gzip.open if path.suffix=='.gz' else open
    with op(path,'rt',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def crops():
    cfg=read(SETTINGS)
    source=Path(cfg['baseline_root'])/'detections'/(CASE+'.json.gz')
    cache=read_detection_cache(source)
    gt_rows={(int(r['gt_id']),int(r['frame'])):r for r in load(OUT/'case/gt_frames.csv.gz')}
    mot={}
    for m in METHODS:
        with (ROOT/'pilot/trackers'/m/'data'/(CASE+'.txt')).open() as f:
            mot[m]={(int(row[0]),int(row[1])):tuple(map(float,row[2:6])) for row in csv.reader(f)}
    panels=((1,32),(1,33),(4,380),(4,381))
    fig,axs=plt.subplots(len(panels),5,figsize=(11.5,9.3))
    for i,(g,t) in enumerate(panels):
        row=gt_rows[g,t]
        b=tuple(float(row['gt_'+k]) for k in 'xywh')
        x,y,w,h=b
        meta=cache['frames'][t-1]
        im=cv2.imread(str(resolve_image(meta['image'],cfg['images_root'])))[:,:,::-1]
        left=max(0,min(im.shape[1]-38,round(x+w/2)-19))
        top=max(0,min(im.shape[0]-32,round(y+h/2)-16))
        for j,ax in enumerate(axs[i]):
            ax.imshow(im[top:top+32,left:left+38],interpolation='nearest',extent=(left-.5,left+37.5,top+31.5,top-.5))
            def draw(box,c,dash=False):
                ax.add_patch(Rectangle(box[:2],*box[2:],facecolor='none',edgecolor=c,linewidth=1.45,linestyle='--' if dash else '-'))
            if j==0:
                draw(b,'#ffe15b')
                det=meta['detections'][int(row['native_det'])]
                draw([det[0],det[1],det[2]-det[0],det[3]-det[1]],'#65ddff',True)
                info=f'GT {g}   frame {t}'
            else:
                m=METHODS[j-1]
                official=row[m+'_tid']
                key=official or row[m+'_same_det_emitted_tid']
                if key:draw(mot[m][t,int(key)],'#f97da4')
                draw(b,'#ffe15b',True)
                info=('TP   IoU '+format(float(row[m+'_iou']),'.3f') if official else
                      ('FN   IoU '+format(float(row[m+'_same_det_iou_to_gt']),'.3f') if key else 'FN   no same-row output'))
            ax.set_xticks([]);ax.set_yticks([])
            ax.set(xlim=(left-.5,left+37.5),ylim=(top+31.5,top-.5))
            ax.set_xlabel(info,fontsize=8.7,labelpad=3)
            if i==0:ax.set_title(('Frozen detection','Original KF','All-score R','Low-only R','High-only R')[j],fontsize=11)
    fig.legend([Line2D([],[],color=c,lw=2) for c in ('#c4a000','#008eb6','#db3c78')],
               ['Official ground truth','Frozen detection','Tracker output'],loc='upper center',ncol=3,bbox_to_anchor=(.5,.965),frameon=False)
    fig.suptitle('DJI_0309_4  |  Same RGB crop across four measurement updates',fontsize=15)
    fig.subplots_adjust(left=.025,right=.99,bottom=.045,top=.87,hspace=.34,wspace=.09)
    fig.savefig(OUT/'case/frame_comparison.png',dpi=160,bbox_inches='tight')
    plt.close(fig)

def main():
    assert read(OUT/'four_way_audit.json')['status']=='PASS'
    metrics=load(OUT/'four_way_metrics.csv')
    seq={r['sequence']:r for r in load(OUT/'full/by_sequence.csv')}
    groups={r['gt_id']:r for r in load(OUT/'case/by_gt.csv')}
    a=read(OUT/'case/association_attribution.json')
    assert a['native_only_tp']==78 and a['r_only_tp']==2
    crops()
    lines=['# VT-Tiny-MOT 观测噪声 R 的漏检归因','',
           '在冻结检测、RGB ReID、图像质量特征和 R 权重后，变更 R 用于哪些已匹配观测。'
           '仅低分采用原生低分关联区间 0.10 < score < 0.50，仅高分采用 score > 0.50，涵盖待确认轨迹的高分更新。出生沿用原始 KF。'
           'DJI_0309_4 的原生与全分数 R 回放和既有预测逐字节一致，四组案例轨迹也和各自的全量结果逐字节一致。','',
           '## 完整 30 段与 DJI_0309_4','',
           '| 范围 | R 开关 | HOTA | IDF1 | MOTA | FP | FN | IDSW |','|---|---|---:|---:|---:|---:|---:|---:|']
    for scope in ('COMBINED_SEQ',CASE):
        for m,label in zip(METHODS,LABELS):
            r=next(r for r in metrics if r['sequence']==scope and r['tracker']==m)
            lines.append(f"| {scope} | {label} | "+' | '.join(format(float(r[k]),'.4f') for k in ('HOTA','IDF1','MOTA'))+
                         ' | '+' | '.join(f"{int(r[k]):,}" for k in ('FP','FN','IDSW'))+' |')
    c=seq['COMBINED_SEQ']
    lines+=['','## 新漏与补回','',
            f"全分数 R 使 {c['lost']} 个原本命中的真值目标帧转为漏检，同时补回 {c['gained']} 个原本漏掉的帧，合计净增 {c['net_fn']}。"
            f"仅低分 R 保住了上述 {c['lost']} 个中的 {c['lost_low_r_hit']} 个，并保留了 {c['gained']} 个补回中的 {c['gained_low_r_hit']} 个。"
            f"相对原始 KF，低分版自身新增 {c['low_r_lost']} 个漏检、补回 {c['low_r_gained']} 个。",'',
            '| 序列 | 全分数 R 新漏 | 全分数 R 补回 | 全分数 R 净增 FN | 仅低分 R 净增 FN |','|---|---:|---:|---:|---:|']
    for s,r in seq.items():
        if s=='COMBINED_SEQ':continue
        lines.append(f"| {s} | {r['lost']} | {r['gained']} | {int(r['net_fn']):+d} | {int(r['low_r_lost'])-int(r['low_r_gained']):+d} |")
    lines+=['','## DJI_0309_4 的两条失效路径','',
            '全分数 R 新漏 78 个真值目标帧、补回 2 个。58 个新增漏检中，原始 KF 使用的检测行在全分数 R 下无输出；另有 20 个仍使用同一检测行输出，但与真值的 IoU 跌破 0.5。'
            '78 帧中有 40 帧仅高分 R 也会漏、仅低分 R 可命中；6 帧方向相反；31 帧两种单独开关都可以命中；1 帧两者均漏。','',
            '| 真值 ID | 新漏 | 补回 | 仅低分 R 保住新漏 | 仅高分 R 保住新漏 |','|---|---:|---:|---:|---:|']
    for g,r in groups.items():
        lines.append(f"| {g} | {r['lost']} | {r['gained']} | {r['lost_low_r_hit']} | {r['lost_high_r_hit']} |")
    lines+=['',
            '- 真值 1，第 32–33 帧。第 32 帧检测得分 0.521，检测中心距离真值 0.758 px，优于先验的 2.427 px。在同一先验下，横向增益由 0.788 降至 0.450，后验中心误差从 1.123 增至 1.680 px。第 33 帧对同一低分候选的关联代价从原生的 0.435 变为全分数 R 的 0.570，跨过门限 0.5。仅高分版也漏，恢复高分原生更新后可命中。',
            '- 真值 4，第 379–394 帧。第 379 帧高分框得分 0.739。同一先验下，R 将后验中心误差从 1.118 增至 1.461 px。第 380 帧两组仍选中同一检测框，输出对真值的 IoU 分别为 0.663 和 0.461。只固定全分数 R 的当帧先验改回原生增益，该帧 IoU 可达 0.582，足以命中。第 381 帧的低分检测对真值 IoU 为 0.808；对该检测的关联代价在原生为 0.281、在全分数 R 为 0.695，跨过同一个门限 0.5。第 381–393 帧全分数 R 没有输出原始 KF 的对应检测行。这 14 个新增漏检在仅高分版全部复现，低分版全部保住。第 394 帧全分数 R 以新 ID 找回。',
            '- 真值 4，第 493–507 帧。第 493 帧低分框得分 0.419，检测中心误差仅 0.538 px，同一先验下 R 使后验中心误差从 0.874 增至 1.385 px。第 495 帧对同一低分候选，原生代价 0.417，全分数 R 为 0.506，跨过 0.5 门限。第 500–507 帧的 8 个损失在两个单独开关下均能保住。此例同时有高低分更新历史，不能将 8 帧单独归给一次低分更新。','',
            '![四组回放的同一原始 RGB 图块](case/frame_comparison.png)','',
            '## 判断','',
            '若继续使用学习 R，优先仅对低分匹配启用，恢复高分匹配的原生 KF 更新。在完整 30 段上，相对原始 KF，低分版少 97 个 FP、12 个 FN、21 次换号，IDF1 增加 0.0640 个百分点，HOTA 下降 0.0386 个百分点。'
            '净少的 21 次换号中有 20 次来自 DJI_0061_2。原始 KF 仍保有最高 HOTA。门控是根据这套官方测试集上的诊断选择，指标反映本次开发对比。','',
            '[四组逐序列指标](four_way_metrics.csv) · [全量目标帧得失](original/events.csv.gz) · [案例逐帧真值与检测索引](case/gt_frames.csv.gz) · [关联成本和同先验更新明细](case/association_attribution.json)']
    (OUT/'漏检归因报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Report and frame crops saved',flush=True)

if __name__=='__main__':main()
