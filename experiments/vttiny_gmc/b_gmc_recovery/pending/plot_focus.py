"""Plot the actual frozen observations that overlap the official GT4 after frame 391."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


root = Path(__file__).resolve().parent/'results'
with (root/'DJI_0030_2_GT4_检测与原ID逐帧核验.csv').open(encoding='utf-8-sig') as f:
    rows = [r for r in csv.DictReader(f) if int(r['frame']) > 391 and int(r['n_det_IoU_at_least_05'])]
events = json.loads((root/'focus2/focus_events.json').read_bytes())['pending_guarded']['events']
floor = next(e['score_floor'] for e in events if e['event']=='defer' and e['tid']==7)
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'], 'font.size':11})
fig, ax = plt.subplots(figsize=(8.6,3.8), layout='constrained')
for i,r in enumerate(rows):
    score = float(r['best_det_score'])
    hit = r['guarded_GT4_tid']=='7'
    color = '#00725E' if hit else '#425A70'
    ax.barh(i,score,height=.44,color=color)
    ax.text(score+.006,i,f"{score:.4f}" + ('  原 ID 7 接回' if hit else ''),va='center',color=color,
            fontsize=10,fontweight='bold' if hit else 'normal')
ax.axvline(floor,ls='--',lw=1.2,color='#975B10',zorder=0,label=f'原 ID 的历史分数门槛 {floor:.4f}')
ax.axvline(.5,ls=':',lw=1.4,color='#5A6169',label='原生高分门槛 0.50')
ax.set(yticks=range(len(rows)),yticklabels=[r['frame'] for r in rows], xlim=(0,.53), xlabel='冻结检测分数',ylabel='视频帧')
ax.invert_yaxis()
ax.legend(frameon=True,facecolor='white',edgecolor='none',framealpha=1,loc='lower right',fontsize=9)
ax.set_title('DJI_0030_2　与真值 4 的 IoU 达到 0.5 的七帧观测',loc='left',fontsize=13,pad=10)
for spine in ('top','right'): ax.spines[spine].set_visible(False)
ax.spines['left'].set_color('#9BA4AC'); ax.spines['bottom'].set_color('#9BA4AC')
fig.savefig(root/'DJI_0030_2_GT4_检测证据.png',dpi=155)
plt.close(fig)
