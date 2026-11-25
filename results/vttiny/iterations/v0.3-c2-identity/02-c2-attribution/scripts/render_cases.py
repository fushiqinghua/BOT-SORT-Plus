"""Draw the case timelines from official CLEAR frame matches."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

root=Path(__file__).resolve().parent
if root.name=='scripts':root=root.parent
data=root/('downloaded' if (root/'downloaded').exists() else 'evidence')/'case_timelines.csv'
font_path=Path('C:/Windows/Fonts/msyh.ttc')
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font_path)).get_name()
else:plt.rcParams['font.family']=['Noto Sans CJK SC','DejaVu Sans']
rows=list(csv.DictReader(data.open(encoding='utf-8-sig')))
cases=[('DJI_0032_3','42','两次恢复在同一目标上受阻'),('DJI_0327_3','80','一次恢复影响后续 25 帧覆盖'),('DJI_0327_3','67','同一序列也出现覆盖收益')]
methods=['B_ref','B_C2','native_only','bank_only','release_bank_history']
labels=['B 参考','完整 C2 + B','仅限制原生模板','仅限制 B 记忆库','B 库放行历史冲突']
palette=['#e5e9ed','#147b75']; cmap=ListedColormap(palette)
fig,axs=plt.subplots(3,1,figsize=(10,7.6))
fig.subplots_adjust(left=.20,right=.97,top=.86,bottom=.09,hspace=.76)
fig.text(.055,.968,'同一真实目标在各配置下的逐帧覆盖',size=17,weight='bold',va='top')
legend=[Patch(facecolor=palette[1],label='匹配到该真实目标'),Patch(facecolor=palette[0],label='该目标有标注，未匹配'),Line2D([],[],marker='D',markersize=5,markerfacecolor='#ffcd59',markeredgecolor='#42371e',linewidth=0,label='由 B 显式恢复')]
fig.legend(handles=legend,loc='upper left',bbox_to_anchor=(.047,.94),ncol=3,frameon=False,fontsize=9)
for ax,(seq,gid,title) in zip(axs,cases):
    sub=sorted((r for r in rows if r['sequence']==seq and r['gt']==gid),key=lambda x:int(x['frame']))
    assert all(r['gt_present']=='True' for r in sub)
    fs=np.array([int(r['frame']) for r in sub])
    matrix=np.array([[bool(r[m+'_id']) for r in sub] for m in methods],int)
    ax.imshow(matrix,aspect='auto',origin='upper',interpolation='nearest',cmap=cmap,norm=BoundaryNorm([-.5,.5,1.5],2),extent=(fs[0]-.5,fs[-1]+.5,4.5,-.5))
    for i,m in enumerate(methods):
        hits=[int(r['frame']) for r in sub if r[m+'_stage']=='recovered']
        ax.scatter(hits,[i]*len(hits),s=23,marker='D',c='#ffcd59',edgecolors='#42371e',linewidths=.4,zorder=3)
    ticks=[int(x) for x in fs if x%5==0]
    ax.set_xticks(ticks);ax.set_yticks(range(5),labels);ax.tick_params(axis='both',length=0,labelsize=9)
    for yy in [.5,1.5,2.5,3.5]:ax.axhline(yy,color='white',lw=1.2)
    ax.set_title(f'{seq}   ·   GT {gid}   |   {title}',loc='left',fontsize=11,pad=7)
    for sp in ax.spines.values():sp.set_visible(False)
axs[-1].set_xlabel('原始序列帧号',fontsize=9)
fig.savefig(root/'典型目标逐帧覆盖.png',dpi=175,facecolor='white')
