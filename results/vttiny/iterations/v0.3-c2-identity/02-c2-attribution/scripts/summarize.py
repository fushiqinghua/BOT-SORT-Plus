from pathlib import Path
import json,gzip,csv
from collections import Counter,defaultdict
import pandas as pd
import numpy as np
job=Path(__file__).resolve().parent
o=job/'evidence';c=job/'controls';r=job.parent/'c2_full_v1/full30'
def load(name):return pd.read_csv(o/(name+'.csv'))
def read(p):return json.loads(p.read_bytes())
def dump(x):return json.dumps(x,ensure_ascii=False,indent=2,default=lambda y:y.item())
def table(x):return x.to_dict('records')
res={}
coll=Counter(); changed=[]
for p in sorted((c/'events/fixed_state').glob('*.jsonl')):
    for line in p.open():
        e=json.loads(line)
        if e['event']!='fixed_assignment':continue
        k=e['stage']
        for v in ('old_count','new_count','changed_costs','discounted','pairs','nonempty'):coll[k,v]+=e[v]
        coll[k,'stage_frames']+=1
        coll[k,'old_only']+=len(e['old_only']);coll[k,'new_only']+=len(e['new_only'])
        coll[k,'cost_changed_frames']+=e['changed_costs']>0
        if e['old_only'] or e['new_only']:coll[k,'decision_changed_frames']+=1;changed.append(e)
res['fixed_state']={s:{k:int(coll[s,k]) for k in ['stage_frames','nonempty','cost_changed_frames','decision_changed_frames','old_count','new_count','old_only','new_only','changed_costs','discounted','pairs']} for s in ['high','unconfirmed']}
(o/'fixed_state_changes.json').write_text(dump(changed))
res['controls']=read(c/'metrics/comparison.json')['results']
res['formal']=read(r/'metrics/comparison.json')['results']
# Verify the newly evaluated trajectories against the official frozen input identities.
pr=read(r/'protocols/B_ref.json')
for n in [k['tracker'] for k in res['controls'] if k['tracker']!='B_ref']:
    alt=read(c/'protocols'/(n+'.json'))
    assert all(pr[key]==alt[key] for key in ('sequences','detector_sha256','rgb_encoder_sha256','detection_fingerprints','feature_hashes','native_config','source_protocol_sha256'))
    assert alt['status']=='COMPLETE' and len(alt['output_prediction_hashes'])==30
assert pr['output_prediction_hashes']==read(c/'protocols/fixed_state.json')['output_prediction_hashes']
assert read(c/'metrics/evaluation_audit.json')['ground_truth']['sha256']==read(r/'metrics/evaluation_audit.json')['ground_truth']['sha256']
pd.DataFrame([{k:v for k,v in a.items() if k!='delta'} for a in res['controls']]).to_csv(o/'controlled_metrics.csv',index=False)
pd.DataFrame(read(c/'metrics/summary.json')['results']).to_csv(o/'controlled_sequences.csv',index=False)
res['control_recoveries']={p.stem:sum(read(x)['event_counts'].get('recovered',0) for x in (c/'timing'/p.stem).glob('*.json')) for p in (c/'protocols').glob('*.json') if p.stem!='B_ref'}
a=load('matched_gates')
b=load('anchor_gates')
res['high_conflict']=dict(previous=table(a.query("method=='B_C2' and stage=='high' and reason=='history_conflict'").groupby('previous_labeled_track',as_index=False)['count'].sum()),anchor=table(b.query("method=='B_C2' and stage=='high' and reason=='history_conflict'").groupby('last_private_write_gt',as_index=False)['count'].sum()))
res['shadow_bank_would_block']=table(a.query("method=='B_C2_shadow' and not gate_allows and custom_wrote").groupby(['stage','previous_labeled_track'],as_index=False)['count'].sum())
res['same_track_gate']=table(a.query("method=='B_C2' and stage=='high'").groupby(['gate_allows','previous_labeled_track'],as_index=False)['count'].sum())
fp=load('paired_fp');res['flow']=table(fp.drop(columns=['sequence']).groupby('comparison',as_index=False).sum())
h=load('per_sequence')
res['stage_totals']=table(h.groupby('method',as_index=False)[['recovered','memory_writes','matches','FP','FN','IDSW']].sum())
t=load('targets')
res['whole_gt']={}
for k,sub in t.groupby('comparison'):
    res['whole_gt'][k]={'new':int(((sub.ref_frames==0)&(sub.var_frames>0)).sum()),'lost':int(((sub.ref_frames>0)&(sub.var_frames==0)).sum()),'more':int((sub.gained>sub.lost).sum()),'less':int((sub.gained<sub.lost).sum())}
sw=load('switch_events')
res['switch_counts']=table(sw.groupby(['comparison','which'],as_index=False).size())
cov=load('changed_gt_frames')
sub=cov[cov.comparison=='B_C2 vs B_ref'].copy()
sub['age']=sub['frame']-sub['last_recovery_frame']
res['lost_gain_breakdown']={}
for k,v in sub.groupby('direction'):
    res['lost_gain_breakdown'][k]={'by_stage':dict(v.stage.value_counts()),'other_row':dict(v.other_same_row.value_counts()),'follows_nonshared_recovery':int(v.last_recovery_not_shared.sum()),'inside10_after_nonshared_recovery':int((v.last_recovery_not_shared&(v.age<=10)).sum()),'sequence_counts':dict(v.sequence.value_counts())}
rec=load('recoveries')
rc=rec.query("comparison=='B_C2 vs B_ref'")
res['recovery_counts']=table(rc.groupby(['side','shared','endpoint'],as_index=False).size())
res['alternative_check']=table(rc.query("side=='ref' and not shared").fillna({'other_check':'unknown/unavailable'}).groupby(['other_check','endpoint'],as_index=False).size())
res['targets_big_loss']=table(t[t.comparison=='B_C2 vs B_ref'].assign(net=lambda x:x.gained-x.lost).nsmallest(8,'net'))
res['targets_big_gain']=table(t[t.comparison=='B_C2 vs B_ref'].assign(net=lambda x:x.gained-x.lost).nlargest(8,'net'))
render=[]
for (seq,gid,d),g in sub.groupby(['sequence','gt','direction']):
    frs=sorted(g.frame)
    for interval in np.split(np.array(frs),np.where(np.diff(frs)!=1)[0]+1):
        render.append(dict(sequence=seq,gt=gid,direction=d,start=interval[0],end=interval[-1],length=len(interval)))
res['long_runs']=sorted(render,key=lambda x:-x['length'])[:18]
res['strong_cases']=table(rc.query("side=='ref' and not shared and endpoint=='same_gt' and other_check=='appearance_contradiction'").head(18).fillna(''))
(o/'summary.json').write_text(dump(res),encoding='utf-8')
print(dump({k:res[k] for k in ['fixed_state','high_conflict','flow','whole_gt','stage_totals','recovery_counts','long_runs']}))
