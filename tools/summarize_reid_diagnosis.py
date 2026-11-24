from pathlib import Path
import shutil,zipfile
from vttiny_tracking_common import ROOT,read,dump,sha
OUT=ROOT/'outputs/reid_drop_diagnosis_v1';BASE=ROOT/'outputs/rgb_test_7trackers_v1';FT=ROOT/'outputs/rgb_test_4trackers_reid_testselect_v2'
stats=read(OUT/'feature_distance_diagnosis.json')['statistics'];interventions=read(OUT/'intervention_summary.json');comparison={r['tracker']:r for r in read(FT/'comparison.json')}
before=read(FT/'baseline_sha256_before.json');assert all(sha(BASE/k)==v for k,v in before.items())
lines=['# ReID 微调后退化诊断','','## 主要结论','','已通过固定检测、固定微调特征、只改变外观门限的干预实验，确认门限与新特征距离分布不兼容是 DeepSORT / StrongSORT 本次退化的主要原因。原始正式实验结果保持不变。','','## 距离统计','','在相同置信度>0.5的检测框上，用 IoU≥0.5 的一对一匹配映射到官方GT身份；统计与最近一次相同身份检测（间隔1–30帧）的距离，共51,520对。两套缓存的检测索引、帧号、offset一致，向量均为单位范数。','', '|统计量|预训练|微调后|','|---|---:|---:|']
for label,key,sub in [('同身份距离中位数','quantiles','p50'),('同身份距离超过0.2的比例(%)','exceeds','0.2'),('同身份距离超过0.4的比例(%)','exceeds','0.4')]:
 a=stats['pretrained/same_id/all'][key][sub];b=stats['finetuned/same_id/all'][key][sub];lines.append(f'|{label}|{a:.4f}|{b:.4f}|')
lines+=['','以上超限比例是检测对层面的诊断，不能直接称为完整跟踪器的拒绝率。DeepSORT还使用历史特征库、运动门控和IoU补匹配；StrongSORT还包含外观EMA及运动代价。','', '不同身份历史候选中的最近距离中位数也由 '+f"{stats['pretrained/nearest_other_id']['quantiles']['p50']:.3f} 增至 {stats['finetuned/nearest_other_id']['quantiles']['p50']:.3f}"+'。该统计未施加运动门控，仅作为外观分布参考。相对排序可以改善，同时固定绝对门限会拒绝更多正确候选。','','## 仅改变门限的全测试集对照','','所有对照使用同一份微调权重、检测缓存和特征缓存。下列为诊断性测试集参数干预，不是独立留出测试，也没有覆盖原基线。','','|方法|权重|外观门限|HOTA|IDF1|MOTA|IDSW|','|---|---|---:|---:|---:|---:|---:|']
for method,default in [('deepsort',.2),('strongsort',.4)]:
 for version,key in [('预训练','pretrained'),('微调：旧门限','finetuned')]:
  r=comparison[method][key];lines.append(f"|{method}|{version}|{default:.1f}|{r['HOTA']:.2f}|{r['IDF1']:.2f}|{r['MOTA']:.2f}|{r['IDSW']}|")
 for r in interventions:
  if not r['tracker'].startswith(method+'_'):continue
  t=int(r['tracker'].split('_t')[1])/10
  protocol=read(OUT/'variants'/r['tracker']/'protocol.json');original=read(FT/'protocol.json')['trackers'][method].copy();actual=protocol['trackers'][method].copy()
  assert actual.pop('max_cos_dist')==t;original.pop('max_cos_dist');assert actual==original
  lines.append(f"|{method}|微调：诊断门限|{t:.1f}|{r['HOTA']:.2f}|{r['IDF1']:.2f}|{r['MOTA']:.2f}|{r['IDSW']}|")
lines+=['','## 训练与选模环节的限制','','身份分类和Triplet Loss约束类别判别及相对距离，未要求同身份余弦距离维持在原来0.2以内。L2归一化固定了长度，不会固定不同样本间的夹角。','', '本次权重按真实框裁剪的身份检索mAP选择；部署使用有定位误差的检测框，且跟踪需要绝对门限、运动门控和轨迹管理。检索mAP提高不能保证跟踪HOTA/IDF1同步提高。训练裁剪短边中位数9像素、测试选模裁剪中位数7像素，框偏差与模糊是后续应检验的因素；本轮尚未通过单因素实验量化它们的贡献。','','## 已核验与后续建议','','- 检测缓存一致，特征索引一致，单位范数正常；此前训练/部署前向一致性与冻结层检查通过。','- 新编码器应配套校准距离门限，并同时观察正确关联、错误关联、HOTA、IDF1和IDSW。','- 当前0.6/0.8仅是本次诊断网格中的参考点，不能视为通用最优参数。','- 按用户既定协议，所有测试集选模/调参都需如实记录。现有七算法正式结果与本诊断结果应分开保存。']
(OUT/'diagnosis.md').write_text('\n'.join(lines),encoding='utf8')
dump(OUT/'integrity.json',dict(status='PASS',original_baseline_unchanged=True,only_tracker_parameter_changed='max_cos_dist',diagnostic_variants=6,frames_per_variant=12590,sequence_count=30))
(OUT/'source').mkdir(exist_ok=True)
for name in ['diagnose_reid_drop.py','summarize_reid_diagnosis.py']:shutil.copy2(ROOT/'tools'/name,OUT/'source'/name)
archive=OUT/'reid_drop_diagnosis.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for path in [OUT/'diagnosis.md',OUT/'feature_distance_diagnosis.json',OUT/'intervention_summary.json',OUT/'intervention_protocol.json',OUT/'integrity.json',*list((OUT/'source').glob('*.py')),*list((OUT/'metrics').rglob('*'))]:
  if path.is_file():z.write(path,path.relative_to(OUT))
print(dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size))
