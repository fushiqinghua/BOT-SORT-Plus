"""Verify the T1 cards, assemble the offline gallery, and package review records."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


HERE = Path(__file__).resolve().parent
FOCUS = ('DJI_0327_3', 'DJI_0061_2')
IDENTITY = {'SUPPORTED', 'AMBIGUOUS', 'DIFFERENT', 'NO_CANDIDATE'}
LOCALIZATION = {'ADEQUATE', 'PARTIAL', 'INSUFFICIENT', 'NOT_APPLICABLE'}
MODEL_FIELDS = ('case_id', 'model_identity', 'model_localization', 'model_note')
HUMAN_FIELDS = ('case_id', 'sample_scope', 'sequence', 'frame', 'human_status', 'human_identity',
                'human_localization', 'human_note', 'reviewer', 'reviewed_at')


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as source:
        return list(csv.DictReader(source))


def write_rows(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def validate_inputs(root, rows):
    files = json.loads((root / 'files.sha256.json').read_text(encoding='utf-8'))['files']
    for name, expected in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Invalid source entry {name}')
        if sha256(root / name) != expected:
            raise ValueError(f'Hash mismatch for {name}')
    if 'case_index.csv' not in files:
        raise ValueError('The case index is missing from the source manifest')
    for case in rows:
        for view in ('blind', 'annotated'):
            image = case[f'{view}_card']
            wanted = f"cards/{case['sample_scope']}/{view}/{case['case_id']}.png"
            if image != wanted or image not in files:
                raise ValueError(f'Card path does not match the verified source {image}')


def read_model(model_dir, ids, previous):
    model = {}
    inputs = sorted(model_dir.glob('*.csv'))
    if not inputs and previous.is_file():
        inputs = [previous]
    for path in inputs:
        for row in read_rows(path):
            if path == previous and row['model_status'] not in ('MODEL_REVIEWED', 'NO_NATIVE_ASSIGNMENT'):
                continue
            key = row['case_id'].strip()
            if key not in ids or key in model:
                raise ValueError(f'Unknown or duplicate model row {key} in {path}')
            if row['model_identity'] not in IDENTITY or row['model_localization'] not in LOCALIZATION:
                raise ValueError(f'Unknown model judgment code for {key} in {path}')
            if not row['model_note'].strip():
                raise ValueError(f'Model review needs a visual rationale for {key} in {path}')
            model[key] = {field: row[field] for field in MODEL_FIELDS}
    return model


def write_reviews(root, samples, model):
    old = root / 'human_review.csv'
    previous = read_rows(old) if old.exists() else []
    preserved = {r['case_id']: r for r in previous}
    if len(preserved) != len(previous) or set(preserved) - {r['case_id'] for r in samples}:
        raise ValueError('The existing human review sheet contains duplicate or unknown IDs')
    human = []
    assisted = []
    for case in samples:
        key = case['case_id']
        common = {field: case[field] for field in ('case_id', 'sample_scope', 'sequence', 'frame')}
        saved = preserved.get(key, {})
        if saved and any(saved.get(field) != value for field, value in common.items()):
            raise ValueError(f'Human row metadata changed for {key}')
        human.append({**common, 'human_status': saved.get('human_status') or 'PENDING_HUMAN_REVIEW',
                      **{field: saved.get(field, '') for field in HUMAN_FIELDS[5:]}})
        item = model.get(key, {})
        status = 'MODEL_REVIEWED' if item else 'NOT_REVIEWED'
        if not case['assigned_det_index'].strip():
            if item and (item['model_identity'], item['model_localization']) != ('NO_CANDIDATE', 'NOT_APPLICABLE'):
                raise ValueError(f'Visual judgment supplied for case without native assigned detection {key}')
            status = 'NO_NATIVE_ASSIGNMENT'
            if not item:
                item = {'model_identity': 'NO_CANDIDATE', 'model_localization': 'NOT_APPLICABLE',
                        'model_note': '原生低分关联未分配检测框，无当前候选可供复核。'}
        elif item.get('model_identity') == 'NO_CANDIDATE':
            raise ValueError(f'Case marked as no candidate has a native assignment {key}')
        assisted.append({**common, 'official_geometric_miss': case['official_geometric_miss'],
                         'model_status': status,
                         **{field: item.get(field, '') for field in MODEL_FIELDS[1:]}})
    write_rows(old, human, HUMAN_FIELDS)
    write_rows(root / 'model_assisted_review.csv', assisted, tuple(assisted[0]))
    return human, assisted


def summarize(assisted, human):
    def stats(rows):
        reviewed = [r for r in rows if r['model_status'] == 'MODEL_REVIEWED']
        misses = [r for r in rows if r['official_geometric_miss'].lower() == 'true']
        reviewed_misses = [r for r in misses if r['model_status'] == 'MODEL_REVIEWED']

        def joint(items):
            return [r['case_id'] for r in items if r['model_identity'] == 'SUPPORTED'
                    and r['model_localization'] == 'ADEQUATE']

        counts = Counter(r['model_status'] for r in rows)
        return dict(cases=len(rows), model_reviewed=len(reviewed), model_pending=counts['NOT_REVIEWED'],
                    no_native_assignment=counts['NO_NATIVE_ASSIGNMENT'],
                    model_identity=dict(Counter(r['model_identity'] for r in reviewed)),
                    model_localization=dict(Counter(r['model_localization'] for r in reviewed)),
                    official_geometric_misses=len(misses), reviewed_geometric_misses=len(reviewed_misses),
                    supported_and_adequate=len(joint(reviewed)),
                    miss_supported_and_adequate=len(joint(reviewed_misses)),
                    miss_supported_and_adequate_ids=joint(reviewed_misses))

    scopes = defaultdict(list)
    sequences = defaultdict(list)
    scope_sequence = defaultdict(list)
    for row in assisted:
        scope, sequence = row['sample_scope'], row['sequence']
        scopes[scope].append(row)
        sequences[sequence].append(row)
        scope_sequence[scope, sequence].append(row)
    return dict(all_cases=stats(assisted), by_scope={k: stats(v) for k, v in sorted(scopes.items())},
                by_sequence={k: stats(v) for k, v in sorted(sequences.items())},
                by_scope_and_sequence={scope: {seq: stats(rows) for (s, seq), rows in sorted(scope_sequence.items())
                                               if s == scope} for scope in sorted(scopes)},
                human_status=dict(Counter(r['human_status'] for r in human)),
                focus_sequence_ids=list(FOCUS), extrapolated_to_population=False, training_started=False)


def write_gallery(root, samples, assisted, human, findings):
    fields = ('case_id', 'sample_scope', 'sequence', 'frame', 'method', 'track_id', 'assigned_det_index',
              'raw_score', 'primary_bucket', 'history_frames', 'gt_id', 'identity_status_gt', 'raw_iou',
              'feasible_iou', 'anchor_covered', 'official_geometric_miss', 'annotated_card', 'blind_card')
    rows = [{**{k: row[k] for k in fields}, **m, 'human_status': h['human_status']}
            for row, m, h in zip(samples, assisted, human, strict=True)]
    data = json.dumps(rows, ensure_ascii=False).replace('<', '\\u003c')
    link = '　<a href="T1_review_findings.md">重点序列复核结论</a>' if findings else ''
    (root / 'index.html').write_text(HTML.replace('<!--ROWS-->', data).replace('<!--FINDINGS-->', link),
                                    encoding='utf-8')


HTML = '''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TCO 逐帧复核图卡</title>
<style>
* {box-sizing:border-box} [hidden] {display:none!important} body {margin:0;background:#edf2f6;color:#182434;font:15px/1.55 system-ui,sans-serif}
header {padding:22px 28px 18px;color:#edf5fc;background:#112538} h1 {margin:0 0 7px;font-size:23px}
header p {margin:0;color:#c8d9e7} nav {display:flex;gap:9px;flex-wrap:wrap;padding:12px 24px;background:#fff;border-bottom:1px solid #ccd7e1}
button,select,input {font:inherit; padding:5px 9px;color:inherit;background:#fff;border:1px solid #adbfce;border-radius:6px} button {cursor:pointer}
button[aria-pressed="true"],button[aria-current="true"] {color:#fff;background:#175184;border-color:#175184}
.workspace {display:grid;grid-template-columns:270px minmax(0,1fr);max-width:1740px;margin:auto;gap:14px;padding:14px}
aside,main {background:#fff;border:1px solid #d2dce5;border-radius:8px;padding:14px} #rows {overflow:auto;max-height:77vh;display:flex;gap:5px;flex-direction:column}
.row {text-align:left;min-width:0} .row small {display:block} .card {width:100%;height:auto;display:block;background:#e6eef4}
#caption,#details {margin:8px 0;color:#354b5d} .toolbar {display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
a {color:#145181} .muted {color:#4f6477} .files {padding-top:12px} #count {margin:0 0 8px}
@media(max-width:790px) {.workspace {grid-template-columns:1fr} #rows {max-height:16vh}}
</style>
</head>
<body>
<header><h1>TCO 逐帧复核图卡</h1><p>先看盲审图，再查看原始低分框、离线可行修正框与官方真值的叠加图。</p></header>
<nav aria-label="筛选案例">
  <button type="button" class="focus" data-sequence="">全部序列</button>
  <button type="button" class="focus" data-sequence="DJI_0327_3">DJI_0327_3</button>
  <button type="button" class="focus" data-sequence="DJI_0061_2">DJI_0061_2</button>
  <select id="scope" aria-label="样本来源"><option value="">所有样本</option><option value="original">原始 160 例</option><option value="focus">重点序列补抽</option></select>
  <input id="search" type="search" aria-label="搜索案例" placeholder="编号 / 序列 / 帧号">
</nav>
<div class="workspace">
  <aside><div id="count"></div><div id="rows"></div></aside>
  <main>
    <div class="toolbar"><button id="prev">上一例</button><button id="next">下一例</button><button id="blind">盲审图</button><button id="annotated">真值叠加图</button><button id="show-model" hidden>显示模型记录</button><a id="original" target="_blank" rel="noopener">查看原尺寸 PNG</a></div>
    <div id="caption"></div><img class="card" id="card" alt="所选案例图卡"><div id="details"></div>
    <div class="files"><a href="README.md">复核说明</a>　<a href="case_index.csv">样本索引</a>　<a href="model_assisted_review.csv">模型辅助复核表</a>　<a href="human_review.csv">人工复核表</a><!--FINDINGS--></div>
  </main>
</div>
<script type="application/json" id="data"><!--ROWS--></script>
<script>
const cases = JSON.parse(document.getElementById('data').textContent);
const $ = id => document.getElementById(id);
const labels = {SUPPORTED:'支持同一目标', AMBIGUOUS:'身份不明确', DIFFERENT:'疑似其他目标', NO_CANDIDATE:'无已分配候选',
  ADEQUATE:'定位依据充分', PARTIAL:'仅有部分定位依据', INSUFFICIENT:'定位依据不足', NOT_APPLICABLE:'不适用'};
let chosen = '', view = 'blind', seq = '', filtered = [], showModel = false;
function setCase(id) { chosen = id; view = 'blind'; showModel = false; refresh(); }
function selectVisible(offset) {
  const index = filtered.findIndex(row => row.case_id === chosen);
  const row = filtered[index + offset];
  if (row) setCase(row.case_id);
}
function refresh() {
  const scope = $('scope').value;
  const search = $('search').value.trim().toLowerCase();
  filtered = cases.filter(r => (!seq || seq === r.sequence) && (!scope || scope === r.sample_scope)
    && (!search || `${r.case_id} ${r.sequence} ${r.frame}`.toLowerCase().includes(search)));
  if (seq) filtered.sort((a,b) => (a.sample_scope === 'focus' ? 0 : 1) - (b.sample_scope === 'focus' ? 0 : 1) || a.frame - b.frame);
  if (!filtered.some(r => r.case_id === chosen)) { chosen = filtered[0]?.case_id || ''; view = 'blind'; showModel = false; }
  $('count').textContent = `共 ${filtered.length} 例`;
  $('rows').replaceChildren(...filtered.map(r => {
    const button = document.createElement('button');
    button.className = 'row';
    button.setAttribute('aria-current', String(r.case_id === chosen));
    button.textContent = r.case_id;
    const sub = document.createElement('small');
    sub.textContent = `${r.sequence} · 帧 ${r.frame}${r.sample_scope === 'focus' ? ' · 重点补抽' : ''}`;
    button.appendChild(sub); button.onclick = () => setCase(r.case_id); return button;
  }));
  document.querySelectorAll('.focus').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.sequence === seq)));
  const row = filtered.find(r => r.case_id === chosen);
  const index = filtered.indexOf(row);
  $('prev').disabled = index <= 0;
  $('next').disabled = index < 0 || index === filtered.length - 1;
  $('blind').disabled = $('annotated').disabled = !row;
  $('blind').setAttribute('aria-pressed', String(view === 'blind'));
  $('annotated').setAttribute('aria-pressed', String(view === 'annotated'));
  $('show-model').hidden = !row || view !== 'annotated';
  $('show-model').setAttribute('aria-pressed', String(showModel));
  $('show-model').textContent = showModel ? '隐藏模型记录' : '显示模型记录';
  $('card').hidden = $('original').hidden = !row;
  if (!row) { $('caption').textContent = '当前筛选条件下无图卡。'; $('details').textContent = ''; return; }
  const active = $('rows').querySelector('[aria-current=true]');
  if (active.offsetTop < $('rows').scrollTop || active.offsetTop + active.offsetHeight > $('rows').scrollTop + $('rows').clientHeight) {
    const start = active.offsetTop - $('rows').offsetTop;
    $('rows').scrollTop = Math.max(0, start - 100);
  }
  $('card').src = row[`${view}_card`]; $('original').href = $('card').src;
  $('caption').textContent = `${row.case_id} · ${row.sequence} · 当前帧 ${row.frame} · 轨迹 ${row.track_id} · ${row.method}`;
  if (view === 'blind') { $('details').textContent = '盲审图隐藏官方真值、可行修正框、复核结论。'; return; }
  const iou = value => value ? Number(value).toFixed(3) : '—';
  const lines = [`官方真值 ID ${row.gt_id || '—'} · 原框 IoU ${iou(row.raw_iou)} · 修正框 IoU ${iou(row.feasible_iou)}`];
  if (row.official_geometric_miss.toLowerCase() === 'true') lines.push('官方评测漏检，且离线几何可达。');
  if (showModel) {
    if (row.model_status === 'MODEL_REVIEWED') {
      lines.push(`模型辅助判断 · ${labels[row.model_identity]} · ${labels[row.model_localization]}`);
      lines.push(row.model_note);
    } else if (row.model_status === 'NO_NATIVE_ASSIGNMENT') {
      lines.push('无原生关联候选 · 未产生图像判断');
      lines.push(row.model_note);
    } else lines.push('模型辅助判断 · 未复核');
  }
  lines.push(`人工复核 · ${row.human_status === 'PENDING_HUMAN_REVIEW' ? '待人工复核' : row.human_status}`);
  $('details').replaceChildren(...lines.map(line => { const p = document.createElement('p'); p.textContent = line; return p; }));
}
$('prev').onclick = () => selectVisible(-1);
$('next').onclick = () => selectVisible(1);
$('blind').onclick = () => {view = 'blind'; showModel = false; refresh();};
$('annotated').onclick = () => {view = 'annotated'; refresh();};
$('show-model').onclick = () => {showModel = !showModel; refresh();};
$('search').oninput = refresh;
$('scope').onchange = refresh;
document.querySelectorAll('.focus').forEach(button => button.onclick = () => {seq = button.dataset.sequence; chosen = ''; refresh();});
refresh();
</script>
</body></html>
'''


README = '''# TCO T1 逐帧复核图卡

解压后打开 `index.html`。选择重点序列或输入案例编号。每次切换案例默认显示盲审图。点击“真值叠加图”后查看离线可行修正和官方真值，点击“查看原尺寸 PNG”可放大检查像素。完成独立判断后，可点击“显示模型记录”查看模型结论。

`original` 对应 T1 的原始 160 例，其中两段重点序列合计 18 例。`focus` 为重点序列另行补抽的 24 例几何漏检，按时间展开，用于检查不同片段的成像和身份歧义。两批样本的统计保留各自分母。

历史图块取自当前时刻之前真实进入轨迹的观测。每个画出的原始低分框来自冻结检测缓存中已分配的检测行。青色修正框是使用官方真值搜索得到的离线几何可行解，仅用于检查在残差约束下定位能改善多少。没有已分配低分框、锚定真值或可行修正时，图卡标明缺项。

`case_index.csv` 记录样本来源、原框和修正框坐标、历史帧、官方配对状态和源图校验值。`model_assisted_review.csv` 记录模型辅助判断。`human_review.csv` 是独立填写的人工复核表，待复核用 `PENDING_HUMAN_REVIEW`。人工复核可填写 `human_identity`、`human_localization`、`human_note`、复核人和日期。更新表格后运行 `reproduce/finalize_review.py` 并传入 `--results-dir` 指向解压目录，可刷新图册。

模型分别判断历史与当前候选是否支持同一目标，以及当前像素的定位依据是否充分。`SUPPORTED` 代表支持，`AMBIGUOUS` 代表证据有歧义，`DIFFERENT` 代表疑似其他目标，`NO_CANDIDATE` 代表无已分配候选。定位依据用 `ADEQUATE`、`PARTIAL`、`INSUFFICIENT`、`NOT_APPLICABLE` 记录。`NO_NATIVE_ASSIGNMENT` 来自原生关联事件，表示此例没有已分配低分框可供判断。`NOT_REVIEWED` 表示有已分配低分框待模型检查。`review_summary.json` 汇总样本内结果。图证判断保留待人工复核状态。

`source_t1` 保存原始抽样清单和 T1 核验摘要。制作阶段核对当前及历史解码图像的哈希，`files.sha256.json` 记录服务器图卡阶段产物哈希。此次工作未启动训练。
'''


def bundle(root, archive):
    archive.parent.mkdir(parents=True, exist_ok=True)
    temp = archive.with_suffix('.partial')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output:
        for file in sorted(root.rglob('*')):
            relative = file.relative_to(root)
            if (file.is_file() and file != archive and file != temp and '__pycache__' not in relative.parts
                    and relative.parts[0] != 'reproduce'):
                output.write(file, file.relative_to(root).as_posix())
        for file in sorted(HERE.glob('*.py')):
            output.write(file, 'reproduce/' + file.name)
    temp.replace(archive)


def finalize(root, model_dir, archive):
    samples = read_rows(root / 'case_index.csv')
    ids = {r['case_id'] for r in samples}
    if len(ids) != len(samples) or not ids:
        raise ValueError('Empty or duplicate case index')
    validate_inputs(root, samples)
    model = read_model(model_dir, ids, root / 'model_assisted_review.csv')
    human, assisted = write_reviews(root, samples, model)
    findings = (root / 'T1_review_findings.md').is_file()
    write_gallery(root, samples, assisted, human, findings)
    notes = README + ('\n[重点序列复核结论](T1_review_findings.md)\n' if findings else '')
    (root / 'README.md').write_text(notes, encoding='utf-8')
    summary = summarize(assisted, human)
    (root / 'review_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if archive is not None:
        bundle(root, archive)
    print(json.dumps(summary['all_cases'], ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=HERE / 'results')
    parser.add_argument('--model-dir', type=Path, default=HERE / 'model_judgements')
    parser.add_argument('--archive', type=Path, default=HERE / 'TCO_T1_review_cards.zip')
    parser.add_argument('--skip-archive', action='store_true')
    args = parser.parse_args()
    finalize(args.results_dir.resolve(), args.model_dir.resolve(), None if args.skip_archive else args.archive.resolve())
