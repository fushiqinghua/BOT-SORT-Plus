"""Retrieve verified RGB/IR frames and annotation context for locked R3 review cards."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import zipfile


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from remote_exec import remote


COLLECT = """
from collections import defaultdict
import gzip
import hashlib
import io
import zipfile

experiment = r.parent/'vttiny_wem_r3_v1'
project = r.parents[1]
raw = project/'data/raw/VT-Tiny-MOT'
named_sources = {
    'capture': experiment/'observer/parity/full.json',
    'labels': experiment/'labels_live/full/summary.json',
    'feature': experiment/'feature_analysis/embeddings/full/manifest.json',
    'rgb_coco': raw/'annotations/instances_00_train2017.json',
    'ir_coco': raw/'annotations/instances_01_train2017.json',
    'intake_images': project/'reports/image_hashes.jsonl',
}

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path):
    return json.loads(path.read_bytes())

def require(condition, message):
    if not condition:
        raise ValueError(message)

digests = {key: sha(path) for key, path in named_sources.items()}
require(all(digests[key] == value for key,value in expected.items()), 'Frozen source digest differs')
parity = read(named_sources['capture'])
manifest = read(named_sources['feature'])
labels = read(named_sources['labels'])
require(parity['status']=='PASS' and parity['online_w_updates']==0 and not labels['official_test_gt_used'],
        'Unexpected tracking or truth scope')
require(labels['source']['capture']==manifest['parity_sha256']==digests['capture'], 'Mismatched R3 captures')

parts = defaultdict(list)
for case in candidates:
    parts[case['partition']].append(case)
for part, part_cases in parts.items():
    path = experiment/'labels_live/full'/part/'candidate_edges.jsonl.gz'
    require(sha(path)==labels['partitions'][part]['labels_sha256'], f'Altered {part} labels')
    requested = {(c['sequence'],c['frame'],c['track_id'],c['det_index']):c for c in part_cases}
    require(len(requested)==len(part_cases), 'Repeated review key')
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for row in map(json.loads, stream):
            key = row['sequence'],row['frame'],row['track_id'],row['det_index']
            picked = requested.pop(key, None)
            if picked is not None:
                require(all(picked.get(k)==v for k,v in row.items()), 'Requested candidate differs from frozen label')
    require(not requested, 'Requested candidate does not exist in its locked split')

grouped = defaultdict(list)
for case in candidates:
    grouped[case['sequence']].append(case)
contexts = {}
required_frames = defaultdict(dict)
for seq, cases in grouped.items():
    proof = parity['references']['B']['files'][seq]
    path = Path(proof['snapshot_file'])
    require(sha(path)==proof['observation_proof']['snapshots_sha256']==manifest['sequences'][seq]['snapshot'],
            f'Altered snapshot for {seq}')
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        shots = {record['frame']: record for record in map(json.loads, stream)}
    require(len(shots)==proof['frames'], f'Incomplete snapshot for {seq}')
    for case in cases:
        frame, det, track = case['frame'], case['det_index'], case['track_id']
        shot = shots[frame]
        require(shot['sequence']==seq and shot['reference']==case['reference'], 'Source context diverged')
        edge, = [x for x in shot['s0_edges'] if (x['track_id'],x['det_index'])==(track,det)]
        candidate = shot['detections'][det]
        require(edge['native_low_feasible'] and edge['gmc_guard_feasible'] and candidate['band']=='S0'
                and candidate['raw_xyxy']==case['raw_box_xyxy'] and candidate['raw_score']==case['raw_score'],
                'Weak box does not match frozen original detector row')
        prior, = [x['selected_xyxy'] for x in shot['post_gmc'] if x['track_id']==track]
        start, = [t for t in shot['start_tracks'] if t['track_id']==track and t['collection']=='active_tracks']
        history = start['raw_history']
        current_sha = manifest['sequences'][seq]['images'][str(frame)]
        required_frames[seq][frame] = (shot['image'], current_sha)
        history_items = []
        for item in history:
            t = item['frame']
            require(t<frame, 'History must be causal')
            old = shots[t]
            accepted, = [a for a in old['accepted'] if a['track_id']==track]
            require(accepted==item, 'Historical raw observation differs from same-frame accepted ledger')
            notes, = [e for e in case['causal_history_evidence'] if e['frame']==t and e['det_index']==item['det_index']]
            required_frames[seq][t] = (old['image'],manifest['sequences'][seq]['images'].get(str(t)))
            history_items.append(dict(**notes, raw_history_box=item['raw_xyxy'],raw_score=item['raw_score']))
        contexts[case['case_id']] = dict(predicted_box=prior,history=history_items)

frame_assets = {}
for camera, key in (('00','rgb_coco'),('01','ir_coco')):
    coco = read(named_sources[key])
    videos = {v['id']:v['name'] for v in coco['videos']}
    selected_images = {}
    for image in coco['images']:
        rel = Path(image['file_name'])
        seq, mode, filename = rel.parts
        frame = int(image['frame_id']) + 1
        if frame not in required_frames.get(seq, {}):
            continue
        require(mode==camera and videos[image['video_id']]==rel.parent.as_posix()
                and int(Path(filename).stem)==image['mot_frame_id'], 'Official COCO frame index mismatch')
        image_path = raw/'train2017'/rel
        rgb_path,rgb_sha = required_frames[seq][frame]
        expected_path = Path(rgb_path)
        if camera=='01':
            expected_path = expected_path.parent.parent/'01'/expected_path.name
        require(image_path == expected_path, 'Paired RGB/IR image filenames disagree')
        encoded_path = Path('raw_frames')/seq/camera/filename
        slot = frame_assets.setdefault((seq,frame), {})
        require(camera not in slot, 'Repeated COCO image at a clip/frame/camera')
        slot[camera] = dict(source=image_path, archive=encoded_path.as_posix(),
                            image_id=image['id'], shape=[image['height'],image['width']],truth=[],
                            expected_intake=f'train2017/{rel.as_posix()}',
                            expected_r3=rgb_sha if camera=='00' else None)
        selected_images[image['id']] = slot[camera]['truth']
    for ann in coco['annotations']:
        if ann['image_id'] not in selected_images:
            continue
        x,y,w,h = ann['bbox']
        selected_images[ann['image_id']].append(dict(gt_id=ann['track_id'],box_xyxy=[x,y,x+w,y+h],
             category_id=ann['category_id'],ignored=bool(ann.get('ignore') or ann.get('iscrowd')),annotation_id=ann['id']))

require(len(frame_assets)==sum(map(len,required_frames.values())), 'Missing indexed image frames')
require(all(set(slot)=={'00','01'} for slot in frame_assets.values()), 'Missing paired image frames')
required_files = {v['expected_intake'] for slot in frame_assets.values() for v in slot.values()}
intake = {}
with named_sources['intake_images'].open(encoding='utf-8') as stream:
    for entry in map(json.loads,stream):
        if entry['path'] in required_files:
            intake[entry['path']] = entry['sha256']
require(set(intake)==required_files, 'Missing archived intake image hash')

for (seq, frame), slot in frame_assets.items():
    require(slot['00']['shape']==slot['01']['shape'], 'RGB/IR dimensions differ')
    for asset in slot.values():
        actual = sha(asset['source'])
        require(actual==intake[asset['expected_intake']] and asset['expected_r3'] in (None, actual),
                'Raw image does not match intake or frozen R3')
        asset['sha256'] = actual

def recorded(seq,frame):
    rgb, ir = (frame_assets[seq,frame][camera] for camera in ('00','01'))
    return dict(frame=frame,rgb_path=rgb['archive'],ir_path=ir['archive'],truth=rgb['truth'],ir_truth=ir['truth'],
                rgb_image_id=rgb['image_id'],ir_image_id=ir['image_id'],image_shape=rgb['shape'])

cards = []
for case in candidates:
    ctx = contexts[case['case_id']]
    history = [dict(recorded(case['sequence'],item['frame']), **{k:v for k,v in item.items() if k!='frame'})
               for item in ctx['history']]
    cards.append(dict(case,current=recorded(case['sequence'],case['frame']),history=history,
                      predicted_box=ctx['predicted_box']))

with io.BytesIO() as buf:
    with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_STORED) as z:
        images = {}
        for slot in frame_assets.values():
            for asset in slot.values():
                z.write(asset['source'],asset['archive'])
                images[asset['archive']] = asset['sha256']
        z.writestr('card_inputs.json',json.dumps(dict(cases=cards,sources=digests,images=images,
                     scope='official training: visual review only'),ensure_ascii=False,indent=2))
    sys.stdout.buffer.write(buf.getvalue())
"""


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_checked(path, data):
    if path.is_file() and path.read_bytes() != data:
        raise ValueError(f"Existing review input differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--project-root", help="Remote VT-Tiny-MOT project root; overrides VT_WEM_REVIEW_PROJECT_ROOT")
    p.add_argument("--candidates", type=Path, default=HERE / "inputs/candidates.json")
    p.add_argument("--r3-root", type=Path, default=HERE.parent,
                   help="Local root of original frozen observer, labels and feature outputs")
    p.add_argument("--output", type=Path, help="Destination for verified source images and private card inputs")
    args = p.parse_args()
    if args.project_root is not None:
        os.environ["VT_WEM_REVIEW_PROJECT_ROOT"] = args.project_root
    packet = json.loads(args.candidates.read_bytes())
    if packet["schema"] != "wem_r3_review_candidates_v1":
        raise ValueError("Unexpected review candidate schema")
    cases = packet["candidates"]
    if args.smoke:
        cases = [next(c for c in cases if c["partition"] == part and c["review_pool"] == pool)
                 for part in ("calibration", "holdout") for pool in ("background", "identity")]
    stage = args.r3_root
    expected = dict(capture=sha(stage / "observer/results/full.json"),
                    labels=sha(stage / "labels/results/full/summary.json"),
                    feature=sha(stage / "results/full/feature_manifest.json"),
                    rgb_coco="5a6696f6cf038301d76d921f7566a1624f5c116f634848a92c2317468f657fad",
                    ir_coco="39f19472ad2dddd5b4cbf1aebb35450ccf412841b9661d4904ffbe08eccce125")
    prefix = ("import json, sys\nfrom pathlib import Path\n"
              f"expected={expected!r}\ncandidates=json.loads({json.dumps(cases, ensure_ascii=True)!r})\n")
    content = remote(prefix + COLLECT)
    output = args.output or HERE / "assets" / ("smoke" if args.smoke else "full")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        paths = archive.namelist()
        for name in paths:
            rel = PurePosixPath(name)
            if rel.is_absolute() or '..' in rel.parts or '\\' in name:
                raise ValueError(f"Unexpected archive path: {name}")
        manifest = json.loads(archive.read("card_inputs.json"))
        if archive.testzip() or len(paths) != len(set(paths)) or set(paths) != {*manifest["images"], "card_inputs.json"}:
            raise ValueError("inconsistent frame archive")
        if {c["case_id"] for c in cases} != {c["case_id"] for c in manifest["cases"]}:
            raise ValueError("Missing case asset")
        for name, digest in manifest["images"].items():
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Image transfer digest differs: {name}")
            write_checked(output / name, data)
        write_checked(output / "card_inputs.json", archive.read("card_inputs.json"))
    print(f"{len(cases)} cards, {len(manifest['images'])} verified original images, {output}")


if __name__ == "__main__":
    main()
