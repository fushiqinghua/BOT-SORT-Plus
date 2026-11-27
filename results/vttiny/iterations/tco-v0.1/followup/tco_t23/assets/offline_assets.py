"""Audit exact RGB overlap and convert official indexed train GT, offline only."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import math
from pathlib import Path
from zipfile import ZipFile

from frozen_inputs import digest, read_json, read_jsonl, write_json


def rgb_paths(raw, split):
    path = raw / 'annotations' / f'instances_00_{split}2017.json'
    source = read_json(path)
    videos = {row['id']: row['name'] for row in source['videos']}
    samples = {}
    for image in source['images']:
        rel = Path(image['file_name'])
        sequence, camera, filename = rel.parts
        if (camera != '00' or videos[image['video_id']] != rel.parent.as_posix()
                or int(Path(filename).stem) != image['mot_frame_id'] or rel in samples):
            raise ValueError(f'Conflicting original RGB COCO index at {rel}')
        samples[rel] = (sequence, int(image['frame_id']) + 1, image['id'])
    return path, source, samples


def source_overlap(raw, out):
    project = raw.parents[2]
    train_path, train_source, train = rgb_paths(raw, 'train')
    dev_path, dev_source, development = rgb_paths(raw, 'test')
    relpaths = {}
    for split, index in (('train', train), ('test', development)):
        for rel, (seq, _, _) in index.items():
            relpaths[f'{split}2017/{rel.as_posix()}'] = (split, seq)
    if len(train) != 34111 or len(development) != 12590:
        raise ValueError('Unexpected official indexed frame counts')
    historical_hash_report = project / 'reports/image_hashes.jsonl'
    old = {row['path']: row['sha256'] for row in read_jsonl(historical_hash_report) if row['path'] in relpaths}
    if set(old) != set(relpaths):
        raise ValueError('Missing historical checksums for official RGB frames')
    grouped = defaultdict(lambda: defaultdict(set))
    checked = sorted(relpaths)
    with ThreadPoolExecutor(max_workers=16) as pool:
        for rel, new in zip(checked, pool.map(lambda rel: digest(raw / rel), checked), strict=True):
            if new != old[rel]:
                raise ValueError(f'Raw image changed since archived intake checksum: {rel}')
            split, seq = relpaths[rel]
            grouped[new][split].add(seq)
    cross, same, dev_pairs = defaultdict(set), set(), set()
    cross_hashes = 0
    for by_split in grouped.values():
        training_ids, dev_ids = by_split['train'], by_split['test']
        if training_ids and dev_ids:
            cross_hashes += 1
            for train_seq in training_ids:
                cross[train_seq].update(dev_ids)
        same.update(combinations(sorted(training_ids), 2))
        dev_pairs.update(combinations(sorted(dev_ids), 2))
    archive = project / 'data/archives/VT-Tiny-MOT.zip'
    with ZipFile(archive) as zip_file:
        archive_suffix_counts = Counter(Path(name).suffix.lower() for name in zip_file.namelist()
                                        if not name.endswith('/'))
    categories = ('images', 'videos')
    result = dict(status='PASS', evidence='fresh complete indexed RGB SHA-256, compared to archived intake bytes',
                  indexed_training_sequences=sorted({v[0] for v in train.values()}),
                  development_sequences=sorted({v[0] for v in development.values()}),
                  cross_split_duplicate_sequences={k: sorted(v) for k, v in sorted(cross.items())},
                  same_train_duplicate_sequence_pairs=sorted(same),
                  same_development_duplicate_sequence_pairs=sorted(dev_pairs),
                  cross_split_duplicate_hash_count=cross_hashes, unique_rgb_byte_hashes=len(grouped),
                  training_rgb_frames=len(train), development_rgb_frames=len(development),
                  checked_historical_rgb_entries=len(checked), historical_hash_report_sha256=digest(historical_hash_report),
                  train_rgb_coco_sha256=digest(train_path), development_rgb_coco_sha256=digest(dev_path),
                  official_coco_fields={split: {key: sorted({name for row in source[key] for name in row})
                                               for key in categories}
                                        for split, source in [('train', train_source), ('test', dev_source)]},
                  official_package_members=dict(archive_suffix_counts),
                  limitations='Exact byte equality does not certify different frames or clips come from different parent recordings.')
    write_json(out, result)
    print({k: v for k, v in result.items() if k not in ('indexed_training_sequences', 'development_sequences')}, flush=True)


def make_training_gt(raw, root, cache_protocol):
    path, coco, index = rgb_paths(raw, 'train')
    protocol = read_json(cache_protocol)
    id_map = {}
    for seq, rows in protocol['sequences'].items():
        for row in rows:
            rel = Path(row['image']).relative_to(raw / 'train2017')
            a_seq, a_frame, image_id = index[rel]
            if (seq, row['frame']) != (a_seq, a_frame) or image_id in id_map:
                raise ValueError(f'Official COCO frame does not match detector input: {row}')
            id_map[image_id] = (seq, a_frame)
    annotations = coco['annotations']
    classes = {row['id'] for row in coco['categories']}
    if classes != set(range(7)):
        raise ValueError('Expected published COCO categories 0 through 6')
    per_track_classes = defaultdict(set)
    by_frame_identity = defaultdict(list)
    stats = defaultdict(Counter)
    max_track_id = max(int(ann['track_id']) for ann in annotations)
    for ann in annotations:
        if ann['image_id'] not in id_map:
            continue
        seq, frame = id_map[ann['image_id']]
        tid, cls = int(ann['track_id']), int(ann['category_id'])
        box = tuple(float(v) for v in ann['bbox'])
        x, y, w, h = box
        if cls not in classes or tid < 0 or not all(map(math.isfinite, box)) or w <= 0 or h <= 0:
            raise ValueError(f'Invalid official RGB annotation {ann["id"]}')
        mark = not bool(ann.get('ignore') or ann.get('iscrowd'))
        per_track_classes[seq, tid].add(cls)
        by_frame_identity[seq, frame, tid].append((box, cls + 1, mark, ann['id']))
        stats[seq]['source_rows'] += 1
        stats[seq]['boxes_beyond_image'] += int(x < 0 or y < 0 or x + w > 640 or y + h > 512)
    conflicts = []
    next_ignore_id = max_track_id + 1
    output_rows = defaultdict(list)
    for (seq, frame, tid), group in sorted(by_frame_identity.items()):
        unique = sorted({(box, cls, mark) for box, cls, mark, _ in group})
        stats[seq]['exact_duplicate_annotation_rows_removed'] += len(group) - len(unique)
        inconsistent_class = len(per_track_classes[seq, tid]) > 1
        identity_conflict = len(unique) > 1
        if identity_conflict or inconsistent_class:
            conflicts.append(dict(sequence=seq, frame=frame, track_id=tid, annotation_ids=[row[3] for row in group],
                                  conflicting_boxes=identity_conflict, class_changes=inconsistent_class))
        for box, cls, mark in unique:
            mark = int(mark and not identity_conflict and not inconsistent_class)
            # TrackEval requires distinct identifiers per frame even for ignored rows.
            out_id = tid
            if not mark and identity_conflict:
                out_id = next_ignore_id
                next_ignore_id += 1
            output_rows[seq].append((frame, out_id, *box, mark, cls, -1))
            stats[seq]['valid_rows' if mark else 'ignored_rows'] += 1
    report = dict(schema='tco_train_coco_rgb_mot_v1', status='PASS', input_annotation_sha256=digest(path),
                  input_cache_protocol_sha256=digest(cache_protocol), sequences=len(protocol['sequences']),
                  frames=len(id_map), output={}, class_mapping='official zero-based category + 1; no category merge here',
                  validity_policy='COCO crowd/ignore, class-inconsistent identities and conflicting frame/ID rows are mark=0; exact duplicate rows deduplicated. Original xywh unchanged.',
                  notes='Derived from official training COCO, independent of the unmodified official test MOT GT.')
    for seq, rows in protocol['sequences'].items():
        out = root / seq / 'gt/gt.txt'
        out.parent.mkdir(parents=True, exist_ok=True)
        converted = output_rows[seq]
        if len({(row[0], row[1]) for row in converted}) != len(converted):
            raise ValueError(f'Duplicate frame/ID after safe conversion: {seq}')
        temp = out.with_suffix('.partial')
        with temp.open('w', encoding='utf-8', newline='\n') as stream:
            for row in sorted(converted):
                stream.write(','.join(map(str, row)) + '\n')
        temp.replace(out)
        report['output'][seq] = dict(frames=len(rows), gt_sha256=digest(out), **stats[seq])
    write_json(root.parent / 'training_gt_conflicts.json', conflicts)
    report['conflicts_json_sha256'] = digest(root.parent / 'training_gt_conflicts.json')
    report['conflict_events'] = len(conflicts)
    write_json(root.parent / 'training_gt_manifest.json', report)
    print({key: value for key, value in report.items() if key != 'output'}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('audit', 'gt'))
    parser.add_argument('--tco-root', type=Path, required=True)
    args = parser.parse_args()
    tco = args.tco_root
    run = tco / 't23_v1'
    raw = Path(read_json(tco / 't01_v1/offline_settings.json')['paths']['images_root'])
    if args.mode == 'audit':
        source_overlap(raw, run / 'source_audit.json')
    else:
        make_training_gt(raw, run / 'training_gt', run / 'training_inputs/protocol.json')


if __name__ == '__main__':
    main()
