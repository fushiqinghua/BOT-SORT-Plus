"""Prepare full-sequence detector and RGB ReID caches for official train clips."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def read_json(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def indexed_paths(raw, modality, split):
    path = raw / 'annotations' / f'instances_{modality}_{split}2017.json'
    data = read_json(path)
    videos = {row['id']: row['name'] for row in data['videos']}
    seen = set()
    for row in data['images']:
        name = Path(row['file_name'])
        if name.parent.as_posix() != videos[row['video_id']] or name.parent.name != modality:
            raise ValueError(f'COCO image/video metadata conflict: {row}')
        if name in seen:
            raise ValueError(f'Duplicate indexed image path: {name}')
        seen.add(name)
    return seen, digest(path)


def prepare(config, out, gpus):
    t0, reference, project, raw = (config[k] for k in ('t0', 'reference', 'project', 'raw'))
    base = read_json(reference / 'protocol.json')
    if (base['weight_sha256'], base['reid_sha256']) != (t0['detector_sha256'], t0['rgb_encoder_sha256']):
        raise ValueError('T0 and baseline detector/ReID identity differ')
    detector = reference / 'weights/rgb_best.pt'
    reid = reference / 'weights/clip_veri.pt'
    if digest(detector) != base['weight_sha256'] or digest(reid) != base['reid_sha256']:
        raise ValueError('Frozen weight checks failed')
    rgb, rgb_sha = indexed_paths(raw, '00', 'train')
    ir, ir_sha = indexed_paths(raw, '01', 'train')
    rgb_test, test_sha = indexed_paths(raw, '00', 'test')
    ir_test, ir_test_sha = indexed_paths(raw, '01', 'test')
    if rgb & rgb_test or ir & ir_test:
        raise ValueError('Canonical official train/test paths overlap')
    paths = defaultdict(list)
    processed = project / 'data/processed_v1'
    files = [processed / f'{split}_pairs.jsonl' for split in ('train', 'val')]
    indexed = {'00': set(), '01': set()}
    for path in files:
        for row in read_jsonl(path):
            if row['official_split'] != 'train':
                raise ValueError(f'Unexpected official split in {path}')
            sequence = row['sequence']
            image, ir_img = Path(row['rgb']), Path(row['ir'])
            assert image.name == ir_img.name == row['source_filename']
            assert image.parent.parent.name == ir_img.parent.parent.name == sequence
            for modality, image_path in (('00', image), ('01', ir_img)):
                key = image_path.relative_to(raw / 'train2017')
                if not image_path.is_file() or key in indexed[modality]:
                    raise ValueError(f'Missing or reused indexed image {image_path}')
                indexed[modality].add(key)
            paths[sequence].append({'frame': row['frame'], 'image': str(image), 'thermal': str(ir_img)})
    if (indexed['00'], indexed['01']) != (rgb, ir) or len(paths) != 85:
        raise ValueError('Processed manifest does not cover exactly the 85 indexed sequences')
    for seq, rows in paths.items():
        rows.sort(key=lambda row: row['frame'])
        if [row['frame'] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f'Non-contiguous frame mapping for {seq}')
    hash_info = project / 'reports/image_hashes.jsonl'
    indexed_checks = {
        f'{split}2017/{path.as_posix()}'
        for split, paths_by_modality in (('train', (rgb, ir)), ('test', (rgb_test, ir_test)))
        for cam in paths_by_modality for path in cam
    }
    all_hashes = {row['path']: row['sha256'] for row in read_jsonl(hash_info) if row['path'] in indexed_checks}
    if set(all_hashes) != indexed_checks:
        raise ValueError(f'Incomplete archived image checksums: {len(all_hashes)} vs {len(indexed_checks)}')
    hashes_test = {key.split('/')[2]: set() for key in all_hashes if key.startswith('test2017/')}
    for key, sha in all_hashes.items():
        if key.startswith('test2017/'):
            hashes_test[key.split('/')[2]].add(sha)
    duplicates = Counter(key.split('/')[1] for key, sha in all_hashes.items()
                         if key.startswith('train2017/') and sha in hashes_test[key.split('/')[2]])
    in_scope = {seq: rows for seq, rows in sorted(paths.items()) if seq not in duplicates}
    totals = Counter({gpu: 0 for gpu in gpus})
    shards = {gpu: [] for gpu in gpus}
    for seq, rows in sorted(in_scope.items(), key=lambda item: (-len(item[1]), item[0])):
        gpu = min(totals, key=lambda k: (totals[k], k))
        shards[gpu].append(seq)
        totals[gpu] += len(rows)
    protocol = {k: base[k] for k in ('weight_sha256', 'reid_sha256', 'trackers', 'detector', 'fps', 'detector_mode')}
    protocol.update(sequences=in_scope, frames=sum(map(len, in_scope.values())),
                    evaluation='Frozen predictions only across complete indexed official-training clips')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'protocol.json', protocol)
    audit = dict(status='PLANNED', indexed_sequences=len(paths), indexed_frames=sum(map(len, paths.values())),
                 frames=protocol['frames'], cached_sequence_plan=len(in_scope),
                 excluded_exact_test_frame_duplicates=dict(duplicates), unused_unindexed_clip='DJI_0061_2',
                 official_image_index_sha256=dict(train_rgb=rgb_sha, train_ir=ir_sha, test_rgb=test_sha, test_ir=ir_test_sha),
                 processed_pair_sha256={p.name: digest(p) for p in files},
                 historical_image_hash_report_sha256=digest(hash_info),
                 detector_sha256=protocol['weight_sha256'], reid_sha256=protocol['reid_sha256'],
                 protocol_sha256=digest(out / 'protocol.json'), shards=shards, frames_by_gpu=dict(totals),
                 source_video_split='Independent pre-extraction asset preparation; no claims on mother-video provenance')
    write_json(out / 'plan_audit.json', audit)
    print(json.dumps({k: v for k, v in audit.items() if k != 'shards'}), flush=True)


def cover(cache, rows, weight_sha):
    expected_paths = [(r['frame'], r['image'], r['thermal']) for r in rows]
    found_paths = [(r['frame'], r['image'], r['thermal']) for r in cache['frames']]
    if expected_paths != found_paths or cache['model_sha256'] != weight_sha:
        raise ValueError('Frozen prediction source paths, frame IDs, or detector identity differ')


def run_shard(cfg, out, gpu):
    sys.path.insert(0, str(cfg['repo']))
    sys.path.insert(0, str(cfg['project'] / 'formal_detection_v1/framework'))
    import cv2
    import numpy as np
    import torch
    from botsort_plus.io import FeatureCache, forbid_ground_truth, read_detection_cache
    from ultralytics import YOLOMM

    sys.addaudithook(forbid_ground_truth)
    plan, protocol = (read_json(out / x) for x in ('plan_audit.json', 'protocol.json'))
    if plan['protocol_sha256'] != digest(out / 'protocol.json'):
        raise ValueError('Inference plan changed after validation')
    checkpoint = cfg['reference'] / 'weights/rgb_best.pt'
    reid = cfg['reference'] / 'weights/clip_veri.pt'
    if (digest(checkpoint), digest(reid)) != (protocol['weight_sha256'], protocol['reid_sha256']):
        raise ValueError('Frozen detector/ReID weight changed')
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    model = YOLOMM(str(checkpoint))
    if model.model.yaml['scale'] != 'm':
        raise ValueError('Detector architecture differs from frozen YOLO11-M')
    det_root, features = out / 'detections', out / 'features'
    det_root.mkdir(exist_ok=True)
    features.mkdir(exist_ok=True)
    sequences = plan['shards'][str(gpu)]
    options = dict(protocol['detector'], device='0', verbose=False, save=False)
    for seq in sequences:
        dest = det_root / f'{seq}.json.gz'
        rows = protocol['sequences'][seq]
        if dest.exists():
            cover(read_detection_cache(dest), rows, protocol['weight_sha256'])
            continue
        records = []
        for start in range(0, len(rows), options['batch']):
            batch = rows[start:start + options['batch']]
            source = {'A': [r['image'] for r in batch], 'B': [r['thermal'] for r in batch]}
            results = model.predict(sources=source, **options)
            if len(results) != len(batch):
                raise ValueError(f'Unexpected detector batch length in {seq}')
            for row, pred in zip(batch, results):
                assert Path(pred.paths['A']).resolve() == Path(row['image']).resolve()
                assert Path(pred.paths['B']).resolve() == Path(row['thermal']).resolve()
                assert pred.orig_imgs['A'].shape[:2] == pred.orig_imgs['B'].shape[:2] == (512, 640)
                boxes = np.asarray(pred.boxes, dtype=np.float32).reshape(-1, 6)
                assert np.isfinite(boxes).all() and np.isin(boxes[:, 5], np.arange(7)).all()
                assert (boxes[:, 2:4] > boxes[:, :2]).all()
                records.append(dict(row, detections=boxes.tolist()))
        temp = dest.with_suffix('.partial')
        with gzip.open(temp, 'wt', encoding='utf-8', compresslevel=2) as stream:
            json.dump(dict(model_sha256=protocol['weight_sha256'], frames=records), stream, ensure_ascii=False,
                      allow_nan=False, separators=(',', ':'))
        temp.replace(dest)
        print('DETECT', seq, len(records), sum(len(row['detections']) for row in records), flush=True)
    del model
    torch.cuda.empty_cache()
    missing = []
    for seq in sequences:
        feature = features / f'{seq}.npz'
        if feature.is_file() and feature.with_suffix('.json').is_file():
            cache = read_detection_cache(det_root / f'{seq}.json.gz')
            FeatureCache(feature, cache, expected_rgb_sha=protocol['reid_sha256'])
        else:
            if feature.exists() or feature.with_suffix('.json').exists():
                raise FileExistsError(f'Interrupted feature pair requires inspection: {feature}')
            missing.append(seq)
    if missing:
        command = [sys.executable, '-Bu', str(cfg['repo'] / 'tools/export_botsort_plus_features.py'),
                   '--detections-dir', str(det_root), '--images-root', str(cfg['raw']), '--rgb-weights', str(reid),
                   '--output-dir', str(features), '--device', 'cuda:0', '--runtime-root', str(cfg['runtime_root']),
                   '--batch-size', '64']
        command.extend(v for seq in missing for v in ('--sequence', seq))
        subprocess.run(command, cwd=cfg['repo'], check=True)
    print('DONE', gpu, len(sequences), flush=True)


def verify(config, out):
    sys.path.insert(0, str(config['repo']))
    from botsort_plus.io import FeatureCache, detection_fingerprint, read_detection_cache
    from botsort_plus.io import forbid_ground_truth

    sys.addaudithook(forbid_ground_truth)
    plan = read_json(out / 'plan_audit.json')
    protocol = read_json(out / 'protocol.json')
    if plan['protocol_sha256'] != digest(out / 'protocol.json'):
        raise ValueError('Inference plan changed after validation')
    counts, per_sequence = Counter(), {}
    for seq, rows in protocol['sequences'].items():
        det_path = out / 'detections' / f'{seq}.json.gz'
        feature_path = out / 'features' / f'{seq}.npz'
        cache = read_detection_cache(det_path)
        cover(cache, rows, protocol['weight_sha256'])
        features = FeatureCache(feature_path, cache, expected_rgb_sha=protocol['reid_sha256'])
        coverage = features.metadata['coverage']
        per_sequence[seq] = dict(frames=len(rows), coverage=coverage, detection_sha256=digest(det_path),
                                ordered_detection_fingerprint=detection_fingerprint(cache),
                                rgb_features_sha256=features.metadata['features_sha256'],
                                rgb_metadata_sha256=digest(feature_path.with_suffix('.json')))
        counts.update(coverage)
    report = dict(status='PASS', sequences=len(per_sequence), frames=protocol['frames'], total_coverage=dict(counts),
                  plan_sha256=digest(out / 'plan_audit.json'), protocol_sha256=plan['protocol_sha256'],
                  detector_sha256=protocol['weight_sha256'], rgb_encoder_sha256=protocol['reid_sha256'],
                  per_sequence=per_sequence)
    write_json(out / 'verified.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'per_sequence'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'shard', 'verify'))
    parser.add_argument('--tco-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[5, 6, 7])
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    root = args.tco_root
    settings = read_json(root / 't01_v1/offline_settings.json')
    raw = Path(settings['paths']['images_root'])
    project = raw.parents[2]
    previous = read_json(root.parent / 'round3_joint_e62bb9b8/settings.json')
    config = dict(project=project, raw=raw, repo=root.parent / 'BOT-SORT-Plus-followup-53faeccb',
                  reference=project / 'outputs/yolo11m_rgb_rgbt_mot_test_v1/rgbt',
                  runtime_root=Path(previous['runtime_root']), t0=read_json(root / 't01_v1/T0/input_manifest.json'))
    if args.stage == 'plan':
        prepare(config, args.output, args.gpus)
    elif args.stage == 'shard':
        if args.gpu is None or os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
            raise ValueError('CUDA_VISIBLE_DEVICES must bind the physical GPU for this shard')
        run_shard(config, args.output, args.gpu)
    else:
        verify(config, args.output)


if __name__ == '__main__':
    main()
