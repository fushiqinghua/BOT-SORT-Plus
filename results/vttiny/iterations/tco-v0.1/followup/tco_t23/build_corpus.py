"""Build one shared causal RGB corpus from frozen training-only candidate and label streams."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path


PARTITIONS = ('fit', 'calibration', 'holdout')
KEYS = ('run_fingerprint', 'method', 'sequence', 'frame', 'track_id', 'det_index')


def sha256(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def records(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def events(folder):
    runtime = records(folder / 'pairs.runtime.jsonl.gz')
    labels = records(folder / 'pairs.labels.jsonl.gz')
    for observation, label in zip(runtime, labels, strict=True):
        if any(observation[key] != label[key] for key in KEYS):
            raise ValueError('Training labels and causal capture keys diverge')
        yield observation, label


def load_image_function(image_root):
    import cv2
    import numpy as np

    @lru_cache(maxsize=18)
    def decode(name):
        reference = Path(name)
        path = (image_root / reference).resolve()
        if not path.is_relative_to(image_root) or 'train2017' not in path.relative_to(image_root).parts:
            raise ValueError(f'Unexpected non-training crop input {path}')
        pixels = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if pixels is None:
            raise ValueError(f'Cannot decode {path}')
        return pixels, hashlib.sha256(pixels.tobytes()).hexdigest()

    def load(row):
        image, checksum = decode(row['frame_image'])
        if checksum != row['image_sha256'] or list(image.shape) != row['image_shape']:
            raise ValueError(f'Historical capture/source RGB mismatch: {row["frame_image"]}')
        return image

    return load


def reviewed_positives(review, votes, label_manifest):
    manifest = read(review / 'manifest.json')
    if manifest['labels_sha256'] != sha256(label_manifest) or manifest['index_sha256'] != sha256(review / 'index.csv'):
        raise ValueError('Independent training review does not belong to these labels')
    with (review / 'index.csv').open(encoding='utf-8-sig', newline='') as stream:
        index = {r['case_id']: r for r in csv.DictReader(stream)}
    decisions = {}
    for file in sorted(votes.glob('*.csv')):
        with file.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                case = row['case_id']
                if (case not in index or case in decisions or not row['note'].strip()
                        or row['blind_identity'] not in ('SUPPORTED', 'AMBIGUOUS', 'DIFFERENT')):
                    raise ValueError(f'Invalid independent training-recheck record {case}')
                decisions[case] = row
    if set(decisions) != set(index):
        raise ValueError(f'Incomplete blinded training review: {len(decisions)}/{len(index)}')
    result = {part: {} for part in PARTITIONS}
    for case, decision in decisions.items():
        if decision['blind_identity'] == 'SUPPORTED':
            metadata = index[case]
            string_key = tuple(metadata[k] for k in KEYS)
            result[metadata['partition']][string_key] = case
    return result


def write_partition(folder, output, selected, source_ids, sequence_ids, load_image, review):
    import numpy as np
    from model import make_sample

    # Sizing pass keeps storage out of memory. The valid prefix is recorded after
    # shared crop eligibility, which must be identical across all model arms.
    capacity = sum(row['online_eligible'] for row in records(folder / 'pairs.runtime.jsonl.gz'))
    if capacity <= 0:
        raise ValueError(f'No eligible events in {folder}')
    output.mkdir()
    counts = Counter()
    arrays = {}
    tracks = {}
    supervised = set()
    used = 0
    lookup = {'ignore': -1, 'negative': 0, 'positive': 1}
    with gzip.open(output / 'row_keys.jsonl.gz', 'wt', encoding='utf-8', compresslevel=2) as index:
        for row, label in events(folder):
            seq = row['sequence']
            if seq not in selected:
                raise ValueError(f'Unregistered sequence in T2 stream: {seq}')
            if row['source_video'] != selected[seq]:
                raise ValueError(f'Source-video group changed during replay: {seq}')
            counts['all_original_edges'] += 1
            if not row['online_eligible']:
                counts['runtime_ineligible'] += 1
                continue
            pixels = load_image(row)
            sample, reason = make_sample(pixels, row['raw_xyxy'], row['pred_xyxy'], row['raw_score'],
                                         row['frame'], row['history'], load_image,
                                         arm='P3', include_current_repeat=True)
            if sample is None:
                counts['crop_fallback_' + str(reason)] += 1
                continue
            key = (row['run_fingerprint'], seq, row['track_id'])
            track_key = tracks.setdefault(key, len(tracks))
            status = label['support_label']
            correction_id = review.get(tuple(str(row[k]) for k in KEYS))
            if correction_id:
                if (status != 'ignore' or not row['assigned'] or not label['same_gt_clear_output']
                        or not label['finite_candidate_crosses_050'] or not 0 < label['raw_iou'] < .5):
                    raise ValueError(f'Invalid positive override {correction_id}')
                supervised.add(correction_id)
                status = 'positive'
                counts['reviewed_low_IoU_positive'] += 1
            target = label['gt_xyxy'] if label['gt_xyxy'] is not None else [0., 0., 0., 0.]
            if status == 'positive' and (target[2] <= target[0] or target[3] <= target[1]):
                raise ValueError(f'Positive label without a valid GT box for {key}')
            sample.update(support_label=np.int8(lookup[status]), gt_xyxy=np.asarray(target, dtype=np.float32),
                          assigned=np.bool_(row['assigned']), source_id=np.int32(source_ids[selected[seq]]),
                          track_key=np.int64(track_key), sequence_id=np.int32(sequence_ids[seq]),
                          frame=np.int32(row['frame']))
            if not arrays:
                arrays = {name: np.lib.format.open_memmap(output / (name + '.npy'), mode='w+',
                          dtype=np.asarray(value).dtype, shape=(capacity, *np.shape(value)))
                          for name, value in sample.items()}
            for name, array in arrays.items():
                array[used] = sample[name]
            counts[status] += 1
            counts[('assigned_' if row['assigned'] else 'unassigned_') + status] += 1
            index.write(json.dumps({k: row[k] for k in KEYS} | dict(
                row=used, history=[(v['frame'], v['det_index']) for v in row['history']],
                gt_id=label['gt_id'], raw_iou=label['raw_iou'],
                feasible_iou=label['feasible_iou'], identity_status=label['identity_status'],
                review_positive_case_id=correction_id),
                ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
            used += 1
    if not used or not counts['positive'] or not counts['negative']:
        raise ValueError(f'T2 lacks known positive or negative examples in {folder}: {counts}')
    counts['reviewed_positive_with_crop_fallback'] = len(set(review.values()) - supervised)
    for array in arrays.values():
        array.flush()
    return dict(events=used, capacity=capacity, counts=dict(counts), arrays={key: dict(
        dtype=str(value.dtype), shape=list(value.shape)) for key, value in arrays.items()},
        row_keys_sha256=sha256(output / 'row_keys.jsonl.gz'))


def prepare_partition(folder, destination, selected, source_ids, sequence_ids, image_root, review):
    import cv2

    cv2.setNumThreads(1)
    return write_partition(folder, destination, selected, source_ids, sequence_ids,
                           load_image_function(image_root), review)


def build(args):
    from model import __file__ as model_source

    split = read(args.split)
    manifest = args.labels / 'training_labels.json'
    summary = read(manifest)
    if split['status'] != 'LOCKED' or summary['split_sha256'] != sha256(args.split):
        raise ValueError('Training label split differs from the locked split')
    groups = split['training_sequence_to_source_video']
    source_ids = {key: i for i, key in enumerate(sorted(set(groups.values())))}
    sequence_ids = {name: i for i, name in enumerate(sorted(groups))}
    independent = reviewed_positives(args.review, args.votes, manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    partitions = {}
    with ProcessPoolExecutor(max_workers=len(PARTITIONS)) as executor:
        tasks = {}
        for part in PARTITIONS:
            folder = args.labels / part
            for key in ('pairs.runtime.jsonl.gz', 'pairs.labels.jsonl.gz'):
                if sha256(folder / key) != summary['partitions'][part]['files'][key]:
                    raise ValueError(f'Changed training candidate source: {part}/{key}')
            selected = {seq: groups[seq] for seq in split['split'][part]['sequences']}
            tasks[part] = executor.submit(prepare_partition, folder, args.output / part, selected,
                                          source_ids, sequence_ids, args.images.resolve(), independent[part])
        for part, task in tasks.items():
            partitions[part] = task.result()
            print(part, partitions[part]['counts'], flush=True)
    result = dict(schema='tco_v0.1_crop_corpus', status='LOCKED', protocol_status=split.get('protocol_status', 'VERIFIED'),
                  model_source_sha256=sha256(model_source), training_labels_sha256=sha256(args.labels / 'training_labels.json'),
                  split_sha256=sha256(args.split), dataset_from_official_training_only=True,
                  partitions=partitions, source_ids=source_ids, sequence_ids=sequence_ids,
                  array_valid_prefix='Only rows before partitions[part].events are valid.',
                  independent_review=dict(manifest_sha256=sha256(args.review / 'manifest.json'),
                      votes_sha256={p.name: sha256(p) for p in sorted(args.votes.glob('*.csv'))},
                      low_iou_positive_scope='Model blind RGB identity support plus historical anchor and CLEAR link; exploratory labels'))
    (args.output / 'crop_manifest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('labels', 'split', 'images', 'output', 'review', 'votes'):
        parser.add_argument('--' + name, type=Path, required=True)
    build(parser.parse_args())
