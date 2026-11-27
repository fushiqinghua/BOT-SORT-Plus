"""Prepare newly admitted training clips and keep the previous diagnosis out of fitting."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import gzip
import json
from pathlib import Path

from build_corpus import load_image_function, read, reviewed_positives, sha256, write_partition


EXTRA_ROLES = ('fit', 'holdout')
NAMES = ('pairs.runtime.jsonl.gz', 'pairs.labels.jsonl.gz', 'episodes.jsonl.gz')
METHOD = 'B_early_moderate'


def dump(path, record):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def partitions(split, previous):
    by_role = {k: v['sequences'] for k, v in split['split'].items()}
    old_by_role = {k: v['sequences'] for k, v in previous['split'].items()}
    added_fit = {k: v for k, v in by_role['fit'].items() if k not in old_by_role['fit']}
    if (any(by_role['fit'].get(k) != v for k, v in old_by_role['fit'].items())
            or by_role['calibration'] != old_by_role['calibration']):
        raise ValueError('Previous fit/calibration source membership changed unexpectedly')
    groups = split['training_sequence_to_source_video']
    for i, name in enumerate(by_role):
        selected = {groups[k] for k in by_role[name]}
        for other in list(by_role)[i+1:]:
            if selected & {groups[k] for k in by_role[other]}:
                raise ValueError('Q2 training-side source grouping overlaps')
    extras = {'fit': added_fit, 'holdout': by_role['holdout']}
    extra_names = set(added_fit) | set(by_role['holdout'])
    if (len(extra_names) != 11 or extra_names != set(previous['excluded_indexed_training_sequences'])
            or set(by_role['holdout']) & set(old_by_role['holdout'])):
        raise ValueError('Revised scope is not the agreed 11 additional clips')
    return extras


def label(args, extras, split):
    import label_t1
    from experiments.vttiny_gmc.round3_joint import analysis

    t0 = read(args.t0 / 'T0/input_manifest.json')
    settings = read(args.t0 / 'offline_settings.json')['paths']
    trackeval, verifier = analysis.load_trackeval(settings['trackeval_root'])
    groups = split['training_sequence_to_source_video']
    runs = args.root / 'extra_capture'
    protocol_path = runs / 'B/protocols' / (METHOD + '.json')
    protocol = read(protocol_path)
    lengths = {k: v for mapping in extras.values() for k, v in mapping.items()}
    receipt = read(runs / 'capture_manifest.json')
    if (receipt['status'] != 'PASS' or receipt['labels_in_prediction'] is not False
            or receipt['split_sha256'] != sha256(args.split)
            or receipt['protocol_sha256'] != sha256(protocol_path)):
        raise ValueError('Extra training capture is not verified for this split')
    if (protocol['sequences'] != lengths or protocol['detector_sha256'] != t0['detector_sha256']
            or protocol['rgb_encoder_sha256'] != t0['rgb_encoder_sha256']):
        raise ValueError('The 11-clip capture differs from the locked scope or weights')
    residual = read(args.repo / 'configs/research/tco_v0.1_protocol.json')['model']['box_residual']
    output = args.root / 'extra_labels'
    output.mkdir(exist_ok=False)
    scopes = {}
    for phase in EXTRA_ROLES:
        destination = output / phase
        destination.mkdir()
        totals, unique = defaultdict(Counter), defaultdict(set)
        context = runs, args.old / 'training_gt', lengths, destination, trackeval, residual, groups, unique
        with ExitStack() as stack:
            streams = [stack.enter_context(gzip.open(destination / n, 'xt', encoding='utf-8', compresslevel=2))
                       for n in NAMES]
            for clip in sorted(extras[phase]):
                cache = args.old / 'training_inputs/detections' / (clip + '.json.gz')
                label_t1.process_sequence(context, 'B', METHOD, clip, protocol, cache, streams, totals)
                print(phase, clip, dict(totals[METHOD, 'sequence', clip]).get('assigned_pairs', 0), flush=True)
        scopes[phase] = dict(sequences=extras[phase], counts=dict(totals[METHOD, 'all', 'ALL']),
                             files={n: sha256(destination/n) for n in NAMES})
    dump(output / 'training_labels.json', dict(partitions=scopes, capture_protocol_sha256=sha256(protocol_path),
        split_sha256=sha256(args.split), trackeval_commit=verifier['commit'], development_gt_read=False))


def review(args):
    from hard_review import dataset_candidates, render
    from build_corpus import KEYS

    labels = args.root / 'extra_labels'
    location = args.root / 'extra_review'
    location.mkdir(exist_ok=False)
    images = read(args.t0 / 'offline_settings.json')['paths']['images_root']
    loader = load_image_function(Path(images).resolve())
    index, summary = [], {}
    for part in EXTRA_ROLES:
        cases, total = dataset_candidates(labels / part, 41, 63)
        summary[part] = dict(reviewable=total, shown=len(cases), groups=len({a['source_video'] for a, _ in cases}))
        index.extend(render(location, part, cases, loader))
    file = location / 'index.csv'
    with file.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case_id', 'partition', *KEYS, 'source_video',
                                                    'gt_id', 'raw_iou', 'feasible_iou'])
        writer.writeheader()
        writer.writerows(index)
    dump(location / 'manifest.json', dict(seed=41, per_partition_budget=63, partitions=summary,
         labels_sha256=sha256(labels / 'training_labels.json'), index_sha256=sha256(file)))
    print(summary, flush=True)


def combine_fit(old, extra, destination, n_old, n_extra):
    import numpy as np

    destination.mkdir()
    names = sorted(x.name for x in old.glob('*.npy'))
    if names != sorted(x.name for x in extra.glob('*.npy')):
        raise ValueError('Prior and additional fit arrays have different schema')
    shapes = {}
    for name in names:
        first, second = (np.load(folder / name, mmap_mode='r', allow_pickle=False) for folder in (old, extra))
        if first.dtype != second.dtype or first.shape[1:] != second.shape[1:]:
            raise ValueError(f'Fit format changed: {name}')
        joined = np.lib.format.open_memmap(destination / name, mode='w+', dtype=first.dtype,
                                           shape=(n_old + n_extra, *first.shape[1:]))
        joined[:n_old] = first[:n_old]
        tail = second[:n_extra]
        if name == 'track_key.npy':
            tail = tail + int(first[:n_old].max()) + 1
        joined[n_old:] = tail
        joined.flush()
        shapes[name[:-4]] = dict(dtype=str(joined.dtype), shape=list(joined.shape))
    return shapes


def corpus(args, extras, split):
    import cv2
    from model import __file__ as model_path

    cv2.setNumThreads(1)
    old = args.old / 'T2_corpus'
    original = read(old / 'crop_manifest.json')
    if original['model_source_sha256'] != sha256(model_path):
        raise ValueError('Changed crop or base model source')
    labels = args.root / 'extra_labels'
    recorded = read(labels / 'training_labels.json')
    if recorded['split_sha256'] != sha256(args.split):
        raise ValueError('New cut differs from the label scope')
    groups = split['training_sequence_to_source_video']
    sources, clips = (dict(original[k]) for k in ('source_ids', 'sequence_ids'))
    for clip in sorted(set(extras['fit']) | set(extras['holdout'])):
        if clip in clips or groups[clip] in original['source_ids']:
            raise ValueError(f'New acceptance input was present in the earlier corpus: {clip}')
        clips[clip] = len(clips)
        sources.setdefault(groups[clip], len(sources))
    votes_dir = args.root / 'extra_votes'
    decisions = reviewed_positives(args.root / 'extra_review', votes_dir, labels / 'training_labels.json')
    images = Path(read(args.t0 / 'offline_settings.json')['paths']['images_root']).resolve()
    loader = load_image_function(images)
    output = args.root / 'corpus'
    intermediate = args.root / 'extra_crops'
    output.mkdir(exist_ok=False)
    intermediate.mkdir(exist_ok=False)
    generated = {}
    for part in EXTRA_ROLES:
        for n in NAMES[:2]:
            if sha256(labels / part / n) != recorded['partitions'][part]['files'][n]:
                raise ValueError(f'Changed labelled event stream: {part}/{n}')
        limited = {clip: groups[clip] for clip in extras[part]}
        generated[part] = write_partition(labels / part, intermediate / part, limited, sources, clips, loader,
                                          decisions[part])
        print(part, generated[part]['counts'], flush=True)
    base, added = original['partitions']['fit'], generated['fit']
    shape = combine_fit(old / 'fit', intermediate / 'fit', output / 'fit', base['events'], added['events'])
    counts = Counter(base['counts']) + Counter(added['counts'])
    n = base['events'] + added['events']
    (output / 'calibration').symlink_to(old / 'calibration', target_is_directory=True)
    (output / 'holdout').symlink_to(intermediate / 'holdout', target_is_directory=True)
    scope = dict(fit=dict(events=n, capacity=n, counts=dict(counts), arrays=shape),
                 calibration=original['partitions']['calibration'], holdout=generated['holdout'])
    dump(output / 'crop_manifest.json', dict(schema=original['schema'], status='LOCKED',
        protocol_status=split['protocol_status'], model_source_sha256=sha256(model_path),
        split_sha256=sha256(args.split), dataset_from_official_training_only=True,
        old_crop_manifest_sha256=sha256(old / 'crop_manifest.json'),
        new_labels_sha256=sha256(labels / 'training_labels.json'),
        independent_votes_sha256={f.name: sha256(f) for f in sorted(votes_dir.glob('*.csv'))},
        partitions=scope, source_ids=sources, sequence_ids=clips,
        old_holdout_role='diagnosis_only_no_active_head_exposure'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('label', 'review', 'corpus'))
    for field in ('root', 'old', 't0', 'repo', 'split'):
        parser.add_argument('--' + field, type=Path, required=True)
    opts = parser.parse_args()
    current = read(opts.split)
    previous = read(opts.old / 'split/training_split.json')
    derived = partitions(current, previous)
    if opts.stage == 'review':
        review(opts)
    else:
        globals()[opts.stage](opts, derived, current)
