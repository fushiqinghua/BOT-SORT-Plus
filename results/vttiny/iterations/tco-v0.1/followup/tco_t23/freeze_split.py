"""Register a TCO training partition from independently sourced video provenance."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random


SEED = 41
PARTS = ('fit', 'calibration', 'holdout')


def read(path):
    return json.loads(path.read_bytes())


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_mapping(path):
    items = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            source = row['source_video'].strip()
            split = row['official_split'].strip()
            identity = row['sequence'].strip()
            evidence = row['provenance'].strip()
            key = (split, identity)
            if key in items or not all((identity, source, evidence)) or split not in ('train', 'test'):
                raise ValueError(f'Duplicate or incomplete video provenance: {key}')
            items[key] = {'source_video': source, 'provenance': evidence}
    return items


def freeze(t1_manifest, mapping_file):
    t1 = read(t1_manifest)
    if t1['schema'] != 'tco_v0.1_split_manifest':
        raise ValueError('Expected the T1 registration manifest')
    train_lengths = t1['official_training_annotation']['frames_by_sequence']
    development = t1['official_development_annotation']['frames_by_sequence']
    if len(train_lengths) != 85 or len(development) != 30 or (set(train_lengths) & set(development)):
        raise ValueError('Verified indexed official train/development sequences differ from registered scope')
    if sum(development.values()) != 12590:
        raise ValueError('Unexpected number of official development frames')
    mapping = load_mapping(mapping_file)
    registered = {(phase, name) for phase, lengths in [('train', train_lengths), ('test', development)] for name in lengths}
    unexpected = set(mapping) - registered
    if unexpected:
        raise ValueError(f'Unregistered or duplicate train/test source rows: {sorted(unexpected)}')
    missing_development = set(development) - {seq for phase, seq in mapping if phase == 'test'}
    if missing_development:
        raise ValueError(f'Development mother-video provenance is incomplete: {sorted(missing_development)}')
    forbidden = {mapping['test', seq]['source_video'] for seq in development}
    allowed = {}
    excluded = {}
    for name in sorted(train_lengths):
        row = mapping.get(('train', name))
        if row is None:
            excluded[name] = 'missing_verified_source_video'
        elif row['source_video'] in forbidden:
            excluded[name] = 'source_video_also_in_development'
        else:
            allowed[name] = row['source_video']
    groups = sorted(set(allowed.values()))
    if len(groups) < 3:
        raise ValueError('At least three eligible mother videos are required for fit/calibration/holdout')
    random.Random(SEED).shuffle(groups)
    n_other = max(1, round(0.15 * len(groups)))
    partitions = dict(zip(PARTS, (groups[2*n_other:], groups[:n_other], groups[n_other:2*n_other]), strict=True))
    splits = {}
    for part, parents in partitions.items():
        lengths = {seq: train_lengths[seq] for seq, parent in allowed.items() if parent in parents}
        splits[part] = dict(source_videos=sorted(parents), sequences=lengths, frames=sum(lengths.values()))
    return dict(schema='tco_v0.1_training_split', status='LOCKED', seed=SEED,
                assignment='Seeded 70/15/15 by source-video counts with rounding to nearest and nonempty partitions',
                split=splits, training_sequence_to_source_video=allowed,
                development_sequence_to_source_video={s: mapping['test', s]['source_video'] for s in development},
                excluded_indexed_training_sequences=excluded,
                excluded_unindexed_training_sequences=sorted(set(t1['official_training_annotation']['registered_sequences'])
                                                             - set(train_lengths)),
                evidence=dict(t1_manifest_sha256=sha256(t1_manifest), source_video_mapping_sha256=sha256(mapping_file)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--t1-manifest', type=Path, required=True)
    parser.add_argument('--source-mapping', type=Path, required=True,
                        help='Verified CSV with official_split,sequence,source_video,provenance')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.t1_manifest, args.source_mapping)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
