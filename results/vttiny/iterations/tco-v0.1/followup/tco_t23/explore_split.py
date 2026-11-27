"""Register the authorized exploratory split using conservative grouping and exact deduplication."""

import argparse
import csv
import json
from pathlib import Path
import re

from freeze_split import freeze, read, sha256


def register(t1_manifest, audit_path, output):
    t1 = read(t1_manifest)
    audit = read(audit_path)
    train = t1['official_training_annotation']['frames_by_sequence']
    dev = t1['official_development_annotation']['frames_by_sequence']
    if (audit['status'] != 'PASS' or set(audit['indexed_training_sequences']) != set(train)
            or set(audit['development_sequences']) != set(dev)
            or audit['training_rgb_frames'] != sum(train.values())
            or audit['development_rgb_frames'] != sum(dev.values())):
        raise ValueError('Exact deduplication audit does not cover the registered input')
    groups = {}
    for sequence in (*train, *dev):
        match = re.fullmatch(r'(DJI_[0-9]+)_[0-9]+', sequence)
        if not match:
            raise ValueError(f'Unknown grouping pattern {sequence}')
        groups[sequence] = match.group(1)
    parents = {group: group for group in groups.values()}

    def find(group):
        while group != parents[group]:
            group = parents[group]
        return group

    def join(left, right):
        earlier, later = sorted((find(groups[left]), find(groups[right])))
        parents[later] = earlier

    for train_seq, dev_seqs in audit['cross_split_duplicate_sequences'].items():
        if train_seq not in train or not set(dev_seqs) <= dev.keys():
            raise ValueError('Duplicate record outside official sequence indexes')
        for dev_seq in dev_seqs:
            join(train_seq, dev_seq)
    for left, right in audit['same_train_duplicate_sequence_pairs']:
        if not {left, right} <= train.keys():
            raise ValueError('Within-training duplicate record outside official index')
        join(left, right)
    output.mkdir(parents=True, exist_ok=False)
    mapping_path = output / 'proxy_group_map.csv'
    with mapping_path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('official_split', 'sequence', 'source_video', 'provenance'))
        for phase, lengths in (('train', train), ('test', dev)):
            for sequence in sorted(lengths):
                writer.writerow((phase, sequence, 'proxy:' + find(groups[sequence]),
                                 'DJI name grouping; exact-duplicate-linked groups merged; not verified original-video metadata'))
    result = freeze(t1_manifest, mapping_path)
    result.update(protocol_status='EXPLORATORY_PROXY_GROUPING',
                  source_video_provenance_verified=False,
                  duplicate_audit_sha256=sha256(audit_path),
                  exclusion_rule='Exclude all proxy groups shared with official test, including byte-duplicate-linked groups',
                  training_data_only_model_selection=True)
    with (output / 'training_split.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--t1-manifest', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = register(args.t1_manifest, args.audit, args.output)
    for part, values in result['split'].items():
        print(part, len(values['source_videos']), len(values['sequences']), values['frames'])
    print('Excluded', len(result['excluded_indexed_training_sequences']))
