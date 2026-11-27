"""Freeze the quality revision split from source metadata, before any new outcomes."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random


SEED = 41


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def partition(lengths, groups):
    return dict(sequences=dict(sorted(lengths.items())),
                source_videos=sorted({groups[s] for s in lengths}),
                frames=sum(lengths.values()))


def freeze(previous, mapping_path, verified_path, source_audit_path):
    old = read(previous)
    verified = read(verified_path)
    audit = read(source_audit_path)
    if old['schema'] != 'tco_v0.1_training_split' or old['status'] != 'LOCKED':
        raise ValueError('Expected the frozen T2 training partition')
    if (old['evidence']['source_video_mapping_sha256'] != digest(mapping_path)
            or old['duplicate_audit_sha256'] != digest(source_audit_path)):
        raise ValueError('Proxy grouping or duplicate audit differs from T2')
    if verified['status'] != 'PASS' or audit['status'] != 'PASS':
        raise ValueError('An audited official training cache is required')
    with mapping_path.open(encoding='utf-8-sig', newline='') as stream:
        groups = {r['sequence']: r['source_video'] for r in csv.DictReader(stream)
                  if r['official_split'] == 'train'}
    indexed = set(audit['indexed_training_sequences'])
    old_training = old['training_sequence_to_source_video']
    extra = set(old['excluded_indexed_training_sequences'])
    dev = old['development_sequence_to_source_video']
    if (len(indexed) != 85 or indexed != set(verified['per_sequence']) or set(groups) != indexed
            or indexed != set(old_training) | extra or set(old_training) & extra or indexed & dev.keys()
            or len(extra) != 11 or set(groups[s] for s in extra) - set(dev.values())
            or audit['cross_split_duplicate_sequences']):
        raise ValueError('Audited 85/30 source contract changed')
    if any(groups[s] != group for s, group in old_training.items()):
        raise ValueError('Previously retained training group assignment changed')
    prior = old['split']
    all_prior = {seq for phase in ('fit', 'calibration', 'holdout') for seq in prior[phase]['sequences']}
    if set(old_training) != all_prior or any(
            set(prior[a]['sequences']) & set(prior[b]['sequences'])
            for a, b in [('fit', 'calibration'), ('fit', 'holdout'), ('calibration', 'holdout')]):
        raise ValueError('Previously locked roles are inconsistent')
    shuffled = sorted({groups[s] for s in extra})
    if len(shuffled) != 8:
        raise ValueError('The 11 newly admitted training clips must cover exactly eight proxy groups')
    random.Random(SEED).shuffle(shuffled)
    acceptance_groups = set(shuffled[:4])
    extra_lengths = {s: verified['per_sequence'][s]['frames'] for s in sorted(extra)}
    new_fit = {s: n for s, n in extra_lengths.items() if groups[s] not in acceptance_groups}
    new_holdout = {s: n for s, n in extra_lengths.items() if groups[s] in acceptance_groups}
    fit = prior['fit']['sequences'] | new_fit
    cal = prior['calibration']['sequences']
    diagnosis = prior['holdout']['sequences']
    selected = set(fit) | set(cal) | set(new_holdout)
    if selected & set(diagnosis) or selected | set(diagnosis) != indexed:
        raise ValueError('The new learning and assessment roles must preserve official scope')
    splits = {name: partition(lengths, groups) for name, lengths in
              [('fit', fit), ('calibration', cal), ('holdout', new_holdout)]}
    if sum(len(p['source_videos']) for p in splits.values()) != len({groups[s] for s in selected}):
        raise ValueError('Proxy groups overlap between fit, calibration and acceptance')
    return dict(
        schema=old['schema'], status='LOCKED', seed=SEED,
        protocol_status='EXPLORATORY_TCO_Q2_TRAIN_TEST_PROXY_OVERLAP',
        assignment='Sort eight newly retained training proxy groups; shuffle with seed 41; '
                   'first four groups are untouched prior-TCO acceptance, final four fit. '
                   'Preserve old fit and calibration roles and quarantine old holdout for diagnosis.',
        split=splits, training_sequence_to_source_video={s: groups[s] for s in sorted(selected)},
        development_sequence_to_source_video=dev,
        diagnosis_only=partition(diagnosis, groups), newly_admitted={
            'fit': partition(new_fit, groups), 'holdout': partition(new_holdout, groups)},
        excluded_unindexed_training_sequences=old['excluded_unindexed_training_sequences'],
        evidence=dict(prior_split_sha256=digest(previous), source_video_mapping_sha256=digest(mapping_path),
                      source_audit_sha256=digest(source_audit_path), frozen_training_cache_sha256=digest(verified_path)),
        duplicate_audit_sha256=digest(source_audit_path), source_video_provenance_verified=False,
        training_data_only_model_selection=True,
        proxy_overlap_exception=dict(
            allowed_groups=sorted({groups[s] for s in extra}),
            authorization='The user requested retaining these 11 official training clips and permitted fit use.',
            official_development_sequences_or_gt_used=False,
            limitation='A proxy group may span training and official test. Official test is development-only. '
                       'Image hash deduplication does not verify disjoint mother videos.'),
        acceptance_provenance='Prior TCO fit/calibration/capture excluded these clips. Frozen detector/ReID may '
                              'have seen official training data. Previous detectors and cache extraction are not '
                              'the TCO quality-head exposure scope.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('old-split', 'proxy-map', 'verified-cache', 'source-audit', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.old_split, args.proxy_map, args.verified_cache, args.source_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for name, part in result['split'].items():
        print(name, len(part['sequences']), len(part['source_videos']), part['frames'])
    print('diagnosis_only', len(result['diagnosis_only']['sequences']))


if __name__ == '__main__':
    main()
