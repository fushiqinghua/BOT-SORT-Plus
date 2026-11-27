"""Capture only the eleven newly admitted annotated training clips with frozen B."""

import argparse
import json
from pathlib import Path
import sys

from freeze_q2 import digest, freeze, read


def capture(args):
    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(args.online))
    import batch
    from experiments.vttiny_gmc.round3_joint.common import write_json

    args.train_features = args.train_baseline / 'features'
    verified_path = args.train_baseline / 'verified.json'
    locked = read(args.split)
    reproduced = freeze(args.old_split, args.proxy_map, verified_path, args.source_audit)
    if locked != reproduced:
        raise ValueError('Quality-revision split differs from the locked prescription')
    extra = locked['newly_admitted']
    lengths = extra['fit']['sequences'] | extra['holdout']['sequences']
    groups = {s: locked['training_sequence_to_source_video'][s] for s in lengths}
    verified = read(verified_path)
    files = {seq: {} for seq in lengths}
    for seq in lengths:
        expected = verified['per_sequence'][seq]
        inputs = {'detection_sha256': args.train_baseline / 'detections' / (seq + '.json.gz'),
                  'rgb_features_sha256': args.train_features / (seq + '.npz'),
                  'rgb_metadata_sha256': args.train_features / (seq + '.json')}
        for key, path in inputs.items():
            checksum = digest(path)
            if expected[key] != checksum:
                raise ValueError(f'Frozen training input differs for {seq}/{key}')
            files[seq][key] = checksum
    settings = batch.train_prediction(args, lengths)
    output = batch.new_output(args.output)
    group_path = batch.settings_files(output, settings, groups)
    shards = batch.collect_shards(args, output, settings, lengths, group_path)
    batch.merge_capture(output, lengths, shards)
    method = batch.METHODS[0]
    protocol = output / 'B' / 'protocols' / f'{method}.json'
    actual = read(protocol)
    if (actual['sequences'] != lengths or actual['detector_sha256'] != verified['detector_sha256']
            or actual['rgb_encoder_sha256'] != verified['rgb_encoder_sha256']):
        raise ValueError('Actual capture altered the frozen training input contract')
    capture_counts = {}
    for seq in lengths:
        metadata = read(output / 'B' / 'tco_capture' / method / (seq + '.json'))
        capture_counts[seq] = {'frames': lengths[seq], 'counts': metadata['counts'],
                               'capture_sha256': metadata['capture_sha256']}
    write_json(output / 'capture_manifest.json', dict(
        status='PASS', split_sha256=digest(args.split), protocol_sha256=digest(protocol),
        source_cache_sha256=digest(verified_path), frames=sum(lengths.values()), sequences=lengths,
        per_sequence_inputs=files, scoped_replayed_sequences=sorted(lengths),
        labels_in_prediction=False, scout_ground_truth_loaded=False,
        frozen_files_preserved=True, capture_counts=capture_counts))
    print(f'PASS {len(lengths)} training clips, {sum(lengths.values())} frames', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('repo', 'online', 'collector', 't0', 'train-baseline', 'images-root',
                   'split', 'old-split', 'proxy-map', 'source-audit', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    options = parser.parse_args()
    if options.jobs < 1:
        parser.error('jobs must be positive')
    options.device, options.cuda_devices = 'cpu', ''
    options.cache_wait_seconds = 60
    capture(options)


if __name__ == '__main__':
    main()
