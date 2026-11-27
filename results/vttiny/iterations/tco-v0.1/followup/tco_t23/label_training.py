"""Label original native low-stage training candidates without exposing GT to replay."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import gzip
import hashlib
import json
from pathlib import Path


METHOD = 'B_early_moderate'
PARTITIONS = ('fit', 'calibration', 'holdout')


def read(path):
    return json.loads(path.read_bytes())


def run(args):
    import label_t1
    from botsort_plus.io import detection_files, sequence_name
    from experiments.vttiny_gmc.round3_joint import analysis

    split = read(args.split)
    if split['schema'] != 'tco_v0.1_training_split' or split['status'] != 'LOCKED':
        raise ValueError('A locked training-only mother-video partition is required')
    groups = split['training_sequence_to_source_video']
    if set(groups.values()) & set(split['development_sequence_to_source_video'].values()):
        raise ValueError('Training and development source videos overlap')
    lengths = {key: length for part in PARTITIONS for key, length in split['split'][part]['sequences'].items()}
    if set(lengths) != set(groups):
        raise ValueError('Partition has inconsistent sequence keys')
    protocol = read(args.runs / 'B/protocols' / (METHOD + '.json'))
    if protocol['sequences'] != lengths:
        raise ValueError('Replay must cover exactly the locked training sequences')
    if protocol['detector_sha256'] != args.detector_sha256 or protocol['rgb_encoder_sha256'] != args.encoder_sha256:
        raise ValueError('Replay does not use the frozen detector and RGB ReID')
    trackeval, evaluator = analysis.load_trackeval(args.trackeval)
    residual = read(args.design)['model']['box_residual']
    caches = {sequence_name(p): p for p in detection_files(args.detections)}
    args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(method=METHOD, trackeval_commit=evaluator['commit'], partitions={},
                   split_sha256=analysis.digest(args.split), detector_sha256=args.detector_sha256,
                   rgb_encoder_sha256=args.encoder_sha256)
    for part in PARTITIONS:
        dest = args.output / part
        dest.mkdir()
        counts, unique = defaultdict(Counter), defaultdict(set)
        current = split['split'][part]['sequences']
        if not current:
            raise ValueError(f'Empty {part} split')
        context = args.runs, args.gt, lengths, dest, trackeval, residual, groups, unique
        names = ('pairs.runtime.jsonl.gz', 'pairs.labels.jsonl.gz', 'episodes.jsonl.gz')
        with ExitStack() as stack:
            streams = [stack.enter_context(gzip.open(dest / n, 'xt', encoding='utf-8', compresslevel=2)) for n in names]
            for seq in sorted(current):
                label_t1.process_sequence(context, 'B', METHOD, seq, protocol, caches[seq], streams, counts)
                print(part, seq, counts[METHOD, 'sequence', seq]['assigned_pairs'], flush=True)
        summary['partitions'][part] = dict(sequences=current, counts=dict(counts[METHOD, 'all', 'ALL']),
                                           files={n: analysis.digest(dest / n) for n in names})
    analysis.dump(args.output / 'training_labels.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('split', 'runs', 'gt', 'trackeval', 'design', 'detections', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--detector-sha256', required=True)
    parser.add_argument('--encoder-sha256', required=True)
    run(parser.parse_args())
