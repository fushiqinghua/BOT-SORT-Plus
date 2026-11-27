"""Run the unmodified T3a stages with the exact authorized Q2 partition."""

import argparse
from pathlib import Path
import sys

from freeze_q2 import digest, freeze


def main():
    wrapper = argparse.ArgumentParser(add_help=False, allow_abbrev=False, description=__doc__)
    for option in ('online-src', 'old-split', 'proxy-map', 'verified-cache', 'source-audit'):
        wrapper.add_argument('--' + option, type=Path, required=True)
    args, forwarded = wrapper.parse_known_args()
    expected = freeze(args.old_split, args.proxy_map, args.verified_cache, args.source_audit)

    sys.path.insert(0, str(args.online_src.resolve()))
    import batch

    def q2_lengths(split, part):
        if split != expected:
            raise ValueError('Replay requires the exact authorized Q2 split')
        if part == 'all':
            return {seq: frames for split_part in split['split'].values()
                    for seq, frames in split_part['sequences'].items()}
        return split['split'][part]['sequences']

    old_sources = batch.runtime_sources
    old_train_prediction = batch.train_prediction

    def q2_sources(options):
        return {**old_sources(options), 'q2_split_wrapper.py': digest(Path(__file__)),
                'q2_split_freezer.py': digest(Path(__file__).with_name('freeze_q2.py'))}

    def q2_train_prediction(options, lengths, *, require_ready=True):
        actual_cache = options.train_baseline/'verified.json'
        if digest(actual_cache) != expected['evidence']['frozen_training_cache_sha256']:
            raise ValueError('The replay cache differs from the one bound to the Q2 split')
        result = old_train_prediction(options, lengths, require_ready=require_ready)
        return result

    batch.selected_lengths = q2_lengths
    batch.runtime_sources = q2_sources
    batch.train_prediction = q2_train_prediction
    # Exact replay, score/quality adoption and stage gates stay in the frozen
    # entry point. Only the independent evidence-backed split check differs.
    sys.argv = [str(Path(batch.__file__)), *forwarded]
    batch.main()


if __name__ == '__main__':
    main()
