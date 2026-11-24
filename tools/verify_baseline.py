"""Recompute the frozen trajectories' HOTA, CLEAR and Identity metrics."""
import argparse
import math
from pathlib import Path

from evaluate_vttiny import evaluate
from vttiny_tracking_common import read


DEFAULT_RESULTS = Path(__file__).resolve().parents[1] / 'results/vttiny/yolo11m_rgb_rgbt'
FIELDS = ('HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'FP', 'FN', 'IDSW')


def verify(results, gt, output, mode):
    trial = results / mode
    protocol = read(trial / 'protocol.json')
    sequence_lengths = {name: len(frames) for name, frames in protocol['sequences'].items()}
    assert len(sequence_lengths) == 30 and sum(sequence_lengths.values()) == 12590
    recorded = read(trial / 'metrics/summary.json')['results']
    expected = {(row['tracker'], row['sequence']): row for row in recorded}
    trackers = sorted({tracker for tracker, _ in expected})
    computed = evaluate('rgb', 'test', trial / 'trackers', trackers, output / mode,
                        seqs=sequence_lengths, gt_folder=gt)['results']
    keys = {(row['tracker'], row['sequence']) for row in computed}
    if keys != set(expected) or len(computed) != len(expected):
        raise ValueError(f'{mode}: recomputation did not cover the full experiment')
    for row in computed:
        key = (row['tracker'], row['sequence'])
        for field in FIELDS:
            recorded_value = expected[key][field]
            if not math.isclose(row[field], recorded_value, rel_tol=0, abs_tol=1e-8):
                raise ValueError(f'{mode}, {key}, {field}: {row[field]} != {recorded_value}')
    print(f'{mode}: {len(trackers)} configurations, {len(computed)} result rows verified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt-folder', type=Path, required=True, help='Official GT/GT_00/test')
    parser.add_argument('--output', type=Path, required=True, help='Write recalculated metrics here')
    parser.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--mode', choices=('rgb', 'rgbt', 'both'), default='both')
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.results.resolve()):
        parser.error('--output must be separate from the stored experiment')
    for selected in ('rgb', 'rgbt') if args.mode == 'both' else (args.mode,):
        verify(args.results, args.gt_folder, args.output, selected)
