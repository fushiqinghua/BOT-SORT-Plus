"""Stage and replay a writer-only low-confidence supplement for frozen BoT-SORT B."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


JOB = Path(__file__).resolve().parent
SUITE = JOB.parents[1]
REPO = SUITE / 'BOT-SORT-Plus'
PROJECT = SUITE.parents[1]
V01 = SUITE.with_name('botsort_plus_v0_1_0')
RUNTIME = PROJECT.parent / 'M3OT_MOT_Baseline_v0_1'
FROZEN = REPO / 'outputs/diagnostics_v0.4_full30_round1/observed'
CODE = JOB / 'overlay'
BASELINE = 'B_early_moderate'
VARIANT = 'B_postlow_output_v1'
SMOKE = ['DJI_0339_2', 'DJI_0327_3']
SPLIT_FIELDS = ('sequences', 'feature_hashes', 'decoded_image_hashes', 'detection_fingerprints')
SHARED_FIELDS = ('native_config', 'detector_sha256', 'rgb_encoder_sha256', 'source_protocol_sha256',
                 'encoder_contracts', 'boxmot_source_hashes')


def read(path):
    return json.loads(path.read_bytes())


def write_new(path, document):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f'Expected exact original source marker: {old!r}')
    return source.replace(old, new, 1)


def update_file(relative, changes):
    path = CODE / relative
    source = path.read_text(encoding='utf-8')
    for old, new in changes:
        source = replace_once(source, old, new)
    path.write_text(source, encoding='utf-8')


def stage():
    CODE.mkdir()
    for name in ('tools', 'botsort_plus'):
        shutil.copytree(REPO / name, CODE / name, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(JOB / 'weak_output.py', CODE / 'botsort_plus/weak_output.py')
    update_file('botsort_plus/adapter_v2.py', [
        ('def update_v2(tracker, dets, img, dense):',
         'def update_v2(tracker, dets, img, dense, weak_output=None):'),
        ('    tracker._update_track_states(removed)\n',
         '    if weak_output is not None:\n'
         '        from .weak_output import observe\n'
         '        observe(weak_output, tracker, frame, all_dets, pool, lost, activated, refound)\n'
         '    tracker._update_track_states(removed)\n'),
    ])
    update_file('botsort_plus/native_adapter.py', [
        ('    def update(self, dets, img, embs=None, features=None):',
         '    def update(self, dets, img, embs=None, features=None, weak_output=None):'),
        ('            return update_v2(self, dets, img, dense)\n',
         '            return update_v2(self, dets, img, dense, weak_output=weak_output)\n'),
    ])
    update_file('tools/run_botsort_plus.py', [
        ('    import cv2\n    import torch\n',
         '    import cv2\n    import torch\n'
         '    from botsort_plus.weak_output import WeakOutput\n'),
        ('            position = position_class',
         f'            weak_output = WeakOutput() if name == {VARIANT!r} else None\n'
         '            position = position_class'),
        ('                        t=time.perf_counter(); prediction=tracker.update(det,image,features=packets); seconds+=time.perf_counter()-t\n',
         '                        t=time.perf_counter()\n'
         '                        prediction=tracker.update(det,image,features=packets,weak_output=weak_output)\n'
         '                        seconds+=time.perf_counter()-t\n'),
        ('                        seen=set()\n',
         '                        weak_events = ()\n'
         '                        if weak_output is not None:\n'
         '                            prediction, weak_events = weak_output.emit(row["frame"], prediction)\n'
         '                        seen=set()\n'),
        ('                        for event in tracker.last_events:\n',
         '                        for event in (*tracker.last_events, *weak_events):\n'),
    ])
    cfg = read(REPO / 'configs/botsort_plus' / (BASELINE + '.json'))
    write_new(JOB / (VARIANT + '.json'), {**cfg, 'name': VARIANT})
    files = ('tools/run_botsort_plus.py', 'botsort_plus/adapter_v2.py',
             'botsort_plus/native_adapter.py', 'botsort_plus/weak_output.py')
    write_new(JOB / 'policy.json', {
        'name': VARIANT, 'frozen_tracker': BASELINE, 'causal': True,
        'observes': 'native newly lost active tracks and explicit B recoveries',
        'applied_after': 'native track processing and optional downstream position update',
        'outputs': 'append only raw detector xyxy and score, use original track ID',
        'state': 'No track IDs, embeddings, candidates, recovery decisions or tracking state are written; ' 
                 'observer metadata is held only by the MOT writer.',
        'score': '[0.05, 0.10)', 'min_prediction_iou': 0.50, 'row_and_track_margin': 0.10,
        'occupied_iou_reject': 0.50, 'age_after_last_native_B_recovery': [1, 3],
        'last_native_observation_lag': 1, 'max_weak_outputs_per_recovery': 2,
        'implementation_sha256': {name: digest(CODE / name) for name in files},
    })


def run(output, sequences):
    command = [sys.executable, '-B', '-X', 'utf8', str(CODE / 'tools/run_botsort_plus.py'),
               '--baseline-root', str(REPO / 'results/vttiny/yolo11m_rgb_rgbt/rgbt'),
               '--features-dir', str(V01 / 'features_rgb'),
               '--images-root', str(PROJECT / 'data/raw/VT-Tiny-MOT'),
               '--config', str(JOB / (VARIANT + '.json')), '--runtime-root', str(RUNTIME),
               '--output-dir', str(output), '--trace']
    if sequences:
        command.extend(['--sequences', *sequences])
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1',
           'PYTHONUSERBASE': str(V01 / 'python_user'), 'TRACKING_RUNTIME_ROOT': str(RUNTIME),
           'YOLO_CONFIG_DIR': str(JOB / 'yolo_settings')}
    with (JOB / (output.name + '.log')).open('x', encoding='utf-8') as log:
        subprocess.run(command, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    protocol = read(output / 'protocols' / (VARIANT + '.json'))
    frozen = read(FROZEN / 'protocols' / (BASELINE + '.json'))
    if protocol['status'] != 'COMPLETE' or protocol['subset'] != bool(sequences):
        raise RuntimeError('Replay protocol did not finish with the requested sequences')
    if protocol['causal'] is not True or protocol['backfilled_frames'] or protocol['interpolation']:
        raise RuntimeError('Expected purely causal frame output')
    for field in SHARED_FIELDS:
        if protocol[field] != frozen[field]:
            raise RuntimeError(f'Frozen replay input differs: {field}')
    for field in SPLIT_FIELDS:
        want = ({s: frozen[field][s] for s in sequences} if sequences else frozen[field])
        if protocol[field] != want:
            raise RuntimeError(f'Frozen replay input differs: {field}')
    if {**protocol['config'], 'name': BASELINE} != frozen['config']:
        raise RuntimeError('Effective configuration differs from the frozen run')
    write_new(JOB / (output.name + '_inputs.json'), {
        'status': 'PASS', 'sequences': protocol['sequences'], 'fields': SHARED_FIELDS + SPLIT_FIELDS,
        'effective_config_equal': True, 'protocol_sha256': digest(output / 'protocols' / (VARIANT + '.json')),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('smoke', 'full'))
    args = parser.parse_args()
    if args.mode == 'smoke':
        stage()
        run(JOB / 'smoke', SMOKE)
    else:
        if read(JOB / 'smoke_inputs.json')['status'] != 'PASS':
            raise RuntimeError('Check the smoke replay before launching all 30 sequences')
        run(JOB / 'variant_full', None)
    print(f'{args.mode}: COMPLETE', flush=True)


if __name__ == '__main__':
    main()
