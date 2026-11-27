"""Fit and freeze the T2 geometry and two RGB arms on official-training crops."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
import csv
import gzip
import hashlib
from itertools import product
import json
from pathlib import Path
from shutil import copyfile
import subprocess
import sys
import time

import numpy as np
import torch

from model import ARMS, THRESHOLD_KEYS, TCO, loss_terms, paired_iou


PARTITIONS = ('fit', 'calibration', 'holdout')
COMMON_ARRAYS = ('geometry', 'raw_xyxy', 'roi_affine', 'valid_xyxy', 'hist_valid', 'hist_dt')
ARM_ARRAYS = dict(P1=(), P2=('current', 'current_repeat'), P3=('current', 'history'))
LABEL_ARRAYS = ('support_label', 'gt_xyxy', 'assigned', 'source_id', 'track_key')
OUTPUT_ARRAYS = ('support', 'quality_raw', 'quality_refined', 'refined_xyxy', 'box_valid')


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


class Corpus:
    def __init__(self, root, count, capacity=None):
        self.count = count
        capacity = count if capacity is None else capacity
        optional = tuple(name for name in ('sequence_id', 'frame') if (root / (name + '.npy')).exists())
        names = (*COMMON_ARRAYS, *ARM_ARRAYS['P2'], 'history', *LABEL_ARRAYS, *optional)
        arrays = {name: np.load(root / (name + '.npy'), mmap_mode='r', allow_pickle=False) for name in names}
        if not 0 < count <= capacity or any(len(value) != capacity for value in arrays.values()):
            raise ValueError(f'Crop manifest and on-disk row counts differ: {root}')
        self.arrays = {name: value[:count] for name, value in arrays.items()}
        if not np.isin(self.arrays['support_label'], (-1, 0, 1)).all():
            raise ValueError(f'Unknown support label encoding: {root}')
        positive = self.arrays['support_label'] == 1
        gt = self.arrays['gt_xyxy'][positive]
        if not (np.isfinite(gt).all() and (gt[:, 2:] > gt[:, :2]).all()):
            raise ValueError(f'Positive samples must have valid uncut GT: {root}')
        self.raw_iou = np.zeros(count, dtype=np.float32)
        raw = torch.from_numpy(self.arrays['raw_xyxy'][positive])
        self.raw_iou[positive] = paired_iou(raw, torch.from_numpy(gt)).numpy()

    def batch(self, arm, rows, device, *, history_rows=None):
        result = {}
        for name in (*COMMON_ARRAYS, *ARM_ARRAYS[arm], 'support_label', 'gt_xyxy'):
            indices = history_rows if name == 'history' and history_rows is not None else rows
            result[name] = torch.from_numpy(self.arrays[name][indices]).to(device)
        return result


def fit_order(corpus, seed, epoch):
    random = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    queues = defaultdict(list)
    rows = np.flatnonzero(corpus.arrays['support_label'] >= 0)
    for index in random.permutation(rows):
        queues[int(corpus.arrays['track_key'][index])].append(index)
    order = np.empty(len(rows), dtype=np.int64)
    groups = list(queues.values())
    cursor = 0
    while groups:
        random.shuffle(groups)
        for group in groups:
            order[cursor] = group.pop()
            cursor += 1
        groups = [group for group in groups if group]
    return order


def batches(rows, batch_size):
    for start in range(0, len(rows), batch_size):
        yield rows[start:start + batch_size]


@torch.inference_mode()
def infer(net, corpus, policy, device, *, donors=None):
    net.eval()
    shape = {'refined_xyxy': (corpus.count, 4)}
    result = {key: np.empty(shape.get(key, (corpus.count,)), dtype=bool if key == 'box_valid' else np.float32)
              for key in OUTPUT_ARRAYS}
    for rows in batches(np.arange(corpus.count), policy['batch_size']):
        batch = corpus.batch(net.arm, rows, device, history_rows=donors[rows] if donors is not None else None)
        values = net(batch)
        for key in result:
            result[key][rows] = values[key].cpu().numpy()
    positive = corpus.arrays['support_label'] == 1
    result['refined_iou'] = np.zeros(corpus.count, dtype=np.float32)
    result['refined_iou'][positive] = paired_iou(torch.from_numpy(result['refined_xyxy'][positive]),
                                                torch.from_numpy(corpus.arrays['gt_xyxy'][positive])).numpy()
    if not all(np.isfinite(value).all() for value in result.values()):
        raise RuntimeError('Nonfinite TCO outputs')
    return result


def adoption(pred, thresholds):
    if thresholds is None:
        return np.zeros(len(pred['box_valid']), dtype=bool)
    return (pred['box_valid'] & (pred['support'] >= thresholds['support'])
            & (pred['quality_refined'] >= thresholds['refined_quality'])
            & (pred['quality_refined'] - pred['quality_raw'] >= thresholds['quality_gain']))


def consequences(corpus, pred, use, scope, policy):
    support = corpus.arrays['support_label']
    threshold = policy['correct_iou_threshold']
    positive = scope & (support == 1)
    negative = scope & (support == 0)
    unknown = scope & (support < 0)
    originally_correct = positive & (corpus.raw_iou >= threshold)
    correctly_refined = pred['refined_iou'] >= threshold
    upward = use & positive & ~originally_correct & correctly_refined
    downward = use & originally_correct & ~correctly_refined
    actual_iou = np.where(use, pred['refined_iou'], corpus.raw_iou)
    count = lambda values: int(values.sum())
    correct_total = count(originally_correct)
    negative_total = count(negative)
    positive_total = count(positive)
    down = count(downward)
    false_use = count(use & negative)
    return dict(events=count(scope), positive=positive_total, negative=negative_total, unknown=count(unknown),
                originally_correct=correct_total, originally_incorrect_positive=positive_total-correct_total,
                adopted=count(scope & use), known_negative_adopted=false_use,
                unknown_adopted=count(use & unknown), upward=count(upward), downward=down,
                net=count(upward) - down, correct_damage_fraction=down / correct_total if correct_total else None,
                known_negative_adoption_fraction=false_use / negative_total if negative_total else None,
                mean_raw_iou=float(corpus.raw_iou[positive].mean()) if positive_total else None,
                mean_applied_iou=float(actual_iou[positive].mean()) if positive_total else None)


def within_budget(metrics, policy):
    return (metrics['correct_damage_fraction'] is not None
            and metrics['correct_damage_fraction'] <= policy['maximum_correct_damage_fraction']
            and metrics['known_negative_adoption_fraction'] is not None
            and metrics['known_negative_adoption_fraction'] <= policy['maximum_known_negative_adoption_fraction'])


def objective(metrics):
    mean_iou = metrics['mean_applied_iou'] if metrics['mean_applied_iou'] is not None else -1
    return (metrics['net'], -metrics['downward'], -metrics['known_negative_adopted'], mean_iou, -metrics['adopted'])


def calibrate(corpus, pred, policy):
    assigned = corpus.arrays['assigned'].astype(bool, copy=False)
    choice = None
    best = consequences(corpus, pred, adoption(pred, None), assigned, policy)
    search = []
    for combination in product(*(policy['threshold_grid'][key] for key in THRESHOLD_KEYS)):
        thresholds = dict(zip(THRESHOLD_KEYS, combination))
        metrics = consequences(corpus, pred, adoption(pred, thresholds), assigned, policy)
        feasible = within_budget(metrics, policy)
        search.append(dict(thresholds=thresholds, within_budget=feasible, **metrics))
        if feasible and metrics['net'] > 0 and objective(metrics) > objective(best):
            choice, best = thresholds, metrics
    return choice, best, search


def reliability(corpus, scores, scope):
    labels = corpus.arrays['support_label']
    known = (labels >= 0) & scope
    labels, values = labels[known], scores[known]
    bins = []
    for lower in range(10):
        selected = np.minimum(9, (values * 10).astype(np.int32)) == lower
        n = int(selected.sum())
        bins.append(dict(low=lower / 10, high=(lower + 1) / 10, count=n,
                         mean_score=float(values[selected].mean()) if n else None,
                         fraction_positive=float(labels[selected].mean()) if n else None))
    return dict(known=int(known.sum()), brier=float(((values - labels) ** 2).mean()) if len(labels) else None,
                bins=bins, expected_calibration_error=(sum(row['count'] * abs(row['mean_score'] - row['fraction_positive'])
                                                           for row in bins if row['count']) / len(labels))
                if len(labels) else None)


def measures(corpus, pred, thresholds, policy):
    use = adoption(pred, thresholds)
    assigned = corpus.arrays['assigned'].astype(bool, copy=False)
    groups = corpus.arrays['source_id']
    result = dict(assigned=consequences(corpus, pred, use, assigned, policy),
                  all_eligible_edges=consequences(corpus, pred, use, np.ones(corpus.count, dtype=bool), policy),
                  assigned_groups={str(group): consequences(corpus, pred, use, assigned & (groups == group), policy)
                                   for group in np.unique(groups)},
                  assigned_support_reliability=reliability(corpus, pred['support'], assigned))
    positive = assigned & (corpus.arrays['support_label'] == 1)
    gt = corpus.arrays['gt_xyxy'][positive]
    if len(gt):
        raw = corpus.arrays['raw_xyxy'][positive]
        chosen = np.where(use[positive, None], pred['refined_xyxy'][positive], raw)
        size = (gt[:, 2:] - gt[:, :2]).min(axis=1)
        center_error = lambda boxes: np.linalg.norm((boxes[:, :2] + boxes[:, 2:] - gt[:, :2] - gt[:, 2:]) / 2, axis=1) / size
        result['assigned'].update(mean_normalized_center_error_raw=float(center_error(raw).mean()),
                                  mean_normalized_center_error_applied=float(center_error(chosen).mean()))
    return result


def prediction_csv(path, corpus, pred, thresholds):
    use = adoption(pred, thresholds)
    fields = ['row', 'source_id', 'track_key', 'assigned', 'support_label', 'support_score', 'quality_raw',
              'quality_refined', 'box_valid', 'adopted', 'fallback_reason', 'raw_iou', 'refined_iou', 'raw_xyxy',
              'refined_xyxy', 'applied_xyxy']
    optional = [name for name in ('sequence_id', 'frame') if name in corpus.arrays]
    with gzip.open(path, 'wt', encoding='utf-8', newline='', compresslevel=2) as stream:
        writer = csv.writer(stream)
        writer.writerow(fields + optional)
        for row in range(corpus.count):
            raw = corpus.arrays['raw_xyxy'][row]
            refined = pred['refined_xyxy'][row]
            support_label = int(corpus.arrays['support_label'][row])
            reason = ('applied' if use[row] else 'calibration_disabled' if thresholds is None else
                      'outside_valid_roi' if not pred['box_valid'][row] else 'evidence_threshold')
            writer.writerow([row, int(corpus.arrays['source_id'][row]), int(corpus.arrays['track_key'][row]),
                             bool(corpus.arrays['assigned'][row]), support_label, float(pred['support'][row]),
                             float(pred['quality_raw'][row]), float(pred['quality_refined'][row]),
                             bool(pred['box_valid'][row]), bool(use[row]), reason,
                             float(corpus.raw_iou[row]) if support_label >= 0 else '',
                             float(pred['refined_iou'][row]) if support_label >= 0 else '',
                             json.dumps(raw.tolist()), json.dumps(refined.tolist()),
                             json.dumps((refined if use[row] else raw).tolist()),
                             *(int(corpus.arrays[name][row]) for name in optional)])


def shuffled_donors(corpus):
    """Choose another pre-frame history from the same sequence and update time."""
    if not {'sequence_id', 'frame'} <= corpus.arrays.keys():
        return None
    tracks = corpus.arrays['track_key']
    by_frame = defaultdict(dict)
    signatures = zip(corpus.arrays['sequence_id'], corpus.arrays['frame'],
                     corpus.arrays['hist_valid'] @ np.array([1, 2, 4]))
    grouping = list(signatures)
    for row, key in enumerate(grouping):
        by_frame[key].setdefault(tracks[row], row)
    donors = np.arange(corpus.count)
    for row, key in enumerate(grouping):
        alternatives = by_frame[key]
        if len(alternatives) > 1:
            donors[row] = next(donor for track, donor in alternatives.items() if track != tracks[row])
    return donors


def shuffle_diagnostic(net, corpus, regular, thresholds, policy, device, output, key):
    donors = shuffled_donors(corpus)
    if donors is None:
        return {'status': 'SOURCE_FRAME_METADATA_UNAVAILABLE'}
    changed = donors != np.arange(corpus.count)
    assigned = changed & corpus.arrays['assigned'].astype(bool, copy=False)
    if not assigned.any():
        return {'status': 'NO_ASSIGNED_SAME_FRAME_ALTERNATIVE_HISTORY'}
    shuffled = infer(net, corpus, policy, device, donors=donors)
    np.savetxt(output / f'{key}_history_donors.csv', np.column_stack((np.flatnonzero(changed), donors[changed])),
               fmt='%d', delimiter=',', header='recipient_row,donor_row', comments='')
    return dict(status='COMPARED_ON_SAME_ASSIGNMENTS_WITH_ALTERNATIVE',
                scope='Assigned holdout rows with a different-track donor at the same current sequence/frame and slot validity',
                donor_age_metadata='Original recipient ages retained; history pixels and masks use the causal donor.',
                regular=consequences(corpus, regular, adoption(regular, thresholds), assigned, policy),
                shuffled=consequences(corpus, shuffled, adoption(shuffled, thresholds), assigned, policy))


def train_arm(arm, seed, fit, calibration, holdout, policy, output, device):
    torch.manual_seed(seed)
    net = TCO(arm).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=policy['learning_rate'], weight_decay=policy['weight_decay'])
    key = f'{arm}_seed{seed}'
    checkpoint_root = output / 'checkpoints' / arm
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_root / f'seed{seed}.pt'
    best, steps, patience, history = None, 0, 0, []
    for epoch in range(1, policy['max_epochs'] + 1):
        net.train()
        samples = 0
        total_loss = 0.0
        for rows in batches(fit_order(fit, seed, epoch), policy['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            batch = fit.batch(arm, rows, device)
            terms = loss_terms(net(batch), batch, batch['support_label'], batch['gt_xyxy'])
            if not torch.isfinite(terms['loss']):
                raise RuntimeError(f'Nonfinite training loss in {key}, step {steps}')
            terms['loss'].backward()
            optimizer.step()
            steps += 1
            samples += len(rows)
            total_loss += terms['loss'].detach().item() * len(rows)
            if steps >= policy['max_steps']:
                break
        cal_pred = infer(net, calibration, policy, device)
        thresholds, cal_metrics, threshold_search = calibrate(calibration, cal_pred, policy)
        improved = best is None or objective(cal_metrics) > objective(best['assigned'])
        if improved:
            patience = 0
            torch.save(dict(arm=arm, seed=seed, epoch=epoch, model_state_dict=net.state_dict()), checkpoint)
            best = dict(arm=arm, seed=seed, epoch=epoch, thresholds=thresholds, assigned=cal_metrics,
                        checkpoint=checkpoint.relative_to(output).as_posix(), checkpoint_sha256=sha256(checkpoint))
            write_json(output / f'{key}_calibration_frozen.json', best)
            write_json(output / f'{key}_calibration_grid.json', threshold_search)
        else:
            patience += 1
        report = dict(arm=arm, seed=seed, epoch=epoch, steps=steps, fit_samples=samples,
                      fit_loss=total_loss / samples, calibration_net=cal_metrics['net'],
                      calibration_adopted=cal_metrics['adopted'], selected=improved)
        history.append(report)
        print(json.dumps(report), flush=True)
        if (steps >= policy['max_steps'] or (epoch >= policy['early_stopping_min_epochs']
                                             and patience >= policy['early_stopping_patience_epochs'])):
            break
    net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)['model_state_dict'])
    cal_pred = infer(net, calibration, policy, device)
    hold_pred = infer(net, holdout, policy, device)
    thresholds = best['thresholds']
    prediction_csv(output / f'{key}_calibration_predictions.csv.gz', calibration, cal_pred, thresholds)
    prediction_csv(output / f'{key}_holdout_predictions.csv.gz', holdout, hold_pred, thresholds)
    result = dict(frozen=best, history=history, calibration=measures(calibration, cal_pred, thresholds, policy),
                  holdout=measures(holdout, hold_pred, thresholds, policy))
    if arm == 'P3' and result['holdout']['assigned']['net'] > 0:
        result['shuffled_history'] = shuffle_diagnostic(net, holdout, hold_pred, thresholds, policy, device, output, key)
    write_json(output / f'{key}_offline_metrics.json', result)
    return result


def raw_reference(corpus, policy):
    pred = dict(support=np.zeros(corpus.count), refined_iou=corpus.raw_iou,
                refined_xyxy=corpus.arrays['raw_xyxy'], box_valid=np.zeros(corpus.count, dtype=bool))
    result = measures(corpus, pred, None, policy)
    del result['assigned_support_reliability']
    return result


def gate(arms, policy):
    scores = {arm: result['holdout']['assigned'] for arm, result in arms.items()}
    lead = scores['P3']
    groups = arms['P3']['holdout']['assigned_groups']
    nets = [metrics['net'] for metrics in groups.values()]
    tests = dict(positive_net=lead['net'] > 0, beats_geometry=lead['net'] > scores['P1']['net'],
                 beats_current_rgb=lead['net'] > scores['P2']['net'], preserves_budget=within_budget(lead, policy),
                 improves_mean_iou=(lead['mean_applied_iou'] is not None and lead['mean_applied_iou'] > lead['mean_raw_iou']),
                 improves_multiple_sources=sum(value > 0 for value in nets) >= 2,
                 survives_removing_strongest_source=lead['net'] > max(nets, default=0))
    return dict(passes=all(tests.values()), conditions=tests,
                group_net={key: value['net'] for key, value in groups.items()})


def parallel_arms(args, seed):
    """Start one fresh training process per arm, each with its own log and device."""
    workers = {}
    logs = args.output / 'arm_logs'
    logs.mkdir(exist_ok=True)
    with ExitStack() as open_logs:
        try:
            for arm, device in zip(ARMS, args.devices, strict=True):
                log = open_logs.enter_context((logs / f'{arm}_seed{seed}.log').open('xb'))
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--corpus', str(args.corpus),
                           '--policy', str(args.policy), '--output', str(args.output), '--threads', str(args.threads),
                           '--device', device, '--worker-arm', arm, '--worker-seed', str(seed)]
                workers[arm] = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                                stderr=subprocess.STDOUT)
            codes = {arm: worker.wait() for arm, worker in workers.items()}
        except BaseException:
            for worker in workers.values():
                if worker.poll() is None:
                    worker.terminate()
            for worker in workers.values():
                worker.wait()
            raise
    failures = {arm: code for arm, code in codes.items() if code}
    if failures:
        raise RuntimeError(f'T2 seed {seed} arm subprocess failed: {failures}. See {logs}')
    return {arm: json.loads((args.output / f'{arm}_seed{seed}_offline_metrics.json').read_bytes()) for arm in ARMS}


def run(args):
    started = time.monotonic()
    manifest_path = args.corpus / 'crop_manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    policy = json.loads(args.policy.read_bytes())
    model_sha = sha256(Path(__file__).with_name('model.py'))
    if (manifest['schema'] != 'tco_v0.1_crop_corpus' or manifest['status'] != 'LOCKED'
            or manifest['dataset_from_official_training_only'] is not True
            or manifest['model_source_sha256'] != model_sha):
        raise ValueError('T2 needs a locked official-training-only corpus built with these crop/model sources')
    if policy['schema'] != 'tco_v0.1_t2_training_policy':
        raise ValueError('Unknown trainer policy schema')
    corpora = {part: Corpus(args.corpus / part, specification['events'], specification.get('capacity'))
               for part, specification in manifest['partitions'].items()}
    if set(corpora) != set(PARTITIONS):
        raise ValueError('Unexpected corpus partition names')
    source_sets = [set(corpora[part].arrays['source_id'].tolist()) for part in PARTITIONS]
    if any(a & b for i, a in enumerate(source_sets) for b in source_sets[i+1:]):
        raise ValueError('Training/cali/holdout source IDs overlap')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    fit, calibration, holdout = (corpora[part] for part in PARTITIONS)
    pp = policy['pilot_seed']
    seeds = [pp, *policy['confirmation_seeds']]
    if args.worker_arm:
        audit = json.loads((args.output / 'model_manifest.json').read_bytes())
        if (audit['crop_manifest_sha256'] != sha256(manifest_path)
                or audit['policy_sha256'] != sha256(args.policy) or args.worker_seed not in seeds):
            raise ValueError('Training worker inputs differ from the frozen master run')
        train_arm(args.worker_arm, args.worker_seed, fit, calibration, holdout, policy, args.output, device)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'checkpoints').mkdir()
    copyfile(args.policy, args.output / 'train_policy.json')
    audit = dict(crop_manifest_sha256=sha256(manifest_path), policy_sha256=sha256(args.policy),
                 trainer_sha256=sha256(Path(__file__)), model_source_sha256=model_sha,
                 torch_version=torch.__version__, devices=dict(zip(ARMS, args.devices or [str(device)] * len(ARMS))),
                 execution_mode='parallel_isolated_arms' if args.devices else 'sequential',
                 partitions=manifest['partitions'],
                 protocol_status=manifest['protocol_status'], official_test_used_for_t2=False)
    write_json(args.output / 'model_manifest.json', audit)
    audit_rows = {}
    for part, corpus in corpora.items():
        labels = corpus.arrays['support_label']
        assigned = corpus.arrays['assigned'].astype(bool, copy=False)
        under = (labels == 1) & (corpus.raw_iou < policy['correct_iou_threshold'])
        audit_rows[part] = dict(events=corpus.count, known_positive=int((labels == 1).sum()),
                                known_negative=int((labels == 0).sum()), unknown=int((labels < 0).sum()),
                                below_050_positive_all=int(under.sum()), below_050_positive_assigned=int((under & assigned).sum()),
                                negative_assigned=int((assigned & (labels == 0)).sum()),
                                source_groups=len(source_sets[PARTITIONS.index(part)]))
    missing_hard_positive = [part for part, row in audit_rows.items() if not row['below_050_positive_assigned']]
    write_json(args.output / 'label_coverage_preflight.json', audit_rows)
    if missing_hard_positive:
        status = 'STOP_T2_INSUFFICIENT_INDEPENDENT_POSITIVE_LABELS'
        write_json(args.output / 'offline_metrics.json', dict(status=status, missing=missing_hard_positive,
                                                             label_coverage=audit_rows, seeds_run=[]))
        (args.output / 'T2_DECISION.md').write_text(
            f'# TCO T2\n\n状态 `{status}`。\n\n没有原框 IoU 低于 0.50 且标签独立确认的原生低分已配对正例的分组为 '
            f'{", ".join(missing_hard_positive)}。未启动训练与门限选择。\n', encoding='utf-8')
        print(status, flush=True)
        return
    results, gates = {}, {}
    for seed in seeds:
        if seed != pp and not gates[str(pp)]['passes']:
            break
        arms = parallel_arms(args, seed) if args.devices else {
            arm: train_arm(arm, seed, fit, calibration, holdout, policy, args.output, device) for arm in ARMS}
        results[str(seed)] = arms
        gates[str(seed)] = gate(arms, policy)
        write_json(args.output / 'partial_progress.json', dict(gates=gates))
    selected = results[str(pp)]['P3']['frozen']
    net_total = {arm: sum(arms[arm]['holdout']['assigned']['net'] for arms in results.values()) for arm in ARMS}
    passed = (gates[str(pp)]['passes'] and sum(value['passes'] for value in gates.values()) >= 2
              and net_total['P3'] > max(net_total['P1'], net_total['P2'], 0))
    status = 'PASS_T2_HOLDOUT_GATE' if passed else 'STOP_T2_INSUFFICIENT_HOLDOUT_INCREMENT'
    summary = dict(status=status, seeds_run=list(results), seed_gates=gates, summed_seed_net=net_total,
                   label_coverage=audit_rows, protocol_status=manifest['protocol_status'],
                   P0_raw={part: raw_reference(corpora[part], policy) for part in ('calibration', 'holdout')},
                   arms=results, elapsed_seconds=round(time.monotonic() - started, 2),
                   t3a_development_authorized=False,
                   note='Passing T2 only enables a separate T3a holdout closed-loop check before full 30 replay.')
    write_json(args.output / 'offline_metrics.json', summary)
    write_json(args.output / 'calibration.json',
               {seed: {arm: item['frozen'] for arm, item in arms.items()} for seed, arms in results.items()})
    if passed:
        write_json(args.output / 'selection.json', dict(**selected, crop_manifest_sha256=audit['crop_manifest_sha256'],
                                                        policy_sha256=audit['policy_sha256'], status=status,
                                                        protocol_status=manifest['protocol_status'],
                                                        calibrated_split='calibration', gate_split='holdout'))
    lines = ['# TCO T2', '', f'状态 `{status}`。', '',
             '| Seed | Arm | Holdout upward | Holdout downward | Net | Adopted | Damage | Negative adoption |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seed, arms in results.items():
        for arm, item in arms.items():
            m = item['holdout']['assigned']
            lines.append(f'| {seed} | {arm} | {m["upward"]} | {m["downward"]} | {m["net"]} | {m["adopted"]} | '
                         f'{m["correct_damage_fraction"]} | {m["known_negative_adoption_fraction"]} |')
    lines.extend(['', '检查点与采用门槛只由 calibration 确定。holdout 使用固定配置。',
                  '未知标签不参与正确率及预算计算。按来源组的结果及停止原因见 offline_metrics.json。', ''])
    (args.output / 'T2_DECISION.md').write_text('\n'.join(lines), encoding='utf-8')
    print(status, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--policy', type=Path, default=Path(__file__).with_name('train_policy.json'))
    parser.add_argument('--output', type=Path, required=True)
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument('--device', default='cuda:0', help='Sequential device, default cuda:0')
    devices.add_argument('--devices', nargs=3, metavar=('P1', 'P2', 'P3'),
                         help='Run isolated arm subprocesses concurrently on these devices')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--worker-arm', choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument('--worker-seed', type=int, help=argparse.SUPPRESS)
    run(parser.parse_args())
