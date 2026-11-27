"""Refit the P3 proposal model and compare two isolated quality objectives."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import torch


MODES = ('absolute_only', 'paired_gain')
PASS_STATUS = 'PASS_T2_HOLDOUT_GATE'
STOP_STATUS = 'STOP_Q2_ACCEPTANCE_FAILED'


def _read(path):
    return json.loads(path.read_bytes())


def _corpus(train, root, manifest, name):
    spec = manifest['partitions'][name]
    return train.Corpus(root / name, spec['events'], spec.get('capacity'))


def _source_ids(corpus):
    return sorted({int(value) for value in corpus.arrays['source_id']})


def _optimizer(net, policy):
    return torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),
                             lr=policy['learning_rate'], weight_decay=policy['weight_decay'])


def _train_epoch(train, net, optimizer, corpus, order, policy, device, loss):
    total = 0.0
    for rows in train.batches(order, policy['batch_size']):
        optimizer.zero_grad(set_to_none=True)
        batch = corpus.batch(net.arm, rows, device)
        terms = loss(net(batch), batch, batch['support_label'], batch['gt_xyxy'])
        if not torch.isfinite(terms['loss']):
            raise RuntimeError('Nonfinite TCO training loss')
        terms['loss'].backward()
        optimizer.step()
        total += terms['loss'].item() * len(rows)
    return total / len(order)


def _new_base(model, train, fit, policy, output, device):
    """Train from fresh weights using fit only and the registered epoch budget."""
    torch.manual_seed(policy['seed'])
    net = model.TCO('P3').to(device)
    optimizer = _optimizer(net, policy)
    path = output / 'checkpoints' / 'base.pt'
    history = []
    for epoch in range(1, policy['base_epochs'] + 1):
        net.train()
        order = train.fit_order(fit, policy['seed'], epoch)
        loss = _train_epoch(train, net, optimizer, fit, order, policy, device, model.loss_terms)
        row = dict(stage='base', epoch=epoch, fit_loss=loss, fit_samples=len(order))
        history.append(row)
        print(json.dumps(row), flush=True)
    torch.save(dict(arm='P3', model_state_dict=net.state_dict()), path)
    train.write_json(output / 'base_history.json', history)
    return path


def _load(model, checkpoint, device):
    net = model.TCO('P3').to(device)
    weights = torch.load(checkpoint, map_location=device, weights_only=True)
    net.load_state_dict(weights['model_state_dict'], strict=True)
    return net, weights['model_state_dict']


def _assert_fixed_proposer(net, base_state):
    if any(not torch.equal(value, base_state[name]) for name, value in net.state_dict().items()
           if not name.startswith('quality_head.')):
        raise ValueError('An isolated quality head changed the encoder, support, or proposal parameters')


def _assert_same_predictions(first, other):
    for name in ('refined_xyxy', 'support', 'box_valid', 'refined_iou'):
        if not np.array_equal(first[name], other[name]):
            raise ValueError(f'Quality-head refit changed a frozen prediction: {name}')


def _refit_head(model, train, quality, fit, cal, base, base_pred, policy, output, device, mode):
    net, base_state = _load(model, base, device)
    net.eval()
    net.requires_grad_(False)
    net.quality_head.requires_grad_(True)
    optimizer = _optimizer(net, policy)
    checkpoint = output / 'checkpoints' / (mode + '.pt')
    best, history = None, []

    def loss(values, batch, labels, gt):
        return quality.loss_terms(values, batch, labels, gt, policy['objective'], mode=mode)

    for epoch in range(1, policy['quality_epochs'] + 1):
        order = train.fit_order(fit, policy['seed'], policy['base_epochs'] + epoch)
        order = order[fit.arrays['assigned'][order]]
        fit_loss = _train_epoch(train, net, optimizer, fit, order, policy, device, loss)
        _assert_fixed_proposer(net, base_state)
        pred = train.infer(net, cal, policy, device)
        _assert_same_predictions(base_pred, pred)
        thresholds, metrics, grid = train.calibrate(cal, pred, policy)
        updated = best is None or train.objective(metrics) > train.objective(best['calibration'])
        if updated:
            torch.save(dict(arm='P3', model_state_dict=net.state_dict(), quality_objective=mode), checkpoint)
            best = dict(arm='P3', epoch=epoch, quality_objective=mode, thresholds=thresholds,
                        calibration=metrics, checkpoint=checkpoint.relative_to(output).as_posix(),
                        checkpoint_sha256=train.sha256(checkpoint))
            train.write_json(output / (mode + '_calibration_grid.json'), grid)
        row = dict(stage=mode, epoch=epoch, fit_loss=fit_loss, fit_samples=len(order),
                   calibration_net=metrics['net'], calibration_adopted=metrics['adopted'], selected=updated)
        history.append(row)
        print(json.dumps(row), flush=True)
    net, _ = _load(model, checkpoint, device)
    pred = train.infer(net, cal, policy, device)
    train.prediction_csv(output / (mode + '_calibration.csv.gz'), cal, pred, best['thresholds'])
    train.write_json(output / (mode + '_history.json'), history)
    return best


def fit(args, model, train, quality, manifest, policy, device, receipt):
    # Keep the holdout files unopened in this process. Only fit and calibration
    # content may influence the model, checkpoint, or operating threshold.
    partitions = {key: _corpus(train, args.corpus, manifest, key) for key in ('fit', 'calibration')}
    fit_rows, cal = (partitions[key] for key in ('fit', 'calibration'))
    sources = {key: _source_ids(value) for key, value in partitions.items()}
    if set(sources['fit']) & set(sources['calibration']):
        raise ValueError('Fit and calibration share source groups')
    if any(not (value.arrays['assigned'] & (value.arrays['support_label'] == 1)).any()
           for value in partitions.values()):
        raise ValueError('Fit and calibration require eligible known-positive observations')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'checkpoints').mkdir()
    base = _new_base(model, train, fit_rows, policy, args.output, device)
    base_net, _ = _load(model, base, device)
    base_pred = train.infer(base_net, cal, policy, device)
    frozen = {mode: _refit_head(model, train, quality, fit_rows, cal, base, base_pred, policy,
                               args.output, device, mode) for mode in MODES}
    train.write_json(args.output / 'calibration_lock.json', dict(
        status='CALIBRATION_LOCKED_NOT_ACCEPTED', model_and_policy=receipt,
        base=dict(checkpoint=base.relative_to(args.output).as_posix(), checkpoint_sha256=train.sha256(base)),
        source_ids=sources, modes=frozen, acceptance_rows_read=False,
    ))
    print('CALIBRATION_LOCKED_NOT_ACCEPTED', flush=True)


def _paired_summary(train, corpus, pred, thresholds, policy):
    allowed = corpus.arrays['assigned'] & (corpus.arrays['support_label'] == 1) & pred['box_valid']
    raw, new = corpus.raw_iou[allowed], pred['refined_iou'][allowed]
    measured = pred['quality_refined'][allowed] - pred['quality_raw'][allowed]
    target = new - raw
    use = train.adoption(pred, thresholds)[allowed]
    cutoff = policy['correct_iou_threshold']
    up, down = (raw < cutoff) & (new >= cutoff), (raw >= cutoff) & (new < cutoff)
    meaningful = abs(target) >= policy['objective']['meaningful_delta']
    return dict(
        known_positive_legal=int(allowed.sum()), meaningful_gain_events=int(meaningful.sum()),
        sign_correct=int(((np.sign(target) == np.sign(measured)) & meaningful).sum()),
        mean_absolute_gain_error=float(abs(measured - target).mean()) if len(target) else None,
        legal_upward=int(up.sum()), legal_downward=int(down.sum()),
        upward_adopted=int((up & use).sum()), downward_adopted=int((down & use).sum()),
        accepted_positive_gain=int((use & (target > 0)).sum()),
        accepted_negative_gain=int((use & (target < 0)).sum()),
    )


def _gate(train, old, paired, policy):
    baseline, trial = old['assigned'], paired['assigned']
    group_nets = [entry['net'] for entry in paired['assigned_groups'].values()]
    average = trial['mean_applied_iou']
    checks = dict(
        positive_net=trial['net'] > 0,
        beats_absolute_only=trial['net'] > baseline['net'],
        preserves_damage_budgets=train.within_budget(trial, policy),
        improves_mean_iou=average is not None and average > trial['mean_raw_iou'],
        matches_or_improves_absolute_mean_iou=average is not None and average >= baseline['mean_applied_iou'],
        net_positive_groups=sum(value > 0 for value in group_nets) >= 2,
        net_positive_without_strongest=trial['net'] > max(group_nets, default=0),
    )
    return dict(passes=all(checks.values()), checks=checks)


def accept(args, model, train, manifest, policy, device, receipt):
    lock_path = args.output / 'calibration_lock.json'
    decision = args.output / 'acceptance.json'
    if decision.exists():
        raise FileExistsError(decision)
    lock = _read(lock_path)
    if (lock['status'] != 'CALIBRATION_LOCKED_NOT_ACCEPTED' or lock['model_and_policy'] != receipt
            or lock['acceptance_rows_read']):
        raise ValueError('Locked receipt differs from acceptance input')
    base_path = args.output / lock['base']['checkpoint']
    if train.sha256(base_path) != lock['base']['checkpoint_sha256']:
        raise ValueError('The frozen proposal checkpoint changed after fitting')
    base, base_state = _load(model, base_path, device)
    models = {}
    for mode in MODES:
        entry = lock['modes'][mode]
        checkpoint = args.output / entry['checkpoint']
        if train.sha256(checkpoint) != entry['checkpoint_sha256']:
            raise ValueError(f'The frozen {mode} checkpoint changed after calibration')
        net, _ = _load(model, checkpoint, device)
        _assert_fixed_proposer(net, base_state)
        models[mode] = net
    heldout = _corpus(train, args.corpus, manifest, 'holdout')
    source_ids = _source_ids(heldout)
    if any(set(source_ids) & set(group) for group in lock['source_ids'].values()):
        raise ValueError('Acceptance groups overlap fitting or calibration')
    base_pred = train.infer(base, heldout, policy, device)
    comparators = {}
    for mode, net in models.items():
        pred = train.infer(net, heldout, policy, device)
        _assert_same_predictions(base_pred, pred)
        thresholds = lock['modes'][mode]['thresholds']
        comparators[mode] = train.measures(heldout, pred, thresholds, policy)
        comparators[mode]['quality_pairs'] = _paired_summary(train, heldout, pred, thresholds, policy)
        train.prediction_csv(args.output / (mode + '_acceptance.csv.gz'), heldout, pred, thresholds)
    gate = _gate(train, comparators['absolute_only'], comparators['paired_gain'], policy)
    status = PASS_STATUS if gate['passes'] else STOP_STATUS
    acceptance = dict(status=status, split='holdout', source_ids=source_ids, gate=gate,
                      metrics=comparators, calibration_lock_sha256=train.sha256(lock_path),
                      acceptance_used_for_fitting_or_tuning=False, same_proposal_and_support=True,
                      label_counts=dict(sorted(Counter(map(int, heldout.arrays['support_label'])).items())))
    if gate['passes']:
        selected = args.output / 'selection.json'
        train.write_json(selected, dict(**lock['modes']['paired_gain'], status=status,
                                       calibrated_split='calibration', gate_split='holdout',
                                       **receipt, calibration_lock_sha256=train.sha256(lock_path)))
        train.write_json(args.output / 't2_gate.json', dict(allow_t3a=True, selection_sha256=train.sha256(selected)))
    train.write_json(decision, acceptance)
    print(status, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('fit', 'accept'))
    parser.add_argument('--base-source', type=Path, required=True, help='Directory with original train.py and model.py')
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy', type=Path, default=Path(__file__).with_name('policy.json'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    sys.path.insert(0, str(args.base_source.resolve()))
    import model
    import train
    import quality_loss

    crop_path = args.corpus / 'crop_manifest.json'
    manifest, policy = _read(crop_path), _read(args.policy)
    receipt = dict(crop_manifest_sha256=train.sha256(crop_path), policy_sha256=train.sha256(args.policy),
                   original_model_sha256=train.sha256(Path(model.__file__)),
                   original_trainer_sha256=train.sha256(Path(train.__file__)),
                   quality_source_sha256=train.sha256(Path(quality_loss.__file__)),
                   quality_runner_sha256=train.sha256(Path(__file__)))
    if (manifest.get('schema') != 'tco_v0.1_crop_corpus' or manifest.get('status') != 'LOCKED'
            or manifest.get('dataset_from_official_training_only') is not True
            or manifest.get('model_source_sha256') != receipt['original_model_sha256']
            or set(manifest['partitions']) != {'fit', 'calibration', 'holdout'}):
        raise ValueError('Expected a new locked official-training crop cohort with the original model fingerprint')
    if (policy.get('schema') != 'tco_q2_paired_quality_v1'
            or policy['correct_iou_threshold'] != policy['objective']['correct_iou_threshold']
            or min(policy['base_epochs'], policy['quality_epochs'], policy['batch_size'], args.threads) < 1):
        raise ValueError('Invalid registered TCO quality training policy')
    quality_loss.validate_config(policy['objective'])
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if args.stage == 'fit':
        fit(args, model, train, quality_loss, manifest, policy, device, receipt)
    else:
        accept(args, model, train, manifest, policy, device, receipt)


if __name__ == '__main__':
    main()
