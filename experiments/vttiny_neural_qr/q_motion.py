"""Train and apply extra positive-semidefinite acceleration noise for Lost tracks."""
from argparse import ArgumentParser
from collections import deque
import gzip
import json
from pathlib import Path
import sys

import numpy as np


HISTORY = 5
MIN_HISTORY = 3
HIDDEN = 8
LIMITS = np.log(np.array([16., 4.]))
MIN_LOG_VARIANCE = -12.
AXES = ('cx', 'cy', 'w', 'h')
FEATURES = ('log_scaled_center_speed', 'log_scaled_shape_speed', 'log_scaled_center_accel',
            'log_scaled_shape_accel', 'log_width', 'log_height')
ACCELERATION = np.concatenate((.5 * np.eye(4), np.eye(4)), axis=0)
UNITS = np.stack([ACCELERATION[:, group] @ ACCELERATION[:, group].T
                  for group in (slice(0, 2), slice(2, 4))])


def history_features(history):
    """Use only posterior states from the current uninterrupted observation history."""
    means = np.asarray([mean for _, mean in history])
    width, height = np.maximum(means[-1, 2:4], 1.)
    scale = np.sqrt(width * height)
    velocity = means[:, 4:8]
    acceleration = np.diff(velocity, axis=0)
    speeds = [np.median(np.linalg.norm(velocity[:, group], axis=1)) for group in (slice(0, 2), slice(2, 4))]
    accelerations = [np.median(np.linalg.norm(acceleration[:, group], axis=1)) for group in (slice(0, 2), slice(2, 4))]
    motion = np.clip(np.log1p(5. * np.asarray(speeds + accelerations) / scale), 0., 4.)
    size = np.clip(np.log(np.array([width, height]) / 12.), -4., 4.)
    return np.concatenate((motion, size))


def record(history, frame, mean):
    if history and frame == history[-1][0]:
        return False
    if history and frame != history[-1][0] + 1:
        history.clear()
    if mean.shape != (8,) or not np.isfinite(mean).all():
        raise ValueError('Expected a finite XYWH state')
    history.append((frame, mean.copy()))
    return True


def covariance_parameters(parameters, standardized, method):
    """Return center and scale acceleration variances in original RGB pixels squared."""
    if method == 'q_constant':
        logits = np.broadcast_to(parameters, (len(standardized), 2))
        hidden = None
    else:
        n_features = len(FEATURES)
        boundary = n_features * HIDDEN
        w1 = parameters[:boundary].reshape(n_features, HIDDEN)
        b1 = parameters[boundary:boundary + HIDDEN]
        w2 = parameters[boundary + HIDDEN:-2].reshape(HIDDEN, 2)
        b2 = parameters[-2:]
        hidden = np.tanh(standardized @ w1 + b1)
        logits = hidden @ w2 + b2
    sigmoid = 1. / (1. + np.exp(-logits))
    variance = np.exp(MIN_LOG_VARIANCE + (LIMITS - MIN_LOG_VARIANCE) * sigmoid)
    return variance, sigmoid, hidden


class QMotion:
    """Apply frozen per-track variances after native predict and before native GMC."""

    def __init__(self, policy, method='q_mlp'):
        if not isinstance(policy, dict):
            policy = json.loads(Path(policy).read_bytes())
        if policy['version'] != 1 or policy['features'] != list(FEATURES):
            raise ValueError('Incompatible Q policy schema')
        if method not in ('q_constant', 'q_mlp'):
            raise ValueError(f'Unknown Q method: {method}')
        self.method = method
        self.offset = np.asarray(policy['feature_mean'], dtype=float)
        self.scale = np.asarray(policy['feature_scale'], dtype=float)
        self.weights = np.asarray(policy['parameters'][method], dtype=float)
        self.injected = 0

    def variances(self, features):
        standardized = (np.atleast_2d(features) - self.offset) / self.scale
        variance, _, _ = covariance_parameters(self.weights, standardized, self.method)
        return variance

    def observe(self, track):
        if not hasattr(track, '_qr_q_history'):
            track._qr_q_history = deque(maxlen=HISTORY)
        history = track._qr_q_history
        if record(history, int(track.frame_id), np.asarray(track.mean)):
            track._qr_q_acceleration = self.variances(history_features(history))[0] if len(history) >= MIN_HISTORY else None

    def apply(self, tracks):
        from boxmot.trackers.botsort.basetrack import TrackState

        for track in tracks:
            variance = getattr(track, '_qr_q_acceleration', None)
            if track.state == TrackState.Lost and variance is not None:
                track.covariance += np.einsum('i,ijk->jk', variance, UNITS)
                self.injected += 1


def make_windows(kf, measurements, truth, warps, horizons, next_state, rng, max_anchors):
    gt = {(int(row.frame), int(row.gid)): np.array([getattr(row, axis) for axis in AXES])
          for row in truth.itertuples()}
    anchors = []
    for gid, track in measurements.groupby('gid'):
        history = deque(maxlen=HISTORY)
        last = -1
        count = 0
        for row in track.sort_values('frame').itertuples():
            frame = int(row.frame)
            z = np.array([getattr(row, axis) for axis in AXES])
            if frame == last + 1:
                mean, cov = next_state(kf, mean, cov, warps[frame], False)
                mean, cov = kf.update(mean, cov, z)
                count += 1
            else:
                mean, cov = kf.initiate(z)
                count = 1
            record(history, frame, mean)
            if len(history) >= MIN_HISTORY and count % 4 == MIN_HISTORY:
                anchors.append((frame, int(gid), mean.copy(), cov.copy(), history_features(history)))
            last = frame

    motion = kf._motion_mat
    for index in sorted(rng.choice(len(anchors), min(max_anchors, len(anchors)), replace=False)):
        anchor, gid, mean, cov, features = anchors[index]
        propagated = np.zeros_like(UNITS)
        for horizon in range(1, max(horizons) + 1):
            frame = anchor + horizon
            if frame not in warps:
                break
            warp = warps[frame]
            lost = horizon > 1
            mean, cov = next_state(kf, mean, cov, warp, lost)
            propagated = warp[0] @ (motion @ propagated @ motion.T + (UNITS if lost else 0.)) @ warp[0].T
            if horizon in horizons and (frame, gid) in gt:
                yield horizon, features, gt[frame, gid] - mean[:4], cov[:4, :4], propagated[:, :4, :4]


def training_windows(seed, max_anchors):
    import cv2
    import pandas as pd
    from qr_settings import KF, MOTION, PROJECT, TRAIN, read, sha

    # The previous motion experiment defines the estimand's native SOF and state transport.
    sys.path.append(str(MOTION))
    from train_q import HORIZONS, KalmanFilterXYWH, SOF, next_state

    pairs = pd.read_pickle(KF/'pairs.pkl').query('iou>=.30 and gap>=.15')
    truth = pd.read_pickle(KF/'truth.pkl')
    plan = read(TRAIN/'window_plan.json')
    info = read(KF/'inputs.json')
    allowed = set()
    for split in ('train', 'val'):
        with (PROJECT/'data/processed_v1'/(split + '_pairs.jsonl')).open(encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line)
                if entry['official_split'] != 'train':
                    raise ValueError('Non-training sequence in training manifest')
                allowed.add(entry['sequence'])
    groups = {split: sorted(set(rows.parent)) for split, rows in pairs.groupby('split')}
    tests = read(PROJECT/'data/processed_v1/splits.json')['splits']['test']
    test_parents = {seq.rsplit('_', 1)[0] for seq in tests}
    if (sha(TRAIN/'window_plan.json') != info['window_plan'] or len(allowed) != plan['official_train']
            or not set(pairs.seq) <= allowed & set(plan['sequences'])
            or set(groups) != {'fit', 'check'} or set(groups['fit']) & set(groups['check'])
            or test_parents & (set(groups['fit']) | set(groups['check']))
            or any(plan['groups'].get(r.parent) != r.split or r.seq.rsplit('_', 1)[0] != r.parent
                   for r in pairs[['seq','parent','split']].drop_duplicates().itertuples())):
        raise ValueError('Training footage, cached input plan or parent-video split disagrees')

    rng = np.random.default_rng(seed)
    cv2.setNumThreads(1)
    kf = KalmanFilterXYWH()
    metadata, samples = [], []
    for seq, rows in pairs.groupby('seq', sort=True):
        path = TRAIN/'training_inputs/detections'/(seq + '.json.gz')
        if sha(path) != info['detection_files'][seq]:
            raise ValueError(f'Training detector cache changed for {seq}')
        with gzip.open(path, 'rt', encoding='utf-8') as file:
            cache = json.load(file)
        if cache['model_sha256'] != info['detector']:
            raise ValueError(f'Training detector mismatch for {seq}')
        cv2.setRNGSeed(0)
        sof = SOF()
        warps = {}
        for expected, stored in zip(plan['sequences'][seq], cache['frames'], strict=True):
            if expected['frame'] != stored['frame'] or expected['image'] != stored['image']:
                raise ValueError(f'Training image order mismatch for {seq}')
            image = cv2.imread(expected['image'])
            if image is None:
                raise FileNotFoundError(expected['image'])
            boxes = np.asarray(stored['detections'], dtype=float).reshape(-1, 6)
            affine = sof.apply(image, boxes)
            warps[expected['frame']] = (np.kron(np.eye(4), affine[:, :2]), affine[:, 2])
        seq_samples = list(make_windows(kf, rows, truth[truth.seq == seq], warps, HORIZONS, next_state, rng, max_anchors))
        samples.extend(item[1:] for item in seq_samples)
        metadata.extend((rows.split.iloc[0], seq, horizon) for horizon, *_ in seq_samples)
        print(seq, len(seq_samples), flush=True)

    if not samples:
        raise ValueError('No complete simulated trajectory gaps from official training')
    features, residual, covariance, units = [np.stack(x) for x in zip(*samples, strict=True)]
    weights = dict(version=1, features=list(FEATURES), groups=groups, horizon_frames=list(HORIZONS), seed=seed,
                   sequences={split: sorted(set(rows.seq)) for split, rows in pairs.groupby('split')},
                   max_anchors_per_sequence=max_anchors, n_history=HISTORY, min_history=MIN_HISTORY,
                   baseline_measurement='native R on observed detections, latent GT target for validation',
                   detector_sha256=info['detector'],
                   inputs={name: sha(KF/name) for name in ('pairs.pkl','truth.pkl','inputs.json')})
    return (features, residual, covariance, units), pd.DataFrame(metadata, columns=['split', 'seq', 'h']), weights


def nll_terms(residual, covariance):
    sign, logdet = np.linalg.slogdet(covariance)
    if np.any(sign != 1):
        raise ValueError('Covariance must be positive definite')
    inverse = np.linalg.inv(covariance)
    whitened = (inverse @ residual[..., None])[..., 0]
    d2 = np.einsum('ni,ni->n', residual, whitened)
    nll = .5 * (logdet + d2 + residual.shape[1] * np.log(2 * np.pi))
    return nll, d2, inverse, whitened, logdet


def objective(parameters, method, data, weights, regularization=.002):
    features, residual, base, units = data
    variance, sigmoid, hidden = covariance_parameters(parameters, features, method)
    covariance = base + np.einsum('na,naij->nij', variance, units)
    nll, _, inverse, whitened, _ = nll_terms(residual, covariance)
    derivative = .5 * (inverse - np.einsum('ni,nj->nij', whitened, whitened))
    d_variance = np.einsum('nij,naij->na', derivative, units)
    d_logits = (weights[:, None] * d_variance * variance
                * (LIMITS - MIN_LOG_VARIANCE) * sigmoid * (1. - sigmoid))
    if method == 'q_constant':
        return float(np.dot(weights, nll)), d_logits.sum(axis=0)
    w2 = parameters[-2 - HIDDEN * 2:-2].reshape(HIDDEN, 2)
    d_hidden = (d_logits @ w2.T) * (1. - hidden ** 2)
    gradient = np.concatenate(((features.T @ d_hidden).ravel(), d_hidden.sum(axis=0),
                               (hidden.T @ d_logits).ravel(), d_logits.sum(axis=0)))
    penalty = np.r_[parameters[:-2], 0., 0.]
    return float(np.dot(weights, nll) + .5 * regularization * (penalty @ penalty)), gradient + regularization * penalty


def fit_parameters(seed, max_iterations, features, residual, base, units, weights):
    from scipy.optimize import minimize

    initial_log_variance = np.array([-4., -4.])
    initial_prob = (initial_log_variance - MIN_LOG_VARIANCE) / (LIMITS - MIN_LOG_VARIANCE)
    initial = np.log(initial_prob / (1. - initial_prob))
    data = features, residual, base, units
    options = {'maxiter': max_iterations, 'ftol': 1e-10}
    constant = minimize(objective, initial, args=('q_constant', data, weights), method='L-BFGS-B',
                        jac=True, bounds=[(-8., 8.)] * 2, options=options)
    rng = np.random.default_rng(seed)
    nweights = len(FEATURES) * HIDDEN + HIDDEN + HIDDEN * 2
    initial = np.concatenate((rng.normal(0., .12, size=nweights), constant.x))
    network = minimize(objective, initial, args=('q_mlp', data, weights), method='L-BFGS-B', jac=True,
                       bounds=[(-8., 8.)] * len(initial), options=options)
    outcomes = {'q_constant': constant, 'q_mlp': network}
    for method, result in outcomes.items():
        if np.any(~np.isfinite(result.x)) or not np.isfinite(result.fun) or result.status not in (0, 1):
            raise ValueError(f'Q fitting failed for {method}: {result.message}')
    return outcomes


def evaluate(data, meta, policy):
    features, residual, base, units = data
    rows = []
    for method in ('native', 'q_constant', 'q_mlp'):
        covariance = base if method == 'native' else base + np.einsum(
            'na,naij->nij', QMotion(policy, method).variances(features), units)
        for (split, horizon), group in meta.groupby(['split', 'h']):
            sel = group.index.to_numpy()
            for block, axes, cutoff in [('center', slice(0, 2), 5.991465), ('scale', slice(2, 4), 5.991465),
                                        ('joint', slice(0, 4), 9.487729)]:
                errors = residual[sel, axes]
                nll, d2, _, _, logdet = nll_terms(errors, covariance[sel, axes, axes])
                area = float(np.mean(np.pi * cutoff * np.exp(.5 * logdet))) if block != 'joint' else None
                rows.append(dict(split=split, h=int(horizon), method=method, block=block, n=len(sel),
                                 sequences=int(group.seq.nunique()), nll=float(np.mean(nll)), coverage95=float(np.mean(d2 <= cutoff)),
                                 mean_area95=area, mean_l2_error=float(np.mean(np.linalg.norm(errors, axis=1)))))
    return rows


def main():
    from qr_settings import OUT, sha, table, write

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=345)
    parser.add_argument('--max-iterations', type=int, default=250)
    parser.add_argument('--max-anchors', type=int, default=160)
    args = parser.parse_args()
    if min(args.max_iterations, args.max_anchors) < 1:
        parser.error('Iteration and anchor limits must be positive')
    data, meta, policy = training_windows(args.seed, args.max_anchors)
    fit = meta.split.eq('fit').to_numpy()
    if not fit.any() or fit.all():
        raise ValueError('Both parent-video pools require eligible gap samples')
    fit_data = tuple(x[fit] for x in data)
    features = fit_data[0]
    offset = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), .08)
    counts = meta.loc[fit].groupby(['seq', 'h']).h.transform('size').to_numpy()
    fit_weights = 1. / counts
    fit_weights /= fit_weights.sum()
    outcomes = fit_parameters(args.seed, args.max_iterations, (features - offset) / scale, *fit_data[1:], fit_weights)
    policy.update(feature_mean=offset.tolist(), feature_scale=scale.tolist(),
                  parameters={key: result.x.tolist() for key, result in outcomes.items()},
                  optimization={key: dict(objective=float(result.fun), iterations=int(result.nit),
                                         converged=bool(result.success)) for key, result in outcomes.items()},
                  samples={key: int(count) for key, count in meta.groupby('split').size().items()})
    report = evaluate(data, meta, policy)
    write(OUT/'q_policy.json', policy)
    table(OUT/'q_holdout.csv', report)
    print('LOCKED', sha(OUT/'q_policy.json'), flush=True)


if __name__ == '__main__':
    main()
