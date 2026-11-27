"""TCO v0.1 crop, network, supervised objectives, and deployment gates."""

from math import log, sqrt

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ARMS = ('P1', 'P2', 'P3')
CURRENT_SIZE = 96
HISTORY_SIZE = 64
MAX_AGE = 12
HISTORY_COUNT = 3
THRESHOLD_KEYS = ('support', 'refined_quality', 'quality_gain')


def _box(box):
    if box is None:
        return None
    values = np.asarray(box, dtype=np.float64)
    if (values.shape != (4,) or not np.isfinite(values).all()
            or not np.all(values[2:] > values[:2])):
        return None
    return values


def roi_transform(box, shape, *, history=False):
    """Affine maps original edges to continuous letterbox edges, matching T0."""
    box = _box(box)
    if box is None:
        return None, 'invalid_box'
    height, width = shape[:2]
    factor, minimum, size = (1.5, 12, HISTORY_SIZE) if history else (2.5, 24, CURRENT_SIZE)
    extent = np.maximum(minimum, factor * (box[2:] - box[:2]))
    if extent.max() > 128:
        return None, 'roi_axis_exceeds_128'
    left_top = (box[:2] + box[2:] - extent) / 2
    right_bottom = left_top + extent
    valid = np.clip(np.concatenate((left_top, right_bottom)), 0, (width, height, width, height))
    if np.prod(valid[2:] - valid[:2]) < 0.5 * np.prod(extent):
        return None, 'roi_valid_fraction_below_half'
    scale = size / extent.max()
    padding = (size - extent * scale) / 2
    affine = np.array([*left_top, scale, *padding], dtype=np.float64)
    return (affine, valid, size), None


def _coverage(rect, size):
    """Exact subpixel rectangle coverage for each unit-area letterbox cell."""
    pixels = np.arange(size)
    coverage_x = np.clip(np.minimum(pixels + 1, rect[2]) - np.maximum(pixels, rect[0]), 0, 1)
    coverage_y = np.clip(np.minimum(pixels + 1, rect[3]) - np.maximum(pixels, rect[1]), 0, 1)
    return np.multiply.outer(coverage_y, coverage_x)


def _map_edges(box, affine):
    source_x, source_y, ratio, pad_x, pad_y = affine
    return (box - (source_x, source_y, source_x, source_y)) * ratio + (pad_x, pad_y, pad_x, pad_y)


def _crop(image, box, transform):
    affine, valid, size = transform
    origin_x, origin_y, scale, pad_x, pad_y = affine
    # OpenCV addresses pixel centers at integers. The box convention addresses
    # the corresponding edges at integers, hence the two half-pixel terms.
    inverse = np.array([
        [1 / scale, 0, origin_x + (0.5 - pad_x) / scale - 0.5],
        [0, 1 / scale, origin_y + (0.5 - pad_y) / scale - 0.5],
    ])
    sampled = cv2.warpAffine(image, inverse, (size, size),
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
    valid_coverage = _coverage(_map_edges(valid, affine), size)
    clipped = np.concatenate((np.maximum(box[:2], valid[:2]), np.minimum(box[2:], valid[2:])))
    proposal_coverage = _coverage(_map_edges(clipped, affine), size)
    sampled[valid_coverage == 0] = 128
    return np.concatenate((sampled[:, :, ::-1].transpose(2, 0, 1),
                           np.rint(255 * valid_coverage)[None].astype(np.uint8),
                           np.rint(255 * proposal_coverage)[None].astype(np.uint8)), axis=0)


def _has_image(value):
    return (isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim == 3
            and value.shape[2] == 3 and min(value.shape[:2]) > 0)


def make_sample(image_bgr, raw_xyxy, pred_xyxy, raw_score, frame, history, load_history, *,
                arm='P3', include_current_repeat=False):
    """Build the same causal input for corpus extraction and T3a, with fallback."""
    if arm not in ARMS:
        raise ValueError(f'Invalid TCO arm {arm}')
    if not _has_image(image_bgr):
        return None, 'current_image_unavailable'
    raw = _box(raw_xyxy)
    predicted = _box(pred_xyxy)
    if predicted is None or not np.isfinite(raw_score):
        return None, 'invalid_geometry'
    transform, reason = roi_transform(raw, image_bgr.shape)
    if reason:
        return None, reason

    past = sorted(history, key=lambda row: row['frame'])
    if any(row['frame'] >= frame for row in past):
        return None, 'noncausal_history'
    tail = [row for row in past if frame - row['frame'] <= MAX_AGE][-HISTORY_COUNT:]
    if len({row['frame'] for row in tail}) != len(tail):
        return None, 'duplicate_history_frame'
    hist_valid = np.zeros(HISTORY_COUNT, dtype=bool)
    hist_dt = np.zeros(HISTORY_COUNT, dtype=np.float32)
    hist_pixels = np.zeros((HISTORY_COUNT, 5, HISTORY_SIZE, HISTORY_SIZE), dtype=np.uint8)
    for slot, observation in enumerate(tail, start=HISTORY_COUNT - len(tail)):
        observation_box = _box(observation.get('raw_xyxy'))
        old_frame = load_history(observation)
        if not _has_image(old_frame):
            continue
        old_transform, old_reason = roi_transform(observation_box, old_frame.shape, history=True)
        if old_reason:
            continue
        if arm == 'P3':
            hist_pixels[slot] = _crop(old_frame, observation_box, old_transform)
        hist_valid[slot] = True
        hist_dt[slot] = frame - observation['frame']
    if hist_valid.sum() < 2:
        return None, 'history_fewer_than_two'

    height, width = image_bgr.shape[:2]
    wh = raw[2:] - raw[:2]
    predicted_wh = predicted[2:] - predicted[:2]
    offset = ((predicted[:2] + predicted[2:]) - (raw[:2] + raw[2:])) / (2 * wh)
    geometry = np.array([raw_score, *np.log(wh / (width, height)), *offset,
                         *np.log(predicted_wh / wh), hist_dt[hist_valid][-1] / MAX_AGE], dtype=np.float32)
    affine, valid, _ = transform
    sample = {'geometry': geometry, 'raw_xyxy': raw.astype(np.float32),
              'roi_affine': affine.astype(np.float32), 'valid_xyxy': valid.astype(np.float32),
              'hist_valid': hist_valid, 'hist_dt': hist_dt}
    if arm != 'P1':
        sample['current'] = _crop(image_bgr, raw, transform)
        sample['history'] = hist_pixels
        if arm == 'P2' or include_current_repeat:
            repeat_transform, repeat_reason = roi_transform(raw, image_bgr.shape, history=True)
            if repeat_reason:
                return None, repeat_reason
            sample['current_repeat'] = _crop(image_bgr, raw, repeat_transform)
    return sample, None


def stack_samples(samples, device=None):
    """Convert a list of uniformly structured NumPy samples to a torch batch."""
    if not samples:
        raise ValueError('Cannot stack zero TCO samples')
    return {key: torch.as_tensor(np.stack([sample[key] for sample in samples]), device=device)
            for key in samples[0]}


def _conv(in_channels, out_channels, *, kernel=3, stride=1):
    return [nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2),
            nn.GroupNorm(8, out_channels), nn.SiLU()]


def _head(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))


def _pixels(crops):
    rgb = crops[:, :3].float() / 127.5 - 1.0
    masks = crops[:, 3:].float() / 255
    return torch.cat((rgb * masks[:, :1], masks), dim=1)


def _masks(crops):
    return F.avg_pool2d(crops[:, 3:5], 4, stride=4).split(1, dim=1)


def _masked_mean(feature, region):
    return (feature * region).sum(dim=(-2, -1)) / region.sum(dim=(-2, -1)).clamp_min(1e-8)


def _residual_box(raw_xyxy, residual):
    size = raw_xyxy[:, 2:] - raw_xyxy[:, :2]
    fraction = residual.to(raw_xyxy.dtype).tanh()
    shift = (size * 0.25).clamp_max(3.0) * fraction[:, :2]
    log_scale = log(1.25) * fraction[:, 2:]
    half_change = size * log_scale.expm1() / 2
    # Express the update relative to existing edges. A zero residual retains
    # those edges exactly instead of recomputing them via center/width.
    refined = torch.cat((raw_xyxy[:, :2] + shift - half_change,
                         raw_xyxy[:, 2:] + shift + half_change), dim=1)
    return refined, torch.cat((shift, log_scale), dim=1)


def _normalized_box(box, affine):
    origin = affine[:, :2].repeat(1, 2)
    padding = affine[:, 3:].repeat(1, 2)
    return (box - origin) * affine[:, 2:3] * (2 / CURRENT_SIZE) + padding * (2 / CURRENT_SIZE) - 1


class TCO(nn.Module):
    """Geometry control or the shared v0.1 temporal CNN for P2 and P3."""

    def __init__(self, arm='P3'):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f'Invalid TCO arm {arm}')
        self.arm = arm
        if arm == 'P1':
            self.geometry_encoder = _head(8, 64, 64)
            self.support_head = _head(64, 64, 1)
            self.box_head = _head(64, 128, 4)
            self.quality_head = _head(68, 128, 1)
            return
        self.encoder = nn.Sequential(*_conv(5, 32, stride=2), *_conv(32, 64, stride=2),
                                     *_conv(64, 64), *_conv(64, 64))
        self.age_encoder = nn.Sequential(nn.Linear(1, 8), nn.SiLU())
        self.token_encoder = _head(136, 64, 64)
        self.query = nn.Linear(64, 64)
        self.key = nn.Linear(64, 64)
        self.value = nn.Linear(64, 64)
        self.spatial_fusion = nn.Sequential(*_conv(133, 64, kernel=1), *_conv(64, 64))
        self.support_head = _head(200, 64, 1)
        self.box_head = _head(200, 128, 4)
        self.quality_head = _head(644, 128, 1)

    def _token(self, feature, valid, proposal, age):
        target = _masked_mean(feature, proposal)
        context = _masked_mean(feature, (valid - proposal).clamp_min(0))
        return self.token_encoder(torch.cat((target, context, self.age_encoder(age / MAX_AGE)), dim=1))

    def _image_features(self, batch):
        current = _pixels(batch['current'])
        feature = self.encoder(current)
        valid, proposal = _masks(current)
        current_query = self.query(self._token(feature, valid, proposal, batch['hist_dt'].new_zeros((len(feature), 1))))

        if self.arm == 'P2':
            repeat = _pixels(batch['current_repeat'])
            history_feature = self.encoder(repeat).repeat_interleave(HISTORY_COUNT, dim=0)
            history_valid, history_proposal = (item.repeat_interleave(HISTORY_COUNT, dim=0) for item in _masks(repeat))
        else:
            history = _pixels(batch['history'].flatten(0, 1))
            history_feature = self.encoder(history)
            history_valid, history_proposal = _masks(history)
        history_tokens = self._token(history_feature, history_valid, history_proposal,
                                     batch['hist_dt'].reshape(-1, 1)).reshape(len(feature), HISTORY_COUNT, 64)
        weights = (self.key(history_tokens) * current_query[:, None, :]).sum(dim=2) / sqrt(64)
        mask = batch['hist_valid']
        # A finite mask avoids all-masked softmax. Usable inputs have at least
        # two slots, but a direct all-empty forward still stays finite.
        attention = weights.masked_fill(~mask, torch.finfo(weights.dtype).min).softmax(dim=1) * mask
        history_query = (attention[:, :, None] * self.value(history_tokens)).sum(dim=1)

        response = (F.normalize(feature, dim=1, eps=1e-6)
                    * F.normalize(history_query, dim=1, eps=1e-6)[:, :, None, None]).sum(dim=1, keepdim=True)
        # Area-pooled mask cells and align_corners=False quality sampling share
        # these continuous cell-center coordinates on the letterbox.
        cells = feature.shape[-1]
        centers = (torch.arange(cells, device=feature.device, dtype=batch['roi_affine'].dtype) + 0.5) * (CURRENT_SIZE / cells)
        affine = batch['roi_affine']
        x = (centers[None, :] - affine[:, 3:4]) / affine[:, 2:3] + affine[:, :1]
        y = (centers[None, :] - affine[:, 4:5]) / affine[:, 2:3] + affine[:, 1:2]
        raw = batch['raw_xyxy']
        middle = (raw[:, :2] + raw[:, 2:]) / 2
        size = raw[:, 2:] - raw[:, :2]
        x = ((x - middle[:, :1]) / size[:, :1])[:, None, None, :].expand(-1, 1, cells, -1)
        y = ((y - middle[:, 1:2]) / size[:, 1:2])[:, None, :, None].expand(-1, 1, -1, cells)
        broadcast = history_query[:, :, None, None].expand(-1, -1, cells, cells)
        fused = self.spatial_fusion(torch.cat((feature * valid, broadcast * valid, response * valid,
                                              proposal, valid, x * valid, y * valid), dim=1))
        pooled = torch.cat((_masked_mean(fused, valid), _masked_mean(fused, proposal), history_query,
                            batch['geometry']), dim=1)
        return pooled, fused * valid, history_query

    def _quality_features(self, box, affine, pooled, fused, history_query):
        roi_box = _normalized_box(box, affine).detach()
        if self.arm == 'P1':
            return torch.cat((pooled, roi_box), dim=1)
        fractions = roi_box.new_tensor((1 / 6, 1 / 2, 5 / 6))
        xs = roi_box[:, :1] + (roi_box[:, 2:3] - roi_box[:, :1]) * fractions
        ys = roi_box[:, 1:2] + (roi_box[:, 3:4] - roi_box[:, 1:2]) * fractions
        grid = torch.stack((xs[:, None, :].expand(-1, 3, -1), ys[:, :, None].expand(-1, -1, 3)), dim=-1)
        local = F.grid_sample(fused, grid, mode='bilinear', padding_mode='zeros', align_corners=False).flatten(1)
        return torch.cat((local, history_query, roi_box), dim=1)

    def forward(self, batch):
        if self.arm == 'P1':
            pooled, fused, history_query = self.geometry_encoder(batch['geometry']), None, None
        else:
            pooled, fused, history_query = self._image_features(batch)
        support_logit = self.support_head(pooled).squeeze(1)
        raw = batch['raw_xyxy']
        refined, delta = _residual_box(raw, self.box_head(pooled))
        boxes = (raw, refined)
        quality = self.quality_head(torch.cat(tuple(
            self._quality_features(box, batch['roi_affine'], pooled, fused, history_query) for box in boxes), dim=0))
        raw_quality, refined_quality = quality.squeeze(1).sigmoid().chunk(2)
        valid_bounds = batch['valid_xyxy']
        box_valid = ((refined[:, :2] >= valid_bounds[:, :2]).all(dim=1)
                     & (refined[:, 2:] <= valid_bounds[:, 2:]).all(dim=1)
                     & (refined[:, 2:] > refined[:, :2]).all(dim=1) & refined.isfinite().all(dim=1))
        return {'support_logit': support_logit, 'support': support_logit.sigmoid(),
                'quality_raw': raw_quality, 'quality_refined': refined_quality,
                'refined_xyxy': refined, 'box_delta': delta, 'box_valid': box_valid}


def paired_iou(a, b, *, generalized=False):
    """Aligned IoU/GIoU on continuous unmodified box coordinates."""
    inter = (torch.minimum(a[:, 2:], b[:, 2:]) - torch.maximum(a[:, :2], b[:, :2])).clamp_min(0).prod(dim=1)
    union = (a[:, 2:] - a[:, :2]).prod(dim=1) + (b[:, 2:] - b[:, :2]).prod(dim=1) - inter
    iou = inter / union.clamp_min(1e-8)
    if not generalized:
        return iou
    enclosing = (torch.maximum(a[:, 2:], b[:, 2:]) - torch.minimum(a[:, :2], b[:, :2])).prod(dim=1)
    return iou - (enclosing - union) / enclosing.clamp_min(1e-8)


def _encode_box(box, raw):
    size = raw[:, 2:] - raw[:, :2]
    center_difference = ((box[:, :2] + box[:, 2:]) - (raw[:, :2] + raw[:, 2:])) / 2
    return torch.cat((center_difference / size, ((box[:, 2:] - box[:, :2]) / size).log()), dim=1)


def loss_terms(output, batch, support_labels, gt_xyxy, sample_weight=None):
    """Compute masked support, box and identity-conditioned quality objectives."""
    known = (support_labels == 0) | (support_labels == 1)
    positive = support_labels == 1
    raw = batch['raw_xyxy']
    refined = output['refined_xyxy']
    logits = output['support_logit']
    labels = support_labels.to(logits.dtype)
    weight = torch.ones_like(logits) if sample_weight is None else sample_weight.to(logits.dtype)

    def mean(values, selected):
        selected_weights = weight[selected]
        return (values * selected_weights).sum() / selected_weights.sum().clamp_min(1e-8)

    support = mean(F.binary_cross_entropy_with_logits(logits[known], labels[known], reduction='none'), known)
    box = mean(F.smooth_l1_loss(_encode_box(refined[positive], raw[positive]),
                               _encode_box(gt_xyxy[positive], raw[positive]), reduction='none').mean(dim=1), positive)
    giou = mean(1 - paired_iou(refined[positive], gt_xyxy[positive], generalized=True), positive)
    with torch.no_grad():
        raw_quality = torch.zeros_like(logits)
        refined_quality = torch.zeros_like(logits)
        raw_quality[positive] = paired_iou(raw[positive], gt_xyxy[positive])
        refined_quality[positive] = paired_iou(refined[positive], gt_xyxy[positive])
    q_raw = mean(F.smooth_l1_loss(output['quality_raw'][known], raw_quality[known], reduction='none'), known)
    q_refined = mean(F.smooth_l1_loss(output['quality_refined'][known], refined_quality[known], reduction='none'), known)
    return {'loss': support + 2 * box + giou + 0.5 * (q_raw + q_refined), 'support': support,
            'box': box, 'giou': giou, 'quality_raw': q_raw, 'quality_refined': q_refined}


def apply_thresholds(output, thresholds):
    """Keep the original measurement unless the calibrated proposal is legal."""
    return (output['box_valid'] & (output['support'] >= thresholds['support'])
            & (output['quality_refined'] >= thresholds['refined_quality'])
            & (output['quality_refined'] - output['quality_raw'] >= thresholds['quality_gain']))
