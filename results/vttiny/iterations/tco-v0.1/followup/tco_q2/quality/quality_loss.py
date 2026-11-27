"""Train the existing shared TCO quality head on paired localization effects."""

import math

import torch
from torch.nn import functional as F

from model import loss_terms as baseline_loss_terms
from model import paired_iou


MODES = ('absolute_only', 'paired_gain')


def validate_config(config):
    """Fail before fitting if the registered quality objective is incomplete."""
    positive = (
        'absolute_weight', 'gain_regression_weight', 'gain_regression_beta',
        'direction_weight', 'meaningful_delta', 'direction_margin', 'crossing_weight',
    )
    if any(not math.isfinite(config[key]) or config[key] <= 0 for key in positive):
        raise ValueError('Quality loss weights, margins, and scales must be finite and positive')
    if not 0 < config['correct_iou_threshold'] < 1 or config['crossing_weight'] < 1:
        raise ValueError('Invalid IoU cutoff or crossing weight')


def _weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def quality_targets(output, batch, support_labels, gt_xyxy, config):
    """Compute detached identity-conditioned IoUs for the exact proposal pair."""
    with torch.no_grad():
        positive = support_labels == 1
        known = positive | (support_labels == 0)
        raw = torch.zeros_like(output['quality_raw'])
        refined = torch.zeros_like(raw)
        raw[positive] = paired_iou(batch['raw_xyxy'][positive], gt_xyxy[positive])
        refined[positive] = paired_iou(output['refined_xyxy'][positive], gt_xyxy[positive])
        pair = positive & output['box_valid']
        crop_cutoff = config['correct_iou_threshold']
        crossing = (raw >= crop_cutoff) ^ (refined >= crop_cutoff)
    return known, pair, raw, refined, crossing


def _paired_gain(output, pair, raw, refined, crossing, weight, config):
    target = (refined - raw)[pair]
    predicted = (output['quality_refined'] - output['quality_raw'])[pair]
    weight = weight[pair] * torch.where(crossing[pair], config['crossing_weight'], 1.0)
    magnitude = _weighted_mean(
        F.smooth_l1_loss(predicted, target, beta=config['gain_regression_beta'], reduction='none'),
        weight,
    )
    improving = target >= config['meaningful_delta']
    damaging = target <= -config['meaningful_delta']
    margin = config['direction_margin']
    # Give both error directions equal influence when they are present.
    # When a batch contains only one direction it receives its ordinary mean.
    one_direction = improving.any().to(weight.dtype) + damaging.any().to(weight.dtype)
    sign = (
        _weighted_mean(F.relu(margin - predicted[improving]), weight[improving])
        + _weighted_mean(F.relu(margin + predicted[damaging]), weight[damaging])
    ) / one_direction.clamp_min(1.0)
    return magnitude, sign


def loss_terms(output, batch, support_labels, gt_xyxy, config, *, mode='paired_gain', sample_weight=None):
    """Return a head-only objective. It has no gradient to box coordinates."""
    if mode not in MODES:
        raise ValueError(f'Unknown quality objective {mode}')
    known, pair, raw, refined, crossing = quality_targets(output, batch, support_labels, gt_xyxy, config)
    weight = (torch.ones_like(raw) if sample_weight is None
              else sample_weight.to(device=raw.device, dtype=raw.dtype))
    q_raw = _weighted_mean(
        F.smooth_l1_loss(output['quality_raw'][known], raw[known], reduction='none'), weight[known],
    )
    q_refined = _weighted_mean(
        F.smooth_l1_loss(output['quality_refined'][known], refined[known], reduction='none'), weight[known],
    )
    loss = config['absolute_weight'] * (q_raw + q_refined)
    result = {'quality_raw': q_raw, 'quality_refined': q_refined}
    if mode == 'paired_gain':
        gain, direction = _paired_gain(output, pair, raw, refined, crossing, weight, config)
        loss = loss + config['gain_regression_weight'] * gain + config['direction_weight'] * direction
        result.update(quality_gain=gain, quality_direction=direction)
    return dict(loss=loss, **result)


def joint_paired_loss(output, batch, support_labels, gt_xyxy, config):
    """Optional fixed-architecture joint variant, preserving the old loc/support losses."""
    baseline = baseline_loss_terms(output, batch, support_labels, gt_xyxy)
    quality = loss_terms(output, batch, support_labels, gt_xyxy, config)
    quality['loss'] = baseline['support'] + 2 * baseline['box'] + baseline['giou'] + quality['loss']
    return dict(support=baseline['support'], box=baseline['box'], giou=baseline['giou'], **quality)
