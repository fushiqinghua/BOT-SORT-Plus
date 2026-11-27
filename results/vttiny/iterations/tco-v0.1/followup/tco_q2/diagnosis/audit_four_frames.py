"""Inspect four archived TCO proposals using official RGB training boxes and pixels."""

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


TARGETS = {('DJI_0083_2', 328, 218, 27): 6600577,
           ('DJI_0083_2', 354, 218, 17): 6600577,
           ('DJI_0183_2', 179, 72, 17): 8001021,
           ('DJI_0183_2', 256, 123, 19): 8001512}
FIELDS = ('sequence', 'frame', 'track_id', 'det_index')
RAW, NEW, GT, ROI = '#ff6662', '#39dbff', '#77ff75', '#e7bc53'


def key(row):
    return tuple(row[name] for name in FIELDS)


def read_gz_csv(path):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def center(box):
    return (box[:2] + box[2:]) / 2


def iou(box, gt):
    area = np.prod(np.maximum(0, np.minimum(box[2:], gt[2:]) - np.maximum(box[:2], gt[:2])))
    return float(area / (np.prod(box[2:] - box[:2]) + np.prod(gt[2:] - gt[:2]) - area))


def read_gt(path, gt_id, frames):
    boxes = {}
    with path.open(encoding='utf-8', newline='') as stream:
        for row in csv.reader(stream):
            frame, track_id = map(int, row[:2])
            if track_id == gt_id and frame in frames and float(row[6]) > 0:
                if frame in boxes:
                    raise ValueError(f'Ambiguous official GT frame {frame} track {track_id}')
                x, y, w, h = map(float, row[2:6])
                boxes[frame] = np.array([x, y, x + w, y + h])
    return boxes


def draw_panel(canvas, draw, image, bounds, location, boxes, caption):
    left, top, right, bottom = bounds
    x, y = location
    scale = 7
    tile = Image.fromarray(image[top:bottom, left:right, ::-1].copy())
    tile = tile.resize(tuple(v * scale for v in tile.size), Image.Resampling.NEAREST)
    canvas.paste(tile, (x, y + 19))
    draw.text((x, y), caption, fill='#f5f6fb')
    for box, color, width in boxes:
        if box is None:
            continue
        screen = (x + (box[0] - left) * scale, y + 19 + (box[1] - top) * scale,
                  x + (box[2] - left) * scale, y + 19 + (box[3] - top) * scale)
        draw.rectangle(screen, outline=color, width=width)


def render(path, case, event, images, gt_by_frame):
    canvas = Image.new('RGB', (1114, 344), '#202533')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    frame = event['frame']
    raw, refined, gt = (np.asarray(case[name]) for name in ('raw_xyxy', 'refined_xyxy', 'gt_xyxy'))
    cx, cy = center(raw)
    height, width = images[frame].shape[:2]
    window = (max(0, round(cx)-15), max(0, round(cy)-15),
              min(width, round(cx)+15), min(height, round(cy)+15))
    draw.text((8, 7), f"{event['sequence']}    f{frame}    native #{event['det_index']}    track {event['track_id']}"
              f"    official GT {case['gt_id']}", fill='white', font=font)
    roi = event['roi']['requested_xyxy']
    draw_panel(canvas, draw, images[frame], window, (8, 35), [], 'Current RGB | native pixels enlarged 7x')
    draw_panel(canvas, draw, images[frame], window, (228, 35),
               [(roi, ROI, 1), (raw, RAW, 3), (refined, NEW, 2), (gt, GT, 2)], 'Aligned comparison | same source pixels')
    for number, past in enumerate(event['history']):
        f = past['frame']
        label = f"Real track history f{f} | det #{past['det_index']}"
        draw_panel(canvas, draw, images[f], window, (448 + 220 * number, 35),
                   [(past['raw_xyxy'], RAW, 2), (gt_by_frame.get(f), GT, 2)], label)
    for x, color, label in [(8, RAW, 'Raw / historical detection'), (241, NEW, 'P3 proposed'),
                             (415, GT, 'Official RGB train GT'), (613, ROI, 'Current model ROI')]:
        draw.rectangle((x, 274, x + 10, 284), fill=color)
        draw.text((x + 16, 273), label, fill='white')
    status = 'ADOPT' if case['adopted'] else 'KEEP RAW'
    draw.text((8, 293), f"det score {event['raw_score']:.4f}  |  support {case['support']:.4f}  |  "
              f"Q raw {case['q_raw']:.5f}  ->  Q proposal {case['q_ref']:.5f}  |  "
              f"predicted gain {case['q_gain']:+.5f}  |  {status}", fill='white', font=font)
    draw.text((8, 316), f"GT IoU {case['raw_iou']:.5f} -> {case['refined_iou']:.5f}  |  "
              f"actual gain {case['true_gain']:+.5f}  |  "
              f"center move {case['proposal_center_shift'][0]:+.3f}, {case['proposal_center_shift'][1]:+.3f} px  |  "
              f"legal ROI: {event['roi']['valid_fraction']:.0%}", fill='white', font=font)
    canvas.save(path)


def inspect_features(t23, rows, cases, frozen):
    import torch
    import model
    from model import TCO
    from train import Corpus

    torch.set_num_threads(1)
    manifest = json.loads((t23 / 'T2_corpus/crop_manifest.json').read_bytes())
    part = manifest['partitions']['holdout']
    if hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest() != manifest['model_source_sha256']:
        raise ValueError('Archived crop/model source hash mismatch')
    corpus = Corpus(t23 / 'T2_corpus/holdout', part['events'], part['capacity'])
    net = TCO(arm='P3').eval()
    checkpoint_path = t23 / 'T2' / frozen['checkpoint']
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != frozen['checkpoint_sha256']:
        raise ValueError('Archived checkpoint hash mismatch')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)
    features = []
    hook = net.quality_head.register_forward_pre_hook(lambda _, args: features.append(args[0].detach()))
    with torch.inference_mode():
        out = net(corpus.batch('P3', np.asarray(rows), torch.device('cpu')))
    hook.remove()
    original, replacement = features[0].chunk(2)
    for idx, case in enumerate(cases):
        q_cpu = [float(out[k][idx]) for k in ('quality_raw', 'quality_refined')]
        cpu_error = max(abs(case[k] - v) for k, v in zip(('q_raw', 'q_ref'), q_cpu, strict=True))
        if cpu_error > 0.002:
            raise ValueError(f'CPU audit scores differ by {cpu_error} from the archived GPU outputs')
        local_raw, local_new = original[idx, :576], replacement[idx, :576]
        case['old_q_features'] = dict(
            raw_ref_sampled_local_cosine=float(torch.nn.functional.cosine_similarity(local_raw, local_new, dim=0)),
            raw_ref_sampled_local_relative_l2=float((local_new-local_raw).norm() / local_raw.norm()),
            identical_history_query=bool(torch.equal(original[idx, 576:640], replacement[idx, 576:640])),
            cpu_quality_values=q_cpu, cpu_vs_archived_gpu_max_abs_q_error=cpu_error)


def build(t23, images_root, output):
    sys.path.insert(0, str(t23 / 'src'))
    from build_corpus import events, load_image_function

    predictions = read_gz_csv(t23 / 'T2/P3_seed41_holdout_predictions.csv.gz')
    frozen = json.loads((t23 / 'T2/P3_seed41_calibration_frozen.json').read_bytes())
    thresholds = frozen['thresholds']
    with gzip.open(t23 / 'T2_corpus/holdout/row_keys.jsonl.gz', 'rt', encoding='utf-8') as stream:
        by_key = {key(row): row for row in map(json.loads, stream) if key(row) in TARGETS}
    records = {key(row): (row, label) for row, label in events(t23 / 'training_labels/holdout')
               if key(row) in TARGETS}
    if set(by_key) != set(TARGETS) or set(records) != set(TARGETS):
        raise ValueError('Missing one or more exact archived candidate keys')
    load = load_image_function(images_root)
    output.mkdir(parents=True, exist_ok=False)
    cases, row_indices = [], []
    for k, gt_id in TARGETS.items():
        archive = by_key[k]
        event, label = records[k]
        entry = predictions[archive['row']]
        assert int(entry['row']) == archive['row'] and entry['assigned'] == 'True' and entry['support_label'] == '1'
        assert entry['box_valid'] == 'True' and event['roi']['eligible'] and event['assigned']
        assert label['gt_id'] == gt_id and event['run_fingerprint'] == archive['run_fingerprint']
        ref_frames = [*event['history'], event]
        frame_images = {row['frame']: load(row) for row in ref_frames}
        gt_by_frame = read_gt(t23 / 'training_gt' / k[0] / 'gt/gt.txt', gt_id, frame_images.keys())
        gt = gt_by_frame[event['frame']]
        raw = np.array(json.loads(entry['raw_xyxy']))
        new = np.array(json.loads(entry['refined_xyxy']))
        assert np.array_equal(gt, label['gt_xyxy']) and np.array_equal(raw, event['raw_xyxy'])
        ir, ip = iou(raw, gt), iou(new, gt)
        assert math.isclose(ir, float(entry['raw_iou']), abs_tol=2e-6)
        assert math.isclose(ip, float(entry['refined_iou']), abs_tol=2e-6)
        qr, qp = float(entry['quality_raw']), float(entry['quality_refined'])
        support = float(entry['support_score'])
        adoption = (support >= thresholds['support'] and qp >= thresholds['refined_quality']
                    and qp-qr >= thresholds['quality_gain'])
        assert adoption == (entry['adopted'] == 'True')
        case = dict(zip(FIELDS, k), gt_id=gt_id, artifact_key=archive, raw_xyxy=raw.tolist(),
                    refined_xyxy=new.tolist(), gt_xyxy=gt.tolist(), kf_predicted_xyxy=event['pred_xyxy'],
                    raw_score=event['raw_score'], roi=event['roi'], history=archive['history'],
                    history_GT_center_offsets={str(f): (center(np.array(p['raw_xyxy']))-center(gt_by_frame[f])).tolist()
                                               for p in event['history'] if (f := p['frame']) in gt_by_frame},
                    gt_need_center_shift=(center(gt)-center(raw)).tolist(), proposal_center_shift=(center(new)-center(raw)).tolist(),
                    proposal_wh_change=((new[2:]-new[:2])-(raw[2:]-raw[:2])).tolist(),
                    raw_iou=ir, refined_iou=ip, true_gain=ip-ir, q_raw=qr, q_ref=qp, q_gain=qp-qr, support=support,
                    adopted=adoption, roi_valid=True, exact_gt_reachable=label['gt_exact_reachable'],
                    feasible_iou=label['feasible_iou'])
        cases.append(case)
        row_indices.append(archive['row'])
        render(output / f'{k[0]}_f{k[1]}.png', case, event, frame_images, gt_by_frame)
    inspect_features(t23, row_indices, cases, frozen)
    known = [r for r in predictions if r['assigned'] == 'True' and r['support_label'] == '1'
             and r['box_valid'] == 'True']
    delta_xy = np.array([[float(r['refined_iou']) - float(r['raw_iou']),
                           float(r['quality_refined']) - float(r['quality_raw'])] for r in known])
    sizable = delta_xy[np.abs(delta_xy[:, 0]) >= 0.02]
    audit = dict(status='PASS', source='old_seed41_training_side_holdout_only', thresholds=thresholds,
                 known_positive_native_pairs_n=len(known),
                 mean_absolute_gt_gain=float(np.mean(np.abs(delta_xy[:, 0]))),
                 mean_absolute_predicted_gain=float(np.mean(np.abs(delta_xy[:, 1]))),
                 gt_and_pred_quality_gain_pearson=float(np.corrcoef(delta_xy.T)[0, 1]),
                 sizable_true_gain_n=len(sizable),
                 sizable_true_gain_sign_agreement=float(np.mean(np.sign(sizable[:, 0]) == np.sign(sizable[:, 1]))),
                 frames=cases)
    (output / 'four_cases.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('t23', 'images', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    build(args.t23, args.images.resolve(), args.output)
