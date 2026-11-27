"""Prepare blinded review strips for known-anchor training proposals below IoU 0.50."""

import argparse
from collections import defaultdict, deque
import csv
import hashlib
import json
from math import ceil, floor
from pathlib import Path
import random

from build_corpus import KEYS, PARTITIONS, events, load_image_function, read, sha256


def dataset_candidates(folder, seed, limit):
    by_group = defaultdict(list)
    for row, label in events(folder):
        # The previous CLEAR link makes this a reviewable hypothesis only. No
        # case becomes training supervision without an independent blind vote.
        if (row['assigned'] and row['online_eligible'] and label['support_label'] == 'ignore'
                and label['same_gt_clear_output'] and label['finite_candidate_crosses_050']
                and label['gt_id'] is not None and 0 < label['raw_iou'] < .50):
            by_group[row['source_video']].append((row, label))
    rng = random.Random(seed)
    queues = []
    for group in sorted(by_group):
        values = by_group[group]
        rng.shuffle(values)
        queues.append(deque(values))
    rng.shuffle(queues)
    selected = []
    while queues and len(selected) < limit:
        selected.extend(queue.popleft() for queue in queues[:limit - len(selected)])
        queues = [q for q in queues if q]
    selected.sort(key=lambda case: (case[0]['sequence'], case[0]['frame'], case[0]['track_id']))
    return selected, sum(len(v) for v in by_group.values())


def paste_crop(page, image, raw, x, y, *, outline=True, w=191, h=108):
    from PIL import Image, ImageDraw

    x1, y1, x2, y2 = raw
    dx, dy = max(20, 2.5 * (x2 - x1)), max(20, 2.5 * (y2 - y1))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = max(0, floor(cx - dx/2)), max(0, floor(cy - dy/2))
    right, bottom = min(image.shape[1], ceil(cx + dx/2)), min(image.shape[0], ceil(cy + dy/2))
    tile = Image.fromarray(image[top:bottom, left:right, ::-1].copy())
    scale = min(w / tile.width, h / tile.height)
    output = tile.resize((round(scale*tile.width), round(scale*tile.height)), Image.Resampling.NEAREST)
    offset = x + (w - output.width)//2, y + (h - output.height)//2
    page.paste(output, offset)
    if outline:
        draw = ImageDraw.Draw(page)
        draw.rectangle([(offset[0] + (x1-left)*scale, offset[1] + (y1-top)*scale),
                        (offset[0] + (x2-left)*scale, offset[1] + (y2-top)*scale)],
                       outline='#ff747d', width=1)


def render(output, part, selected, load_image):
    from PIL import Image, ImageDraw

    index = []
    size = 7
    for offset in range(0, len(selected), size):
        rows = selected[offset:offset+size]
        page = Image.new('RGB', (1015, len(rows)*150), '#202733')
        draw = ImageDraw.Draw(page)
        for i, (row, label) in enumerate(rows):
            key = {name: row[name] for name in KEYS}
            case = part[:3] + '_' + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:10]
            line = i*150
            draw.text((8, line + 4), f'{case}  {row["sequence"]}  frame {row["frame"]}  track {row["track_id"]}', fill='white')
            images = [(row, 'Current pixels'), (row, 'Current low detection')]
            images += [(obs, f'History f{obs["frame"]}') for obs in row['history']]
            for j, (obs, title) in enumerate(images):
                x = 4 + 202*j
                draw.text((x, line + 21), title, fill='#dddddd')
                image = load_image(obs)
                paste_crop(page, image, obs['raw_xyxy'], x, line+36, outline=j != 0)
            index.append(dict(case_id=case, partition=part, **key, source_video=row['source_video'],
                              gt_id=label['gt_id'], raw_iou=label['raw_iou'], feasible_iou=label['feasible_iou']))
        page.save(output / f'{part}_{offset//size+1:02}.png', compress_level=3)
    return index


def build(args):
    args.output.mkdir(parents=True, exist_ok=False)
    loader = load_image_function(args.images.resolve())
    index, summary = [], {}
    for part in PARTITIONS:
        cases, total = dataset_candidates(args.labels / part, 41, args.limit)
        summary[part] = dict(reviewable=total, shown=len(cases), groups=len({a['source_video'] for a,b in cases}))
        index.extend(render(args.output, part, cases, loader))
    with (args.output / 'index.csv').open('x', encoding='utf-8', newline='') as stream:
        fields = ['case_id','partition', *KEYS, 'source_video', 'gt_id', 'raw_iou', 'feasible_iou']
        out = csv.DictWriter(stream, fieldnames=fields)
        out.writeheader()
        out.writerows(index)
    (args.output / 'manifest.json').write_text(json.dumps(dict(seed=41, per_partition_budget=args.limit,
        partitions=summary, labels_sha256=sha256(args.labels / 'training_labels.json'),
        index_sha256=sha256(args.output / 'index.csv')), indent=2) + '\n', encoding='utf-8')
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('labels', 'images', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--limit', type=int, default=63)
    build(parser.parse_args())
