"""Render individual cases and compact visual comparison sheets for W R3."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from PIL import Image, ImageDraw

from render_cards import (
    BACKGROUND, MUTED, _box, _crop, _load, _map_crop, _picture, _place,
    _target_gt, _text, _tile, render_card,
)


HERE = Path(__file__).resolve().parent
PANELS = 8
PANEL_WIDTH = 184
PANEL_HEIGHT = 156
ROW_HEIGHT = 187
PER_SHEET = 5


def image_path(record, mode):
    return record["rgb_path" if mode == "RGB" else "ir_path"]


def draw_context(canvas, bounds, mode, record, region, weak, prior, case, use_overlays, history):
    image = _load(str(image_path(record, mode)))
    rgb_size = _load(str(record["rgb_path"])).size
    tag = "f" + str(record["frame"])
    if use_overlays:
        target = case.get("gt_id") if mode == "RGB" else None
        title = f"{tag} {mode} with boxes"
        index = record.get("det_index") if history else case["det_index"]
        _picture(canvas, bounds, title, record, mode, rgb_size, region, weak, prior, index, target, history=history)
    else:
        area = _tile(canvas, bounds, f"{tag} {mode} pixels")
        _place(canvas, image, area, _map_crop(region, rgb_size, image.size))


def sheet(cases, output):
    width = PANEL_WIDTH * PANELS + 14
    canvas = Image.new("RGB", (width, ROW_HEIGHT * len(cases) + 8), BACKGROUND)
    ink = ImageDraw.Draw(canvas)
    for offset, case in enumerate(cases):
        y = offset * ROW_HEIGHT + 5
        current = case["current"]
        history = case["history"][-2:]
        weak = _box(case["raw_box_xyxy"])
        pred = _box(case["predicted_box"])
        note = (f"{case['case_id']} | {case['sequence']} f{case['frame']} T{case['track_id']}"
                f" | weak row {case['det_index']} | best RGB IoU {case['best_gt_iou']:.3f}"
                f" | nearby RGB GT {case['best_candidate_gt_id']} | prior anchor {case['gt_id']}")
        _text(ink, (7, y), note, 17)

        def box(pos):
            x = 7 + PANEL_WIDTH * pos
            return x, y + 25, x + PANEL_WIDTH - 5, y + 25 + PANEL_HEIGHT

        rgb_size = _load(str(current["rgb_path"])).size
        region = _crop(rgb_size, [weak, pred, _target_gt(current["truth"], case.get("gt_id"))])
        for pos, (mode, overlay) in enumerate((('RGB', False), ('RGB', True), ('IR', False), ('IR', True))):
            draw_context(canvas, box(pos), mode, current, region, weak, pred, case, overlay, False)
        for pos, item in enumerate([None] * (2 - len(history)) + history):
            for mod_index, mode in enumerate(("RGB", "IR")):
                bounds = box(4 + pos * 2 + mod_index)
                if item is None:
                    area = _tile(canvas, bounds, f"History {mode}")
                    _text(ink, (area[0] + 7, area[1] + 8), "No observation", 13, MUTED)
                    continue
                observed = _box(item["raw_history_box"])
                earlier_size = _load(str(item["rgb_path"])).size
                old_region = _crop(earlier_size, [observed, _target_gt(item["truth"], case.get("gt_id"))])
                draw_context(canvas, bounds, mode, item, old_region, observed, None, case, True, True)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def read_assets(root):
    cases = json.loads((root / "card_inputs.json").read_bytes())["cases"]
    for case in cases:
        case["method"] = case["reference"]
        for frame in [case["current"], *case["history"]]:
            for key in ("rgb_path", "ir_path"):
                frame[key] = root / frame[key]
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--assets", type=Path, help="Private asset directory created by collect_frames.py")
    p.add_argument("--output", type=Path, help="Rendered card output directory")
    args = p.parse_args()
    stage = "smoke" if args.smoke else "full"
    cases = read_assets(args.assets or HERE / "assets" / stage)
    root = args.output or HERE / "rendered" / stage
    grouped = defaultdict(list)
    for case in cases:
        part, pool = case["partition"], case["review_pool"]
        grouped[part, pool].append(case)
        render_card(case, root / part / pool / f"{case['case_id']}.png")
    for (part, pool), group in grouped.items():
        for start in range(0, len(group), PER_SHEET):
            sheet(group[start:start + PER_SHEET], root / part / "sheets" / f"{pool}_{start // PER_SHEET + 1:02d}.png")
    print(f"Rendered {len(cases)} review cards to {root}")


if __name__ == "__main__":
    main()
