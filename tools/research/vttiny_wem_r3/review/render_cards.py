"""Render paired native-pixel VT-Tiny-MOT evidence for W R3 review."""

from functools import lru_cache
from math import ceil, floor, isfinite
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BACKGROUND = "#101720"
PANEL = "#1d2734"
EMPTY = "#111720"
TEXT = "#f4f7fa"
MUTED = "#b5c3d2"
WEAK = "#ff6378"
OBSERVATION = "#b9acff"
TRUTH = "#94f08c"
IGNORED = "#a8aeb4"
PREDICTION = "#5fdbed"
TRACK = "#f6bd65"
CROP = "#e3edf4"
RESAMPLE = getattr(Image, "Resampling", Image)


@lru_cache(maxsize=None)
def _font(size):
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


@lru_cache(maxsize=24)
def _load(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def _label(text, fallback="-"):
    if text is None or text == "":
        return fallback
    return str(text).encode("ascii", "replace").decode("ascii")


def _text(draw, location, value, size=15, color=TEXT):
    draw.text(location, _label(value), fill=color, font=_font(size))


def _box(raw):
    if raw is None or len(raw) != 4:
        return None
    values = tuple(map(float, raw))
    if all(map(isfinite, values)) and values[0] < values[2] and values[1] < values[3]:
        return values
    return None


def _crop(size, boxes):
    valid = [item for raw in boxes if (item := _box(raw)) is not None]
    width, height = size
    if not valid:
        return (0, 0, width, height)
    x0 = min(box[0] for box in valid)
    y0 = min(box[1] for box in valid)
    x1 = max(box[2] for box in valid)
    y1 = max(box[3] for box in valid)
    mx, my = min(max((x0 + x1) / 2, 0), width), min(max((y0 + y1) / 2, 0), height)
    r = max(18, (x1 - x0) * .82, (y1 - y0) * .82)
    return max(0, floor(mx - r)), max(0, floor(my - r)), min(width, ceil(mx + r)), min(height, ceil(my + r))


def _map_box(box, from_size, to_size):
    sx = to_size[0] / from_size[0]
    sy = to_size[1] / from_size[1]
    return box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy


def _map_crop(region, rgb_size, panel_size):
    x0, y0, x1, y1 = _map_box(region, rgb_size, panel_size)
    return floor(x0), floor(y0), ceil(x1), ceil(y1)


def _tile(canvas, bounds, title, subtitle=None):
    left, top, right, bottom = bounds
    ink = ImageDraw.Draw(canvas)
    ink.rectangle((left, top, right - 1, bottom - 1), fill=PANEL)
    _text(ink, (left + 9, top + 7), title, 16)
    view = (left + 7, top + 32, right - 7, bottom - (25 if subtitle else 7))
    ink.rectangle((view[0], view[1], view[2] - 1, view[3] - 1), fill=EMPTY)
    if subtitle:
        _text(ink, (left + 9, bottom - 20), subtitle, 12, MUTED)
    return view


def _place(canvas, source, bounds, region=None):
    rect = region or (0, 0, *source.size)
    pixels = source.crop(rect) if region else source
    left, top, right, bottom = bounds
    scale = min((right - left) / pixels.width, (bottom - top) / pixels.height)
    size = max(1, round(pixels.width * scale)), max(1, round(pixels.height * scale))
    at = left + (right - left - size[0]) // 2, top + (bottom - top - size[1]) // 2
    canvas.paste(pixels.resize(size, RESAMPLE.NEAREST if region else RESAMPLE.LANCZOS), at)
    return rect, at, size


def _stroke(ink, start, end, color, dashed):
    if not dashed:
        ink.line((*start, *end), fill=color, width=2)
        return
    x0, y0 = start
    x1, y1 = end
    if y0 == y1:
        for x in range(x0, x1 + 1, 10):
            ink.line((x, y0, min(x1, x + 5), y0), fill=color, width=2)
    else:
        for y in range(y0, y1 + 1, 10):
            ink.line((x0, y, x0, min(y1, y + 5)), fill=color, width=2)


def _projected_box(box, transform):
    (left, top, right, bottom), (x, y), (width, height) = transform
    a, b, c, d = box
    if c <= left or a >= right or d <= top or b >= bottom:
        return None
    clipped = (
        x + min(width - 1, round((max(left, a) - left) * width / (right - left))),
        y + min(height - 1, round((max(top, b) - top) * height / (bottom - top))),
        x + min(width - 1, round((min(right, c) - left) * width / (right - left))),
        y + min(height - 1, round((min(bottom, d) - top) * height / (bottom - top))),
    )
    return clipped, (a >= left, b >= top, c <= right, d <= bottom)


def _overlay(canvas, transform, layers, labels=True):
    ink = ImageDraw.Draw(canvas)
    tags = []
    for box, label, color, dashed in layers:
        shown = _projected_box(box, transform)
        if shown is None:
            continue
        (x0, y0, x1, y1), (show_left, show_top, show_right, show_bottom) = shown
        if show_top:
            _stroke(ink, (x0, y0), (x1, y0), color, dashed)
        if show_bottom:
            _stroke(ink, (x0, y1), (x1, y1), color, dashed)
        if show_left:
            _stroke(ink, (x0, y0), (x0, y1), color, dashed)
        if show_right:
            _stroke(ink, (x1, y0), (x1, y1), color, dashed)
        if labels and label and (show_left or show_top or show_right or show_bottom):
            tags.append((label, color, (x0, y0, x1, y1)))
    if not tags:
        return

    _, (origin_x, origin_y), (width, height) = transform
    placed = []
    font = _font(12)
    for label, color, (x0, y0, x1, y1) in reversed(tags):
        left, top, right, bottom = ink.textbbox((0, 0), label, font=font)
        w, h = right - left + 6, bottom - top + 4
        if w > width or h > height:
            continue
        candidates = [(x0, y0 - h), (x0, y1 + 1), (x1 + 1, y0), (x0 - w, y0), (x0, y0 + 2)]
        for x, y in candidates:
            a = min(max(x, origin_x), origin_x + width - w)
            b = min(max(y, origin_y), origin_y + height - h)
            c, d = a + w, b + h
            if any(a < z[2] and c > z[0] and b < z[3] and d > z[1] for z in placed):
                continue
            ink.rectangle((a, b, c - 1, d - 1), fill=EMPTY)
            ink.text((a + 3 - left, b + 2 - top), label, font=font, fill=color)
            placed.append((a, b, c, d))
            break


def _layers(record, modality, rgb_size, image_size, weak, predicted, raw_index, selected_gt, history):
    items = []
    truth_key = "truth" if modality == "RGB" else "ir_truth"
    for row in record.get(truth_key, ()) or ():
        box = _box(row.get("box_xyxy"))
        if box is None:
            continue
        ignored = row.get("ignored", False)
        truth = row.get("gt_id")
        highlighted = selected_gt is not None and str(truth) == str(selected_gt)
        prefix = "IGN" if ignored else "GT"
        items.append((box, f"{prefix} {_label(truth)}", IGNORED if ignored else TRUTH, not highlighted or ignored))
    for row in record.get("tracks", ()) or ():
        box = _box(row.get("box_xyxy", row.get("xyxy")))
        if box:
            items.append((_map_box(box, rgb_size, image_size), f"T {_label(row.get('track_id'))}", TRACK, True))
    if predicted:
        items.append((_map_box(predicted, rgb_size, image_size), "Pred", PREDICTION, True))
    if weak:
        name = "Obs" if history else "W"
        items.append((_map_box(weak, rgb_size, image_size), f"{name} {_label(raw_index)}",
                      OBSERVATION if history else WEAK, False))
    return items


def _picture(canvas, bounds, title, record, modality, rgb_size, region, weak, predicted, index,
             selected_gt, *, full=False, history=False):
    path = record.get("rgb_path" if modality == "RGB" else "ir_path")
    if path is None or path == "":
        area = _tile(canvas, bounds, title)
        _text(ImageDraw.Draw(canvas), (area[0] + 9, area[1] + 9), f"No paired {modality} image", 13, MUTED)
        return
    image = _load(str(path))
    crop = _map_crop(region, rgb_size, image.size)
    caption = f"{image.width} x {image.height} native px" if full else (
        f"crop {crop[0]},{crop[1]} | {crop[2] - crop[0]} x {crop[3] - crop[1]} native px"
    )
    area = _tile(canvas, bounds, title, caption)
    transform = _place(canvas, image, area, None if full else crop)
    if full:
        _overlay(canvas, transform, [(crop, "", CROP, True)], labels=False)
        return
    layers = _layers(record, modality, rgb_size, image.size, weak, predicted, index, selected_gt, history)
    _overlay(canvas, transform, layers)


def _target_gt(rows, target):
    if target is None:
        return None
    return next((row.get("box_xyxy") for row in rows or () if str(row.get("gt_id")) == str(target)), None)


def _slots(count, y0, y1):
    # Four wide panels preserve complete images and six narrower ones preserve
    # the cropped historical RGB and IR views on a single card.
    x0, x1, gap = 18, 1422, 10
    width = (x1 - x0 - (count - 1) * gap) / count
    return [(round(x0 + k * (width + gap)), y0, round(x0 + k * (width + gap) + width), y1)
            for k in range(count)]


def render_card(case: dict, output: Path) -> None:
    """Write one RGB/IR card using original weak detections and separate official sensor labels."""
    current = case["current"]
    if isinstance(current, list):
        current, = current
    frame = int(current["frame"])
    history = sorted(case.get("history", ()), key=lambda row: int(row["frame"]))[-3:]
    if any(int(old["frame"]) >= frame for old in history):
        raise ValueError(f"Historical observation is not causal in {case['case_id']}")
    rgb_size = _load(str(current["rgb_path"])).size
    weak = _box(case.get("raw_box_xyxy"))
    predicted = _box(case.get("predicted_box"))
    target = case.get("gt_id")
    crop = _crop(rgb_size, [weak, predicted, _target_gt(current.get("truth"), target)])
    raw_index = case.get("det_index", case.get("raw_det_index"))

    canvas = Image.new("RGB", (1440, 824), BACKGROUND)
    ink = ImageDraw.Draw(canvas)
    seq = case.get("sequence") or case.get("seq")
    _text(ink, (18, 13), f"{case['case_id']}   |   {_label(seq)}   |   Frame {frame}", 25)
    _text(ink, (18, 47), f"Review pool: {_label(case.get('review_pool'))}   |   Partition: {_label(case.get('partition'))}"
          f"   |   Source video: {_label(case.get('source_video'))}   |   {_label(case.get('method'))}", 15, MUTED)
    score = case.get("raw_score")
    score_text = "-" if score is None or score == "" else f"{float(score):.4f}"
    _text(ink, (18, 68), f"Track {_label(case.get('track_id'))}   |   Weak detection row {_label(raw_index)}"
          f"   |   Original score {score_text}   |   Anchor GT {_label(target)}", 15, MUTED)

    panels = _slots(4, 96, 452)
    captions = (("RGB", "Current RGB full", True), ("IR", "Current IR full", True),
                ("RGB", "Current RGB crop", False), ("IR", "Current IR crop", False))
    for panel, (modality, title, full) in zip(panels, captions):
        sensor_gt = target if modality == "RGB" else case.get("ir_gt_id")
        _picture(canvas, panel, title, current, modality, rgb_size, crop, weak, predicted, raw_index, sensor_gt,
                 full=full)

    history_panels = _slots(6, 499, 752)
    for location in range(3):
        row = history[location] if location < len(history) else None
        if row is None:
            for offset, modality in enumerate(("RGB", "IR")):
                area = _tile(canvas, history_panels[location * 2 + offset], f"History {modality}")
                _text(ink, (area[0] + 8, area[1] + 8), "No recorded observation", 13, MUTED)
            continue
        prior = int(row["frame"])
        old_box = _box(row.get("raw_history_box"))
        old_size = _load(str(row["rgb_path"])).size
        old_crop = _crop(old_size, [old_box, _target_gt(row.get("truth"), target)])
        meta = f"f{prior} (-{frame - prior}) | {_label(row.get('stage'))} | support {_label(row.get('support'))}"
        x = history_panels[location * 2][0]
        _text(ink, (x, 474), meta, 14, MUTED)
        for offset, modality in enumerate(("RGB", "IR")):
            sensor_gt = target if modality == "RGB" else case.get("ir_gt_id")
            _picture(canvas, history_panels[location * 2 + offset], f"History {modality}", row, modality,
                     old_size, old_crop, old_box, None, row.get("det_index"), sensor_gt, history=True)

    entries = ((WEAK, "W: original weak detection"), (OBSERVATION, "Obs: observed history"),
               (TRUTH, "GT: official ID per sensor"), (IGNORED, "IGN: ignored GT"),
               (PREDICTION, "Pred: predicted box"), (TRACK, "T: tracker ID"), (CROP, "Full-frame crop"))
    x = 18
    for color, label in entries:
        ink.rectangle((x, 770, x + 10, 778), fill=color)
        _text(ink, (x + 16, 765), label, 13)
        x += round(ink.textlength(label, font=_font(13))) + 31
    _text(ink, (18, 795), "RGB-coordinate overlays on IR are visual references. Sensor offset is possible. "
          "IR GT uses its own official annotations. Crops use nearest-neighbor zoom.", 13, MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
