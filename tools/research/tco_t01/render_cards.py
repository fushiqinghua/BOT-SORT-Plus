"""Compose TCO review cards from verified RGB frames and offline geometry."""

from math import ceil, floor, isfinite

from PIL import Image, ImageDraw, ImageFont


BACKGROUND = "#141923"
PANEL = "#202835"
IMAGE_BACKGROUND = "#0e131a"
TEXT = "#f1f4f8"
MUTED = "#b4c1ce"
RAW = "#ff6476"
CORRECTION = "#52e4fa"
TRUTH = "#95ee84"
OTHER = "#f9bd61"
HISTORY = "#a9acff"
RESAMPLING = getattr(Image, "Resampling", Image)
NEAREST = RESAMPLING.NEAREST
LANCZOS = RESAMPLING.LANCZOS


def _font(size):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


FONT_TITLE = _font(25)
FONT_LABEL = _font(17)
FONT_BODY = _font(14)
FONT_SMALL = _font(12)


def _box(value):
    if value is None or len(value) != 4:
        return None
    box = tuple(float(item) for item in value)
    if not all(map(isfinite, box)) or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _region(size, box, padding, minimum):
    width, height = size
    if box is None:
        return (0, 0, width, height)
    left, top, right, bottom = box
    span_x = max(minimum, (right - left) * padding)
    span_y = max(minimum, (bottom - top) * padding)
    center_x = min(width, max(0, (left + right) / 2))
    center_y = min(height, max(0, (top + bottom) / 2))
    return (
        max(0, floor(center_x - span_x / 2)),
        max(0, floor(center_y - span_y / 2)),
        min(width, ceil(center_x + span_x / 2)),
        min(height, ceil(center_y + span_y / 2)),
    )


def _place(canvas, source, destination, source_box=None, *, pixelated):
    left, top, right, bottom = destination
    bounds = source_box or (0, 0, *source.size)
    region = source.crop(bounds) if source_box else source
    factor = min((right - left) / region.width, (bottom - top) / region.height)
    size = (max(1, round(region.width * factor)), max(1, round(region.height * factor)))
    position = (left + (right - left - size[0]) // 2, top + (bottom - top - size[1]) // 2)
    tile = region.resize(size, NEAREST if pixelated else LANCZOS)
    canvas.paste(tile, position)
    return (bounds, position, size)


def _outline(canvas, transform, box, color, *, mode="solid"):
    if box is None:
        return
    (left, top, right, bottom), (origin_x, origin_y), (width, height) = transform
    if box[2] <= left or box[3] <= top or box[0] >= right or box[1] >= bottom:
        return
    scale_x, scale_y = width / (right - left), height / (bottom - top)
    x1 = origin_x + min(width - 1, round((max(box[0], left) - left) * scale_x))
    y1 = origin_y + min(height - 1, round((max(box[1], top) - top) * scale_y))
    x2 = origin_x + min(width - 1, round((min(box[2], right) - left) * scale_x))
    y2 = origin_y + min(height - 1, round((min(box[3], bottom) - top) * scale_y))
    ink = ImageDraw.Draw(canvas)
    if mode == "solid":
        ink.rectangle((x1, y1, x2, y2), outline=color, width=2)
        return
    dash, gap = (10, 5) if mode == "dash" else (3, 5)
    for x in range(x1, x2 + 1, dash + gap):
        end = min(x2, x + dash)
        ink.line((x, y1, end, y1), fill=color, width=2)
        ink.line((x, y2, end, y2), fill=color, width=2)
    for y in range(y1, y2 + 1, dash + gap):
        end = min(y2, y + dash)
        ink.line((x1, y, x1, end), fill=color, width=2)
        ink.line((x2, y, x2, end), fill=color, width=2)


def _panel(canvas, bounds, label, note=None, *, compact=False):
    x, y, right, bottom = bounds
    ink = ImageDraw.Draw(canvas)
    ink.rectangle((x, y, right - 1, bottom - 1), fill=PANEL)
    ink.text((x + 9, y + 6), label, fill=TEXT, font=FONT_SMALL if compact else FONT_LABEL)
    image_top = y + (25 if compact else 33)
    image_bottom = bottom - (24 if note else 7)
    if note:
        ink.text((x + 9, bottom - 20), note, fill=MUTED, font=FONT_SMALL)
    image_bounds = (x + 6, image_top, right - 6, image_bottom)
    ink.rectangle((image_bounds[0], image_top, image_bounds[2] - 1, image_bottom - 1),
                  fill=IMAGE_BACKGROUND)
    return image_bounds


def _caption(canvas, bounds, value):
    left, top, _, _ = bounds
    ImageDraw.Draw(canvas).text((left + 9, top + 9), value, fill=MUTED, font=FONT_SMALL)


def _history_tile(canvas, bounds, observation, source, current_frame, *, compact=False):
    if observation is None:
        area = _panel(canvas, bounds, "History", compact=compact)
        _caption(canvas, area, "No recorded earlier observation")
        return
    frame, stage = observation["frame"], observation.get("source_stage", "unspecified")
    label = f"History f{frame} (-{current_frame - frame})"
    score = observation.get("raw_score")
    note = f"{stage}, det {observation.get('det_index', '?')}"
    if score is not None and not compact:
        note += f", score {float(score):.3f}"
    area = _panel(canvas, bounds, label, note, compact=compact)
    raw = _box(observation.get("raw_xyxy"))
    region = _region(source.size, raw, padding=2.5, minimum=24)
    transform = _place(canvas, source, area, region, pixelated=True)
    _outline(canvas, transform, raw, HISTORY, mode="dash")


def _expanded_region(size, base_region, boxes):
    """Ensure references can be seen even when they fall outside the blind crop."""
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return base_region
    left, top, right, bottom = base_region
    if all(left <= b[0] and top <= b[1] and b[2] <= right and b[3] <= bottom for b in boxes):
        return base_region
    return (
        min(left, max(0, floor(min(b[0] for b in boxes) - 3))),
        min(top, max(0, floor(min(b[1] for b in boxes) - 3))),
        max(right, min(size[0], ceil(max(b[2] for b in boxes) + 3))),
        max(bottom, min(size[1], ceil(max(b[3] for b in boxes) + 3))),
    )


def _coordinates(box):
    if box is None:
        return "unavailable"
    return "(" + ", ".join(f"{number:.1f}" for number in box) + ")"


def _score(value):
    return "n/a" if value is None else f"{float(value):.3f}"


def _legend(image, episode, edge, raw, correction, anchor, nearby, detection_scope):
    ink = ImageDraw.Draw(image)
    rows = [
        (RAW, f"{detection_scope}   {_coordinates(raw)}   IoU {_score(edge.get('raw_iou', episode.get('raw_iou')))}"),
        (CORRECTION, f"Offline bounded correction   {_coordinates(correction)}"
         f"   IoU {_score(edge.get('feasible_iou', episode.get('feasible_iou')))}"
         if correction is not None else
         "Offline bounded correction unavailable for this track/detection pair"),
        (TRUTH, f"Official anchor GT {episode.get('gt_id')}   {_coordinates(anchor)}"
         if anchor is not None else "Official current anchor GT unavailable"),
    ]
    if nearby:
        extra = f" and {len(nearby) - 12} more" if len(nearby) > 12 else ""
        rows.append((OTHER, "Other official GT in view   "
                     + ", ".join(str(item["gt_id"]) for item in nearby[:12]) + extra))
    for index, (color, line) in enumerate(rows):
        y = 724 + 22 * index
        ink.rectangle((23, y + 4, 31, y + 12), fill=color)
        ink.text((42, y), line, fill=TEXT, font=FONT_BODY)
    ink.text((23, 821), "Correction is an offline geometric candidate derived with GT. "
             "Image evidence is assessed using the separate blind view.", fill=MUTED, font=FONT_SMALL)


def render_case(case, load_image):
    """Return annotated, blind, and compact blind views as RGB Pillow images."""
    sample, episode = case["sample"], case["episode"]
    frame, track, detection = case["frame"], case["track"], case.get("detection")
    edge = case.get("edge_label") or {}
    current = load_image(frame)
    history = sorted(case.get("history", ()), key=lambda item: item["frame"])
    history_sources = [(observation, load_image(observation)) for observation in history]
    if len(history_sources) > 3:
        raise ValueError("The TCO history contract permits at most three observations")
    history_sources += [(None, None)] * (3 - len(history_sources))

    raw = _box(detection.get("raw_xyxy")) if detection else None
    correction = _box(edge.get("feasible_xyxy"))
    anchor = _box(episode.get("gt_xyxy"))
    region = _region(current.size, raw or _box(track.get("pred_xyxy")), padding=3.0, minimum=32)
    annotated_region = _expanded_region(current.size, region, (raw, correction, anchor))
    nearby = [
        item for item in case.get("nearby_gt", ())
        if item["gt_id"] != episode.get("gt_id") and _box(item["xyxy"]) is not None
        and item["xyxy"][2] > annotated_region[0] and item["xyxy"][3] > annotated_region[1]
        and item["xyxy"][0] < annotated_region[2] and item["xyxy"][1] < annotated_region[3]
    ]
    current_frame = int(frame["frame"])
    track_id = track["track_id"]
    assigned_det = episode.get("assigned_det_index")
    if detection is None:
        detection_scope = "No assigned low detection"
    elif detection["det_index"] == assigned_det:
        detection_scope = f"Assigned low det {assigned_det}"
    else:
        detection_scope = f"Unassigned low candidate det {detection['det_index']}"
    raw_note = detection_scope
    if detection is not None:
        raw_note += f", score {_score(detection.get('raw_score'))}"
    crop_note = f"Source ({region[0]}, {region[1]})  {region[2] - region[0]} x {region[3] - region[1]} px"
    scope = "Focus sequence supplement" if case.get("scope") == "focus" else "Original review sample"
    identity = f"{sample['case_id']}  |  {sample['sequence']}  |  frame {current_frame}  |  track {track_id}"

    views = {}
    for annotated in (False, True):
        view = "annotated" if annotated else "blind"
        image = Image.new("RGB", (1350, 846 if annotated else 755), BACKGROUND)
        ink = ImageDraw.Draw(image)
        ink.text((18, 14), identity, fill=TEXT, font=FONT_TITLE)
        ink.text((20, 49), f"{sample['method']}  /  {scope}  /  {view.capitalize()} view",
                 fill=MUTED, font=FONT_BODY)

        area = _panel(image, (18, 80, 620, 460), "Current RGB frame",
                      f"Source {current.width} x {current.height} px")
        _place(image, current, area, pixelated=False)
        area = _panel(image, (632, 80, 977, 460), "Current crop, no boxes", crop_note)
        _place(image, current, area, region, pixelated=True)
        geometry_note = "View expanded to contain reference geometry" if annotated_region != region else raw_note
        area = _panel(image, (989, 80, 1332, 460),
                      "Current geometry comparison" if annotated else "Original low detection",
                      geometry_note if annotated else raw_note)
        transform = _place(image, current, area, annotated_region if annotated else region, pixelated=True)
        if annotated:
            for item in nearby:
                _outline(image, transform, _box(item["xyxy"]), OTHER, mode="dash")
        _outline(image, transform, raw, RAW)
        if annotated:
            _outline(image, transform, anchor, TRUTH, mode="dots")
            _outline(image, transform, correction, CORRECTION, mode="dash")

        for index, (observation, source) in enumerate(history_sources):
            x = 18 + index * 442
            _history_tile(image, (x, 472, x + 430, 711), observation, source, current_frame)
        if annotated:
            _legend(image, episode, edge, raw, correction, anchor, nearby, detection_scope)
        else:
            ink.text((23, 724), "History shows only accepted observations before the current frame. "
                     "The purple outlines mark the recorded historical observations.",
                     fill=MUTED, font=FONT_SMALL)
        views[view] = image

    review = Image.new("RGB", (960, 202), BACKGROUND)
    ImageDraw.Draw(review).text((9, 4), identity, fill=TEXT, font=FONT_BODY)
    area = _panel(review, (7, 25, 191, 197), "Current pixels", compact=True)
    _place(review, current, area, region, pixelated=True)
    area = _panel(review, (197, 25, 381, 197), "Current low detection",
                  "No assigned low detection" if detection is None else f"det {detection['det_index']}",
                  compact=True)
    transform = _place(review, current, area, region, pixelated=True)
    _outline(review, transform, raw, RAW)
    for index, (observation, source) in enumerate(history_sources):
        x = 387 + index * 190
        _history_tile(review, (x, 25, x + 184, 197), observation, source, current_frame, compact=True)
    views["review"] = review
    return views
