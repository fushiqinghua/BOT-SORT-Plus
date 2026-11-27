"""Package the frozen T1 sample and a targeted RGB image review."""

import argparse
from collections import Counter
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import cv2
from PIL import Image

from assemble_cases import assemble_cases
from render_cards import render_case


FOCUS = ('DJI_0327_3', 'DJI_0061_2')
REVIEW_GROUPS = (*FOCUS, 'others_1', 'others_2', 'others_3')


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def image_loader(dataset):
    @lru_cache(maxsize=12)
    def decode(path):
        pixels = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if pixels is None:
            raise ValueError(f'Cannot read source frame {path}')
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        image = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
        return list(pixels.shape), digest, image

    def load(record):
        reference = Path(record['frame_image'])
        path = (dataset / reference).resolve()
        if not path.is_relative_to(dataset):
            raise ValueError(f'Unexpected source frame path {path}')
        shape, digest, image = decode(path)
        if shape != record['image_shape'] or digest != record['image_sha256']:
            raise ValueError(f'Source frame disagrees with frozen capture {path}')
        return image

    return load


def review_metadata(case, group):
    sample = case['sample']
    episode = case['episode']
    label = case['edge_label'] or {}
    det = case['detection'] or {}
    history = sorted(case['history'], key=lambda observation: observation['frame'])
    case_id = sample['case_id']
    return dict(
        case_id=case_id, sample_scope=case['scope'], review_group=group,
        sequence=sample['sequence'], frame=episode['frame'], method=sample['method'],
        track_id=episode['track_id'], assigned_det_index=det.get('det_index'),
        raw_score=det.get('raw_score'), detector_class=det.get('raw_class'),
        primary_bucket=episode['primary_bucket'],
        history_frames=','.join(str(observation['frame']) for observation in history),
        history_stages=','.join(observation['source_stage'] for observation in history),
        history_xyxy=json.dumps([observation['raw_xyxy'] for observation in history]),
        history_image_sha256=','.join(observation['image_sha256'] for observation in history),
        current_source=case['frame']['frame_image'], current_image_sha256=case['frame']['image_sha256'],
        run_fingerprint=case['frame']['run_fingerprint'], gt_id=episode['gt_id'],
        gt_short_side_bin=episode['gt_short_side_bin'], anchor_present=episode['anchor_present'],
        anchor_covered=episode['anchor_covered'], reference_low_output_status=episode['reference_low_output_status'],
        official_geometric_miss=bool(episode['geometry_only_bounded_cross_050'] and not episode['anchor_covered']),
        identity_status_gt=episode['identity_status'], raw_iou=episode['raw_iou'],
        feasible_iou=episode['feasible_iou'], nearest_other_iou=episode['nearest_other_iou'],
        bounded_cross_050=episode['bounded_cross_050'], gt_exact_reachable=episode['gt_exact_reachable'],
        raw_xyxy=json.dumps(det.get('raw_xyxy')), offline_feasible_xyxy=json.dumps(label.get('feasible_xyxy')),
        gt_xyxy=json.dumps(episode['gt_xyxy']),
        original_inclusion_probability=sample.get('inclusion_probability') if case['scope'] == 'original' else None,
        annotated_card=f"cards/{case['scope']}/annotated/{case_id}.png",
        blind_card=f"cards/{case['scope']}/blind/{case_id}.png",
    )


def review_group(cases):
    groups = {key: [] for key in REVIEW_GROUPS}
    other_index = 0
    for case in cases:
        sequence = case['sample']['sequence']
        if sequence in FOCUS:
            group = sequence
        else:
            group = REVIEW_GROUPS[len(FOCUS) + other_index % 3]
            other_index += 1
        groups[group].append(case)
    for sequence in FOCUS:
        groups[sequence].sort(key=lambda c: (c['scope'] != 'focus', c['episode']['frame'], c['sample']['case_id']))
    return groups


def save_sheet(sheet_root, group, number, review_images):
    height = sum(p.height for p in review_images) + 8 * (len(review_images) - 1)
    sheet = Image.new('RGB', (max(p.width for p in review_images), height), '#273141')
    y = 0
    for image in review_images:
        sheet.paste(image, (0, y))
        y += image.height + 8
    sheet.save(sheet_root / f'{group}_{number:02}.png', compress_level=3)


def package(root, output):
    manifest = {'files': {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}}
    listing = root / 'files.sha256.json'
    listing.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    temp = output.with_suffix('.partial')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for key in [*manifest['files'], listing.name]:
            archive.write(root / key, key)
    temp.replace(output)
    return sha256(output)


def build(root, work, focus_count):
    work.mkdir(parents=True, exist_ok=True)
    output = work / 'bundle'
    output.mkdir(exist_ok=False)
    source_t1 = root / 'T1'
    settings = json.loads((root / 'offline_settings.json').read_bytes())
    load_image = image_loader(Path(settings['paths']['images_root']).resolve())
    cases = assemble_cases(root, focus_count)
    counts = Counter(case['scope'] for case in cases)
    if counts != {'original': 160, 'focus': 2 * focus_count}:
        raise ValueError(f'Unexpected sample counts {counts}')
    print('Case joins verified', dict(counts), flush=True)
    for scope in counts:
        for view in ('annotated', 'blind'):
            (output / 'cards' / scope / view).mkdir(parents=True)
    sheets = output / 'model_blind_sheets'
    sheets.mkdir()
    rows = []
    for group, selected in review_group(cases).items():
        sheet = []
        for index, case in enumerate(selected, 1):
            card = render_case(case, load_image)
            row = review_metadata(case, group)
            for view in ('annotated', 'blind'):
                card[view].save(output / row[f'{view}_card'], compress_level=3)
            sheet.append(card['review'])
            if index % 6 == 0 or index == len(selected):
                save_sheet(sheets, group, (index + 5) // 6, sheet)
                sheet.clear()
            rows.append(row)
        print(f'Rendered {group} {len(selected)}', flush=True)
    rows.sort(key=lambda row: (row['sample_scope'] != 'original', row['case_id']))
    original_samples = list(csv.DictReader((source_t1 / 'review_sample.csv').open(encoding='utf-8-sig')))
    if [r['case_id'] for r in rows if r['sample_scope'] == 'original'] != [r['case_id'] for r in original_samples]:
        raise ValueError('The rendered cards do not reproduce the original sample IDs and order')
    if len(set(r['case_id'] for r in rows)) != len(cases):
        raise ValueError('Duplicate card IDs')
    write_csv(output / 'case_index.csv', rows)
    original = output / 'source_t1'
    original.mkdir()
    for name in ('review_sample.csv', 'review_sample.json', 'offline_verification.json', 'T1_DECISION.md'):
        shutil.copyfile(source_t1 / name, original / name)
    details = dict(original_cases=counts['original'], focus_supplement_cases=counts['focus'],
                   views_per_case=2, blind_sheet_count=len(list(sheets.glob('*.png'))),
                   gt_uses='Offline witness and official CLEAR coverage labels',
                   corrected_boxes_are='GT-assisted feasible witnesses from T1, never runtime predictions',
                   native_history='Actual accepted detector measurements before current frame',
                   source_frame_verification='All decoded current and historical RGB source frames verified against T0 capture SHA-256',
                   source_sample_sha256=sha256(original / 'review_sample.csv'),
                   source_scripts_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
                   training_started=False)
    (output / 'build_verification.json').write_text(json.dumps(details, indent=2) + '\n', encoding='utf-8')
    digest = package(output, work / 'TCO_review_images.zip')
    (work / 'verified.json').write_text(json.dumps(dict(**details, archive_sha256=digest), indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('t01_root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--focus-per-sequence', type=int, default=12)
    args = parser.parse_args()
    build(args.t01_root, args.output, args.focus_per_sequence)
