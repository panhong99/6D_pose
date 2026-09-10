"""Common overlay and lossless binary mask exports, kept outside infer timing."""
import json
import textwrap
from pathlib import Path

import cv2
import numpy as np


def write_rgb(path, rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f'Could not write image: {path}')


def write_mask(path, mask):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8) * 255):
        raise OSError(f'Could not write mask: {path}')


def draw_result(rgb, result, title=''):
    image = rgb.copy()
    mask = result.get('mask')
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != rgb.shape[:2]:
            raise ValueError('Mask must have the original image height and width')
        image[mask] = np.round(0.60 * image[mask] + 0.40 * np.array([40, 220, 90])).astype(np.uint8)
    box = result.get('bbox')
    if box is not None:
        x1, y1, x2, y2 = np.round(box).astype(int)
        cv2.rectangle(image, (x1, y1), (min(x2, image.shape[1]-1), min(y2, image.shape[0]-1)),
                      (30, 235, 100), 2)
    # Header is outside the source image: no pixels of a small cube are obscured.
    width = max(400, image.shape[1])
    status = 'ERROR' if result.get('error') else ('SUCCESS' if result['success'] else 'MISS')
    latency = result.get('latency_ms')
    latency_text = f'{latency:.1f} ms' if latency is not None else 'n/a ms'
    lines = textwrap.wrap(title, width=max(25, int((width - 20) / 8))) or ['Detection']
    lines += [f"{status} | conf={result['confidence']:.3f} | {latency_text}"]
    if result.get('error'):
        lines += textwrap.wrap(str(result['error']), width=max(25, int((width-20)/8)))[:2]
    header = 14 + 22 * len(lines)
    canvas = np.full((header + image.shape[0], width, 3), 28, dtype=np.uint8)
    canvas[header:, :image.shape[1]] = image
    for index, line in enumerate(lines):
        cv2.putText(canvas, line, (10, 23 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def side_by_side(images):
    height = max(image.shape[0] for image in images)
    canvas = np.full((height, sum(image.shape[1] for image in images), 3), 28, dtype=np.uint8)
    offset = 0
    for image in images:
        canvas[:image.shape[0], offset:offset+image.shape[1]] = image
        offset += image.shape[1]
    return canvas


def save_single_result(rgb, result, output_dir, stem, title=''):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / f'{stem}_overlay.png'
    write_rgb(overlay_path, draw_result(rgb, result, title))
    record = {key: value for key, value in result.items() if key != 'mask'}
    record.update(has_bbox=result['bbox'] is not None, has_mask=result['mask'] is not None,
                  overlay_path=overlay_path.name, mask_path=None)
    if result['mask'] is not None:
        mask_path = output_dir / f'{stem}_mask.png'
        write_mask(mask_path, result['mask'])
        record['mask_path'] = mask_path.name
    with (output_dir / f'{stem}.json').open('w', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return record
