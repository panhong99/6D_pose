"""Evaluate markerless Q3/Q4 wrist photos from the opposite-side view.

The 19 photos do not have hand-labelled tine keypoints.  This script therefore
reports geometric completeness and numbering acceptance, not independent
accuracy.  Physical quadrant names are preserved: a Q3 camera view evaluates
the Q4 target, and a Q4 camera view evaluates the Q3 target.
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch

import eval as E
import slots as S
from train import TineNet


def put(image, text, xy, colour=(255, 255, 255), scale=0.5):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                colour, 1, cv2.LINE_AA)


def aggregate(records, mode, view):
    selected = [record[mode] for record in records if record['view'] == view]
    return {
        'n': len(selected),
        'target': selected[0]['target'],
        'strict_answered': sum(item['strict_answer'] for item in selected),
        'total_found': sum(item['found'] for item in selected),
        'total_expected': sum(item['expected'] for item in selected),
        'mean_coverage': float(np.mean([item['coverage'] for item in selected])),
        'whole_rows': sum(item['whole_rows'] for item in selected),
        'agreeing_frames': sum(item['rows_agree'] for item in selected),
        'ambiguous_rows': sum(item['ambiguous_rows'] for item in selected),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--source', default='calibration/data/basket/wrist_look')
    parser.add_argument('--th', type=float, default=0.3)
    parser.add_argument('--tta', action='store_true')
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    E.TTA = args.tta
    net = TineNet(pretrained=False).cuda().eval()
    net.load_state_dict(torch.load(args.ckpt, map_location='cuda', weights_only=True))
    facts = S.basket_facts('tip')

    records = []
    tiles = []
    for photo in sorted(source.glob('20261026_*.raw.png')):
        match = re.search(r'_(Q[34])\.raw\.png$', photo.name)
        if not match:
            continue
        view = match.group(1)
        cross = 'Q3' if view == 'Q4' else 'Q4'
        metadata = json.loads(Path(str(photo).replace('.raw.png', '.json')).read_text())
        t_base_camera = (np.asarray(metadata['T_base_flange'])
                         @ np.asarray(metadata['T_flange_cam']))
        t_base_basket = np.asarray(metadata['T_base_basket'], float)
        image = cv2.imread(str(photo))
        predictions = E.predict(net, image, args.th)['tip']
        points = (S.to_rail_plane(predictions[:, :2], t_base_camera,
                                  t_base_basket, facts['rail_z'])
                  if len(predictions) else np.zeros((0, 3)))
        away = bool((t_base_basket[:3, 0]
                     @ np.r_[t_base_basket[:2, 3], 0.0]) >= 0)

        modes = []
        for mode, target in [('same', view), ('cross', cross)]:
            numbered, info = S.number_rows(points, facts, target, away)
            strict = len(info) == 2 and all(row['confident'] for row in info.values())
            expected = sum(facts['rows'][(target, row)]['n'] for row in (0, 1))
            found = sum(row['found'] for row in info.values())
            modes.append({
                'mode': mode,
                'target': target,
                'strict_answer': strict,
                'expected': expected,
                'found': found,
                'coverage': found / expected,
                'whole_rows': sum(bool(row['whole_row']) for row in info.values()),
                'rows_agree': (len(info) == 2
                               and all(bool(row.get('rows_agree'))
                                       for row in info.values())),
                'ambiguous_rows': sum(bool(row.get('ambiguous'))
                                      for row in info.values()),
                'rows': {
                    str(row_id): {
                        key: (float(value) if isinstance(value, (float, np.floating))
                              else value)
                        for key, value in row.items()
                    }
                    for row_id, row in info.items()
                },
                'numbers': {
                    str(row_id): [int(number) for number, _ in values]
                    for row_id, values in numbered.items()
                },
            })

        records.append({
            'photo': photo.name,
            'view': view,
            'tip_predictions': len(predictions),
            'same': modes[0],
            'cross': modes[1],
        })

        visual = image.copy()
        for x, y, _ in predictions:
            cv2.circle(visual, (round(x), round(y)), 4, (0, 0, 255),
                       -1, cv2.LINE_AA)
        numbered, _ = S.number_rows(points, facts, cross, away)
        strict = modes[1]['strict_answer']
        for values in numbered.values():
            for number, index in values:
                put(visual, str(number),
                    (round(predictions[index, 0]) + 5,
                     round(predictions[index, 1]) + 14),
                    (0, 220, 0) if strict else (0, 170, 255), 0.45)
        bar = np.zeros((52, visual.shape[1], 3), np.uint8)
        put(bar,
            f'VIEW {view} -> TARGET {cross} | '
            f'found {modes[1]["found"]}/{modes[1]["expected"]} | '
            f'whole {modes[1]["whole_rows"]}/2 | strict {strict}',
            (8, 31), (255, 255, 255), 0.62)
        tile = np.vstack([bar, visual])
        tiles.append(cv2.resize(tile, (640, 386), interpolation=cv2.INTER_AREA))
        cv2.imwrite(str(output / f'{photo.stem.replace(".raw", "")}_'
                                  f'view_{view}_target_{cross}.jpg'), tile)

    if not records:
        raise RuntimeError(f'no Q3/Q4 photos found under {source}')
    aggregates = {
        f'{view}_{mode}': aggregate(records, mode, view)
        for view in ('Q3', 'Q4') for mode in ('same', 'cross')
    }
    cross_items = [record['cross'] for record in records]
    summary = {
        'checkpoint': str(Path(args.ckpt)),
        'threshold': args.th,
        'tta': args.tta,
        'metric_note': (
            'The 19 photos have no hand-labelled keypoints. Coverage, strict '
            'acceptance, and ambiguity are geometric consistency metrics and '
            'must not be reported as independent accuracy.'
        ),
        'cross_total': {
            'frames': len(cross_items),
            'strict_answered': sum(item['strict_answer'] for item in cross_items),
            'found': sum(item['found'] for item in cross_items),
            'expected': sum(item['expected'] for item in cross_items),
            'whole_rows': sum(item['whole_rows'] for item in cross_items),
            'ambiguous_rows': sum(item['ambiguous_rows'] for item in cross_items),
        },
        'aggregate': aggregates,
        'records': records,
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([
        np.hstack(tiles[index:index + 2])
        for index in range(0, len(tiles), 2)
    ])
    cv2.imwrite(str(output / 'cross_view_contact_sheet.jpg'), grid)
    print(json.dumps({'cross_total': summary['cross_total'],
                      'aggregate': aggregates}, indent=2))
    print('OUT', output)


if __name__ == '__main__':
    main()
