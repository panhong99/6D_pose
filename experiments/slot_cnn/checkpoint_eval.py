"""Fast per-checkpoint real-photo metrics for a running slot-CNN training job.

Unlike eval.py, this intentionally skips the five-threshold sweep and image
overlays.  It runs inference once per labelled photo, then reuses those
predictions for detection and tine-number metrics.  Full visual evaluation is
still done for selected/final checkpoints with eval.py.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import data as D
import eval as E
from train import KEYS, TineNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--th', type=float, default=0.3)
    p.add_argument('--tol', type=float, default=8.0)
    p.add_argument('--trials', type=int, default=100)
    p.add_argument('--tta', action='store_true')
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    E.TTA = a.tta
    net = TineNet(pretrained=False).cuda().eval()
    net.load_state_dict(torch.load(a.ckpt, map_location='cuda', weights_only=True))

    predictions = {}
    rows = []
    for image_path, _, pins, quadrant in D.real_tests():
        image = cv2.imread(image_path)
        pred = E.predict(net, image, a.th)
        predictions[quadrant] = pred
        labelled = [point for pin in pins for point in pin[1:] if point]
        hull = cv2.convexHull(np.asarray(labelled, dtype=np.float32))
        region = np.zeros(image.shape[:2], np.uint8)
        cv2.fillConvexPoly(region, hull.astype(np.int32), 1)
        region = cv2.dilate(region, np.ones((61, 61), np.uint8))
        rec = {'photo': Path(image_path).name, 'label_quadrant': quadrant}
        for kind in KEYS:
            gt = [point[1 if kind == 'root' else 2] for point in pins
                  if point[1 if kind == 'root' else 2]]
            gt = np.asarray(gt, dtype=float).reshape(-1, 2)
            points = pred[kind]
            inside = points[
                region[
                    np.clip(points[:, 1].astype(int), 0, image.shape[0] - 1),
                    np.clip(points[:, 0].astype(int), 0, image.shape[1] - 1),
                ] > 0
            ] if len(points) else points
            matches, distances = E.match(inside, gt, a.tol)
            # The v4 annotations cover one quadrant rather than the complete
            # image.  Keep the historical in-area precision for comparable
            # checkpoint selection, but also expose every prediction in the
            # frame as requested by the review.  ``unmatched_full_frame`` is
            # deliberately conservative: a real tine outside the annotated
            # quadrant is counted as unmatched because this four-photo set
            # has no ground truth for it.
            rec[kind] = {
                'labels': len(gt),
                'found': len(matches),
                'pred_in_area': len(inside),
                'pred_full_frame': len(points),
                'unmatched_in_area': len(inside) - len(matches),
                'unmatched_full_frame': len(points) - len(matches),
                'recall': len(matches) / max(1, len(gt)),
                'precision': len(matches) / max(1, len(inside)),  # compatibility: in-area precision
                'precision_in_area': len(matches) / max(1, len(inside)),
                'precision_full_frame_conservative': len(matches) / max(1, len(points)),
                'err_px_median': float(np.median(distances)) if len(distances) else None,
            }
        rows.append(rec)

    total = {}
    for kind in KEYS:
        labels = sum(row[kind]['labels'] for row in rows)
        found = sum(row[kind]['found'] for row in rows)
        predicted_area = sum(row[kind]['pred_in_area'] for row in rows)
        predicted_full = sum(row[kind]['pred_full_frame'] for row in rows)
        total[kind] = {
            'labels': labels,
            'found': found,
            'recall': found / max(1, labels),
            'pred_in_area': predicted_area,
            'pred_full_frame': predicted_full,
            'unmatched_in_area': predicted_area - found,
            'unmatched_full_frame': predicted_full - found,
            'precision': found / max(1, predicted_area),  # compatibility: in-area precision
            'precision_in_area': found / max(1, predicted_area),
            'precision_full_frame_conservative': found / max(1, predicted_full),
        }

    numbers = {
        kind: E.number_check({q: pred[kind] for q, pred in predictions.items()},
                             trials=a.trials, kind=kind)
        for kind in KEYS
    }
    numbers['both'] = E.number_check(predictions, trials=a.trials, kind='both')
    for result in numbers.values():
        for quadrant in result.values():
            quadrant.pop('info', None)

    result = {
        'ckpt': a.ckpt,
        'th': a.th,
        'tol_px': a.tol,
        'tta': a.tta,
        'precision_note': (
            'precision_in_area uses the annotated quadrant hull dilated by 30 px. '
            'precision_full_frame_conservative counts predictions outside that area as unmatched, '
            'although the rest of the frame is not exhaustively labelled.'
        ),
        'total': total,
        'photos': rows,
        'tine_numbers': numbers,
    }
    (out / 'metrics.json').write_text(json.dumps(result, indent=1))
    print(json.dumps({'total': total, 'photos': rows, 'tine_numbers': numbers}, indent=1))


if __name__ == '__main__':
    main()
