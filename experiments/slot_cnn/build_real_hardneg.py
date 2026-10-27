"""Build a small sim + real hard-negative set for slot-CNN fine tuning.

The real images use the current checkpoint predictions as pseudo-positive keypoints.
Only manually reviewed false tip endpoints are removed.  Output is a new dataset;
source renders, photos, labels, and checkpoints are never modified.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402
import eval as E  # noqa: E402
from train import TineNet  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sim', required=True)
    p.add_argument('--photos', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--sim-count', type=int, default=3000)
    p.add_argument('--real-repeat', type=int, default=100)
    p.add_argument('--seed', type=int, default=27)
    p.add_argument('--remove-radius-px', type=float, default=5.0)
    p.add_argument('--th', type=float, default=0.3)
    p.add_argument('--tta', action='store_true')
    a = p.parse_args()

    sim, photos, out = Path(a.sim), Path(a.photos), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    sim_labels = json.loads((sim / 'labels.json').read_text())
    rng = np.random.default_rng(a.seed)
    selected = rng.choice(len(sim_labels['samples']), min(a.sim_count, len(sim_labels['samples'])), replace=False)
    samples = []
    next_id = 0
    for index in selected:
        sample = dict(sim_labels['samples'][int(index)])
        source = sim / f"{sample['i']:06d}.jpg"
        target = out / f'{next_id:06d}.jpg'
        os.link(source, target)
        sample['i'] = next_id
        sample['source'] = 'sim_v5_cross'
        samples.append(sample)
        next_id += 1

    review = json.loads(Path(a.labels).read_text())
    by_photo = {}
    for item in review['frames']:
        by_photo.setdefault(item['photo'], []).append(np.asarray(item['false_tip_uv'], float))

    E.TTA = a.tta
    net = TineNet(pretrained=False).cuda().eval()
    net.load_state_dict(torch.load(a.ckpt, map_location='cuda', weights_only=True))
    real_samples = []
    for name, false_uv in sorted(by_photo.items()):
        image = cv2.imread(str(photos / name))
        if image is None:
            raise FileNotFoundError(photos / name)
        pred = E.predict(net, image, a.th)
        keep_tip = np.ones(len(pred['tip']), bool)
        for uv in false_uv:
            distance = np.linalg.norm(pred['tip'][:, :2] - uv, axis=1)
            if not len(distance) or distance.min() > a.remove_radius_px:
                raise RuntimeError(f'{name}: reviewed false endpoint was not predicted within radius')
            keep_tip[int(distance.argmin())] = False
        resized = cv2.resize(image, (D.OW, D.OH - 4), interpolation=cv2.INTER_AREA)
        resized = np.r_[resized, np.zeros((4, D.OW, 3), np.uint8)]
        image_id = next_id
        cv2.imwrite(str(out / f'{image_id:06d}.jpg'), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        tines = []
        for kind in ('root', 'tip'):
            points = pred[kind][keep_tip] if kind == 'tip' else pred[kind]
            for x, y, _ in points:
                rec = {'root': None, 'root_vis': False, 'tip': None, 'tip_vis': False}
                rec[kind] = [float(x * E.SCALE), float(y * E.SCALE)]
                rec[kind + '_vis'] = True
                tines.append(rec)
        sample = {'i': image_id, 'profile': 'real_hard_negative', 'source_photo': name,
                  'removed_false_tips': len(false_uv), 'tines': tines}
        real_samples.append(sample)
        samples.extend([sample] * a.real_repeat)
        next_id += 1

    manifest = {
        'size': sim_labels['size'],
        'samples': samples,
        'construction': {
            'sim_source': str(sim), 'sim_unique': len(selected),
            'real_unique': len(real_samples), 'real_repeat': a.real_repeat,
            'teacher_checkpoint': a.ckpt, 'teacher_tta': a.tta,
            'manual_endpoint_labels': a.labels,
            'note': ('Real positives are teacher pseudo-labels. Manually reviewed endpoint false positives are excluded '
                     'and become heatmap negatives; this is not independent ground truth.'),
        },
    }
    (out / 'labels.json').write_text(json.dumps(manifest, indent=1))
    (out / 'real_samples.json').write_text(json.dumps(real_samples, indent=1))
    print(json.dumps(manifest['construction'], indent=1))


if __name__ == '__main__':
    main()
