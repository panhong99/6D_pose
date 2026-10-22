"""(10/20 보스: 바구니에 아루코가 없다고 가정) marker-free basket pose from the fixed D455 (calibration/basket_vision.py):
offline check against the marker pose saved with every real run (no robot, no camera):

    ~/miniconda3/envs/foundationpose/bin/python experiments/basket_free/fit.py [--out DIR] [--show N]

The method lives in calibration/basket_vision.py.  Basket frame (rl/basket.yaml full): origin = rim centre at the rail
top, x robot side -> far side, y left seen from the robot, the blue handle on -y.  (A first rim-rectangle-only fit gave
xy median 12 mm and 19/145 turned 90 deg: dropped.)"""
import argparse
import json
from pathlib import Path

import sys

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'calibration'))
from basket_vision import Detector, estimate, model_points  # noqa: E402
RUNS = ROOT / 'calibration/data/real_pipeline'
SIZE = np.array([0.53, 0.52])                       # rim x (rails), y (handle side) [m]
def frames():
    """(name, rgb, depth, K, T_base_cam, T_base_basket marker, joints deg) of every saved fixed-camera frame with a marker pose."""
    for npz in sorted(RUNS.glob('*/fixed_*.npz')):
        meta = npz.with_suffix('.json')
        if not meta.is_file():
            continue
        m = json.loads(meta.read_text())
        if m.get('T_base_basket') is None or m.get('T_base_cam') is None:
            continue
        if (npz.parent / 'basket_vision.json').is_file():   # (10/22) that run's basket pose came from this method, not markers
            continue
        z = np.load(npz)
        yield (f'{npz.parent.name}/{npz.stem}', z['rgb'], z['depth'], z['K'].astype(float),
               np.asarray(m['T_base_cam'], float), np.asarray(m['T_base_basket'], float), m.get('posj'))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--out', default=str(ROOT / 'calibration/data/basket/basket_free_20261020'))
    a.add_argument('--show', type=int, default=6)
    a.add_argument('--since', default='20261019_1700', help='runs from this time on (current marker layout = 10/19 16:5x)')
    a.add_argument('--plate_colours', default='', help="(10/22) '', 'red', 'red_white': pixels dropped from the rack fit")
    a.add_argument('--min_cover', type=float, default=None, help='(10/22) basket_vision.MIN_COVER occlusion gate')
    a.add_argument('--refit_mm', type=float, default=None, help='(10/22) drop points farther than this from the fitted model, refit')
    a = a.parse_args()
    import basket_vision
    basket_vision.PLATE_COLOURS = a.plate_colours
    basket_vision.REFIT_MM = a.refit_mm
    basket_vision.MIN_COVER = a.min_cover
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    z_rim = yaml.safe_load((ROOT / 'calibration/config.yaml').read_text())['basket']['fixed_z_m']
    det = Detector()
    rows = []
    M = model_points()
    for i, (name, rgb, depth, K, Tbc, Tm, posj) in enumerate(f for f in frames() if f[0][:13] >= a.since):
        e = estimate(det, rgb, depth, K, Tbc, z_rim, posj, M)
        row = dict(name=name, ok=e is not None)
        if e:
            d = e['T'][:2, 3] - Tm[:2, 3]
            yaw_m = np.degrees(np.arctan2(Tm[1, 0], Tm[0, 0]))
            yaw_e = np.degrees(np.arctan2(e['T'][1, 0], e['T'][0, 0]))
            row.update(refused=e['refused'], dx_mm=d[0] * 1000, dy_mm=d[1] * 1000, dyaw_deg=(yaw_e - yaw_m + 180) % 360 - 180,
                       conf=e['conf'], cost=e['cost'], n_top=e['n_top'], n_handle=e['n_handle'],
                       marker_xy=Tm[:2, 3].tolist(), marker_yaw=yaw_m)
            if i < a.show:
                draw(out / f'{i:03d}.png', rgb, K, Tbc, Tm, e['T'])
        rows.append(row)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()
                          if k not in ('marker_xy',)}), flush=True)
    (out / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    s = dict(frames=len(rows), found=sum(r['ok'] for r in rows))
    for tag, keep in (('all', lambda r: True), ('offset_fit_1019', lambda r: r['name'] < '20261020'),
                      ('test_1020', lambda r: r['name'] >= '20261020'), ('arm_above', lambda r: 'fixed_above' in r['name']),
                      ('plates_in_after_release', lambda r: 'after_release' in r['name'])):
        ok = [r for r in rows if r['ok'] and keep(r)]
        acc = [r for r in ok if not r.get('refused')]
        if not acc:
            continue
        xy = np.array([np.hypot(r['dx_mm'], r['dy_mm']) for r in acc])
        yw = np.abs([r['dyaw_deg'] for r in acc])
        s[tag] = dict(frames=len(ok), accepted=len(acc), refused=len(ok) - len(acc),
                      xy_mm_median=float(np.median(xy)), xy_mm_p90=float(np.percentile(xy, 90)), xy_mm_max=float(xy.max()),
                      yaw_deg_median=float(np.median(yw)), yaw_deg_p90=float(np.percentile(yw, 90)),
                      yaw_deg_max=float(yw.max()), wrong_over_20mm_or_3deg=int(((xy > 20) | (yw > 3)).sum()))
    (out / 'summary.json').write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


def draw(path, rgb, K, Tbc, Tm, Te):
    img = rgb[:, :, ::-1].copy()
    hx, hy = np.array([0.53, 0.52]) / 2
    corners = np.array([[hx, hy, 0, 1], [-hx, hy, 0, 1], [-hx, -hy, 0, 1], [hx, -hy, 0, 1]])
    Tcb = np.linalg.inv(Tbc)
    for T, col in ((Tm, (0, 200, 0)), (Te, (0, 0, 255))):
        p = (Tcb @ T @ corners.T)[:3]
        uv = (K @ (p / p[2]))[:2].T.astype(int)
        cv2.polylines(img, [uv.reshape(-1, 1, 2)], True, col, 1)
        o = (Tcb @ T @ np.array([[0, 0, 0, 1], [0.15, 0, 0, 1], [0, -0.15, 0, 1]]).T)[:3]
        o = (K @ (o / o[2]))[:2].T.astype(int)
        cv2.line(img, tuple(o[0]), tuple(o[1]), col, 2)                  # basket +x
        cv2.circle(img, tuple(o[2]), 4, col, -1)                          # handle side (-y)
    cv2.imwrite(str(path), img)


if __name__ == '__main__':
    main()
