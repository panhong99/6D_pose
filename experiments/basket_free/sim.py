"""(10/22 보스: 마커 없는 바구니를 sim에서 안정화) the fixed D455 view of the basket model anywhere in reach, rendered
(nvdiffrast, the slot CNN scene of experiments/slot_cnn/data.py: randomised tines, plates standing in the basket, white
clutter, textured table) with D455-like depth, then calibration/basket_vision.py on it against the true pose:

    ~/miniconda3/envs/foundationpose/bin/python experiments/basket_free/sim.py --n 200 --out calibration/data/basket/sim_eval_v1
    ~/miniconda3/envs/foundationpose/bin/python experiments/basket_free/sim.py --n 8 --sheet --out /tmp/x   # look first

Each scene: 5 renders (light, noise, depth dropout change) -> the same per-frame estimate + combine as the robot.
Reported: found (rack mask), accepted, xy / yaw error of the accepted, wrong-but-accepted (> 20 mm or > 3 deg), why refused."""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'calibration'))
sys.path.insert(0, str(ROOT / 'experiments/slot_cnn'))
import basket_vision as bv  # noqa: E402
import data as sd  # noqa: E402

W, H = 640, 480


def fixed_camera():
    """K of the real fixed D455 frames and the newest T_base_cam."""
    from calib_core import latest_t_base_cam
    z = np.load(ROOT / 'calibration/data/basket/vision_debug_20261022_1445/fixed_snapshot.npz')
    Tbc = np.asarray(json.loads(Path(latest_t_base_cam()).read_text())['T_base_cam'], float)
    return z['K'].astype(float), Tbc


def sample_basket(rng, z_rim, x=(0.72, 1.08), y=(-0.42, 0.20), yaw=(-50, 50)):
    """A basket pose in reach (base frame; the handle side -y turns with yaw)."""
    a = np.radians(rng.uniform(*yaw))
    T = np.eye(4)
    T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    T[:3, 3] = [rng.uniform(*x), rng.uniform(*y), z_rim]
    return T


def stand_plate(m, rng, T_bb, colour=None):
    """The plate to pick on its stand at the usual spot (base ~(0.90, +0.34)), as in every real frame (basket frame)."""
    c_base = np.array([rng.uniform(0.86, 0.94), rng.uniform(0.30, 0.38), 0.075, 1.0])
    c = (np.linalg.inv(T_bb) @ c_base)[:3]
    R = np.linalg.inv(T_bb)[:3, :3] @ sd._frame_z(np.array([0, 0, 1.0]) + rng.normal(0, 0.03, 3))
    v, nn, ff = sd._unit_cyl(48)
    col = colour if colour is not None else (np.array([0.72, 0.03, 0.04]) if rng.random() < 0.5 else np.full(3, 0.92))
    m.add((v * [0.134, 0.134, 0.022] - [0, 0, 0.011]) @ R.T + c, nn @ R.T, ff, col)


def d455_depth(depth, rng, fat=7):
    """Ideal z -> D455-like: stereo 'fattening' (thin wires spread to the nearest surface over ~fat px: the real rack
    depth is a filled sheet, ~11k points), random dropout, z-dependent correlated noise; 0 = no depth."""
    d = depth.copy()
    d[d > 3.0] = 3.0
    k = int(rng.integers(fat - 2, fat + 3)) | 1
    near = -cv2.dilate(-d.astype(np.float32), np.ones((k, k), np.uint8))   # nearest surface around each pixel
    d = np.where(near < d - 0.03, near, d)                                  # foreground wins over a far background
    drop = rng.random(d.shape) < rng.uniform(0.02, 0.10)
    noise = cv2.GaussianBlur(rng.normal(0, 1, d.shape).astype(np.float32), (0, 0), 2.0) * 2.0
    d = d + noise * (0.0015 + 0.0025 * d ** 2)
    d[drop | (d >= 2.9)] = 0
    return d.astype(np.float32)


def scene(rng, parts, B, tines, T_bb):
    m, ts, inserted = sd.build_scene(rng, parts, B, tines, profile='v4')
    stand_plate(m, rng, T_bb)
    return m, inserted


def render(rd, m, T_cam_world, tex, rng, K):
    sd.W, sd.H = W, H                                   # the renderer reads the module image size
    img, depth = rd.render(m, T_cam_world, tex, rng, K=K)
    img = img * rng.uniform(0.8, 1.2) * rng.uniform(0.92, 1.08, 3)
    img = img + rng.normal(0, rng.uniform(0.004, 0.02), img.shape)
    rgb = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return rgb, d455_depth(depth, rng), depth


class MaskDetector:
    """YOLOE stand-in (the renders are too far from the real look for it: rack conf <= 0.09): the true rack silhouette as
    one filled blob that keeps plates standing in it (YOLOE did that on 10/22), and the stand plate as a 'plate'."""

    def set(self, depth, K, T_bb, Tbc, rng):
        h, w = depth.shape
        v, u = np.mgrid[:h, :w]
        ok = depth < 3.0
        Pc = np.stack([(u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth], -1)
        Pb = Pc @ Tbc[:3, :3].T + Tbc[:3, 3]
        Pk = (Pb - T_bb[:3, 3]) @ T_bb[:3, :3]
        rack = ok & (np.abs(Pk[..., 0]) < 0.28) & (np.abs(Pk[..., 1]) < 0.28) & (Pk[..., 2] > -0.14) & (Pk[..., 2] < 0.16)
        rack = cv2.morphologyEx(rack.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        cs, _ = cv2.findContours(rack, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.rack = None
        if cs:
            blob = np.zeros((h, w), np.uint8)
            cv2.drawContours(blob, [cv2.convexHull(max(cs, key=cv2.contourArea))], -1, 1, -1)
            self.rack = cv2.erode(blob, np.ones((3, 3), np.uint8), iterations=int(rng.integers(0, 3))).astype(bool)
        plate = ok & (Pb[..., 2] > 0.03) & (np.hypot(Pb[..., 0] - 0.90, Pb[..., 1] - 0.34) < 0.2) & ~(self.rack if self.rack is not None else False)
        self.plate_masks = [(plate, 0.9)] if plate.sum() > 200 else []
        self.conf = float(rng.uniform(0.5, 0.9))

    def masks(self, rgb, conf=0.05):
        plates = self.plate_masks[0][0] if self.plate_masks else np.zeros(rgb.shape[:2], bool)
        return self.rack, plates, (self.conf if self.rack is not None else 0.0)


def error(T_est, T_true):
    E = np.asarray(T_est) @ np.linalg.inv(bv.STRUCT_TO_FRAME)  # the sim model has no structure -> frame offset
    dxy = 1000 * float(np.linalg.norm(E[:2, 3] - T_true[:2, 3]))
    dyaw = np.degrees(np.arctan2(E[1, 0], E[0, 0]) - np.arctan2(T_true[1, 0], T_true[0, 0]))
    return dxy, float((dyaw + 180) % 360 - 180)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--n', type=int, default=100)
    a.add_argument('--frames', type=int, default=5)
    a.add_argument('--seed', type=int, default=0)
    a.add_argument('--out', required=True)
    a.add_argument('--sheet', action='store_true', help='save the renders (rgb + depth) for a look')
    a.add_argument('--yaw', type=float, nargs=2, default=(-50, 50))
    a.add_argument('--yoloe', action='store_true', help='real YOLOE on the renders (domain gap: mostly not found)')
    a.add_argument('--plate_colours', default='', help="'', 'red', 'red_white' (basket_vision.PLATE_COLOURS)")
    a.add_argument('--refit_mm', type=float, default=None, help='basket_vision.REFIT_MM')
    a = a.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    K, Tbc = fixed_camera()
    from calib_core import load_config
    z_rim = float(load_config()['basket']['fixed_z_m'])
    parts, B = sd.basket_parts(False)
    tines = sd.tine_table(parts, B)
    rd, M = sd.Renderer(), bv.model_points()
    det = bv.Detector() if a.yoloe else MaskDetector()
    bv.PLATE_COLOURS, bv.REFIT_MM = a.plate_colours, a.refit_mm
    rng = np.random.default_rng(a.seed)
    rows = []
    for i in range(a.n):
        T_bb = sample_basket(rng, z_rim, yaw=a.yaw)
        m, inserted = scene(rng, parts, B, tines, T_bb)
        tex = sd.table_texture(rng)
        Tcw = np.linalg.inv(np.linalg.inv(T_bb) @ Tbc)
        res, t0 = [], time.time()
        for k in range(a.frames):
            rgb, depth, ideal = render(rd, m, Tcw, tex, rng, K)
            if not a.yoloe:
                det.set(ideal, K, T_bb, Tbc, rng)
            if a.sheet and k == 0:
                cv2.imwrite(str(out / f'{i:04d}_rgb.jpg'), rgb[..., ::-1])
                np.savez_compressed(out / f'{i:04d}.npz', rgb=rgb, depth=depth, K=K, T_base_basket=T_bb, T_base_cam=Tbc)
            e = bv.estimate(det, rgb, depth, K, Tbc, z_rim, None, M)
            res.append(e if e is not None else dict(refused='rack not found'))
        c = bv.combine(res, z_rim, 6.0, 1.5)
        found = sum(1 for r in res if r.get('refused') != 'rack not found')
        row = dict(i=i, x=round(T_bb[0, 3], 3), y=round(T_bb[1, 3], 3),
                   yaw=round(float(np.degrees(np.arctan2(T_bb[1, 0], T_bb[0, 0]))), 1), inserted=inserted,
                   found=found, ok=bool(c.get('ok')), why=c.get('why', ''), s=round(time.time() - t0, 1),
                   conf=[round(float(r.get('conf', 0)), 2) for r in res])
        if 'T' in c:
            row['xy_mm'], row['yaw_deg'] = error(c['T'], T_bb)
            row['spread_mm'], row['spread_deg'] = round(c['spread_mm'], 1), round(c['spread_deg'], 2)
        per = [error(r['T'], T_bb) for r in res if 'T' in r]
        row['frame_xy_mm'] = [round(p[0], 1) for p in per]
        rows.append(row)
        print(json.dumps(row), flush=True)
    (out / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    s = summary(rows)
    (out / 'summary.json').write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


def summary(rows):
    acc = [r for r in rows if r['ok']]
    xy = np.array([r['xy_mm'] for r in acc]) if acc else np.zeros(0)
    yw = np.abs([r['yaw_deg'] for r in acc]) if acc else np.zeros(0)
    why = {}
    for r in rows:
        if not r['ok']:
            for w in (r['why'] or 'unknown').split('; '):
                k = w.split(':')[0]
                why[k] = why.get(k, 0) + 1
    return dict(scenes=len(rows), all_frames_found=sum(r['found'] == len(r['conf']) for r in rows),
                accepted=len(acc), wrong_accepted=int(sum((xy > 20) | (yw > 3))) if acc else 0,
                xy_mm_median=float(np.median(xy)) if acc else None, xy_mm_p90=float(np.percentile(xy, 90)) if acc else None,
                yaw_deg_median=float(np.median(yw)) if acc else None, yaw_deg_p90=float(np.percentile(yw, 90)) if acc else None,
                refused=why,
                by_inserted={n: f"{sum(r['ok'] for r in rows if r['inserted'] == n)}/{sum(r['inserted'] == n for r in rows)}"
                             for n in sorted({r['inserted'] for r in rows})})


if __name__ == '__main__':
    main()
