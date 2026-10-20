"""(10/20 보스: 바구니에 아루코가 없다고 가정) basket pose from the fixed D455 alone: YOLOE prompt mask + depth ->
base-frame points -> the known 53 x 52 cm rim rectangle fitted in x, y, yaw (z = config basket.fixed_z_m).  Offline
check against the marker pose saved with every real run (no robot, no camera):

    ~/miniconda3/envs/foundationpose/bin/python experiments/basket_free/fit.py [--out DIR] [--show N]

Basket frame (rl/basket.yaml full): origin = rim centre at the rail top, x robot side -> far side, y left seen from
the robot, the blue handle on -y.  The rim is nearly square, so the 90-degree turn is told apart by the blue handle."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'calibration/data/real_pipeline'
SIZE = np.array([0.53, 0.52])                       # rim x (rails), y (handle side) [m]
PROMPTS = ['dish drying rack', 'plate']
MIN_HANDLE, MIN_POINTS = 20, 6000                   # below: refuse (look again with the arm out of the view)
# One-time offset of the rack structure (rl/assets/basket_full.obj walls/rails/handle) from the frame the slots are
# defined in: median of 25 marker frames on 10/19 17:00-19:30 (model_v5 run): model fit = marker + (-3.0, -24.4) mm.
STRUCT_TO_FRAME = np.eye(4)
STRUCT_TO_FRAME[:2, 3] = [0.0030, 0.0244]             # rack mask; plates standing in it are cut out


def frames():
    """(name, rgb, depth, K, T_base_cam, T_base_basket marker, joints deg) of every saved fixed-camera frame with a marker pose."""
    for npz in sorted(RUNS.glob('*/fixed_*.npz')):
        meta = npz.with_suffix('.json')
        if not meta.is_file():
            continue
        m = json.loads(meta.read_text())
        if m.get('T_base_basket') is None or m.get('T_base_cam') is None:
            continue
        z = np.load(npz)
        yield (f'{npz.parent.name}/{npz.stem}', z['rgb'], z['depth'], z['K'].astype(float),
               np.asarray(m['T_base_cam'], float), np.asarray(m['T_base_basket'], float), m.get('posj'))


class Detector:
    def __init__(self):
        from ultralytics import YOLOE
        self.m = YOLOE(str(ROOT / 'yoloe-v8l-seg.pt'))
        self.m.set_classes(PROMPTS, self.m.get_text_pe(PROMPTS))

    def masks(self, rgb, conf=0.1):
        """(rack mask or None, plate mask union, rack confidence)."""
        r = self.m.predict(rgb[:, :, ::-1].copy(), conf=conf, verbose=False, retina_masks=True)[0]
        h, w = rgb.shape[:2]
        rack, best, plates = None, 0.0, np.zeros((h, w), bool)
        if r.masks is None:
            return rack, plates, best
        for mk, c, k in zip(r.masks.data.cpu().numpy() > 0.5, r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            if int(k) == 0 and c > best:
                rack, best = mk, float(c)
            elif int(k) == 1:
                plates |= mk
        return rack, plates, best


def cloud(mask, depth, K, Tbc, stride=1):
    v, u = np.nonzero(mask[::stride, ::stride])
    v, u = v * stride, u * stride
    d = depth[v, u].astype(float)
    ok = (d > 0.2) & (d < 2.5)
    u, v, d = u[ok], v[ok], d[ok]
    P = np.c_[(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d]
    return P @ Tbc[:3, :3].T + Tbc[:3, 3], np.c_[u, v]


def rect_fit(xy, size=SIZE):
    """Centre and yaw (mod 90 deg) of the size rectangle whose outline the points hug best (robust, trimmed)."""
    best = None
    for yaw in np.radians(np.arange(0.0, 90.0, 0.5)):
        c, s = np.cos(yaw), np.sin(yaw)
        loc = xy @ np.array([[c, -s], [s, c]])          # points in the rotated frame
        for sz in (size, size[::-1]):
            lo, hi = np.percentile(loc, 2, axis=0), np.percentile(loc, 98, axis=0)
            ctr = (lo + hi) / 2                          # centre from the robust extents, then the outline distance
            cost = np.mean(np.minimum(outline_dist(loc - ctr, sz), 0.03))
            if best is None or cost < best[0]:
                best = (cost, yaw, ctr, sz)
    cost, yaw, ctr, sz = best
    for _ in range(3):                                   # refine yaw and centre on the outline (coordinate search)
        for step in (np.radians(0.25), np.radians(0.1)):
            for dyaw in (-step, 0.0, step):
                c, s = np.cos(yaw + dyaw), np.sin(yaw + dyaw)
                loc = xy @ np.array([[c, -s], [s, c]])
                for dc in np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]) * 0.002:
                    k = np.mean(np.minimum(outline_dist(loc - ctr - dc, sz), 0.03))
                    if k < cost - 1e-9:
                        cost, yaw, ctr = k, yaw + dyaw, ctr + dc
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]]) @ ctr, yaw, cost, sz


def outline_dist(p, sz):
    """Distance of 2D points (rectangle frame, centred) to the outline of a sz rectangle."""
    q = np.abs(p) - sz / 2
    outside = np.linalg.norm(np.maximum(q, 0), axis=1)
    inside = -np.minimum(q.max(1), 0)
    return outside + inside


def resolve_turn(ctr, yaw, sz, handle_xy):
    """Basket yaw from the rectangle's mod-90 yaw: the blue handle side is basket -y, and the 0.53 m side is x."""
    cands = []
    for k in range(4):
        a = yaw + k * np.pi / 2
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        if handle_xy is not None and len(handle_xy):
            hy = np.median((handle_xy - ctr) @ R[:, 1])     # handle offset along basket y
            cands.append((hy, a))
        else:
            cands.append((0.0, a))
    return min(cands)[1]                                    # most negative y = handle side


def blue(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[..., 0] > 95) & (hsv[..., 0] < 125) & (hsv[..., 1] > 120) & (hsv[..., 2] > 60)


def model_points(n=20000, z_min=-0.13):
    """Basket model surface (rl/assets/basket_full.obj, basket frame), floor grid left out (it touches the table)."""
    import trimesh
    m = trimesh.load(str(ROOT / 'rl/assets/basket_full.obj'), force='mesh')
    p, _ = trimesh.sample.sample_surface(m, n, seed=0)
    return p[p[:, 2] > z_min]


def arm_points(posj_deg):
    """Robot links + the gripper cylinder in base at the capture joints (scene points near them are the arm)."""
    import sys
    sys.path.insert(0, str(ROOT / 'calibration'))
    from robot_model import CollisionChecker, Kinematics
    global _CH
    if '_CH' not in globals():
        _CH = CollisionChecker(Kinematics(), [], bounds=((0, 0.1), (0, 0.1), (0, 0.1)))
        _CH.set_tool(0.165, 0.06)
    return _CH.robot_points(np.radians(posj_deg))


def icp_xy_yaw(S, M, yaw0, t0, z_rim, iters=30, trim=0.7):
    """Scene points S (base) onto the model M (basket frame) in x, y, yaw at a fixed height: trimmed point-to-point."""
    from scipy.spatial import cKDTree
    yaw, t = yaw0, np.asarray(t0, float)
    for _ in range(iters):
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        loc = np.c_[(S[:, :2] - t) @ R, S[:, 2] - z_rim]          # scene in the basket frame
        d, j = cKDTree(M).query(loc)
        keep = d <= np.quantile(d, trim)
        a, b = loc[keep, :2], M[j[keep], :2]                       # loc -> model: small 2D rigid update
        ca, cb = a.mean(0), b.mean(0)
        H = (a - ca).T @ (b - cb)
        dth = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
        # model = Rd (loc - ca) + cb  ->  basket frame turns by -dth
        Rd = np.array([[np.cos(dth), -np.sin(dth)], [np.sin(dth), np.cos(dth)]])
        # new scene->basket map: Rd R^T (x - t) - Rd ca + cb  =  R'^T (x - t')
        Rn = R @ Rd.T
        tn = t + R @ (ca - Rd.T @ cb)
        yaw, t = np.arctan2(Rn[1, 0], Rn[0, 0]), tn
        if abs(dth) < 1e-5 and np.linalg.norm(tn - t) < 1e-5:
            break
    c, s = np.cos(yaw), np.sin(yaw)
    loc = np.c_[(S[:, :2] - t) @ np.array([[c, -s], [s, c]]), S[:, 2] - z_rim]
    d, _ = cKDTree(M).query(loc)
    return yaw, t, float(np.mean(np.sort(d)[:int(trim * len(d))]))


def raster(xy, lo, n, cell):
    img = np.zeros((n, n))
    ij = np.floor((xy - lo) / cell).astype(int)
    ok = (ij >= 0).all(1) & (ij < n).all(1)
    np.add.at(img, (ij[ok, 0], ij[ok, 1]), 1.0)
    return img


def global_search(P, M, z_rim, cell=0.005, half=0.5, step_deg=1.0, keep=6, band=(-0.07, 0.03)):
    """Top-view correlation of the upper structure (rails, handle, wall tops, tine tops) over every yaw and shift:
    -> the `keep` best (yaw, centre) seeds for the ICP.  Rewards overlap both ways, so a turned fit that leaves the
    model's rails or tine rows without scene points loses (the ICP alone sits in +-20 deg wire-pitch minima)."""
    from scipy.ndimage import gaussian_filter
    from scipy.signal import fftconvolve
    S = P[(P[:, 2] - z_rim > band[0]) & (P[:, 2] - z_rim < band[1])]
    Mu = M[(M[:, 2] > band[0]) & (M[:, 2] < band[1])]
    ctr = np.median(S[:, :2], axis=0)
    n = int(2 * half / cell)
    lo = ctr - half
    Si = gaussian_filter(np.minimum(raster(S[:, :2], lo, n, cell), 3.0), 1.5)
    Si -= Si.mean()
    out = []
    for yd in np.arange(0.0, 360.0, step_deg):
        a = np.radians(yd)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        m = Mu[:, :2] @ R.T
        Mi = gaussian_filter(np.minimum(raster(m, np.array([-half, -half]), n, cell), 3.0), 1.5)
        Mi /= np.linalg.norm(Mi) + 1e-9
        c = fftconvolve(Si, Mi[::-1, ::-1], mode='same')
        k = np.unravel_index(np.argmax(c), c.shape)
        shift = (np.array(k) - n // 2) * cell
        out.append((c[k], a, ctr + shift))
    out.sort(key=lambda r: -r[0])
    seeds = []
    for sc, a, t in out:                                           # distinct seeds (yaw > 5 deg apart)
        if all(abs((a - b + np.pi) % (2 * np.pi) - np.pi) > np.radians(5) for b, _ in seeds):
            seeds.append((a, t))
        if len(seeds) >= keep:
            break
    return [(a, t) for a, t in seeds]


def estimate_model(det, rgb, depth, K, Tbc, z_rim, posj=None, M=None):
    """Whole-model fit: YOLOE rack mask - plates - arm -> points -> ICP from 12 yaws; the best trimmed fit wins."""
    from scipy.spatial import cKDTree
    rack, plates, conf = det.masks(rgb)
    if rack is None:
        return None
    m = cv2.dilate(rack.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~plates
    P, _ = cloud(m, depth, K, Tbc)
    P = P[(P[:, 2] > 0.015) & (P[:, 2] < z_rim + 0.03)]
    if posj is not None:
        d, _ = cKDTree(arm_points(posj)).query(P)
        P = P[d > 0.04]
    if len(P) < 100:
        return None
    ctr = np.median(P[:, :2], axis=0)
    seeds = global_search(P, M, z_rim)
    fits = sorted((icp_xy_yaw(P, M, y0, t0, z_rim) for y0, t0 in seeds), key=lambda r: r[2])
    yaw, t, cost = fits[0]
    Hb, _ = cloud(m & blue(rgb), depth, K, Tbc)                    # the blue handle is basket -y: pick the turn of the
    Hb = Hb[(Hb[:, 2] > 0.03) & (Hb[:, 2] < z_rim + 0.03)]         # good fits (cost within 30 %) that puts it there
    n_handle = len(Hb)
    if n_handle >= 20:
        def handle_y(r):
            c, s = np.cos(r[0]), np.sin(r[0])
            return np.median((Hb[:, :2] - r[1]) @ np.array([-s, c]))
        flip = icp_xy_yaw(P, M, yaw + np.pi, t, z_rim)             # the rim is nearly 180-deg symmetric: the handle decides
        yaw, t, cost = min((fits[0], flip), key=handle_y)
    T = np.eye(4)
    T[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    T[:2, 3], T[2, 3] = t, z_rim
    T = T @ STRUCT_TO_FRAME                                          # rack structure -> the slot (marker) frame
    why = [w for w, bad in (('handle not seen', n_handle < MIN_HANDLE), ('few points', len(P) < MIN_POINTS)) if bad]
    return dict(T=T, conf=conf, cost=cost, n_top=len(P), n_handle=n_handle, refused=', '.join(why))


def estimate(det, rgb, depth, K, Tbc, z_rim):
    rack, plates, conf = det.masks(rgb)
    if rack is None:
        return None
    m = cv2.dilate(rack.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~plates
    P, _ = cloud(m, depth, K, Tbc)
    top = P[(P[:, 2] > z_rim - 0.03) & (P[:, 2] < z_rim + 0.02)]     # rails + wall tops around the rim height
    if len(top) < 50:
        return None
    ctr, yaw, cost, sz = rect_fit(top[:, :2])
    Hb, _ = cloud(m & blue(rgb), depth, K, Tbc)
    Hb = Hb[(Hb[:, 2] > z_rim - 0.08) & (Hb[:, 2] < z_rim + 0.03)]
    a = resolve_turn(ctr, yaw, sz, Hb[:, :2] if len(Hb) >= 20 else None)
    T = np.eye(4)
    T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    T[:2, 3], T[2, 3] = ctr, z_rim
    return dict(T=T, conf=conf, cost=cost, n_top=len(top), n_handle=len(Hb))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--out', default=str(ROOT / 'calibration/data/basket/basket_free_20261020'))
    a.add_argument('--show', type=int, default=6)
    a.add_argument('--method', choices=('rim', 'model'), default='model')
    a.add_argument('--since', default='20261019_1700', help='runs from this time on (current marker layout = 10/19 16:5x)')
    a = a.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    z_rim = yaml.safe_load((ROOT / 'calibration/config.yaml').read_text())['basket']['fixed_z_m']
    det = Detector()
    rows = []
    M = model_points()
    for i, (name, rgb, depth, K, Tbc, Tm, posj) in enumerate(f for f in frames() if f[0][:13] >= a.since):
        e = estimate_model(det, rgb, depth, K, Tbc, z_rim, posj, M) if a.method == 'model' else \
            estimate(det, rgb, depth, K, Tbc, z_rim)
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
                      ('test_1020', lambda r: r['name'] >= '20261020')):
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
    hx, hy = SIZE / 2
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
