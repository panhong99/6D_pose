"""(10/20 보스: 바구니에 아루코가 없다고 가정) basket pose from the fixed D455 RGB-D alone, no markers.

YOLOE prompt mask ('dish drying rack', plates cut out) + depth -> base-frame points, the arm (FK at the capture joints)
cut out -> top-view correlation of the basket model's upper structure over every yaw -> 3-DOF ICP (x, y, yaw; z =
config basket.fixed_z_m) -> the blue handle (basket -y) picks between the fit and its 180-degree turn.  A one-time offset
maps the rack structure to the slot frame.  Checked offline against the marker pose of 56 real frames
(experiments/basket_free/fit.py): 10/20 test 23/24 accepted, xy median 3.4 mm, max 7.3 mm, yaw max 2.1 deg.

    python calibration/basket_vision.py --share /dev/shm/fixed_latest.npz --T_base_cam X.json --posj j1 .. j6 --out res.json
        n fresh frames from the perception process's share -> the median pose, refused if the frames disagree.
Opt-in (config basket.source: vision); the marker path is unchanged."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ['dish drying rack', 'plate']             # rack mask; plates standing in it are cut out
MIN_HANDLE, MIN_POINTS = 20, 6000                   # below: refuse (look again with the arm out of the view)
# One-time offset of the rack structure (rl/assets/basket_full.obj walls/rails/handle) from the frame the slots are
# defined in: median of 25 marker frames on 10/19 17:00-19:30 (experiments/basket_free model_v5): fit = marker + (-3.0, -24.4) mm.
STRUCT_TO_FRAME = np.eye(4)
STRUCT_TO_FRAME[:2, 3] = [0.0030, 0.0244]


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
        step = np.linalg.norm(tn - t)
        yaw, t = np.arctan2(Rn[1, 0], Rn[0, 0]), tn
        if abs(dth) < 1e-6 and step < 1e-6:                       # (point-to-point creeps along wires: many small steps)
            break
    c, s = np.cos(yaw), np.sin(yaw)                               # (a Powell polish on the truncated distance was
    loc = np.c_[(S[:, :2] - t) @ np.array([[c, -s], [s, c]]), S[:, 2] - z_rim]   # tried: 10/20 xy median 3.4 -> 5.8 mm)
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


def estimate(det, rgb, depth, K, Tbc, z_rim, posj=None, M=None):
    """Whole-model fit: YOLOE rack mask - plates - arm -> points -> top-view correlation seeds -> ICP; the handle picks
    the 180-degree turn.  -> dict(T basket frame in base, ..., refused = '' or why) or None (no rack found)."""
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


def live(share, Tbc, posj, z_rim, n=5, gap_s=0.6, timeout_s=40.0, max_spread_mm=6.0, max_spread_deg=1.5):
    """n fresh frames of the fixed camera share -> median pose; refused if any frame is refused or they disagree."""
    det, M = Detector(), model_points()
    t0, last, res = time.time(), 0.0, []                           # frames newer than the loaded models only
    while len(res) < n:
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f'basket vision: {len(res)}/{n} frames in {timeout_s:.0f} s')
        try:
            z = np.load(share)
            t = float(z['t'])
        except Exception:
            time.sleep(0.1)
            continue
        if t <= last or t < t0:
            time.sleep(0.1)
            continue
        last = t
        e = estimate(det, z['rgb'], z['depth'], z['K'].astype(float), Tbc, z_rim, posj, M)
        res.append(e if e is not None else dict(refused='rack not found'))
        time.sleep(gap_s)
    why = sorted({r['refused'] for r in res if r.get('refused')})
    if why:
        return dict(ok=False, why='; '.join(why), frames=len(res))
    xy = np.array([r['T'][:2, 3] for r in res])
    yaw = np.array([np.arctan2(r['T'][1, 0], r['T'][0, 0]) for r in res])
    yaw = yaw[0] + (yaw - yaw[0] + np.pi) % (2 * np.pi) - np.pi
    mxy, myaw = np.median(xy, 0), float(np.median(yaw))
    spread_mm = float(np.abs(xy - mxy).max() * 1000)
    spread_deg = float(np.degrees(np.abs(yaw - myaw).max()))
    T = np.eye(4)
    T[:2, :2] = [[np.cos(myaw), -np.sin(myaw)], [np.sin(myaw), np.cos(myaw)]]
    T[:2, 3], T[2, 3] = mxy, z_rim
    ok = spread_mm <= max_spread_mm and spread_deg <= max_spread_deg
    return dict(ok=ok, why='' if ok else f'frames disagree: {spread_mm:.1f} mm, {spread_deg:.1f} deg', T=T.tolist(),
                spread_mm=spread_mm, spread_deg=spread_deg, frames=len(res),
                conf=float(np.median([r['conf'] for r in res])), n_points=int(np.median([r['n_top'] for r in res])))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--share', default='/dev/shm/fixed_latest.npz')
    a.add_argument('--T_base_cam', required=True, help='json with T_base_cam')
    a.add_argument('--posj', type=float, nargs=6, help='joints deg at the capture (the arm is cut out)')
    a.add_argument('--z_rim', type=float, required=True)
    a.add_argument('--n', type=int, default=5)
    a.add_argument('--out', required=True)
    a = a.parse_args()
    Tbc = np.asarray(json.loads(Path(a.T_base_cam).read_text())['T_base_cam'], float)
    try:
        r = live(a.share, Tbc, a.posj, a.z_rim, a.n)
    except Exception as exc:
        r = dict(ok=False, why=f'{type(exc).__name__}: {exc}')
    Path(a.out).write_text(json.dumps(r))
    print(json.dumps({k: v for k, v in r.items() if k != 'T'}), flush=True)


if __name__ == '__main__':
    main()
