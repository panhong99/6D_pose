"""(10/23 보스, 11/3 시연) bowls in the basket quadrants -> which quadrant is empty, seen by the fixed D455 + YOLOE.

The perception process owns the camera and shares its frame (/dev/shm/fixed_latest.npz, every 0.5 s).  This process
runs YOLOE ('bowl', 'plate', 'dish drying rack') on every new shared frame and
  - (with a window) draws the boxes and the four quadrants (green = empty, red = bowl, yellow = chosen) on the fixed
    view for the spectators, until the pipeline that started it ends (--until_pid; Q / ESC only without it);
  - writes the state (per quadrant: bowl points, empty; the bowls as boxes for the collision model) to --state.
The basket pose (for the quadrants) comes from the pipeline's basket_vision.json (--basket), read when it appears.

    python calibration/demo_watch.py --T_base_cam X.json --basket run/basket_vision.json --state /dev/shm/demo_quadrants.json
    python calibration/demo_watch.py --frame saved.npz --T_base_cam X.json --basket res.json --png out.png   (offline check)
Opt-in (pipeline --quadrant auto / --demo_view); nothing else reads it."""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ['bowl', 'plate', 'dish drying rack']
COLOURS = {'bowl': (255, 140, 0), 'plate': (60, 200, 255), 'dish drying rack': (180, 180, 180)}   # RGB
# basket frame (slot frame, z = rim): x robot side -> far side, y left seen from the robot (plate_plans.basket_gaps)
HALF = (0.265, 0.26)
SPLIT = (0.035, 0.0095)                            # x: the clip between near / far, y: between pin pairs A (+0.129) / B (-0.11)
QUADRANTS = {'Q1': ('far', 'right'), 'Q2': ('far', 'left'), 'Q3': ('near', 'left'), 'Q4': ('near', 'right')}


class Detector:
    def __init__(self):
        from ultralytics import YOLOE
        self.m = YOLOE(str(ROOT / 'yoloe-v8l-seg.pt'))
        self.m.set_classes(PROMPTS, self.m.get_text_pe(PROMPTS))

    def __call__(self, rgb, conf=0.15):
        """-> [(name, conf, (x0, y0, x1, y1), mask bool HxW), ...]"""
        r = self.m.predict(rgb[:, :, ::-1].copy(), conf=conf, verbose=False, retina_masks=True)[0]
        if r.boxes is None or not len(r.boxes):
            return []
        masks = r.masks.data.cpu().numpy() > 0.5 if r.masks is not None else [None] * len(r.boxes)
        return [(PROMPTS[int(k)], float(c), tuple(int(v) for v in b), m) for k, c, b, m in
                zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy(), masks)]


def quadrant_of(xy):
    """Basket-frame xy -> 'Q1'..'Q4', or None outside the basket."""
    x, y = xy
    if abs(x) > HALF[0] or abs(y) > HALF[1]:
        return None
    fn, lr = 'near' if x < SPLIT[0] else 'far', 'left' if y > SPLIT[1] else 'right'
    return next(q for q, v in QUADRANTS.items() if v == (fn, lr))


def quadrant_corners(q):
    """Corners (basket frame, rim height) of a quadrant."""
    fn, lr = QUADRANTS[q]
    x = (-HALF[0], SPLIT[0]) if fn == 'near' else (SPLIT[0], HALF[0])
    y = (SPLIT[1], HALF[1]) if lr == 'left' else (-HALF[1], SPLIT[1])
    return np.array([[x[0], y[0], 0], [x[1], y[0], 0], [x[1], y[1], 0], [x[0], y[1], 0]])


def bowl_quadrants(dets, depth, K, Tbc, Tb, min_conf=0.25, min_points=80, share=0.25):
    """Bowls -> {quadrant: bowl points there}; a bowl counts in every quadrant holding >= `share` of its points."""
    from basket_vision import cloud
    out = {q: 0 for q in QUADRANTS}
    Tbi = np.linalg.inv(Tb)
    for name, c, _, mk in dets:
        if name != 'bowl' or c < min_conf or mk is None:
            continue
        P, _ = cloud(cv2.erode(mk.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool), depth, K, Tbc)
        P = P[P[:, 2] > 0.0]
        if len(P) < min_points:
            continue
        L = P @ Tbi[:3, :3].T + Tbi[:3, 3]
        qs = [quadrant_of(p[:2]) for p in L]
        for q in QUADRANTS:
            n = sum(1 for v in qs if v == q)
            if n >= share * len(qs):
                out[q] += n
    return out


def bowl_boxes(dets, depth, K, Tbc, posj=None):
    """Bowls -> oriented boxes (name, centre, size, R, conf, n) in base, like basket_vision.obstacle_boxes."""
    from types import SimpleNamespace
    from basket_vision import obstacle_boxes
    det = SimpleNamespace(plate_masks=[(m, c) for n, c, _, m in dets if n == 'bowl' and m is not None])
    return [(f'bowl_{i}',) + tuple(b[1:]) for i, b in enumerate(obstacle_boxes(det, None, depth, K, Tbc, posj))]


def choose(counts, allowed=('Q3', 'Q4'), min_points=150):
    """The empty quadrants in `allowed` order -> the first one, or None."""
    empty = [q for q in allowed if counts.get(q, 0) < min_points]
    return (empty[0] if empty else None), empty


def draw(rgb, dets, K, Tbc, Tb, counts, chosen, note=''):
    img = rgb.copy()
    for name, c, (x0, y0, x1, y1), _ in dets:
        col = COLOURS[name]
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
        cv2.putText(img, f'{name} {c:.2f}', (x0, max(14, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    if Tb is not None:
        Tcb = np.linalg.inv(Tbc) @ Tb
        for q in QUADRANTS:
            P = quadrant_corners(q) @ Tcb[:3, :3].T + Tcb[:3, 3]
            uv = (P @ K.T)[:, :2] / P[:, 2:3]
            full = counts.get(q, 0) >= 150
            col = (255, 220, 0) if q == chosen else (230, 40, 40) if full else (40, 210, 60)
            cv2.polylines(img, [uv.astype(np.int32)], True, col, 4 if q == chosen else 2)
            for i, s in enumerate((q, 'bowl' if full else 'empty')):
                (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                u, v = uv.mean(0).astype(int)
                cv2.putText(img, s, (u - w // 2, v - 4 + i * (h + 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    lines = [f'empty: {" ".join(q for q in QUADRANTS if counts.get(q, 0) < 150) or "-"}',
             f'target: {chosen or "-"}'] + ([note] if note else [])
    for i, s in enumerate(lines):
        cv2.putText(img, s, (10, 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return img


def load_basket(path):
    try:
        r = json.loads(Path(path).read_text())
        return np.asarray(r['T'], float) if r.get('ok') else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_state(path, state):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(state))
    os.replace(tmp, path)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--share', default='/dev/shm/fixed_latest.npz')
    a.add_argument('--frame', help='offline: one saved frame (npz rgb/depth/K) instead of the share')
    a.add_argument('--T_base_cam', required=True)
    a.add_argument('--basket', help="basket_vision.json (T); quadrants drawn once it exists")
    a.add_argument('--posj', type=float, nargs=6, help='joints deg (the arm is cut out of the bowl boxes)')
    a.add_argument('--state', default='/dev/shm/demo_quadrants.json')
    a.add_argument('--allowed', nargs='+', default=['Q3', 'Q4'])
    a.add_argument('--chosen_file', help='the pipeline writes the quadrant it took here (drawn yellow)')
    a.add_argument('--until_pid', type=int, help='exit when this process ends (the pipeline)')
    a.add_argument('--no_window', action='store_true')
    a.add_argument('--png', help='offline: save the overlay here')
    a = a.parse_args()
    Tbc = np.asarray(json.loads(Path(a.T_base_cam).read_text())['T_base_cam'], float)
    det, last, n = Detector(), 0.0, 0
    win = 'demo: fixed camera'
    if not a.no_window and not a.png:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 960)
    while True:
        if a.until_pid and not Path(f'/proc/{a.until_pid}').exists():
            break
        try:
            z = np.load(a.frame or a.share)
            t = float(z['t']) if 't' in z else 0.0
        except (OSError, ValueError, KeyError):
            time.sleep(0.05)
            continue
        if not a.frame and t <= last:
            if not a.no_window and (cv2.waitKey(30) & 0xff) in (ord('q'), 27) and not a.until_pid:
                break
            continue
        last = t
        rgb, depth, K = z['rgb'], z['depth'], z['K'].astype(float)
        dets = det(rgb)
        Tb = load_basket(a.basket) if a.basket else None
        counts = bowl_quadrants(dets, depth, K, Tbc, Tb) if Tb is not None else {}
        chosen_by_rule, empty = choose(counts, a.allowed) if Tb is not None else (None, [])
        taken = Path(a.chosen_file).read_text().strip().split(':')[0] if a.chosen_file and Path(a.chosen_file).exists() else None
        n += 1
        write_state(a.state, dict(t=t, frame=n, basket=None if Tb is None else Tb.tolist(), counts=counts, empty=empty,
                                  chosen=chosen_by_rule, bowls=bowl_boxes(dets, depth, K, Tbc, a.posj),
                                  dets=[(nm, round(c, 3), b) for nm, c, b, _ in dets]))
        img = draw(rgb, dets, K, Tbc, Tb, counts, taken or chosen_by_rule,
                   '' if Tb is not None else 'basket: measuring ...')
        if a.png:
            cv2.imwrite(a.png, img[:, :, ::-1])
            print(json.dumps(dict(counts=counts, empty=empty, chosen=chosen_by_rule, dets=len(dets))), flush=True)
            break
        if not a.no_window:
            cv2.imshow(win, img[:, :, ::-1])
            if (cv2.waitKey(1) & 0xff) in (ord('q'), 27) and not a.until_pid:   # the pipeline's view stays to the end
                break
        if a.frame:
            break
    if not a.no_window and not a.png:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
