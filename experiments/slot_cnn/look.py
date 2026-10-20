"""(10/20 보스, BQ) one wrist photo -> numbered slots (pins) with their base-frame pose, next to where the basket model
puts them.  Read-only: no robot, no camera; the photo and its metadata json are saved ones.

    ~/miniconda3/envs/foundationpose/bin/python experiments/slot_cnn/look.py PHOTO.raw.png [--quadrant Q4] \
        [--ckpt A.pt,B.pt] [--basket vision.json] [--out DIR]

PHOTO.json next to the photo gives T_base_flange, T_flange_cam and the basket pose (marker or vision; --basket
overrides it with a basket_vision.json result).  Answers only when both tine rows of the quadrant are seen end to end
and agree (experiments/slot_cnn/slots.py); otherwise says why and gives no numbers.  Pin p = the gap between tines p
and p+1 counted from the robot-far end (calibration/plate_plans.pin_gap)."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402
import eval as E  # noqa: E402
import slots as S  # noqa: E402
from train import TineNet  # noqa: E402

CKPT = 'calibration/data/slot_cnn/run_v1/ep08.pt,calibration/data/slot_cnn/run_v2/last.pt'   # 10/21 best tips: ensemble


def quadrant_in_view(P_all, facts, Tbb_c):
    """The quadrant whose rows hold the most tips (when --quadrant is not given)."""
    best = None
    for q in ('Q1', 'Q2', 'Q3', 'Q4'):
        rows, info = S.number_rows(P_all, facts, q, away=(Tbb_c[:3, 0] @ np.r_[Tbb_c[:2, 3], 0.0]) >= 0)
        n = sum(i['found'] for i in info.values())
        if best is None or n > best[0]:
            best = (n, q)
    return best[1]


def main():
    a = argparse.ArgumentParser()
    a.add_argument('photo')
    a.add_argument('--quadrant', choices=('Q1', 'Q2', 'Q3', 'Q4'))
    a.add_argument('--ckpt', default=CKPT)
    a.add_argument('--basket', help='basket_vision.json (ok, T) instead of the pose in the photo json')
    a.add_argument('--th', type=float, default=0.3)
    a.add_argument('--out')
    a = a.parse_args()
    root = Path(__file__).resolve().parents[2]
    photo = Path(a.photo)
    meta = json.loads(Path(str(photo).replace('.raw.png', '.json')).read_text())
    Tbc = np.asarray(meta['T_base_flange']) @ np.asarray(meta['T_flange_cam'])
    Tbb = np.asarray(json.loads(Path(a.basket).read_text())['T'] if a.basket else meta['T_base_basket'], float)
    nets = []
    for c in a.ckpt.split(','):
        n_ = TineNet(pretrained=False).cuda().eval()
        n_.load_state_dict(torch.load(str(root / c) if not Path(c).is_absolute() else c, map_location='cuda'))
        nets.append(n_)
    img = cv2.imread(str(photo))
    pr = E.predict(nets, img, a.th)['tip']
    facts = S.basket_facts('tip')
    P = S.to_rail_plane(pr[:, :2], Tbc, Tbb, facts['rail_z']) if len(pr) else np.zeros((0, 3))
    q = a.quadrant or quadrant_in_view(P, facts, Tbb)
    away = bool((Tbb[:3, 0] @ np.r_[Tbb[:2, 3], 0.0]) >= 0)
    rows, info = S.number_rows(P, facts, q, away)
    ok = all(i['confident'] for i in info.values()) and len(info) == 2
    res = dict(photo=str(photo), quadrant=q, answer=bool(ok), rows={int(r): {k: (float(v) if isinstance(v, (np.floating, float))
                                                                               else v) for k, v in i.items()}
                                                                    for r, i in info.items()})
    if ok:                                                               # slots sit at the roots: tips moved back by
        fr = S.basket_facts('root')                                      # the root -> tip offset seen in this photo
        rr = E.predict(nets, img, a.th)['root']
        Pr = S.to_rail_plane(rr[:, :2], Tbc, Tbb, fr['rail_z']) if len(rr) else np.zeros((0, 3))
        dx, n_pair = S.root_tip_offset(Pr, P)
        if dx is None:
            sign = 1.0 if facts['rows'][(q, 0)]['far'] else -1.0
            dx = sign * (facts['first'][q] - fr['first'][q])
        res['root_to_tip_dx_mm'] = round(dx * 1000, 1)
        res['root_to_tip_from'] = f'{n_pair} root/tip pairs in this photo' if n_pair else 'basket model lean'
        Pslot = P.copy()
        Pslot[:, 0] -= dx
        pins = S.slot_poses(Pslot, rows, facts, q, Tbb)
        sys.path.insert(0, str(root / 'calibration'))
        import plate_plans as pp
        gaps = pp.basket_gaps()[q]
        model = {}
        for p in pins:                                                   # the basket model's gap for the same pin
            g = gaps[pp.pin_gap(Tbb, q, p)]
            model[p] = (Tbb @ np.r_[g[:2], facts['floor_z'], 1.0])[:3]
        res['pins'] = {int(p): dict(xyz_m=np.round(v['xyz'], 4).tolist(), yaw_deg=round(v['yaw_deg'], 2),
                                    model_minus_seen_mm=np.round((model[p] - np.asarray(v['xyz']))[:2] * 1000, 1).tolist())
                       for p, v in pins.items()}
    out = Path(a.out) if a.out else photo.parent / 'slot_cnn_look'
    out.mkdir(parents=True, exist_ok=True)
    vis = img.copy()
    for x, y, s in pr:
        cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)
    for r, lst in rows.items():
        for num, i in lst:
            cv2.putText(vis, str(num), (int(pr[i, 0]) + 4, int(pr[i, 1]) - 4), 0, 0.45,
                        (0, 200, 0) if ok else (0, 160, 255), 1)
    if ok:
        Tcb = np.linalg.inv(Tbc)
        for p, v in res['pins'].items():
            uv, z = D.project(np.asarray(v['xyz_m'])[None], Tcb)
            cv2.drawMarker(vis, tuple(uv[0].astype(int)), (255, 0, 255), cv2.MARKER_CROSS, 10, 2)
            cv2.putText(vis, f'pin{p}', tuple(uv[0].astype(int) + [6, 12]), 0, 0.4, (255, 0, 255), 1)
    stem = photo.name.replace('.raw.png', '')
    cv2.imwrite(str(out / f'{stem}_slots.jpg'), vis)
    (out / f'{stem}_slots.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
