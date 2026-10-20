"""(10/20 보스, BQ) the sim-trained tine network on real wrist photos.

    ~/miniconda3/envs/foundationpose/bin/python experiments/slot_cnn/eval.py --ckpt calibration/data/slot_cnn/run_v1/last.pt

Test = Codex's v4 labels (4 photos, 84 tines; root/tip pixel points, null where hidden).  A prediction is matched to a
label of its kind within --tol px (original 1280 x 720 pixels, one-to-one, Hungarian).  Each photo labels one quadrant's
two rows only, so precision counts predictions inside that labelled area (label hull + 30 px); recall counts all labels.
Unlabelled real photos (pre-insert, insert height) get drawn overlays only."""
import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402
from train import KEYS, STRIDE, TineNet, to_tensor  # noqa: E402

SCALE = D.OW / D.W                                   # 0.75


def predict(net, bgr, th=0.3):
    """Original-pixel keypoints per kind: {key: (n, 3) [u, v, score]}."""
    x = cv2.resize(bgr, (D.OW, D.OH - 4), interpolation=cv2.INTER_AREA)
    x = np.r_[x, np.zeros((4, D.OW, 3), np.uint8)]
    with torch.no_grad():
        o = net(to_tensor(x)[None].cuda()).float()[0]
    out = {}
    for c, k in enumerate(KEYS):
        h = torch.sigmoid(o[c])
        peak = (h == torch.nn.functional.max_pool2d(h[None, None], 3, 1, 1)[0, 0]) & (h > th)
        iy, ix = torch.nonzero(peak, as_tuple=True)
        off = torch.sigmoid(o[2 + 2 * c:4 + 2 * c])[:, iy, ix]
        u = (ix.float() + off[0]) * STRIDE / SCALE
        v = (iy.float() + off[1]) * STRIDE / SCALE
        out[k] = torch.stack([u, v, h[iy, ix]], 1).cpu().numpy()
    return out


def match(pred, gt, tol):
    if not len(pred) or not len(gt):
        return [], np.zeros(0)
    d = np.linalg.norm(pred[:, None, :2] - gt[None], axis=2)
    r, c = linear_sum_assignment(np.where(d <= tol, d, 1e6))
    ok = d[r, c] <= tol
    return list(zip(r[ok], c[ok])), d[r[ok], c[ok]]


def evaluate(net, out, th, tol):
    rows, per = [], []
    for img_path, Tcb, pins, q in D.real_tests():
        img = cv2.imread(img_path)
        pr = predict(net, img, th)
        gts = {'root': np.array([p[1] for p in pins if p[1]], float), 'tip': np.array([p[2] for p in pins if p[2]], float)}
        allp = np.r_[gts['root'], gts['tip']]
        hull = cv2.convexHull(allp.astype(np.float32))
        region = np.zeros(img.shape[:2], np.uint8)
        cv2.fillConvexPoly(region, hull.astype(np.int32), 1)
        region = cv2.dilate(region, np.ones((61, 61), np.uint8))
        vis = img.copy()
        rec = dict(photo=Path(img_path).name, label_quadrant=q)
        for k, col in (('root', (255, 120, 0)), ('tip', (0, 0, 255))):
            p, g = pr[k], gts[k]
            inside = p[region[np.clip(p[:, 1].astype(int), 0, 719), np.clip(p[:, 0].astype(int), 0, 1279)] > 0] if len(p) else p
            m, d = match(inside, g, tol)
            rec[k] = dict(labels=len(g), found=len(m), pred_in_area=len(inside), recall=len(m) / max(1, len(g)),
                          precision=len(m) / max(1, len(inside)), err_px_median=float(np.median(d)) if len(d) else None)
            for x, y in g:
                cv2.circle(vis, (int(x), int(y)), 6, (0, 220, 0), 1)
            for x, y, s in p:
                cv2.circle(vis, (int(round(x)), int(round(y))), 2, col, -1)
        cv2.polylines(vis, [hull.astype(np.int32)], True, (200, 200, 200), 1)
        cv2.imwrite(str(out / f'real_{q}.jpg'), vis)
        rows.append(rec)
    tot = {}
    for k in KEYS:
        L = sum(r[k]['labels'] for r in rows)
        Fd = sum(r[k]['found'] for r in rows)
        P = sum(r[k]['pred_in_area'] for r in rows)
        tot[k] = dict(labels=L, found=Fd, recall=Fd / max(1, L), precision=Fd / max(1, P))
    return rows, tot


def overlays(net, out, th, k=12):
    files = sorted(glob.glob(str(D.DATA / 'real_pipeline/*/wrist_pre.raw.png')))[-6:] + \
        sorted(glob.glob(str(D.DATA / 'real_pipeline/2026102*/wrist_above.raw.png')))[-6:]
    tiles = []
    for f in files[:k]:
        img = cv2.imread(f)
        pr = predict(net, img, th)
        for key, col in (('root', (255, 120, 0)), ('tip', (0, 0, 255))):
            for x, y, s in pr[key]:
                cv2.circle(img, (int(round(x)), int(round(y))), 3, col, -1)
        tiles.append(cv2.resize(img, (640, 360)))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles) - len(tiles) % 3, 3)]
    cv2.imwrite(str(out / 'unlabelled_overlays.jpg'), np.vstack(rows))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--ckpt', required=True)
    a.add_argument('--out')
    a.add_argument('--th', type=float, default=0.3)
    a.add_argument('--tol', type=float, default=8.0)
    a = a.parse_args()
    out = Path(a.out or Path(a.ckpt).parent / f'eval_{Path(a.ckpt).stem}')
    out.mkdir(parents=True, exist_ok=True)
    net = TineNet(pretrained=False).cuda().eval()
    net.load_state_dict(torch.load(a.ckpt, map_location='cuda'))
    sweep = {}
    for th in (0.1, 0.2, 0.3, 0.4, 0.5):
        _, tot = evaluate(net, out, th, a.tol)
        sweep[th] = tot
    rows, tot = evaluate(net, out, a.th, a.tol)
    overlays(net, out, a.th)
    preds = {q: predict(net, cv2.imread(p), a.th) for p, _, _, q in D.real_tests()}
    numbers = {k: number_check({q: v[k] for q, v in preds.items()}, trials=100, kind=k) for k in KEYS}
    numbers['both'] = number_check(preds, trials=100, kind='both')
    numbers = {k: {q: {kk: vv for kk, vv in v.items() if kk != 'info'} for q, v in r.items()} for k, r in numbers.items()}
    res = dict(ckpt=a.ckpt, th=a.th, tol_px=a.tol, total=tot, photos=rows, threshold_sweep=sweep, tine_numbers=numbers)
    (out / 'metrics.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(numbers, indent=1))
    print(json.dumps(dict(total=tot, sweep={k: {kk: (round(v['recall'], 3), round(v['precision'], 3)) for kk, v in s.items()}
                                            for k, s in sweep.items()}), indent=1))



def number_check(pred_roots=None, trials=200, xy_mm=4.0, yaw_deg=1.0, seed=0, kind='root'):
    """Tine numbers from roots + a coarse basket pose (marker pose moved by vision-like errors, experiments/basket_free:
    xy sd 4 mm capped 7.3, yaw sd 1 deg capped 2.1) vs the v4 numbers.  pred_roots None = the labelled roots (geometry
    alone); else {label quadrant: (n, 2) CNN roots}, each matched to a label first."""
    import slots as S
    facts = S.basket_facts('tip' if kind == 'both' else kind)
    rng = np.random.default_rng(seed)
    col = 1 if kind == 'root' else 2
    uv = None
    res = {}
    for img_path, Tcb, pins, q in D.real_tests():
        meta = json.loads(Path(json.loads((D.V4 / f'{q}.json').read_text())['capture_metadata']).read_text())
        Tbc = np.asarray(meta['T_base_flange']) @ np.asarray(meta['T_flange_cam'])
        Tbb = np.asarray(meta['T_base_basket'])
        lab = [(int(p[0][1:]), np.array(p[col])) for p in pins if p[col]]
        if kind == 'both':
            pass
        elif pred_roots is None:
            uv, truth = np.array([p for _, p in lab]), np.array([n for n, _ in lab])
        else:                                                # every detection is numbered; matched ones are scored
            uv = pred_roots[q][:, :2]
            m, _ = match(uv, np.array([p for _, p in lab]), 8.0)
            truth = np.full(len(uv), -1)
            for i, j in m:
                truth[i] = lab[j][0]
        if kind == 'both':                                   # tips + roots whose tip was missed (moved to the tip plane)
            labs = {kk: [(int(p[0][1:]), np.array(p[c])) for p in pins if p[c]] for kk, c in (('root', 1), ('tip', 2))}
            tr = {}
            for kk in ('root', 'tip'):
                uv_k = pred_roots[q][kk][:, :2]
                m, _ = match(uv_k, np.array([p for _, p in labs[kk]]), 8.0)
                t_ = np.full(len(uv_k), -1)
                for i, j in m:
                    t_[i] = labs[kk][j][0]
                tr[kk] = (uv_k, t_)
            fr, ft = S.basket_facts('root'), S.basket_facts('tip')
        right = wrong = refused = 0
        for _ in range(trials):
            e = np.clip(rng.normal(0, xy_mm / 1000, 2), -0.0073, 0.0073)
            a = np.radians(np.clip(rng.normal(0, yaw_deg), -2.1, 2.1))
            dT = np.eye(4)
            dT[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
            dT[:2, 3] = e
            Tc = Tbb @ dT                                                   # coarse (vision) basket pose
            if kind == 'both':
                Pr = S.to_rail_plane(tr['root'][0], Tbc, Tc, fr['rail_z']) if len(tr['root'][0]) else np.zeros((0, 3))
                Pt = S.to_rail_plane(tr['tip'][0], Tbc, Tc, ft['rail_z']) if len(tr['tip'][0]) else np.zeros((0, 3))
                P, truth = S.merge_root_tip(Pr, Pt, fr, ft, q, return_src=True)
                truth = np.array([tr['tip'][1][i] if src == 't' else tr['root'][1][i] for src, i in truth])
            else:
                P = S.to_rail_plane(uv, Tbc, Tc, facts['rail_z'])
            away = (Tc[:3, 0] @ np.r_[Tc[:2, 3], 0.0]) >= 0
            rows, info = S.number_rows(P, facts, q, away)
            for row, lst in rows.items():
                if not info[row]['confident']:
                    refused += len(lst)
                    continue
                for num, i in lst:
                    if truth[i] < 0:                         # unmatched detection (other row / false): not scored
                        continue
                    right += int(num == truth[i])
                    wrong += int(num != truth[i])
        res[q] = dict(roots=len(uv) if uv is not None else -1, right=right, wrong=wrong, refused=refused,
                      acc_of_answered=right / max(1, right + wrong), info=info)
    return res


if __name__ == '__main__':
    main()
