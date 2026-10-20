"""(10/20 보스, BQ) tine roots in a wrist photo -> numbered tines and slot (pin) poses in the robot base frame.

Inputs: root pixels (CNN or labels), the camera pose at the photo (FK @ T_flange_cam), a COARSE basket pose (no markers:
experiments/basket_free/fit.py, 3-7 mm / <= 2 deg) and the basket's fixed facts (rl/basket.yaml full: rail height, row
y, clip x, tines per row).  The coarse pose only says which row a root is on and where the clip end is; the slot
positions come from the photo.

Numbering = the pipeline's: tine 1 = the robot-far end of the row, pin p = the gap between tines p and p+1
(calibration/plate_plans.pin_gap).  Index from the clip end: k = round((|x - clip| - first) / pitch) + 1, pitch = median
spacing in the photo; the result is refused (confident False) if the clip-to-first-tine offset it implies is more than
pitch/2 - 3 mm from the model's, or two roots claim one number."""
import numpy as np

import data as D


# (10/20 night) keypoint-free zones in the basket frame: (min distance from the clip, max |x|).  The clip and the end
# walls gave false tips (run_v2: clip side 8-23 mm, walls |x| 224-235 mm); real tips sit 30-39 mm from the clip and at
# |x| <= 213 mm on the near half, 230 mm on the far half (Q1 7 tines at 24 mm) (v4 labels).  NOTE: set by looking at the 4 test photos, so the test numbers below are optimistic.
KEEP_OUT = {'tip': (0.025, {True: 0.232, False: 0.218}), 'root': (0.010, {True: 0.240, False: 0.235})}   # far / near half


def basket_facts(kind='root'):
    """Model facts for one keypoint kind (root on the rail / tip): per row y, count, side; per quadrant the clip -> first
    tine distance; the keypoint plane height (basket frame)."""
    parts, B = D.basket_parts()
    f = B['full']
    tines = D.tine_table(parts, B)
    rail = B['pins']['rail_z']
    cx = f.get('clip_x', 0.0)
    rows, first, zs = {}, {}, []
    for t in tines:
        p = D.root_tip(t, rail)[0 if kind == 'root' else 1]
        zs.append(p[2])
        rows.setdefault((t['q'], t['row']), []).append(p)
    out = {}
    for (q, row), P in rows.items():
        P = np.array(P)
        out[(q, row)] = dict(y=float(np.median(P[:, 1])), n=len(P), far=q in ('Q1', 'Q2'))
        first[q] = float(np.abs(P[:, 0] - cx).min())
    return dict(rows=out, clip_x=cx, rail_z=float(np.median(zs)), floor_z=f['floor_z'], first=first,
                pitch=f.get('tine_pitch', {}), kind=kind)


def to_rail_plane(uv, T_base_cam, T_base_basket, z_rail_basket, K=D.K_WRIST):
    """Pixels -> points on the rail plane, basket frame."""
    uv = np.asarray(uv, float).reshape(-1, 2)
    ray_c = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))]
    T = np.linalg.inv(T_base_basket) @ T_base_cam                        # camera in the basket frame
    o, d = T[:3, 3], ray_c @ T[:3, :3].T
    s = (z_rail_basket - o[2]) / d[:, 2]
    return o + d * s[:, None]


def number_rows(P, facts, quadrant, away=True, row_tol=0.025):
    """Basket-frame roots (n, 3) -> {row: [(tine number, point index)]}, info."""
    cx = facts['clip_x']
    out, info = {}, {}
    ys = [facts['rows'][(quadrant, r)]['y'] for r in (0, 1)]
    side = (P[:, 0] > cx) == facts['rows'][(quadrant, 0)]['far']
    near, far = KEEP_OUT[facts.get('kind', 'root')]                         # no tine keypoint at the clip or the walls
    far = far[facts['rows'][(quadrant, 0)]['far']]
    zone = (np.abs(P[:, 0] - cx) >= near) & (np.abs(P[:, 0]) <= far)
    cand = np.nonzero(side & zone & (np.abs(P[:, 1] - np.mean(ys)) < abs(ys[1] - ys[0]) / 2 + 0.035))[0]
    groups = {0: [], 1: []}                                                 # the two rows: split the seen points in y at
    if len(cand):                                                           # their widest gap (the real rows sit 1-2 cm
        o = cand[np.argsort(P[cand, 1])]                                    # off the model's, 10/19 v4 labels)
        g = np.diff(P[o, 1])
        cut = int(np.argmax(g)) + 1 if len(g) and g.max() > 0.04 else (len(o) if P[o, 1].mean() < np.mean(ys) else 0)
        lo_row = int(np.argmin(ys))
        groups[lo_row], groups[1 - lo_row] = list(o[:cut]), list(o[cut:])
    for row in (0, 1):
        R = facts['rows'][(quadrant, row)]
        sel = np.array(groups[row], int)
        if not len(sel):
            continue
        d = np.abs(P[sel, 0] - cx)
        o = np.argsort(d)
        sel, d = sel[o], d[o]
        p0 = float(facts['pitch'].get(quadrant, 0.0175))
        best = None                                                         # periodic lattice through the most points
        for pitch in np.arange(0.8 * p0, 1.25 * p0 + 1e-9, 0.00025):
            for ph in np.arange(0.0, pitch, 0.0005):
                r = (d - ph) / pitch
                res = np.abs(r - np.round(r)) * pitch
                inl = res < 0.0045
                sc = (inl.sum(), -res[inl].sum())
                if best is None or sc > best[0]:
                    best = (sc, pitch, ph, inl)
        _, pitch, ph, inl = best
        sel, d = sel[inl], d[inl]
        kr = np.round((d - ph) / pitch).astype(int)
        keep = np.r_[True, np.diff(kr) > 0]                               # one point per lattice place
        sel, d, kr = sel[keep], d[keep], kr[keep]
        first = facts['first'][quadrant]
        ambiguous = False
        if kr[-1] - kr[0] + 1 > R['n']:                                     # longer than the row (a false point at an end):
            wins = [(np.sum((kr >= w) & (kr < w + R['n'])),                # the n-long window with the most points, then the
                     -abs(ph + pitch * w - first), w) for w in range(kr[0], kr[-1] - R['n'] + 2)]   # model's first offset
            top = max(c for c, _, _ in wins)
            if sum(c == top for c, _, _ in wins) > 1:                       # two windows hold as many points: which end
                ambiguous = True                                            # is false cannot be told -> no answer
            w = max(wins)[2]
            m = (kr >= w) & (kr < w + R['n'])
            sel, d, kr = sel[m], d[m], kr[m]
        whole = kr[-1] - kr[0] + 1 == R['n'] and len(sel) >= R['n'] - 1     # both row ends seen: count them
        k = kr - kr[0] + 1 if whole else np.round((d - first) / pitch).astype(int) + 1
        off = float(np.median(d - pitch * (k - 1)))
        # (10/20 night) the model's clip -> first-tine distance is off by up to 17 mm (v4 labels) and real tines sit up to
        # +-5 mm off a regular pitch: an anchor from the model is a guess, so only a row seen end to end is trusted
        ok = whole and not ambiguous and (k >= 1).all() and (k <= R['n']).all()
        clip_is_far = (not R['far']) == away                               # near quadrants: the clip is the robot-far end
        num = k if clip_is_far else R['n'] + 1 - k
        out[row] = sorted(zip(num.tolist(), sel.tolist()))
        info[row] = dict(pitch_mm=pitch * 1000, first_mm=off * 1000, confident=bool(ok), whole_row=bool(whole), found=len(sel),
                         of=R['n'])
    # the two rows of a plate pair carry tine k at the same x (v4 labels: L1 and R1 both 4.6 mm from the clip): the same
    # numbering in both rows must put them there, else neither row is trusted
    if len(info) == 2:
        a, b = info[0], info[1]
        same = abs(a['first_mm'] - b['first_mm']) < 6.0 and abs(a['pitch_mm'] - b['pitch_mm']) < 1.5
        both = a['confident'] and b['confident']                            # both rows seen end to end, and agreeing
        for r in (0, 1):
            info[r]['rows_agree'] = bool(same)
            info[r]['confident'] = bool(both and same)
    else:
        for r in info:
            info[r]['rows_agree'] = False
            info[r]['confident'] = False                                    # one row alone is not trusted
    return out, info


def slot_poses(P, rows, facts, quadrant, T_base_basket):
    """Pin p (gap between tines p and p+1) -> base-frame (x, y, z on the basket floor, yaw deg) from a line fit per row."""
    fits = {}
    for row, lst in rows.items():
        if len(lst) < 3:
            continue
        num = np.array([n for n, _ in lst], float)
        xy = P[[i for _, i in lst], :2]
        A = np.c_[np.ones_like(num), num]
        cx, *_ = np.linalg.lstsq(A, xy[:, 0], rcond=None)                  # x(num), y(num): straight row
        cy, *_ = np.linalg.lstsq(A, xy[:, 1], rcond=None)
        fits[row] = (cx, cy)
    if not fits:
        return {}
    n = facts['rows'][(quadrant, 0)]['n']
    out = {}
    for p in range(1, n):
        pts = [np.array([cx[0] + cx[1] * (p + 0.5), cy[0] + cy[1] * (p + 0.5)]) for cx, cy in fits.values()]
        c = np.mean(pts, axis=0)
        dirs = [np.array([cx[1], cy[1]]) for cx, cy in fits.values()]
        dv = np.mean([d / np.linalg.norm(d) * np.sign(d[0] or 1) for d in dirs], axis=0)
        pb = T_base_basket @ np.r_[c, facts['floor_z'], 1.0]
        db = T_base_basket[:3, :3] @ np.r_[dv, 0.0]
        out[p] = dict(xyz=pb[:3].tolist(), yaw_deg=float(np.degrees(np.arctan2(db[1], db[0]))))
    return out


def merge_root_tip(Pr, Pt, facts_r, facts_t, quadrant, dy_tol=0.008, dx_tol=0.012, return_src=False):
    """Tips (tip plane) + the roots whose tip was not found, moved by the root -> tip offset measured in this photo
    (median of root/tip pairs; the model's lean is the first guess).  Basket frame points, tip plane.
    (10/20 night: run_v1 ep08 numbers no better than tips alone, Q3 11 wrong: not used for the answer.)"""
    exp = np.median([facts_t['first'][quadrant]]) - np.median([facts_r['first'][quadrant]])
    sign = 1.0 if facts_t['rows'][(quadrant, 0)]['far'] else -1.0          # away from the clip on both halves
    exp_dx = sign * exp
    src_t = [('t', i) for i in range(len(Pt))]
    if not len(Pr):
        return (Pt, src_t) if return_src else Pt
    if not len(Pt):
        P = np.c_[Pr[:, 0] + exp_dx, Pr[:, 1], np.full(len(Pr), facts_t['rail_z'])]
        return (P, [('r', i) for i in range(len(Pr))]) if return_src else P
    pairs, used = [], np.zeros(len(Pt), bool)
    for i, r in enumerate(Pr):
        d = Pt[:, :2] - r[:2]
        ok = (np.abs(d[:, 1]) < dy_tol) & (np.abs(d[:, 0] - exp_dx) < dx_tol)
        if ok.any():
            j = int(np.argmin(np.where(ok, np.abs(d[:, 0] - exp_dx), np.inf)))
            pairs.append((i, j, d[j, 0]))
            used[j] = True
    dx = float(np.median([p[2] for p in pairs])) if len(pairs) >= 3 else exp_dx
    paired = {p[0] for p in pairs}
    rest = [i for i in range(len(Pr)) if i not in paired]
    extra = np.array([[Pr[i, 0] + dx, Pr[i, 1], facts_t['rail_z']] for i in rest]).reshape(-1, 3)
    P = np.r_[Pt, extra]
    return (P, src_t + [('r', i) for i in rest]) if return_src else P
