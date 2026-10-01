"""Offline forensic report for real eye-to-hand samples (read-only, no camera, no robot).

    conda run -n foundationpose python calibration/analyze_handeye_samples.py \
        [--samples calibration/data/handeye_samples.json] [--out calibration/data/handeye_forensics.json]

Why the checks below work without knowing T_base_cam or T_flange_marker:
  * For any two samples i, j the robot motion A = inv(Tbf_i) Tbf_j and the marker motion
    B = inv(Tcm_i) Tcm_j are similar matrices (A X = X B), so their rotation ANGLES are equal.
    This tests the Euler convention, typos and robot/camera pairing per pair of samples.
  * A constant TCP offset (pendant shows TCP instead of flange) or a constant user/world frame
    is absorbed by T_flange_marker / T_base_cam, so it can NOT cause a large residual.
  * A wrong marker size scales every camera translation by the same factor and leaves the
    rotations untouched; it is estimated as a free scale here.
The input file is only read.  Nothing here changes calibration results.
"""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from calib_utils import inv_T, load_json, make_T, rot_angle_deg, save_json

DATA = Path(__file__).resolve().parent / 'data'
CONVENTIONS = ['ZYZ', 'zyz', 'ZYX', 'zyx', 'XYZ', 'xyz']     # upper = intrinsic, lower = extrinsic (scipy)
# Consensus thresholds (diagnostic only, not a pass criterion).  A 100 mm marker at 0.8-1.4 m is
# 65-110 px wide, so its single-view rotation is only good to a few degrees.
INLIER_MM, INLIER_DEG = 15.0, 5.0


def posx_T(posx, conv='ZYZ'):
    return make_T(R.from_euler(conv, posx[3:], degrees=True).as_matrix(), np.asarray(posx[:3]) / 1000.0)


def angle_mismatch(Tbf, Tcm):
    """|angle(robot motion) - angle(marker motion)| for every pair, deg (n x n)."""
    n = len(Tbf)
    M = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        a = rot_angle_deg((inv_T(Tbf[i]) @ Tbf[j])[:3, :3])
        b = rot_angle_deg((inv_T(Tcm[i]) @ Tcm[j])[:3, :3])
        M[i, j] = M[j, i] = abs(a - b)
    return M


def closed_form(Tbf, Tcm):
    """T_flange_marker from A X = X B (rotation by SVD over rotation vectors, always proper), then T_base_cam."""
    A, B = [], []
    for i, j in itertools.combinations(range(len(Tbf)), 2):
        a, b = inv_T(Tbf[i]) @ Tbf[j], inv_T(Tcm[i]) @ Tcm[j]
        if rot_angle_deg(a[:3, :3]) > 3 and rot_angle_deg(b[:3, :3]) > 3:
            A.append(a)
            B.append(b)
    if len(A) < 2:
        return None
    H = sum(np.outer(R.from_matrix(b[:3, :3]).as_rotvec(), R.from_matrix(a[:3, :3]).as_rotvec())
            for a, b in zip(A, B))
    U, _, Vt = np.linalg.svd(H)
    Rx = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    C = np.vstack([a[:3, :3] - np.eye(3) for a in A])
    d = np.concatenate([Rx @ b[:3, 3] - a[:3, 3] for a, b in zip(A, B)])
    Tfm = make_T(Rx, np.linalg.lstsq(C, d, rcond=None)[0])
    Tbc_i = [Tbf[i] @ Tfm @ inv_T(Tcm[i]) for i in range(len(Tbf))]
    U, _, Vt = np.linalg.svd(sum(T[:3, :3] for T in Tbc_i))
    Rb = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return make_T(Rb, np.median([T[:3, 3] for T in Tbc_i], axis=0)), Tfm


def residuals(Tbf, Tcm, Tbc, Tfm, scale=1.0):
    pos, rot = [], []
    for a, b in zip(Tbf, Tcm):
        b = b.copy()
        b[:3, 3] *= scale
        via_robot, via_cam = a @ Tfm, Tbc @ b
        pos.append(np.linalg.norm(via_robot[:3, 3] - via_cam[:3, 3]) * 1000)
        rot.append(rot_angle_deg(via_robot[:3, :3].T @ via_cam[:3, :3]))
    return np.array(pos), np.array(rot)


def refine(Tbf, Tcm, Tbc, Tfm, fit_scale=False):
    """Least squares on all given samples (plain L2: no hidden down-weighting). Returns Tbc, Tfm, scale."""
    def unpack(x):
        return (make_T(R.from_rotvec(x[:3]).as_matrix(), x[3:6]), make_T(R.from_rotvec(x[6:9]).as_matrix(), x[9:12]),
                x[12] if fit_scale else 1.0)

    def f(x):
        bc, fm, s = unpack(x)
        out = []
        for a, b in zip(Tbf, Tcm):
            b = b.copy()
            b[:3, 3] *= s
            d = inv_T(a @ fm) @ bc @ b
            out += [R.from_matrix(d[:3, :3]).as_rotvec() * 0.1, d[:3, 3]]   # 0.1 m per rad
        return np.concatenate(out)

    x0 = np.concatenate([R.from_matrix(Tbc[:3, :3]).as_rotvec(), Tbc[:3, 3],
                         R.from_matrix(Tfm[:3, :3]).as_rotvec(), Tfm[:3, 3]] + ([[1.0]] if fit_scale else []))
    return unpack(least_squares(f, x0).x)


def solve(Tbf, Tcm, fit_scale=False):
    cf = closed_form(Tbf, Tcm)
    if cf is None:
        return None
    Tbc, Tfm, s = refine(Tbf, Tcm, *cf, fit_scale=fit_scale)
    pos, rot = residuals(Tbf, Tcm, Tbc, Tfm, s)
    return dict(Tbc=Tbc, Tfm=Tfm, scale=s, pos=pos, rot=rot)


def consensus(Tbf, Tcm, trials=3000, seed=0):
    """Largest set of samples that agree with ONE (T_base_cam, T_flange_marker) - diagnostic, not a filter."""
    rng = np.random.default_rng(seed)
    n, best = len(Tbf), []
    for _ in range(trials):
        idx = sorted(rng.choice(n, 4, replace=False))
        cf = closed_form([Tbf[i] for i in idx], [Tcm[i] for i in idx])
        if cf is None:
            continue
        pos, rot = residuals(Tbf, Tcm, *cf)
        inl = list(np.nonzero((pos < 3 * INLIER_MM) & (rot < 3 * INLIER_DEG))[0])
        if len(inl) > len(best):
            best = inl
    for _ in range(3):                                   # re-fit on the set, re-select with the final thresholds
        if len(best) < 4:
            break
        sol = solve([Tbf[i] for i in best], [Tcm[i] for i in best])
        pos, rot = residuals(Tbf, Tcm, sol['Tbc'], sol['Tfm'])
        best = list(np.nonzero((pos < INLIER_MM) & (rot < INLIER_DEG))[0])
    return best


def ippe_alternatives(T, K, size):
    """Both planar-pose solutions for the marker corners that T projects to (IPPE two-fold ambiguity)."""
    h = size / 2.0
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
    img, _ = cv2.projectPoints(obj, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, None)
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img.reshape(4, 2), K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    sols = [make_T(cv2.Rodrigues(r)[0], t) for r, t in zip(rvecs, tvecs)]
    other = [S for S in sols if rot_angle_deg(S[:3, :3].T @ T[:3, :3]) > 0.5]
    return other[0] if other else None, [float(e) for e in np.ravel(errs)]


def fmt(sol):
    return (f'pos mean {sol["pos"].mean():7.1f} mm max {sol["pos"].max():7.1f} | '
            f'rot mean {sol["rot"].mean():6.2f} deg max {sol["rot"].max():6.2f}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--samples', type=Path, default=DATA / 'handeye_samples.json')
    p.add_argument('--out', type=Path, default=DATA / f'handeye_forensics_{time.strftime("%Y%m%d_%H%M%S")}.json')
    p.add_argument('--marker_size', type=float, default=0.10, help='value used at capture (for the IPPE test only)')
    p.add_argument('--intrinsics', type=Path, default=None, help='real D455 intrinsics json (K); newest in data/real_env')
    a = p.parse_args()
    if a.out.resolve() in {(DATA / 'eye_to_hand' / 'result.json').resolve(), a.samples.resolve()}:
        raise SystemExit('refusing to write over a protected file')

    raw = a.samples.read_bytes()
    S = json.loads(raw)
    report = dict(samples=str(a.samples), samples_sha256=hashlib.sha256(raw).hexdigest(), n=len(S),
                  created=time.strftime('%Y-%m-%d %H:%M:%S'), note='diagnostic only; not a calibration result')
    Tcm = [np.array(s['T_cam_marker']) for s in S]
    np.set_printoptions(precision=4, suppress=True)
    print(f'{len(S)} samples from {a.samples}  sha256 {report["samples_sha256"][:16]}...\n')

    # 1. stored T_base_flange really is posx_to_T(posx)?
    stored = max(np.abs(np.array(s['T_base_flange']) - posx_T(s['posx'])).max() for s in S)
    print(f'[1] stored T_base_flange vs ZYZ(posx): max diff {stored:.2e}')
    report['stored_T_base_flange_matches_ZYZ'] = bool(stored < 1e-9)

    # 2. Euler convention (rotation-angle invariant, independent of X)
    print('\n[2] Euler convention: |angle(robot motion) - angle(marker motion)| over all pairs')
    conv = {}
    for c in CONVENTIONS:
        M = angle_mismatch([posx_T(s['posx'], c) for s in S], Tcm)
        v = M[np.triu_indices(len(S), 1)]
        conv[c] = dict(median=float(np.median(v)), mean=float(v.mean()), frac_below_2deg=float((v < 2).mean()))
        print(f'    {c}: median {np.median(v):6.2f}  mean {v.mean():6.2f}  pairs < 2 deg {100 * (v < 2).mean():5.1f} %')
    report['euler_convention'] = conv
    Tbf = [posx_T(s['posx']) for s in S]
    M = angle_mismatch(Tbf, Tcm)

    # 3. same camera view but different robot pose (or vice versa) -> pairing error
    print('\n[3] pairing check (camera barely changed but robot moved, or the reverse)')
    pairs = []
    for i, j in itertools.combinations(range(len(S)), 2):
        dc = np.linalg.norm(Tcm[i][:3, 3] - Tcm[j][:3, 3]) * 1000
        rc = rot_angle_deg(Tcm[i][:3, :3].T @ Tcm[j][:3, :3])
        dr = np.linalg.norm(Tbf[i][:3, 3] - Tbf[j][:3, 3]) * 1000
        rr = rot_angle_deg(Tbf[i][:3, :3].T @ Tbf[j][:3, :3])
        if (dc < 5 and rc < 1) != (dr < 5 and rr < 1):
            pairs.append(dict(i=i, j=j, cam_mm=dc, cam_deg=rc, robot_mm=dr, robot_deg=rr,
                              times=[S[i].get('time'), S[j].get('time')]))
            print(f'    #{i} & #{j} ({S[i].get("time")}, {S[j].get("time")}): camera {dc:.1f} mm / {rc:.1f} deg, '
                  f'robot {dr:.1f} mm / {rr:.1f} deg  <-- impossible for a rigid marker')
    if not pairs:
        print('    none')
    report['pairing_conflicts'] = pairs

    # 4. per-sample agreement with the others
    print('\n[4] per sample: median angle mismatch to all others, #others agreeing within 2 deg')
    per = []
    for i in range(len(S)):
        m = np.delete(M[i], i)
        per.append(dict(i=i, median_deg=float(np.median(m)), agree=int((m < 2).sum())))
        print(f'    #{i:2d} {S[i].get("time", "")}  median {np.median(m):6.2f} deg  agree {(m < 2).sum():2d}  '
              f'tilt {S[i]["tilt_deg"]:5.1f}  dist {S[i]["dist_m"]:.3f}  posx {np.round(S[i]["posx"], 2).tolist()}')
    report['per_sample'] = per

    # 5. robot pose typed for a neighbouring capture (index shift)?
    print('\n[5] index shift: angle mismatch median when camera #i is paired with robot #(i+k)')
    shift = {}
    for k in (-2, -1, 0, 1, 2):
        idx = [i for i in range(len(S)) if 0 <= i + k < len(S)]
        v = angle_mismatch([Tbf[i + k] for i in idx], [Tcm[i] for i in idx])[np.triu_indices(len(idx), 1)]
        shift[k] = float(np.median(v))
        print(f'    k={k:+d}: median {np.median(v):6.2f} deg  pairs < 2 deg {100 * (v < 2).mean():5.1f} %')
    report['index_shift_median_deg'] = shift

    # 6. solves
    print('\n[6] solves (plain least squares on the listed samples)')
    sol_all = solve(Tbf, Tcm)
    print(f'    all {len(S):2d}            : {fmt(sol_all)}')
    sol_scale = solve(Tbf, Tcm, fit_scale=True)
    print(f'    all + marker scale  : {fmt(sol_scale)}  scale {sol_scale["scale"]:.4f}')
    core = consensus(Tbf, Tcm)
    report['solves'] = dict(all=dict(pos_mean=float(sol_all['pos'].mean()), pos_max=float(sol_all['pos'].max()),
                                     rot_mean=float(sol_all['rot'].mean()), rot_max=float(sol_all['rot'].max())))
    report['consensus_set'] = [int(i) for i in core]
    print(f'    largest consistent set (<{INLIER_MM:.0f} mm, <{INLIER_DEG:.0f} deg): {len(core)} samples {core}')
    if len(core) >= 4:
        sc = solve([Tbf[i] for i in core], [Tcm[i] for i in core], fit_scale=True)
        print(f'      refit on that set : {fmt(sc)}  scale {sc["scale"]:.4f}')
        print(f'      T_base_cam translation {np.round(sc["Tbc"][:3, 3], 4).tolist()} m, '
              f'camera +z in base {np.round(sc["Tbc"][:3, 2], 3).tolist()}')
        print(f'      T_flange_marker translation {np.round(sc["Tfm"][:3, 3] * 1000, 1).tolist()} mm')
        pos, rot = residuals(Tbf, Tcm, sc['Tbc'], sc['Tfm'], sc['scale'])
        print('      residual of EVERY sample against that set:')
        for i in range(len(S)):
            print(f'        #{i:2d}  {pos[i]:8.1f} mm  {rot[i]:7.2f} deg{"" if i in core else "   (outside set)"}')
        report['consensus_solve'] = dict(T_base_cam=sc['Tbc'].tolist(), T_flange_marker=sc['Tfm'].tolist(),
                                         scale=float(sc['scale']), pos_mm=pos.tolist(), rot_deg=rot.tolist(),
                                         warning='exploratory: chosen by consensus, NOT validated, do not use')

    # 7. planar-marker flip (IPPE) on the samples outside the set
    kpath = a.intrinsics or (sorted((DATA / 'real_env').glob('real_intrinsics_*.json')) or [None])[-1]
    if kpath is not None and len(core) >= 4:
        K = np.array(load_json(kpath)['K'])
        print(f'\n[7] IPPE flip test (K from {kpath.name}, marker {a.marker_size} m): does the 2nd solution fit?')
        flips = []
        for i in range(len(S)):
            if i in core:
                continue
            alt, errs = ippe_alternatives(Tcm[i], K, a.marker_size)
            if alt is None:
                continue
            p0, r0 = residuals([Tbf[i]], [Tcm[i]], sc['Tbc'], sc['Tfm'], sc['scale'])
            p1, r1 = residuals([Tbf[i]], [alt], sc['Tbc'], sc['Tfm'], sc['scale'])
            flips.append(dict(i=i, as_captured=[float(p0[0]), float(r0[0])], flipped=[float(p1[0]), float(r1[0])],
                              reproj_px=errs))
            print(f'    #{i:2d} as captured {p0[0]:7.1f} mm {r0[0]:6.2f} deg | flipped {p1[0]:7.1f} mm {r1[0]:6.2f} deg'
                  f' | reproj of both {np.round(errs, 2).tolist()} px')
        report['ippe_flip'] = flips

    # 8. re-pairing: which typed robot pose fits each camera view under the consensus model?
    if len(core) >= 4:
        print('\n[8] re-pairing: best robot pose #j for camera view #i (consensus model)')
        cost = np.full((len(S), len(S)), np.inf)
        for i in range(len(S)):
            for j in range(len(S)):
                pj, rj = residuals([Tbf[j]], [Tcm[i]], sc['Tbc'], sc['Tfm'], sc['scale'])
                if pj[0] < 2 * INLIER_MM and rj[0] < 2 * INLIER_DEG:
                    cost[i, j] = pj[0] + 10 * rj[0]
        pairs, used = [], set()
        for i in range(len(S)):
            j = int(np.argmin(cost[i]))
            note = ''
            if not np.isfinite(cost[i, j]):
                note, j = 'no typed robot pose fits this view', None
            elif j in used:
                note = f'robot #{j} already used (same view captured twice)'
            pr = residuals([Tbf[j]], [Tcm[i]], sc['Tbc'], sc['Tfm'], sc['scale']) if j is not None else None
            if j is not None and j not in used:
                used.add(j)
                pairs.append((i, j))
            print(f'    camera #{i:2d} -> robot #{"--" if j is None else f"{j:2d}"}'
                  + (f'  {pr[0][0]:6.1f} mm {pr[1][0]:5.2f} deg' if pr else '')
                  + (f'  shift {j - i:+d}' if j is not None and j != i else '') + (f'  ({note})' if note else ''))
        unused = sorted(set(range(len(S))) - used)
        print(f'    robot poses not matched to any view: {unused}')
        report['repairing'] = dict(pairs=pairs, robot_unmatched=unused)
        if len(pairs) >= 6:
            A = [Tbf[j] for _, j in pairs]
            B = [Tcm[i] for i, _ in pairs]
            fit = solve(A, B, fit_scale=True)
            loo_p, loo_r = [], []
            for k in range(len(pairs)):            # leave-one-out: residual of a pair NOT used in the fit
                rest = [m for m in range(len(pairs)) if m != k]
                f = solve([A[m] for m in rest], [B[m] for m in rest], fit_scale=True)
                p_, r_ = residuals([A[k]], [B[k]], f['Tbc'], f['Tfm'], f['scale'])
                loo_p.append(p_[0])
                loo_r.append(r_[0])
            loo_p, loo_r = np.array(loo_p), np.array(loo_r)
            print(f'    re-paired solve ({len(pairs)} pairs): {fmt(fit)}  scale {fit["scale"]:.4f}')
            print(f'    leave-one-out: pos mean {loo_p.mean():.1f} mm, median {np.median(loo_p):.1f}, max {loo_p.max():.1f} '
                  f'| rot mean {loo_r.mean():.2f} deg, max {loo_r.max():.2f}')
            print(f'    T_base_cam (EXPLORATORY, do not use):\n{np.round(fit["Tbc"], 4)}')
            report['repaired_solve'] = dict(
                n=len(pairs), T_base_cam=fit['Tbc'].tolist(), T_flange_marker=fit['Tfm'].tolist(),
                scale=float(fit['scale']), pos_mean=float(fit['pos'].mean()), pos_max=float(fit['pos'].max()),
                rot_mean=float(fit['rot'].mean()), rot_max=float(fit['rot'].max()),
                loo_pos_mm=loo_p.tolist(), loo_rot_deg=loo_r.tolist(),
                warning='exploratory: pairing inferred from data, NOT validated; re-capture with automatic pose read')

    save_json(a.out, report)
    print(f'\nsaved diagnostic report: {a.out}')


if __name__ == '__main__':
    main()
