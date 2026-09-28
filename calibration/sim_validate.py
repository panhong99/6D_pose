"""Synthetic validation of handeye_core (no robot/camera needed).

Known ground truth T_base_cam, T_ee_board -> generate poses with noise ->
calibrate -> compare to truth. Sweeps method / #poses / rotation diversity / noise.
Usage: python sim_validate.py [--trials 200] [--seed 0]
"""
import argparse, json
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from handeye_core import *


def rand_rot(rng, max_deg):
    """random rotation with angle<=max_deg about a random axis"""
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    return Rot.from_rotvec(axis * np.radians(rng.uniform(0, max_deg))).as_matrix()


def noisy(T, rng, rot_deg, t_mm):
    dR = Rot.from_rotvec(rng.normal(size=3) * np.radians(rot_deg) / np.sqrt(3)).as_matrix()
    dT = make_T(dR, rng.normal(size=3) * t_mm / 1000 / np.sqrt(3))
    return T @ dT


def gen(rng, n, rot_range, cam_rot_noise, cam_t_noise, rob_rot_noise, rob_t_noise, T_base_cam, T_ee_board):
    Tbe_l, Tcb_l = [], []
    # board facing the camera (board z toward camera, i.e. pointing back along -z_cam), then random tilt
    R_face = Rot.from_euler("x", 180, degrees=True).as_matrix()
    while len(Tbe_l) < n:
        t = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.1, 0.1), rng.uniform(0.5, 0.9)])
        Tcb = make_T(rand_rot(rng, rot_range) @ R_face, t)
        Tbe = T_base_cam @ Tcb @ inv_T(T_ee_board)          # true robot pose
        Tbe_l.append(noisy(Tbe, rng, rob_rot_noise, rob_t_noise))
        Tcb_l.append(noisy(Tcb, rng, cam_rot_noise, cam_t_noise))  # measured board pose
    return Tbe_l, Tcb_l


def run(trials, seed, n, rot_range, cam_rot, cam_t, rob_rot=0.05, rob_t=0.1, methods=METHODS):
    rng = np.random.default_rng(seed)
    out = {m: [] for m in methods}
    for _ in range(trials):
        T_base_cam = make_T(Rot.from_euler("xyz", [200, 10, 90], degrees=True).as_matrix() * 1.0, [0.8, 0.0, 0.6])
        T_ee_board = make_T(Rot.from_euler("xyz", [5, -3, 10], degrees=True).as_matrix(), [0.0, 0.02, 0.08])
        Tbe, Tcb = gen(rng, n, rot_range, cam_rot, cam_t, rob_rot, rob_t, T_base_cam, T_ee_board)
        for m in methods:
            try:
                Te, _ = calibrate_eye_to_hand(Tbe, Tcb, m)
                out[m].append(pose_error(Te, T_base_cam))
            except cv2.error:
                out[m].append((np.nan, np.nan))
    return {m: (np.nanmedian(np.array(v)[:, 0]), np.nanmedian(np.array(v)[:, 1])) for m, v in out.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print("== [1] Noise-free sanity (should be ~0) ==")
    r = run(20, a.seed, 15, 45, 0, 0, 0, 0)
    for m, (re, te) in r.items(): print(f"  {m:11s} rot {re:.4f} deg  trans {te:.4f} mm")

    print("\n== [2] Realistic noise (cam rot 0.5deg, trans 2mm; robot 0.05deg/0.1mm), 15 poses, tilt<=45deg ==")
    r = run(a.trials, a.seed, 15, 45, 0.5, 2.0)
    for m, (re, te) in r.items(): print(f"  {m:11s} rot {re:.3f} deg  trans {te:.2f} mm")

    print("\n== [3] #poses sweep (PARK, tilt<=45deg) ==")
    for n in [4, 6, 10, 15, 20, 30]:
        re, te = run(a.trials, a.seed, n, 45, 0.5, 2.0, methods=["PARK"])["PARK"]
        print(f"  N={n:2d}  rot {re:.3f} deg  trans {te:.2f} mm")

    print("\n== [4] Rotation diversity sweep (PARK, N=15) ==")
    for rr in [5, 15, 30, 45, 60]:
        re, te = run(a.trials, a.seed, 15, rr, 0.5, 2.0, methods=["PARK"])["PARK"]
        print(f"  tilt<={rr:2d}deg  rot {re:.3f} deg  trans {te:.2f} mm")

    print("\n== [5] Board-detection noise sweep (PARK, N=15, tilt<=45deg) ==")
    for cr, ct in [(0.1, 0.5), (0.5, 2), (1.0, 5), (2.0, 10)]:
        re, te = run(a.trials, a.seed, 15, 45, cr, ct, methods=["PARK"])["PARK"]
        print(f"  cam noise {cr}deg/{ct}mm  rot {re:.3f} deg  trans {te:.2f} mm")
