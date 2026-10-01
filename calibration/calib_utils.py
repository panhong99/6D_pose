"""Eye-to-hand (fixed camera) hand-eye calibration helpers for Doosan M1013 + D455.

Chain (4x4 homogeneous, translation in meters):
    T_cam_marker = inv(T_base_cam) @ T_base_flange @ T_flange_marker
Goal: T_base_cam.  T_flange_marker (where the ArUco sits on the flange) is a byproduct.
Once known:  T_base_obj = T_base_cam @ T_cam_obj   (T_cam_obj = FoundationPose output).

Doosan poses are posx = [x, y, z (mm), a, b, c (deg)] with intrinsic ZYZ Euler angles.

Self-test (no hardware):  python calibration/calib_utils.py
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


# ------------------------------------------------------------------ transforms

def make_T(Rm, t):
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def inv_T(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def rot_angle_deg(Rm):
    return float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))))


def average_poses(Ts):
    """Mean of noisy poses: median translation, SVD-projected mean rotation."""
    Ts = np.asarray(Ts)
    U, _, Vt = np.linalg.svd(Ts[:, :3, :3].mean(axis=0))
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt
    return make_T(Rm, np.median(Ts[:, :3, 3], axis=0))


# ------------------------------------------------------------------ Doosan posx

def posx_to_T(posx):
    """Doosan [x,y,z (mm), a,b,c (deg, intrinsic ZYZ)] -> 4x4 in meters."""
    x, y, z, a, b, c = [float(v) for v in posx]
    return make_T(R.from_euler('ZYZ', [a, b, c], degrees=True).as_matrix(),
                  np.array([x, y, z]) / 1000.0)


def T_to_posx(T):
    a, b, c = R.from_matrix(T[:3, :3]).as_euler('ZYZ', degrees=True)
    return [*(T[:3, 3] * 1000.0), a, b, c]


# ------------------------------------------------------------------ ArUco marker

def make_detector(dictionary='DICT_4X4_50'):
    aruco = cv2.aruco
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return aruco.ArucoDetector(aruco.getPredefinedDictionary(getattr(aruco, dictionary)), params)


def detect_marker(bgr, K, detector, marker_id, marker_size):
    """Pose of one ArUco marker in the camera frame (image must be undistorted).

    Returns dict(T=T_cam_marker, corners, reproj_px, tilt_deg) or None.  tilt_deg is the
    angle between the marker normal and the viewing ray (0 = facing the camera).
    """
    corners, ids, _ = detector.detectMarkers(bgr)
    if ids is None or marker_id not in ids.flatten():
        return None
    c = corners[list(ids.flatten()).index(marker_id)].reshape(4, 2).astype(np.float64)
    h = marker_size / 2.0
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, c, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    T = make_T(cv2.Rodrigues(rvec)[0], tvec)
    ray = T[:3, 3] / np.linalg.norm(T[:3, 3])
    tilt = float(np.degrees(np.arccos(np.clip(abs(T[:3, 2] @ ray), 0, 1))))
    return dict(T=T, corners=c, reproj_px=float(np.linalg.norm(proj.reshape(4, 2) - c, axis=1).mean()),
                tilt_deg=tilt)


# ------------------------------------------------------------------ solver

def _park_martin(A_list, B_list):
    """Closed-form AX = XB (Park & Martin). Returns X (4x4)."""
    M = np.zeros((3, 3))
    for A, B in zip(A_list, B_list):
        M += np.outer(R.from_matrix(B[:3, :3]).as_rotvec(), R.from_matrix(A[:3, :3]).as_rotvec())
    w, V = np.linalg.eigh(M.T @ M)
    Rx = V @ np.diag(1 / np.sqrt(w)) @ V.T @ M.T
    U, _, Vt = np.linalg.svd(Rx)          # inconsistent data can give det -1: project onto SO(3)
    Rx = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    C =np.vstack([A[:3, :3] - np.eye(3) for A in A_list])
    d = np.concatenate([Rx @ B[:3, 3] - A[:3, 3] for A, B in zip(A_list, B_list)])
    return make_T(Rx, np.linalg.lstsq(C, d, rcond=None)[0])


def _residual_vec(x, Tbf, Tcm, rot_w):
    Tbc, Tfm = make_T(R.from_rotvec(x[:3]).as_matrix(), x[3:6]), make_T(R.from_rotvec(x[6:9]).as_matrix(), x[9:12])
    Tcb = inv_T(Tbc)
    out = []
    for a, b in zip(Tbf, Tcm):
        d = inv_T(Tcb @ a @ Tfm) @ b
        out += [R.from_matrix(d[:3, :3]).as_rotvec() * rot_w, d[:3, 3]]
    return np.concatenate(out)


def solve_eye_to_hand(T_base_flange, T_cam_marker, refine=True, min_rot_deg=3.0):
    """Return (T_base_cam, T_flange_marker) from N >= 3 pose pairs."""
    Tbf, Tcm = [np.asarray(T) for T in T_base_flange], [np.asarray(T) for T in T_cam_marker]
    if len(Tbf) < 3:
        raise ValueError('Need at least 3 samples')
    # T_cm_i^-1 T_cm_j = X^-1 (T_bf_i^-1 T_bf_j) X   ->   A X = X B with X = T_flange_marker
    A_list, B_list = [], []
    for i in range(len(Tbf)):
        for j in range(i + 1, len(Tbf)):
            A, B = inv_T(Tbf[i]) @ Tbf[j], inv_T(Tcm[i]) @ Tcm[j]
            if rot_angle_deg(A[:3, :3]) >= min_rot_deg and rot_angle_deg(B[:3, :3]) >= min_rot_deg:
                A_list.append(A)
                B_list.append(B)
    if len(A_list) < 3:
        raise ValueError('Poses are too similar: rotate the flange more between samples')
    Tfm = _park_martin(A_list, B_list)
    Tbc = inv_T(average_poses([Tcm[i] @ inv_T(Tfm) @ inv_T(Tbf[i]) for i in range(len(Tbf))]))
    if refine:
        x0 = np.concatenate([R.from_matrix(Tbc[:3, :3]).as_rotvec(), Tbc[:3, 3],
                             R.from_matrix(Tfm[:3, :3]).as_rotvec(), Tfm[:3, 3]])
        sol = least_squares(_residual_vec, x0, args=(Tbf, Tcm, 0.1), loss='soft_l1', f_scale=0.005)
        x = sol.x
        Tbc = make_T(R.from_rotvec(x[:3]).as_matrix(), x[3:6])
        Tfm = make_T(R.from_rotvec(x[6:9]).as_matrix(), x[9:12])
    return Tbc, Tfm


def evaluate(T_base_flange, T_cam_marker, T_base_cam, T_flange_marker):
    """Per-sample consistency (no ground truth needed).

    pos_err_mm: marker position in the base frame seen via the robot vs via the camera.
    rot_err_deg: marker orientation disagreement between the two routes.
    """
    pos, rot = [], []
    for a, b in zip(T_base_flange, T_cam_marker):
        via_robot, via_cam = a @ T_flange_marker, T_base_cam @ b
        pos.append(np.linalg.norm(via_robot[:3, 3] - via_cam[:3, 3]) * 1000)
        rot.append(rot_angle_deg(via_robot[:3, :3].T @ via_cam[:3, :3]))
    return np.array(pos), np.array(rot)


def pair_angle_mismatch_deg(Tbf_new, Tcm_new, T_base_flange, T_cam_marker):
    """Median |angle(robot motion) - angle(marker motion)| of a new sample against stored ones.

    A X = X B makes both angles equal for any T_flange_marker, so a large value means the robot
    pose does not belong to this camera view (stale/typed pose, marker slipped).  None if < 1 pair.
    """
    d = [abs(rot_angle_deg((inv_T(a) @ Tbf_new)[:3, :3]) - rot_angle_deg((inv_T(b) @ Tcm_new)[:3, :3]))
         for a, b in zip(T_base_flange, T_cam_marker)]
    return float(np.median(d)) if d else None


def leave_one_out(T_base_flange, T_cam_marker):
    """Residual of each sample under a solve WITHOUT that sample (pos mm, rot deg arrays)."""
    pos, rot = [], []
    for k in range(len(T_base_flange)):
        rest = [i for i in range(len(T_base_flange)) if i != k]
        Tbc, Tfm = solve_eye_to_hand([T_base_flange[i] for i in rest], [T_cam_marker[i] for i in rest])
        p, r = evaluate([T_base_flange[k]], [T_cam_marker[k]], Tbc, Tfm)
        pos.append(p[0])
        rot.append(r[0])
    return np.array(pos), np.array(rot)


def rotation_spread_deg(T_base_flange):
    Rs = [T[:3, :3] for T in T_base_flange]
    return max(rot_angle_deg(a.T @ b) for a in Rs for b in Rs)


# ------------------------------------------------------------------ files

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_json(path):
    return json.loads(Path(path).read_text())


def result_dict(T_base_cam, T_flange_marker, pos_err, rot_err, n, meta=None):
    return dict(T_base_cam=T_base_cam.tolist(), T_flange_marker=T_flange_marker.tolist(),
                n_samples=n, pos_err_mm=dict(mean=float(pos_err.mean()), max=float(pos_err.max())),
                rot_err_deg=dict(mean=float(rot_err.mean()), max=float(rot_err.max())),
                created=time.strftime('%Y-%m-%d %H:%M:%S'), meta=meta or {})


# ------------------------------------------------------------------ self-test

def _selftest():
    rng = np.random.default_rng(0)
    print('posx round trip ...', end=' ')
    for _ in range(50):
        p = [*rng.uniform(-800, 800, 3), rng.uniform(-180, 180), rng.uniform(10, 170), rng.uniform(-180, 180)]
        q = T_to_posx(posx_to_T(p))
        assert np.allclose(posx_to_T(p), posx_to_T(q), atol=1e-9)
    print('ok')

    print('marker detection on a synthetic image ...', end=' ')
    K = np.array([[900., 0, 640], [0, 900., 360], [0, 0, 1]])
    size = 0.10  # the pipeline itself is exact; small markers just add sub-pixel noise
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 1, 200)
    marker = cv2.copyMakeBorder(marker, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)  # white quiet zone
    true = make_T(R.from_euler('xyz', [200, 15, 10], degrees=True).as_matrix(), [0.02, -0.01, 0.5])
    h = size / 2 * 280 / 200          # 40px quiet zone on each side of the 200px marker
    q3 = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    px = (K @ (true[:3, :3] @ q3.T + true[:3, 3:4])).T
    px = (px[:, :2] / px[:, 2:3]).astype(np.float32)
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [279, 0], [279, 279], [0, 279]]), px)
    img = cv2.warpPerspective(marker, H, (1280, 720), borderValue=255)
    det = detect_marker(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), K, make_detector(), 1, size)
    assert det is not None, 'marker not detected'
    dt = np.linalg.norm(det['T'][:3, 3] - true[:3, 3]) * 1000
    dr = rot_angle_deg(det['T'][:3, :3].T @ true[:3, :3])
    assert dt < 3 and dr < 3, f'marker pose off: {dt:.2f} mm, {dr:.2f} deg'
    print(f'ok  (pos err {dt:.2f} mm, rot err {dr:.2f} deg, reproj {det["reproj_px"]:.2f} px)')

    print('hand-eye solve on synthetic data:')
    Tbc = make_T(R.from_euler('xyz', [-125, 5, 178], degrees=True).as_matrix(), [0.9, 0.1, 0.6])
    Tfm = make_T(R.from_euler('xyz', [10, -5, 30], degrees=True).as_matrix(), [0.02, -0.01, 0.03])
    for noise_rot, noise_pos in ((0.0, 0.0), (0.3, 1.0), (0.6, 2.5)):
        Tbf, Tcm = [], []
        for _ in range(20):
            Rf = R.from_euler('xyz', [180, 0, 0], degrees=True) * R.from_rotvec(rng.normal(0, 0.5, 3))
            a = make_T(Rf.as_matrix(), [rng.uniform(0.4, 0.8), rng.uniform(-0.3, 0.3), rng.uniform(0.3, 0.7)])
            b = inv_T(Tbc) @ a @ Tfm
            b = b @ make_T(R.from_rotvec(rng.normal(0, np.radians(noise_rot) / 1.7, 3)).as_matrix(),
                           rng.normal(0, noise_pos / 1000 / 1.7, 3))
            Tbf.append(a)
            Tcm.append(b)
        est_bc, est_fm = solve_eye_to_hand(Tbf, Tcm)
        d = inv_T(Tbc) @ est_bc
        pe, re = evaluate(Tbf, Tcm, est_bc, est_fm)
        print(f'  noise {noise_rot:.1f} deg / {noise_pos:.1f} mm -> T_base_cam error '
              f'{np.linalg.norm(d[:3, 3]) * 1000:.2f} mm, {rot_angle_deg(d[:3, :3]):.2f} deg | '
              f'residual mean {pe.mean():.2f} mm {re.mean():.2f} deg')
        if noise_rot == 0.0:
            assert np.linalg.norm(d[:3, 3]) < 1e-4 and rot_angle_deg(d[:3, :3]) < 1e-2, 'noise-free solve is inexact'
    print('self-test passed')


if __name__ == '__main__':
    _selftest()
