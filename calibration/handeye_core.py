"""Eye-to-hand (fixed camera) hand-eye calibration core.

Chain (all 4x4 homogeneous, translation in meters):
    T_cam_board = inv(T_base_cam) @ T_base_ee @ T_ee_board
Unknowns: T_base_cam (goal), T_ee_board (byproduct, constant).

Inputs per pose i:
    T_base_ee[i]   : robot flange pose in base frame
    T_cam_board[i] : board pose in camera frame (solvePnP)
"""
import cv2
import numpy as np

METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def calibrate_eye_to_hand(T_base_ee, T_cam_board, method="PARK"):
    """Return (T_base_cam, T_ee_board).

    Maps onto cv2.calibrateHandEye's eye-in-hand formulation:
      OpenCV "gripper2base" <- inv(T_base_ee)  (= T_ee_base)
      OpenCV "target2cam"   <- T_cam_board
      OpenCV output "cam2gripper" <- T_base_cam
    """
    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for Tbe, Tcb in zip(T_base_ee, T_cam_board):
        Teb = inv_T(Tbe)
        R_g2b.append(Teb[:3, :3]); t_g2b.append(Teb[:3, 3].reshape(3, 1))
        R_t2c.append(Tcb[:3, :3]); t_t2c.append(Tcb[:3, 3].reshape(3, 1))
    R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=METHODS[method])
    T_base_cam = make_T(R, t)
    # constant T_ee_board, averaged over poses (chain rearranged)
    Ts = [inv_T(Tbe) @ T_base_cam @ Tcb for Tbe, Tcb in zip(T_base_ee, T_cam_board)]
    T_ee_board = make_T(_mean_rot([T[:3, :3] for T in Ts]), np.mean([T[:3, 3] for T in Ts], axis=0))
    return T_base_cam, T_ee_board


def _mean_rot(Rs):
    M = np.mean(Rs, axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def consistency_residual(T_base_ee, T_cam_board, T_base_cam, T_ee_board):
    """Per-pose residual of the chain. Returns (rot_deg[], trans_mm[]).
    Real-data quality metric (no ground truth needed)."""
    rot, trans = [], []
    for Tbe, Tcb in zip(T_base_ee, T_cam_board):
        pred = inv_T(T_base_cam) @ Tbe @ T_ee_board
        d = inv_T(pred) @ Tcb
        rot.append(np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1))))
        trans.append(np.linalg.norm(d[:3, 3]) * 1000)
    return np.array(rot), np.array(trans)


def pose_error(T_est, T_gt):
    d = inv_T(T_gt) @ T_est
    rot = np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)))
    return rot, np.linalg.norm(d[:3, 3]) * 1000  # deg, mm
