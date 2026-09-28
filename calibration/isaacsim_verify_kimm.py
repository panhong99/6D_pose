"""Detect the ChArUco board in images captured by isaacsim_capture_kimm.py,
run calibrate_eye_to_hand, and compare T_base_cam against the sim ground truth.

Run in an env with a recent OpenCV (contrib aruco), e.g.:
  conda run -n foundationpose python calibration/isaacsim_verify_kimm.py \
      --capture-dir calibration/sim_capture
"""
import argparse
import json
import os
import sys

import cv2
import cv2.aruco as aruco
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from handeye_core import METHODS, calibrate_eye_to_hand, consistency_residual, pose_error

parser = argparse.ArgumentParser()
parser.add_argument("--capture-dir", default=os.path.join(os.path.dirname(__file__), "sim_capture"))
parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "isaacsim_verify_result.txt"))
args = parser.parse_args()

with open(os.path.join(args.capture_dir, "board_config.json")) as f:
    bc = json.load(f)
with open(os.path.join(args.capture_dir, "gt.json")) as f:
    gt = json.load(f)
with open(os.path.join(args.capture_dir, "poses.json")) as f:
    poses = json.load(f)

K = np.array(gt["K"])
dist = np.zeros(5)
T_base_cam_gt = np.array(gt["T_base_cam"])

aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, bc["dictionary"]))
board = aruco.CharucoBoard((bc["squares_x"], bc["squares_y"]), bc["square_length"], bc["marker_length"], aruco_dict)
detector = aruco.CharucoDetector(board)

T_base_ee_list, T_cam_board_list, used_idx = [], [], []
for i, p in enumerate(poses):
    img_path = os.path.join(args.capture_dir, p["img"])
    img = cv2.imread(img_path)
    ch_corners, ch_ids, m_corners, m_ids = detector.detectBoard(img)
    if ch_ids is None or len(ch_ids) < 6:
        print(f"  [{i:03d}] {p['img']}: detection failed ({0 if ch_ids is None else len(ch_ids)} corners), skipped")
        continue
    obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        print(f"  [{i:03d}] {p['img']}: solvePnP failed, skipped")
        continue
    R, _ = cv2.Rodrigues(rvec)
    T_cam_board = np.eye(4)
    T_cam_board[:3, :3] = R
    T_cam_board[:3, 3] = tvec.reshape(3)
    T_base_ee_list.append(np.array(p["T_base_ee"]))
    T_cam_board_list.append(T_cam_board)
    used_idx.append(i)
    print(f"  [{i:03d}] {p['img']}: {len(ch_ids)} corners, board z={tvec[2, 0]:.3f} m")

lines = []
lines.append(f"images: {len(poses)} captured, {len(used_idx)} used after ChArUco detection")
lines.append(f"K = {K.tolist()}")
lines.append(f"T_base_cam (ground truth):\n{T_base_cam_gt}")
lines.append("")

if len(used_idx) < 4:
    lines.append(f"ERROR: only {len(used_idx)} usable poses (<4), cannot run calibrateHandEye")
    print("\n".join(lines))
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    sys.exit(1)

for m in METHODS:
    try:
        T_base_cam_est, T_ee_board_est = calibrate_eye_to_hand(T_base_ee_list, T_cam_board_list, method=m)
    except cv2.error as e:
        lines.append(f"{m:11s} FAILED: {e}")
        continue
    rot_err, trans_err_mm = pose_error(T_base_cam_est, T_base_cam_gt)
    rres, tres = consistency_residual(T_base_ee_list, T_cam_board_list, T_base_cam_est, T_ee_board_est)
    line = (
        f"{m:11s} rot_err {rot_err:.3f} deg  trans_err {trans_err_mm:.2f} mm   "
        f"(chain residual median: rot {np.median(rres):.3f} deg, trans {np.median(tres):.2f} mm)"
    )
    print(line)
    lines.append(line)

with open(args.out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nresult written to {args.out}")
