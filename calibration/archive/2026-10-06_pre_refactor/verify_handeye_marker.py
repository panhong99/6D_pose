"""Check the real hand-eye result with the flange ArUco only (no cube, no FoundationPose).  Read-only.

Robot says the flange is at posx P (read it on the pendant, or from ros2_flange_udp.py).  The camera sees the marker
fixed to the flange.  Two routes to the marker pose in the robot base frame must agree:
    via the robot : T_base_flange(P) @ T_flange_marker              (T_flange_marker from the hand-eye result)
    via the camera: T_base_cam @ T_cam_marker                        (marker detected now)
    python calibration/verify_handeye_marker.py --posx "x y z a b c" [--marker_size 0.10] [--n_frames 40]
Needs the D455 free (stop real_time_project/main.py and any preview first).  Nothing is sent to the robot or ROS.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'real_time_project'))
from real_time_utils import D455Source  # noqa: E402

from calib_utils import average_poses, detect_marker, inv_T, load_json, make_detector, posx_to_T, rot_angle_deg  # noqa: E402
from real_pose_dryrun import latest_t_base_cam  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--posx', required=True, help='"x y z a b c" tool-flange pose in the base frame (mm, deg, ZYZ)')
    p.add_argument('--t_base_cam', type=Path, default=None)
    p.add_argument('--marker_size', type=float, default=0.10)
    p.add_argument('--marker_id', type=int, default=1)
    p.add_argument('--n_frames', type=int, default=40)
    p.add_argument('--width', type=int, default=1280, help='keep the calibration resolution (intrinsics come from the stream)')
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--serial', default='338122300585')
    p.add_argument('--save', type=Path, default=None, help='save the annotated frame here')
    return p.parse_args(argv)


def main():
    args = parse_args()
    path = args.t_base_cam or latest_t_base_cam()
    res = load_json(path)
    T_bc, T_fm = np.array(res['T_base_cam']), np.array(res['T_flange_marker'])
    vals = [float(v) for v in args.posx.replace(',', ' ').split()]
    if len(vals) != 6:
        raise SystemExit('--posx needs 6 numbers')
    T_bf = posx_to_T(vals)
    cam = D455Source(args.width, args.height, 30, args.serial)
    detector = make_detector('DICT_4X4_50')
    Ts, last = [], None
    try:
        t_end = time.time() + 15
        while len(Ts) < args.n_frames and time.time() < t_end:
            frame = cam.read()
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            det = detect_marker(bgr, frame.K, detector, args.marker_id, args.marker_size)
            if det is not None:
                Ts.append(det['T'])
                last = (bgr, det, frame.K)
    finally:
        cam.close()
    if len(Ts) < args.n_frames // 2:
        raise SystemExit(f'marker id {args.marker_id} seen in only {len(Ts)}/{args.n_frames} frames: is the flange plate '
                         f'in view and the camera free?')
    T_cm = average_poses(Ts)
    jitter = np.std([t[:3, 3] for t in Ts], axis=0) * 1000
    via_cam, via_robot = T_bc @ T_cm, T_bf @ T_fm
    d = (via_cam[:3, 3] - via_robot[:3, 3]) * 1000
    ray = T_cm[:3, 3] / np.linalg.norm(T_cm[:3, 3])
    R = T_bc[:3, :3]
    d_cam = R.T @ d / 1000.0 * 1000            # difference expressed in camera axes (x right, y down, z forward), mm
    np.set_printoptions(precision=2, suppress=True)
    print(f'T_base_cam: {path}  (hand-eye residual {res["pos_err_mm"]["mean"]:.1f} mm mean)')
    print(f'marker seen in {len(Ts)} frames; camera->marker {np.round(T_cm[:3, 3] * 1000, 1).tolist()} mm '
          f'(distance {np.linalg.norm(T_cm[:3, 3]) * 1000:.0f} mm), tilt {np.degrees(np.arccos(abs(T_cm[:3, 2] @ ray))):.0f} deg, '
          f'jitter {np.round(jitter, 2).tolist()} mm')
    print(f'marker in base via ROBOT  [mm]: {np.round(via_robot[:3, 3] * 1000, 1).tolist()}   (flange posx {vals})')
    print(f'marker in base via CAMERA [mm]: {np.round(via_cam[:3, 3] * 1000, 1).tolist()}')
    print(f'camera - robot [mm]: dx {d[0]:+7.1f}  dy {d[1]:+7.1f}  dz {d[2]:+7.1f}   NORM {np.linalg.norm(d):.1f} mm')
    print(f'   in camera axes [mm] (x right, y down, z = along the optical axis): {np.round(d_cam, 1).tolist()}')
    print(f'marker orientation difference: {rot_angle_deg(via_robot[:3, :3].T @ via_cam[:3, :3]):.2f} deg')
    if args.save and last is not None:
        bgr, det, K = last
        cv2.polylines(bgr, [det['corners'].astype(np.int32)], True, (0, 255, 0), 2)
        cv2.imwrite(str(args.save), bgr)
        print(f'saved {args.save}')


if __name__ == '__main__':
    main()
