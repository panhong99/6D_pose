"""Cube position error: FoundationPose x T_base_cam  vs  ArUco-on-cube seen by the WRIST camera.  Read-only.

    ROS2 terminal 1:  bringup                       (python3 calibration/ros2_flange_udp.py must stream UDP 5006)
    conda terminal 2: python real_time_project/main.py --serial 338122300585 --width 640 --height 480   (UDP 5005)
    conda terminal 3: python calibration/cube_gt_compare.py [--marker_size 0.045] [--cube_edge 0.057]

GT   : wrist camera sees the marker on the cube's TOP face:  T_base_marker = T_base_flange @ T_flange_cam @ T_cam_marker
       (T_flange_cam from calibration/wrist_handeye.py); cube centre = marker origin moved cube_edge/2 along -marker z.
FP   : T_base_cube = T_base_cam @ T_cam_cube (stable FoundationPose pose from main.py), for every T_base_cam variant.
Per cube position: put the robot where the wrist camera sees the marker obliquely (0.3-0.6 m), still 2 s, press G;
then move the robot out of the fixed camera's view, wait for a stable pose, press F; press N to store the position.
The cube must not move between G and F.  Keys: G gt | F fp | N next position | U undo | S summary | Q quit.
Nothing is sent to the robot or ROS.  Output: data/gt_compare/<time>/{positions.json, summary.txt}.
"""
import argparse
import collections
import json
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from auto_handeye_capture import PoseStream, capture
from calib_utils import detect_marker, load_json, make_detector, make_T, rot_angle_deg, save_json
from wrist_handeye import WristCam

DATA = Path(__file__).resolve().parent / 'data'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--wrist_result', type=Path, default=None, help='default: newest data/wrist_handeye/*/result.json')
    p.add_argument('--t_base_cam', type=Path, nargs='*', default=None,
                   help='variants to compare (default: every T_base_cam*.json of the newest real_handeye session)')
    p.add_argument('--serial', default='338122303684', help='WRIST camera serial')
    p.add_argument('--marker_id', type=int, default=1)
    p.add_argument('--marker_size', type=float, default=0.045)
    p.add_argument('--marker_scale', type=float, default=None,
                   help='override the marker_scale stored in the wrist result (must match how it was solved)')
    p.add_argument('--cube_edge', type=float, default=0.057)
    p.add_argument('--dictionary', default='DICT_4X4_50')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--joint_port', type=int, default=5006)
    p.add_argument('--pose_port', type=int, default=5005)
    p.add_argument('--stable_n', type=int, default=8)
    p.add_argument('--stable_mm', type=float, default=3.0)
    p.add_argument('--stable_deg', type=float, default=3.0)
    p.add_argument('--n_frames', type=int, default=20)
    p.add_argument('--max_jitter_mm', type=float, default=1.0)
    p.add_argument('--min_tilt_deg', type=float, default=10.0)
    p.add_argument('--still_mm', type=float, default=0.3)
    p.add_argument('--still_deg', type=float, default=0.1)
    p.add_argument('--out', type=Path, default=None)
    return p.parse_args(argv)


def newest(pattern):
    found = sorted(DATA.glob(pattern))
    if not found:
        raise SystemExit(f'nothing matches data/{pattern}')
    return found[-1]


class FpStream:
    """Mesh-to-camera poses from main.py (UDP 5005); stable = last N poses agree in the camera frame."""

    def __init__(self, port, n, mm, deg):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', port))
        self.sock.setblocking(False)
        self.hist, self.n, self.mm, self.deg = collections.deque(maxlen=n), n, mm, deg
        self.last = 0.0
        self.last_rx = 0.0

    def poll(self):
        while True:
            try:
                msg = json.loads(self.sock.recv(65535))
                T = np.asarray(msg['pose'], dtype=float)
                stamp = float(msg['timestamp']['sec']) + float(msg['timestamp']['nanosec']) / 1e9
            except BlockingIOError:
                return
            except (ValueError, KeyError, TypeError):
                continue
            if (T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1])
                    or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-3) or stamp <= self.last):
                self.hist.clear()
                continue
            if self.last and stamp - self.last > 1.0:
                self.hist.clear()
            self.last, self.last_rx = stamp, time.time()
            self.hist.append(T)

    def stable(self):
        """Median camera-frame pose, or None when old / not enough / not agreeing."""
        if len(self.hist) < self.n or time.time() - self.last_rx > 1.0:
            return None
        ref = self.hist[-1]
        if (max(np.linalg.norm(h[:3, 3] - ref[:3, 3]) * 1000 for h in self.hist) > self.mm
                or max(rot_angle_deg(h[:3, :3].T @ ref[:3, :3]) for h in self.hist) > self.deg):
            return None
        out = ref.copy()
        out[:3, 3] = np.median([h[:3, 3] for h in self.hist], axis=0)
        return out


def compare(T_fp, T_gt):
    """Position error vector (mm, base frame), top-face normal angle (deg), yaw about the normal mod 90 (deg)."""
    d = (T_fp[:3, 3] - T_gt[:3, 3]) * 1000
    n = T_gt[:3, 2]
    k = int(np.argmax(np.abs(T_fp[:3, :3].T @ n)))                 # FP cube axis that is the face normal
    ax = T_fp[:3, k] * np.sign(T_fp[:3, k] @ n)
    tilt = float(np.degrees(np.arccos(np.clip(ax @ n, -1, 1))))
    others = [i for i in range(3) if i != k]
    v = T_fp[:3, others[0]] - (T_fp[:3, others[0]] @ n) * n
    yaw = float(np.degrees(np.arctan2(v @ np.cross(n, T_gt[:3, 0]), v @ T_gt[:3, 0])))
    return d, tilt, (yaw + 45) % 90 - 45


def summary(positions, variants):
    lines = [f'{len(positions)} cube positions (error = FoundationPose - GT, base frame, mm)']
    for name in variants:
        lines.append(f'\n== T_base_cam: {name}')
        lines.append(' #   GT x/y/z (mm)           dx      dy      dz   horiz   total  normal_deg  yaw90_deg')
        rows = []
        for i, pos in enumerate(positions):
            gt = np.array(pos['T_base_cube_gt'])
            d, tilt, yaw = compare(np.array(pos['fp'][name]['T_base_cube']), gt)
            rows.append((d, tilt, yaw))
            lines.append(f'{i:2d}  {gt[0, 3] * 1000:7.1f} {gt[1, 3] * 1000:6.1f} {gt[2, 3] * 1000:5.1f}  '
                         f'{d[0]:7.1f} {d[1]:7.1f} {d[2]:7.1f} {np.hypot(d[0], d[1]):7.1f} {np.linalg.norm(d):7.1f}  '
                         f'{tilt:8.1f} {yaw:10.1f}')
        D = np.array([r[0] for r in rows])
        lines.append(f'mean error  : dx {D[:, 0].mean():+.1f} dy {D[:, 1].mean():+.1f} dz {D[:, 2].mean():+.1f} mm '
                     f'(constant bias if these are large and the scatter below is small)')
        lines.append(f'scatter (std): dx {D[:, 0].std():.1f} dy {D[:, 1].std():.1f} dz {D[:, 2].std():.1f} mm   '
                     f'mean horiz {np.mean(np.hypot(D[:, 0], D[:, 1])):.1f}  mean total {np.mean(np.linalg.norm(D, axis=1)):.1f} '
                     f'max total {np.max(np.linalg.norm(D, axis=1)):.1f} mm')
    return '\n'.join(lines)


def main():
    args = parse_args()
    wrist_path = args.wrist_result or newest('wrist_handeye/*/result.json')
    wr = load_json(wrist_path)
    X, scale = np.array(wr['T_flange_cam']), wr['meta'].get('marker_scale', 1.0)
    if args.marker_scale is not None:
        scale = args.marker_scale
    paths = args.t_base_cam or sorted(newest('real_handeye/*/T_base_cam.json').parent.glob('T_base_cam*.json'))
    variants = {p.stem: np.array(load_json(p)['T_base_cam']) for p in paths}
    out = args.out or DATA / 'gt_compare' / time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    print(f'wrist result {wrist_path} (fit {wr["pos_err_mm"]["mean"]:.2f} mm, marker_scale {scale}, marker '
          f'{args.marker_size * 1000:.1f} mm, cube {args.cube_edge * 1000:.0f} mm)\nT_base_cam variants: {list(variants)}\n'
          f'out {out}', flush=True)

    robot, fp = PoseStream(args.joint_port), FpStream(args.pose_port, args.stable_n, args.stable_mm, args.stable_deg)
    cam, detector = WristCam(args.serial, args.width, args.height, args.fps), make_detector(args.dictionary)
    cap_args = SimpleNamespace(n_frames=args.n_frames, marker_id=args.marker_id, marker_size=args.marker_size,
                               still_mm=args.still_mm, still_deg=args.still_deg, max_jitter_mm=args.max_jitter_mm,
                               min_tilt_deg=args.min_tilt_deg)
    positions, gt, fpv = [], None, None
    log = lambda m: print(m, flush=True)  # noqa: E731
    try:
        while True:
            fp.poll()
            frame = cam.read()
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            det = detect_marker(bgr, frame.K, detector, args.marker_id, args.marker_size)
            if det is not None:
                cv2.polylines(bgr, [det['corners'].astype(np.int32)], True, (0, 255, 0), 2)
            stab = fp.stable()
            text = (f'pos {len(positions)}  GT {"OK" if gt else "-"}  FP {"OK" if fpv else "-"}  '
                    f'robot still {robot.still_for(args.still_mm, args.still_deg):.1f}s  '
                    f'FP stream {"STABLE" if stab is not None else "not stable/none"}  '
                    + (f'marker tilt {det["tilt_deg"]:.0f}' if det else 'marker not seen'))
            cv2.putText(bgr, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow('cube_gt_compare', bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('g'):
                sample, err = capture(cam, detector, cap_args, robot, log)
                if sample is None:
                    log(f'GT failed: {err}')
                    continue
                Tcm = np.array(sample['T_cam_marker'])
                Tcm[:3, 3] *= scale
                T_marker = np.array(sample['T_base_flange']) @ X @ Tcm
                T_cube = T_marker @ make_T(np.eye(3), [0, 0, -args.cube_edge / 2])
                gt = dict(T_base_marker=T_marker.tolist(), T_base_cube_gt=T_cube.tolist(), posx=sample['posx'],
                          dist_m=sample['dist_m'], tilt_deg=sample['tilt_deg'], jitter_mm=sample['jitter_mm'])
                log(f'GT cube centre (base, mm): {np.round(T_cube[:3, 3] * 1000, 1).tolist()}  '
                    f'(marker top z {T_marker[2, 3] * 1000:.1f}; normal tilt from vertical '
                    f'{np.degrees(np.arccos(np.clip(T_marker[2, 2], -1, 1))):.1f} deg)  dist {sample["dist_m"]:.2f} m')
            if key == ord('f'):
                if stab is None:
                    log('FP not stable yet (main.py running? cube visible? robot out of the way?)')
                    continue
                fpv = dict(T_cam_cube=stab.tolist(), **{name: dict(T_base_cube=(Tbc @ stab).tolist())
                                                          for name, Tbc in variants.items()})
                log('FP cube centre (base, mm) per T_base_cam: ' + '; '.join(
                    f'{n}: {np.round((Tbc @ stab)[:3, 3] * 1000, 1).tolist()}' for n, Tbc in variants.items()))
            if key == ord('n'):
                if gt is None or fpv is None:
                    log('need both G and F for this position first')
                    continue
                positions.append(dict(T_base_cube_gt=gt['T_base_cube_gt'], gt=gt, T_cam_cube=fpv['T_cam_cube'],
                                      fp={n: fpv[n] for n in variants}, time=time.strftime('%H:%M:%S')))
                gt = fpv = None
                save_json(out / 'positions.json', dict(positions=positions, wrist_result=str(wrist_path),
                                                       marker_scale=scale, cube_edge=args.cube_edge))
                log(f'stored position {len(positions) - 1}. Move the cube to a new place, then G / F again.')
                log(summary(positions, list(variants)))
            if key == ord('u') and positions:
                positions.pop()
                save_json(out / 'positions.json', dict(positions=positions, wrist_result=str(wrist_path),
                                                       marker_scale=scale, cube_edge=args.cube_edge))
                log(f'undo -> {len(positions)} positions')
            if key == ord('s') and positions:
                log(summary(positions, list(variants)))
    finally:
        cam.close()
        cv2.destroyAllWindows()
        if positions:
            text = summary(positions, list(variants))
            (out / 'summary.txt').write_text(text)
            log(text + f'\nsaved {out}')


if __name__ == '__main__':
    main()
