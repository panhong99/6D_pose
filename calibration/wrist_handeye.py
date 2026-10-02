"""Eye-in-hand calibration of the WRIST D455 (T_flange_cam) with one fixed ArUco marker.  Read-only.

    conda terminal:  conda run -n foundationpose python calibration/wrist_handeye.py --marker_size 0.045
    ROS2 terminal:   python3 calibration/ros2_flange_udp.py        (bringup running; read-only pose stream, UDP 5006)
    re-solve later:  python calibration/wrist_handeye.py --solve calibration/data/wrist_handeye/<time>/samples.json

YOU move the robot (pendant, slowest speed); this script only watches.  The marker (the one on the cube) stays
FIXED in the world, so for every sample  T_base_flange @ T_flange_cam @ T_cam_marker = T_base_marker  (constant).
The solve returns T_flange_cam and that constant T_base_marker; the residual is how far the per-sample
T_base_marker scatters (no ground truth needed).  The cube must not move during capture.

A sample is taken automatically when the robot has been still for --still_s, the pose differs from every stored
sample by >= --min_new_mm or >= --min_new_deg, the marker is seen obliquely (tilt >= --min_tilt_deg) and steadily.
Keys: S solve | U undo last | P pause/resume auto | Q/ESC quit.
Output: data/wrist_handeye/<YYYYmmdd_HHMMSS>/{samples.json, result.json}.  Nothing is sent to the robot or ROS.
"""
import argparse
import sys
import time
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'real_time_project'))
import pyrealsense2 as rs  # noqa: E402
from real_time_utils import rectification_maps  # noqa: E402

from auto_handeye_capture import PoseStream, capture, novelty  # noqa: E402
from calib_utils import (_park_martin, average_poses, detect_marker, inv_T, load_json, make_detector, make_T,  # noqa: E402
                         posx_to_T, rot_angle_deg, rotation_spread_deg, save_json)

DATA = Path(__file__).resolve().parent / 'data'
Frame = namedtuple('Frame', 'rgb K')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--solve', type=Path, default=None, help='samples.json to (re-)solve; no camera is opened')
    p.add_argument('--serial', default='338122303684', help='WRIST camera serial')
    p.add_argument('--udp_port', type=int, default=5006)
    p.add_argument('--marker_id', type=int, default=1)
    p.add_argument('--marker_size', type=float, default=0.045, help='black-square edge, m (MEASURE the printout)')
    p.add_argument('--dictionary', default='DICT_4X4_50')
    p.add_argument('--marker_scale', type=float, default=1.0,
                   help='solve-time correction: multiplies the camera->marker distance (= true printed edge / --marker_size)')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--n_frames', type=int, default=20, help='frames averaged per sample')
    p.add_argument('--max_jitter_mm', type=float, default=1.0)
    p.add_argument('--min_tilt_deg', type=float, default=15.0)
    p.add_argument('--max_dist', type=float, default=0.7,
                   help='ignore samples farther than this (m): a 45 mm marker is inaccurate beyond ~0.7 m; 0 = no limit')
    p.add_argument('--min_dist', type=float, default=0.3, help='ignore samples closer than this (m)')
    p.add_argument('--still_s', type=float, default=1.5)
    p.add_argument('--still_mm', type=float, default=0.3)
    p.add_argument('--still_deg', type=float, default=0.1)
    p.add_argument('--min_new_mm', type=float, default=20.0)
    p.add_argument('--min_new_deg', type=float, default=8.0)
    p.add_argument('--target_samples', type=int, default=20)
    p.add_argument('--session', type=Path, default=None, help='existing session dir = resume (adds to its samples.json)')
    return p.parse_args(argv)


class WristCam:
    """Colour-only D455 stream, undistorted to the pinhole K (same mapping the other scripts use)."""

    def __init__(self, serial, width, height, fps):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        profile = self.pipeline.start(cfg)
        try:
            print('Camera:', profile.get_device().get_info(rs.camera_info.name), serial, flush=True)
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float32)
            self.map1, self.map2 = rectification_maps(intr, self.K)
        except Exception:
            self.pipeline.stop()
            raise

    def read(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        while True:
            newer = self.pipeline.poll_for_frames()
            if not newer:
                break
            frames = newer
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError('no colour frame')
        rgb = np.asanyarray(color.get_data()).copy()
        if self.map1 is not None:
            rgb = cv2.remap(rgb, self.map1, self.map2, cv2.INTER_LINEAR)
        return Frame(rgb, self.K)

    def close(self):
        self.pipeline.stop()


# ------------------------------------------------------------------ solver

def _residual(x, Tbf, Tcm, rot_w):
    X, Z = make_T(R.from_rotvec(x[:3]).as_matrix(), x[3:6]), make_T(R.from_rotvec(x[6:9]).as_matrix(), x[9:12])
    Zi = inv_T(Z)
    out = []
    for a, b in zip(Tbf, Tcm):
        d = Zi @ a @ X @ b
        out += [R.from_matrix(d[:3, :3]).as_rotvec() * rot_w, d[:3, 3]]
    return np.concatenate(out)


def solve_eye_in_hand(Tbf, Tcm, refine=True):
    """(T_flange_cam, T_base_marker) from N >= 3 samples of a fixed marker."""
    Tbf, Tcm = [np.asarray(T, dtype=np.float64) for T in Tbf], [np.asarray(T, dtype=np.float64) for T in Tcm]
    if len(Tbf) < 3:
        raise ValueError('need at least 3 samples')
    if rotation_spread_deg(Tbf) < 15.0:
        raise ValueError('flange orientations are too similar: rotate the flange more between samples')
    # Tbf_i X Tcm_i = Tbf_j X Tcm_j  ->  (Tbf_i^-1 Tbf_j) X = X (Tcm_i Tcm_j^-1)
    A_list, B_list = [], []
    for i in range(len(Tbf)):
        for j in range(i + 1, len(Tbf)):
            A, B = inv_T(Tbf[i]) @ Tbf[j], Tcm[i] @ inv_T(Tcm[j])
            if rot_angle_deg(A[:3, :3]) >= 3.0 and rot_angle_deg(B[:3, :3]) >= 3.0:
                A_list.append(A)
                B_list.append(B)
    if len(A_list) < 3:
        raise ValueError('poses are too similar: rotate the flange more between samples')
    X = _park_martin(A_list, B_list)
    Z = average_poses([a @ X @ b for a, b in zip(Tbf, Tcm)])
    if refine:
        x0 = np.concatenate([R.from_matrix(X[:3, :3]).as_rotvec(), X[:3, 3],
                             R.from_matrix(Z[:3, :3]).as_rotvec(), Z[:3, 3]])
        x = least_squares(_residual, x0, args=(Tbf, Tcm, 0.1), loss='soft_l1', f_scale=0.005).x
        X = make_T(R.from_rotvec(x[:3]).as_matrix(), x[3:6])
        Z = make_T(R.from_rotvec(x[6:9]).as_matrix(), x[9:12])
    return X, Z


def evaluate(Tbf, Tcm, X, Z):
    """Per-sample disagreement of T_base_marker with the solved constant Z (pos mm, rot deg)."""
    pos, rot = [], []
    for a, b in zip(Tbf, Tcm):
        m = a @ X @ b
        pos.append(np.linalg.norm(m[:3, 3] - Z[:3, 3]) * 1000)
        rot.append(rot_angle_deg(Z[:3, :3].T @ m[:3, :3]))
    return np.array(pos), np.array(rot)


def leave_one_out(Tbf, Tcm):
    pos, rot = [], []
    for k in range(len(Tbf)):
        rest = [i for i in range(len(Tbf)) if i != k]
        X, Z = solve_eye_in_hand([Tbf[i] for i in rest], [Tcm[i] for i in rest])
        p, r = evaluate([Tbf[k]], [Tcm[k]], X, Z)
        pos.append(p[0])
        rot.append(r[0])
    return np.array(pos), np.array(rot)


def solve_report(samples, out, args, log):
    Tbf = [np.array(s['T_base_flange']) for s in samples]
    Tcm = [np.array(s['T_cam_marker']) for s in samples]
    for T in Tcm:
        T[:3, 3] *= args.marker_scale
    if len(samples) < 6:
        log(f'need >= 6 samples to solve (have {len(samples)})')
        return
    try:
        X, Z = solve_eye_in_hand(Tbf, Tcm)
        loo_p, loo_r = leave_one_out(Tbf, Tcm)
    except ValueError as exc:
        log(f'solve failed: {exc}')
        return
    pos, rot = evaluate(Tbf, Tcm, X, Z)
    log(f'\n{len(samples)} samples, flange rotation spread {rotation_spread_deg(Tbf):.0f} deg')
    log(' #   fit_mm  fit_deg   LOO_mm  LOO_deg  tilt  dist_m')
    for i, s in enumerate(samples):
        log(f'{i:2d} {pos[i]:8.2f} {rot[i]:8.2f} {loo_p[i]:8.2f} {loo_r[i]:8.2f}  {s["tilt_deg"]:4.0f}  {s["dist_m"]:.3f}')
    log(f'fit residual  : pos mean {pos.mean():.2f} mm (max {pos.max():.2f}), rot mean {rot.mean():.2f} deg (max {rot.max():.2f})')
    log(f'leave-one-out : pos mean {loo_p.mean():.2f} mm (median {np.median(loo_p):.2f}, max {loo_p.max():.2f}), '
        f'rot mean {loo_r.mean():.2f} deg (max {loo_r.max():.2f})')
    np.set_printoptions(precision=5, suppress=True)
    log(f'T_flange_cam (camera in the flange frame, m):\n{X}')
    log(f'T_base_marker (marker centre/orientation in the robot base, m):\n{Z}')
    res = dict(T_flange_cam=X.tolist(), T_base_marker=Z.tolist(), n_samples=len(samples),
               pos_err_mm=dict(mean=float(pos.mean()), max=float(pos.max())),
               rot_err_deg=dict(mean=float(rot.mean()), max=float(rot.max())),
               leave_one_out=dict(pos_mm=loo_p.tolist(), rot_deg=loo_r.tolist()),
               created=time.strftime('%Y-%m-%d %H:%M:%S'),
               meta=dict(marker_id=args.marker_id, marker_size=args.marker_size, marker_scale=args.marker_scale, dictionary=args.dictionary,
                         serial=args.serial, resolution=[args.width, args.height],
                         mode='eye-in-hand, fixed marker, human-moved robot, ROS2 read-only pose'))
    save_json(out, res)
    log(f'saved: {out}  (not validated: check the residuals and compare with an independent measurement)')


# ------------------------------------------------------------------ selftest / main

def _selftest():
    rng = np.random.default_rng(1)
    X = make_T(R.from_euler('xyz', [5, -8, 12], degrees=True).as_matrix(), [0.03, -0.05, 0.08])
    Z = make_T(R.from_euler('xyz', [0, 180, 20], degrees=True).as_matrix(), [1.0, 0.05, 0.03])
    Tbf, Tcm = [], []
    for _ in range(14):
        rot = R.from_euler('xyz', rng.uniform(-35, 35, 3) + [0, 180, 0], degrees=True).as_matrix()
        a = make_T(rot, Z[:3, 3] + rng.uniform(-0.1, 0.1, 3) + [0, 0, 0.35])
        Tbf.append(a)
        Tcm.append(inv_T(a @ X) @ Z)
    Xs, Zs = solve_eye_in_hand(Tbf, Tcm)
    return (np.linalg.norm(Xs[:3, 3] - X[:3, 3]) * 1000, rot_angle_deg(Xs[:3, :3].T @ X[:3, :3]),
            np.linalg.norm(Zs[:3, 3] - Z[:3, 3]) * 1000)


def main():
    args = parse_args()
    if args.solve:
        samples = [x for x in load_json(args.solve)['samples']
                   if x['dist_m'] <= (args.max_dist or 1e9) and x['dist_m'] >= args.min_dist]
        print(f'{len(samples)} samples with {args.min_dist} <= dist <= {args.max_dist} m')
        solve_report(samples, args.solve.with_name('result.json'), args, print)
        return
    session = args.session or DATA / 'wrist_handeye' / time.strftime('%Y%m%d_%H%M%S')
    session.mkdir(parents=True, exist_ok=True)
    logf = open(session / 'capture.log', 'a')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    robot = PoseStream(args.udp_port)
    cam = WristCam(args.serial, args.width, args.height, args.fps)
    detector = make_detector(args.dictionary)
    samples = load_json(session / 'samples.json')['samples'] if (session / 'samples.json').is_file() else []
    paused, cooldown, reported = False, 0.0, False
    log(f'session {session}  marker {args.marker_size * 1000:.1f} mm id {args.marker_id}  wrist {args.serial}')
    log('Move the robot with the pendant (slowest). View the marker obliquely from 10-40 cm, vary the flange '
        'ROTATION a lot (tilt about all axes), hold still ~2 s per pose. Keep the cube fixed.')
    try:
        while True:
            frame = cam.read()
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            det = detect_marker(bgr, frame.K, detector, args.marker_id, args.marker_size)
            pose, still = robot.latest(), robot.still_for(args.still_mm, args.still_deg)
            if det is not None:
                cv2.polylines(bgr, [det['corners'].astype(np.int32)], True, (0, 255, 0), 2)
            status = (f'samples {len(samples)}/{args.target_samples}  '
                      f'{"PAUSED " if paused else ""}{"NO ROBOT POSE " if pose is None else ""}still {still:.1f}s  '
                      + (f'tilt {det["tilt_deg"]:.0f} dist {np.linalg.norm(det["T"][:3, 3]):.2f}m' if det else 'marker not seen'))
            cv2.putText(bgr, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if (not paused and det is not None and pose is not None and still >= args.still_s
                    and time.time() > cooldown and det['tilt_deg'] >= args.min_tilt_deg
                    and args.min_dist <= np.linalg.norm(det['T'][:3, 3]) <= (args.max_dist or 1e9)):
                d_mm, d_deg = novelty(posx_to_T(pose[1]), samples)
                if d_mm >= args.min_new_mm or d_deg >= args.min_new_deg:
                    sample, err = capture(cam, detector, args, robot, log)
                    if sample is None:
                        log(f'  skipped: {err}')
                        cooldown = time.time() + 2.0
                    else:
                        samples.append(sample)
                        save_json(session / 'samples.json', dict(samples=samples))
                        log(f'  sample {len(samples) - 1}: tilt {sample["tilt_deg"]:.0f} dist {sample["dist_m"]:.3f} m '
                            f'jitter {sample["jitter_mm"]:.2f} mm  posx {np.round(sample["posx"], 1).tolist()}')
            cv2.imshow('wrist_handeye', bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('p'):
                paused = not paused
            if key == ord('u') and samples:
                samples.pop()
                save_json(session / 'samples.json', dict(samples=samples))
                log(f'undo -> {len(samples)} samples')
            if key == ord('s') or (len(samples) >= args.target_samples and not reported):
                reported = True
                solve_report(samples, session / 'result.json', args, log)
    finally:
        cam.close()
        cv2.destroyAllWindows()
        if len(samples) >= 6:
            solve_report(samples, session / 'result.json', args, log)
        log(f'session dir: {session}')


if __name__ == '__main__':
    main()
