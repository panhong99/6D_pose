"""Hands-free real eye-to-hand capture: YOU move the robot, this script only watches.

    ./calibration/run_real_handeye_auto.sh                   (starts the pose stream + this script)
    conda run -n foundationpose python calibration/auto_handeye_capture.py --marker_size 0.10

Robot pose comes from ros2_flange_udp.py (read-only get_current_tool_flange_posx(DR_BASE) over
localhost UDP 5006), so the camera view and robot pose can never get out of step the way typed
pendant values did (2026-10-01: from sample #12 on the typed pose belonged to the next view).

A sample is captured automatically when ALL hold:
  * robot pose unchanged for --still_s (<= --still_mm / --still_deg),
  * it differs from every stored sample by >= --min_new_mm or >= --min_new_deg,
  * the marker is visible with tilt >= --min_tilt_deg, frame jitter <= --max_jitter_mm,
  * the robot did not move during the averaging (pose re-read after the frames),
  * pairing check: median |angle(robot motion) - angle(marker motion)| to the stored samples
    <= --max_pair_mismatch_deg (needs >= 3 stored samples), and the stored samples' solve predicts
    the new marker position within --max_pred_mm (needs >= 6 stored samples).
At --target_samples (and on S) it solves and reports leave-one-out residuals.
Keys: S solve | U undo last | P pause/resume auto | Q/ESC quit.
Output: data/real_handeye/<YYYYmmdd_HHMMSS>/{samples.json, T_base_cam.json, capture.log}.
It never writes data/handeye_samples.json, data/T_base_cam.json or data/eye_to_hand/*.
"""
import argparse
import collections
import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

from calib_utils import (average_poses, detect_marker, evaluate, leave_one_out, load_json, make_detector,
                         pair_angle_mismatch_deg, posx_to_T, result_dict, rot_angle_deg, rotation_spread_deg,
                         save_json, solve_eye_to_hand)

DATA = Path(__file__).resolve().parent / 'data'
PROTECTED = {DATA / 'handeye_samples.json', DATA / 'T_base_cam.json',
             DATA / 'eye_to_hand' / 'result.json', DATA / 'eye_to_hand' / 'T_base_cam.txt'}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--udp_port', type=int, default=5006)
    p.add_argument('--marker_id', type=int, default=1)
    p.add_argument('--marker_size', type=float, required=True, help='black-square edge, m (MEASURE it)')
    p.add_argument('--dictionary', default='DICT_4X4_50')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--serial', default='338122300585')
    p.add_argument('--n_frames', type=int, default=20, help='frames averaged per capture')
    p.add_argument('--max_jitter_mm', type=float, default=1.5)
    p.add_argument('--min_tilt_deg', type=float, default=15.0)
    p.add_argument('--still_s', type=float, default=1.5)
    p.add_argument('--still_mm', type=float, default=0.3)
    p.add_argument('--still_deg', type=float, default=0.1)
    p.add_argument('--min_new_mm', type=float, default=40.0)
    p.add_argument('--min_new_deg', type=float, default=10.0)
    p.add_argument('--max_pair_mismatch_deg', type=float, default=8.0)
    p.add_argument('--max_pred_mm', type=float, default=25.0,
                   help='reject if the stored samples predict this marker position worse than this (>= 6 stored)')
    p.add_argument('--target_samples', type=int, default=25)
    p.add_argument('--session', type=Path, default=None, help='session dir (default: new data/real_handeye/<time>)')
    return p.parse_args(argv)


class PoseStream:
    """Latest flange posx from ros2_flange_udp.py plus a short history for the stillness test."""

    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('127.0.0.1', port))
        self.hist = collections.deque(maxlen=400)
        self.lock = threading.Lock()
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        while True:
            try:
                msg = json.loads(self.sock.recv(4096))
                posx = [float(v) for v in msg['posx']]
                assert len(posx) == 6
            except (ValueError, KeyError, TypeError, AssertionError):
                continue
            with self.lock:
                self.hist.append((time.time(), posx, msg.get('posj')))

    def latest(self, max_age=0.5):
        with self.lock:
            if not self.hist or time.time() - self.hist[-1][0] > max_age:
                return None
            return self.hist[-1]

    def still_for(self, mm, deg):
        """Seconds the pose has stayed within mm/deg of the newest pose (0 if no fresh data)."""
        with self.lock:
            hist = list(self.hist)
        if not hist or time.time() - hist[-1][0] > 0.5:
            return 0.0
        T0, t_still = posx_to_T(hist[-1][1]), hist[-1][0]
        for t, posx, _ in reversed(hist):
            T = posx_to_T(posx)
            if (np.linalg.norm(T[:3, 3] - T0[:3, 3]) * 1000 > mm
                    or rot_angle_deg(T[:3, :3].T @ T0[:3, :3]) > deg):
                break
            t_still = t
        return time.time() - t_still


def novelty(Tbf, samples):
    """(min translation mm, rotation deg of that nearest sample) to stored samples."""
    best = (np.inf, np.inf)
    for s in samples:
        T = np.array(s['T_base_flange'])
        d = (np.linalg.norm(T[:3, 3] - Tbf[:3, 3]) * 1000, rot_angle_deg(T[:3, :3].T @ Tbf[:3, :3]))
        if max(d[0] / 40.0, d[1] / 10.0) < max(best[0] / 40.0, best[1] / 10.0):
            best = d
    return best


def predicted_error_mm(sample, samples):
    """Marker position disagreement of a new sample under a solve of the stored samples only."""
    try:
        Tbc, Tfm = solve_eye_to_hand([np.array(s['T_base_flange']) for s in samples],
                                     [np.array(s['T_cam_marker']) for s in samples])
    except ValueError:
        return None
    pos, _ = evaluate([np.array(sample['T_base_flange'])], [np.array(sample['T_cam_marker'])], Tbc, Tfm)
    return float(pos[0])


def capture(camera, detector, args, robot, log):
    import cv2
    start = robot.latest()
    if start is None:
        return None, 'no fresh robot pose'
    Ts, tilts, deadline = [], [], time.time() + 6.0
    while len(Ts) < args.n_frames and time.time() < deadline:
        frame = camera.read()
        det = detect_marker(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), frame.K, detector,
                            args.marker_id, args.marker_size)
        if det is not None:
            Ts.append(det['T'])
            tilts.append(det['tilt_deg'])
    end = robot.latest()
    if end is None:
        return None, 'robot pose stream stopped'
    if len(Ts) < 0.6 * args.n_frames:
        return None, f'marker seen in only {len(Ts)}/{args.n_frames} frames'
    a, b = posx_to_T(start[1]), posx_to_T(end[1])
    if (np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000 > args.still_mm
            or rot_angle_deg(a[:3, :3].T @ b[:3, :3]) > args.still_deg):
        return None, 'robot moved during capture'
    T = average_poses(Ts)
    jitter = float(np.std([t[:3, 3] for t in Ts], axis=0).mean() * 1000)
    tilt = float(np.mean(tilts))
    if jitter > args.max_jitter_mm:
        return None, f'marker jitter {jitter:.2f} mm'
    if tilt < args.min_tilt_deg:
        return None, f'marker too head-on (tilt {tilt:.0f} deg)'
    return dict(posx=end[1], posj=end[2], T_base_flange=b.tolist(), T_cam_marker=T.tolist(), jitter_mm=jitter,
                jitter_deg=float(np.mean([rot_angle_deg(T[:3, :3].T @ t[:3, :3]) for t in Ts])),
                tilt_deg=tilt, dist_m=float(np.linalg.norm(T[:3, 3])), source='ros2 get_current_tool_flange_posx',
                robot_stamp=end[0], time=time.strftime('%H:%M:%S')), None


def solve_report(samples, out, args, log):
    Tbf = [np.array(s['T_base_flange']) for s in samples]
    Tcm = [np.array(s['T_cam_marker']) for s in samples]
    if len(samples) < 6:
        log(f'need >= 6 samples to solve (have {len(samples)})')
        return
    try:
        Tbc, Tfm = solve_eye_to_hand(Tbf, Tcm)
        loo_p, loo_r = leave_one_out(Tbf, Tcm)
    except ValueError as exc:
        log(f'solve failed: {exc}')
        return
    pos, rot = evaluate(Tbf, Tcm, Tbc, Tfm)
    log(f'\n{len(samples)} samples, flange rotation spread {rotation_spread_deg(Tbf):.0f} deg')
    log(' #   fit_mm  fit_deg   LOO_mm  LOO_deg  tilt  dist_m')
    for i, s in enumerate(samples):
        log(f'{i:2d} {pos[i]:8.2f} {rot[i]:8.2f} {loo_p[i]:8.2f} {loo_r[i]:8.2f}  {s["tilt_deg"]:4.0f}  {s["dist_m"]:.3f}')
    log(f'fit residual   : pos mean {pos.mean():.2f} mm (max {pos.max():.2f}), rot mean {rot.mean():.2f} deg (max {rot.max():.2f})')
    log(f'leave-one-out  : pos mean {loo_p.mean():.2f} mm (median {np.median(loo_p):.2f}, max {loo_p.max():.2f}), '
        f'rot mean {loo_r.mean():.2f} deg (max {loo_r.max():.2f})')
    np.set_printoptions(precision=5, suppress=True)
    log(f'T_base_cam:\n{Tbc}')
    res = result_dict(Tbc, Tfm, pos, rot, len(samples), dict(
        marker_id=args.marker_id, marker_size=args.marker_size, dictionary=args.dictionary,
        resolution=[args.width, args.height], mode='auto_capture (human-moved robot, ROS2 read-only pose)'))
    res['leave_one_out'] = dict(pos_mm=loo_p.tolist(), rot_deg=loo_r.tolist())
    save_json(out, res)
    log(f'saved: {out}  (not validated: check against an independent measurement before use)')


def main():
    args = parse_args()
    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'real_time_project'))
    from real_time_utils import D455Source

    session = args.session or DATA / 'real_handeye' / time.strftime('%Y%m%d_%H%M%S')
    session.mkdir(parents=True, exist_ok=True)
    samples_path, out_path, log_path = session / 'samples.json', session / 'T_base_cam.json', session / 'capture.log'
    if {samples_path.resolve(), out_path.resolve()} & {p.resolve() for p in PROTECTED}:
        raise SystemExit('refusing to write a protected file')
    samples = load_json(samples_path) if samples_path.exists() else []
    logf = open(log_path, 'a')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    log(f'session {session} ({len(samples)} samples loaded) | marker {args.marker_size} m id {args.marker_id}')
    robot = PoseStream(args.udp_port)
    detector = make_detector(args.dictionary)
    camera = D455Source(args.width, args.height, args.fps, args.serial)
    flash_until = 0.0
    paused, last_reason, solved_at = False, '', len(samples) if len(samples) >= args.target_samples else -1
    log('Move the robot yourself. S solve | U undo | P pause | Q quit')
    try:
        while True:
            frame = camera.read()
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            det = detect_marker(bgr, frame.K, detector, args.marker_id, args.marker_size)
            pose = robot.latest()
            still = robot.still_for(args.still_mm, args.still_deg)
            state, colour = '', (0, 255, 255)
            if pose is None:
                state, colour = 'NO ROBOT POSE (start ros2_flange_udp.py)', (0, 0, 255)
            elif det is None:
                state = 'marker NOT visible'
            elif det['tilt_deg'] < args.min_tilt_deg:
                state = f'tilt {det["tilt_deg"]:.0f} deg < {args.min_tilt_deg:.0f}: tilt the flange'
            else:
                Tbf = posx_to_T(pose[1])
                dmm, ddeg = novelty(Tbf, samples)
                if dmm < args.min_new_mm and ddeg < args.min_new_deg:
                    state = f'too close to a stored sample ({dmm:.0f} mm, {ddeg:.0f} deg): move on'
                elif still < args.still_s:
                    state = f'hold still {still:.1f}/{args.still_s:.1f} s'
                elif paused:
                    state = 'PAUSED (P)'
                else:
                    sample, reason = capture(camera, detector, args, robot, log)
                    if sample is not None and len(samples) >= 3:
                        mis = pair_angle_mismatch_deg(np.array(sample['T_base_flange']), np.array(sample['T_cam_marker']),
                                                      [np.array(s['T_base_flange']) for s in samples],
                                                      [np.array(s['T_cam_marker']) for s in samples])
                        sample['pair_mismatch_deg'] = mis
                        if mis > args.max_pair_mismatch_deg:
                            reason, sample = f'pairing check failed ({mis:.1f} deg): marker slipped?', None
                    if sample is not None and len(samples) >= 6:
                        pred = predicted_error_mm(sample, samples)
                        sample['predicted_error_mm'] = pred
                        if pred is not None and pred > args.max_pred_mm:
                            reason, sample = f'disagrees with stored samples by {pred:.0f} mm: marker slipped?', None
                    if sample is None:
                        if reason != last_reason:
                            log(f'rejected: {reason}')
                        last_reason = reason
                        state, colour = f'rejected: {reason}', (0, 0, 255)
                    else:
                        samples.append(sample)
                        save_json(samples_path, samples)
                        last_reason = ''
                        log(f'captured #{len(samples) - 1}: tilt {sample["tilt_deg"]:.0f} deg, dist {sample["dist_m"]:.3f} m, '
                            f'jitter {sample["jitter_mm"]:.2f} mm, pair {sample.get("pair_mismatch_deg") or 0:.1f} deg, '
                            f'posx {np.round(sample["posx"], 2).tolist()}')
                        flash_until = time.time() + 2.0
                        state, colour = f'CAPTURED #{len(samples) - 1} - move to the next pose', (0, 255, 0)
                        if len(samples) >= args.target_samples and solved_at != len(samples):
                            solve_report(samples, out_path, args, log)
                            solved_at = len(samples)
            if det is not None:
                cv2.polylines(bgr, [det['corners'].astype(np.int32)], True, (0, 255, 0), 2)
                cv2.drawFrameAxes(bgr, frame.K, None, cv2.Rodrigues(det['T'][:3, :3])[0], det['T'][:3, 3],
                                  args.marker_size)
            if time.time() < flash_until:      # no speaker needed: green frame + big banner for 2 s
                cv2.rectangle(bgr, (0, 0), (bgr.shape[1] - 1, bgr.shape[0] - 1), (0, 255, 0), 24)
                cv2.putText(bgr, f'CAPTURED #{len(samples) - 1}  -  move to next pose', (40, bgr.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4)
            cv2.putText(bgr, f'samples {len(samples)}/{args.target_samples}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(bgr, state, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            cv2.imshow('auto hand-eye capture (robot moved by human)', bgr)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), 27):
                break
            if key == ord('p'):
                paused = not paused
                log('paused' if paused else 'resumed')
            elif key == ord('u') and samples:
                samples.pop()
                save_json(samples_path, samples)
                log(f'removed last sample ({len(samples)} left)')
            elif key == ord('s'):
                solve_report(samples, out_path, args, log)
    finally:
        camera.close()
        cv2.destroyAllWindows()
        logf.close()


if __name__ == '__main__':
    main()
