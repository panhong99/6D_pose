"""Fixed-camera T_base_cam from a loose ArUco marker, using the WRIST camera as the measuring instrument.  Read-only.

Why: the gripper hides the flange, so no marker can sit there.  Instead a marker lies anywhere on the table where BOTH
cameras see it:   T_base_marker = T_base_flange @ T_flange_cam(wrist hand-eye) @ T_cam_marker(wrist)
                  T_base_cam   = T_base_marker @ inv(T_cam_marker(fixed))
One marker position gives one estimate; use >= 5 spread over the workspace and the script averages them and reports the
spread (residual per position).  The marker must NOT move between G and F of the same position.

    ROS2 terminal 1: bringup + `python3 calibration/ros2_flange_udp.py` (UDP 5006)       [Human starts, read-only]
    conda terminal 2: python calibration/fixed_cam_from_wrist.py [--marker_mm 100 --marker_scale 0.98]
Per position: jog the robot (Human, pendant) so the WRIST camera sees the marker obliquely (0.3-0.7 m), hold still, press G;
              then press F (fixed camera captures the same marker); press N to store the position.
Keys: G wrist | F fixed | N next position | U undo | S summary + save | Q quit.   Nothing is sent to the robot or ROS.
Output (never overwrites existing results): data/real_handeye/<time>_fromwrist/{T_base_cam.json, positions.json}.
--selftest checks the solver on synthetic data (no hardware).
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'real_time_project'))
from auto_handeye_capture import PoseStream, average_poses, capture  # noqa: E402
from calib_utils import detect_marker, inv_T, load_json, make_detector, rot_angle_deg, save_json  # noqa: E402
from wrist_handeye import WristCam  # noqa: E402

DATA = Path(__file__).resolve().parent / 'data'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--wrist_result', type=Path, default=None, help='default: newest data/wrist_handeye/*/result.json')
    p.add_argument('--wrist_serial', default='338122303684')
    p.add_argument('--fixed_serial', default='338122300585')
    p.add_argument('--marker_id', type=int, default=1)
    p.add_argument('--marker_mm', type=float, default=100.0, help='nominal printed side of the black square')
    p.add_argument('--marker_scale', type=float, default=0.98,
                   help='printed/nominal size (previous print of this marker measured ~98 mm, hand-eye fit 0.977)')
    p.add_argument('--dictionary', default='DICT_4X4_100')
    p.add_argument('--joint_port', type=int, default=5006)
    p.add_argument('--n_frames', type=int, default=20)
    p.add_argument('--max_jitter_mm', type=float, default=2.0)
    p.add_argument('--min_tilt_deg', type=float, default=10.0)
    p.add_argument('--still_mm', type=float, default=0.3)
    p.add_argument('--still_deg', type=float, default=0.1)
    p.add_argument('--pixels_per_cell', type=int, default=10, help='ArUco bit-sampling resolution (default 10; OpenCV default 4 missed this marker)')
    p.add_argument('--no_fixed_window', action='store_true', help='do not keep the fixed camera streaming (open it only on F)')
    p.add_argument('--resolve', type=Path, default=None, help='session dir with positions.json: recompute (no cameras, no robot)')
    p.add_argument('--selftest', action='store_true')
    return p.parse_args(argv)


def estimates(positions, scale):
    """Per-position T_base_cam for a given marker scale (scale multiplies both cameras' marker distances)."""
    out = []
    for q in positions:
        Tw = np.array(q['wrist_T_cam_marker'])
        Tw[:3, 3] *= scale
        Tf = np.array(q['fixed_T_cam_marker'])
        Tf[:3, 3] *= scale
        T_base_marker = np.array(q['T_base_flange']) @ np.array(q['T_flange_cam']) @ Tw
        out.append(T_base_marker @ inv_T(Tf))
    return out


def solve(positions, scale):
    est = estimates(positions, scale)
    T = average_poses(est)
    res = [(float(np.linalg.norm(e[:3, 3] - T[:3, 3]) * 1000), float(rot_angle_deg(T[:3, :3].T @ e[:3, :3]))) for e in est]
    return T, res


def report(positions, scale):
    lines = []
    T, res = solve(positions, scale)
    lines.append(f'{len(positions)} positions, marker {scale * 100:.1f} % of nominal')
    lines.append('T_base_cam position [m]: ' + str(np.round(T[:3, 3], 4).tolist()))
    for i, (d, a) in enumerate(res):
        lines.append(f'  position {i}: differs from the mean by {d:6.1f} mm / {a:4.2f} deg')
    lines.append(f'  mean spread {np.mean([r[0] for r in res]):.1f} mm (max {max(r[0] for r in res):.1f}), '
                 f'{np.mean([r[1] for r in res]):.2f} deg')
    if len(positions) >= 3:
        lines.append('marker-scale scan (spread of the per-position estimates; the true scale makes it smallest):')
        scan = [(s, np.mean([r[0] for r in solve(positions, s)[1]])) for s in np.arange(0.94, 1.0601, 0.005)]
        best = min(scan, key=lambda x: x[1])
        lines.append('  ' + '  '.join(f'{s:.3f}:{d:.1f}' for s, d in scan[::2]) + f'   best {best[0]:.3f} ({best[1]:.1f} mm)')
    return T, '\n'.join(lines)


def solve_pnp(positions, scale, K, idx=None):
    """T_base_cam by PnP on ALL marker corners: 3D corners in base from the wrist measurement, 2D pixels the fixed camera
    saw (re-projected from its stored marker pose).  Uses the corners' spread over the workspace, so one noisy marker
    orientation does not become a camera-position error (the plain pose average does: error ~ rotation noise x range)."""
    idx = list(range(len(positions))) if idx is None else idx
    size = positions[0].get('marker_size_m', 0.1)
    h, hn = size * scale / 2, size / 2
    corn = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    corn_nom = np.array([[-hn, hn, 0], [hn, hn, 0], [hn, -hn, 0], [-hn, -hn, 0]])
    obj, img = [], []
    for i in idx:
        q = positions[i]
        Tw = np.array(q['wrist_T_cam_marker'])
        Tw[:3, 3] *= scale
        Tbm = np.array(q['T_base_flange']) @ np.array(q['T_flange_cam']) @ Tw
        obj.append((Tbm[:3, :3] @ corn.T).T + Tbm[:3, 3])
        Tf = np.array(q['fixed_T_cam_marker'])
        cam = (Tf[:3, :3] @ corn_nom.T).T + Tf[:3, 3]
        px = (K @ cam.T).T
        img.append(px[:, :2] / px[:, 2:3])
    obj, img = np.vstack(obj), np.vstack(img)
    ok, rv, tv = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_EPNP)
    ok, rv, tv = cv2.solvePnP(obj, img, K, None, rv, tv, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    Tcb = np.eye(4)
    Tcb[:3, :3], Tcb[:3, 3] = cv2.Rodrigues(rv)[0], tv.ravel()
    proj, _ = cv2.projectPoints(obj, rv, tv, K, None)
    err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1).reshape(-1, 4).mean(1)
    return inv_T(Tcb), err


def report_pnp(positions, scale, K):
    T, err = solve_pnp(positions, scale, K)
    lines = [f'PnP on all marker corners: T_base_cam position {np.round(T[:3, 3], 4).tolist()} m, optical axis (base) '
             f'{np.round(T[:3, 2], 3).tolist()}',
             f'  reprojection per position [px]: {np.round(err, 2).tolist()}  (mean {err.mean():.2f})']
    if len(positions) >= 4:
        lines.append('  leave-one-position-out (camera position / rotation change):')
        for i in range(len(positions)):
            Ti, _ = solve_pnp(positions, scale, K, [j for j in range(len(positions)) if j != i])
            lines.append(f'    without {i}: {np.linalg.norm(Ti[:3, 3] - T[:3, 3]) * 1000:5.1f} mm / {rot_angle_deg(T[:3, :3].T @ Ti[:3, :3]):.2f} deg')
    return T, '\n'.join(lines)


def load_K():
    return np.array(load_json(DATA / 'real_d455_intrinsics.json')['K'])


def grab_fixed(serial, marker_id, marker_size, detector, n):
    """Open the fixed camera, average n detections of the marker, close it (kept off while the wrist preview runs)."""
    from real_time_utils import D455Source
    cam = D455Source(1280, 720, 30, serial)
    Ts, tilts = [], []
    try:
        for i in range(n + 25):
            f = cam.read()
            if i < 25:
                continue
            det = detect_marker(cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR), f.K, detector, marker_id, marker_size)
            if det is not None:
                Ts.append(det['T'])
                tilts.append(det['tilt_deg'])
    finally:
        cam.close()
    if len(Ts) < 0.6 * n:
        return None, f'marker seen in only {len(Ts)}/{n} frames'
    jitter = float(np.std([t[:3, 3] for t in Ts], axis=0).mean() * 1000)
    return dict(T=average_poses(Ts), tilt=float(np.mean(tilts)), jitter_mm=jitter, dist=float(np.linalg.norm(Ts[0][:3, 3]))), None


def grab_fixed_live(cam, marker_id, marker_size, detector, n):
    """Average n marker detections from the already-running fixed-camera stream (no re-opening)."""
    Ts, tilts = [], []
    deadline = time.time() + 8.0
    while len(Ts) < n and time.time() < deadline:
        f = cam.read()
        det = detect_marker(cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR), f.K, detector, marker_id, marker_size)
        if det is not None:
            Ts.append(det['T'])
            tilts.append(det['tilt_deg'])
    if len(Ts) < 0.6 * n:
        return None, f'marker seen in only {len(Ts)}/{n} frames'
    jitter = float(np.std([t[:3, 3] for t in Ts], axis=0).mean() * 1000)
    return dict(T=average_poses(Ts), tilt=float(np.mean(tilts)), jitter_mm=jitter, dist=float(np.linalg.norm(Ts[0][:3, 3]))), None


def selftest():
    rng = np.random.default_rng(1)
    from scipy.spatial.transform import Rotation as R
    T_base_cam = np.eye(4)
    T_base_cam[:3, :3] = R.from_euler('zyx', [150, -5, -125], degrees=True).as_matrix()
    T_base_cam[:3, 3] = [1.2, 0.3, 0.9]
    X = np.eye(4)
    X[:3, :3] = R.from_euler('xyz', [-3, 5, 90], degrees=True).as_matrix()
    X[:3, 3] = [0.03, -0.05, 0.08]
    true_scale, positions = 0.98, []
    for _ in range(8):
        Tm = np.eye(4)                                                    # marker lying on the table, random yaw
        Tm[:3, :3] = R.from_euler('z', rng.uniform(0, 360), degrees=True).as_matrix() @ R.from_euler('x', 180 + rng.uniform(-25, 25), degrees=True).as_matrix()
        Tm[:3, 3] = [rng.uniform(0.7, 1.1), rng.uniform(-0.3, 0.3), 0.05]
        Tf = inv_T(T_base_cam) @ Tm
        Tf_meas = Tf.copy()
        Tf_meas[:3, 3] = Tf[:3, 3] / true_scale + rng.normal(0, 0.001, 3) / true_scale     # printed marker smaller -> looks farther
        T_base_flange = np.eye(4)
        T_base_flange[:3, :3] = R.from_euler('zyx', [rng.uniform(-10, 10), rng.uniform(40, 80), rng.uniform(-10, 10)], degrees=True).as_matrix()
        T_base_flange[:3, 3] = [rng.uniform(0.4, 0.7), rng.uniform(-0.2, 0.2), rng.uniform(0.3, 0.5)]
        Tw = inv_T(T_base_flange @ X) @ Tm
        Tw_meas = Tw.copy()
        Tw_meas[:3, 3] = Tw[:3, 3] / true_scale + rng.normal(0, 0.001, 3) / true_scale
        positions.append(dict(T_base_flange=T_base_flange.tolist(), T_flange_cam=X.tolist(), wrist_T_cam_marker=Tw_meas.tolist(),
                              fixed_T_cam_marker=Tf_meas.tolist()))
    T, res = solve(positions, true_scale)
    err = np.linalg.norm(T[:3, 3] - T_base_cam[:3, 3]) * 1000
    rot = rot_angle_deg(T[:3, :3].T @ T_base_cam[:3, :3])
    _, text = report(positions, 1.0)
    print(text)
    print(f'selftest (true scale {true_scale}): position error {err:.2f} mm, rotation error {rot:.3f} deg  ->',
          'OK' if err < 3 and rot < 0.3 else 'FAIL')
    scan = {round(s, 3): np.mean([r[0] for r in solve(positions, s)[1]]) for s in np.arange(0.94, 1.0601, 0.005)}
    print('scale scan minimum at', min(scan, key=scan.get), '(expected near', true_scale, ')')


def main():
    a = parse_args()
    if a.selftest:
        return selftest()
    if a.resolve:
        d = load_json(a.resolve / 'positions.json')
        P, sc = d['positions'], d['marker_scale']
        for q in P:
            q['marker_size_m'] = d['marker_mm'] / 1000
        Ta, text = report(P, sc)
        print(text)
        Tp, textp = report_pnp(P, sc, load_K())
        print(textp)
        save_json(a.resolve / 'T_base_cam_pnp.json', dict(T_base_cam=Tp.tolist(), n_positions=len(P),
                                                         meta=dict(method='PnP on all marker corners (base from wrist, pixels from fixed cam)',
                                                                   marker_mm=d['marker_mm'], marker_scale=sc, created=time.strftime('%Y-%m-%d %H:%M:%S'),
                                                                   note='resolved offline from positions.json')))
        print('saved', a.resolve / 'T_base_cam_pnp.json')
        return
    wrist_path = a.wrist_result or sorted((DATA / 'wrist_handeye').glob('*/result.json'))[-1]
    X = np.array(load_json(wrist_path)['T_flange_cam'])
    out = DATA / 'real_handeye' / (time.strftime('%Y%m%d_%H%M%S') + '_fromwrist')
    out.mkdir(parents=True, exist_ok=True)
    size = a.marker_mm / 1000.0
    print(f'wrist hand-eye: {wrist_path}\nmarker {a.marker_mm:g} mm nominal x {a.marker_scale}, id {a.marker_id}, {a.dictionary}\nout {out}', flush=True)
    robot = PoseStream(a.joint_port)
    cam, detector = WristCam(a.wrist_serial, 1280, 720, 30), make_detector(a.dictionary, a.pixels_per_cell)
    fcam = None if a.no_fixed_window else WristCam(a.fixed_serial, 1280, 720, 30)      # colour-only stream, live window
    cap_args = SimpleNamespace(n_frames=a.n_frames, marker_id=a.marker_id, marker_size=size, still_mm=a.still_mm,
                               still_deg=a.still_deg, max_jitter_mm=a.max_jitter_mm, min_tilt_deg=a.min_tilt_deg)
    positions, wrist, fixed = [], None, None
    log = lambda m: print(m, flush=True)  # noqa: E731
    try:
        while True:
            frame = cam.read()
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            det = detect_marker(bgr, frame.K, detector, a.marker_id, size)
            if det is not None:
                cv2.polylines(bgr, [det['corners'].astype(np.int32)], True, (0, 255, 0), 2)
            text = (f'pos {len(positions)}  wrist {"OK" if wrist else "-"}  fixed {"OK" if fixed else "-"}  '
                    f'robot still {robot.still_for(a.still_mm, a.still_deg):.1f}s  '
                    + (f'tilt {det["tilt_deg"]:.0f} dist {det["T"][2, 3]:.2f}m' if det else 'marker not seen'))
            cv2.putText(bgr, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow('fixed_cam_from_wrist (wrist view)', bgr)
            if fcam is not None:
                ff = fcam.read()
                fbgr = cv2.cvtColor(ff.rgb, cv2.COLOR_RGB2BGR)
                fdet = detect_marker(fbgr, ff.K, detector, a.marker_id, size)
                if fdet is not None:
                    cv2.polylines(fbgr, [fdet['corners'].astype(np.int32)], True, (0, 255, 0), 2)
                cv2.putText(fbgr, 'FIXED camera  ' + (f'marker tilt {fdet["tilt_deg"]:.0f} dist {fdet["T"][2, 3]:.2f}m' if fdet else 'marker not seen'),
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow('fixed camera view', cv2.resize(fbgr, None, fx=0.75, fy=0.75))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('g'):
                s, err = capture(cam, detector, cap_args, robot, log)
                if s is None:
                    log(f'wrist capture failed: {err}')
                    continue
                wrist = dict(T_base_flange=s['T_base_flange'], posj=s['posj'], wrist_T_cam_marker=s['T_cam_marker'],
                             wrist_tilt=s['tilt_deg'], wrist_dist=s['dist_m'], wrist_jitter_mm=s['jitter_mm'])
                Tw = np.array(s['T_cam_marker'])
                Tw[:3, 3] *= a.marker_scale
                Tm = np.array(s['T_base_flange']) @ X @ Tw
                log(f'wrist: marker in base [mm] {np.round(Tm[:3, 3] * 1000, 1).tolist()}, normal z {Tm[2, 2]:+.2f} '
                    f'(-1 = lying face up), tilt {s["tilt_deg"]:.0f} deg, dist {s["dist_m"]:.2f} m, jitter {s["jitter_mm"]:.2f} mm')
            if key == ord('f'):
                f, err = (grab_fixed_live(fcam, a.marker_id, size, detector, a.n_frames) if fcam is not None
                          else grab_fixed(a.fixed_serial, a.marker_id, size, detector, a.n_frames))
                if f is None:
                    log(f'fixed capture failed: {err}')
                    continue
                fixed = dict(fixed_T_cam_marker=f['T'].tolist(), fixed_tilt=f['tilt'], fixed_dist=f['dist'], fixed_jitter_mm=f['jitter_mm'])
                log(f'fixed: marker at {f["dist"]:.3f} m, tilt {f["tilt"]:.0f} deg, jitter {f["jitter_mm"]:.2f} mm')
            if key == ord('n'):
                if not (wrist and fixed):
                    log('need both G and F for this position')
                    continue
                positions.append(dict(**wrist, **fixed, T_flange_cam=X.tolist()))
                wrist = fixed = None
                save_json(out / 'positions.json', dict(marker_mm=a.marker_mm, marker_scale=a.marker_scale, positions=positions))
                log(f'stored position {len(positions) - 1}; move the marker to a new place')
            if key == ord('u') and positions:
                positions.pop()
                log('removed last position')
            if key == ord('s') and len(positions) >= 1:
                T, text = report(positions, a.marker_scale)
                log(text)
                save_json(out / 'T_base_cam.json', dict(T_base_cam=T.tolist(), n_positions=len(positions),
                                                       meta=dict(method='wrist camera locates a loose marker, fixed camera observes it',
                                                                 marker_mm=a.marker_mm, marker_scale=a.marker_scale, marker_id=a.marker_id,
                                                                 dictionary=a.dictionary, wrist_result=str(wrist_path),
                                                                 wrist_serial=a.wrist_serial, fixed_serial=a.fixed_serial,
                                                                 created=time.strftime('%Y-%m-%d %H:%M:%S'))))
                log(f'saved {out / "T_base_cam.json"} (pose average)')
                Tp, textp = report_pnp(positions, a.marker_scale, load_K())
                log(textp)
                save_json(out / 'T_base_cam_pnp.json', dict(T_base_cam=Tp.tolist(), n_positions=len(positions),
                                                           meta=dict(method='PnP on all marker corners (base from wrist, pixels from fixed cam)',
                                                                     marker_mm=a.marker_mm, marker_scale=a.marker_scale, wrist_result=str(wrist_path),
                                                                     created=time.strftime('%Y-%m-%d %H:%M:%S'))))
                log(f'saved {out / "T_base_cam_pnp.json"}  <- prefer this one')
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
