"""Camera calibrations of the real cell (conda env `foundationpose`).  Read-only: the Human moves the robot with the
pendant (slowest speed); these tools only watch the cameras and the flange pose stream (`run.sh stream`, UDP 5006).

  python calibration/calibrate.py wrist                 wrist D455 eye-in-hand T_flange_cam (auto-capture, fixed marker)
  python calibration/calibrate.py wrist --solve <samples.json> [--out <result.json>]      re-solve, no camera
  python calibration/calibrate.py fixed                 fixed D455 T_base_cam from wrist-located loose markers (PnP)
  python calibration/calibrate.py fixed --resolve <session dir> [--out <json>]           recompute, no camera
  python calibration/calibrate.py gt                    FoundationPose x T_base_cam vs wrist-GT marker on the cube
  python calibration/calibrate.py selftest              synthetic solver checks (no hardware)

wrist : the marker stays FIXED; for every sample T_base_flange @ X @ T_cam_marker = T_base_marker (constant).  A sample is
        taken automatically when the robot is still, the view is new, oblique and steady.  Keys: S solve | U undo |
        P pause | Q quit.  Output data/wrist_handeye/<time>/{samples.json, result.json, capture.log}.
fixed : per marker position press G (wrist measures the marker in base), F (fixed camera sees it), N (store); the marker
        must not move between G and F.  S solves (pose average + PnP on all corners, PnP preferred).  Keys G F N U S Q.
        Output data/real_handeye/<time>_fromwrist/{positions.json, T_base_cam.json, T_base_cam_pnp.json}.
gt    : per cube position: G (wrist sees the cube-top marker), move the robot out of the fixed view, F (stable FP pose),
        N (store).  Output data/gt_compare/<time>/{positions.json, summary.txt}.  Needs real_time_project/main.py.
Numbers (serials, marker, capture thresholds): calibration/config.yaml.  Existing results are never overwritten.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from calib_core import (DATA, detect_marker, evaluate_eye_in_hand, fixed_cam_average, fixed_cam_pnp, latest_t_base_cam, leave_one_out,
                        load_config, load_fixed_K, load_json, make_detector, make_T, newest_wrist_result, posx_to_T,
                        rot_angle_deg, rotation_spread_deg, save_json, selftest, solve_eye_in_hand)


def refuse_overwrite(path, force):
    if Path(path).exists() and not force:
        raise SystemExit(f'{path} exists: pass --out <new file> (or --force to replace it)')


def window(name, bgr, text, det=None, scale=1.0):
    import cv2
    if det is not None:
        cv2.polylines(bgr, [det['corners'].astype(np.int32)], True, (0, 255, 0), 2)
    cv2.putText(bgr, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow(name, bgr if scale == 1.0 else cv2.resize(bgr, None, fx=scale, fy=scale))
    return cv2.waitKey(1) & 0xFF


class Logger:
    def __init__(self, path=None):
        self.f = open(path, 'a') if path else None

    def __call__(self, msg):
        print(msg, flush=True)
        if self.f:
            self.f.write(msg + '\n')
            self.f.flush()


# ================================================================ wrist: eye-in-hand T_flange_cam

def wrist_solve_report(samples, out, meta, scale, log):
    Tbf = [np.array(s['T_base_flange']) for s in samples]
    Tcm = [np.array(s['T_cam_marker']) for s in samples]
    for T in Tcm:
        T[:3, 3] *= scale
    if len(samples) < 6:
        log(f'need >= 6 samples to solve (have {len(samples)})')
        return
    try:
        X, Z = solve_eye_in_hand(Tbf, Tcm)
        loo_p, loo_r = leave_one_out(solve_eye_in_hand, evaluate_eye_in_hand, Tbf, Tcm)
    except ValueError as exc:
        log(f'solve failed: {exc}')
        return
    pos, rot = evaluate_eye_in_hand(Tbf, Tcm, X, Z)
    log(f'\n{len(samples)} samples, flange rotation spread {rotation_spread_deg(Tbf):.0f} deg, marker scale {scale}')
    log(' #   fit_mm  fit_deg   LOO_mm  LOO_deg  tilt  dist_m')
    for i, s in enumerate(samples):
        log(f'{i:2d} {pos[i]:8.2f} {rot[i]:8.2f} {loo_p[i]:8.2f} {loo_r[i]:8.2f}  {s["tilt_deg"]:4.0f}  {s["dist_m"]:.3f}')
    log(f'fit residual  : pos mean {pos.mean():.2f} mm (max {pos.max():.2f}), rot mean {rot.mean():.2f} deg (max {rot.max():.2f})')
    log(f'leave-one-out : pos mean {loo_p.mean():.2f} mm (median {np.median(loo_p):.2f}, max {loo_p.max():.2f}), '
        f'rot mean {loo_r.mean():.2f} deg (max {loo_r.max():.2f})')
    np.set_printoptions(precision=5, suppress=True)
    log(f'T_flange_cam (camera in the flange frame, m):\n{X}\nT_base_marker (m):\n{Z}')
    save_json(out, dict(T_flange_cam=X.tolist(), T_base_marker=Z.tolist(), n_samples=len(samples),
                        pos_err_mm=dict(mean=float(pos.mean()), max=float(pos.max())),
                        rot_err_deg=dict(mean=float(rot.mean()), max=float(rot.max())),
                        leave_one_out=dict(pos_mm=loo_p.tolist(), rot_deg=loo_r.tolist()),
                        created=time.strftime('%Y-%m-%d %H:%M:%S'), meta=dict(meta, marker_scale=scale)))
    log(f'saved: {out}  (check the residuals; compare with an independent measurement before relying on it)')


def run_wrist(a, cfg):
    mk, w, cap, cams = cfg['marker'], cfg['wrist'], cfg['capture'], cfg['cameras']
    scale = mk['scale'] if a.marker_scale is None else a.marker_scale
    meta = dict(marker_id=mk['id'], marker_size=mk['size_m'], dictionary=mk['dictionary'], serial=cams['wrist_serial'],
                resolution=[cams['width'], cams['height']], mode='eye-in-hand, fixed marker, human-moved robot, ROS2 read-only pose')
    in_range = lambda d: w['min_dist'] <= d <= (w['max_dist'] or 1e9)                     # noqa: E731
    if a.solve:
        samples = [s for s in load_json(a.solve)['samples'] if in_range(s['dist_m'])]
        out = a.out or a.solve.with_name('result.json')
        refuse_overwrite(out, a.force)
        print(f'{len(samples)} samples with {w["min_dist"]} <= dist <= {w["max_dist"]} m')
        wrist_solve_report(samples, out, meta, scale, print)
        return

    import cv2
    from sensors import ColorCam, PoseStream, capture, novelty
    session = a.session or DATA / 'wrist_handeye' / time.strftime('%Y%m%d_%H%M%S')
    session.mkdir(parents=True, exist_ok=True)
    log = Logger(session / 'capture.log')
    robot = PoseStream(cfg['udp']['flange'])
    cam = ColorCam(cams['wrist_serial'], cams['width'], cams['height'], cams['fps'])
    detector = make_detector(mk['dictionary'], mk['pixels_per_cell'])
    samples = load_json(session / 'samples.json')['samples'] if (session / 'samples.json').is_file() else []
    paused, cooldown, reported = False, 0.0, False
    log(f'session {session}  marker {mk["size_m"] * 1000:.1f} mm id {mk["id"]}  wrist {cams["wrist_serial"]}')
    log('Move the robot with the pendant (slowest). View the marker obliquely, vary the flange ROTATION a lot '
        '(tilt about all axes), hold still ~2 s per pose. Keep the marker fixed.')
    try:
        while True:
            frame = cam.read()
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            det = detect_marker(bgr, frame.K, detector, mk['id'], mk['size_m'])
            pose, still = robot.latest(), robot.still_for(cap['still_mm'], cap['still_deg'])
            text = (f'samples {len(samples)}/{w["target_samples"]}  {"PAUSED " if paused else ""}'
                    f'{"NO ROBOT POSE " if pose is None else ""}still {still:.1f}s  '
                    + (f'tilt {det["tilt_deg"]:.0f} dist {np.linalg.norm(det["T"][:3, 3]):.2f}m' if det else 'marker not seen'))
            if (not paused and det is not None and pose is not None and still >= cap['still_s'] and time.time() > cooldown
                    and det['tilt_deg'] >= w['min_tilt_deg'] and in_range(np.linalg.norm(det['T'][:3, 3]))):
                d_mm, d_deg = novelty(posx_to_T(pose[1]), samples)
                if d_mm >= w['min_new_mm'] or d_deg >= w['min_new_deg']:
                    sample, err = capture(cam, detector, robot, mk['id'], mk['size_m'], cap['n_frames'], cap['still_mm'],
                                          cap['still_deg'], w['max_jitter_mm'], w['min_tilt_deg'])
                    if sample is None:
                        log(f'  skipped: {err}')
                        cooldown = time.time() + 2.0
                    else:
                        samples.append(sample)
                        save_json(session / 'samples.json', dict(samples=samples))
                        log(f'  sample {len(samples) - 1}: tilt {sample["tilt_deg"]:.0f} dist {sample["dist_m"]:.3f} m '
                            f'jitter {sample["jitter_mm"]:.2f} mm  posx {np.round(sample["posx"], 1).tolist()}')
            key = window('calibrate wrist', bgr, text, det)
            if key in (ord('q'), 27):
                break
            if key == ord('p'):
                paused = not paused
            if key == ord('u') and samples:
                samples.pop()
                save_json(session / 'samples.json', dict(samples=samples))
                log(f'undo -> {len(samples)} samples')
            if key == ord('s') or (len(samples) >= w['target_samples'] and not reported):
                reported = True
                wrist_solve_report(samples, session / 'result.json', meta, scale, log)
    finally:
        cam.close()
        cv2.destroyAllWindows()
        if len(samples) >= 6:
            wrist_solve_report(samples, session / 'result.json', meta, scale, log)
        log(f'session dir: {session}')


# ================================================================ fixed: T_base_cam from wrist-located markers

def fixed_report(positions, scale, K):
    """Pose-average and PnP solutions with their spread / leave-one-position-out tables."""
    Ta, res = fixed_cam_average(positions, scale)
    lines = [f'{len(positions)} positions, marker {scale * 100:.1f} % of nominal',
             f'pose average: T_base_cam position {np.round(Ta[:3, 3], 4).tolist()} m']
    lines += [f'  position {i}: differs from the mean by {d:6.1f} mm / {r:4.2f} deg' for i, (d, r) in enumerate(res)]
    if len(positions) >= 3:
        scan = [(s, np.mean([r[0] for r in fixed_cam_average(positions, s)[1]])) for s in np.arange(0.94, 1.0601, 0.005)]
        best = min(scan, key=lambda x: x[1])
        lines.append('  marker-scale scan (smallest spread = true scale): '
                     + '  '.join(f'{s:.3f}:{d:.1f}' for s, d in scan[::2]) + f'   best {best[0]:.3f}')
    Tp, err = fixed_cam_pnp(positions, scale, K)
    lines += [f'PnP on all corners: T_base_cam position {np.round(Tp[:3, 3], 4).tolist()} m, optical axis '
              f'{np.round(Tp[:3, 2], 3).tolist()}', f'  reprojection per position [px]: {np.round(err, 2).tolist()} (mean {err.mean():.2f})']
    if len(positions) >= 4:
        lines.append('  leave-one-position-out (camera position / rotation change):')
        for i in range(len(positions)):
            Ti, _ = fixed_cam_pnp(positions, scale, K, [j for j in range(len(positions)) if j != i])
            lines.append(f'    without {i}: {np.linalg.norm(Ti[:3, 3] - Tp[:3, 3]) * 1000:5.1f} mm / '
                         f'{rot_angle_deg(Tp[:3, :3].T @ Ti[:3, :3]):.2f} deg')
    return Ta, Tp, '\n'.join(lines)


def run_fixed(a, cfg):
    mk, f, cap, cams = cfg['marker'], cfg['fixed'], cfg['capture'], cfg['cameras']
    K = load_fixed_K(cfg)
    if a.resolve:
        d = load_json(a.resolve / 'positions.json')
        for q in d['positions']:
            q['marker_size_m'] = d['marker_mm'] / 1000
        out = a.out or a.resolve / 'T_base_cam_pnp.json'
        refuse_overwrite(out, a.force)
        _, Tp, text = fixed_report(d['positions'], d['marker_scale'], K)
        print(text)
        save_json(out, dict(T_base_cam=Tp.tolist(), n_positions=len(d['positions']), meta=dict(
            method='PnP on all marker corners (base from wrist, pixels from fixed cam)', marker_mm=d['marker_mm'],
            marker_scale=d['marker_scale'], created=time.strftime('%Y-%m-%d %H:%M:%S'), note='resolved offline from positions.json')))
        print('saved', out)
        return

    import cv2
    from sensors import ColorCam, PoseStream, average_marker, capture
    scale = mk['scale'] if a.marker_scale is None else a.marker_scale
    wrist_path = a.wrist_result or newest_wrist_result()
    X = np.array(load_json(wrist_path)['T_flange_cam'])
    out = DATA / 'real_handeye' / (time.strftime('%Y%m%d_%H%M%S') + '_fromwrist')
    out.mkdir(parents=True, exist_ok=True)
    size = mk['size_m']
    log = Logger()
    log(f'wrist hand-eye: {wrist_path}\nmarker {size * 1000:g} mm nominal x {scale}, id {mk["id"]}, {mk["dictionary"]}\nout {out}')
    robot = PoseStream(cfg['udp']['flange'])
    cam = ColorCam(cams['wrist_serial'], cams['width'], cams['height'], cams['fps'])
    fcam = None if a.no_fixed_window else ColorCam(cams['fixed_serial'], cams['width'], cams['height'], cams['fps'])
    detector = make_detector(mk['dictionary'], mk['pixels_per_cell'])
    positions, wrist, fixed = [], None, None
    try:
        while True:
            frame = cam.read()
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            det = detect_marker(bgr, frame.K, detector, mk['id'], size)
            key = window('calibrate fixed (wrist view)', bgr,
                         f'pos {len(positions)}  wrist {"OK" if wrist else "-"}  fixed {"OK" if fixed else "-"}  robot still '
                         f'{robot.still_for(cap["still_mm"], cap["still_deg"]):.1f}s  '
                         + (f'tilt {det["tilt_deg"]:.0f} dist {det["T"][2, 3]:.2f}m' if det else 'marker not seen'), det)
            if fcam is not None:
                ff = fcam.read()
                fbgr = cv2.cvtColor(ff.rgb, cv2.COLOR_RGB2BGR)
                fdet = detect_marker(fbgr, ff.K, detector, mk['id'], size)
                window('fixed camera view', fbgr, 'FIXED camera  ' + (f'marker tilt {fdet["tilt_deg"]:.0f} dist {fdet["T"][2, 3]:.2f}m'
                                                                     if fdet else 'marker not seen'), fdet, scale=0.75)
            if key in (ord('q'), 27):
                break
            if key == ord('g'):
                s, err = capture(cam, detector, robot, mk['id'], size, cap['n_frames'], cap['still_mm'], cap['still_deg'],
                                 f['max_jitter_mm'], f['min_tilt_deg'])
                if s is None:
                    log(f'wrist capture failed: {err}')
                    continue
                wrist = dict(T_base_flange=s['T_base_flange'], posj=s['posj'], wrist_T_cam_marker=s['T_cam_marker'],
                             wrist_tilt=s['tilt_deg'], wrist_dist=s['dist_m'], wrist_jitter_mm=s['jitter_mm'])
                Tw = np.array(s['T_cam_marker'])
                Tw[:3, 3] *= scale
                Tm = np.array(s['T_base_flange']) @ X @ Tw
                log(f'wrist: marker in base [mm] {np.round(Tm[:3, 3] * 1000, 1).tolist()}, normal z {Tm[2, 2]:+.2f} '
                    f'(-1 = face up), tilt {s["tilt_deg"]:.0f} deg, dist {s["dist_m"]:.2f} m, jitter {s["jitter_mm"]:.2f} mm')
            if key == ord('f'):
                if fcam is not None:
                    fx, err = average_marker(fcam, detector, mk['id'], size, cap['n_frames'])
                else:                                               # open the fixed camera only for this capture
                    tmp = ColorCam(cams['fixed_serial'], cams['width'], cams['height'], cams['fps'])
                    try:
                        for _ in range(25):                         # auto-exposure settle
                            tmp.read()
                        fx, err = average_marker(tmp, detector, mk['id'], size, cap['n_frames'])
                    finally:
                        tmp.close()
                if fx is None:
                    log(f'fixed capture failed: {err}')
                    continue
                fixed = dict(fixed_T_cam_marker=fx['T'].tolist(), fixed_tilt=fx['tilt'], fixed_dist=fx['dist'], fixed_jitter_mm=fx['jitter_mm'])
                log(f'fixed: marker at {fx["dist"]:.3f} m, tilt {fx["tilt"]:.0f} deg, jitter {fx["jitter_mm"]:.2f} mm')
            if key == ord('n'):
                if not (wrist and fixed):
                    log('need both G and F for this position')
                    continue
                positions.append(dict(**wrist, **fixed, T_flange_cam=X.tolist()))
                wrist = fixed = None
                save_json(out / 'positions.json', dict(marker_mm=size * 1000, marker_scale=scale, positions=positions))
                log(f'stored position {len(positions) - 1}; move the marker to a new place')
            if key == ord('u') and positions:
                positions.pop()
                log('removed last position')
            if key == ord('s') and positions:
                for q in positions:
                    q['marker_size_m'] = size
                Ta, Tp, text = fixed_report(positions, scale, load_fixed_K(cfg))
                log(text)
                meta = dict(marker_mm=size * 1000, marker_scale=scale, marker_id=mk['id'], dictionary=mk['dictionary'],
                            wrist_result=str(wrist_path), wrist_serial=cams['wrist_serial'], fixed_serial=cams['fixed_serial'],
                            created=time.strftime('%Y-%m-%d %H:%M:%S'))
                save_json(out / 'T_base_cam.json', dict(T_base_cam=Ta.tolist(), n_positions=len(positions),
                                                       meta=dict(meta, method='per-position pose average')))
                save_json(out / 'T_base_cam_pnp.json', dict(T_base_cam=Tp.tolist(), n_positions=len(positions),
                                                           meta=dict(meta, method='PnP on all marker corners (base from wrist, pixels from fixed cam)')))
                log(f'saved {out}/T_base_cam.json (average) and T_base_cam_pnp.json  <- prefer PnP')
    finally:
        cam.close()
        if fcam is not None:
            fcam.close()
        cv2.destroyAllWindows()


# ================================================================ gt: FoundationPose x T_base_cam vs wrist GT

def gt_compare(T_fp, T_gt):
    """Position error vector (mm, base frame), top-face normal angle (deg), yaw about the normal mod 90 (deg)."""
    d = (T_fp[:3, 3] - T_gt[:3, 3]) * 1000
    n = T_gt[:3, 2]
    k = int(np.argmax(np.abs(T_fp[:3, :3].T @ n)))                 # FP cube axis that is the face normal
    tilt = float(np.degrees(np.arccos(np.clip((T_fp[:3, k] * np.sign(T_fp[:3, k] @ n)) @ n, -1, 1))))
    v = T_fp[:3, [i for i in range(3) if i != k][0]]
    v = v - (v @ n) * n
    yaw = float(np.degrees(np.arctan2(v @ np.cross(n, T_gt[:3, 0]), v @ T_gt[:3, 0])))
    return d, tilt, (yaw + 45) % 90 - 45


def gt_summary(positions, variants):
    lines = [f'{len(positions)} cube positions (error = FoundationPose - GT, base frame, mm)']
    for name in variants:
        lines += [f'\n== T_base_cam: {name}', ' #   GT x/y/z (mm)           dx      dy      dz   horiz   total  normal_deg  yaw90_deg']
        rows = []
        for i, pos in enumerate(positions):
            gt = np.array(pos['T_base_cube_gt'])
            d, tilt, yaw = gt_compare(np.array(pos['fp'][name]['T_base_cube']), gt)
            rows.append(d)
            lines.append(f'{i:2d}  {gt[0, 3] * 1000:7.1f} {gt[1, 3] * 1000:6.1f} {gt[2, 3] * 1000:5.1f}  {d[0]:7.1f} {d[1]:7.1f} '
                         f'{d[2]:7.1f} {np.hypot(d[0], d[1]):7.1f} {np.linalg.norm(d):7.1f}  {tilt:8.1f} {yaw:10.1f}')
        D = np.array(rows)
        lines.append(f'mean error  : dx {D[:, 0].mean():+.1f} dy {D[:, 1].mean():+.1f} dz {D[:, 2].mean():+.1f} mm (bias if large and the scatter is small)')
        lines.append(f'scatter (std): dx {D[:, 0].std():.1f} dy {D[:, 1].std():.1f} dz {D[:, 2].std():.1f} mm   mean horiz '
                     f'{np.mean(np.hypot(D[:, 0], D[:, 1])):.1f}  mean total {np.mean(np.linalg.norm(D, axis=1)):.1f}  max total {np.max(np.linalg.norm(D, axis=1)):.1f} mm')
    return '\n'.join(lines)


def run_gt(a, cfg):
    import cv2
    from sensors import ColorCam, FpStream, PoseStream, capture
    g, cap, cams, st = cfg['gt_compare'], cfg['capture'], cfg['cameras'], cfg['stability']
    wrist_path = a.wrist_result or newest_wrist_result()
    wr = load_json(wrist_path)
    X, scale = np.array(wr['T_flange_cam']), wr['meta'].get('marker_scale', 1.0)
    paths = a.t_base_cam or sorted(latest_t_base_cam().parent.glob('T_base_cam*.json'))
    variants = {p.stem: np.array(load_json(p)['T_base_cam']) for p in paths}
    out = a.out or DATA / 'gt_compare' / time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    log = Logger()
    log(f'wrist result {wrist_path} (fit {wr["pos_err_mm"]["mean"]:.2f} mm, marker_scale {scale}); cube marker '
        f'{g["marker_size_m"] * 1000:.1f} mm, cube {g["cube_edge_m"] * 1000:.0f} mm\nT_base_cam variants: {list(variants)}\nout {out}')
    robot, fp = PoseStream(cfg['udp']['flange']), FpStream(cfg['udp']['fp_pose'], st['n'], st['mm'], st['deg'])
    cam = ColorCam(cams['wrist_serial'], cams['width'], cams['height'], cams['fps'])
    detector = make_detector(g['dictionary'])
    positions, gt, fpv = [], None, None

    def store():
        save_json(out / 'positions.json', dict(positions=positions, wrist_result=str(wrist_path), marker_scale=scale,
                                               cube_edge=g['cube_edge_m']))
    try:
        while True:
            fp.poll()
            frame = cam.read()
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            det = detect_marker(bgr, frame.K, detector, g['marker_id'], g['marker_size_m'])
            stab = fp.stable()
            key = window('calibrate gt', bgr, f'pos {len(positions)}  GT {"OK" if gt else "-"}  FP {"OK" if fpv else "-"}  '
                         f'robot still {robot.still_for(cap["still_mm"], cap["still_deg"]):.1f}s  FP '
                         f'{"STABLE" if stab is not None else "not stable/none"}  '
                         + (f'marker tilt {det["tilt_deg"]:.0f}' if det else 'marker not seen'), det)
            if key in (ord('q'), 27):
                break
            if key == ord('g'):
                s, err = capture(cam, detector, robot, g['marker_id'], g['marker_size_m'], cap['n_frames'], cap['still_mm'],
                                 cap['still_deg'], g['max_jitter_mm'], g['min_tilt_deg'])
                if s is None:
                    log(f'GT failed: {err}')
                    continue
                Tcm = np.array(s['T_cam_marker'])
                Tcm[:3, 3] *= scale
                T_marker = np.array(s['T_base_flange']) @ X @ Tcm
                T_cube = T_marker @ make_T(np.eye(3), [0, 0, -g['cube_edge_m'] / 2])
                gt = dict(T_base_marker=T_marker.tolist(), T_base_cube_gt=T_cube.tolist(), posx=s['posx'],
                          dist_m=s['dist_m'], tilt_deg=s['tilt_deg'], jitter_mm=s['jitter_mm'])
                log(f'GT cube centre (base, mm): {np.round(T_cube[:3, 3] * 1000, 1).tolist()}  (marker normal tilt '
                    f'{np.degrees(np.arccos(np.clip(T_marker[2, 2], -1, 1))):.1f} deg, dist {s["dist_m"]:.2f} m)')
            if key == ord('f'):
                if stab is None:
                    log('FP not stable yet (main.py running? cube visible? robot out of the way?)')
                    continue
                fpv = dict(T_cam_cube=stab.tolist(), **{n: dict(T_base_cube=(T @ stab).tolist()) for n, T in variants.items()})
                log('FP cube centre (base, mm): ' + '; '.join(f'{n}: {np.round((T @ stab)[:3, 3] * 1000, 1).tolist()}'
                                                              for n, T in variants.items()))
            if key == ord('n'):
                if gt is None or fpv is None:
                    log('need both G and F for this position first')
                    continue
                positions.append(dict(T_base_cube_gt=gt['T_base_cube_gt'], gt=gt, T_cam_cube=fpv['T_cam_cube'],
                                      fp={n: fpv[n] for n in variants}, time=time.strftime('%H:%M:%S')))
                gt = fpv = None
                store()
                log(f'stored position {len(positions) - 1}. Move the cube, then G / F again.\n' + gt_summary(positions, variants))
            if key == ord('u') and positions:
                positions.pop()
                store()
                log(f'undo -> {len(positions)} positions')
            if key == ord('s') and positions:
                log(gt_summary(positions, variants))
    finally:
        cam.close()
        cv2.destroyAllWindows()
        if positions:
            (out / 'summary.txt').write_text(gt_summary(positions, variants))
            log(gt_summary(positions, variants) + f'\nsaved {out}')


# ================================================================ main

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=('wrist', 'fixed', 'gt', 'selftest'))
    p.add_argument('--marker_scale', type=float, default=None, help='override config marker.scale')
    p.add_argument('--solve', type=Path, default=None, help='wrist: re-solve this samples.json (no camera)')
    p.add_argument('--session', type=Path, default=None, help='wrist: resume this session dir')
    p.add_argument('--resolve', type=Path, default=None, help='fixed: recompute from <session>/positions.json (no camera)')
    p.add_argument('--no_fixed_window', action='store_true', help='fixed: open the fixed camera only at F')
    p.add_argument('--wrist_result', type=Path, default=None, help='fixed/gt: default newest data/wrist_handeye/*/result.json')
    p.add_argument('--t_base_cam', type=Path, nargs='*', default=None, help='gt: variants (default all of the newest session)')
    p.add_argument('--out', type=Path, default=None, help='output file/dir (re-solves: required unless --force)')
    p.add_argument('--force', action='store_true', help='allow replacing an existing result')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = load_config()
    {'wrist': run_wrist, 'fixed': run_fixed, 'gt': run_gt, 'selftest': lambda a, c: selftest()}[args.mode](args, config)
