"""D455 camera tools (conda env `foundationpose`, camera only, no robot).

  python calibration/camera_tools.py preview [--all | --serial S [--serial S2]]   live view; S saves colour+depth+K (one --serial: data/tray_capture/, full 1280x720)
  python calibration/camera_tools.py marker [--serial S] [--marker_mm 100]        live ArUco check + averaged report
  python calibration/camera_tools.py probe [--serial S | --image F] [--marker_mm M]   identify an unknown marker board
  python calibration/camera_tools.py locate                                       plate position vs the RL sim spot

preview : one camera (default the first found) with depth colour map and centre depth; S saves to data/real_env/.
          --all / --serial: several cameras side by side (no saving).
marker  : axes, distance, tilt (marker normal vs viewing ray), reprojection and position jitter once per second; on Q
          prints the pose averaged over the last 30 detections (camera frame) and saves an annotated image to
          data/board_probe/.  A head-on marker (tilt < ~15 deg) has an ambiguous rotation.
probe   : tries every predefined ArUco/AprilTag dictionary, prints IDs, marker size in pixels and the layout in marker
          units (pass --marker_mm for millimetres).  Saves raw + annotated images to data/board_probe/.
locate  : needs real_time_project/main.py running on the fixed camera with the 029 plate mesh (UDP 5005).  Prints the
          plate bottom-centre in the robot base (FoundationPose x T_base_cam) and how far it is from the spot the RL
          policy was trained for (rl/config.yaml geometry.plate_xy, box_h).  Use it to place the real plate + box.
Defaults (serials, marker id / dictionary / size): calibration/config.yaml.
"""
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from calib_core import DATA, detect_marker, load_config, make_detector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'real_time_project'))


# ---------------------------------------------------------------- preview

def preview(a):
    import pyrealsense2 as rs
    available = [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()
                 if 'D455' in d.get_info(rs.camera_info.name)]
    if a.all or a.serial:
        serials = available if a.all else a.serial
        missing = [s for s in serials if s not in available]
        if missing or not serials:
            raise SystemExit(f'D455 not connected: {missing}; found {available or "none"}')
        return preview_multiple(serials)
    out = DATA / 'real_env'
    out.mkdir(parents=True, exist_ok=True)
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    serial = profile.get_device().get_info(rs.camera_info.serial_number)
    align = rs.align(rs.stream.color)
    win = f'D455 {serial}  [s] save  [q] quit'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1600, 450)
    saved, flash_until = 0, 0.0
    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
            vis = cv2.applyColorMap(cv2.convertScaleAbs(np.clip(depth_m, 0, 3.0), alpha=255 / 3.0), cv2.COLORMAP_JET)
            vis[depth_m == 0] = 0
            view = color.copy()
            h, w = view.shape[:2]
            cx, cy = w // 2, h // 2
            cv2.drawMarker(view, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 50, 1)
            patch = depth_m[cy - 5:cy + 6, cx - 5:cx + 6]
            patch = patch[patch > 0]
            cv2.putText(view, f'centre depth {f"{np.median(patch):.3f} m" if patch.size else "none"} | valid '
                              f'{100 * float((depth_m > 0).mean()):.0f}% | saved {saved}', (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if time.time() < flash_until:
                cv2.rectangle(view, (0, 0), (w - 1, h - 1), (0, 255, 0), 12)
            cv2.imshow(win, np.hstack([view, vis]))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                stamp = time.strftime('%Y%m%d_%H%M%S')
                intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
                cv2.imwrite(str(out / f'real_color_{stamp}.png'), color)
                np.save(out / f'real_depth_{stamp}.npy', depth_m)
                (out / f'real_intrinsics_{stamp}.json').write_text(json.dumps(dict(
                    K=[[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dist=list(intr.coeffs),
                    width=intr.width, height=intr.height, depth_scale=depth_scale, serial=serial), indent=1))
                saved, flash_until = saved + 1, time.time() + 0.4
                print(f'saved {stamp}', flush=True)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def preview_multiple(serials):
    import pyrealsense2 as rs
    pipelines = []
    try:
        align = rs.align(rs.stream.color)
        for serial in serials:
            pipeline, config = rs.pipeline(), rs.config()
            config.enable_device(serial)
            w, h = (1280, 720) if len(serials) == 1 else (640, 480)          # a single camera streams full resolution
            config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 15)
            config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 15)
            profile = pipeline.start(config)
            pipelines.append((pipeline, serial, profile.get_device().first_depth_sensor().get_depth_scale()))
            print(f'D455 {serial} preview started ({w}x{h}, 15 FPS)  [s] saves colour + depth + K to data/tray_capture/', flush=True)
        win = 'D455 preview  [s] save  [q / Esc] close'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1600, 500)
        out = DATA / 'tray_capture'
        out.mkdir(parents=True, exist_ok=True)
        save_now, flash_until = False, 0.0
        while True:
            panels = []
            for pipeline, serial, scale in pipelines:
                frames = align.process(pipeline.wait_for_frames())
                color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
                panel = np.zeros((480, 640, 3), dtype=np.uint8)
                if color_frame and depth_frame:
                    panel = np.asanyarray(color_frame.get_data()).copy()
                    if save_now:
                        stamp = time.strftime('%Y%m%d_%H%M%S')
                        intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
                        cv2.imwrite(str(out / f'{stamp}_{serial}_color.png'), panel)
                        np.save(out / f'{stamp}_{serial}_depth.npy',
                                np.asanyarray(depth_frame.get_data()).astype(np.float32) * scale)
                        (out / f'{stamp}_{serial}.json').write_text(json.dumps(dict(
                            serial=serial, K=[[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
                            dist=list(intr.coeffs), width=intr.width, height=intr.height, depth_scale=scale), indent=1))
                        print(f'saved {out / (stamp + "_" + serial + "_color.png")}', flush=True)
                        flash_until = time.time() + 0.4
                    valid = 100 * float((np.asanyarray(depth_frame.get_data()) > 0).mean())
                    cv2.drawMarker(panel, (320, 240), (0, 255, 0), cv2.MARKER_CROSS, 30, 1)
                    cv2.putText(panel, f'D455 {serial} | depth valid {valid:.0f}%', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                panels.append(panel)
            save_now = False
            view = np.hstack(panels)
            if time.time() < flash_until:
                cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), (0, 255, 0), 12)
            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            save_now = key == ord('s')
    finally:
        for pipeline, _, _ in reversed(pipelines):
            pipeline.stop()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------- marker

def marker(a, cfg):
    from real_time_utils import D455Source
    mk = cfg['marker']
    size = (a.marker_mm or mk['size_m'] * 1000) / 1000.0
    detector = make_detector(a.dictionary or mk['dictionary'], mk['pixels_per_cell'])
    print('Opening D455 (rectification maps take a few seconds) ...', flush=True)
    cam = D455Source(1280, 720, 30, a.serial)
    print(f'K=\n{cam.K}\nQ/ESC: quit', flush=True)
    history, recent, seen, total, t_report, last = deque(maxlen=30), deque(maxlen=30), 0, 0, time.time(), None
    try:
        while True:
            frame = cam.read()
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            det = detect_marker(bgr, frame.K, detector, mk['id'] if a.marker_id is None else a.marker_id, size)
            total += 1
            text = 'marker not found'
            if det is not None:
                seen += 1
                T = det['T']
                history.append(T[:3, 3] * 1000)
                recent.append(det)
                cv2.drawFrameAxes(bgr, frame.K, None, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], size)
                cv2.polylines(bgr, [det['corners'].astype(np.int32)], True, (0, 255, 0), 2)
                jitter = np.std(history, axis=0).mean() if len(history) >= 5 else float('nan')
                text = (f'dist {np.linalg.norm(T[:3, 3]) * 100:.1f}cm tilt {det["tilt_deg"]:.0f}deg '
                        f'reproj {det["reproj_px"]:.2f}px jitter {jitter:.2f}mm')
                last = bgr.copy()
            cv2.putText(bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('marker check', bgr)
            if time.time() - t_report > 1.0:
                print(f'[{seen}/{total} frames detected] {text}', flush=True)
                seen, total, t_report = 0, 0, time.time()
            if cv2.waitKey(1) & 0xff in (ord('q'), 27):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()
    if recent:
        t = np.array([d['T'][:3, 3] for d in recent])
        n = np.array([d['T'][:3, 2] for d in recent])
        print(f'last {len(recent)} detections: centre (camera frame) {t.mean(0).round(4).tolist()} m, distance '
              f'{np.linalg.norm(t.mean(0)):.3f} m, jitter std {(t.std(0) * 1000).round(2).tolist()} mm, tilt '
              f'{np.mean([d["tilt_deg"] for d in recent]):.1f} deg, reprojection {np.mean([d["reproj_px"] for d in recent]):.2f} px, '
              f'normal {n.mean(0).round(3).tolist()}')
        out = DATA / 'board_probe'
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / f'{time.strftime("%Y%m%d_%H%M%S")}_marker_{a.serial}.png'), last)


# ---------------------------------------------------------------- probe

def probe(a):
    if a.image:
        img = cv2.imread(str(a.image))
    else:
        from real_time_utils import D455Source
        cam = D455Source(1280, 720, 30, a.serial)
        for _ in range(30):                                     # let auto-exposure settle
            frame = cam.read()
        cam.close()
        img = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    best = None
    for name in [n for n in dir(cv2.aruco) if n.startswith('DICT_')]:
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        corners, ids, _ = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)), params).detectMarkers(gray)
        if ids is not None and (best is None or len(ids) > best[1]):
            best = (name, len(ids), corners, ids.flatten())
    out = DATA / 'board_probe'
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    cv2.imwrite(str(out / f'{stamp}_raw.png'), img)
    if best is None:
        print('NO markers found in any dictionary. Check: in view, sharp, no glare, > 25 px per marker.')
        return
    name, n, corners, ids = best
    cv2.aruco.drawDetectedMarkers(img, corners, ids.reshape(-1, 1))
    cv2.imwrite(str(out / f'{stamp}_annotated.png'), img)
    c = np.array([c.reshape(4, 2) for c in corners])
    sides = np.linalg.norm(c - np.roll(c, 1, axis=1), axis=2).mean(1)
    print(f'dictionary {name}: {n} markers, ids {sorted(ids.tolist())}\nmarker side in pixels: min {sides.min():.0f} '
          f'median {np.median(sides):.0f} max {sides.max():.0f}')
    if n >= 2:                                                  # layout in units of the largest marker's side
        ref = c[np.argmax(sides)]
        H, _ = cv2.findHomography(ref, np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float))
        plane = cv2.perspectiveTransform(c.mean(1).reshape(-1, 1, 2), H).reshape(-1, 2)
        unit = a.marker_mm or 1.0
        print(f'layout relative to marker id {ids[np.argmax(sides)]} ({"mm" if a.marker_mm else "marker-side units"}):')
        for i in np.lexsort((plane[:, 0].round(1), plane[:, 1].round(1))):
            print(f'  id {ids[i]:3d}: x {plane[i, 0] * unit:8.2f}  y {plane[i, 1] * unit:8.2f}')
    print(f'images: {out}/{stamp}_raw.png, {stamp}_annotated.png')


# ---------------------------------------------------------------- locate

def locate(a, cfg):
    """Live plate bottom-centre in base vs the RL training spot (camera + FoundationPose only, no robot)."""
    import trimesh
    import yaml
    from calib_core import cpath, latest_t_base_cam, load_json
    from sensors import FpStream
    rl = yaml.safe_load((Path(__file__).resolve().parent.parent / 'rl' / 'config.yaml').read_text())['geometry']
    target = np.array([*rl['plate_xy'], rl['table_top_z'] + rl['box_h']])
    lo, hi = trimesh.load(cpath(cfg['paths']['plate_obj']), force='mesh').bounds
    bottom_centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2], 1.0])     # sim plate root, mesh frame
    t_file = latest_t_base_cam()
    T_base_cam = np.array(load_json(t_file)['T_base_cam'])
    st = cfg['stability']
    fp = FpStream(cfg['udp']['fp_pose'], st['n'], st['mm'], st['deg'])
    print(f'T_base_cam: {t_file}\ntarget (RL plate root) [m]: {np.round(target, 3).tolist()}  (+-10 mm trained)\n'
          'waiting for FoundationPose on UDP %d ... Ctrl+C to stop' % cfg['udp']['fp_pose'], flush=True)
    last = 0.0
    try:
        while True:
            fp.poll()
            T = fp.stable()
            if T is not None and time.time() - last > 0.5:
                last = time.time()
                Tb = T_base_cam @ T
                p = (Tb @ bottom_centre)[:3]
                d = (p - target) * 1000
                tilt = np.degrees(np.arccos(np.clip(Tb[2, 2], -1, 1)))
                yaw = np.degrees(np.arctan2(Tb[1, 0], Tb[0, 0]))
                ok = abs(d[0]) <= 10 and abs(d[1]) <= 10 and abs(d[2]) <= 15 and tilt < 5
                print(f'plate root [m] {np.round(p, 4).tolist()}  move by x {-d[0]:+6.1f} mm  y {-d[1]:+6.1f} mm  '
                      f'(z {d[2]:+5.1f} mm)  yaw {yaw:+6.1f} deg  tilt {tilt:4.1f} deg  {"OK" if ok else ""}', flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass


def basket_measure(cfg, n_frames=None):
    """Mean base-frame positions of the basket rim markers over n_frames (fixed D455).  Positions only: a single
    small oblique marker can flip its orientation (IPPE), positions stay stable.  Returns {id: xyz}."""
    import cv2
    from calib_core import latest_t_base_cam, load_json
    from sensors import ColorCam
    b = cfg['basket']
    ids_wanted = set(b['marker_ids'].values())
    Tbc = np.array(load_json(latest_t_base_cam())['T_base_cam'])
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg['marker']['dictionary']))
    prm = cv2.aruco.DetectorParameters()
    prm.perspectiveRemovePixelPerCell = cfg['marker']['pixels_per_cell']
    det = cv2.aruco.ArucoDetector(d, prm)
    s = b['marker_size_m'] / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
    cam = ColorCam(cfg['cameras']['fixed_serial'])
    seen = {}
    try:
        for _ in range(30):
            cam.read()
        for _ in range(n_frames or b['n_frames']):
            fr = cam.read()
            corners, ids, _ = det.detectMarkers(cv2.cvtColor(fr.rgb, cv2.COLOR_RGB2GRAY))
            for c, i in zip(corners, [] if ids is None else ids.ravel()):
                if int(i) in ids_wanted:
                    ok, rv, tv = cv2.solvePnP(obj, c.reshape(4, 2), fr.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if ok:
                        seen.setdefault(int(i), []).append((Tbc @ np.r_[tv.ravel(), 1.0])[:3])
    finally:
        cam.close()
    return {i: np.median(v, 0) for i, v in seen.items() if len(v) >= 3}


def basket_frame_from(P, b):
    """Basket frame (4x4, base) from marker positions: origin = mean of the four, x = robot side -> far side,
    z = world up."""
    m = b['marker_ids']
    o = np.mean([P[i] for i in m.values()], 0)
    x = P[m['far']] - P[m['robot_side']]
    x[2] = 0.0
    x /= np.linalg.norm(x)
    z = np.array([0, 0, 1.0])
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
    T[:3, 3] = o
    return T


def basket_fit(P, layout):
    """Basket pose from >= 2 seen markers and their layout positions (planar rigid fit, z = world up)."""
    ids = [i for i in P if str(i) in layout]
    if len(ids) < 2:
        raise RuntimeError(f'basket: only {len(ids)} known markers visible')
    A = np.array([layout[str(i)][:2] for i in ids])
    B = np.array([P[i][:2] for i in ids])
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R2 = (U @ Vt).T
    if np.linalg.det(R2) < 0:
        Vt[-1] *= -1
        R2 = (U @ Vt).T
    T = np.eye(4)
    T[:2, :2] = R2
    T[:2, 3] = cb - R2 @ ca
    T[2, 3] = np.mean([P[i][2] - layout[str(i)][2] for i in ids])
    res = max(np.linalg.norm((T @ np.r_[layout[str(i)], 1.0])[:3] - P[i]) for i in ids)
    return T, res, ids


def basket_pins_in_base(T_base_basket):
    """Pin (base, top) points of the parametric basket model (rl/basket_env.full_basket_parts, numpy only) in base."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rl'))
    import basket_env as be
    r = be.B['full']['pin_r']
    pins = [(p0, p1) for kind, p0, p1, *rest in be.full_basket_parts() if kind == 'cyl' and rest and abs(rest[0] - r) < 1e-9]
    T = np.asarray(T_base_basket, float)
    return [((T @ np.r_[p0, 1])[:3], (T @ np.r_[p1, 1])[:3]) for p0, p1 in pins]


def wrist_snapshot(cfg, T_base_flange, T_base_basket, path, n_warm=15, after=None, share=Path('/dev/shm/wrist_latest.npz')):
    """(10/12, read-only) one wrist-D455 colour frame at a stop above the basket, saved with the model pins projected
    (green, from the marker / taught basket pose + wrist hand-eye) so the Human sees pin-vs-plate alignment, and a json
    with everything needed to fit the real pins later.  Returns the png path."""
    from calib_core import load_json, newest_wrist_result
    from sensors import ColorCam
    Tfc = np.array(load_json(newest_wrist_result())['T_flange_cam'])
    frame = None
    if after is not None:                                         # perception owns the wrist camera: its shared frame
        t0 = time.time()
        while frame is None and time.time() - t0 < 3:
            try:
                z = np.load(share)
                if float(z['t']) > after:
                    from sensors import Frame
                    frame = Frame(z['rgb'], z['K'])
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(0.1)
    if frame is None:
        cam = ColorCam(cfg['cameras']['wrist_serial'])
        try:
            for _ in range(n_warm):                               # auto exposure settles
                cam.read()
            frame = cam.read()
        finally:
            cam.close()
    Tcb = np.linalg.inv(np.asarray(T_base_flange, float) @ Tfc)
    img = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    raw = img.copy()
    for p0, p1 in basket_pins_in_base(T_base_basket):
        a, b = (Tcb @ np.r_[p0, 1])[:3], (Tcb @ np.r_[p1, 1])[:3]
        if a[2] <= 0.05 or b[2] <= 0.05:
            continue
        ua, ub = frame.K @ (a / a[2]), frame.K @ (b / b[2])
        cv2.line(img, (int(ua[0]), int(ua[1])), (int(ub[0]), int(ub[1])), (0, 255, 0), 1)
        cv2.circle(img, (int(ub[0]), int(ub[1])), 3, (0, 255, 0), 1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path.with_suffix('.raw.png')), raw)
    cv2.imwrite(str(path), img)
    path.with_suffix('.json').write_text(json.dumps(dict(T_base_flange=np.asarray(T_base_flange).tolist(), T_flange_cam=Tfc.tolist(),
                                                         T_base_basket=np.asarray(T_base_basket).tolist(), K=frame.K.tolist()), indent=1))
    return path


def plate_gap_dx(png, plate_bottom_z, plate_centre_y, pitch=0.02):
    """(10/12 real) pre-stop check on a wrist_snapshot: along each of the two slot pin rows (model pins of the basket
    pose stored with the photo, the rows nearest plate_centre_y on either side) find the plate's +x-most red pixel
    (the leaning plate's lowest rim side), back-project it to plate_bottom_z and compare with the nearest pin-gap
    centre.  Returns dict(dx = mean move along base x that centres the rim in the gap, rows=[...]) or None."""
    png = Path(png)
    d = json.loads(png.with_suffix('.json').read_text())
    img = cv2.imread(str(png.with_suffix('.raw.png')))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red = (((hsv[..., 0] < 8) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 60)).astype(np.uint8)
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(red)
    red = lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) if n_lab > 1 else red > 0     # the plate = biggest red blob
    Tc = np.array(d['T_base_flange']) @ np.array(d['T_flange_cam'])
    Tcb, K = np.linalg.inv(Tc), np.array(d['K'])

    def proj(p):
        c = Tcb @ np.r_[p, 1.0]
        return (K @ (c[:3] / c[2]))[:2]

    def back(u, v, z):
        r = Tc[:3, :3] @ np.linalg.solve(K, [u, v, 1.0])
        return Tc[:3, 3] + r * (z - Tc[2, 3]) / r[2]
    Tb = np.array(d['T_base_basket'])
    tops = [p1 for _, p1 in basket_pins_in_base(Tb)]
    ys = np.array([(np.linalg.inv(Tb) @ np.r_[p, 1.0])[1] for p in tops])     # rows are straight in the basket frame
    plate_centre_y = float((np.linalg.inv(Tb) @ np.r_[Tb[0, 3], plate_centre_y, 0, 1.0])[1])   # same frame (yaw ~1 deg)
    rows = []
    for side in (-1, 1):                                        # nearest row below / above the plate centre in y
        cand = ys[(ys - plate_centre_y) * side > 0.02]
        if not len(cand):
            continue
        yr = cand[np.argmin(np.abs(cand - plate_centre_y))]
        row = np.array([p for p, y in zip(tops, ys) if abs(y - yr) < 0.005])
        row = row[np.abs(row[:, 0] - np.median(row[:, 0])) < 0.15]          # this half of the basket (centre clip gap)
        xs = np.sort(row[:, 0])
        mid = np.r_[row[:, :2].mean(0), plate_bottom_z]
        uv, du = proj(mid), proj(mid + [0.05, 0, 0]) - proj(mid)       # image direction of base +x along the row
        n = np.array([-du[1], du[0]]) / np.linalg.norm(du)
        hh, ww = red.shape
        vv, uu = np.nonzero(red)
        on = np.abs((uu - uv[0]) * n[0] + (vv - uv[1]) * n[1]) < 3       # red pixels on the row line
        if on.sum() < 5:
            continue
        k = np.argmax((uu[on] - uv[0]) * du[0] + (vv[on] - uv[1]) * du[1])
        edge = back(uu[on][k], vv[on][k], plate_bottom_z)[0]
        gaps = (xs[:-1] + xs[1:]) / 2
        gaps = gaps[np.diff(xs) < 1.5 * pitch]
        g = gaps[np.argmin(np.abs(gaps - edge))]
        rows.append(dict(row_y_basket=round(float(yr), 3), edge_x=round(float(edge), 4), gap_x=round(float(g), 4), dx=round(float(g - edge), 4)))
    if not rows:
        return None
    return dict(dx=float(np.mean([r['dx'] for r in rows])), rows=rows)


def pre_photo(a, cfg):
    """(10/12) pipeline helper run as a child process (keeps OpenCV out of the pipeline process, which segfaulted once
    after a photo + measurement): --req json {T_base_flange, T_base_basket, png, after, gap: {plate_bottom_z,
    plate_centre_y} | null} -> writes {png, gap} to <png>.result.json."""
    r = json.loads(Path(a.req).read_text())
    png = wrist_snapshot(cfg, np.array(r['T_base_flange']), np.array(r['T_base_basket']), r['png'], after=r['after'])
    gap = plate_gap_dx(png, r['gap']['plate_bottom_z'], r['gap']['plate_centre_y']) if r.get('gap') else None
    Path(str(png) + '.result.json').write_text(json.dumps(dict(png=str(png), gap=gap)))


def wrist_look(a, cfg):
    """(10/12, read-only) wrist photo at the robot's CURRENT pose (Human jogs it with the pendant; needs
    `run.sh stream` for the joints) with the model pins drawn (green), basket = taught pose -> data/basket/wrist_look/."""
    from robot_model import Kinematics
    from sensors import PoseStream
    stream = PoseStream(cfg['udp']['flange'])
    t0 = time.time()
    while (s := stream.latest(1.0)) is None or s[2] is None:
        if time.time() - t0 > 20:
            raise SystemExit('no joints on UDP flange: start `bash calibration/run.sh stream` first')
        time.sleep(0.1)
    Tbb = np.array(json.loads((DATA / 'basket' / 'slot_01_in_basket.json').read_text())['T_base_basket_at_teach'])
    Tbb[:3, 3] += cfg['plate_ik_bc'].get('basket_shift_m', [0.0, 0.0, 0.0])   # where the basket is now (10/12)
    png = wrist_snapshot(cfg, Kinematics().fk(np.radians(s[2])), Tbb, DATA / 'basket' / 'wrist_look' / (time.strftime('%Y%m%d_%H%M%S') + '.png'))
    print(f'wrist photo: {png}  (joints {np.round(s[2], 1).tolist()})', flush=True)


def basket_fit_robust(P, layout, max_res=0.008):
    """(10/12) basket_fit with the residual taken in xy (the fit is planar; single-camera ArUco z is the noisy part)
    and, if all markers do not agree within max_res, the best 3-marker subset (one marker misread ~1 cm when the
    basket stands elsewhere).  Returns (T, xy residual, ids used)."""
    def xy_res(T, ids):
        return max(np.linalg.norm(((T @ np.r_[layout[str(i)], 1.0])[:3] - P[i])[:2]) for i in ids)
    T, _, ids = basket_fit(P, layout)
    best = (T, xy_res(T, ids), ids)
    if best[1] > max_res and len(ids) >= 4:
        for drop in ids:
            sub = {i: P[i] for i in ids if i != drop}
            T2, _, ids2 = basket_fit(sub, layout)
            if xy_res(T2, ids2) < best[1]:
                best = (T2, xy_res(T2, ids2), ids2)
    return best


def basket(a, cfg):
    """--define: measure the four markers, define the basket frame, write the layout (and, with --golden, slot 1's
    taught keyposes in the basket frame).  Without --define: the current basket pose from the stored layout."""
    import json as _json
    from calib_core import cpath
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    b = cfg['basket']
    P = basket_measure(cfg)
    print('markers seen:', {i: np.round(v, 4).tolist() for i, v in sorted(P.items())}, flush=True)
    layout_path = cpath(b['layout'])
    if a.define:
        missing = [k for k, i in b['marker_ids'].items() if i not in P]
        if missing:
            raise SystemExit(f'--define needs all four markers; missing {missing}')
        T = basket_frame_from(P, b)
        Ti = np.linalg.inv(T)
        layout = {str(i): (Ti @ np.r_[P[i], 1.0])[:3].round(5).tolist() for i in b['marker_ids'].values()}
        layout_path.parent.mkdir(parents=True, exist_ok=True)
        layout_path.write_text(_json.dumps(layout, indent=1))
        print(f'basket frame (base): origin {np.round(T[:3, 3], 4).tolist()}  x {np.round(T[:3, 0], 3).tolist()}\n'
              f'layout -> {layout_path}', flush=True)
        if a.golden:
            from robot_model import Kinematics
            import plate_plans as PP
            t = _json.loads(cpath(a.golden).read_text())['poses']
            kin = Kinematics()
            slot = {n: (Ti @ PP._tcp(kin, t[n]['posj'])).round(6).tolist() for n in ('carry', 'above', 'insert', 'retreat')}
            out = layout_path.parent / 'slot_01_in_basket.json'
            out.write_text(_json.dumps(dict(source=str(a.golden), T_base_basket_at_teach=T.round(6).tolist(),
                                            tcp_in_basket=slot, posj_at_teach={n: t[n]['posj'] for n in slot}), indent=1))
            print(f'slot 1 (golden teaching) in the basket frame -> {out}', flush=True)
        return
    if a.define_current:    # (10/12) markers re-taped: store them in the frame of the basket AS IT IS NOW, known from the
        # teaching frame + basket_shift_m (wrist photos: model pins on the real ones), so --place basket's delta needs no hand shift
        missing = [k for k, i in b['marker_ids'].items() if i not in P]
        if missing:
            raise SystemExit(f'--define_current needs all four markers; missing {missing}')
        T = np.array(_json.loads((DATA / 'basket' / 'slot_01_in_basket.json').read_text())['T_base_basket_at_teach'])
        T[:3, 3] += cfg['plate_ik_bc'].get('basket_shift_m', [0.0, 0.0, 0.0])
        Ti = np.linalg.inv(T)
        layout = {str(i): (Ti @ np.r_[P[i], 1.0])[:3].round(5).tolist() for i in b['marker_ids'].values()}
        new_path = layout_path.with_name(f'basket_layout_{time.strftime("%Y%m%d_%H%M%S")}.json')
        new_path.write_text(_json.dumps(dict(layout, note='markers in the teaching basket frame moved by basket_shift_m'), indent=1))
        T2, res, ids = basket_fit(P, layout)
        print(f'layout -> {new_path}  (self-fit residual {res * 1000:.1f} mm; set config basket.layout to it)', flush=True)
        return
    layout = _json.loads(layout_path.read_text())
    T, res, ids = basket_fit(P, layout)
    yaw = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
    print(f'basket pose: origin {np.round(T[:3, 3], 4).tolist()}  yaw {yaw:+.1f} deg  from markers {ids}  '
          f'fit residual {res * 1000:.1f} mm', flush=True)


if __name__ == '__main__':
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=('preview', 'marker', 'probe', 'locate', 'basket', 'wrist_look', 'pre_photo'))
    p.add_argument('--define', action='store_true', help='basket: define the basket frame / marker layout now')
    p.add_argument('--define_current', action='store_true', help='basket: marker layout in today\'s known basket frame (10/12)')
    p.add_argument('--golden', default=None, help='basket --define: also store this full teaching\'s slot in the basket frame')
    p.add_argument('--serial', action='append' if 'preview' in sys.argv else 'store',
                   default=None if 'preview' in sys.argv else cfg['cameras']['fixed_serial'])
    p.add_argument('--all', action='store_true', help='preview: every connected D455')
    p.add_argument('--image', type=Path, default=None, help='probe: offline image')
    p.add_argument('--req', default=None, help='pre_photo: request json (pipeline)')
    p.add_argument('--marker_id', type=int, default=None)
    p.add_argument('--marker_mm', type=float, default=None)
    p.add_argument('--dictionary', default=None)
    a = p.parse_args()
    if a.mode == 'preview':
        preview(a)
    elif a.mode == 'marker':
        marker(a, cfg)
    elif a.mode == 'locate':
        locate(a, cfg)
    elif a.mode == 'basket':
        basket(a, cfg)
    elif a.mode == 'wrist_look':
        wrist_look(a, cfg)
    elif a.mode == 'pre_photo':
        pre_photo(a, cfg)
    else:
        probe(a)
