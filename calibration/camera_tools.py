"""D455 camera tools (conda env `foundationpose`, camera only, no robot).

  python calibration/camera_tools.py preview [--all | --serial S [--serial S2]]   live view; S saves colour+depth+K (one --serial: data/tray_capture/, full 1280x720)
  python calibration/camera_tools.py marker [--serial S] [--marker_mm 100]        live ArUco check + averaged report
  python calibration/camera_tools.py probe [--serial S | --image F] [--marker_mm M]   identify an unknown marker board
  python calibration/camera_tools.py locate                                       plate position vs the RL sim spot
  python calibration/camera_tools.py pins --out NEW_DIR [--layout MEASURED.json]  saved wrist photos only; numbered diagnostic overlays
  python calibration/camera_tools.py basket_stats --out NEW_DIR                   fixed camera repeatability; --marker_size_m is actual printed size
  python calibration/camera_tools.py pin_layout --labels LABELS.json --out NEW_DIR   ray/height-plane measurement from checked pixel labels
  python calibration/camera_tools.py plate_angle --req wrist_above.req.json --out NEW_DIR  RGB disk silhouette yaw diagnostic
  python calibration/camera_tools.py plate_plane --rgb RGB.png --depth DEPTH.npy --meta SNAPSHOT.json --out NEW_DIR  saved RGB-D plane diagnostic
  python calibration/camera_tools.py slot_targets --out NEW.json  reproducible quadrant/pin-to-gap table

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


def basket_measure(cfg, n_frames=None, return_samples=False):
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
    up = int(b.get('detect_upsample', 1))                    # (10/14) 5 cm markers: 2x image + sub-pixel corners
    if up > 1:
        prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        prm.minMarkerPerimeterRate = 0.01
    det = cv2.aruco.ArucoDetector(d, prm)
    s = b['marker_size_m'] / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
    cam = ColorCam(cfg['cameras']['fixed_serial'])
    seen, frame_ids = {}, []
    try:
        for _ in range(30):
            cam.read()
        for _ in range(n_frames or b['n_frames']):
            fr = cam.read()
            detected_ids = []
            g = cv2.cvtColor(fr.rgb, cv2.COLOR_RGB2GRAY)
            corners, ids, _ = det.detectMarkers(cv2.resize(g, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up > 1 else g)
            for c, i in zip(corners, [] if ids is None else ids.ravel()):
                if int(i) in ids_wanted:
                    ok, rv, tv = cv2.solvePnP(obj, (c.reshape(4, 2) / up).astype(np.float32), fr.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if ok:
                        seen.setdefault(int(i), []).append((Tbc @ np.r_[tv.ravel(), 1.0])[:3])
                        detected_ids.append(int(i))
            frame_ids.append(detected_ids)
    finally:
        cam.close()
    positions = {i: np.median(v, 0) for i, v in seen.items() if len(v) >= 3}
    return (positions, seen, frame_ids) if return_samples else positions


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
        while frame is None and time.time() - t0 < 8:            # (10/19 17:4x: the after-release frame came later than 3 s,
                                                                  # the fallback found the wrist camera busy -> no photo)
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


def pre_photo(a, cfg):
    """(10/12) pipeline helper run as a child process (keeps OpenCV out of the pipeline process):
    --req json {T_base_flange, T_base_basket, png, after}."""
    r = json.loads(Path(a.req).read_text())
    wrist_snapshot(cfg, np.array(r['T_base_flange']), np.array(r['T_base_basket']), r['png'], after=r['after'])


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


# ---------------------------------------------------------------- offline wrist pin pairs

def _pin_project(points, Tcb, K):
    xyz = np.asarray(points) @ Tcb[:3, :3].T + Tcb[:3, 3]
    uv = xyz @ K.T
    return uv[:, :2] / xyz[:, 2:3], xyz[:, 2]


def _pixel_to_basket_plane(uv, z_local, T_base_basket, T_base_cam, K):
    """Intersect a camera ray with a known basket-height plane; no depth is inferred from the image."""
    uv = np.asarray(uv, float)
    Tbb, Tbc, K = (np.asarray(v, float) for v in (T_base_basket, T_base_cam, K))
    if uv.shape != (2,) or K.shape != (3, 3) or not np.isfinite(uv).all():
        raise ValueError('pixel and saved 3x3 intrinsics required')
    direction = Tbc[:3, :3] @ (np.linalg.inv(K) @ np.r_[uv, 1.0])
    plane_z = (Tbb @ np.array([0., 0., z_local, 1.]))[2]
    if abs(direction[2]) < 1e-8:
        raise ValueError('camera ray nearly parallel to pin height plane')
    scale = (plane_z - Tbc[2, 3]) / direction[2]
    if scale <= 0:
        raise ValueError('pin height plane behind camera')
    base_point = Tbc[:3, 3] + scale * direction
    return (np.linalg.inv(Tbb) @ np.r_[base_point, 1.])[:3]


def pin_layout_measure(labels_path, out):
    """Convert manually checked tip/root pixels to 3D. Labels remain provenance, never inferred ground truth."""
    labels_path, out = Path(labels_path), Path(out)
    labels = json.loads(labels_path.read_text())
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rl'))
    import basket_env as be
    floor = be.B['full']['floor_z']
    root_z = floor + be.B['pins']['rail_z']
    top = floor + be.B['pins']['height'] * np.cos(np.radians(be.B['pins']['lean_deg']))
    measured = []
    seen = set()
    for photo in labels['photos']:
        path = Path(photo['image'])
        if not path.name.endswith('.raw.png'):
            raise ValueError(f'Expected original wrist .raw.png: {path}')
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        height, width = image.shape[:2]
        meta = json.loads(path.with_name(path.name[:-8] + '.json').read_text())
        Tbb = np.asarray(meta['T_base_basket'], float)
        Tbc = np.asarray(meta['T_base_flange'], float) @ np.asarray(meta['T_flange_cam'], float)
        K = np.asarray(meta['K'], float)
        if Tbb.shape != (4, 4) or Tbc.shape != (4, 4) or K.shape != (3, 3) or not all(
                np.isfinite(v).all() for v in (Tbb, Tbc, K)):
            raise ValueError(f'Invalid saved camera or basket transform: {path}')
        for pin in photo['pins']:
            if pin.get('review') != 'checked':
                continue
            if pin.get('row') not in ('pair_A_row_plus', 'pair_A_row_minus', 'pair_B_row_plus', 'pair_B_row_minus'):
                raise ValueError(f'Invalid pin row: {pin.get("row")}')
            for name in ('root_uv', 'tip_uv'):
                u, v = np.asarray(pin[name], float)
                if not (np.isfinite([u, v]).all() and 0 <= u < width and 0 <= v < height):
                    raise ValueError(f'{name} outside saved image: {path}')
            key = (str(path.resolve()), pin['row'], tuple(pin['tip_uv']))
            if key in seen:
                raise ValueError(f'Duplicate checked pin: {key}')
            seen.add(key)
            root = _pixel_to_basket_plane(pin['root_uv'], root_z, Tbb, Tbc, K)
            tip = _pixel_to_basket_plane(pin['tip_uv'], top, Tbb, Tbc, K)
            dy = abs(root[1] - tip[1])
            base_at_floor = root + (tip-root)*(floor-root_z)/(top-root_z)
            measured.append(dict(image=str(path), row=pin['row'], root_uv=pin['root_uv'], tip_uv=pin['tip_uv'],
                                 root_at_rail_basket_xyz=root.tolist(), base_basket_xyz=base_at_floor.tolist(),
                                 top_basket_xyz=tip.tolist(),
                                 root_x_m=float(root[0]), tip_x_m=float(tip[0]),
                                 transverse_disagreement_mm=float(dy * 1000),
                                 lean_deg=float(np.degrees(np.arctan2(root[0] - tip[0], tip[2] - root[2]))),
                                 review=pin['review']))
    if not measured or labels.get('counts', {}).get('total', len(measured)) != len(measured):
        raise ValueError('No checked pins or labels.counts.total disagrees with checked pins')
    result = dict(source_labels=str(labels_path.resolve()), assumed_base_z_m=floor,
                  assumed_visible_root_z_m=root_z, assumed_tip_z_m=top,
                  method='saved camera ray intersection with visible comb-rail root and tip height planes; '
                         'floor base extrapolated along measured segment',
                  accuracy_note='Pixel clicks and assumed heights are not independently measured 3D ground truth.',
                  pins=measured)
    out.mkdir(parents=True, exist_ok=False)
    (out/'pin_layout_measured.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(out=str(out), checked_pins=len(measured)), ensure_ascii=False))
    return result


def _pin_image_context(img, cfg):
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0.6)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    excluded = ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165)) & (hsv[:, :, 1] > 60)
    excluded = cv2.dilate(excluded.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg['marker']['dictionary']))
    corners, _, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(gray)
    for c in corners:
        c = c.reshape(-1, 2)
        c = (c-c.mean(0))*1.25+c.mean(0)
        cv2.fillConvexPoly(excluded.view(np.uint8), c.astype(np.int32), 1)
    edges = cv2.Canny(gray, 45, 110)
    edges[excluded] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, minLineLength=12, maxLineGap=4)
    return gray.astype(np.float32), excluded, [] if lines is None else lines.reshape(-1, 4)


def _pin_longest_run(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, bool), False]))
    return int((edges[1::2]-edges[::2]).max()) if len(edges) else 0


def _pin_longest_run_bridged(mask, max_gap=2):
    """Longest support after bridging only short, fully internal profile gaps."""
    bridged = np.asarray(mask, bool).copy()
    if max_gap < 0:
        raise ValueError('max_gap must be non-negative')
    missing = np.flatnonzero(~bridged)
    if len(missing):
        starts = missing[np.r_[True, np.diff(missing) > 1]]
        ends = missing[np.r_[np.diff(missing) > 1, True]]
        for start, end in zip(starts, ends):
            if start > 0 and end < len(bridged)-1 and end-start+1 <= max_gap:
                bridged[start:end+1] = True
    return _pin_longest_run(bridged)


def _solve_pin_pixel_offset(delta_px, jacobian_px_per_mm, search_mm, max_residual_px=12.0, search_y_mm=None):
    """Solve a local basket x/y tip offset from pixels; return None if ill-conditioned/out of bounds."""
    J = np.asarray(jacobian_px_per_mm, float)
    delta = np.asarray(delta_px, float)
    if J.shape != (2, 2) or delta.shape != (2,) or not np.isfinite(J).all() or not np.isfinite(delta).all():
        return None
    if np.linalg.cond(J) > 40:
        return None
    correction = np.linalg.solve(J, delta)
    residual = delta - J @ correction
    bounds = np.array([search_mm, search_mm if search_y_mm is None else search_y_mm], float)
    if np.any(np.abs(correction) >= bounds) or np.linalg.norm(residual) > max_residual_px:
        return None
    return correction, residual


def _pin_pair_offset(first_xy, second_xy, tolerance_mm=3.0):
    """Combine two row-wise image offsets while preserving their disagreement for review."""
    a, b = np.asarray(first_xy, float), np.asarray(second_xy, float)
    if a.shape != (2,) or b.shape != (2,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    disagreement = np.abs(a-b)
    return dict(candidate_xy_mm=np.mean([a, b], axis=0), disagreement_xy_mm=disagreement,
                consistent=bool(np.all(disagreement <= tolerance_mm)))


def _pin_match(img, context, p0, p1, Tcb, K, axis, search_mm=8.0, axis_y=None, search_radius_px=19,
               search_y_mm=None):
    """Local dark-wire ridge fit. No depth estimate from a guessed plate plane; offsets translate the 3D pin.
    Evaluate only the exposed upper tine, away from floor mesh and rail. Reject plate/marker occlusion and weak fits."""
    gray, excluded, lines = context
    h, w = gray.shape
    # A 2-D fit must use the image-line endpoint to resolve both basket axes together.
    # The stripe scan below intentionally remains the legacy one-axis implementation.
    if axis_y is not None:
        result = _pin_line_match(context, p0, p1, Tcb, K, axis, search_mm,
                                 axis_y=axis_y, search_radius_px=search_radius_px,
                                 search_y_mm=search_y_mm)
        if result is not None:
            return result
        # Preserve the per-pin output contract when the Hough search has no usable line.
        model_uv, depth = _pin_project(np.stack([p0, p1]), Tcb, K)
        visible = bool(np.all(depth > 0.05) and np.all(model_uv[:, 0] >= 0)
                       and np.all(model_uv[:, 0] < img.shape[1])
                       and np.all(model_uv[:, 1] >= 0) and np.all(model_uv[:, 1] < img.shape[0]))
        return dict(status='weak_line' if visible else 'out_of_view', correction_mm=None,
                    candidate_mm=None, ridge_score=0.0, support=0.0,
                    visible_fraction=1.0 if visible else 0.0,
                    model_uv=model_uv.tolist(), detected_uv=model_uv.tolist())
    t = np.linspace(0.53, 0.98, 40)
    points = p0[None]+t[:, None]*(p1-p0)[None]
    model, depth = _pin_project(points, Tcb, K)
    tangent = model[-1]-model[0]
    length = float(np.linalg.norm(tangent))
    if depth.min() <= 0.05 or length < 12:
        return dict(status='out_of_view', correction_mm=None)
    normal = np.array([-tangent[1], tangent[0]])/length
    half_width = np.clip(float(K[0, 0])*0.00175/depth, 1.1, 5.0)
    offsets = np.arange(-search_mm, search_mm+0.01, 0.5)
    scores, supports, visibility, continuous = [], [], [], []
    def sample(a, xy):
        return cv2.remap(a, xy[:, 0].astype(np.float32)[None], xy[:, 1].astype(np.float32)[None],
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).ravel()
    for dx in offsets:
        uv, z = _pin_project(points+axis[None]*dx/1000, Tcb, K)
        valid = (uv[:, 0] > 8) & (uv[:, 0] < w-9) & (uv[:, 1] > 8) & (uv[:, 1] < h-9) & (z > 0.05)
        valid &= sample(excluded.astype(np.float32), uv) < 0.1
        visibility.append(float(valid.mean()))
        if valid.sum() < 28:
            scores.append(-1.0); supports.append(0.0); continuous.append(0.0); continue
        centre = sample(gray, uv)
        left = sample(gray, uv-normal[None]*(half_width+2.0)[:, None])
        right = sample(gray, uv+normal[None]*(half_width+2.0)[:, None])
        contrast = np.minimum(left, right)-centre
        support = (contrast > 12) & (centre < 145)
        supports.append(float(support[valid].mean()))
        continuous.append(_pin_longest_run_bridged(support & valid, max_gap=2)/len(t))
        scores.append(float(np.clip(contrast[valid], 0, 80).mean()))
    peak = int(np.argmax(scores))
    lo = hi = peak
    while lo > 0 and scores[lo-1] >= scores[peak]-2.5:
        lo -= 1
    while hi < len(scores)-1 and scores[hi+1] >= scores[peak]-2.5:
        hi += 1
    # A dark wire has a flat ridge top: use its centre, not the first pixel of the plateau.
    j = int(round((lo+hi)/2))
    score, support = scores[j], supports[j]
    plateau_width = float(offsets[hi]-offsets[lo])
    status = 'detected'
    if max(visibility) < 0.7:
        status = 'occluded'
    elif score < 14 or support < 0.55 or continuous[j] < 0.45:
        status = 'weak_line'
    elif abs(offsets[j]) >= search_mm-0.1 or lo == 0 or hi == len(offsets)-1:
        status = 'search_boundary'
    else:
        competitors = [v for v, dx in zip(scores, offsets) if abs(dx-offsets[j]) >= max(3.0, plateau_width/2+2.0)]
        if plateau_width > 6.0 or (competitors and score-max(competitors) < 2.5):
            status = 'ambiguous'
    fitted, _ = _pin_project(np.stack([p0, p1])+axis[None]*offsets[j]/1000, Tcb, K)
    if status in ('weak_line', 'search_boundary', 'ambiguous'):
        fallback = _pin_line_match(context, p0, p1, Tcb, K, axis, search_mm)
        if fallback is not None:
            return fallback
    return dict(status=status, correction_mm=float(offsets[j]) if status == 'detected' else None,
                candidate_mm=float(offsets[j]), ridge_score=round(score, 3), support=round(support, 3),
                continuity_ratio=round(float(continuous[j]), 3),
                visible_fraction=round(visibility[j], 3), model_uv=_pin_project(np.stack([p0, p1]), Tcb, K)[0].tolist(),
                detected_uv=fitted.tolist())


def _pin_line_match(context, p0, p1, Tcb, K, axis, search_mm, axis_y=None, search_radius_px=19,
                    search_y_mm=None):
    """Hough fallback for small errors transverse to the x-only model fit. Report unresolved pixel residual explicitly."""
    gray, excluded, lines = context
    uv, depth = _pin_project(np.stack([p0+(p1-p0)*0.53, p1]), Tcb, K)
    tangent = uv[1]-uv[0]
    length = np.linalg.norm(tangent)
    direction = tangent/length
    normal = np.array([-direction[1], direction[0]])
    Jx = _pin_project(p1[None]+axis[None]*0.001, Tcb, K)[0][0]-uv[1]
    if np.linalg.norm(Jx) < 0.3:
        return None
    J = Jx
    if axis_y is not None:
        Jy = _pin_project(p1[None]+np.asarray(axis_y)[None]*0.001, Tcb, K)[0][0]-uv[1]
        J = np.column_stack([Jx, Jy])
        if not np.isfinite(J).all() or np.linalg.cond(J) > 40:
            return None
    candidates = []
    h, w = gray.shape
    for line in lines:
        a, b = line[:2].astype(float), line[2:].astype(float)
        d = b-a; L = np.linalg.norm(d)
        if L < 0.55*length or L > 2.5*length or abs(np.dot(d/L, direction)) < 0.90:
            continue
        tip = a if np.linalg.norm(a-uv[1]) < np.linalg.norm(b-uv[1]) else b
        if np.linalg.norm(tip-uv[1]) > search_radius_px:
            continue
        t = np.linspace(0.12, 0.9, 24)
        coords = a[None]+t[:, None]*d
        if (coords[:, 0].min() < 6 or coords[:, 0].max() >= w-6 or coords[:, 1].min() < 6 or coords[:, 1].max() >= h-6):
            continue
        def sample(arr, xy):
            return cv2.remap(arr, xy[:, 0].astype(np.float32)[None], xy[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR).ravel()
        if (sample(excluded.astype(np.float32), coords) > 0.1).mean() > 0.2:
            continue
        best = None
        for shift in (-2, -1, 0, 1, 2):
            xy = coords+normal[None]*shift
            centre = sample(gray, xy)
            contrast = np.minimum(sample(gray, xy+normal[None]*4), sample(gray, xy-normal[None]*4))-centre
            score = np.clip(contrast, 0, 80).mean()
            support = ((contrast > 12) & (centre < 145)).mean()
            if best is None or score > best[0]:
                best = (score, support, tip+normal*shift)
        if best[0] < 20 or best[1] < 0.65:
            continue
        xy = coords+normal[None]*np.dot(best[2]-tip, normal)
        ridge_mask = (np.minimum(sample(gray, xy+normal[None]*4), sample(gray, xy-normal[None]*4))-sample(gray, xy) > 12)
        if _pin_longest_run_bridged(ridge_mask, max_gap=2)/len(ridge_mask) < 0.45:
            continue
        delta = best[2]-uv[1]
        if axis_y is None:
            correction = np.array([float(np.dot(delta, J)/np.dot(J, J))])
            residual = delta-J*correction[0]
        else:
            solved = _solve_pin_pixel_offset(delta, J, search_mm, search_y_mm=search_y_mm)
            if solved is None:
                continue
            correction, residual = solved
        if axis_y is None and (np.any(np.abs(correction) >= search_mm) or np.linalg.norm(residual) > 12):
            continue
        candidates.append((float(best[0]), float(best[1]), correction, residual, best[2]))
    if not candidates:
        return None
    candidates.sort(key=lambda c: -c[0])
    best = candidates[0]
    if any(c[0] > best[0]-3 and np.linalg.norm(c[2]-best[2]) > 3 for c in candidates[1:]):
        return None
    base_uv = _pin_project(np.stack([p0, p1]), Tcb, K)[0]
    corrected = base_uv+(best[4]-base_uv[1])[None]
    result = dict(status='detected', method='hough_ridge', correction_mm=round(float(best[2][0]), 3), candidate_mm=round(float(best[2][0]), 3),
                ridge_score=round(best[0], 3), support=round(best[1], 3), visible_fraction=1.0,
                transverse_residual_px=round(float(np.linalg.norm(best[3])), 3),
                model_uv=base_uv.tolist(), detected_uv=corrected.tolist())
    if axis_y is not None:
        result['correction_xy_mm'] = np.round(best[2], 3).tolist()
    return result


def _gap_xy_offset(model_xy, detections):
    """Return measured-minus-model gap offset; x needs both bracket tines in each row, y needs both rows."""
    model_xy = np.asarray(model_xy, float)
    by_row = {}
    for item in detections:
        if item.get('status') != 'detected' or 'correction_xy_mm' not in item:
            continue
        by_row.setdefault(item['row'], []).append(item)
    row_centres = {}
    for row, items in by_row.items():
        edges = {item['edge'] for item in items}
        if not items:
            continue
        offsets = np.asarray([item['correction_xy_mm'] for item in items], float) / 1000.0
        local = np.asarray([item['model_local_xy_m'] for item in items], float) + offsets
        row_centres[row] = dict(xy=np.mean(local, axis=0).tolist(), edges=sorted(edges), count=len(items))
    if len(row_centres) < 2:
        return dict(status='insufficient_rows', measured_xy_m=None, measured_minus_model_mm=None,
                    x_mm=None, y_mm=None, row_centres=row_centres)
    centre_xy = np.mean([v['xy'] for v in row_centres.values()], axis=0)
    rows_complete = all({'low_x', 'high_x'} <= set(v['edges']) for v in row_centres.values())
    return dict(status='detected' if rows_complete else 'partial', measured_xy_m=centre_xy.tolist(),
                measured_minus_model_mm=np.round((centre_xy-model_xy)*1000, 3).tolist(),
                x_mm=round(float((centre_xy[0]-model_xy[0])*1000), 3) if rows_complete else None,
                y_mm=round(float((centre_xy[1]-model_xy[1])*1000), 3), row_centres=row_centres)


def _checked_gap_root_offset(audit_path, quadrant, pin):
    """Use previously human-checked 3-D tine roots; this is a separate geometry prior, not photo detection."""
    audit_path = Path(audit_path)
    if not audit_path.is_file():
        return dict(status='missing_checked_layout', source=str(audit_path))
    audit = json.loads(audit_path.read_text())
    by_row = {}
    for item in audit.get('pins', []):
        if item.get('quadrant') != quadrant or item.get('visibility') != 'complete':
            continue
        by_row.setdefault(item['image_row'], {})[int(item['number'])] = item
    row_centres = {}
    for row, pins in by_row.items():
        if pin not in pins or pin+1 not in pins:
            continue
        pair = (pins[pin], pins[pin+1])
        actual = np.mean([np.asarray(v['root_basket_xyz_mm'], float)[:2] for v in pair], axis=0)
        model = np.mean([[float(v['root_model_x_mm']), float(v['model_row_y_mm'])] for v in pair], axis=0)
        row_centres[row] = dict(actual_xy_mm=actual.tolist(), model_xy_mm=model.tolist(),
                                measured_minus_model_mm=(actual-model).tolist(),
                                source_pin_numbers=[pin, pin+1])
    if len(row_centres) < 2:
        return dict(status='insufficient_checked_rows', source=str(audit_path), row_centres=row_centres)
    actual = np.mean([v['actual_xy_mm'] for v in row_centres.values()], axis=0)
    model = np.mean([v['model_xy_mm'] for v in row_centres.values()], axis=0)
    actual_z = float(np.mean([np.mean([audit_pin['root_basket_xyz_mm'][2] for audit_pin in
                                      (by_row[row][pin], by_row[row][pin+1])]) for row in row_centres]))
    return dict(status='measured_prior', source=str(audit_path.resolve()), rows=len(row_centres),
                visible_checked_tines=2*len(row_centres), actual_gap_root_centre_basket_xy_mm=actual.tolist(),
                model_gap_root_centre_basket_xy_mm=model.tolist(),
                root_z_basket_mm=actual_z,
                measured_minus_model_xy_mm=np.round(actual-model, 3).tolist(), row_centres=row_centres,
                provenance='mean of the human-checked rail-root coordinates for adjacent tine numbers pin,pin+1 in both basket rows; not detected in this current wrist photo')


def wrist_gap_error(meta_path, out, quadrant=None, pin=None, search_mm=35.0, layout_audit=None):
    """Offline-only 2D model-vs-visible-tine gap measurement from one saved wrist-above photo."""
    meta_path, out = Path(meta_path), Path(out)
    meta = json.loads(meta_path.read_text())
    raw_path = meta_path.with_name(meta_path.name.replace('.json', '.raw.png'))
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    run_dir = meta_path.parent
    states = run_dir/'robot_state.jsonl'
    done = None
    if states.is_file():
        import json as _json
        for line in states.read_text().splitlines():
            try:
                event = _json.loads(line)
            except ValueError:
                continue
            if event.get('kind') == 'event' and event.get('event') == 'done' and event.get('insert') == 'quadrant':
                done = event
    quadrant = quadrant or (done or {}).get('quadrant')
    pin = int(pin if pin is not None else (done or {}).get('pin', 0))
    if quadrant not in ('Q1', 'Q2', 'Q3', 'Q4') or pin < 1:
        raise ValueError('need a saved quadrant-insert completion or explicit --quadrant and --pin')
    img = cv2.imread(str(raw_path))
    if img is None:
        raise FileNotFoundError(raw_path)
    Tbb = np.asarray(meta['T_base_basket'], float)
    Tbc = np.asarray(meta['T_base_flange'], float) @ np.asarray(meta['T_flange_cam'], float)
    K = np.asarray(meta['K'], float)
    if Tbb.shape != (4, 4) or Tbc.shape != (4, 4) or K.shape != (3, 3) or not all(
            np.isfinite(v).all() for v in (Tbb, Tbc, K)):
        raise ValueError('saved T_base_basket, flange/camera transforms and K are required')
    from plate_plans import basket_gaps, pin_gap, robot_quadrants
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'rl'))
    import basket_env as be
    model_qd = robot_quadrants(Tbb)[quadrant]
    gap_index = int(pin_gap(Tbb, model_qd, pin))
    model_gap = np.asarray(basket_gaps()[model_qd][gap_index], float)
    if search_mm <= 0 or search_mm > 50:
        raise ValueError('search_mm must be within (0, 50]')
    # Select the four model tines bracketing this numbered slot: two x-adjacent tips on each of its two y rows.
    geometry = basket_pins_in_base(Tbb)
    Tbi = np.linalg.inv(Tbb)
    pair_y = be.B['full']['pair_centres_y'][0 if model_qd in ('Q2', 'Q3') else 1]
    local_pins = []
    for p0, p1 in geometry:
        local = (Tbi @ np.r_[p0, 1.0])[:3]
        is_far = model_qd in ('Q1', 'Q2')
        if ((local[0] > be.B['full'].get('clip_x', 0.0)) == is_far
                and abs(local[1] - pair_y) < 0.06):
            local_pins.append((local, p0, p1))
    xvals = sorted({round(float(v[0][0]), 6) for v in local_pins})
    if gap_index + 1 >= len(xvals):
        raise ValueError(f'{model_qd} slot index {gap_index} has no two model tine boundaries')
    x_edges = (xvals[gap_index], xvals[gap_index+1])
    row_ys = [float(pair_y+dy) for dy in be.B['pins']['rows_u']]
    axis_x, axis_y = Tbb[:3, 0], Tbb[:3, 1]
    Tcb = np.linalg.inv(Tbc)
    context = _pin_image_context(img, load_config())
    detections = []
    for row_id, y in enumerate(row_ys):
        for edge, x in zip(('low_x', 'high_x'), x_edges):
            local, p0, p1 = min(local_pins, key=lambda v: abs(v[0][0]-x)+abs(v[0][1]-y))
            if abs(local[0]-x) > 0.002 or abs(local[1]-y) > 0.002:
                raise ValueError('could not associate model tine with one of four target-gap corners')
            match = _pin_match(img, context, p0, p1, Tcb, K, axis_x, search_mm=search_mm,
                               axis_y=axis_y, search_radius_px=70)
            model_tip_uv = _pin_project(np.asarray([p1]), Tcb, K)[0][0]
            detection = dict(match or {})
            detection.update(row=f'row_{row_id}', edge=edge, model_local_xy_m=local[:2].tolist(),
                             model_tip_uv=model_tip_uv.tolist(),
                             status='unseen' if match is None else match['status'])
            detections.append(detection)
    offset = _gap_xy_offset(model_gap[:2], detections)
    if layout_audit is None:
        layout_audit = Path(__file__).resolve().parent/'data/basket/realQ_verified_3d_model_audit_20261014_v5/verified_pin_3d_model_comparison.json'
    checked_prior = _checked_gap_root_offset(layout_audit, quadrant, pin)
    view = img.copy()
    model_world = (Tbb @ np.r_[model_gap[:2], np.mean([be.B['full']['floor_z']+be.B['pins']['height'] for _ in row_ys]), 1.0])[:3]
    model_uv = _pin_project(model_world[None], Tcb, K)[0][0]
    for item in detections:
        p = tuple(np.rint(item['model_tip_uv']).astype(int))
        colour = (0, 180, 255) if item['status'] == 'detected' else (120, 120, 120)
        cv2.circle(view, p, 6, colour, 2)
        if item['status'] == 'detected':
            q = tuple(np.rint(item['detected_uv'][1]).astype(int))
            cv2.arrowedLine(view, p, q, (30, 220, 40), 2, tipLength=.25)
            cv2.putText(view, f"{item['row']} {item['edge']} {item['correction_xy_mm'][0]:+.1f},{item['correction_xy_mm'][1]:+.1f}mm",
                        (p[0]+5, p[1]-4), cv2.FONT_HERSHEY_SIMPLEX, .42, (30, 220, 40), 1, cv2.LINE_AA)
    mc = tuple(np.rint(model_uv).astype(int))
    cv2.drawMarker(view, mc, (255, 80, 0), cv2.MARKER_CROSS, 22, 2)
    if checked_prior['status'] == 'measured_prior':
        root_z = checked_prior['root_z_basket_mm']/1000.0
        prior_model = np.r_[np.asarray(checked_prior['model_gap_root_centre_basket_xy_mm'])/1000.0, root_z]
        prior_actual = np.r_[np.asarray(checked_prior['actual_gap_root_centre_basket_xy_mm'])/1000.0, root_z]
        prior_uv, _ = _pin_project((Tbb @ np.c_[np.stack([prior_model, prior_actual]), np.ones(2)].T).T[:, :3], Tcb, K)
        pm, pa = [tuple(np.rint(v).astype(int)) for v in prior_uv]
        cv2.drawMarker(view, pm, (255, 80, 0), cv2.MARKER_CROSS, 18, 2)
        cv2.drawMarker(view, pa, (255, 0, 255), cv2.MARKER_DIAMOND, 16, 2)
        cv2.arrowedLine(view, pm, pa, (255, 0, 255), 2, tipLength=.25)
        cv2.putText(view, f'checked 3D roots pin {pin}: dx/dy '
                    f'{checked_prior["measured_minus_model_xy_mm"][0]:+.1f}/'
                    f'{checked_prior["measured_minus_model_xy_mm"][1]:+.1f} mm',
                    (12, 52), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 0, 255), 2, cv2.LINE_AA)
    candidate_count = sum(v['status'] == 'detected' for v in detections)
    text = f'{quadrant} pin {pin} -> {model_qd}, photo candidates {candidate_count}/4 (unverified); '
    text += (f'delta slot xy {offset["x_mm"]},{offset["y_mm"]} mm' if offset['status'] == 'detected'
             else f'delta status {offset["status"]}; partial y={offset["y_mm"]} mm')
    cv2.putText(view, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2, cv2.LINE_AA)
    result = dict(source=str(meta_path.resolve()), image=str(raw_path.resolve()),
                  commanded_quadrant=quadrant, commanded_pin=pin, model_quadrant=model_qd,
                  model_gap_index_robot_to_far=gap_index, model_gap_centre_basket_xy_m=model_gap[:2].tolist(),
                  frame='basket/slot aligned: x along tine row, y transverse across rows; positive delta = physical minus model',
                  photo_candidate_matches=candidate_count, photo_candidate_count=4, photo_visible_tines_verified=None,
                  detection_confidence='unverified: no current-photo ground-truth labels; Hough candidates may be adjacent wires or rails',
                  detections=detections, photo_measurement=offset, checked_3d_layout_prior=checked_prior,
                  accuracy_note='Photo detections are unlabelled Hough candidates and are not accuracy evidence. The separate checked_3d_layout_prior uses earlier human-checked roots, assumes the same basket geometry, and is not a current-photo verification. Marker pose, hand-eye, height and basket differences remain uncertainty. Diagnostic only; no command correction.')
    out.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(out/'gap_error.png'), view)
    (out/'gap_error.json').write_text(json.dumps(result, indent=2))
    return result


def _gap_number_candidate(Tbb, quadrant, x_m, gaps=None):
    """Find a globally numbered model gap candidate and its Voronoi-cell tolerance."""
    from plate_plans import basket_gaps, pin_gap

    gaps = basket_gaps() if gaps is None else gaps
    candidates = []
    for number in range(1, len(gaps[quadrant]) + 1):
        index = pin_gap(Tbb, quadrant, number)
        candidates.append((abs(float(gaps[quadrant][index][0]) - x_m), number,
                           float(gaps[quadrant][index][0])))
    candidates.sort()
    error, number, expected_x = candidates[0]
    # Refuse an address when the measured/model x has crossed a nearest-slot boundary.
    neighbours = [abs(expected_x - x) for _, _, x in candidates[1:]]
    tolerance = 0.5 * min(neighbours) if neighbours else 0.0
    if error >= tolerance:
        return None, error, tolerance
    return number, error, tolerance


def _assign_gap_numbers(Tbb, groups, gaps=None):
    """Assign stable global IDs; duplicate measurements competing for one gap remain unnumbered."""
    by_candidate = {}
    for group in groups:
        candidate, error, tolerance = _gap_number_candidate(
            Tbb, group['quadrant'], group['local_x_m'], gaps)
        group.update(number=None, number_candidate=candidate,
                     number_source='nearest_parametric_gap_x',
                     number_match_error_mm=round(error * 1000, 3),
                     number_tolerance_mm=round(tolerance * 1000, 3))
        if candidate is not None:
            by_candidate.setdefault((group['quadrant'], candidate), []).append(group)
    for (_, candidate), matched in by_candidate.items():
        if len(matched) == 1:
            matched[0]['number'] = candidate


def _gap_groups_from_tine_pairs(Tbb, tine_pairs, gaps=None):
    """Build insertion-gap candidates from adjacent cross-row tine stations.

    A cross-row pair is one tine station, not a slot. Slot N lies between stations
    N and N+1, so only their midpoint is passed to the global gap number matcher.
    This remains an image diagnostic; the measured corrections are not a robot target.
    """
    from plate_plans import basket_gaps, pin_gap

    gaps = basket_gaps() if gaps is None else gaps
    out = []
    for quadrant in ('Q1', 'Q2', 'Q3', 'Q4'):
        stations = sorted((g for g in tine_pairs if g['quadrant'] == quadrant),
                          key=lambda g: float(g['local_x_m']))
        gap_x = sorted(float(g[0]) for g in gaps[quadrant])
        station_pitch = float(np.median(np.diff(gap_x))) if len(gap_x) > 1 else None
        for left, right in zip(stations[:-1], stations[1:]):
            station_delta = float(right['local_x_m'] - left['local_x_m'])
            if station_pitch is not None and not 0.5 * station_pitch <= station_delta <= 1.5 * station_pitch:
                continue  # A missing/spurious station must not create a false intervening gap.
            left_xy = left.get('candidate_correction_xy_mm')
            right_xy = right.get('candidate_correction_xy_mm')
            left_ok = left.get('status') == 'detected' and left.get('correction_xy_mm') is not None
            right_ok = right.get('status') == 'detected' and right.get('correction_xy_mm') is not None
            candidate_xy = (np.mean([left_xy, right_xy], axis=0).astype(float)
                            if left_xy is not None and right_xy is not None else None)
            correction_xy = (np.mean([left['correction_xy_mm'], right['correction_xy_mm']], axis=0).astype(float)
                             if left_ok and right_ok else None)
            row_ys = [*left.get('row_y_m', []), *right.get('row_y_m', [])]
            out.append(dict(quadrant=quadrant,
                            station_pair_ids=[left['station_id'], right['station_id']],
                            model_pin_ids=[*left['model_pin_ids'], *right['model_pin_ids']],
                            midpoint_uv=np.mean([left['midpoint_uv'], right['midpoint_uv']], axis=0).tolist(),
                            local_x_m=float((left['local_x_m'] + right['local_x_m']) / 2),
                            row_y_m=sorted(float(y) for y in row_ys),
                            candidate_correction_xy_mm=candidate_xy.tolist() if candidate_xy is not None else None,
                            correction_xy_mm=correction_xy.tolist() if correction_xy is not None else None,
                            correction_mm=float(correction_xy[0]) if correction_xy is not None else None,
                            status='detected' if left_ok and right_ok else
                                   ('pair_disagreement' if candidate_xy is not None else 'partial_or_occluded'),
                            gap_centre_base_xyz=None, candidate_gap_centre_base_xyz=None))
    _assign_gap_numbers(Tbb, out, gaps)
    Tbb = np.asarray(Tbb, float)
    axis, axis_y = Tbb[:3, 0], Tbb[:3, 1]
    for group in out:
        candidate_xy = group['candidate_correction_xy_mm']
        if candidate_xy is None or group['number_candidate'] is None:
            continue
        model_gap = np.asarray(gaps[group['quadrant']][pin_gap(Tbb, group['quadrant'],
                                                               group['number_candidate'])], float)
        base = (Tbb @ np.r_[model_gap, 1.0])[:3]
        shift = axis * (candidate_xy[0] / 1000.0) + axis_y * (candidate_xy[1] / 1000.0)
        group['candidate_gap_centre_base_xyz'] = (base + shift).tolist()
        if group['status'] == 'detected' and group['number'] is not None:
            group['gap_centre_base_xyz'] = group['candidate_gap_centre_base_xyz']
    return out


def _pin_quadrant(x_m, y_m, clip_x):
    """Classify basket-frame rows around the modeled centre clip, not the basket origin."""
    if x_m > clip_x:
        return 'Q2' if y_m >= 0 else 'Q1'
    return 'Q3' if y_m >= 0 else 'Q4'


def wrist_pins(req, cfg, out, image=None, rotate='auto', search_mm=8.0, layout=None,
               model_search_2d=False):
    """Stored image only. Pair model rows by equal x or checked measured rows by nearby x.
    Display top-to-bottom IDs within each quadrant.
    Auto rotation aligns the two physical rows left-to-right in the saved display. Raw pixel geometry stays in JSON.
    Rejected/occluded pins retain their model IDs; no automatic robot correction is performed."""
    if not 0 < search_mm <= 8:
        raise ValueError('search_mm must be in (0, 8] to avoid adjacent 20 mm pitch pins')
    req = Path(req)
    r = json.loads(req.read_text())
    png = Path(r['png'])
    meta_path = png.with_suffix('.json')
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    from calib_core import load_json, cpath
    handeye = cpath('data/wrist_handeye/20261006_111318_clean/result.json')
    Tfc = np.array(meta.get('T_flange_cam', load_json(handeye)['T_flange_cam']), float)
    K = np.asarray(meta.get('K', r.get('K')), float)
    if K.shape != (3, 3):
        raise ValueError('Saved camera K required; do not infer intrinsics from image size')
    raw_path = Path(image) if image else png.with_suffix('.raw.png')
    img = cv2.imread(str(raw_path))
    if img is None:
        raise FileNotFoundError(raw_path)
    Tbb = np.array(r['T_base_basket'], float)
    Tcb = np.linalg.inv(np.array(r['T_base_flange'], float) @ Tfc)
    Tbi = np.linalg.inv(Tbb)
    axis = Tbb[:3, 0]
    axis_y = Tbb[:3, 1]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rl'))
    import basket_env as be
    clip_x = float(be.B['full'].get('clip_x', 0.0))
    from plate_plans import basket_gaps
    numbered_gaps = basket_gaps()
    context = _pin_image_context(img, cfg)
    pins = []
    pin_tops_base = []
    if layout is None:
        pin_geometry = [(p0, p1, None) for p0, p1 in basket_pins_in_base(Tbb)]
    else:
        measured = json.loads(Path(layout).read_text())
        pin_geometry = []
        for entry in measured['pins']:
            if entry.get('review') != 'checked' or entry.get('row') not in (
                    'pair_A_row_plus', 'pair_A_row_minus', 'pair_B_row_plus', 'pair_B_row_minus'):
                raise ValueError('Measured layout requires checked pins with one of four named rows')
            endpoints = [np.asarray(entry[k], float) for k in ('base_basket_xyz', 'top_basket_xyz')]
            if any(v.shape != (3,) or not np.isfinite(v).all() for v in endpoints):
                raise ValueError('Measured pin endpoints must be finite basket-frame xyz')
            p0, p1 = [(Tbb @ np.r_[v, 1.])[:3] for v in endpoints]
            pin_geometry.append((p0, p1, entry['row']))
    for i, (p0, p1, row_name) in enumerate(pin_geometry):
        local = (Tbi @ np.r_[p0, 1])[:3]
        if layout is None and model_search_2d:
            # The basket model has a measured across-row residual in saved wrist views.
            # Keep along-row search below half pitch, but let the image line correct up to
            # 20 mm across rows (row spacing is much larger); labels are never used to seed it.
            match = _pin_match(img, context, p0, p1, Tcb, K, axis, search_mm,
                               axis_y=axis_y, search_radius_px=70, search_y_mm=20.0)
        else:
            match = _pin_match(img, context, p0, p1, Tcb, K, axis, search_mm)
        pins.append(dict(model_id=i, local_xyz=local.tolist(), row=row_name, **match))
        pin_tops_base.append(np.asarray(p1, float))
    groups = []
    if layout is None:
        # Model pins share exactly the same x on the two rows of each pair.
        rows = sorted(set(round(p['local_xyz'][1], 5) for p in pins))
        pair_ids = []
        for row0, row1 in zip(rows[::2], rows[1::2]):
            a = {round(p['local_xyz'][0], 5): p for p in pins if round(p['local_xyz'][1], 5) == row0}
            b = {round(p['local_xyz'][0], 5): p for p in pins if round(p['local_xyz'][1], 5) == row1}
            pair_ids.extend((a[x], b[x]) for x in sorted(a.keys() & b.keys()))
    else:
        # Measured pins can have slightly different x in the two physical rows.
        pair_ids = []
        for pair_name in ('A', 'B'):
            a = sorted((p for p in pins if p['row'] == f'pair_{pair_name}_row_minus'),
                       key=lambda p: p['local_xyz'][0])
            b = sorted((p for p in pins if p['row'] == f'pair_{pair_name}_row_plus'),
                       key=lambda p: p['local_xyz'][0])
            unused = set(range(len(b)))
            for p in a:
                if not unused:
                    break
                j = min(unused, key=lambda k: abs(b[k]['local_xyz'][0] - p['local_xyz'][0]))
                if abs(b[j]['local_xyz'][0] - p['local_xyz'][0]) <= 0.008:
                    unused.remove(j)
                    pair_ids.append((p, b[j]))
    for station_id, (p, q) in enumerate(pair_ids):
        x = round((p['local_xyz'][0] + q['local_xyz'][0]) / 2, 5)
        row0, row1 = sorted((p['local_xyz'][1], q['local_xyz'][1]))
        if 'model_uv' not in p or 'model_uv' not in q:
            continue
        yc = (row0+row1)/2
        # Same basket-frame convention as plate_plans.basket_gaps: +y is left.
        quadrant = _pin_quadrant(x, yc, clip_x)
        confirmed = p['status'] == q['status'] == 'detected'
        corrections = [np.asarray(p.get('correction_xy_mm', [p['correction_mm'], 0.0]), float),
                       np.asarray(q.get('correction_xy_mm', [q['correction_mm'], 0.0]), float)]
        pair_offset = _pin_pair_offset(*corrections) if confirmed else None
        if confirmed and not pair_offset['consistent']:
            confirmed = False
            status = 'pair_disagreement'
        else:
            status = 'detected' if confirmed else 'partial_or_occluded'
        correction_xy_mm = pair_offset['candidate_xy_mm'] if confirmed else None
        dx_mm = float(correction_xy_mm[0]) if confirmed else None
        # The observed correction is along basket +x only. Rejected pairs have no
        # usable target; do not silently return an uncorrected model point.
        groups.append(dict(quadrant=quadrant, station_id=station_id,
                           model_pin_ids=[p['model_id'], q['model_id']],
                           midpoint_uv=np.mean([p['model_uv'][1], q['model_uv'][1]], 0).tolist(),
                           status=status, correction_mm=dx_mm,
                           correction_xy_mm=correction_xy_mm.tolist() if confirmed else None,
                           candidate_correction_xy_mm=(pair_offset['candidate_xy_mm'].tolist()
                                                       if pair_offset is not None else None),
                           correction_disagreement_xy_mm=(pair_offset['disagreement_xy_mm'].tolist()
                                                         if pair_offset is not None else None),
                           local_x_m=x, row_y_m=[row0, row1]))
    for quadrant in ('Q1', 'Q2', 'Q3', 'Q4'):
        ordered_stations = sorted((g for g in groups if g['quadrant'] == quadrant),
                                  key=lambda g: float(g['local_x_m']), reverse=True)
        for tine_number, station in enumerate(ordered_stations, 1):
            station['tine_pair_number'] = tine_number
    gap_groups = _gap_groups_from_tine_pairs(Tbb, groups, numbered_gaps)
    # Display rotation is explicit provenance; no calibration transform changes.
    if rotate == 'auto':
        vectors = [np.array(pins[g['model_pin_ids'][1]]['model_uv'][1])-pins[g['model_pin_ids'][0]]['model_uv'][1] for g in groups]
        rotate = '90' if vectors and np.median([abs(v[1])-abs(v[0]) for v in vectors]) > 0 else '0'
    rotation = int(rotate)
    h, w = img.shape[:2]
    def display(uv):
        u, v = uv
        return np.array([u, v]) if rotation == 0 else (np.array([h-1-v, u]) if rotation == 90 else np.array([v, w-1-u]))
    view = img.copy() if rotation == 0 else cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE if rotation == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE)
    for g in groups:
        u, v = display(g['midpoint_uv'])
        g['visible_in_image'] = bool(0 <= u < view.shape[1] and 0 <= v < view.shape[0])
    for g in gap_groups:
        u, v = display(g['midpoint_uv'])
        g['visible_in_image'] = bool(0 <= u < view.shape[1] and 0 <= v < view.shape[0])
    for quadrant in ('Q1', 'Q2', 'Q3', 'Q4'):
        visible = [g for g in groups if g['quadrant'] == quadrant and g['visible_in_image']]
        for g in sorted(visible, key=lambda g: tuple(display(g['midpoint_uv'])[::-1])):
            colour = (30, 210, 40) if g['status'] == 'detected' else (0, 140, 255)
            points = []
            for pid in g['model_pin_ids']:
                pin = pins[pid]
                uv = pin['detected_uv'] if pin['status'] == 'detected' else pin['model_uv']
                a, b = [tuple(np.rint(display(x)).astype(int)) for x in uv]
                cv2.line(view, a, b, colour, 2 if pin['status'] == 'detected' else 1)
                cv2.circle(view, b, 4, colour, 1)
                points.append(b)
            midpoint = np.rint(np.mean(points, 0)).astype(int)
            label = f"{quadrant}/T{g['tine_pair_number']} " + (f"{g['correction_mm']:+.1f}mm" if g['status'] == 'detected' else '?')
            cv2.putText(view, label, tuple(midpoint+np.array([5, -5])), cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)
    for g in gap_groups:
        u, v = display(g['midpoint_uv'])
        if not g['visible_in_image']:
            continue
        colour = (30, 210, 40) if g['status'] == 'detected' else (0, 190, 255)
        marker = tuple(np.rint([u, v]).astype(int))
        cv2.circle(view, marker, 5, colour, 1)
        label = f"{g['quadrant']}/G{g['number'] if g['number'] is not None else '?'}"
        cv2.putText(view, label, tuple(np.rint([u, v]).astype(int)+np.array([5, 12])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)
    for g in gap_groups:
        if g['number'] is None or not g['visible_in_image']:
            g['gap_centre_base_xyz'] = None
            g['candidate_gap_centre_base_xyz'] = None
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(out/'pairs.png'), view)
    result = dict(req=str(req.resolve()), image=str(raw_path.resolve()), handeye=str(handeye),
                  display_rotation_deg=rotation, numbering='quadrant, top-to-bottom in displayed image; hidden IDs retained',
                  quadrant_basis='basket frame: +x far, +y left; Q1 far/right, Q2 far/left, Q3 near/left, Q4 near/right',
                  correction_axis=('basket +x,+y; 2D local line fit, diagnostic only'
                                   if layout is None and model_search_2d else
                                   'basket +x; diagnostic only'),
                  model_search_2d=bool(layout is None and model_search_2d),
                  continuity_gap_bridge_samples=2,
                  search_mm=search_mm,
                  search_y_mm=20.0 if layout is None and model_search_2d else None,
                  gap_centre_definition=('midpoint model basket_gaps floor center, shifted by mean candidate correction from four adjacent tine lines; diagnostic only'),
                  visually_detected_pairs=sum(g['status'] == 'detected' and g['visible_in_image'] for g in groups),
                  confirmed_pairs=sum(g['status'] == 'detected' and g['number'] is not None
                                      and g['visible_in_image'] for g in gap_groups),
                  confirmed_tine_pairs=sum(g['status'] == 'detected' and g['visible_in_image'] for g in groups),
                  confirmed_gaps=sum(g['status'] == 'detected' and g['number'] is not None
                                     and g['visible_in_image'] for g in gap_groups),
                  visible_model_pairs=sum(g['visible_in_image'] and g['number'] is not None
                                          for g in gap_groups) if layout is None else None,
                  visible_layout_pairs=sum(g['visible_in_image'] and g['number'] is not None
                                           for g in gap_groups),
                  visible_pairs=sum(g['visible_in_image'] for g in groups),
                  visible_tine_pairs=sum(g['visible_in_image'] for g in groups),
                  visible_gaps=sum(g['visible_in_image'] and g['number_candidate'] is not None
                                   for g in gap_groups),
                  pins=pins, pairs=groups, gaps=gap_groups,
                  layout_source=str(Path(layout).resolve()) if layout else 'parametric_model',
                  pair_semantics='pairs are same-x cross-row tine stations; they are not numbered insertion gaps',
                  number_assignment='gaps are formed between adjacent x-ordered tine stations, then matched to the nearest global model gap; partial visibility does not restart numbering; competing groups for one slot stay unnumbered',
                  accuracy_note=('Measured layout comes from manually labelled images and assumed height planes; '
                                 'using its source image for detection is in-sample and not an independent accuracy test.'
                                 if layout else 'No measured 3D ground truth; mm offsets are model-to-image fits, not absolute accuracy.'))
    (out/'pairs.json').write_text(json.dumps(result, indent=2))
    return result


def pins_mode(a, cfg):
    if a.image and not a.req:
        raise ValueError('--image requires --req; one image cannot stand in for a batch')
    reqs = [Path(a.req)] if a.req else sorted((DATA/'real_pipeline').glob('2026101[23]_*/wrist_*.req.json'))
    out = a.out or DATA/'basket'/f'pins_{time.strftime("%Y%m%d_%H%M%S")}'
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    rows = []
    for req in reqs:
        try:
            r = wrist_pins(req, cfg, out/f'{req.parent.name}_{req.stem}', a.image, a.rotate,
                           a.search_mm, a.layout, a.model_search_2d)
            offsets = [g['correction_mm'] for g in r['gaps'] if g['status'] == 'detected' and g['number'] is not None]
            rows.append(dict(photo=str(req), detected_tine_pairs=r['confirmed_tine_pairs'],
                             visible_tine_pairs=r['visible_tine_pairs'], detected_gaps=r['confirmed_gaps'],
                             visible_gaps=r['visible_gaps'],
                             median_gap_dx_mm=float(np.median(offsets)) if offsets else None,
                             display_rotation=r['display_rotation_deg']))
        except (FileNotFoundError, ValueError) as e:
            rows.append(dict(photo=str(req), error=str(e)))
    import csv
    columns = ['photo', 'detected_tine_pairs', 'visible_tine_pairs', 'detected_gaps', 'visible_gaps',
               'median_gap_dx_mm', 'display_rotation', 'error']
    with (out/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps(dict(out=str(out), photos=len(rows), detected_gaps=sum(r.get('detected_gaps', 0) for r in rows)), ensure_ascii=False))


# ---------------------------------------------------------------- read-only marker-size repeatability experiment

def basket_stats(a, cfg):
    """Repeat fixed-camera measurements without changing saved calibration or marker layout.
    The size argument must match the physical black-square edge; hypothetical small sizes on a large print are invalid."""
    import copy
    import csv
    from calib_core import cpath, latest_t_base_cam
    cfg = copy.deepcopy(cfg)
    if a.marker_size_m is not None:
        if not 0.01 <= a.marker_size_m <= 0.15:
            raise ValueError('marker_size_m must be measured black-square edge, 0.01..0.15 m')
        cfg['basket']['marker_size_m'] = a.marker_size_m
    if not 2 <= a.repeats <= 100 or not 3 <= a.frames <= 120:
        raise ValueError('Use 2..100 repeats and 3..120 frames per measurement')
    out = a.out or DATA/'basket'/f'marker_stats_{time.strftime("%Y%m%d_%H%M%S")}'
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    wanted = set(cfg['basket']['marker_ids'].values())
    layout = json.loads(cpath(cfg['basket']['layout']).read_text())
    samples, repeat_positions, fits, frame_ids = {}, {}, [], []
    for repeat in range(a.repeats):
        P, seen, ids = basket_measure(cfg, a.frames, return_samples=True)
        frame_ids.extend(ids)
        for i, points in seen.items():
            samples.setdefault(i, []).extend(points)
        for i, position in P.items():
            repeat_positions.setdefault(i, []).append(position.tolist())
        try:
            T, residual, used = basket_fit_robust(P, layout)
            fits.append(dict(repeat=repeat, residual_mm=float(residual*1000), used_ids=used, T_base_basket=T.tolist()))
        except RuntimeError as e:
            fits.append(dict(repeat=repeat, error=str(e)))
        print(f'[basket_stats] {repeat+1}/{a.repeats}: seen {sorted(P)}', flush=True)
    rows = []
    for i in sorted(wanted):
        xyz = np.asarray(samples.get(i, []), float)
        repeats = np.asarray(repeat_positions.get(i, []), float)
        std = xyz.std(0, ddof=1)*1000 if len(xyz) > 1 else [None]*3
        median_std = repeats.std(0, ddof=1)*1000 if len(repeats) > 1 else [None]*3
        rows.append(dict(marker_id=i, detections=len(xyz), detection_rate=len(xyz)/len(frame_ids),
                         std_x_mm=std[0], std_y_mm=std[1], std_z_mm=std[2],
                         repeat_std_x_mm=median_std[0], repeat_std_y_mm=median_std[1], repeat_std_z_mm=median_std[2]))
    residuals = [r['residual_mm'] for r in fits if 'residual_mm' in r]
    result = dict(marker_size_m=cfg['basket']['marker_size_m'], fixed_serial=cfg['cameras']['fixed_serial'],
                  calibration=str(latest_t_base_cam()), layout=str(cpath(cfg['basket']['layout'])),
                  repeats=a.repeats, frames_per_repeat=a.frames, total_frames=len(frame_ids),
                  all_four_rate=float(np.mean([wanted.issubset(ids) for ids in frame_ids])),
                  fit_success_rate=len(residuals)/a.repeats, residual_mean_mm=float(np.mean(residuals)) if residuals else None,
                  residual_max_mm=max(residuals) if residuals else None, markers=rows, fits=fits,
                  samples={str(k): np.asarray(v).tolist() for k, v in samples.items()},
                  frame_detected_ids=frame_ids, accuracy_note='Repeatability only; fixed bias and absolute accuracy need an external reference.')
    (out/'summary.json').write_text(json.dumps(result, indent=2))
    with (out/'markers.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k: result[k] for k in ('marker_size_m', 'total_frames', 'all_four_rate', 'residual_mean_mm', 'residual_max_mm')}, indent=2))
    return result


def _red_plate_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    red = (((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 170))
           & (hsv[:, :, 1] >= 85) & (hsv[:, :, 2] >= 35)).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(red)
    if count < 2:
        raise ValueError('No red plate component in RGB image')
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < 200:
        raise ValueError('Red plate component has fewer than 200 pixels')
    return components == largest


def plate_plane(rgb, depth, K, T_base_cam, T_base_basket, depth_range_m=(0.20, 0.45),
                surface_tolerance_m=0.005):
    """Diagnostic normal of a red plate from colour-aligned metric depth.

    All transforms must describe the *same captured frame*. No robot command or
    automatic wrist correction is made here. The normal sign is chosen toward
    basket +x, the intended plate-plane normal for a vertical insertion.
    """
    rgb, depth = np.asarray(rgb), np.asarray(depth, float)
    K, Tbc, Tbb = (np.asarray(v, float) for v in (K, T_base_cam, T_base_basket))
    if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
        raise ValueError('RGB and aligned metric depth must have matching dimensions')
    if K.shape != (3, 3) or Tbc.shape != (4, 4) or Tbb.shape != (4, 4):
        raise ValueError('K must be 3x3 and poses must be 4x4')
    if not all(np.isfinite(v).all() for v in (K, Tbc, Tbb, depth)):
        raise ValueError('Inputs must be finite; invalid depth must be zero')
    if min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError('Camera focal lengths must be positive')
    near, far = map(float, depth_range_m)
    if not (0 < near < far <= 3.0):
        raise ValueError('Depth range must satisfy 0 < near < far <= 3 m')
    if not (0.001 <= surface_tolerance_m <= 0.02):
        raise ValueError('Surface tolerance must be between 1 and 20 mm')
    mask = _red_plate_mask(rgb)
    valid = mask & (depth >= near) & (depth <= far)
    if valid.sum() < 200 or valid.sum() / mask.sum() < 0.20:
        raise ValueError(f'Insufficient red-plate depth in {near:.2f}–{far:.2f} m range')
    v, u = np.nonzero(valid)
    z = depth[valid]
    points = np.column_stack(((u - K[0, 2]) * z / K[0, 0],
                              (v - K[1, 2]) * z / K[1, 1], z))
    rng = np.random.default_rng(20261014)
    sample = points[rng.choice(len(points), min(len(points), 6000), replace=False)]
    best = np.zeros(len(sample), bool)
    threshold = float(surface_tolerance_m)  # curved dishes need a declared larger approximation tolerance
    for _ in range(300):
        tri = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal /= norm
        inliers = np.abs((sample - tri[0]) @ normal) <= threshold
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 200 or best.mean() < 0.50:
        raise ValueError('No dominant red-plate plane; depth may be from background')
    centre = sample[best].mean(axis=0)
    _, singular, vt = np.linalg.svd(sample[best] - centre, full_matrices=False)
    if singular[1] < 0.02:
        raise ValueError('Plane points span too narrow an area to determine orientation')
    normal_cam = vt[-1]
    normal_base = Tbc[:3, :3] @ normal_cam
    normal_basket = Tbb[:3, :3].T @ normal_base
    if normal_basket[0] < 0:
        normal_cam, normal_base, normal_basket = -normal_cam, -normal_base, -normal_basket
    normal_basket /= np.linalg.norm(normal_basket)
    residual = np.abs((points - centre) @ normal_cam)
    # A nearly horizontal plate has no observable in-plane yaw from its normal.
    yaw_observable = bool(np.hypot(normal_basket[0], normal_basket[1]) >= 0.1)
    return dict(yaw_error_deg=(float(np.degrees(np.arctan2(normal_basket[1], normal_basket[0])))
                               if yaw_observable else None),
                yaw_observable=yaw_observable,
                tilt_error_deg=float(np.degrees(np.arcsin(np.clip(normal_basket[2], -1, 1)))),
                tilt_from_horizontal_deg=float(np.degrees(np.arccos(np.clip(abs(normal_base[2]), 0, 1)))),
                normal_basket=normal_basket.tolist(), normal_base=normal_base.tolist(),
                normal_camera=normal_cam.tolist(), red_pixels=int(mask.sum()),
                valid_red_depth_pixels=int(valid.sum()), valid_red_depth_fraction=float(valid.sum() / mask.sum()),
                inlier_fraction=float(np.mean(residual <= threshold)),
                inlier_rmse_mm=float(1000 * np.sqrt(np.mean(residual[residual <= threshold] ** 2))),
                median_depth_m=float(np.median(z)), depth_range_m=[near, far],
                surface_tolerance_m=threshold,
                note='Saved RGB-D and synchronized poses only; diagnostic, not a wrist motion command')


def slot_targets(out, slide_path=None, tines_path=None, audit_path=None):
    """Build a stage-3 pin-number to basket-frame gap table from reviewed labels and 3D tine rows.

    The output path is exclusive: choose a new path for every run so previous results are never overwritten.
    This offline file operation does not access a camera, robot, simulator, or GPU.
    """
    import hashlib

    root = Path(__file__).resolve().parents[1]
    slide_path = Path(slide_path or root/'calibration/data/basket/boss_slide_20261014/tines_refined_codex.json')
    tines_path = Path(tines_path or root/'calibration/data/basket/wrist_look/tines_3d_claude_20261014.json')
    audit_path = Path(audit_path or root/'calibration/data/basket/realQ_verified_3d_model_audit_20261014_v5/verified_pin_3d_model_comparison.json')
    out = Path(out)
    if not out.is_absolute():
        out = root/out
    slide, measured, audit = (json.loads(p.read_text()) for p in (slide_path, tines_path, audit_path))
    # plate_plans is the single source for model gap ordering and pin_gap numbering.
    from plate_plans import basket_gaps, pin_gap
    import numpy as np

    gaps = basket_gaps()
    counts = {q: len(v) for q, v in gaps.items()}
    expected_tines = {q: count + 1 for q, count in counts.items()}
    slide_checks = {}
    labels = slide.get('labels', [])
    for q, n_tines in expected_tines.items():
        rows = {}
        for label in labels:
            if label.get('quadrant') == q:
                rows.setdefault(label.get('row'), []).append(int(label['num']))
        row_checks = {str(row): sorted(nums) == list(range(1, n_tines + 1))
                      and len(nums) == n_tines and len(set(nums)) == n_tines
                      for row, nums in rows.items()}
        slide_checks[q] = {
            'model_slots': counts[q], 'model_tines_per_row': n_tines,
            'slide_numbering_by_row': {str(row): sorted(nums) for row, nums in sorted(rows.items())},
            'both_rows_match_model_sequence': len(row_checks) == 2 and all(row_checks.values()),
        }
        if not slide_checks[q]['both_rows_match_model_sequence']:
            raise ValueError(f'{q} boss-slide labels do not provide two complete numbered rows: {row_checks}')

    # Basket +x points away from the robot; the measured reference yaw is +0.1 degree.
    angle = np.radians(0.1)
    Tbb = np.eye(4)
    Tbb[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    row_points = measured.get('quadrants', {})
    targets = []
    for q in ('Q1', 'Q2', 'Q3', 'Q4'):
        for number in range(1, counts[q] + 1):
            model_index = int(pin_gap(Tbb, q, number))
            model_gap = np.asarray(gaps[q][model_index], dtype=float)
            used_rows, measured_centres = [], []
            extrapolated = False
            for row_name, pins in sorted(row_points.get(q, {}).items()):
                by_number = {int(pin['num']): np.asarray(pin['base'][:2], dtype=float) for pin in pins}
                nums = np.asarray(sorted(by_number), dtype=float)
                coords = np.asarray([by_number[int(n)] for n in nums], dtype=float)
                if len(nums) < 2:
                    continue
                fit_x, fit_y = np.polyfit(nums, coords[:, 0], 1), np.polyfit(nums, coords[:, 1], 1)
                pair, estimated = [], []
                for tine_number in (number, number + 1):
                    if tine_number in by_number:
                        point = by_number[tine_number]
                    else:
                        point = np.asarray([np.polyval(fit_x, tine_number), np.polyval(fit_y, tine_number)])
                        estimated.append(tine_number)
                    pair.append(point)
                centre = np.mean(pair, axis=0)
                measured_centres.append(centre)
                extrapolated |= bool(estimated)
                used_rows.append({
                    'row': row_name, 'estimated_tines': estimated,
                    'tine_base_xy_m': {str(n): [round(float(v), 6) for v in point]
                                       for n, point in zip((number, number + 1), pair)},
                    # Match numpy's decimal rounding used by the reviewed source table.
                    'row_gap_centre_xy_m': [float(np.round(v, 6)) for v in centre],
                })
            if not row_points.get(q):
                target = model_gap.copy()
                status, evidence, delta, error = 'model_only_no_3d', 'rl_basket_gaps_model', None, None
            else:
                if len(measured_centres) != 2:
                    raise ValueError(f'{q} pin {number}: expected two 3D tine rows, got {len(measured_centres)}')
                xy = np.mean(measured_centres, axis=0)
                target = np.r_[xy, model_gap[2]]
                status = 'measured_with_linear_edge_extrapolation' if extrapolated else 'measured'
                evidence = '3d_measured_base_midpoint'
                delta = [round(float(v), 2) for v in (target[:2] - model_gap[:2]) * 1000]
                error = round(float(np.linalg.norm(target[:2] - model_gap[:2]) * 1000), 2)
            targets.append({
                'quadrant': q, 'pin_number': number,
                'tines_bracketing_gap': [number, number + 1],
                'model_gap_index_robot_to_far': model_index,
                'gap_centre_basket_m': [round(float(v), 6) for v in target],
                'evidence': evidence, 'status': status, 'rows_used': used_rows,
                'model_gap_centre_basket_m': [round(float(v), 6) for v in model_gap],
                'measured_minus_model_mm': delta, 'measured_model_xy_error_mm': error,
            })

    assumptions = audit.get('assumptions', {})
    def source_name(path):
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)
    result = {
        'title': 'Quadrant/pin-number slot target map for basket insertion stage 3',
        'units': 'm in basket frame; x away from robot, y left, z up',
        'layout': {'basket_xy_m': [0.807, -0.192], 'dish_xy_m': [0.848, 0.339],
                   'basket_yaw_deg': 0.1, 'basket_x_away_from_robot': True},
        'numbering_rule': ('Pin N target lies between tines N and N+1; N=1 is farthest from robot. '
                           'pin_gap(Tbb, model_qd, N) selects robot-to-far model gap index.'),
        'sources': {
            'boss_slide_labels': source_name(slide_path),
            'boss_slide_sha256': hashlib.sha256(slide_path.read_bytes()).hexdigest(),
            '3d_tine_result': source_name(tines_path),
            '3d_tine_sha256': hashlib.sha256(tines_path.read_bytes()).hexdigest(),
            'verified_3d_audit': source_name(audit_path),
            'verified_3d_audit_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            'planner': 'calibration/plate_plans.py basket_gaps/pin_gap',
            'planner_data': 'rl/basket.yaml + rl/basket_env.py',
            'basket_yaml': {'path': 'rl/basket.yaml', 'sha256': hashlib.sha256((root/'rl/basket.yaml').read_bytes()).hexdigest()},
            'basket_geometry': {'path': 'rl/basket_env.py', 'sha256': hashlib.sha256((root/'rl/basket_env.py').read_bytes()).hexdigest()},
            'pin_gap_implementation': {'path': 'calibration/plate_plans.py', 'sha256': hashlib.sha256(Path(__file__).with_name('plate_plans.py').read_bytes()).hexdigest()},
        },
        'slide_numbering_checks': slide_checks,
        'model_slot_counts': counts,
        'targets': targets,
        'limitations': [
            'Q1 has no 3D measured tine coordinates; model coordinates only.',
            'Q2/Q3/Q4 3D backprojection depends on the saved wrist-camera calibration and assumed height planes.',
            'Missing edge tines are linearly extrapolated from that row and explicitly marked; check rows_used.',
            'Measured targets average base/floor XY midpoints of tines N and N+1 across the two rail rows; geometric target only, not collision or placement success.',
        ],
        '3d_backprojection_assumptions': assumptions,
        '3d_label_counts': audit.get('counts', {}),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'out': str(out), 'target_count': len(targets), 'quadrants': counts,
                      'all_slide_sequences_valid': all(x['both_rows_match_model_sequence'] for x in slide_checks.values())}))
    return result


def tine_number(pins, p0, cx):
    """보스 numbering of the tine at base p0 in its row: 1 = farthest from the robot in every quadrant (far rows Q1/Q2 run
    outward from the centre clip, near rows Q3/Q4 toward the robot).  pins: [(base, tip)] in the basket frame."""
    far = bool(p0[0] > cx)
    row = sorted((abs(q0[0] - cx) for q0, _ in pins if (q0[0] > cx) == far and abs(q0[1] - p0[1]) < 1e-6), reverse=far)
    return row.index(abs(p0[0] - cx)) + 1


def model_overlay(req, out=None):
    """Saved wrist photo + current basket tines; numbers follow the boss's far-to-near rule."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rl'))
    import basket_env as be
    req = Path(req)
    m = json.loads(req.read_text())
    img = cv2.imread(str(Path(m['png']).with_suffix('.raw.png') if 'png' in m else req.with_suffix('.raw.png')))
    if img is None:
        raise FileNotFoundError(f'raw photo for {req}')
    meta = json.loads(Path(m['png']).with_suffix('.json').read_text()) if 'png' in m and Path(m['png']).with_suffix('.json').exists() else m
    Tb = np.asarray(meta['T_base_basket'], float)
    Tcb = np.linalg.inv(np.asarray(meta['T_base_flange'], float) @ np.asarray(meta['T_flange_cam'], float))
    K = np.asarray(meta['K'], float)
    f = be.B['full']
    cx, r = f.get('clip_x', 0.0), f['pin_r']
    pins = [(np.asarray(p0), np.asarray(p1)) for k, p0, p1, *rest in be.full_basket_parts() if k == 'cyl' and rest and abs(rest[0] - r) < 1e-9]
    colours = {'Q1': (0, 255, 0), 'Q2': (255, 0, 255), 'Q3': (0, 200, 255), 'Q4': (255, 128, 0)}
    for p0, p1 in pins:
        far, left = p0[0] > cx, p0[1] > 0
        qd = {(True, False): 'Q1', (True, True): 'Q2', (False, True): 'Q3', (False, False): 'Q4'}[(far, left)]
        a, b = (Tcb @ Tb @ np.r_[p0, 1])[:3], (Tcb @ Tb @ np.r_[p1, 1])[:3]
        if a[2] < 0.05 or b[2] < 0.05:
            continue
        ua, ub = K @ (a / a[2]), K @ (b / b[2])
        cv2.line(img, (int(ua[0]), int(ua[1])), (int(ub[0]), int(ub[1])), colours[qd], 2)
        cv2.putText(img, str(tine_number(pins, p0, cx)), (int(ua[0]) + 3, int(ua[1]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colours[qd], 1)
    out = Path(out) if out else req.with_name(req.stem.replace('.req', '') + '_model.png')
    cv2.imwrite(str(out), img)
    print(f'model overlay -> {out}', flush=True)
    return out


def plate_plane_mode(a):
    if not all((a.rgb, a.depth, a.meta, a.out)):
        raise ValueError('plate_plane requires --rgb, --depth, --meta and --out')
    meta = json.loads(Path(a.meta).read_text())
    if 'T_base_cam' in meta:
        Tbc = np.asarray(meta['T_base_cam'], float)
    else:
        Tbc = np.asarray(meta['T_base_flange'], float) @ np.asarray(meta['T_flange_cam'], float)
    rgb = cv2.imread(str(a.rgb))
    if rgb is None:
        raise FileNotFoundError(a.rgb)
    depth = np.load(a.depth)
    result = plate_plane(rgb, depth, meta['K'], Tbc, meta['T_base_basket'],
                         (a.depth_min_m, a.depth_max_m), a.surface_tolerance_mm / 1000)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    view = rgb.copy()
    view[_red_plate_mask(rgb)] = (0.55 * view[_red_plate_mask(rgb)] +
                                  0.45 * np.array([220, 20, 220])).astype(np.uint8)
    yaw_text = f"{result['yaw_error_deg']:+.2f}" if result['yaw_observable'] else 'unobservable'
    cv2.putText(view, f"yaw {yaw_text}  tilt {result['tilt_error_deg']:+.2f} deg",
                (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out / 'plate_plane.png'), view)
    (out / 'plate_plane.json').write_text(json.dumps(result, indent=2))
    return result


def plate_angle(req, out, radius_m=0.134):
    """Estimate disk-plane yaw from one saved RGB silhouette, with the planned perpendicular orientation as prior.

    The inverse projection has a second yaw branch. This diagnostic only fits ±40° around basket +y
    and reports sensitivity to colour threshold; it must not directly command a wrist joint.
    """
    from scipy.optimize import least_squares
    req, out = Path(req), Path(out)
    r = json.loads(req.read_text())
    png = Path(r['png'])
    meta = json.loads(png.with_suffix('.json').read_text())
    raw = png.with_name(png.stem + '.raw.png')
    image = cv2.imread(str(raw))
    if image is None:
        raise FileNotFoundError(raw)
    K = np.asarray(meta['K'], float)
    Tbc = np.asarray(meta['T_base_flange'], float) @ np.asarray(meta['T_flange_cam'], float)
    Tbb = np.asarray(meta['T_base_basket'], float)
    R = Tbc[:3, :3].T @ Tbb[:3, :3]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    phase = np.linspace(0, 2*np.pi, 128, endpoint=False)
    cos_phase, sin_phase = np.cos(phase), np.sin(phase)

    def projection(state):
        cx, cy, cz, yaw = state
        tangent = R @ np.array([np.cos(yaw), np.sin(yaw), 0.0])
        up = R @ np.array([0.0, 0.0, 1.0])
        points = np.array([cx, cy, cz])[None] + radius_m * (
            cos_phase[:, None] * tangent + sin_phase[:, None] * up)
        if np.min(points[:, 2]) <= 0.05:
            raise ValueError('Projected plate behind camera')
        uvw = points @ K.T
        return uvw[:, :2] / uvw[:, 2:3]

    estimates = []
    for sat in (65, 80, 95):
        red = (((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165)) &
               (hsv[:, :, 1] > sat) & (hsv[:, :, 2] > 45)).astype(np.uint8)
        _, components, stats, _ = cv2.connectedComponentsWithStats(red)
        if len(stats) < 2 or int(stats[1:, 4].max()) < 10000:
            raise ValueError('No large red plate silhouette in saved image')
        component = 1 + int(np.argmax(stats[1:, 4]))
        contours, _ = cv2.findContours((components == component).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        observed = cv2.fitEllipse(max(contours, key=cv2.contourArea))
        center = np.asarray(observed[0], float)
        sizes = np.asarray(observed[1], float)
        minor_dir = np.array([np.cos(np.radians(observed[2])), np.sin(np.radians(observed[2]))])

        def ellipse_properties(state):
            xy = projection(state)
            mean = xy.mean(axis=0)
            eigenvalues, eigenvectors = np.linalg.eigh(np.cov(xy.T))
            return mean, 2*np.sqrt(2*eigenvalues), eigenvectors[:, 0], xy

        def residual(state):
            mean, lengths, direction, _ = ellipse_properties(state)
            cross = direction[0]*minor_dir[1] - direction[1]*minor_dir[0]
            return np.r_[(mean-center)/15.0, (lengths-sizes)/15.0, cross*3.0]

        # Circle plane should be perpendicular to basket x; a farther yaw branch is underdetermined
        # by one RGB view and is deliberately excluded rather than presented as a unique solution.
        depth = max(0.2, min(0.6, 2*radius_m*K[0, 0]/sizes[1]))
        start = [(center[0]-K[0, 2])*depth/K[0, 0],
                 (center[1]-K[1, 2])*depth/K[1, 1], depth, np.pi/2]
        fit = least_squares(residual, start,
                            bounds=([-0.4, -0.4, 0.17, np.radians(50)],
                                    [0.4, 0.4, 0.8, np.radians(130)]), max_nfev=250)
        mean, lengths, _, poly = ellipse_properties(fit.x)
        estimates.append(dict(saturation_threshold=sat, plate_plane_yaw_basket_deg=float(np.degrees(fit.x[3])),
                              perpendicular_error_deg=float(np.degrees(fit.x[3])-90.0),
                              fit_residual_norm=float(np.linalg.norm(residual(fit.x))),
                              observed_ellipse=dict(center_px=center.tolist(), axes_px=sizes.tolist(),
                                                    minor_axis_deg=float(observed[2])),
                              projected_ellipse=dict(center_px=mean.tolist(), axes_px=lengths.tolist()),
                              projected_outline_px=poly.tolist(), red_component_px=int(stats[component, 4])))
    median = float(np.median([x['perpendicular_error_deg'] for x in estimates]))
    # One ellipse cannot separate yaw from a small out-of-vertical dish tilt.
    # Demonstrate the ambiguity with a joint fit before any controller consumes this diagnostic.
    observed_mid = estimates[1]['observed_ellipse']
    centre_mid = np.asarray(observed_mid['center_px'], float)
    axes_mid = np.asarray(observed_mid['axes_px'], float)
    angle_mid = float(observed_mid['minor_axis_deg'])

    def tilted_outline(state):
        cx, cy, cz, yaw, tilt = state
        tangent = R @ np.array([np.cos(yaw), np.sin(yaw), 0.0])
        normal = R @ np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        up = np.cos(tilt)*(R @ np.array([0.0, 0.0, 1.0])) + np.sin(tilt)*normal
        points = np.array([cx, cy, cz])[None] + radius_m*(
            cos_phase[:, None]*tangent + sin_phase[:, None]*up)
        uvw = points @ K.T
        return (uvw[:, :2]/uvw[:, 2:3]).astype(np.float32)

    def tilted_residual(state):
        fitted = cv2.fitEllipse(tilted_outline(state))
        angle_difference = (fitted[2]-angle_mid+90.0) % 180.0-90.0
        return np.r_[(np.asarray(fitted[0])-centre_mid)/15.0,
                     (np.asarray(fitted[1])-axes_mid)/15.0,
                     angle_difference/15.0]

    upright = estimates[1]
    z_guess = max(0.2, min(0.6, 2*radius_m*K[0, 0]/axes_mid[1]))
    joint_start = [(centre_mid[0]-K[0, 2])*z_guess/K[0, 0],
                   (centre_mid[1]-K[1, 2])*z_guess/K[1, 1], z_guess,
                   np.radians(upright['plate_plane_yaw_basket_deg']), 0.0]
    joint = least_squares(tilted_residual, joint_start,
                          bounds=([-0.4, -0.4, 0.17, np.radians(50), np.radians(-15)],
                                  [0.4, 0.4, 0.8, np.radians(130), np.radians(15)]),
                          max_nfev=300, diff_step=1e-4)
    joint_fit = dict(yaw_error_deg=float(np.degrees(joint.x[3])-90.0),
                     out_of_vertical_tilt_deg=float(np.degrees(joint.x[4])),
                     residual_norm=float(np.linalg.norm(tilted_residual(joint.x))))
    view = image.copy()
    xy = np.rint(np.asarray(estimates[1]['projected_outline_px'])).astype(np.int32)
    cv2.polylines(view, [xy], True, (20, 220, 50), 2)
    cv2.polylines(view, [np.rint(tilted_outline(joint.x)).astype(np.int32)], True, (255, 180, 40), 2)
    cv2.ellipse(view, tuple(np.rint(estimates[1]['observed_ellipse']['center_px']).astype(int)),
                tuple(np.rint(np.asarray(estimates[1]['observed_ellipse']['axes_px'])/2).astype(int)),
                estimates[1]['observed_ellipse']['minor_axis_deg'], 0, 360, (0, 230, 255), 2)
    cv2.putText(view, f'if upright: yaw error {median:+.1f} deg; tilt-fit: {joint_fit["yaw_error_deg"]:+.1f} deg',
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
    result = dict(req=str(req.resolve()), image=str(raw.resolve()), radius_assumption_m=radius_m,
                  method='red connected-component contour; ellipse fit; perspective disk fit around planned 90 deg',
                  perpendicular_error_deg=median, threshold_sensitivity_deg=float(np.ptp(
                      [x['perpendicular_error_deg'] for x in estimates])), fits=estimates,
                  joint_yaw_tilt_fit=joint_fit,
                  conclusion='Yaw is not identifiable from this one RGB silhouette without an independent tilt measurement.',
                  limits='Single RGB view has yaw-branch ambiguity; disk assumed circular and 268 mm wide. '
                         'Gripper occlusion, dish depth and unknown plate tilt add unmeasured bias. Diagnostic only; '
                         'do not command wrist correction from the upright-only estimate.')
    out.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(out/'plate_angle.png'), view)
    (out/'plate_angle.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(out=str(out), upright_only_error_deg=median,
                          joint_yaw_tilt_fit=joint_fit, conclusion=result['conclusion'])))
    return result


def markers_on_planes(P, layout, T_base_cam, basket_z):
    """(10/19 17:3x, 보스: the blocks had not moved) marker centres moved along the camera ray onto their known heights
    (fixed basket z + layout z).  A small oblique 5 cm marker's single-camera depth was off by up to 6 cm (marker 2:
    -54 mm y, -47 mm z) and every fit failed; the ray direction is good: ray/plane fits the layout to 0.8 mm."""
    o = np.asarray(T_base_cam, float)[:3, 3]
    out = {}
    for i, p in P.items():
        if str(i) not in layout:
            out[i] = p
            continue
        z = basket_z + float(layout[str(i)][2])
        d = np.asarray(p, float) - o
        out[i] = o + d * (z - o[2]) / d[2] if abs(d[2]) > 1e-6 else np.asarray(p, float)
    return out


def basket_consistency(P, layout, tol=0.010, pair_tol=0.006):
    """(10/19 보스: generalise the moved-marker-block fix) which basket marker blocks still match the stored layout.
    Pairs whose xy distance matches the layout within pair_tol are consistent; the largest consistent group (>= 2
    markers, never all from one fit) is taken as unmoved and the others as moved.  Returns a dict: verdict ok /
    moved / unsure, group, moved, pair table, the full-fit xy residual and the basket pose from the group."""
    import itertools
    ids = sorted(i for i in P if str(i) in layout)
    xy = lambda v: np.asarray(v, float)[:2]                                          # noqa: E731
    v3 = lambda v: np.asarray(v, float)[:3]                                          # noqa: E731
    # (10/19 17:2x) 3-D pair distances: a knocked-over block moved 47 mm down, its xy distance to one marker barely changed
    pairs = [(a, b, float(np.linalg.norm(v3(P[a]) - v3(P[b]))), float(np.linalg.norm(v3(layout[str(a)]) - v3(layout[str(b)]))))
             for a, b in itertools.combinations(ids, 2)]
    ok_pair = {(a, b) for a, b, dm, dl in pairs if abs(dm - dl) <= pair_tol}
    T, _, _ = basket_fit(P, layout)
    full = max(float(np.linalg.norm(((T @ np.r_[layout[str(i)], 1.0])[:3] - P[i])[:2])) for i in ids)
    out = dict(ids=ids, pairs=pairs, full_xy_res=full, verdict='ok', group=ids, moved=[], T=T)
    if full <= tol:
        return out
    group = []
    for n in range(len(ids) - 1, 1, -1):                # largest group whose every pair agrees with the layout
        cands = [g for g in itertools.combinations(ids, n) if all((a, b) in ok_pair for a, b in itertools.combinations(g, 2))]
        if cands:
            span = lambda g: max(d for a, b, d, _ in pairs if a in g and b in g)   # noqa: E731
            group = list(max(cands, key=span))            # the widest group: best yaw
            break
    if not group:
        out.update(verdict='unsure', group=[], moved=ids)
        return out
    Tg, _, _ = basket_fit({i: P[i] for i in group}, layout)
    out.update(verdict='moved', group=group, moved=[i for i in ids if i not in group], T=Tg)
    return out


def basket_check(cfg, relayout=False):
    """(10/19 보스) pre-demo check of the basket marker blocks against the stored layout; --relayout stores the moved
    blocks where they are now, in the basket frame of the unmoved ones (new file, config switched after a typed y).
    Exit status: 0 ok, 3 moved (relayout possible), 4 unsure / too few markers."""
    import json as _json
    from calib_core import cpath
    b = cfg['basket']
    from calib_core import latest_t_base_cam, load_json
    layout = _json.loads(cpath(b['layout']).read_text())
    Tbc = np.array(load_json(latest_t_base_cam())['T_base_cam'])
    P = markers_on_planes(basket_measure(cfg), layout, Tbc, b.get('fixed_z_m', 0.144))
    names = {v: k for k, v in b['marker_ids'].items()}
    tol = b.get('max_fit_residual_m', 0.010)
    if len([i for i in P if str(i) in layout]) < 2:
        print(f'markers seen {sorted(P)}: too few for the basket (run.sh markers)', flush=True)
        return 4
    r = basket_consistency(P, layout, tol)
    print(f'layout {b["layout"]}', flush=True)
    for a, c, dm, dl in r['pairs']:
        print(f'  {a}-{c}: now {dm * 1000:6.1f} mm, layout {dl * 1000:6.1f} mm, diff {(dm - dl) * 1000:+6.1f}', flush=True)
    missing = sorted(set(names) - set(P))
    if missing:
        print(f'  not seen: {missing}', flush=True)
    if r['verdict'] == 'ok':
        print(f'BASKET OK: markers {r["ids"]} fit {r["full_xy_res"] * 1000:.1f} mm (<= {tol * 1000:.0f})', flush=True)
        return 0
    if r['verdict'] == 'unsure':
        print(f'BASKET UNSURE: fit {r["full_xy_res"] * 1000:.1f} mm and no two markers still match the layout - '
              'several blocks or the basket moved; fix the blocks or re-measure the basket frame', flush=True)
        return 4
    moved = ', '.join(f'{i} ({names.get(i, "?")})' for i in r['moved'])
    print(f'MARKER BLOCK MOVED: {moved}; unmoved {r["group"]} (fit {r["full_xy_res"] * 1000:.1f} mm > {tol * 1000:.0f})', flush=True)
    if len(r['group']) == 2:
        print('  only two unmoved markers: the basket frame from them is weaker (check the next placement)', flush=True)
    Tg = np.array(r['T'])
    Tg[2, 3] = b.get('fixed_z_m', Tg[2, 3])
    Ti = np.linalg.inv(Tg)
    new = {str(i): (Ti @ np.r_[P[i], 1.0])[:3].round(5).tolist() for i in r['ids']}
    for i in r['moved']:
        d = (np.array(new[str(i)]) - np.array(layout[str(i)]))[:2] * 1000
        print(f'  marker {i}: moved {d[0]:+.1f} / {d[1]:+.1f} mm (basket x / y)', flush=True)
    if not relayout:
        print('  -> bash calibration/run.sh basket_relayout   (stores the moved blocks where they are now)', flush=True)
        return 3
    if input('type y to store the new layout and switch config basket.layout to it: ').strip() != 'y':
        print('not stored', flush=True)
        return 3
    out = cpath(b['layout']).parent / f'basket_layout_5cm_{time.strftime("%Y%m%d_%H%M%S")}_relayout.json'
    out.write_text(_json.dumps(dict(new, note=f'basket_check relayout: moved {r["moved"]}, frame from unmoved {r["group"]} '
                                         f'(was {b["layout"]})'), indent=1))
    cfg_path = Path(__file__).resolve().parent / 'config.yaml'
    text = cfg_path.read_text()
    line = next(l for l in text.splitlines() if l.startswith('  layout: '))
    rel = str(out.relative_to(Path(__file__).resolve().parent))
    cfg_path.write_text(text.replace(line, f'  layout: {rel}   # ({time.strftime("%m/%d %H:%M")} basket_relayout: moved '
                                           f'{r["moved"]}, frame from {r["group"]})\n  # was:{line[1:]}', 1))
    print(f'layout -> {out}\nconfig basket.layout switched (old line kept as a comment)', flush=True)
    return 0


def markers_live(cfg):
    """(10/19 보스, pre-demo check) live fixed-camera view of the 4 basket markers with the basket_measure detector:
    each marker green with its base x/y, the top bar OK/-- per marker.  Read-only; q / ESC quits (frees the camera)."""
    import time
    from calib_core import latest_t_base_cam, load_json
    from sensors import ColorCam
    b = cfg['basket']
    names = {v: k for k, v in b['marker_ids'].items()}
    Tbc = np.array(load_json(latest_t_base_cam())['T_base_cam'])
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg['marker']['dictionary']))
    prm = cv2.aruco.DetectorParameters()
    prm.perspectiveRemovePixelPerCell = cfg['marker']['pixels_per_cell']
    up = int(b.get('detect_upsample', 1))
    if up > 1:
        prm.cornerRefinementMethod, prm.minMarkerPerimeterRate = cv2.aruco.CORNER_REFINE_SUBPIX, 0.01
    det = cv2.aruco.ArucoDetector(d, prm)
    s = b['marker_size_m'] / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
    import json as _json
    from calib_core import cpath
    layout = _json.loads(cpath(b['layout']).read_text())
    hist = {}                                               # (10/19 보스) last 2 s of positions per marker: layout check
    cam = ColorCam(cfg['cameras']['fixed_serial'])
    win = 'basket markers (fixed cam) - q quits'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    t0 = time.time()
    try:
        while True:
            fr = cam.read()
            bgr = cv2.cvtColor(fr.rgb, cv2.COLOR_RGB2BGR)
            g = cv2.cvtColor(fr.rgb, cv2.COLOR_RGB2GRAY)
            corners, ids, _ = det.detectMarkers(cv2.resize(g, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up > 1 else g)
            seen = set()
            for c, i in zip(corners, [] if ids is None else ids.ravel()):
                c, i = (c.reshape(4, 2) / up).astype(np.float32), int(i)
                col = (0, 255, 0) if i in names else (0, 165, 255)
                cv2.polylines(bgr, [c.astype(np.int32)], True, col, 2)
                txt = f'{i} {names.get(i, "other")}'
                if i in names:
                    ok, _, tv = cv2.solvePnP(obj, c, fr.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if ok:
                        pb = (Tbc @ np.r_[tv.ravel(), 1.0])[:3]
                        seen.add(i)
                        hist.setdefault(i, []).append((time.time(), pb))
                        txt += f' ({pb[0]:.3f},{pb[1]:+.3f})'
                cv2.putText(bgr, txt, tuple(c[0].astype(int) + [0, -8]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            status = '  '.join(f'{i}:{names[i]} {"OK" if i in seen else "--"}' for i in sorted(names))
            now = time.time()
            P = {i: np.mean([p_ for t_, p_ in v if now - t_ < 2.0], axis=0) for i, v in hist.items()
                 if any(now - t_ < 2.0 for t_, _ in v)}
            hist = {i: [(t_, p_) for t_, p_ in v if now - t_ < 2.0] for i, v in hist.items()}
            if len(seen) < len(names):
                verdict, col = '   MISSING', (0, 0, 255)
            else:                                           # seen is not enough: the 4 must also match the stored layout
                r = basket_consistency(markers_on_planes(P, layout, Tbc, b.get('fixed_z_m', 0.144)), layout,
                                       b.get('max_fit_residual_m', 0.010))
                verdict, col = (('   ALL 4 OK, LAYOUT OK', (0, 200, 0)) if r['verdict'] == 'ok' else
                                (f'   LAYOUT NG {r["full_xy_res"] * 1000:.0f} mm, moved {r["moved"]} (empty basket -> basket_relayout)',
                                 (0, 0, 255)))
            cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(bgr, status + verdict, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.imshow(win, bgr)
            if time.time() - t0 > 2:
                print('markers seen', sorted(seen), verdict.strip(), flush=True)
                t0 = time.time()
            if cv2.waitKey(1) & 0xff in (ord('q'), 27):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=('preview', 'marker', 'markers_live', 'basket_check', 'probe', 'locate', 'basket', 'wrist_look', 'pre_photo', 'pins', 'basket_stats', 'pin_layout', 'plate_angle', 'plate_plane', 'model_overlay', 'slot_targets', 'gap_error'))
    p.add_argument('--relayout', action='store_true', help='basket_check: store moved marker blocks where they are now')
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
    p.add_argument('--out', type=Path, help='pins / basket_stats: new output directory')
    p.add_argument('--labels', type=Path, help='pin_layout: checked pin root/tip pixel labels JSON')
    p.add_argument('--layout', type=Path, help='pins: measured pin layout JSON, offline diagnostic only')
    p.add_argument('--quadrant', choices=('Q1', 'Q2', 'Q3', 'Q4'), help='gap_error: commanded robot quadrant override')
    p.add_argument('--pin', type=int, help='gap_error: commanded pin number override')
    p.add_argument('--gap_search_mm', type=float, default=35.0,
                   help='gap_error: local 2-D projected tip search radius in basket x/y, millimetres')
    p.add_argument('--plate_radius_m', type=float, default=0.134, help='plate_angle: assumed dish outer radius')
    p.add_argument('--rgb', type=Path, help='plate_plane: saved BGR/RGB image readable by OpenCV')
    p.add_argument('--depth', type=Path, help='plate_plane: colour-aligned depth in metres as .npy')
    p.add_argument('--meta', type=Path, help='plate_plane: same-frame K, T_base_basket and T_base_cam or flange+hand-eye')
    p.add_argument('--depth_min_m', type=float, default=.20, help='plate_plane: nearest accepted metric depth')
    p.add_argument('--depth_max_m', type=float, default=.45, help='plate_plane: farthest accepted metric depth')
    p.add_argument('--surface_tolerance_mm', type=float, default=5.0,
                   help='plate_plane: allowed distance to fitted plane; declare larger tolerance for curved dish')
    p.add_argument('--rotate', choices=('auto', '0', '90', '270'), default='auto', help='pins: display rotation, raw projection unchanged')
    p.add_argument('--search_mm', type=float, default=8.0, help='pins: maximum local correction along basket x')
    p.add_argument('--model_search_2d', action='store_true',
                   help='pins: experimental model-only 2D Hough fit (basket x ±search_mm, y ±20 mm)')
    p.add_argument('--marker_size_m', type=float, help='basket_stats: actual printed black square side, override for this run only')
    p.add_argument('--repeats', type=int, default=10, help='basket_stats: repeat measurements')
    p.add_argument('--frames', type=int, default=20, help='basket_stats: frames in each measurement')
    a = p.parse_args()
    if a.mode == 'preview':
        preview(a)
    elif a.mode == 'marker':
        marker(a, cfg)
    elif a.mode == 'markers_live':
        markers_live(cfg)
    elif a.mode == 'basket_check':
        sys.exit(basket_check(cfg, a.relayout))
    elif a.mode == 'locate':
        locate(a, cfg)
    elif a.mode == 'basket':
        basket(a, cfg)
    elif a.mode == 'wrist_look':
        wrist_look(a, cfg)
    elif a.mode == 'basket_stats':
        basket_stats(a, cfg)
    elif a.mode == 'pins':
        pins_mode(a, cfg)
    elif a.mode == 'pin_layout':
        if not a.labels or not a.out:
            p.error('pin_layout requires --labels and --out')
        pin_layout_measure(a.labels, a.out)
    elif a.mode == 'plate_angle':
        if not a.req or not a.out:
            p.error('plate_angle requires --req and --out')
        plate_angle(a.req, a.out, a.plate_radius_m)
    elif a.mode == 'gap_error':
        if not a.req or not a.out:
            p.error('gap_error requires --req wrist_above.json and --out <new directory>')
        result = wrist_gap_error(a.req, a.out, a.quadrant, a.pin, a.gap_search_mm)
        print(json.dumps(dict(out=str(a.out), quadrant=result['commanded_quadrant'], pin=result['commanded_pin'],
                              photo_candidate_matches=result['photo_candidate_matches'],
                              photo_measurement=result['photo_measurement'],
                              checked_3d_layout_prior=result['checked_3d_layout_prior']),
                         ensure_ascii=False))
    elif a.mode == 'model_overlay':
        model_overlay(a.req, a.out)
    elif a.mode == 'slot_targets':
        if not a.out:
            p.error('slot_targets requires --out <new JSON path>')
        slot_targets(a.out)
    elif a.mode == 'plate_plane':
        print(json.dumps(plate_plane_mode(a), indent=2))
    elif a.mode == 'pre_photo':
        pre_photo(a, cfg)
    else:
        probe(a)
