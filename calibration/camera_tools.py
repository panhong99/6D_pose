"""D455 camera tools (conda env `foundationpose`, camera only, no robot).

  python calibration/camera_tools.py preview [--all | --serial S [--serial S2]]   live view; S saves colour+depth+K (one --serial: data/tray_capture/, full 1280x720)
  python calibration/camera_tools.py marker [--serial S] [--marker_mm 100]        live ArUco check + averaged report
  python calibration/camera_tools.py probe [--serial S | --image F] [--marker_mm M]   identify an unknown marker board
  python calibration/camera_tools.py locate                                       plate position vs the RL sim spot
  python calibration/camera_tools.py pins --out NEW_DIR [--layout MEASURED.json]  saved wrist photos only; numbered diagnostic overlays
  python calibration/camera_tools.py basket_stats --out NEW_DIR                   fixed camera repeatability; --marker_size_m is actual printed size
  python calibration/camera_tools.py pin_layout --labels LABELS.json --out NEW_DIR   ray/height-plane measurement from checked pixel labels
  python calibration/camera_tools.py plate_angle --req wrist_above.req.json --out NEW_DIR  RGB disk silhouette yaw diagnostic

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
            corners, ids, _ = det.detectMarkers(cv2.cvtColor(fr.rgb, cv2.COLOR_RGB2GRAY))
            for c, i in zip(corners, [] if ids is None else ids.ravel()):
                if int(i) in ids_wanted:
                    ok, rv, tv = cv2.solvePnP(obj, c.reshape(4, 2), fr.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
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
    for photo in labels['photos']:
        path = Path(photo['image'])
        if not path.name.endswith('.raw.png'):
            raise ValueError(f'Expected original wrist .raw.png: {path}')
        meta = json.loads(path.with_name(path.name[:-8] + '.json').read_text())
        Tbb = np.asarray(meta['T_base_basket'], float)
        Tbc = np.asarray(meta['T_base_flange'], float) @ np.asarray(meta['T_flange_cam'], float)
        K = np.asarray(meta['K'], float)
        for pin in photo['pins']:
            if pin.get('review') != 'checked':
                continue
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


def _pin_match(img, context, p0, p1, Tcb, K, axis, search_mm=8.0):
    """Local dark-wire ridge fit. No depth estimate from a guessed plate plane; offsets translate the 3D pin.
    Evaluate only the exposed upper tine, away from floor mesh and rail. Reject plate/marker occlusion and weak fits."""
    gray, excluded, lines = context
    h, w = gray.shape
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
        continuous.append(_pin_longest_run(support & valid)/len(t))
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
                visible_fraction=round(visibility[j], 3), model_uv=_pin_project(np.stack([p0, p1]), Tcb, K)[0].tolist(),
                detected_uv=fitted.tolist())


def _pin_line_match(context, p0, p1, Tcb, K, axis, search_mm):
    """Hough fallback for small errors transverse to the x-only model fit. Report unresolved pixel residual explicitly."""
    gray, excluded, lines = context
    uv, depth = _pin_project(np.stack([p0+(p1-p0)*0.53, p1]), Tcb, K)
    tangent = uv[1]-uv[0]
    length = np.linalg.norm(tangent)
    direction = tangent/length
    normal = np.array([-direction[1], direction[0]])
    J = (_pin_project(p1[None]+axis[None]*0.001, Tcb, K)[0][0]-uv[1])
    if np.linalg.norm(J) < 0.3:
        return None
    candidates = []
    h, w = gray.shape
    for line in lines:
        a, b = line[:2].astype(float), line[2:].astype(float)
        d = b-a; L = np.linalg.norm(d)
        if L < 0.55*length or L > 2.5*length or abs(np.dot(d/L, direction)) < 0.90:
            continue
        tip = a if np.linalg.norm(a-uv[1]) < np.linalg.norm(b-uv[1]) else b
        if np.linalg.norm(tip-uv[1]) > 19:
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
        if _pin_longest_run(ridge_mask)/len(ridge_mask) < 0.45:
            continue
        delta = best[2]-uv[1]
        dx = float(np.dot(delta, J)/np.dot(J, J))
        residual = delta-J*dx
        if abs(dx) >= search_mm or np.linalg.norm(residual) > 12:
            continue
        candidates.append((float(best[0]), float(best[1]), dx, residual, best[2]))
    if not candidates:
        return None
    candidates.sort(key=lambda c: -c[0])
    best = candidates[0]
    if any(c[0] > best[0]-3 and abs(c[2]-best[2]) > 3 for c in candidates[1:]):
        return None
    base_uv = _pin_project(np.stack([p0, p1]), Tcb, K)[0]
    corrected = base_uv+(best[4]-base_uv[1])[None]
    return dict(status='detected', method='hough_ridge', correction_mm=round(best[2], 3), candidate_mm=round(best[2], 3),
                ridge_score=round(best[0], 3), support=round(best[1], 3), visible_fraction=1.0,
                transverse_residual_px=round(float(np.linalg.norm(best[3])), 3),
                model_uv=base_uv.tolist(), detected_uv=corrected.tolist())


def wrist_pins(req, cfg, out, image=None, rotate='auto', search_mm=8.0, layout=None):
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
    for p, q in pair_ids:
        x = round((p['local_xyz'][0] + q['local_xyz'][0]) / 2, 5)
        row0, row1 = sorted((p['local_xyz'][1], q['local_xyz'][1]))
        if 'model_uv' not in p or 'model_uv' not in q:
            continue
        yc = (row0+row1)/2
        # Same basket-frame convention as plate_plans.basket_gaps: +y is left.
        quadrant = ('Q2' if yc >= 0 else 'Q1') if x >= 0 else ('Q3' if yc >= 0 else 'Q4')
        confirmed = p['status'] == q['status'] == 'detected'
        corrections = [p['correction_mm'], q['correction_mm']]
        if confirmed and abs(corrections[0]-corrections[1]) > 3.0:
            confirmed = False
            status = 'pair_disagreement'
        else:
            status = 'detected' if confirmed else 'partial_or_occluded'
        dx_mm = float(np.mean(corrections)) if confirmed else None
        # The observed correction is along basket +x only. Rejected pairs have no
        # usable target; do not silently return an uncorrected model point.
        gap_centre = ((pin_tops_base[p['model_id']] + pin_tops_base[q['model_id']]) / 2
                      + axis * (dx_mm / 1000)).tolist() if confirmed else None
        groups.append(dict(quadrant=quadrant, model_pin_ids=[p['model_id'], q['model_id']],
                           midpoint_uv=np.mean([p['model_uv'][1], q['model_uv'][1]], 0).tolist(),
                           status=status, correction_mm=dx_mm,
                           gap_centre_base_xyz=gap_centre,
                           local_x_m=x, row_y_m=[row0, row1]))
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
    for quadrant in ('Q1', 'Q2', 'Q3', 'Q4'):
        visible = [g for g in groups if g['quadrant'] == quadrant and 0 <= display(g['midpoint_uv'])[0] < view.shape[1] and 0 <= display(g['midpoint_uv'])[1] < view.shape[0]]
        for number, g in enumerate(sorted(visible, key=lambda g: tuple(display(g['midpoint_uv'])[::-1])), 1):
            g['number'] = number
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
            label = f"{quadrant}/{number} " + (f"{g['correction_mm']:+.1f}mm" if g['status'] == 'detected' else '?')
            cv2.putText(view, label, tuple(midpoint+np.array([5, -5])), cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)
    for g in groups:
        if 'number' not in g:
            g['gap_centre_base_xyz'] = None  # Off-image match is not an addressable gap.
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(out/'pairs.png'), view)
    result = dict(req=str(req.resolve()), image=str(raw_path.resolve()), handeye=str(handeye),
                  display_rotation_deg=rotation, numbering='quadrant, top-to-bottom in displayed image; hidden IDs retained',
                  quadrant_basis='basket frame: +x far, +y left; Q1 far/right, Q2 far/left, Q3 near/left, Q4 near/right',
                  correction_axis='basket +x; diagnostic only', search_mm=search_mm,
                  gap_centre_definition=('midpoint of two measured pin tips, shifted by confirmed basket +x image correction; diagnostic only'
                                         if layout else 'midpoint of two model pin tips, shifted by confirmed basket +x image correction; diagnostic only'),
                  confirmed_pairs=sum(g['status'] == 'detected' and 'number' in g for g in groups),
                  visible_model_pairs=sum('number' in g for g in groups) if layout is None else None,
                  visible_layout_pairs=sum('number' in g for g in groups), pins=pins, pairs=groups,
                  layout_source=str(Path(layout).resolve()) if layout else 'parametric_model',
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
            r = wrist_pins(req, cfg, out/f'{req.parent.name}_{req.stem}', a.image, a.rotate, a.search_mm, a.layout)
            offsets = [g['correction_mm'] for g in r['pairs'] if g['status'] == 'detected' and 'number' in g]
            rows.append(dict(photo=str(req), detected_pairs=r['confirmed_pairs'], visible_pairs=r['visible_layout_pairs'],
                             median_dx_mm=float(np.median(offsets)) if offsets else None, display_rotation=r['display_rotation_deg']))
        except (FileNotFoundError, ValueError) as e:
            rows.append(dict(photo=str(req), error=str(e)))
    import csv
    columns = ['photo', 'detected_pairs', 'visible_pairs', 'median_dx_mm', 'display_rotation', 'error']
    with (out/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps(dict(out=str(out), photos=len(rows), detected_pairs=sum(r.get('detected_pairs', 0) for r in rows)), ensure_ascii=False))


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


if __name__ == '__main__':
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=('preview', 'marker', 'probe', 'locate', 'basket', 'wrist_look', 'pre_photo', 'pins', 'basket_stats', 'pin_layout', 'plate_angle'))
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
    p.add_argument('--plate_radius_m', type=float, default=0.134, help='plate_angle: assumed dish outer radius')
    p.add_argument('--rotate', choices=('auto', '0', '90', '270'), default='auto', help='pins: display rotation, raw projection unchanged')
    p.add_argument('--search_mm', type=float, default=8.0, help='pins: maximum local correction along basket x')
    p.add_argument('--marker_size_m', type=float, help='basket_stats: actual printed black square side, override for this run only')
    p.add_argument('--repeats', type=int, default=10, help='basket_stats: repeat measurements')
    p.add_argument('--frames', type=int, default=20, help='basket_stats: frames in each measurement')
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
    elif a.mode == 'pre_photo':
        pre_photo(a, cfg)
    else:
        probe(a)
