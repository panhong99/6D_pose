"""Sensor inputs for the conda tools (read-only: cameras and localhost UDP; nothing is ever sent to the robot).

  ColorCam      colour-only D455 stream, undistorted to the pinhole K (wrist camera, fixed-camera live view)
  PoseStream    flange posx/posj from `ros2_nodes.py flange` (UDP 5006) + a short history for the stillness test
  FpStream      FoundationPose mesh-to-camera poses from real_time_project/main.py (UDP 5005); stable = last n agree
  Receiver      exclusive non-blocking JSON datagram reader (the pipeline's own poses / joints)
  capture       average n marker detections while checking the robot did not move (one calibration sample)
  novelty       distance of a flange pose to the stored samples (auto-capture only takes new views)
  average_marker  average n detections from an open stream (fixed camera at F)
"""
import collections
import json
import socket
import sys
import threading
import time
from collections import namedtuple
from pathlib import Path

import numpy as np

from calib_core import average_poses, detect_marker, posx_to_T, rot_angle_deg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'real_time_project'))
Frame = namedtuple('Frame', 'rgb K')


class ColorCam:
    """Colour-only D455 stream undistorted to K (same rectification as real_time_project)."""

    def __init__(self, serial, width=1280, height=720, fps=30):
        import pyrealsense2 as rs
        from real_time_utils import rectification_maps
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
        import cv2
        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        while True:                                     # drop queued frames: always the newest view
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


class PoseStream:
    """Latest flange pose from `ros2_nodes.py flange`; a background thread keeps ~20 s of history."""

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
        """(receive time, posx, posj) or None when older than max_age."""
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
            if np.linalg.norm(T[:3, 3] - T0[:3, 3]) * 1000 > mm or rot_angle_deg(T[:3, :3].T @ T0[:3, :3]) > deg:
                break
            t_still = t
        return time.time() - t_still


class FpStream:
    """FoundationPose poses (mesh-to-camera 4x4 + timestamp) from main.py; stable() = median of n agreeing poses."""

    def __init__(self, port, n, mm, deg, symmetric_z=False):
        self.symmetric_z = symmetric_z                     # round object: compare only the mesh z axis (yaw is arbitrary)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', port))
        self.sock.setblocking(False)
        self.hist, self.n, self.mm, self.deg = collections.deque(maxlen=n), n, mm, deg
        self.last, self.last_rx = 0.0, 0.0

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

    def _angle(self, a, b):
        if self.symmetric_z:
            return float(np.degrees(np.arccos(np.clip(a[:3, 2] @ b[:3, 2], -1.0, 1.0))))
        return rot_angle_deg(a[:3, :3].T @ b[:3, :3])

    def stable(self):
        """Median camera-frame pose, or None when old / too few / not agreeing."""
        if len(self.hist) < self.n or time.time() - self.last_rx > 1.0:
            return None
        ref = self.hist[-1]
        if (max(np.linalg.norm(h[:3, 3] - ref[:3, 3]) * 1000 for h in self.hist) > self.mm
                or max(self._angle(h, ref) for h in self.hist) > self.deg):
            return None
        out = ref.copy()
        out[:3, 3] = np.median([h[:3, 3] for h in self.hist], axis=0)
        return out


class Receiver:
    """Exclusive bind (a leftover planner must not steal this session's packets); packets() yields dict datagrams."""

    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind(('127.0.0.1', port))
            self.sock.setblocking(False)
        except BaseException:
            self.sock.close()
            raise

    def packets(self):
        while True:
            try:
                raw = self.sock.recv(65535)
            except BlockingIOError:
                return
            try:
                msg = json.loads(raw)
                if isinstance(msg, dict):
                    yield msg
            except (ValueError, UnicodeDecodeError):
                continue


def novelty(Tbf, samples):
    """(min translation mm, rotation deg of that nearest sample) to stored samples (40 mm ~ 10 deg weighting)."""
    best = (np.inf, np.inf)
    for s in samples:
        T = np.array(s['T_base_flange'])
        d = (np.linalg.norm(T[:3, 3] - Tbf[:3, 3]) * 1000, rot_angle_deg(T[:3, :3].T @ Tbf[:3, :3]))
        if max(d[0] / 40.0, d[1] / 10.0) < max(best[0] / 40.0, best[1] / 10.0):
            best = d
    return best


def capture(camera, detector, robot, marker_id, marker_size, n_frames, still_mm, still_deg, max_jitter_mm, min_tilt_deg):
    """One sample: n_frames marker poses averaged; refused if the robot moved, the marker is unsteady / head-on.
    Returns (sample dict, None) or (None, reason)."""
    import cv2
    start = robot.latest()
    if start is None:
        return None, 'no fresh robot pose'
    Ts, tilts, deadline = [], [], time.time() + 6.0
    while len(Ts) < n_frames and time.time() < deadline:
        frame = camera.read()
        det = detect_marker(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), frame.K, detector, marker_id, marker_size)
        if det is not None:
            Ts.append(det['T'])
            tilts.append(det['tilt_deg'])
    end = robot.latest()
    if end is None:
        return None, 'robot pose stream stopped'
    if len(Ts) < 0.6 * n_frames:
        return None, f'marker seen in only {len(Ts)}/{n_frames} frames'
    a, b = posx_to_T(start[1]), posx_to_T(end[1])
    if np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000 > still_mm or rot_angle_deg(a[:3, :3].T @ b[:3, :3]) > still_deg:
        return None, 'robot moved during capture'
    T = average_poses(Ts)
    jitter, tilt = float(np.std([t[:3, 3] for t in Ts], axis=0).mean() * 1000), float(np.mean(tilts))
    if jitter > max_jitter_mm:
        return None, f'marker jitter {jitter:.2f} mm'
    if tilt < min_tilt_deg:
        return None, f'marker too head-on (tilt {tilt:.0f} deg)'
    return dict(posx=end[1], posj=end[2], T_base_flange=b.tolist(), T_cam_marker=T.tolist(), jitter_mm=jitter,
                jitter_deg=float(np.mean([rot_angle_deg(T[:3, :3].T @ t[:3, :3]) for t in Ts])),
                tilt_deg=tilt, dist_m=float(np.linalg.norm(T[:3, 3])), source='ros2 get_current_tool_flange_posx',
                robot_stamp=end[0], time=time.strftime('%H:%M:%S')), None


def average_marker(cam, detector, marker_id, marker_size, n, timeout_s=8.0):
    """Average n detections from an open stream: (dict(T, tilt, jitter_mm, dist), None) or (None, reason)."""
    import cv2
    Ts, tilts, deadline = [], [], time.time() + timeout_s
    while len(Ts) < n and time.time() < deadline:
        f = cam.read()
        det = detect_marker(cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR), f.K, detector, marker_id, marker_size)
        if det is not None:
            Ts.append(det['T'])
            tilts.append(det['tilt_deg'])
    if len(Ts) < 0.6 * n:
        return None, f'marker seen in only {len(Ts)}/{n} frames'
    return dict(T=average_poses(Ts), tilt=float(np.mean(tilts)), jitter_mm=float(np.std([t[:3, 3] for t in Ts], axis=0).mean() * 1000),
                dist=float(np.linalg.norm(Ts[0][:3, 3]))), None


class RobotRecorder:
    """Background thread: every UDP sample of ros2_nodes.py flange (posj deg, flange posx mm/deg, t), the live FoundationPose
    plate root in base (kind=fp, <= 10 Hz, while the tracking window stays open) plus pipeline events
    (leg start/end, gripper, plan) as JSON lines - real-robot data for later training / sim-to-real checks."""

    def __init__(self, receiver, path, extra=None):
        import threading
        self.receiver, self.file, self.extra, self.t_extra = receiver, path.open('a'), extra, 0.0
        self.lock, self.leg, self.n, self.last = threading.Lock(), None, 0, None
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _write(self, rec):
        with self.lock:
            self.file.write(json.dumps(rec) + '\n')
            self.file.flush()

    def _run(self):
        while self.running:
            for msg in self.receiver.packets():
                self.n += 1
                self.last = msg
                self._write(dict(kind='state', leg=self.leg, t=msg.get('t'), posj=msg.get('posj'), posx=msg.get('posx'),
                                 tool_force=msg.get('tool_force'), ext_torque=msg.get('ext_torque')))
            if self.extra is not None and time.time() - self.t_extra > 0.1:              # FP plate pose at <= 10 Hz
                self.t_extra = time.time()
                try:
                    sample = self.extra()
                except Exception as exc:                                              # never stop the recorder
                    sample = dict(error=str(exc))
                if sample:
                    self._write(dict(kind='fp', leg=self.leg, t=self.t_extra, **sample))
            time.sleep(0.01)

    def event(self, name, **info):
        if name == 'leg_start':
            self.leg = info.get('leg')
        elif name == 'leg_end':
            self.leg = None
        self._write(dict(kind='event', event=name, t=time.time(), **info))

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)
        self._write(dict(kind='event', event='recorder_stop', t=time.time(), samples=self.n))
        self.file.close()
