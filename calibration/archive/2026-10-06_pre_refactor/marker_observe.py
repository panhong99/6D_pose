"""Read-only: observe ONE ArUco marker with a D455 and report its pose in the camera frame (no robot).

python calibration/marker_observe.py --serial 338122300585 --marker_id 1 --marker_mm 100 [--dictionary DICT_4X4_100]

Averages 30 frames; prints distance, tilt (angle between marker normal and viewing ray), reprojection error, position
jitter (std, mm) and the marker centre/normal in the camera frame.  Saves an annotated image to data/board_probe/.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'real_time_project'))
sys.path.insert(0, str(HERE))
from real_time_utils import D455Source  # noqa: E402
from calib_utils import detect_marker, make_detector  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--serial', required=True)
p.add_argument('--marker_id', type=int, default=1)
p.add_argument('--marker_mm', type=float, default=100.0)
p.add_argument('--dictionary', default='DICT_4X4_100')
p.add_argument('--frames', type=int, default=30)
a = p.parse_args()
cam = D455Source(1280, 720, 30, a.serial)
det = make_detector(a.dictionary)
poses, reps, tilts, last = [], [], [], None
for i in range(a.frames + 20):
    f = cam.read()
    if i < 20:
        continue                                                  # auto-exposure settle
    bgr = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR)
    r = detect_marker(bgr, f.K, det, a.marker_id, a.marker_mm / 1000.0)
    if r is not None:
        poses.append(r['T']); reps.append(r['reproj_px']); tilts.append(r['tilt_deg']); last = bgr
K = f.K
cam.close()
print(f'camera {a.serial}: fx {K[0, 0]:.1f} fy {K[1, 1]:.1f}; detected in {len(poses)}/{a.frames} frames')
if not poses:
    raise SystemExit('marker not detected')
T = np.array(poses)
t = T[:, :3, 3]
n = T[:, :3, 2]
print(f'marker centre in camera frame [m]: {t.mean(0).round(4).tolist()}  distance {np.linalg.norm(t.mean(0)):.3f} m')
print(f'jitter std [mm]: {(t.std(0) * 1000).round(2).tolist()}   tilt {np.mean(tilts):.1f} deg   reprojection {np.mean(reps):.2f} px')
print(f'marker normal (camera frame): {n.mean(0).round(3).tolist()}')
cv2.imwrite(str(HERE / 'data' / 'board_probe' / f'{time.strftime("%Y%m%d_%H%M%S")}_observe_{a.serial}.png'), last)
