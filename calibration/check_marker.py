"""Step 1: check that the D455 sees the flange ArUco marker and that its pose is stable.

python calibration/check_marker.py                      # DICT_4X4_50, id 1, 25 mm, 1280x720
python calibration/check_marker.py --marker_size 0.05

Shows the marker axes and, per second, distance / tilt / reprojection error / jitter.
Look for: small jitter (std of position, mm) while the marker is still, and a tilt of
20-65 deg (a marker facing the camera head-on has an ambiguous rotation).  Q/ESC: quit.
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'real_time_project'))
from real_time_utils import D455Source  # aligned + distortion-rectified frames, same K as main.py
from calib_utils import detect_marker, make_detector


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--marker_id', type=int, default=1)
    p.add_argument('--marker_size', type=float, default=0.025, help='side of the black square, meters')
    p.add_argument('--dictionary', default='DICT_4X4_50')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--serial', default='')
    return p.parse_args(argv)


def main():
    a = parse_args()
    detector = make_detector(a.dictionary)
    print('Opening D455 (rectification maps are computed once, this can take a few seconds) ...', flush=True)
    camera = D455Source(a.width, a.height, a.fps, a.serial)
    print(f'K=\n{camera.K}\nQ/ESC: quit', flush=True)
    history, seen, total, last_report = deque(maxlen=30), 0, 0, time.time()
    try:
        while True:
            frame = camera.read()
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            det = detect_marker(bgr, frame.K, detector, a.marker_id, a.marker_size)
            total += 1
            text = 'marker not found'
            if det is not None:
                seen += 1
                T = det['T']
                history.append(T[:3, 3] * 1000)
                cv2.drawFrameAxes(bgr, frame.K, None, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], a.marker_size)
                cv2.polylines(bgr, [det['corners'].astype(np.int32)], True, (0, 255, 0), 2)
                jitter = np.std(history, axis=0).mean() if len(history) >= 5 else float('nan')
                text = (f'dist {np.linalg.norm(T[:3, 3]) * 100:.1f}cm tilt {det["tilt_deg"]:.0f}deg '
                        f'reproj {det["reproj_px"]:.2f}px jitter {jitter:.2f}mm')
            cv2.putText(bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('marker check', bgr)
            if time.time() - last_report > 1.0:
                print(f'[{seen}/{total} frames detected] {text}', flush=True)
                seen, total, last_report = 0, 0, time.time()
            if cv2.waitKey(1) & 0xff in (ord('q'), 27):
                break
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
