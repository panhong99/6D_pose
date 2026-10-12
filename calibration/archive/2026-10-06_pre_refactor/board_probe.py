"""Identify an unknown multi-marker board from one D455 colour frame (read-only, no robot).

python calibration/board_probe.py --serial 338122300585          # fixed camera
python calibration/board_probe.py --serial 338122303684          # wrist camera
python calibration/board_probe.py --image some.png               # offline

Tries every predefined ArUco/AprilTag dictionary and prints, for the best one: the marker IDs, how many markers, each
marker's pixel size, and the layout (marker centres in units of the median marker side, from a homography of the board
plane).  A real-world scale needs ONE caliper measurement of the printed marker (black square side, mm): pass it with
--marker_mm to get the full board geometry in metres.  Saves an annotated image to calibration/data/board_probe/.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'real_time_project'))

DICTS = [n for n in dir(cv2.aruco) if n.startswith('DICT_')]


def grab(serial, width, height):
    from real_time_utils import D455Source
    cam = D455Source(width, height, 30, serial)
    for _ in range(30):                                   # let auto-exposure settle
        frame = cam.read()
    cam.close()
    return cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)


def detect_all(gray):
    best = None
    for name in DICTS:
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)), params)
        corners, ids, _ = det.detectMarkers(gray)
        n = 0 if ids is None else len(ids)
        if n and (best is None or n > best[1]):
            best = (name, n, corners, ids.flatten())
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--serial', default='')
    p.add_argument('--image', type=Path, default=None)
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--marker_mm', type=float, default=None, help='caliper-measured side of ONE printed black square, mm')
    a = p.parse_args()

    if a.image:
        img = cv2.imread(str(a.image))
    else:
        print('opening D455 ...', flush=True)
        img = grab(a.serial, a.width, a.height)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    best = detect_all(gray)
    out = HERE / 'data' / 'board_probe'
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    cv2.imwrite(str(out / f'{stamp}_raw.png'), img)
    if best is None:
        print('NO markers found in any dictionary. Check: board in view and sharp, not glare, not too small (> 25 px per marker).')
        return
    name, n, corners, ids = best
    cv2.aruco.drawDetectedMarkers(img, corners, ids.reshape(-1, 1))
    cv2.imwrite(str(out / f'{stamp}_annotated.png'), img)
    c = np.array([c.reshape(4, 2) for c in corners])                            # (n, 4, 2)
    sides = np.linalg.norm(c - np.roll(c, 1, axis=1), axis=2).mean(1)
    print(f'dictionary {name}: {n} markers, ids {sorted(ids.tolist())}')
    print(f'marker side in pixels: min {sides.min():.0f} median {np.median(sides):.0f} max {sides.max():.0f}')
    if n < 2:
        print('only one marker visible - move the camera so the whole board is in view for the layout.')
        return
    # board layout: homography from the image to a fronto-parallel plane using the marker centres' mutual geometry is
    # unknowable without a model, so estimate it as: fit a plane homography that makes the markers axis-aligned squares
    # of equal side.  Use the first marker as reference, then refine on all corners (least squares on side lengths).
    centres = c.mean(1)
    ref = c[np.argmax(sides)]
    side_ref = float(np.linalg.norm(ref[0] - ref[1]))
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    H, _ = cv2.findHomography(ref, square)                                       # image -> marker units of the reference marker
    plane = cv2.perspectiveTransform(centres.reshape(-1, 1, 2), H).reshape(-1, 2)
    order = np.lexsort((plane[:, 0].round(1), plane[:, 1].round(1)))
    unit = (a.marker_mm or 1.0)
    print(f'layout (marker centres relative to marker id {ids[np.argmax(sides)]}, in marker-side units'
          f'{"" if a.marker_mm is None else f", x {a.marker_mm:g} mm = mm"}):')
    for i in order:
        print(f'  id {ids[i]:3d}: x {plane[i, 0] * unit:8.2f}  y {plane[i, 1] * unit:8.2f}')
    if a.marker_mm is None:
        print('Measure the printed black square of any marker with calipers and re-run with --marker_mm <value> for the geometry in mm.')
    print(f'images saved in {out}/ ({stamp}_raw.png, {stamp}_annotated.png)')


if __name__ == '__main__':
    main()
