"""Precompute the work-cell distance fields for the RL safety shield (run with the conda python: numpy + scipy).

Output rl/assets/cell_sdf.npz: two 1 cm Euclidean distance grids in the robot base frame (m to the nearest occupied voxel)
  frame : steel frame uprights + rails (calibration/workcell.frame_layout + data/real_cell_measured.json)
  table : table slab (top z = -0.020)
and M1013 link collision samples (calibration/data/m1013_collision_points.npz), down-sampled per link.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
CAL = HERE.parent / 'calibration'
sys.path.insert(0, str(CAL))
import workcell as wc  # noqa: E402

RES = 0.01
LO = np.array([-0.6, -0.9, -0.15])
HI = np.array([1.8, 0.9, 1.7])
PTS_PER_LINK = 200

cell = dict(wc.FRAME_CELL)
cell.update({k: v for k, v in json.loads((CAL / 'data' / 'real_cell_measured.json').read_text()).items() if not k.startswith('_')})
boxes = wc.frame_layout(cell)
shape = np.ceil((HI - LO) / RES).astype(int)
centres = LO + (np.indices(shape).transpose(1, 2, 3, 0) + 0.5) * RES


def edt(selected):
    occ = np.zeros(shape, dtype=bool)
    for _, c, s in selected:
        occ |= np.all(np.abs(centres - np.asarray(c)) <= np.asarray(s) / 2, axis=-1)
    return ndimage.distance_transform_edt(~occ, sampling=RES).astype(np.float32)


frame = edt([b for b in boxes if b[0] != 'table'])
table = edt([b for b in boxes if b[0] == 'table'])
pts = np.load(CAL / 'data' / 'm1013_collision_points.npz')
rng = np.random.default_rng(0)
links = {f'link_{i}': pts[f'link_{i}'][rng.choice(len(pts[f'link_{i}']), PTS_PER_LINK, replace=False)] for i in range(1, 7)}
np.savez_compressed(HERE / 'assets' / 'cell_sdf.npz', lo=LO, res=RES, frame=frame, table=table,
                    boxes=json.dumps([[n, list(map(float, c)), list(map(float, s))] for n, c, s in boxes]), **links)
print('grid', shape, 'frame boxes', [b[0] for b in boxes if b[0] != 'table'])
