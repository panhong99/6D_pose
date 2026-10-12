"""Voxel safety shield for the M1013 + 2F-85 (torch, batched over envs).

Any arm joint target whose robot points come closer than FRAME_MARGIN to the steel frame or TABLE_MARGIN to the table
(1 cm distance fields from make_cell_sdf.py, robot base frame) is rejected; the arm keeps its last safe target instead.
Robot points: down-sampled M1013 link collision samples + a conservative box around the 2F-85 (link_6 frame).
FK follows calibration/kinematics.Kinematics (URDF revolute joints) and is checked against Isaac in check_shield.py.
"""
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
URDF = Path('/home/pan/pan/doosan_ws/install/dsr_description2/share/dsr_description2/urdf/m1013.urdf')
FRAME_MARGIN = 0.05      # m, steel frame keep-out (Human: never touch the frame)
TABLE_MARGIN = 0.02      # m, table keep-out (fingertips are ~5 cm above the table at the rim grasp)
GRIPPER_DIR = HERE.parent / 'assets' / 'robotiq_2f85'
GRIPPER_SURFACE_VOXEL = 0.003   # 3 mm surface sampling, finer than the 10 mm cell SDF


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _box_surface(lims, step=0.02):
    (x0, x1), (y0, y1), (z0, z1) = lims
    xs, ys, zs = (np.arange(a, b + 1e-9, step) for a, b in lims)
    pts = []
    for x in (x0, x1):
        pts += [(x, y, z) for y in ys for z in zs]
    for y in (y0, y1):
        pts += [(x, y, z) for x in xs for z in zs]
    for z in (z0, z1):
        pts += [(x, y, z) for x in xs for y in ys]
    return np.unique(np.round(np.array(pts), 4), axis=0)


def _quat_to_R(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _read_stl_vertices(path):
    data = Path(path).read_bytes()
    count = int(np.frombuffer(data[80:84], '<u4')[0])
    if 84 + 50 * count == len(data):
        rec = np.frombuffer(data[84:], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), count=count)
        return rec['v'].reshape(-1, 3).astype(np.float32) * 0.001
    return np.asarray([[float(v) for v in line.split()[1:4]] for line in data.decode().splitlines()
                       if line.strip().startswith('vertex')], dtype=np.float32) * 0.001


def _gripper_surface_points():
    """Actual open Robotiq visual meshes, expressed in link_6 frame and voxel-decimated.

    The former enclosing box put an empty box corner below the table and blocked rim entry about 5 cm early.
    Mesh vertices keep the conservative shield tied to the gripper's occupied shape instead.
    """
    root = ET.parse(GRIPPER_DIR / '2f85.xml').getroot()
    mesh_files = {Path(mesh.get('file')).stem: mesh.get('file') for mesh in root.iter('mesh') if mesh.get('file')}
    points = []

    def walk(body, parent_T):
        pos = np.fromstring(body.get('pos', '0 0 0'), sep=' ')
        quat = np.fromstring(body.get('quat', '1 0 0 0'), sep=' ')
        T = parent_T.copy()
        R = _quat_to_R(quat)
        T[:3, :3] = parent_T[:3, :3] @ R
        T[:3, 3] = parent_T[:3, :3] @ pos + parent_T[:3, 3]
        for geom in body.findall('geom'):
            name = geom.get('mesh')
            if geom.get('class') != 'visual' or name not in mesh_files:
                continue
            verts = _read_stl_vertices(GRIPPER_DIR / 'assets' / mesh_files[name])
            gpos = np.fromstring(geom.get('pos', '0 0 0'), sep=' ')
            gquat = np.fromstring(geom.get('quat', '1 0 0 0'), sep=' ')
            gR = _quat_to_R(gquat)
            points.append(verts @ gR.T @ T[:3, :3].T + (T[:3, :3] @ gpos + T[:3, 3]))
        for child in body.findall('body'):
            walk(child, T)

    # Human-confirmed 20 mm flange coupling, followed by the MJCF base_mount and base offsets.
    mount = np.eye(4)
    mount[:3, 3] = (0.0, 0.0, 0.020)
    walk(root.find('worldbody').find('body'), mount)
    points = np.concatenate(points).astype(np.float32)
    # Approximate the 20 mm AGC coupling as a 40 mm radius cylinder between link_6 and base_mount.
    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    ring = np.column_stack([0.040 * np.cos(angles), 0.040 * np.sin(angles)])
    coupling = np.concatenate([np.column_stack([ring, np.full(len(ring), z)]) for z in (0.0, 0.010, 0.020)])
    points = np.concatenate([points, coupling.astype(np.float32)])
    keys = np.floor(points / GRIPPER_SURFACE_VOXEL).astype(np.int32)
    _, keep = np.unique(keys, axis=0, return_index=True)
    return points[keep]


class SafetyShield:
    def __init__(self, device):
        self.device = device
        origins, axes = [], []
        for j in ET.parse(URDF).getroot().findall('joint'):
            if j.get('type') != 'revolute':
                continue
            o = j.find('origin')
            xyz = [float(v) for v in o.get('xyz', '0 0 0').split()]
            rpy = [float(v) for v in o.get('rpy', '0 0 0').split()]
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = _rpy_to_R(*rpy), xyz
            origins.append(T)
            axes.append([float(v) for v in j.find('axis').get('xyz').split()])
        assert len(origins) == 6
        self.origins = torch.tensor(np.array(origins), dtype=torch.float32, device=device)       # (6, 4, 4)
        self.axes = torch.tensor(np.array(axes), dtype=torch.float32, device=device)            # (6, 3)
        d = np.load(HERE / 'assets' / 'cell_sdf.npz')
        self.lo = torch.tensor(d['lo'], dtype=torch.float32, device=device)
        self.res = float(d['res'])
        # grid_sample wants (N, C, D, H, W) = (1, 1, z, y, x)
        self.sdf = {k: torch.tensor(d[k], dtype=torch.float32, device=device).permute(2, 1, 0)[None, None].contiguous()
                    for k in ('frame', 'table')}
        self.size = torch.tensor(d['frame'].shape, dtype=torch.float32, device=device)            # (nx, ny, nz)
        pts = [torch.tensor(d[f'link_{i}'], dtype=torch.float32, device=device) for i in range(1, 7)]
        pts[5] = torch.cat([pts[5], torch.tensor(_gripper_surface_points(), dtype=torch.float32, device=device)])
        self.link_pts = pts                                                                      # per link, local frame

    def link_frames(self, q):
        """q (N, 6) rad -> list of 6 link transforms (N, 4, 4) in the base frame."""
        n = q.shape[0]
        T = torch.eye(4, device=self.device).expand(n, 4, 4)
        out = []
        for i in range(6):
            a = self.axes[i]
            th = q[:, i]
            K = torch.zeros(n, 3, 3, device=self.device)
            K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -a[2], a[1], -a[0]
            K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = a[2], -a[1], a[0]
            s, c = torch.sin(th)[:, None, None], torch.cos(th)[:, None, None]
            R = torch.eye(3, device=self.device) + s * K + (1 - c) * (K @ K)                      # Rodrigues
            J = torch.eye(4, device=self.device).repeat(n, 1, 1)
            J[:, :3, :3] = R
            T = T @ self.origins[i] @ J
            out.append(T)
        return out

    def _lookup(self, grid, p):
        """Trilinear distance at points p (N, M, 3) in the base frame; outside the grid = far."""
        idx = (p - self.lo) / self.res - 0.5                                                    # voxel-centre coords
        g = 2.0 * idx / (self.size - 1) - 1.0                                                   # align_corners=True
        v = F.grid_sample(grid.expand(p.shape[0], -1, -1, -1, -1), g[:, :, None, None, :], mode='bilinear',
                          padding_mode='border', align_corners=True)                           # (N, 1, M, 1, 1)
        return v[:, 0, :, 0, 0]

    def clearance(self, q):
        """(N, 6) joint angles -> (frame clearance, table clearance), each (N,) metres."""
        frames = self.link_frames(q)
        world = torch.cat([pts @ T[:, :3, :3].transpose(1, 2) + T[:, None, :3, 3]
                           for pts, T in zip(self.link_pts, frames)], dim=1)                    # (N, M, 3)
        return self._lookup(self.sdf['frame'], world).amin(1), self._lookup(self.sdf['table'], world).amin(1)

    def safe(self, q):
        f, t = self.clearance(q)
        return (f >= FRAME_MARGIN) & (t >= TABLE_MARGIN)
