"""M1013 model for planning: kinematics, the static work cell with voxel collision checks, and joint-path timing.
Pure numpy/scipy (no Isaac, no ROS) so the conda tools, Isaac scripts and rl/ can all import it.

  kinematics  Kinematics (URDF FK / IK), HOME_JOINTS_DEG, cube_yaw_deg, grasp_frames (top-down pregrasp above a cube)
  work cell   FRAME_CELL, frame_layout (table, frame uprights / rails), D455_BOX, load_link_points, CollisionChecker
              (distance field of the cell + samples of the robot's collision meshes; straight / RRT plan, tidy, simplify)
  trajectory  arc_length, polyline_path, smooth_path (PCHIP), time_parameterize (speed / accel limited), joint_limits_check

  python calibration/robot_model.py reach [--centre X Y Z]    IK + cell-clearance map around a cube position
  python calibration/robot_model.py points                    rebuild data/m1013_collision_points.npz (needs trimesh+pycollada)

Frames: robot base, metres, x toward the object, +y left, z up, base bottom z = 0; table top z = -0.020 (20 mm plate).
Kinematics checked on the real robot: q = 0 -> flange (0.14, 34.8, 1452.5) mm model vs (0.02, 34.5, 1452.5) controller.
"""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from calib_core import DATA, load_config, load_json, make_T

C = load_config()
HOME_JOINTS_DEG = tuple(C['robot']['home_joints_deg'])
M1013_SHARE = Path(C['paths']['m1013_share'])
DEFAULT_URDF = M1013_SHARE / 'urdf' / 'm1013.urdf'


# ================================================================ kinematics

class Kinematics:
    def __init__(self, urdf=DEFAULT_URDF):
        self.joints = []
        for j in ET.parse(urdf).getroot().findall('joint'):
            if j.get('type') != 'revolute':
                continue
            o = j.find('origin')
            xyz = np.fromstring(o.get('xyz', '0 0 0'), sep=' ')
            rpy = np.fromstring(o.get('rpy', '0 0 0'), sep=' ')
            lim = j.find('limit')
            self.joints.append((j.find('child').get('link'), make_T(Rotation.from_euler('xyz', rpy).as_matrix(), xyz),
                                np.fromstring(j.find('axis').get('xyz'), sep=' '),
                                float(lim.get('lower')), float(lim.get('upper'))))
        assert len(self.joints) == 6

    def frames(self, q):
        t = np.eye(4)
        result = {'base_link': t.copy()}
        for angle, (name, origin, axis, _, _) in zip(q, self.joints):
            t = t @ origin @ make_T(Rotation.from_rotvec(axis * angle).as_matrix(), [0, 0, 0])
            result[name] = t.copy()
        return result

    def fk(self, q):
        return self.frames(q)['link_6']

    def ik(self, target, seed):
        def residual(q):
            actual = self.fk(q)
            return np.r_[actual[:3, 3] - target[:3, 3],
                         Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec() * .25]
        lo, hi = np.array([j[3] for j in self.joints]), np.array([j[4] for j in self.joints])
        seed = np.clip(seed, lo + 1e-6, hi - 1e-6)            # a seed at/over a limit (e.g. a taught pose) is refused by scipy
        fit = least_squares(residual, seed, bounds=(lo, hi), max_nfev=250)
        if np.linalg.norm(residual(fit.x)) > 1e-4:
            raise RuntimeError('Unreachable IK target')
        return fit.x

    def limits_deg(self):
        return [(np.degrees(j[3]), np.degrees(j[4])) for j in self.joints]


TOP_DOWN = np.diag([1.0, -1.0, -1.0])


def cube_yaw_deg(base_obj):
    """Yaw of the cube's side faces about base Z, modulo 90 deg, in [-45, 45); and the tilt of its up axis (deg).
    Uses the most horizontal axis after dropping the most vertical one, whichever face FoundationPose calls 'up'."""
    rot = base_obj[:3, :3]
    up = int(np.argmax(np.abs(rot[2])))
    side = rot[:, (up + 1) % 3]
    yaw = np.degrees(np.arctan2(side[1], side[0]))
    return float((yaw + 45.0) % 90.0 - 45.0), float(np.degrees(np.arccos(min(1.0, abs(rot[2, up])))))


def grasp_frames(kin, cube_centre, yaw_deg, tool_length_m, pregrasp_m):
    """Top-down wrist (j4 ~ 0, j5 points the tool down) with j6 on a cube face.
    Returns (yaw, pregrasp flange T, grasp flange T, q_pregrasp, q_grasp) for the face-aligned yaw (every 90 deg)
    needing the least j6 twist.  TCP = tool_length_m along flange +z; pregrasp raises it pregrasp_m in base Z."""
    seed = np.radians([np.degrees(np.arctan2(cube_centre[1], cube_centre[0])), 0, 90, 0, 90, 0])
    best = None
    for k in range(4):
        yaw = yaw_deg + 90.0 * k
        rotation = Rotation.from_euler('z', yaw, degrees=True).as_matrix() @ TOP_DOWN
        flange_up = np.array([0.0, 0.0, tool_length_m])               # tool points down, flange above the TCP
        pre = make_T(rotation, np.asarray(cube_centre) + [0, 0, pregrasp_m] + flange_up)
        grasp = make_T(rotation, np.asarray(cube_centre) + flange_up)
        try:
            q_pre = kin.ik(pre, seed)
            q_grasp = kin.ik(grasp, q_pre)
        except RuntimeError:
            continue
        twist = abs((np.degrees(q_pre[5]) + 180.0) % 360.0 - 180.0)
        if best is None or twist < best[0]:
            best = (twist, ((yaw + 180.0) % 360.0 - 180.0, pre, grasp, q_pre, q_grasp))
    if best is None:
        raise RuntimeError('No reachable top-down grasp yaw for this cube')
    return best[1]


# ================================================================ work cell

MESH_DIR = M1013_SHARE / 'meshes' / 'm1013_collision'
LINK_MESHES = {                                     # collision meshes per URDF link (m1013.urdf), exported in mm
    'link_1': ['MF1013_1_0.dae'],
    'link_2': ['MF1013_2_0.dae', 'MF1013_2_1.dae', 'MF1013_2_2.dae'],
    'link_3': ['MF1013_3_0.dae'],
    'link_4': ['MF1013_4_0.dae', 'MF1013_4_1.dae'],
    'link_5': ['MF1013_5_0.dae'],
    'link_6': ['MF1013_6_0.dae'],
}
# link_1 / link_2 only spin about their own axes 80 mm above the table and > 0.7 m from the frame: counting them
# would eat the keep-out margin without ever being able to move away from the table.
STATIC_LINKS = ('link_1', 'link_2')
PLAN_MARGIN_FACTOR = 1.3                            # plan with extra room: edges are only sampled every few degrees
POINT_CACHE = DATA / 'm1013_collision_points.npz'   # Isaac python has no pycollada: samples are cached once

# MEASURED (Human 2026-10-01/02): table 1.80 x 1.20 x 0.10 m, robot on a 20 mm plate on the table, uprights at
# x = 0.85 / 1.65, y = +-0.56, frame 1.20 m high.  ASSUMED: 40 mm profile, base 0.30 m from the table edge, rails.
# data/real_cell_measured.json overrides any key (paths.cell_json).
FRAME_CELL = dict(
    table_size=(1.80, 1.20), table_thickness=0.10, base_plate_thickness=0.020, robot_from_table_edge=0.30,
    frame_centre_xy=(1.25, 0.0), frame_size=(0.84, 1.16), frame_height=1.20, profile=0.04,
    collision_margin_m=0.10, omit_boxes=(),
)
D455_BOX = (0.029, 0.124, 0.026)                    # depth x width x height of the D455 body


def measured_cell(cell_json=None):
    """FRAME_CELL updated with the measured json (default: config paths.cell_json)."""
    from calib_core import cpath
    cell = dict(FRAME_CELL)
    path = Path(cell_json) if cell_json else cpath(C['paths']['cell_json'])
    if path.is_file():
        cell.update({k: v for k, v in load_json(path).items() if not k.startswith('_')})
    return cell


def frame_layout(cell=FRAME_CELL):
    """Static boxes of the work cell: list of (name, centre[3], size[3])."""
    tx, ty = cell['table_size']
    edge = cell['robot_from_table_edge']
    cx, cy = cell['frame_centre_xy']
    fx, fy = cell['frame_size']
    h, p = cell['frame_height'], cell['profile']
    z0 = -cell['base_plate_thickness']                     # table top; the frame stands on the table
    boxes = [('table', [tx / 2 - edge, 0.0, z0 - cell['table_thickness'] / 2], [tx, ty, cell['table_thickness']])]
    xs = (cx - fx / 2 + p / 2, cx + fx / 2 - p / 2)
    ys = (cy - fy / 2 + p / 2, cy + fy / 2 - p / 2)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            boxes.append((f'upright_{i}{j}', [x, y, z0 + h / 2], [p, p, h]))
    top = z0 + h - p / 2
    for i, x in enumerate(xs):
        boxes.append((f'rail_y_{i}', [x, cy, top], [p, fy - 2 * p, p]))
    for j, y in enumerate(ys):
        boxes.append((f'rail_x_{j}', [cx, y, top], [fx - 2 * p, p, p]))
    omit = set(cell.get('omit_boxes', ()))
    return [b for b in boxes if b[0] not in omit]


def sample_link_points(points_per_link=600, seed=0):
    """Surface samples + some vertices of each link's collision mesh (needs trimesh + pycollada)."""
    import trimesh
    rng = np.random.default_rng(seed)
    points = {}
    for link, files in LINK_MESHES.items():
        mesh = trimesh.util.concatenate([trimesh.load(str(MESH_DIR / f), force='mesh') for f in files])
        pts, _ = trimesh.sample.sample_surface(mesh, points_per_link, seed=int(rng.integers(1 << 31)))
        verts = mesh.vertices[rng.choice(len(mesh.vertices), points_per_link // 4)]
        points[link] = np.vstack([pts, verts]) * 0.001
    return points


def load_link_points(points_per_link=600, seed=0):
    """Cached samples when they match the request, else sample them (cache only if absent)."""
    if POINT_CACHE.is_file():
        data = np.load(POINT_CACHE)
        if int(data['points_per_link']) == points_per_link and int(data['seed']) == seed:
            return {link: data[link] for link in LINK_MESHES}
    points = sample_link_points(points_per_link, seed)
    if not POINT_CACHE.is_file():                   # never replace the default cache with other settings
        POINT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(POINT_CACHE, points_per_link=points_per_link, seed=seed, **points)
    return points


class CollisionChecker:
    """Distance-field check of M1013 collision-mesh samples (+ optional tool cylinder) against static boxes."""

    def __init__(self, kin, boxes, resolution=0.01, margin=None, points_per_link=600,
                 bounds=((-0.6, 1.8), (-0.9, 0.9), (-0.15, 1.7)), seed=0):
        margin = FRAME_CELL['collision_margin_m'] if margin is None else margin
        self.kin, self.margin, self.res = kin, margin, resolution
        self.plan_margin = margin * PLAN_MARGIN_FACTOR
        self.lo = np.array([b[0] for b in bounds], dtype=float)
        shape = np.ceil((np.array([b[1] for b in bounds]) - self.lo) / resolution).astype(int)
        occ = np.zeros(shape, dtype=bool)
        centres = self.lo + (np.indices(shape).transpose(1, 2, 3, 0) + 0.5) * resolution
        for box in boxes:                               # (name, centre, size[, R]): R = 3x3 box axes in base (10/13 basket walls)
            _, centre, size = box[:3]
            d = centres - np.asarray(centre)
            if len(box) > 3:
                d = d @ np.asarray(box[3])
            occ |= np.all(np.abs(d) <= np.asarray(size) / 2, axis=-1)
        self.occupancy = occ
        self.distance = ndimage.distance_transform_edt(~occ, sampling=resolution).astype(np.float32)   # m to nearest occupied
        self.boxes = boxes
        self.link_points = {k: v for k, v in load_link_points(points_per_link, seed).items() if k not in STATIC_LINKS}
        self.tool = None

    def set_tool(self, length, radius):
        """Approximate a gripper/hand as a cylinder of points along flange +z."""
        if length <= 0:
            self.tool = None
            return
        z = np.linspace(0.0, length, max(2, int(length / 0.01)))
        a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        ring = np.c_[radius * np.cos(a), radius * np.sin(a)]
        self.tool = np.vstack([np.c_[ring, np.full(len(a), zz)] for zz in z] + [[0, 0, zz] for zz in z])

    def robot_points(self, q):
        frames = self.kin.frames(q)
        clouds = [pts @ frames[link][:3, :3].T + frames[link][:3, 3] for link, pts in self.link_points.items()]
        if self.tool is not None:
            t = frames['link_6']
            clouds.append(self.tool @ t[:3, :3].T + t[:3, 3])
        return np.vstack(clouds)

    def clearance(self, q):
        """Smallest distance (m) from any robot sample to the static boxes."""
        pts = self.robot_points(q)
        idx = np.floor((pts - self.lo) / self.res).astype(int)
        inside = np.all((idx >= 0) & (idx < np.array(self.distance.shape)), axis=1)
        d = np.full(len(pts), np.inf, dtype=np.float32)
        d[inside] = self.distance[tuple(idx[inside].T)]
        return float(d.min())

    def free(self, q):
        """Planning test: clearance with extra room beyond the verified margin."""
        return self.clearance(q) >= self.plan_margin

    def path_clearance(self, path):
        return min(self.clearance(q) for q in path)

    def plan(self, start, goal, step_deg=2.0, max_nodes=4000, seed=0):
        """Straight joint line if collision-free, else bidirectional RRT + shortcut + tidy."""
        if self._edge_free(start, goal, step_deg):
            return [start, goal], 'straight'
        rng = np.random.default_rng(seed)
        lower = np.array([j[3] for j in self.kin.joints])
        upper = np.array([j[4] for j in self.kin.joints])
        span_lo = np.maximum(lower, np.minimum(start, goal) - np.radians(90))   # sample near the start/goal box
        span_hi = np.minimum(upper, np.maximum(start, goal) + np.radians(90))
        trees, parents = [[start], [goal]], [[-1], [-1]]
        for _ in range(max_nodes):
            sample = rng.uniform(span_lo, span_hi)
            a, b = (0, 1) if len(trees[0]) <= len(trees[1]) else (1, 0)
            new = self._extend(trees[a], parents[a], sample, step_deg)
            if new is None:
                continue
            if self._connect(trees[b], parents[b], new, step_deg):
                path_a = self._trace(trees[a], parents[a], len(trees[a]) - 1)
                path_b = self._trace(trees[b], parents[b], len(trees[b]) - 1)
                path = path_a[::-1] + path_b[1:] if a == 0 else path_b[::-1] + path_a[1:]
                return self.tidy(self._shortcut(path, step_deg, rng)), 'rrt+tidy'
        raise RuntimeError('No collision-free joint path found (RRT budget exhausted)')

    def tidy(self, path, step_deg=4.0, iters=20, subdivide=4):
        """Elastic-band relaxation: pull each interior waypoint toward its neighbours' midpoint while both edges stay
        collision-free (removes the sampling planner's wrist flicks; the clearance margin still holds)."""
        pts = [np.asarray(path[0], dtype=float)]
        for a, b in zip(path[:-1], path[1:]):
            pts += [a + (b - a) * k / subdivide for k in range(1, subdivide + 1)]
        for _ in range(iters):
            moved = 0.0
            for i in range(1, len(pts) - 1):
                mid = (pts[i - 1] + pts[i + 1]) / 2
                for f in (1.0, 0.5, 0.25):
                    cand = pts[i] + (mid - pts[i]) * f
                    if (self.free(cand) and self._edge_free(pts[i - 1], cand, step_deg)
                            and self._edge_free(cand, pts[i + 1], step_deg)):
                        moved += float(np.abs(cand - pts[i]).max())
                        pts[i] = cand
                        break
            if moved < np.radians(0.05):
                break
        return pts

    def simplify(self, waypoints, max_joint_deg=45.0, min_clearance=None):
        """Fewest point-to-point (straight joint) moves through the waypoints: greedily jump to the farthest later
        waypoint whose straight move keeps min_clearance (default: the margin) and turns no joint more than
        max_joint_deg.  Never less safe than the input path."""
        need = self.margin if min_clearance is None else min_clearance
        wp = [np.asarray(w, dtype=float) for w in waypoints]
        keep, i = [wp[0]], 0
        while i < len(wp) - 1:
            best = i + 1
            for j in range(len(wp) - 1, i, -1):
                if np.degrees(np.abs(wp[j] - wp[i]).max()) > max_joint_deg and j > i + 1:
                    continue
                if self.path_clearance(np.array([wp[i] + (wp[j] - wp[i]) * t for t in np.linspace(0, 1, max(
                        2, int(np.ceil(np.degrees(np.abs(wp[j] - wp[i]).max()) / 0.25)) + 1))])) >= need:
                    best = j
                    break
            keep.append(wp[best])
            i = best
        return keep

    def _edge_free(self, a, b, step_deg):
        """Interior samples need the planning margin, the two ends only the verified one."""
        n = max(2, int(np.ceil(np.degrees(np.abs(b - a).max()) / step_deg)) + 1)
        for k, t in enumerate(np.linspace(0, 1, n)):
            if self.clearance(a + (b - a) * t) < (self.margin if k in (0, n - 1) else self.plan_margin):
                return False
        return True

    def _extend(self, tree, parents, target, step_deg, max_step_deg=10.0):
        i = int(np.argmin([np.abs(n - target).max() for n in tree]))
        near = tree[i]
        delta = target - near
        new = near + delta * min(1.0, np.radians(max_step_deg) / max(np.abs(delta).max(), 1e-9))
        if not self._edge_free(near, new, step_deg):
            return None
        tree.append(new)
        parents.append(i)
        return new

    def _connect(self, tree, parents, target, step_deg):
        while True:
            new = self._extend(tree, parents, target, step_deg)
            if new is None:
                return False
            if np.abs(new - target).max() < 1e-9:
                return True

    @staticmethod
    def _trace(tree, parents, i):
        path = []
        while i != -1:
            path.append(tree[i])
            i = parents[i]
        return path

    def _shortcut(self, path, step_deg, rng, rounds=60):
        path = list(path)
        for _ in range(rounds):
            if len(path) < 3:
                break
            i, j = sorted(rng.choice(len(path), 2, replace=False))
            if j - i > 1 and self._edge_free(path[i], path[j], step_deg):
                path = path[:i + 1] + path[j:]
        return path

    def export(self, path):
        """Occupancy grid + box list for reuse (np.load / json)."""
        path = Path(path)
        np.savez_compressed(path, occupancy=self.occupancy, origin=self.lo, resolution=self.res, margin=self.margin)
        path.with_suffix('.json').write_text(json.dumps(
            dict(frame='robot base, metres, z up', resolution=self.res, origin=self.lo.tolist(),
                 shape=list(self.occupancy.shape), margin=self.margin,
                 boxes=[dict(name=b[0], centre=list(map(float, b[1])), size=list(map(float, b[2]))) for b in self.boxes]),
            indent=1))


class MultiChecker(CollisionChecker):
    """(10/13) several checkers (own boxes, margins, tool clouds) as one for planning: clearance = the smallest
    clearance / margin ratio, so 1.0 is "at the margin" of the tightest checker."""

    def __init__(self, checkers):
        self.checkers, self.kin = checkers, checkers[0].kin
        self.margin, self.plan_margin = 1.0, PLAN_MARGIN_FACTOR

    def clearance(self, q):
        return min(c.clearance(q) / c.margin for c in self.checkers)


def build_checker(kin, T_base_cam, margin_m, tool_length_m=0.0, tool_radius_m=0.05, cell_json=None):
    """Checker for the measured cell plus the fixed D455 body as an obstacle.  Returns (checker, cell)."""
    cell = measured_cell(cell_json)
    cell['collision_margin_m'] = margin_m
    FRAME_CELL.update(cell)                          # CollisionChecker reads the module-level margin default
    boxes = frame_layout(cell)
    boxes.append(('d455_body', np.asarray(T_base_cam)[:3, 3].tolist(), list(D455_BOX)))
    checker = CollisionChecker(kin, boxes, margin=margin_m)
    checker.set_tool(tool_length_m, tool_radius_m)
    return checker, cell


# ================================================================ trajectory (joint radians, dt seconds)

def _dedupe(waypoints):
    pts = [np.asarray(waypoints[0], dtype=float)]
    for p in waypoints[1:]:
        p = np.asarray(p, dtype=float)
        if np.abs(p - pts[-1]).max() > 1e-9:
            pts.append(p)
    if len(pts) == 1:
        pts.append(pts[0].copy())
    return np.asarray(pts)


def arc_length(path):
    """Cumulative L2 joint-space distance along path (starts at 0)."""
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]


def polyline_path(waypoints, step_deg=0.25):
    """Dense straight-segment path through the waypoints (has corners)."""
    pts = _dedupe(waypoints)
    dense = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(np.degrees(np.abs(b - a).max()) / step_deg)))
        dense += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.asarray(dense)


def smooth_path(waypoints, step_deg=0.25):
    """Dense corner-free (C1) path; PCHIP is shape preserving, so a joint never leaves the range of two waypoints."""
    pts = _dedupe(waypoints)
    if len(pts) < 3:
        return polyline_path(pts, step_deg)
    u = arc_length(pts)
    travel = np.degrees(np.abs(np.diff(pts, axis=0)).max(axis=1)).sum()
    return PchipInterpolator(u, pts, axis=0)(np.linspace(0.0, u[-1], max(2, int(np.ceil(travel / step_deg)) + 1)))


def time_parameterize(path, max_speed, max_accel, dt, smooth_time=0.4):
    """Joint positions every dt along path within joint speed (rad/s) / accel (rad/s^2) limits: per-sample path speed
    limited by the fastest joint and by centripetal acceleration at bends, forward/backward passes for the accel
    limit, then a smooth_time moving average (jerk limit); starts and ends at rest."""
    path = np.asarray(path, dtype=float)
    s = arc_length(path)
    if s[-1] < 1e-9:
        return np.repeat(path[:1], 2, axis=0)
    eps = 1e-9
    d1 = np.gradient(path, s, axis=0)
    d2 = np.gradient(d1, s, axis=0)
    v_ref = max_speed / max(float(np.abs(d1).max()), eps)
    v_lim = np.minimum((max_speed / np.maximum(np.abs(d1), eps)).min(axis=1),
                       np.sqrt(max_accel / np.maximum(np.abs(d2), eps)).min(axis=1))
    sd2_cap = np.maximum(v_lim, 0.01 * v_ref) ** 2
    floor = (0.005 * v_ref) ** 2                    # never stall completely

    def accel_range(i, sd2):
        b, c = d1[i], d2[i] * sd2
        ok = np.abs(b) > eps
        if not ok.any():
            return -np.inf, np.inf
        r1, r2 = (-max_accel - c[ok]) / b[ok], (max_accel - c[ok]) / b[ok]
        return float(np.minimum(r1, r2).max()), float(np.maximum(r1, r2).min())

    n = len(s)
    sd2 = np.zeros(n)
    sd2[0] = floor
    for i in range(n - 1):
        _, hi = accel_range(i, sd2[i])
        sd2[i + 1] = min(sd2_cap[i + 1], max(sd2[i] + 2 * (s[i + 1] - s[i]) * hi, floor))
    sd2[-1] = min(sd2[-1], floor)
    for i in range(n - 2, -1, -1):
        lo, _ = accel_range(i + 1, sd2[i + 1])
        sd2[i] = min(sd2[i], max(sd2[i + 1] - 2 * (s[i + 1] - s[i]) * lo, floor))
    sd = np.sqrt(np.maximum(sd2, floor))
    t = np.r_[0.0, np.cumsum(2 * np.diff(s) / (sd[:-1] + sd[1:]))]
    grid = np.arange(0.0, t[-1] + dt, dt)
    frames = np.column_stack([np.interp(grid, t, path[:, j]) for j in range(path.shape[1])])
    k = int(round(smooth_time / dt))
    if k > 1:                                       # trailing average, ends held at rest
        padded = np.vstack([np.repeat(frames[:1], k - 1, axis=0), frames, np.repeat(frames[-1:], k, axis=0)])
        frames = np.column_stack([np.convolve(padded[:, j], np.ones(k) / k, mode='valid') for j in range(path.shape[1])])
        moving = np.abs(np.diff(frames, axis=0)).max(axis=1) > 1e-7
        frames = frames[:(int(np.nonzero(moving)[0].max()) + 2 if moving.any() else 2)]
    frames[-1] = path[-1]
    return frames


def joint_limits_check(frames, dt, window=0.2):
    """(max joint speed rad/s, max joint accel rad/s^2), accel measured over `window` seconds."""
    v = np.diff(frames, axis=0) / dt
    k = max(1, int(round(window / dt)))
    if len(v) <= k:
        return float(np.abs(v).max()), 0.0
    smooth = np.apply_along_axis(lambda x: np.convolve(x, np.ones(k) / k, mode='valid'), 0, v)
    return float(np.abs(v).max()), float(np.abs(np.diff(smooth, axis=0) / dt).max())


# ================================================================ CLI

def reach_map(a):
    """Can the arm reach the cube if the camera-based position is a few cm off?  IK + joint limits + cell clearance
    over a +-span grid (`.` ok, `C` too close to the cell, `L` no IK / joint limit) per pregrasp height."""
    import time
    from calib_core import latest_t_base_cam
    if a.centre:
        centre, src = np.array(a.centre), 'command line'
    else:
        plans = sorted((DATA / 'real_pipeline').glob('*/plan_*.json')) + sorted((DATA / 'real_dryrun').glob('*/plan_*.json'))
        plans = [p for p in plans if load_json(p).get('cube_centre_m')]
        if not plans:
            raise SystemExit('no plan with a cube centre found: pass --centre X Y Z')
        centre, src = np.array(load_json(plans[-1])['cube_centre_m']), str(plans[-1])
    pl = C['planner']
    kin = Kinematics()
    checker, _ = build_checker(kin, load_json(latest_t_base_cam())['T_base_cam'], pl['margin_m'],
                               pl['tool_length_m'], pl['tool_radius_m'])
    lo, hi = np.array(kin.limits_deg()).T
    offsets = np.round(np.arange(-a.span, a.span + 1e-9, a.step), 4)
    out = DATA / 'real_dryrun' / f'ik_reach_{time.strftime("%Y%m%d_%H%M%S")}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = ['height_m,dx_m,dy_m,x_m,y_m,status,clearance_mm,j1,j2,j3,j4,j5,j6']
    print(f'centre {np.round(centre, 3).tolist()} m (from {src}); window +-{a.span} m, step {a.step} m, margin {pl["margin_m"] * 1000:.0f} mm\n')
    for h in a.heights:
        print(f'pregrasp {h * 1000:.0f} mm above the cube centre   (columns dx, rows dy, +y up)')
        print('        ' + ' '.join(f'{dx * 100:+5.0f}' for dx in offsets))
        counts = dict(ok=0, C=0, L=0)
        for dy in offsets[::-1]:
            line = []
            for dx in offsets:
                c = centre + [dx, dy, 0.0]
                status, clr, q = 'L', np.nan, None
                try:
                    q = np.degrees(grasp_frames(kin, c, 0.0, pl['tool_length_m'], h)[3])
                    clr = np.inf if a.ik_only else checker.clearance(np.radians(q))
                    status = 'L' if not ((q > lo + 2) & (q < hi - 2)).all() else ('C' if clr < checker.margin else 'ok')
                except RuntimeError:
                    status = 'L'
                counts[status] += 1
                line.append(' . ' if status == 'ok' else f' {status} ')
                rows.append(f'{h},{dx},{dy},{c[0]:.4f},{c[1]:.4f},{status},{clr * 1000 if np.isfinite(clr) else ""},'
                            + (','.join(f'{v:.2f}' for v in q) if q is not None else ',,,,,'))
            print(f'  {dy * 100:+5.0f}  ' + ' '.join(f'{x:>5}' for x in line))
        print(f'  -> ok {counts["ok"]}, too close to cell (C) {counts["C"]}, no IK / joint limit (L) {counts["L"]}\n')
    out.write_text('\n'.join(rows) + '\n')
    print(f'details: {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=('reach', 'points'))
    p.add_argument('--centre', type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'), help='reach: cube centre, base frame')
    p.add_argument('--span', type=float, default=0.20)
    p.add_argument('--step', type=float, default=0.05)
    p.add_argument('--heights', type=float, nargs='+', default=(0.10, 0.15))
    p.add_argument('--ik_only', action='store_true', help='reach: ignore the work cell')
    args = p.parse_args()
    if args.mode == 'reach':
        reach_map(args)
    else:
        POINT_CACHE.unlink(missing_ok=True)
        print('wrote', POINT_CACHE, {k: v.shape for k, v in load_link_points().items()})
