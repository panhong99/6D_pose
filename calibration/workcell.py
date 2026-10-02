"""Static work cell (table, aluminium frame, D455 body) and voxel collision checks of the M1013.

All lengths are metres in the robot base frame (x toward the object, +y left, z up, robot base bottom at z = 0).
The base stands on a 20 mm steel plate, so the table top is at z = -base_plate_thickness (Human, 2026-10-02).
Boxes -> 1 cm occupancy grid -> distance field; samples on the robot's collision meshes are moved by the URDF
kinematics and looked up in the field, so a joint path can be checked before anything is sent to the controller.
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

MESH_DIR = Path('/home/pan/pan/doosan_ws/install/dsr_description2/share/dsr_description2/meshes/m1013_collision')
# Collision meshes per URDF link (from m1013.urdf <collision>); exported in millimetres.
LINK_MESHES = {
    'link_1': ['MF1013_1_0.dae'],
    'link_2': ['MF1013_2_0.dae', 'MF1013_2_1.dae', 'MF1013_2_2.dae'],
    'link_3': ['MF1013_3_0.dae'],
    'link_4': ['MF1013_4_0.dae', 'MF1013_4_1.dae'],
    'link_5': ['MF1013_5_0.dae'],
    'link_6': ['MF1013_6_0.dae'],
}
MESH_SCALE = 0.001
# link_1 only spins about the vertical axis on top of the base plate: its gap to the
# table (80 mm with the 20 mm plate under the base) never changes, and counting it would eat the keep-out margin.
# link_2 (the shoulder housing) is the same: a cylinder turning about its own axis 80 mm above the table at
# every pose (measured at the zero and approach poses), and >0.7 m from the frame, so a keep-out margin above
# 80 mm could never be met by the robot's own base.
STATIC_LINKS = ('link_1', 'link_2')
PLAN_MARGIN_FACTOR = 1.3   # plan with extra room: edges are only sampled every few degrees
# Surface samples of the meshes above, in link frames (metres).  Isaac's Python
# has no pycollada, so the samples are cached once from the conda env:
#   conda run -n foundationpose python calibration/workcell.py
POINT_CACHE = Path(__file__).resolve().parent / 'data' / 'm1013_collision_points.npz'

# Work cell.  MEASURED by Human on 2026-10-01: table 1.80 x 1.20 m and 0.10 m thick, the robot stands directly on the
# table top, uprights at x = 0.85 / 1.65 m and y = +-0.56 m (robot base centre = origin), frame height 1.20 m.
# ASSUMED (not measured): 40x40 mm profile, base 0.30 m from the table's short edge, rail layout.
# Override any key with a JSON file (data/real_cell_measured.json) through --cell_json.
FRAME_CELL = dict(
    table_size=(1.80, 1.20),
    table_thickness=0.10,
    base_plate_thickness=0.020,      # steel plate between robot base and table: table top z = -0.020 (Human 2026-10-02)
    robot_from_table_edge=0.30,      # ASSUMED
    frame_centre_xy=(1.25, 0.0),     # centre of the four uprights
    frame_size=(0.84, 1.16),         # upright centre lines 0.80 x 1.12 m + one profile width
    frame_height=1.20,
    profile=0.04,                    # ASSUMED
    collision_margin_m=0.10,         # keep-out distance around every static box
    omit_boxes=(),
)
D455_BOX = (0.029, 0.124, 0.026)     # depth x width x height of the D455 body, m


def frame_layout(cell=FRAME_CELL):
    """Static boxes of the work cell: list of (name, centre[3], size[3])."""
    tx, ty = cell['table_size']
    edge = cell['robot_from_table_edge']
    cx, cy = cell['frame_centre_xy']
    fx, fy = cell['frame_size']
    h, p = cell['frame_height'], cell['profile']
    z0 = -cell['base_plate_thickness']                     # table top in the base frame; the frame stands on the table
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
    """Surface samples + some vertices of each link's collision mesh (needs pycollada)."""
    import trimesh
    rng = np.random.default_rng(seed)
    points = {}
    for link, files in LINK_MESHES.items():
        mesh = trimesh.util.concatenate([trimesh.load(str(MESH_DIR / f), force='mesh') for f in files])
        pts, _ = trimesh.sample.sample_surface(mesh, points_per_link, seed=int(rng.integers(1 << 31)))
        verts = mesh.vertices[rng.choice(len(mesh.vertices), points_per_link // 4)]
        points[link] = np.vstack([pts, verts]) * MESH_SCALE
    return points


def load_link_points(points_per_link=600, seed=0):
    """Cached samples when they match the request, else sample them (cache only if absent)."""
    if POINT_CACHE.is_file():
        data = np.load(POINT_CACHE)
        if int(data['points_per_link']) == points_per_link and int(data['seed']) == seed:
            return {link: data[link] for link in LINK_MESHES}
    points = sample_link_points(points_per_link, seed)
    if not POINT_CACHE.is_file():  # never replace the default cache with other settings
        POINT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(POINT_CACHE, points_per_link=points_per_link, seed=seed, **points)
    return points


class CollisionChecker:
    """Distance-field check of M1013 collision-mesh samples against static boxes."""

    def __init__(self, kin, boxes, resolution=0.01, margin=None, points_per_link=600,
                 bounds=((-0.6, 1.8), (-0.9, 0.9), (-0.15, 1.7)), seed=0):
        margin = FRAME_CELL['collision_margin_m'] if margin is None else margin
        self.kin, self.margin, self.res = kin, margin, resolution
        self.plan_margin = margin * PLAN_MARGIN_FACTOR
        self.lo = np.array([b[0] for b in bounds], dtype=float)
        shape = np.ceil((np.array([b[1] for b in bounds]) - self.lo) / resolution).astype(int)
        occ = np.zeros(shape, dtype=bool)
        centres = self.lo + (np.indices(shape).transpose(1, 2, 3, 0) + 0.5) * resolution
        for _, centre, size in boxes:
            inside = np.all(np.abs(centres - np.asarray(centre)) <= np.asarray(size) / 2, axis=-1)
            occ |= inside
        self.occupancy = occ
        # Distance (m) from each free voxel centre to the nearest occupied one.
        self.distance = ndimage.distance_transform_edt(~occ, sampling=resolution).astype(np.float32)
        self.boxes = boxes
        self.link_points = {k: v for k, v in load_link_points(points_per_link, seed).items()
                            if k not in STATIC_LINKS}
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
        """Straight joint line if collision-free, else bidirectional RRT + shortcut."""
        if self._edge_free(start, goal, step_deg):
            return [start, goal], 'straight'
        rng = np.random.default_rng(seed)
        lower = np.array([j[3] for j in self.kin.joints])
        upper = np.array([j[4] for j in self.kin.joints])
        # Sample near the start/goal box (wrapped joints otherwise waste samples).
        span_lo = np.maximum(lower, np.minimum(start, goal) - np.radians(90))
        span_hi = np.minimum(upper, np.maximum(start, goal) + np.radians(90))
        trees = [[start], [goal]]
        parents = [[-1], [-1]]
        for _ in range(max_nodes):
            sample = rng.uniform(span_lo, span_hi)
            a, b = (0, 1) if len(trees[0]) <= len(trees[1]) else (1, 0)
            new = self._extend(trees[a], parents[a], sample, step_deg)
            if new is None:
                continue
            reached = self._connect(trees[b], parents[b], new, step_deg)
            if reached:
                path_a = self._trace(trees[a], parents[a], len(trees[a]) - 1)
                path_b = self._trace(trees[b], parents[b], len(trees[b]) - 1)
                path = path_a[::-1] + path_b[1:] if a == 0 else path_b[::-1] + path_a[1:]
                return self.tidy(self._shortcut(path, step_deg, rng)), 'rrt+tidy'
        raise RuntimeError('No collision-free joint path found (RRT budget exhausted)')

    def tidy(self, path, step_deg=4.0, iters=20, subdivide=4):
        """Elastic-band relaxation of a collision-free polyline.

        A sampling planner leaves random intermediate poses (wrist flicks).  Each
        interior waypoint is pulled towards the midpoint of its neighbours as far
        as both edges stay collision-free, which removes those detours while the
        clearance margin still holds.
        """
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
        """Fewest point-to-point (straight joint) moves through the planner's waypoints.

        Greedy: from the current point jump to the farthest later waypoint so that the straight joint move
        keeps ``min_clearance`` (default: the verified margin) along its whole length and no joint turns more
        than ``max_joint_deg`` in one move (small moves are easy to check by eye on the pendant).  Points that
        cannot be skipped stay, so the result is never less safe than the planner's path.
        """
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
        """Interior samples need the planning margin, the two ends only the verified one
        (a start/goal pose chosen by the caller may sit between the two)."""
        n = max(2, int(np.ceil(np.degrees(np.abs(b - a).max()) / step_deg)) + 1)
        for k, t in enumerate(np.linspace(0, 1, n)):
            need = self.margin if k in (0, n - 1) else self.plan_margin
            if self.clearance(a + (b - a) * t) < need:
                return False
        return True

    def _extend(self, tree, parents, target, step_deg, max_step_deg=10.0):
        dists = [np.abs(n - target).max() for n in tree]
        i = int(np.argmin(dists))
        near = tree[i]
        delta = target - near
        scale = min(1.0, np.radians(max_step_deg) / max(np.abs(delta).max(), 1e-9))
        new = near + delta * scale
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
        """Occupancy grid + box list for reuse on the real cell (np.load / json)."""
        path = Path(path)
        np.savez_compressed(path, occupancy=self.occupancy, origin=self.lo, resolution=self.res,
                            margin=self.margin)
        path.with_suffix('.json').write_text(json.dumps(
            dict(frame='robot base, metres, z up', resolution=self.res, origin=self.lo.tolist(),
                 shape=list(self.occupancy.shape), margin=self.margin,
                 boxes=[dict(name=n, centre=list(map(float, c)), size=list(map(float, s)))
                        for n, c, s in self.boxes]), indent=1))
