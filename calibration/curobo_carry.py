"""(10/20 보스: cuRobo first, RRT second; opt-in) the basket carry from cuRobo V2, checked here with the planner's own
checkers.  Off unless CARRY_PLANNER=curobo (environment) or plate_ik_bc.basket_carry_planner: curobo; without it the
carry is exactly the RRT one.  cuRobo runs in its own env (.envs/curobo-bm-20261020, torch + GPU) as one worker process
per pipeline run (calibration/curobo_worker.py), started at the first carry: ~5 s, then ~0.2 s per carry.

Model (experiments/curobo_bm/README.md, 133-plan stand sweep): arm = 24 spheres per moving link from our checker's mesh
samples + the gripper cylinder; held plate = spheres on link_6; obstacles = the held checker boxes grown by part of the
checker margin (frame 0.5, table/basket 1.0); the basket floor = a cost on the plate spheres only.  Any returned path
must still pass MultiChecker(held + FloorCheck) every 0.5 deg and the wrist limit in plate_plans, else the RRT runs."""
import atexit
import json
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PLATE_R = 0.134                                       # as plate_plans.PLATE_R (2.5 cm thick disc)
_WORKER = None


def enabled(q):
    return (os.environ.get('CARRY_PLANNER') or q.get('basket_carry_planner', 'rrt')) == 'curobo'


def _fit(points, n, seed=0):
    from scipy.cluster.vq import kmeans2
    centres, label = kmeans2(points, n, seed=seed, minit='++')
    out = []
    for k in range(n):
        member = points[label == k]
        if len(member):
            c = member.mean(axis=0)
            out.append(c.tolist() + [float(np.linalg.norm(member - c, axis=1).max())])
    return out


def robot_spheres(q, per_link=24):
    """Moving links as our checker samples them (link_3..6) + the gripper cylinder along link_6 +z."""
    from robot_model import STATIC_LINKS, load_link_points
    sp = {link: _fit(pts, per_link) for link, pts in load_link_points().items() if link not in STATIC_LINKS}
    L, r = q.get('tool_length_m', 0.165), q.get('tool_radius_m', 0.06)
    sp['link_6'] += [[0.0, 0.0, float(z), float(r)] for z in np.arange(0.0, L + 1e-9, r * 0.6)]
    return sp


def plate_spheres(H, tool, r=0.0135, step=0.018):
    """Held plate as rings of spheres in the link_6 frame; the rim ring reaches PLATE_R."""
    pts = [[0.0, 0.0]]
    for ring in (PLATE_R - r, 0.085, 0.045):
        m = max(6, int(np.ceil(2 * np.pi * ring / step)))
        a = np.linspace(0, 2 * np.pi, m, endpoint=False)
        pts += np.c_[ring * np.cos(a), ring * np.sin(a)].tolist()
    P = np.c_[np.asarray(pts), np.full(len(pts), 0.0025), np.ones(len(pts))] @ (np.asarray(tool) @ np.asarray(H)).T
    return [p[:3].tolist() + [r] for p in P]


def cuboids(held, grow):
    """The held checkers' boxes, each grown by grow[i] x that checker's margin (cuRobo then needs distance > 0)."""
    from scipy.spatial.transform import Rotation
    out = []
    for i, ch in enumerate(held.checkers):
        g = grow[i] if i < len(grow) else 1.0
        for b in ch.boxes:
            R = np.asarray(b[3]) if len(b) > 3 else np.eye(3)
            x, y, z, w = Rotation.from_matrix(R).as_quat()
            out.append(dict(name=f'{i}_{b[0]}', dims=(np.asarray(b[2], float) + 2 * g * ch.margin).tolist(),
                            pose=[float(v) for v in b[1]] + [float(w), float(x), float(y), float(z)]))
    return out


class _Worker:
    def __init__(self, q):
        cfg = q.get('basket_curobo', {}) or {}
        py = Path(cfg.get('python', ROOT / '.envs/curobo-bm-20261020/bin/python'))
        if not py.is_file():
            raise RuntimeError(f'cuRobo env missing: {py} (experiments/curobo_bm/setup_env.sh)')
        log = open(ROOT / 'calibration' / 'data' / 'curobo_worker.log', 'a')
        self.p = subprocess.Popen([str(py), '-u', str(Path(__file__).with_name('curobo_worker.py'))], cwd=ROOT,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        atexit.register(self.close)
        from robot_model import DEFAULT_URDF
        r = self.ask(dict(cmd='init', urdf=str(DEFAULT_URDF), spheres=robot_spheres(q),
                          activation_m=cfg.get('activation_m', 0.05), attempts=cfg.get('attempts', 5)))
        if not r.get('ok'):
            raise RuntimeError(f'cuRobo init: {r.get("why")}')

    def ask(self, m):
        if self.p.poll() is not None:
            raise RuntimeError(f'cuRobo worker exited ({self.p.returncode}), see calibration/data/curobo_worker.log')
        self.p.stdin.write(json.dumps(m) + '\n')
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError('cuRobo worker gave no answer, see calibration/data/curobo_worker.log')
        return json.loads(line)

    def close(self):
        if self.p.poll() is None:
            self.p.stdin.close()
            try:
                self.p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.p.kill()


def carry(q, held, H, tool, Tbb, z_floor, half, start, goal):
    """Joint path start -> goal (rad) from cuRobo, or RuntimeError (the caller then runs the RRT)."""
    global _WORKER
    if _WORKER is None:
        _WORKER = _Worker(q)
    cfg = q.get('basket_curobo', {}) or {}
    r = _WORKER.ask(dict(cmd='plan', start=np.asarray(start, float).tolist(), goal=np.asarray(goal, float).tolist(),
                         cuboids=cuboids(held, cfg.get('grow', (0.5, 1.0, 1.0))), plate=plate_spheres(H, tool),
                         floor=dict(T_basket_inv=np.linalg.inv(np.asarray(Tbb, float)).tolist(),
                                    half=np.asarray(half, float).tolist(), z=float(z_floor))))
    if not r.get('ok'):
        raise RuntimeError(f'curobo: {r.get("why")}')
    path = np.asarray(r['path'], float)
    gap = max(np.abs(path[0] - start).max(), np.abs(path[-1] - goal).max())
    if gap > np.radians(0.5):
        raise RuntimeError(f'curobo: ends off by {np.degrees(gap):.2f} deg')
    path[0], path[-1] = start, goal                   # float32 ends -> the exact joints (checked above)
    return [p for p in path], r['s']
