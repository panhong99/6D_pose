"""(10/20 보스: cuRobo vs our RRT vs PyRoki) scene export and exact gate, run in the foundationpose env (no torch/JAX).

    export SWEEP_DIR OUT_DIR   -> OUT_DIR/robot_spheres.json + OUT_DIR/scenes.json (one entry per planned sweep job:
                                  start/goal joints of the carry, inflated world boxes, plate spheres, legacy RRT metrics)
    gate   OUT_DIR             -> reads OUT_DIR/curobo_paths.json, checks every path with the planner's own checkers
                                  (MultiChecker held + FloorCheck, joint limits, J4/J6 excursion), writes OUT_DIR/table.jsonl

The scene is rebuilt like experiments/pyroki_bm/benchmark_saved.py (same camera, boxes, floor) so the three planners
see the same problem.  Offline only: nothing here talks to the robot."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'calibration')]
from calib_core import latest_t_base_cam, load_json  # noqa: E402
from plate_plans import C, PLATE_R, FloorCheck, basket_boxes, pin_gap, plate_cloud, quadrant_checkers  # noqa: E402
from robot_model import Kinematics, load_link_points, STATIC_LINKS  # noqa: E402


# ---------------------------------------------------------------- spheres

def fit_spheres(points, n, seed=0):
    """Cover a link's surface samples with n spheres: k-means centres, radius = farthest member (conservative)."""
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
    """Moving links as our checker sees them (link_3..6 mesh samples) + the gripper cylinder along link_6 +z."""
    spheres = {link: fit_spheres(pts, per_link) for link, pts in load_link_points().items() if link not in STATIC_LINKS}
    L, r = q.get('tool_length_m', 0.165), q.get('tool_radius_m', 0.06)
    spheres['link_6'] += [[0.0, 0.0, float(z), float(r)] for z in np.arange(0.0, L + 1e-9, r * 0.6)]
    return spheres


def plate_spheres(H, tool, r=0.0135, step=0.018):
    """Held plate (2.5 cm thick disc) as rings of spheres in the link_6 frame; the rim ring reaches PLATE_R."""
    pts = [[0.0, 0.0]]
    for ring in (PLATE_R - r, 0.085, 0.045):
        m = max(6, int(np.ceil(2 * np.pi * ring / step)))
        a = np.linspace(0, 2 * np.pi, m, endpoint=False)
        pts += np.c_[ring * np.cos(a), ring * np.sin(a)].tolist()
    P = np.c_[np.asarray(pts), np.full(len(pts), 0.0025), np.ones(len(pts))] @ (tool @ H).T
    return [p[:3].tolist() + [r] for p in P]


# ---------------------------------------------------------------- scene per saved plan

def build(plan_path, standing):
    source = json.loads(Path(plan_path).read_text())
    q = {**C['plate_rl'], **C['plate_ik_bc']}
    basket = np.asarray(source['basket_T'], float)
    plates = [(qd, pin_gap(basket, qd, int(pin))) for qd, pin in (s.split(':') for s in standing)]
    H = np.asarray(source['plate_in_tcp'], float)
    tool = np.eye(4)
    tool[2, 3] = q.get('tool_length_m', 0.165)
    Tbc = np.asarray(load_json(latest_t_base_cam())['T_base_cam'], float)
    held, _, _ = quadrant_checkers(Tbc, q, basket, H, tool, plates=plates)
    cloud = plate_cloud(H, tool)
    bb = basket_boxes(basket, plates=plates)
    z_floor = max(c[2] + s[2] / 2 for n, c, s, _ in bb if n.startswith('marker_')) + q.get('basket_carry_clear_m', 0.05)
    quadrant = source.get('quadrant') or source['quadrant_plan'].get('quadrant')
    if plates and quadrant in q.get('basket_carry_rrt_over_plates', ()):
        z_floor = max(z_floor, max(c[2] + s[2] / 2 for n, c, s, _ in bb if n.startswith('plate_'))
                      + q.get('basket_carry_plate_clear_m', 0.05))
    if plates:
        z_floor += (q.get('basket_carry_rrt_extra_m', {}) or {}).get(quadrant, 0.0)
    floor = FloorCheck(Kinematics(), cloud, basket, z_floor)
    z_bottom = min(c[2] - s[2] / 2 for n, c, s, _ in bb if n.startswith('wall_'))
    saved = np.radians(np.asarray(source['legs']['to_above']['waypoints_deg'], float))
    k = int(source['quadrant_plan']['drop_start'])
    return dict(source=source, q=q, held=held, floor=floor, basket=basket, H=H, tool=tool, z_floor=z_floor,
                carry=saved[:k + 1], quadrant=quadrant, z_bottom=z_bottom)


def world_cuboids(sc, grow=(0.5, 1.0, 1.0), pad=0.0):
    """Every held-checker box, grown by that checker's margin (cuRobo then needs distance > 0), plus the floor over the
    basket as a box up to z_floor (it limits the plate; the gripper is above the plate there).  grow: fraction of the
    margin added to each box, per held checker (frame, table, basket); pad: floor box beyond the basket footprint (FloorCheck pads 8 cm, for the plate only)."""
    from scipy.spatial.transform import Rotation
    out = []
    for i, ch in enumerate(sc['held'].checkers):
        for b in ch.boxes:
            R = np.asarray(b[3]) if len(b) > 3 else np.eye(3)
            x, y, z, w = Rotation.from_matrix(R).as_quat()
            out.append(dict(name=f'{i}_{b[0]}', dims=(np.asarray(b[2], float) + 2 * grow[i] * ch.margin).tolist(),
                            pose=list(map(float, b[1])) + [float(w), float(x), float(y), float(z)]))
    T = sc['basket']
    half = sc['floor'].half - 0.08 + pad
    z0 = sc['z_bottom']
    top = sc['z_floor'] - 0.003                           # the carry ends right at the floor (drop start): leave it free
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    out.append(dict(name='floor_over_basket', dims=[2 * half[0], 2 * half[1], top - z0],
                    pose=T[:3, 3].tolist()[:2] + [(top + z0) / 2, float(w), float(x), float(y), float(z)]))
    return out


# ---------------------------------------------------------------- exact gate (same rules as pyroki_bm.trajopt_backend.validate)

def densify(path, step_deg):
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.degrees(np.max(np.abs(b - a))) / step_deg)))
        out.extend(a + (b - a) * (i / n) for i in range(1, n + 1))
    return np.asarray(out)


def resample(path, n=101):
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    keep = np.r_[True, np.diff(s) > 1e-12]
    if s[-1] <= 1e-12:
        return np.repeat(path[:1], n, axis=0)
    return np.stack([np.interp(np.linspace(0, s[-1], n), s[keep], path[keep, i]) for i in range(6)], axis=1)


def metrics(path, sc, max_wrist_deg=100.0):
    path = np.asarray(path, float)
    kin = sc['floor'].kin
    lo = np.array([j[3] for j in kin.joints])
    hi = np.array([j[4] for j in kin.joints])
    d2 = densify(path, 2.0)
    start = sc['carry'][0]
    wrist = float(np.max(np.degrees(np.abs(d2[:, [3, 5]] - start[[3, 5]]))))
    held = min(float(sc['held'].clearance(x)) for x in d2)
    fl2 = min(float(sc['floor'].clearance(x)) for x in d2)
    fl05 = min(float(sc['floor'].clearance(x)) for x in densify(path, 0.5))
    ends = float(max(np.abs(path[0] - start).max(), np.abs(path[-1] - sc['carry'][-1]).max()))
    within = bool(np.all(d2 >= lo - 1e-9) and np.all(d2 <= hi + 1e-9))
    sm = resample(path)
    return dict(valid=bool(within and wrist <= max_wrist_deg and held >= 1 and fl2 >= 1 and ends < 1e-3),
                floor_dense_ok=bool(fl05 >= 1), within_limits=within, max_wrist_deg=wrist, min_held_clearance=held,
                min_floor_clearance=fl2, min_floor_clearance_05deg=fl05, end_error_rad=ends,
                joint_travel_deg=float(np.degrees(np.linalg.norm(np.diff(path, axis=0), axis=1)).sum()),
                acceleration_l2_101=float(np.linalg.norm(np.diff(sm, n=2, axis=0))),
                jerk_l2_101=float(np.linalg.norm(np.diff(sm, n=3, axis=0))), points=len(path))


def jobs(sweep):
    for j in map(json.loads, open(Path(sweep) / 'results.jsonl')):
        if j.get('ok') and j.get('plan'):
            yield f"{j['quadrant']}_{j['region']}_{j['r']:.2f}_{j['az']:.0f}", j


def export(sweep, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    q = {**C['plate_rl'], **C['plate_ik_bc']}
    (out / 'robot_spheres.json').write_text(json.dumps(robot_spheres(q)))
    scenes = []
    for tag, j in jobs(sweep):
        sc = build(ROOT / j['plan'], j.get('standing', []))
        scenes.append(dict(job=tag, plan=j['plan'], standing=j.get('standing', []),
                           start=sc['carry'][0].tolist(), goal=sc['carry'][-1].tolist(),
                           cuboids=world_cuboids(sc), plate_spheres=plate_spheres(sc['H'], sc['tool']),
                           legacy=metrics(sc['carry'], sc)))
        print(tag, scenes[-1]['legacy']['valid'], file=sys.stderr)
    (out / 'scenes.json').write_text(json.dumps(dict(sweep=str(sweep), scenes=scenes)))


def gate(out):
    out = Path(out)
    meta = {s['job']: s for s in json.loads((out / 'scenes.json').read_text())['scenes']}
    res = json.loads((out / 'curobo_paths.json').read_text())
    with open(out / 'table.jsonl', 'w') as f:
        for r in res['results']:
            s = meta[r['job']]
            row = dict(job=r['job'], legacy=s['legacy'], curobo_s=r['solve_s'], curobo_success=r['success'],
                       curobo_status=r.get('status'))
            if r['success']:
                sc = build(ROOT / s['plan'], s['standing'])
                row['curobo'] = metrics(np.asarray(r['path']), sc)
            f.write(json.dumps(row) + '\n')
            print(r['job'], r['success'], row.get('curobo', {}).get('valid'), file=sys.stderr)


if __name__ == '__main__':
    {'export': lambda: export(sys.argv[2], sys.argv[3]), 'gate': lambda: gate(sys.argv[2])}[sys.argv[1]]()
