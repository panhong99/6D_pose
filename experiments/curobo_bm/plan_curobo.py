"""(10/20 보스) cuRobo V2 carry planning on the scenes written by scene.py export; run in .envs/curobo-bm-20261020.

    python plan_curobo.py OUT_DIR            -> OUT_DIR/curobo_paths.json (path per job, solve time, GPU memory)
    python plan_curobo.py OUT_DIR --batch    -> OUT_DIR/batch_scaling.json: N problems with N different worlds in one
                                                call (BatchMotionPlanner, multi_env) for N = 1..128: time + GPU memory

Same start/goal joints as our RRT carry (lift end -> top of the vertical drop).  The M1013 moving links + gripper
cylinder are spheres (robot_spheres.json), the held plate is spheres on an `attached_object` link under link_6, the
cell/basket/markers/standing plates are cuboids grown by the checker margins and the floor over the basket is a box.
scene.py gate then judges every path with our own checkers.  Offline only."""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
URDF = Path('/home/pan/pan/doosan_ws/install/dsr_description2/share/dsr_description2/urdf/m1013.urdf')
JOINTS = [f'joint_{i}' for i in range(1, 7)]
PLATE_SLOTS = 100
ACT = float(next((a.split('=')[1] for a in sys.argv if a.startswith('--act=')), 0.005))   # collision cost starts here (m)


def bare_urdf(out):
    """URDF without visual/collision meshes: cuRobo only needs the joint tree (spheres come from robot_spheres.json)."""
    text = re.sub(r'<(visual|collision)>.*?</\1>', '', URDF.read_text(), flags=re.S)
    p = out / 'm1013_bare.urdf'
    p.write_text(text)
    return p


def robot_cfg(out, spheres):
    links = list(spheres) + ['attached_object']
    kin = dict(
        format_version=2.0, urdf_path=str(bare_urdf(out)), asset_root_path=str(out), base_link='base_link',
        tool_frames=['link_6'], collision_link_names=links, mesh_link_names=[], lock_joints=None,
        collision_sphere_buffer=0.0,
        collision_spheres={k: [dict(center=s[:3], radius=s[3]) for s in v] for k, v in spheres.items()},
        extra_collision_spheres=dict(attached_object=PLATE_SLOTS),
        extra_links=dict(attached_object=dict(fixed_transform=[0, 0, 0, 1, 0, 0, 0], joint_name='attach_joint',
                                              joint_type='FIXED', link_name='attached_object',
                                              parent_link_name='link_6')),
        self_collision_buffer={k: 0.0 for k in links}, self_collision_ignore={k: [o for o in links if o != k and {k, o} != {'link_3', 'attached_object'}] for k in links},
        # ^ our checker has no self check; cuRobo needs one pair: the held plate vs the forearm (a real contact anyway)
        cspace=dict(joint_names=JOINTS, default_joint_position=[0.0, 0.0, 1.5708, 0.0, 1.5708, 0.0],
                    cspace_distance_weight=[1.0] * 6, null_space_weight=[1.0] * 6,
                    max_acceleration=15.0, max_jerk=500.0, position_limit_clip=0.0))
    return dict(robot_cfg=dict(kinematics=kin))


def main(out):
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.scene import Cuboid, Scene
    from curobo.types import JointState
    out = Path(out).resolve()
    spheres = json.loads((out / 'robot_spheres.json').read_text())
    scenes = json.loads((out / 'scenes.json').read_text())['scenes']
    n_box = max(len(s['cuboids']) for s in scenes)
    t0 = time.perf_counter()
    cfg = MotionPlannerCfg.create(robot=robot_cfg(out, spheres), collision_cache={'cuboid': n_box + 4},
                                  self_collision_check=False, optimizer_collision_activation_distance=ACT)
    planner = MotionPlanner(cfg)
    planner.warmup(enable_graph=True, num_warmup_iterations=2)
    setup_s = time.perf_counter() - t0
    dev = planner.default_joint_state.position.device
    results = []
    torch.cuda.reset_peak_memory_stats()
    for s in scenes:
        planner.update_world(Scene(cuboid=[Cuboid(name=c['name'], dims=c['dims'], pose=c['pose'])
                                           for c in s['cuboids']]))
        start = JointState.from_position(torch.tensor([s['start']], device=dev, dtype=torch.float32), joint_names=JOINTS)
        goal = JointState.from_position(torch.tensor([s['goal']], device=dev, dtype=torch.float32), joint_names=JOINTS)
        plate = torch.tensor(s['plate_spheres'], device=dev, dtype=torch.float32)
        planner.attachment_manager.update(plate, start, link_name='attached_object')
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = planner.plan_cspace(goal, start, max_attempts=5, enable_graph_attempt=1)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t
        ok = r is not None and bool(r.success.any())
        row = dict(job=s['job'], success=ok, solve_s=wall,
                   status=None if r is None else str(getattr(r, 'status', None)))
        if ok:
            pos = r.js_solution.position
            idx = int(torch.nonzero(r.success.reshape(-1))[0])
            path = pos.reshape(-1, pos.shape[-2], pos.shape[-1])[idx].cpu().numpy().astype(float)
            row['end_err_rad'] = float(max(np.abs(path[0] - s['start']).max(), np.abs(path[-1] - s['goal']).max()))
            path[0], path[-1] = s['start'], s['goal']      # float32 ends -> the exact joints; the gap is reported above
            row['path'] = path.tolist()
        results.append(row)
        print(s['job'], ok, f'{wall:.3f}s', file=sys.stderr, flush=True)
    (out / 'curobo_paths.json').write_text(json.dumps(dict(
        setup_s=setup_s, activation_m=ACT, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
        peak_gpu_mb=torch.cuda.max_memory_allocated() / 2 ** 20, results=results)))
    print(f'{sum(r["success"] for r in results)}/{len(results)} setup {setup_s:.1f}s', file=sys.stderr)


def batch_scaling(out, sizes=(1, 4, 16, 32, 64, 128)):
    """(보스: parallel envs in sim training -> GPU memory per env?) one process, one robot model, N worlds."""
    from curobo.batch_motion_planner import BatchMotionPlanner, MotionPlannerCfg
    from curobo.types import JointState
    out = Path(out).resolve()
    spheres = json.loads((out / 'robot_spheres.json').read_text())
    scenes = json.loads((out / 'scenes.json').read_text())['scenes']
    n_box = max(len(s['cuboids']) for s in scenes)
    rows = []
    for n in sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        batch = [scenes[i % len(scenes)] for i in range(n)]
        worlds = [dict(cuboid={c['name']: dict(dims=c['dims'], pose=c['pose']) for c in s['cuboids']}) for s in batch]
        cfg = MotionPlannerCfg.create(robot=robot_cfg(out, spheres), scene_model=worlds,   # one world per env
                                      collision_cache={'cuboid': n_box + 4},
                                      self_collision_check=False, optimizer_collision_activation_distance=ACT,
                                      max_batch_size=n, multi_env=True)
        planner = BatchMotionPlanner(cfg)
        dev = planner.default_joint_state.position.device
        start = JointState.from_position(torch.tensor([s['start'] for s in batch], device=dev, dtype=torch.float32),
                                         joint_names=JOINTS)
        goal = JointState.from_position(torch.tensor([s['goal'] for s in batch], device=dev, dtype=torch.float32),
                                        joint_names=JOINTS)
        planner.attachment_manager.update(torch.tensor(batch[0]['plate_spheres'], device=dev, dtype=torch.float32),
                                          start, link_name='attached_object')
        planner.plan_cspace(goal, start, max_attempts=1)            # warm-up (CUDA graph capture)
        setup = time.perf_counter() - t0
        times = []
        for _ in range(3):
            torch.cuda.synchronize()
            t = time.perf_counter()
            r = planner.plan_cspace(goal, start, max_attempts=1)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        solved = int(r.success.reshape(n, -1).any(dim=-1).sum()) if r is not None else 0
        rows.append(dict(n=n, solved=solved, setup_s=setup, solve_s=float(np.median(times)),
                         peak_mb=(torch.cuda.max_memory_allocated() - base) / 2 ** 20,
                         reserved_mb=torch.cuda.memory_reserved() / 2 ** 20))
        print(rows[-1], file=sys.stderr, flush=True)
        planner.destroy()
        del planner, cfg
    (out / 'batch_scaling.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(0), rows=rows), indent=1))


if __name__ == '__main__':
    batch_scaling(sys.argv[1]) if '--batch' in sys.argv else main(sys.argv[1])
