"""(10/20 보스: cuRobo first, RRT second) cuRobo V2 carry planner process, run in .envs/curobo-bm-20261020 (torch + cuRobo);
calibration/curobo_carry.py starts it and talks one JSON line per request on stdin/stdout.  Offline planning only: it
never talks to the robot.

    {"cmd": "init", "urdf": ..., "spheres": {link: [[x, y, z, r], ...]}, "activation_m": 0.05, "attempts": 5}
    {"cmd": "plan", "start": [6 rad], "goal": [6 rad], "cuboids": [{name, dims, pose[x y z qw qx qy qz]}],
     "plate": [[x, y, z, r] in link_6], "floor": {"T_basket_inv": 4x4, "half": [hx, hy], "z": m}}
 -> {"ok": true, "path": [[6 rad], ...], "s": seconds}  or  {"ok": false, "why": "..."}

The basket floor is a cost on the held-plate spheres only (FloorCheck applies to the plate, not to the arm): over the
basket footprint + pad, a plate sphere's lowest point below z costs weight * depth, in the optimizer and as a constraint
in cuRobo's feasibility checks (the graph planner's too)."""
import json
import re
import sys
import time
from pathlib import Path

import torch

JOINTS = [f'joint_{i}' for i in range(1, 7)]
PLATE_SLOTS = 100
FLOOR_WEIGHT = 1e5                                   # as trajopt's scene collision (lbfgs_bspline_trajopt.yml)
FLOOR_EDGE_M = 0.02                                  # (10/20 실기 Q4 white) a rim point in the band's corner, the spheres
                                                     # 0.2 mm outside: the floor area for the spheres is 2 cm wider
F = {}                                               # floor tensors: made once, updated in place (CUDA graphs)


def patch_floor_cost():
    from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
    orig = RobotCostManager.compute_costs

    def compute_costs(self, state, cost_collection=None, goal=None, **kw):
        cc = orig(self, state, cost_collection, goal, **kw)
        sp = state.robot_spheres
        if sp is None or 'idx' not in F:
            return cc
        p = sp[..., F['idx'], :]                                           # held plate spheres, world frame
        loc = p[..., :3] @ F['R'].T + F['t']                               # basket frame
        r = p[..., 3]
        edge = r + FLOOR_EDGE_M                                            # the spheres stop short of the rim between
        over = (loc[..., 0].abs() <= F['half'][0] + edge) & (loc[..., 1].abs() <= F['half'][1] + edge) & (r > 0)
        depth = torch.relu(F['z'] - (p[..., 2] - r)) * over
        cc.add(F['w'] * depth.sum(-1, keepdim=True), 'plate_floor')
        return cc
    RobotCostManager.compute_costs = compute_costs


def robot_cfg(urdf, spheres, folder):
    text = re.sub(r'<(visual|collision)>.*?</\1>', '', Path(urdf).read_text(), flags=re.S)
    bare = folder / 'm1013_bare.urdf'
    bare.write_text(text)
    links = list(spheres) + ['attached_object']
    keep = {'link_3', 'attached_object'}                # cuRobo needs one self pair: the held plate vs the forearm
    kin = dict(
        format_version=2.0, urdf_path=str(bare), asset_root_path=str(folder), base_link='base_link',
        tool_frames=['link_6'], collision_link_names=links, mesh_link_names=[], lock_joints=None,
        collision_sphere_buffer=0.0,
        collision_spheres={k: [dict(center=s[:3], radius=s[3]) for s in v] for k, v in spheres.items()},
        extra_collision_spheres=dict(attached_object=PLATE_SLOTS),
        extra_links=dict(attached_object=dict(fixed_transform=[0, 0, 0, 1, 0, 0, 0], joint_name='attach_joint',
                                              joint_type='FIXED', link_name='attached_object',
                                              parent_link_name='link_6')),
        self_collision_buffer={k: 0.0 for k in links},
        self_collision_ignore={k: [o for o in links if o != k and {k, o} != keep] for k in links},
        cspace=dict(joint_names=JOINTS, default_joint_position=[0.0, 0.0, 1.5708, 0.0, 1.5708, 0.0],
                    cspace_distance_weight=[1.0] * 6, null_space_weight=[1.0] * 6,
                    max_acceleration=15.0, max_jerk=500.0, position_limit_clip=0.0))
    return dict(robot_cfg=dict(kinematics=kin))


class Worker:
    def init(self, m):
        import tempfile
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        patch_floor_cost()
        self.folder = Path(tempfile.mkdtemp(prefix='curobo_worker_'))
        self.attempts = int(m.get('attempts', 5))
        cfg = MotionPlannerCfg.create(robot=robot_cfg(m['urdf'], m['spheres'], self.folder),
                                      collision_cache={'cuboid': int(m.get('max_boxes', 64))}, self_collision_check=False,
                                      optimizer_collision_activation_distance=float(m.get('activation_m', 0.05)))
        self.planner = MotionPlanner(cfg)
        dev = self.planner.default_joint_state.position.device
        kp = self.planner.kinematics.config.kinematics_config
        F.update(idx=kp.get_sphere_index_from_link_name('attached_object').to(dev),
                 R=torch.eye(3, device=dev), t=torch.zeros(3, device=dev), half=torch.zeros(2, device=dev),
                 z=torch.zeros((), device=dev), w=torch.zeros((), device=dev))
        self.planner.warmup(enable_graph=True, num_warmup_iterations=2)
        return dict(ok=True)

    def plan(self, m):
        from curobo.scene import Cuboid, Scene
        from curobo.types import JointState
        dev = self.planner.default_joint_state.position.device
        f32 = dict(device=dev, dtype=torch.float32)
        self.planner.update_world(Scene(cuboid=[Cuboid(name=c['name'], dims=c['dims'], pose=c['pose'])
                                                for c in m['cuboids']]))
        fl = m.get('floor')
        Ti = torch.tensor(fl['T_basket_inv'], **f32) if fl else torch.eye(4, **f32)
        F['R'].copy_(Ti[:3, :3])
        F['t'].copy_(Ti[:3, 3])
        F['half'].copy_(torch.tensor(fl['half'] if fl else [0.0, 0.0], **f32))
        F['z'].fill_(float(fl['z']) if fl else 0.0)
        F['w'].fill_(FLOOR_WEIGHT if fl else 0.0)
        start = JointState.from_position(torch.tensor([m['start']], **f32), joint_names=JOINTS)
        goal = JointState.from_position(torch.tensor([m['goal']], **f32), joint_names=JOINTS)
        self.planner.attachment_manager.update(torch.tensor(m['plate'], **f32), start, link_name='attached_object')
        t = time.perf_counter()
        r = self.planner.plan_cspace(goal, start, max_attempts=self.attempts, enable_graph_attempt=1)
        torch.cuda.synchronize()
        s = time.perf_counter() - t
        if r is None or not bool(r.success.any()):
            return dict(ok=False, why='no feasible trajectory (start/goal blocked or not found)', s=s)
        pos = r.js_solution.position
        i = int(torch.nonzero(r.success.reshape(-1))[0])
        path = pos.reshape(-1, pos.shape[-2], pos.shape[-1])[i].double().cpu().tolist()
        return dict(ok=True, path=path, s=s)


def main():
    w = Worker()
    out = sys.stdout
    sys.stdout = sys.stderr                              # library prints never reach the JSON channel
    for line in sys.stdin:
        if not line.strip():
            continue
        m = json.loads(line)
        try:
            res = getattr(w, m['cmd'])(m)
        except Exception as e:                           # reported to the caller, which falls back to the RRT
            res = dict(ok=False, why=f'{type(e).__name__}: {e}'[:300])
        out.write(json.dumps(res) + '\n')
        out.flush()


if __name__ == '__main__':
    main()
