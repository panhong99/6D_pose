"""Basket insertion in Isaac (2026-10-13), Isaac Lab python:

    ~/IsaacLab/isaaclab.sh -p rl/basket/expert.py replay --headless          P1: the real runs of 10/12 in sim
    ~/IsaacLab/isaaclab.sh -p rl/basket/expert.py collect --headless --out IL/data_basket_v2_<date> [--dart] [--batches N]

replay: every labelled real run (calibration/data/real_pipeline/labels_*.jsonl) is rebuilt in sim: the basket at its
  marker pose, the plate where FoundationPose saw it, and the robot driven through the run's own joint waypoints
  (grasp plan + basket plan legs, gripper events in between).  Several envs per run with a small plate jitter.
  The sim result (inserted or not) is compared with the real label -> does the sim reproduce the real robot?
collect: (added next) feedback expert in the slot frame -> demos for train.py bc.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RL = HERE.parent
ROOT = RL.parent
sys.path[:0] = [str(HERE), str(RL)]


def parse_args():
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=('replay', 'collect', 'realplan'))
    p.add_argument('--labels', default=str(ROOT / 'calibration/data/real_pipeline/labels_20261012.jsonl'))
    p.add_argument('--plans', default=None, help='replay: jsonl of planned setups {name, plate, Tb, plan} (10/13 quadrant plans: '
                   'one plan.json with to_grasp/lift/to_above/descend/ascend) instead of real runs')
    p.add_argument('--per_run', type=int, default=16, help='replay: envs per real run')
    p.add_argument('--jitter_mm', type=float, default=2.0, help='replay: plate xy jitter (FP noise)')
    p.add_argument('--speed', type=float, default=20.0, help='sim joint speed for transport legs (deg/s)')
    p.add_argument('--out', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--save_starts', action='store_true', help='collect: stop at `above` and save RL start states')
    p.add_argument('--req', default=None, help='realplan: request json')
    p.add_argument('--residual', default=None, help='realplan: residual RL checkpoint on top of --policy (BC)')
    p.add_argument('--policy', default=None, help='collect: a BC checkpoint drives instead of the expert (evaluation)')
    p.add_argument('--attach', action='store_true', help='replay: rigid in-hand plate after the lift')
    p.add_argument('--rot_sign', type=float, default=1.0)
    p.add_argument('--batches', type=int, default=1, help='collect: batches of config expert.envs envs')
    p.add_argument('--dart', action='store_true', help='collect: noisy execution, clean labels')
    p.add_argument('--episode_s', type=float, default=30.0, help='collect: insertion time limit')
    p.add_argument('--library', default=str(ROOT / 'calibration/data/plan_library/20261008_174557_380970/library.npz'))
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


# ---------------------------------------------------------------- shared sim helpers
def sim_steps(env, k=None):
    for _ in range(k or env.cfg.decimation):
        attach_update(env)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
    env.sim.render()


def attach(env, mask, noise_mm=0.0, noise_deg=0.0, seed=0):
    """(10/12 night) rigid in-hand plate: the sim pinch let the plate swing like a pendulum (tool turned, plate did
    not; the real one follows).  From now on the plate pose = TCP pose x H, H = the in-hand pose measured now, perturbed
    by +-noise (the real FP / grasp error).  attach_update() applies it every physics step; release with detach()."""
    import math
    import torch
    import plate_env as pe
    from isaaclab.utils.math import quat_apply, quat_conjugate, quat_from_angle_axis, quat_mul
    tp, tq = pe.tcp_pose(env)
    pl = env.scene['plate']
    qc = quat_conjugate(tq)
    rel_p = quat_apply(qc, pl.data.root_pos_w - tp)
    rel_q = quat_mul(qc, pl.data.root_quat_w)
    g = torch.Generator(device=env.device).manual_seed(seed)
    n = env.num_envs
    if noise_deg > 0:
        ax = torch.randn(n, 3, generator=g, device=env.device)
        ax = ax / ax.norm(dim=1, keepdim=True)
        ang = (torch.rand(n, generator=g, device=env.device) * 2 - 1) * math.radians(noise_deg)
        rel_q = quat_mul(rel_q, quat_from_angle_axis(ang, ax))
    if noise_mm > 0:
        rel_p = rel_p + (torch.rand(n, 3, generator=g, device=env.device) * 2 - 1) * noise_mm / 1000
    env._attach = dict(on=torch.as_tensor(mask, device=env.device).clone(), p=rel_p, q=rel_q)
    attach_update(env)


def detach(env, mask):
    if hasattr(env, '_attach'):
        env._attach['on'] &= ~mask


def attach_update(env):
    a = getattr(env, '_attach', None)
    if a is None or not bool(a['on'].any()):
        return
    import torch
    import plate_env as pe
    from isaaclab.utils.math import quat_apply, quat_mul
    tp, tq = pe.tcp_pose(env)
    pl = env.scene['plate']
    pose = torch.cat([tp + quat_apply(tq, a['p']), quat_mul(tq, a['q'])], 1)
    ids = torch.nonzero(a['on']).squeeze(1)
    pl.write_root_pose_to_sim(pose[ids], env_ids=ids)
    pl.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), env_ids=ids)


def set_basket(env, T_list, ids=None):
    """Basket of env ids[i] (default: all) at the base-frame pose T_list[i] (marker frame, 4x4); env._slot follows."""
    import math
    import numpy as np
    import torch
    import basket_env as be
    n, dev = env.num_envs, env.device
    o_s, yaw_s, _ = be.slot_frame()
    Tn = be.basket_nominal_T()
    o_b, yaw_b = Tn[:3, 3], math.radians(be.B['full']['pose_base'][3])
    rel = np.asarray(o_s[:2]) - o_b[:2]
    ids = list(range(n)) if ids is None else [int(i) for i in ids]
    pose = torch.zeros(len(ids), 7, device=dev)
    for j, (i, T) in enumerate(zip(ids, T_list)):
        T = np.asarray(T)
        yaw = math.atan2(T[1, 0], T[0, 0])
        d = yaw - yaw_b
        pose[j, :3] = torch.tensor(T[:3, 3], dtype=torch.float32, device=dev)
        pose[j, 3], pose[j, 6] = math.cos(yaw / 2), math.sin(yaw / 2)
        c, s = math.cos(d), math.sin(d)
        env._slot[i, 0] = float(T[0, 3] + c * rel[0] - s * rel[1])
        env._slot[i, 1] = float(T[1, 3] + s * rel[0] + c * rel[1])
        env._slot[i, 2] = float(o_s[2] + T[2, 3] - o_b[2])
        env._slot[i, 3] = yaw_s + d
    idt = torch.tensor(ids, device=dev)
    pose[:, :3] += env.scene.env_origins[idt]
    env.scene['basket'].write_root_pose_to_sim(pose, env_ids=idt)


def drive(env, paths_deg, closed, speed, hold=15):
    """Every env along its own joint waypoint path (list of (k, 6) deg arrays, None = stay) in the same number of
    steps (the longest path at `speed` deg/s), fingers closed / open per env, then `hold` settle steps."""
    import numpy as np
    import torch
    import plate_env as pe
    robot = env.scene['robot']
    aj = env.action_manager.get_term('arm_action')._joint_ids
    fj = [robot.joint_names.index(j) for j in ('left_finger_joint', 'right_finger_joint')]
    q = robot.data.joint_pos.clone()
    now = torch.rad2deg(q[:, aj]).cpu().numpy()
    dense = []
    for k, p in enumerate(paths_deg):
        pts = np.vstack([now[k], p]) if p is not None and len(p) else now[k][None]
        seg = np.abs(np.diff(pts, axis=0)).max(1) if len(pts) > 1 else np.zeros(0)
        dense.append((pts, np.r_[0, np.cumsum(seg)]))
    total = max(float(s[-1]) for _, s in dense)
    steps = max(1, int(total / speed * 30))
    for i in range(1, steps + 1):
        for k, (pts, s) in enumerate(dense):
            if s[-1] <= 0:
                continue
            t = min(s[-1], total * i / steps)
            j = min(np.searchsorted(s, t, 'right') - 1, len(pts) - 2)
            f = (t - s[j]) / max(s[j + 1] - s[j], 1e-9)
            q[k, aj] = torch.tensor(np.radians(pts[j] + (pts[j + 1] - pts[j]) * f), dtype=torch.float32, device=env.device)
        shut = torch.as_tensor(closed, device=env.device)
        if getattr(env, '_attach', None) is not None:
            shut = shut & ~env._attach['on']
        q[:, fj] = torch.where(shut[:, None],
                               torch.full_like(q[:, fj], pe.FINGER_CLOSED), torch.zeros_like(q[:, fj]))
        robot.set_joint_position_target(q)
        sim_steps(env)
    for _ in range(hold):
        sim_steps(env)


def inserted(env, still_window=30):
    """rl/basket.yaml success: lowest rim point in the gap, low enough, plane close to the slot plane, at rest."""
    import torch
    import basket_env as be
    p0 = env.scene['plate'].data.root_pos_w.clone()
    for _ in range(still_window):
        sim_steps(env)
    st = be.basket_state(env)
    sc = be.B['success']
    still = (env.scene['plate'].data.root_pos_w - p0).norm(dim=1) < 0.01
    # (10/12 night) measured tines lean (-x 12.5 deg): the gap at height h is shifted -h*tan(lean) along n, and a seated
    # plate rests near the 49 mm rail -> judge the lowest point against the leaning gap; plate within 25 deg of upright
    import math
    import plate_env as pe
    lean = math.tan(math.radians(be.B['pins'].get('lean_deg', 0.0)))
    gap_err = (st['d_bottom'] + st['low'].clamp(min=0) * lean).abs()
    upright = pe.body_axis(env.scene['plate'].data.root_quat_w, 2)[:, 2].abs() < math.sin(math.radians(25))
    ok = ((gap_err < sc['bottom_tol_m']) & (st['d_u'].abs() < sc['centre_tol_m'])
          & (st['low'] < sc['max_height_m'] + 0.005) & upright & still)
    st['gap_err'] = gap_err
    return ok, st


def E_CFG():
    import env as E
    return E.CFG


def make_cfg(n, viewer, torsion_m):
    """basket env cfg + a torsional friction patch on the plate collider (10/12 night): the sim pinch held the rim at two
    points and the plate swung like a pendulum about the line between the fingers (a turned tool did not turn the
    plate, the real one does); the real 2F-85 pads press a ~22 mm rubber patch."""
    import isaaclab.sim as sim_utils
    import basket_env as be
    cfg = be.make_basket_cfg(n, viewer=viewer)
    cp = cfg.scene.plate.spawn.collision_props or sim_utils.CollisionPropertiesCfg()
    cp.torsional_patch_radius = torsion_m
    cp.min_torsional_patch_radius = torsion_m
    cfg.scene.plate.spawn.collision_props = cp
    return cfg

# ---------------------------------------------------------------- P1 replay of the real runs
def run_replay(args):
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    import basket_env as be
    import plate_env as pe
    runs = [] if args.plans else [json.loads(l) for l in open(args.labels)]
    rows = []
    for r in ([json.loads(l) for l in open(args.plans)] if args.plans else []):   # planned, not run: 'outcome' = planned
        legs = json.loads(Path(r['plan']).read_text())['legs']
        rows.append(dict(name=r['name'], outcome='planned', plate=r['plate'], Tb=r['Tb'], grasp=legs, slot=legs))
    for r in runs:
        d = Path(r['run']) if Path(r['run']).is_absolute() else ROOT / 'calibration/data/real_pipeline' / r['run']
        ev = [json.loads(x) for x in open(d / 'robot_state.jsonl') if '"event"' in x]
        grasp = json.loads(Path(next(e['plan'] for e in ev if e['event'] == 'plan')).read_text())['legs']
        slot = json.loads(Path(next(e['plan'] for e in ev if e['event'] == 'slot_plan')).read_text())['legs']
        photo = d / 'wrist_above.json' if (d / 'wrist_above.json').exists() else d / 'wrist_pre.json'
        rows.append(dict(name=d.name, outcome=r['outcome'], plate=next(e for e in ev if e['event'] == 'plan')['plate_root_fp'],
                         Tb=json.loads(photo.read_text())['T_base_basket'], grasp=grasp, slot=slot))
    n = len(rows) * args.per_run
    env = ManagerBasedRLEnv(make_cfg(n, not args.headless, E_CFG()['expert'].get('pad_torsion_radius_m', 0.01)))
    env.reset()
    dev = env.device
    who = np.repeat(np.arange(len(rows)), args.per_run)
    set_basket(env, [rows[i]['Tb'] for i in who])
    rng = np.random.default_rng(args.seed)
    pose = torch.zeros(n, 7, device=dev)
    for k, i in enumerate(who):
        jit = rng.normal(0, args.jitter_mm / 1000, 2) if k % args.per_run else np.zeros(2)   # first env of a run: exact
        pose[k, :2] = torch.tensor(np.asarray(rows[i]['plate'][:2]) + jit, dtype=torch.float32)
    pose[:, 2], pose[:, 3] = pe.REST_Z, 1.0
    pose[:, :3] += env.scene.env_origins
    env.scene['plate'].write_root_pose_to_sim(pose)
    env.scene['plate'].write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
    pe.box_under_plate(env, torch.arange(n, device=dev))
    robot = env.scene['robot']
    q = robot.data.default_joint_pos.clone()                   # home
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    for _ in range(30):
        sim_steps(env)
    L = lambda leg, src: [np.asarray(rows[i][src][leg]['waypoints_deg']) for i in who]   # noqa: E731
    open_ = np.zeros(n, bool)
    drive(env, L('to_grasp', 'grasp'), open_, args.speed)
    drive(env, [None] * n, ~open_, args.speed, hold=30)       # close (stalls on the rim)
    drive(env, L('lift', 'grasp'), ~open_, args.speed, hold=60)
    lift = env.scene['plate'].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - pe.REST_Z
    held = (lift > 0.05).cpu().numpy()
    if args.attach:                                            # rigid in-hand plate (see attach), no extra noise
        attach(env, torch.as_tensor(held, device=dev))
    drive(env, L('to_above', 'slot'), ~open_, args.speed)
    if all('descend_pre' in rows[i]['slot'] for i in who):    # taught slot paths only (quadrant plans: to_above -> descend)
        drive(env, L('descend_pre', 'slot'), ~open_, args.speed / 4)
    st_pre = be.basket_state(env)
    drive(env, L('descend', 'slot'), ~open_, args.speed / 20, hold=30)
    if args.attach:
        detach(env, torch.ones(n, dtype=torch.bool, device=dev))
    drive(env, [None] * n, open_, args.speed, hold=45)         # release
    drive(env, L('ascend', 'slot'), open_, args.speed / 4)
    ok, st = inserted(env)
    ok = ok.cpu().numpy() & held
    # the real plates did not always go into the taught gap (tilt + 4 cm along the rows): judge "in A gap" too
    pitch = be.B['pins']['pitch']
    gap_err = ((st['d_bottom'] + pitch / 2) % pitch - pitch / 2).abs()
    sc = be.B['success']
    any_gap = ((gap_err < sc['bottom_tol_m']) & (st['low'] < sc['max_height_m']) & (st['normal_err'] < sc['max_normal_err_deg'])
               & (st['d_u'].abs() < sc['centre_tol_m'])).cpu().numpy() & held
    np.set_printoptions(precision=1, suppress=True)
    print('[replay] final d_bottom mm', (st['d_bottom'] * 1000).cpu().numpy()[::args.per_run], flush=True)
    print('[replay] final gap err mm  ', (gap_err * 1000).cpu().numpy()[::args.per_run], flush=True)
    print('[replay] final low mm      ', (st['low'] * 1000).cpu().numpy()[::args.per_run], flush=True)
    print('[replay] final normal deg  ', st['normal_err'].cpu().numpy()[::args.per_run], ' d_u mm', (st['d_u'] * 1000).cpu().numpy()[::args.per_run], flush=True)
    print('\n[replay] run                              real      sim inserted  (held)  pre d_bottom mm  pre normal err deg', flush=True)
    res = []
    for i, r in enumerate(rows):
        m = who == i
        res.append(dict(run=r['name'], real=r['outcome'], sim_rate=float(ok[m].mean()), any_gap=float(any_gap[m].mean()), held=float(held[m].mean()),
                        pre_d_bottom_mm=float(st_pre['d_bottom'][torch.as_tensor(m)].mean() * 1000),
                        pre_normal_err=float(st_pre['normal_err'][torch.as_tensor(m)].mean())))
        print(f"[replay] {r['name']:32s} {r['outcome'][:8]:8s}  {ok[m].mean():5.0%} any gap {any_gap[m].mean():4.0%} ({held[m].mean():4.0%})  "
              f"{res[-1]['pre_d_bottom_mm']:+7.1f}        {res[-1]['pre_normal_err']:5.1f}", flush=True)
    good = [x for x in res if x['real'] == 'success']
    gate = sum(x['sim_rate'] >= 0.8 for x in good)
    print(f'[replay] P1 gate: {gate}/{len(good)} real successes reproduced with sim rate >= 80 % (need >= 9)', flush=True)
    out = Path(args.out or RL / 'logs/basket/replay_20261012.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(results=res, gate=gate, need=9), indent=1))
    env.close()


# ---------------------------------------------------------------- feedback expert -> demos for train.py bc
def run_collect(args):
    """One batch per call of the outer loop: grasp (library replay) -> basket at a random pose -> planned transport to
    `above` (joint IK, basket_tasks) -> FEEDBACK insertion recorded at 30 Hz in the slot frame:
      lateral   : move the plate's lowest rim point onto the gap centre (n) and the slot centre (u)
      rotation  : turn the tool to the insert orientation
      descent   : free above the tine tips; near them only while the lowest point is within 3 mm of the gap centre
      release   : open at target depth, wait 1 s, rise 4 s
    The expert decides from the CURRENT state only (no retry counters: BC must be able to copy it).  DART: the executed
    step gets Gaussian noise, the recorded label stays the clean expert step."""
    import math
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_from_matrix, quat_mul
    sys.path.insert(0, str(ROOT / 'calibration'))
    from robot_model import Kinematics
    import basket_env as be
    import basket_tasks as bt
    import plate_env as pe
    import env as E
    C = E.CFG
    ex = C['expert']
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = ManagerBasedRLEnv(make_cfg(ex['envs'], not args.headless, ex.get('pad_torsion_radius_m', 0.01)))
    n, dev, org = env.num_envs, env.device, env.scene.env_origins
    robot, plate = env.scene['robot'], env.scene['plate']
    aj = env.action_manager.get_term('arm_action')._joint_ids
    l6 = robot.body_names.index('link_6')
    kin = Kinematics()
    golden = json.loads((be.RL_DIR / be.B['golden_teach']).read_text())['poses']
    seeds = [np.radians(golden['above']['posj']), np.radians(golden['insert']['posj'])]
    Tn = be.basket_nominal_T()
    centre = Tn[:3, 3] + np.array(ex.get('basket_centre_offset_m', [-0.11, 0.006, 0.0]))
    step_max = torch.as_tensor(C['action']['step_max'], device=dev)
    rail = be.B['pins'].get('rail_z', 0.049)
    tip = rail + 0.066                                            # tine tips ~11.5 cm above the floor
    target_low = ex.get('target_low_m', 0.052)
    # (10/12 night) geometry check (numpy, plate mesh vs leaning tines + rails): a plate tilted 12.8 deg (taught) jams
    # at ~71 mm, a near-vertical one (-5..+5 deg) goes down to ~46 mm (rests on the 49 mm rail) = the real recipe
    be.B['slot']['target_tilt_deg'] = ex.get('plate_tilt_deg', 0.0)
    # and the plate's concave side must face +n (away from the robot): facing -n it jams at ~70 mm (same check)
    bt.FORCE_PLUS_N = True
    bt.MIN_TURN = ex.get('min_turn', False)              # round plate: pick the in-plane turn needing the least wrist turn
    lean = math.tan(math.radians(be.B['pins'].get('lean_deg', 0.0)))
    dt = env.step_dt
    policy = None
    if args.policy:
        import model as Mdl
        policy, _ = Mdl.load_bc(args.policy, device=dev)
        print(f'[eval] BC policy {args.policy} drives the insertion', flush=True)
    for b in range(args.batches):
        rng = np.random.default_rng(args.seed * 1000 + b)
        s0 = bt.grasped_start(env, args, rng)
        attach(env, s0['ok'], ex['in_hand_mm'], ex['in_hand_deg'], seed=args.seed * 1000 + b)
        Ts = []                                                   # random reachable-ish basket poses around the centre
        for k in range(n):
            T = Tn.copy()
            yaw = math.radians(be.B['full']['pose_base'][3] + rng.uniform(-1, 1) * ex['basket_yaw_deg'])
            T[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
            T[:3, 3] = centre + [rng.uniform(-1, 1) * ex['basket_xy_m'][0], rng.uniform(-ex['basket_xy_m'][1], 0.05), 0.0]
            Ts.append(T)
        set_basket(env, Ts)
        for _ in range(10):
            sim_steps(env)
        # planned transport to `above` (taught-order, basket_tasks), per env joint IK
        tp, tq = pe.tcp_pose(env)
        W = lambda pos, quat, k: bt._T(bt._quat_to_R(quat[k].cpu().numpy()), (pos[k] - org[k]).cpu().numpy())  # noqa: E731
        plans, q_ins = [], tq.clone()
        live = s0['ok'].cpu().numpy().copy()
        for k in range(n):
            if not live[k]:
                plans.append(None)
                continue
            tcp_T = W(tp, tq, k)
            f_t = np.linalg.inv(tcp_T) @ W(robot.data.body_pos_w[:, l6], robot.data.body_quat_w[:, l6], k)
            try:
                ins, above, v = bt.insertion_targets(env, k, tcp_T, W(plate.data.root_pos_w, plate.data.root_quat_w, k), f_t)
                seed = robot.data.joint_pos[k, aj].cpu().numpy().astype(float)
                qs = []
                for wi, T_ in enumerate(bt.transport_poses(tcp_T, above, 0.40)):
                    seed = bt.ik_robust(kin, T_ @ f_t, seed, seeds, wi)
                    qs.append(np.degrees(seed))
                plans.append(np.array(qs))
                q_ins[k] = quat_from_matrix(torch.tensor(ins[:3, :3], dtype=torch.float32, device=dev)[None])[0]
            except RuntimeError:
                plans.append(None)
                live[k] = False
        closed = np.ones(n, bool)
        drive(env, [p if p is not None else None for p in plans], closed, args.speed, hold=20)
        if args.save_starts:                                    # RL start states: robot at `above`, plate attached
            a_ = env._attach
            np.savez_compressed(out / f'starts_{args.seed:02d}{b:03d}.npz', live=live,
                                q=robot.data.joint_pos.cpu().numpy(), basket=np.stack(Ts), slot=env._slot.cpu().numpy(),
                                att_p=a_['p'].cpu().numpy(), att_q=a_['q'].cpu().numpy(), q_ins=q_ins.cpu().numpy())
            print(f'[starts] batch {b}: {int(live.sum())} start states', flush=True)
            continue
        # feedback insertion, recorded
        arm = env.action_manager.get_term('arm_action')
        arm.reset(torch.arange(n, device=dev))
        arm.orientation_locked[:] = False                      # the grasp-time rim lock must not freeze the turn
        # the RL shield (frame/table SDF margins) rejected ~25/64 targets right above the basket and froze those arms;
        # the insertion moves only cm in free space (real basket plans: >= 125 mm from the frame) -> off here
        arm._safe_target = robot.data.joint_pos[:, aj].clone()
        arm.shield.safe = lambda q_: torch.ones(q_.shape[0], dtype=torch.bool, device=q_.device)
        released = torch.zeros(n, dtype=torch.bool, device=dev)
        t_rel = torch.zeros(n, device=dev)
        liv = torch.as_tensor(live, device=dev)
        O, A, M = [], [], []
        dart = args.dart
        for step in range(int(args.episode_s / dt)):
            obs, st = E.observe(env, released)
            R = E.slot_R(env)
            # lateral (slot frame): lowest point onto the gap centre along n, plate centre onto the slot centre along u
            # the gap centre leans with the tines: at height h it sits -h*tan(lean) along n from the floor slot line
            d_goal = -st['low'].clamp(min=0.0, max=tip) * lean
            e_n = st['d_bottom'] - d_goal
            lat = torch.stack([-e_n * 2.0, -st['d_u'] * 1.0, torch.zeros_like(st['low'])], 1) * dt
            # rotation toward the insert orientation (env-frame axis-angle -> slot frame)
            # keep the PLATE upright with its normal along the rows (n): rotate the tool by the plate-normal error
            # (the plate can turn in the fingers, so the tool target alone is not enough)
            nrm_e = pe.body_axis(plate.data.root_quat_w, 2)
            tgt = R[:, :, 0]                                     # concave side toward +n (debug print)
            _, tq_ = pe.tcp_pose(env)
            rot_env = axis_angle_from_quat(quat_mul(q_ins, quat_conjugate(tq_))) * 3.0 * dt
            rot = torch.einsum('nji,nj->ni', R, rot_env)
            high = st['low'] > tip + 0.015
            aligned = e_n.abs() < 0.003
            upright = nrm_e[:, 2].abs() < 0.10                      # plate within ~6 deg of upright
            vz = torch.where(high, torch.full_like(t_rel, -0.03), torch.where(aligned & upright, torch.full_like(t_rel, -0.008),
                                                                               torch.zeros_like(t_rel)))
            vz = torch.where(released, torch.where(t_rel > 1.0, torch.full_like(t_rel, 0.03), torch.zeros_like(t_rel)), vz)
            stepv = torch.cat([lat, rot], 1)
            stepv[:, 2] = vz * dt
            stepv[released, :2] = 0.0
            stepv[released, 3:] = 0.0
            stepv = E.clip_step(stepv)
            open_ = released.float()
            label = torch.cat([stepv, open_[:, None]], 1)
            exe = stepv.clone()
            if policy is not None:                               # eval: the BC drives (the expert step is only logged)
                with torch.no_grad():
                    pa, pg = policy(obs)
                exe = E.clip_step(pa * policy.step_max)
                pol_open = (pg > 0) | released
                open_ = pol_open.float()
            if dart:
                exe = E.clip_step(exe + torch.randn_like(exe) * torch.as_tensor(ex['dart_noise'], device=dev))
            exe[~liv] = 0.0
            O.append(obs.cpu().numpy()); A.append(label.cpu().numpy()); M.append((liv & (t_rel < 5.0)).cpu().numpy())
            act_env = E.to_env_action(env, exe, open_)
            act_env[env._attach['on'], 6] = 1.0                 # attached: fingers open (no fight with the teleport)
            env.step(act_env)
            attach_update(env)
            if args.debug and step % 30 == 0:
                k_ = np.flatnonzero(live)[:4]
                tpd, _ = pe.tcp_pose(env)
                kk = int(k_[0])
                tz = pe.body_axis(pe.tcp_pose(env)[1], 2)
                print(f"[dbg3] tool_z . plate_n per env {((tz * nrm_e).sum(1)[k_]).cpu().numpy().round(3).tolist()}  tool_z_x {tz[k_, 0].cpu().numpy().round(3).tolist()}", flush=True)
                print(f"[dbg2] nrm {nrm_e[kk].cpu().numpy().round(3).tolist()} tgt {tgt[kk].cpu().numpy().round(3).tolist()} "
                      f"rot_env {(rot_env[kk] / dt).cpu().numpy().round(3).tolist()} tool_z {pe.body_axis(pe.tcp_pose(env)[1], 2)[kk].cpu().numpy().round(3).tolist()} "
                      f"q {torch.rad2deg(robot.data.joint_pos[kk, aj]).cpu().numpy().round(1).tolist()}", flush=True)
                print(f"[dbg] t {step * dt:4.1f} low {(st['low'][k_] * 1000).round().tolist()} d_bot {(st['d_bottom'][k_] * 1000).round().tolist()} "
                      f"d_u {(st['d_u'][k_] * 1000).round().tolist()} nerr {st['normal_err'][k_].round().tolist()} rel {released[k_].tolist()} "
                      f"rot {(rot[k_].norm(dim=1) / dt).round(decimals=2).tolist()} lock {arm.orientation_locked[k_].tolist()} blk {int(arm.blocked.sum())} tcpz {((tpd - org)[k_, 2] * 1000).round().tolist()}", flush=True)
            land = ~released & ((st['low'] <= target_low) if policy is None else (open_ > 0.5))
            released |= land
            detach(env, land)
            t_rel = torch.where(released, t_rel + dt, t_rel)
            if bool((released | ~liv).all()) and float(t_rel[liv].min() if liv.any() else 9) >= 5.0:
                break
        ok, st = inserted(env)
        ok = ok & liv
        import plate_env as pe_
        sc_ = be.B['success']
        up_ = pe_.body_axis(plate.data.root_quat_w, 2)[:, 2].abs() < math.sin(math.radians(25))
        f_ = liv & ~ok
        nr_ = torch.nonzero(f_ & ~released).squeeze(1)[:8]
        if len(nr_):
            _, tq_e = pe_.tcp_pose(env)
            rerr = axis_angle_from_quat(quat_mul(q_ins, quat_conjugate(tq_e))).norm(dim=1)
            print(f"[collect] stuck: low {(st['low'][nr_] * 1000).round().tolist()} e_n {((st['d_bottom'] + st['low'].clamp(min=0, max=tip) * lean)[nr_] * 1000).round().tolist()} "
                  f"plate nz {pe_.body_axis(plate.data.root_quat_w, 2)[nr_, 2].round(decimals=2).tolist()} tool rot err deg {torch.rad2deg(rerr[nr_]).round().tolist()}", flush=True)
        print(f"[collect] failures {int(f_.sum())}: never released {int((f_ & ~released).sum())}, gap > tol {int((f_ & (st['gap_err'] >= sc_['bottom_tol_m'])).sum())}, "
              f"too high {int((f_ & (st['low'] >= sc_['max_height_m'] + 0.005)).sum())}, |d_u| {int((f_ & (st['d_u'].abs() >= sc_['centre_tol_m'])).sum())}, "
              f"not upright {int((f_ & ~up_).sum())}", flush=True)
        obs_a, act_a, mask_a = np.stack(O, 1), np.stack(A, 1), np.stack(M, 1)
        np.savez_compressed(out / f'batch_{args.seed:02d}{b:03d}.npz', obs=obs_a.astype(np.float32), act=act_a.astype(np.float32),
                            mask=mask_a, success=ok.cpu().numpy(), dart=dart, live=live)
        print(f'[collect] batch {b}: grasped {int(s0["ok"].sum())}/{n}, planned {int(live.sum())}, inserted {int(ok.sum())}, '
              f'steps {obs_a.shape[1]}  dart={dart}', flush=True)
    env.close()


# ---------------------------------------------------------------- P4: real-robot insertion plan from the learned policy
def run_realplan(args):
    """Real measurements -> sim rollout of the learned policy -> joint path for the real mover (goal-1 style).
    --req json: T_base_basket (marker fit), grasp_deg (real grasp joints), plate_root_fp (FoundationPose at the grasp),
    above_deg (golden plan to_above end), out (plan.json).  Env 0 is exact; envs 1.. add in-hand noise (+-2 mm / 2 deg)
    as a robustness check.  The target gap = sim slot origin + slot_shift_pitches (P1: the real intended gap = +2)."""
    import math
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    sys.path.insert(0, str(ROOT / 'calibration'))
    from robot_model import Kinematics
    import basket_env as be
    import plate_env as pe
    import env as E
    import model as Mdl
    req = json.loads(Path(args.req).read_text())
    n = args.per_run
    env = ManagerBasedRLEnv(make_cfg(n, not args.headless, E.CFG['expert'].get('pad_torsion_radius_m', 0.01)))
    env.reset()
    dev = env.device
    robot = env.scene['robot']
    arm = env.action_manager.get_term('arm_action')
    aj = arm._joint_ids
    arm.shield.safe = lambda q_: torch.ones(q_.shape[0], dtype=torch.bool, device=q_.device)
    set_basket(env, [np.asarray(req['T_base_basket'])] * n)
    shift = req.get('slot_shift_pitches', 2) * be.B['pins']['pitch']
    env._slot[:, 0] += shift * torch.cos(env._slot[:, 3])
    env._slot[:, 1] += shift * torch.sin(env._slot[:, 3])
    q = robot.data.default_joint_pos.clone()
    q[:, aj] = torch.tensor(np.radians(req['above_deg']), dtype=torch.float32, device=dev)
    fj = [robot.joint_names.index(j) for j in ('left_finger_joint', 'right_finger_joint')]
    q[:, fj] = 0.0
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    # plate in the hand: FoundationPose root at the grasp, flat, x toward the plate centre (round plate: yaw free)
    kin = Kinematics()
    T6 = kin.fk(np.radians(req['grasp_deg']))
    T_tcp = T6 @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.165], [0, 0, 0, 1.0]])
    c = np.asarray(req['plate_root_fp'], float)
    d = c[:2] - T_tcp[:2, 3]
    d = np.r_[d / np.linalg.norm(d), 0.0]
    Tp = np.eye(4)
    Tp[:3, 0], Tp[:3, 1], Tp[:3, 2], Tp[:3, 3] = d, np.cross([0, 0, 1.0], d), [0, 0, 1.0], c
    H = np.linalg.inv(T_tcp) @ Tp
    from scipy.spatial.transform import Rotation
    # (10/13 night) the sim TCP frame is the FK tool frame turned about the tool axis (z matches, x/y do not):
    # express the in-hand pose in the sim frame: H_sim = R_off^T H, R_off = R_fk^T R_sim at the same joints
    for _ in range(30):
        sim_steps(env)
    _, tq0 = pe.tcp_pose(env)
    w_, x_, y_, z_ = tq0[0].cpu().numpy()
    R_sim = Rotation.from_quat([x_, y_, z_, w_]).as_matrix()
    R_fk = (kin.fk(np.radians(req['above_deg'])))[:3, :3]
    R_off = R_fk.T @ R_sim
    T_off = np.eye(4)
    T_off[:3, :3] = R_off
    H = np.linalg.inv(T_off) @ H
    print(f'[realplan] sim TCP frame vs FK: {np.degrees(np.arccos(np.clip((np.trace(R_off) - 1) / 2, -1, 1))):.1f} deg', flush=True)
    hq = Rotation.from_matrix(H[:3, :3]).as_quat()            # x y z w
    env._attach = dict(on=torch.ones(n, dtype=torch.bool, device=dev),
                       p=torch.tensor(H[:3, 3], dtype=torch.float32, device=dev).repeat(n, 1),
                       q=torch.tensor([hq[3], hq[0], hq[1], hq[2]], dtype=torch.float32, device=dev).repeat(n, 1))
    if n > 1:                                                  # robustness variants: in-hand noise on envs 1..
        rng = torch.Generator(device=dev).manual_seed(0)
        env._attach['p'][1:] += (torch.rand(n - 1, 3, generator=rng, device=dev) * 2 - 1) * 0.002
    for _ in range(15):
        sim_steps(env)
    arm.reset(torch.arange(n, device=dev))
    arm.orientation_locked[:] = False
    arm._safe_target = robot.data.joint_pos[:, aj].clone()
    nrm = pe.body_axis(env.scene['plate'].data.root_quat_w, 2)
    facing = float((nrm[0] * E.slot_R(env)[0, :, 0]).sum())
    tp_, tq_ = pe.tcp_pose(env)
    T_above = kin.fk(np.radians(req['above_deg'])) @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.165], [0, 0, 0, 1.0]])
    pl_ = env.scene['plate'].data.root_pos_w[0] - env.scene.env_origins[0]
    print(f"[realplan] sim tcp {(tp_[0] - env.scene.env_origins[0]).cpu().numpy().round(3).tolist()} vs FK tcp {T_above[:3, 3].round(3).tolist()}; "
          f"sim tool z {pe.body_axis(tq_, 2)[0].cpu().numpy().round(2).tolist()} vs FK {T_above[:3, 2].round(2).tolist()}", flush=True)
    print(f"[realplan] plate root {pl_.cpu().numpy().round(3).tolist()} expected {(T_above @ T_off @ H)[:3, 3].round(3).tolist()}; normal {nrm[0].cpu().numpy().round(2).tolist()} "
          f"expected {(T_above @ T_off @ H)[:3, 2].round(2).tolist()}; slot {env._slot[0].cpu().numpy().round(3).tolist()}", flush=True)
    print(f'[realplan] plate normal . n = {facing:+.2f} (policy trained with the concave side toward +n)', flush=True)
    # the policy was trained from the expert's `above` (plate upright over the gap, ~11 cm up): IK there first
    import basket_tasks as bt
    be.B['slot']['target_tilt_deg'] = E.CFG['expert'].get('plate_tilt_deg', 0.0)
    bt.FORCE_PLUS_N = True
    org = env.scene.env_origins
    W = lambda pos, quat, k: bt._T(bt._quat_to_R(quat[k].cpu().numpy()), (pos[k] - org[k]).cpu().numpy())  # noqa: E731
    l6 = robot.body_names.index('link_6')
    tp0, tq0 = pe.tcp_pose(env)
    tcp_T = W(tp0, tq0, 0)
    f_t = np.linalg.inv(tcp_T) @ W(robot.data.body_pos_w[:, l6], robot.data.body_quat_w[:, l6], 0)
    pl = env.scene['plate']
    ins, above, v = bt.insertion_targets(env, 0, tcp_T, W(pl.data.root_pos_w, pl.data.root_quat_w, 0), f_t)
    fk_check = np.linalg.norm(kin.fk(np.radians(req['above_deg']))[:3, 3] - W(robot.data.body_pos_w[:, l6], robot.data.body_quat_w[:, l6], 0)[:3, 3])
    seed = np.radians(req['above_deg'])
    golden = json.loads((be.RL_DIR / be.B['golden_teach']).read_text())['poses']
    try:
        q_start = bt.ik_robust(kin, above @ f_t, seed, [np.radians(golden['above']['posj']), np.radians(golden['insert']['posj'])], 'policy_start')
    except RuntimeError as e:   # basket too far (x >= ~1.15 m in the 10/12 runs): the pipeline uses the taught path
        Path(req['out']).write_text(json.dumps(dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), source='realplan (policy rollout in sim)',
                                                    sim_success_exact=False, sim_success_variants=0.0, final_gap_mm=[], final_low_mm=[],
                                                    reason=f'policy start pose unreachable: {e}'), indent=1))
        print(f'[realplan] policy start pose unreachable ({e}) -> {req["out"]}', flush=True)
        env.close()
        return
    print(f"[realplan] link_6 sim vs FK {fk_check * 1000:.1f} mm; policy start tcp {above[:3, 3].round(3).tolist()} "
          f"(max joint move {np.abs(np.degrees(q_start) - req['above_deg']).max():.1f} deg)", flush=True)
    detach(env, torch.zeros(n, dtype=torch.bool, device=dev))
    drive(env, [np.degrees(q_start)[None]] * n, np.zeros(n, bool), 10.0, hold=15)
    to_start = [list(req['above_deg']), np.degrees(q_start).round(3).tolist()]
    arm.reset(torch.arange(n, device=dev))
    arm.orientation_locked[:] = False
    arm._safe_target = robot.data.joint_pos[:, aj].clone()
    bc, meta = Mdl.load_bc(args.policy, device=dev)
    ac = Mdl.ResidualActorCritic(bc, E.CFG['rl']['residual_scale'], meta['hidden'], meta['activation']).to(dev)
    if args.residual:
        ac.load_state_dict(torch.load(args.residual, map_location=dev)['state'])
    sm = torch.as_tensor(E.CFG['action']['step_max'], device=dev)
    released = torch.zeros(n, dtype=torch.bool, device=dev)
    t_rel = torch.zeros(n, device=dev)
    path, rel_step = [], None
    dt = env.step_dt
    for step in range(int(25 / dt)):
        o, st = E.observe(env, released)
        with torch.no_grad():
            a = ac.mean(o)
        cont = a[:, :6].clamp(-1, 1)
        opening = (a[:, 6] > 0) & ~released
        exe = cont * sm
        exe[released] = 0.0
        exe[released & (t_rel > 1.0), 2] = sm[2]
        act = E.to_env_action(env, exe, (released | opening).float())
        act[env._attach['on'], 6] = 1.0
        env.step(act)
        attach_update(env)
        if bool(opening[0]) and rel_step is None:
            rel_step = len(path)
        released |= opening
        detach(env, opening)
        t_rel = torch.where(released, t_rel + dt, t_rel)
        path.append(torch.rad2deg(robot.data.joint_pos[0, aj]).cpu().numpy())
        if bool(released.all()) and float(t_rel.min()) > 3.0:
            break
    lean = math.tan(math.radians(be.B['pins'].get('lean_deg', 0.0)))
    st = be.basket_state(env)
    gap = (st['d_bottom'] + st['low'].clamp(min=0) * lean).abs()
    ok = released & (gap < 0.0085) & (st['low'] < 0.06) & (st['d_u'].abs() < 0.03)
    path = np.array(path)
    res = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), source='realplan (policy rollout in sim)',
               sim_success_exact=bool(ok[0]), sim_success_variants=float(ok.float().mean()), plate_facing=facing,
               release_index=rel_step, final_gap_mm=(gap * 1000).cpu().numpy().round(1).tolist(),
               final_low_mm=(st['low'] * 1000).cpu().numpy().round(1).tolist())
    if rel_step is not None:
        down = path[:rel_step + 1]
        up = path[rel_step + 1:]
        k = max(1, len(down) // 60)
        res['legs'] = dict(to_policy_start=dict(waypoints_deg=to_start), policy_descend=dict(waypoints_deg=np.vstack([down[::k], down[-1]]).round(3).tolist()),
                           policy_ascend=dict(waypoints_deg=(up[::max(1, len(up) // 20)].round(3).tolist() if len(up) else [])))
    Path(req['out']).write_text(json.dumps(res, indent=1))
    print(f"[realplan] exact {res['sim_success_exact']}, variants {res['sim_success_variants']:.2f}, release at step {rel_step}, "
          f"gap mm {res['final_gap_mm'][:4]} low mm {res['final_low_mm'][:4]} -> {req['out']}", flush=True)
    env.close()


if __name__ == '__main__':
    a = parse_args()
    from isaaclab.app import AppLauncher
    app = AppLauncher(a).app
    if a.mode == 'replay':
        run_replay(a)
    elif a.mode == 'realplan':
        run_realplan(a)
    else:
        run_collect(a)
    app.close()
