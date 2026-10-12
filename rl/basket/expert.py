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
from pathlib import Path

HERE = Path(__file__).resolve().parent
RL = HERE.parent
ROOT = RL.parent
sys.path[:0] = [str(HERE), str(RL)]


def parse_args():
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=('replay', 'collect'))
    p.add_argument('--labels', default=str(ROOT / 'calibration/data/real_pipeline/labels_20261012.jsonl'))
    p.add_argument('--per_run', type=int, default=16, help='replay: envs per real run')
    p.add_argument('--jitter_mm', type=float, default=2.0, help='replay: plate xy jitter (FP noise)')
    p.add_argument('--speed', type=float, default=20.0, help='sim joint speed for transport legs (deg/s)')
    p.add_argument('--out', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--batches', type=int, default=1, help='collect: batches of config expert.envs envs')
    p.add_argument('--dart', action='store_true', help='collect: noisy execution, clean labels')
    p.add_argument('--episode_s', type=float, default=30.0, help='collect: insertion time limit')
    p.add_argument('--library', default=str(ROOT / 'calibration/data/plan_library/20261008_174557_380970/library.npz'))
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


# ---------------------------------------------------------------- shared sim helpers
def sim_steps(env, k=None):
    for _ in range(k or env.cfg.decimation):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
    env.sim.render()


def set_basket(env, T_list):
    """Basket of env i at the base-frame pose T_list[i] (marker frame, 4x4); env._slot follows like randomize_basket."""
    import math
    import numpy as np
    import torch
    import basket_env as be
    n, dev = env.num_envs, env.device
    o_s, yaw_s, _ = be.slot_frame()
    Tn = be.basket_nominal_T()
    o_b, yaw_b = Tn[:3, 3], math.radians(be.B['full']['pose_base'][3])
    rel = np.asarray(o_s[:2]) - o_b[:2]
    pose = torch.zeros(n, 7, device=dev)
    for i, T in enumerate(T_list):
        T = np.asarray(T)
        yaw = math.atan2(T[1, 0], T[0, 0])
        d = yaw - yaw_b
        pose[i, :3] = torch.tensor(T[:3, 3], dtype=torch.float32, device=dev)
        pose[i, 3], pose[i, 6] = math.cos(yaw / 2), math.sin(yaw / 2)
        c, s = math.cos(d), math.sin(d)
        env._slot[i, 0] = float(T[0, 3] + c * rel[0] - s * rel[1])
        env._slot[i, 1] = float(T[1, 3] + s * rel[0] + c * rel[1])
        env._slot[i, 2] = float(o_s[2] + T[2, 3] - o_b[2])
        env._slot[i, 3] = yaw_s + d
    pose[:, :3] += env.scene.env_origins
    env.scene['basket'].write_root_pose_to_sim(pose)


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
        q[:, fj] = torch.where(torch.as_tensor(closed, device=env.device)[:, None],
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
    ok = ((st['d_bottom'].abs() < sc['bottom_tol_m']) & (st['d_u'].abs() < sc['centre_tol_m'])
          & (st['low'] < sc['max_height_m']) & (st['normal_err'] < sc['max_normal_err_deg']) & still)
    return ok, st


# ---------------------------------------------------------------- P1 replay of the real runs
def run_replay(args):
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    import basket_env as be
    import plate_env as pe
    runs = [json.loads(l) for l in open(args.labels)]
    rows = []
    for r in runs:
        d = Path(r['run']) if Path(r['run']).is_absolute() else ROOT / 'calibration/data/real_pipeline' / r['run']
        ev = [json.loads(x) for x in open(d / 'robot_state.jsonl') if '"event"' in x]
        grasp = json.loads(Path(next(e['plan'] for e in ev if e['event'] == 'plan')).read_text())['legs']
        slot = json.loads(Path(next(e['plan'] for e in ev if e['event'] == 'slot_plan')).read_text())['legs']
        photo = d / 'wrist_above.json' if (d / 'wrist_above.json').exists() else d / 'wrist_pre.json'
        rows.append(dict(name=d.name, outcome=r['outcome'], plate=next(e for e in ev if e['event'] == 'plan')['plate_root_fp'],
                         Tb=json.loads(photo.read_text())['T_base_basket'], grasp=grasp, slot=slot))
    n = len(rows) * args.per_run
    env = ManagerBasedRLEnv(be.make_basket_cfg(n, viewer=not args.headless))
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
    drive(env, L('to_above', 'slot'), ~open_, args.speed)
    drive(env, L('descend_pre', 'slot'), ~open_, args.speed / 4)
    st_pre = be.basket_state(env)
    drive(env, L('descend', 'slot'), ~open_, args.speed / 20, hold=30)
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
    env = ManagerBasedRLEnv(be.make_basket_cfg(ex['envs'], viewer=not args.headless))
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
    target_low = ex.get('target_low_m', 0.030)
    dt = env.step_dt
    for b in range(args.batches):
        rng = np.random.default_rng(args.seed * 1000 + b)
        s0 = bt.grasped_start(env, args, rng)
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
        # feedback insertion, recorded
        arm = env.action_manager.get_term('arm_action')
        arm.reset(torch.arange(n, device=dev))
        released = torch.zeros(n, dtype=torch.bool, device=dev)
        t_rel = torch.zeros(n, device=dev)
        liv = torch.as_tensor(live, device=dev)
        O, A, M = [], [], []
        dart = args.dart
        for step in range(int(args.episode_s / dt)):
            obs, st = E.observe(env, released)
            R = E.slot_R(env)
            # lateral (slot frame): lowest point onto the gap centre along n, plate centre onto the slot centre along u
            lat = torch.stack([-st['d_bottom'] * 2.0, -st['d_u'] * 1.0, torch.zeros_like(st['low'])], 1) * dt
            # rotation toward the insert orientation (env-frame axis-angle -> slot frame)
            _, tq_ = pe.tcp_pose(env)
            rot_env = axis_angle_from_quat(quat_mul(q_ins, quat_conjugate(tq_))) * 1.0 * dt * 3.0
            rot = torch.einsum('nji,nj->ni', R, rot_env)
            high = st['low'] > tip + 0.015
            aligned = st['d_bottom'].abs() < 0.003
            vz = torch.where(high, torch.full_like(t_rel, -0.03), torch.where(aligned, torch.full_like(t_rel, -0.012),
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
            if dart:
                exe = E.clip_step(exe + torch.randn_like(exe) * torch.as_tensor(ex['dart_noise'], device=dev))
            exe[~liv] = 0.0
            O.append(obs.cpu().numpy()); A.append(label.cpu().numpy()); M.append((liv & (t_rel < 5.0)).cpu().numpy())
            env.step(E.to_env_action(env, exe, open_))
            if args.debug and step % 30 == 0:
                k_ = np.flatnonzero(live)[:4]
                tpd, _ = pe.tcp_pose(env)
                print(f"[dbg] t {step * dt:4.1f} low {(st['low'][k_] * 1000).round().tolist()} d_bot {(st['d_bottom'][k_] * 1000).round().tolist()} "
                      f"d_u {(st['d_u'][k_] * 1000).round().tolist()} nerr {st['normal_err'][k_].round().tolist()} rel {released[k_].tolist()} "
                      f"rot {(rot[k_].norm(dim=1) / dt).round(decimals=2).tolist()} tcpz {((tpd - org)[k_, 2] * 1000).round().tolist()}", flush=True)
            land = ~released & (st['low'] <= target_low)
            released |= land
            t_rel = torch.where(released, t_rel + dt, t_rel)
            if bool((released | ~liv).all()) and float(t_rel[liv].min() if liv.any() else 9) >= 5.0:
                break
        ok, st = inserted(env)
        ok = ok & liv
        obs_a, act_a, mask_a = np.stack(O, 1), np.stack(A, 1), np.stack(M, 1)
        np.savez_compressed(out / f'batch_{args.seed:02d}{b:03d}.npz', obs=obs_a.astype(np.float32), act=act_a.astype(np.float32),
                            mask=mask_a, success=ok.cpu().numpy(), dart=dart, live=live)
        print(f'[collect] batch {b}: grasped {int(s0["ok"].sum())}/{n}, planned {int(live.sum())}, inserted {int(ok.sum())}, '
              f'steps {obs_a.shape[1]}  dart={dart}', flush=True)
    env.close()


if __name__ == '__main__':
    a = parse_args()
    from isaaclab.app import AppLauncher
    app = AppLauncher(a).app
    if a.mode == 'replay':
        run_replay(a)
    else:
        run_collect(a)
    app.close()
