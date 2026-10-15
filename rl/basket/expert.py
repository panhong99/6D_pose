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
    p.add_argument('--g0', action='store_true', help='replay: record teacher slot-frame actions and replay them from the same hover state through SafeDiffIK')
    p.add_argument('--closed_loop', action='store_true', help='replay --plans: at the stop above the basket read the plate in the hand and '
                   're-solve the descent so it stands over its target gap (stand-in for measuring the plate there)')
    p.add_argument('--cl_dz', type=float, default=0.0, help='--closed_loop: raise the insert end by this [m] (plate bottom vs the rail)')
    p.add_argument('--cl_lean', type=float, default=1.0, help='--closed_loop: plate lean as a fraction of its quadrant\'s tine lean (1 = parallel to the tines)')
    p.add_argument('--cl_dx', type=float, default=0.0, help='--closed_loop: move the aim along the basket rows (+x) [m]')
    p.add_argument('--cl_iter', type=int, default=1, help='--closed_loop: measure + re-aim this many times at the above stop')
    p.add_argument('--cl_heights', default='', help='--closed_loop: re-aim height above the insert per iteration [m], e.g. "0.11,0.07"')
    p.add_argument('--cl_edge_q', default='', help='--closed_loop: clamp the shifted aim to the quadrant gaps, this far inside the far-end gap [m], e.g. "Q2:0.018" (unlisted: no clamp)')
    p.add_argument('--cl_noise', type=float, nargs=2, default=None, metavar=('MM', 'DEG'), help='--closed_loop: rms noise added to the measured in-hand plate pose (position mm, rotation deg) - real FP robustness')
    p.add_argument('--cl_noise_fp', default='', help='--closed_loop: Codex AC empirical FP noise model json (heavy-tailed, basket axes)')
    p.add_argument('--cl_gate', type=float, default=None, help='--closed_loop with --cl_blind: use the measured in-hand pose unless it is more than this [mm] from the mean (outlier gate)')
    p.add_argument('--cl_blind', default='', help='--closed_loop: json {model quadrant: 4x4 plate-in-TCP} used instead of measuring the plate (no in-hand sensing)')
    p.add_argument('--cl_top_q', default='', help='--closed_loop: per model quadrant top re-grip on/off, e.g. "Q1:0" (overrides --cl_top; Q1 rolled along its 3 cm rows)')
    p.add_argument('--cl_pin_dx', default='', help='--closed_loop: json {model quadrant: {pin: aim dx [m]}} (overrides --cl_dx_q for the listed pins)')
    p.add_argument('--cl_dz_q', default='', help='--closed_loop: per model quadrant release height offset [m], e.g. "Q3:0.01,Q4:0.01"')
    p.add_argument('--cl_lean_q', default='', help='--closed_loop: per model quadrant tine-lean factor, e.g. "Q1:-1,Q2:0" (overrides --cl_lean)')
    p.add_argument('--cl_dx_q', default='', help='--closed_loop: per model quadrant aim, e.g. "Q3:-0.03,Q4:-0.035" (overrides --cl_dx)')
    p.add_argument('--cl_top', action='store_true', help='--closed_loop: turn the plate about its normal so the grip is at its top')
    p.add_argument('--cl_policy', default='', help='--closed_loop: (10/15 insert RL) policy .pt from train.py insert_rl: per env aim dx/dy, '
                   'lean, dz, yaw corrections on top of the per-quadrant settings, sampled once per env (actions + rewards -> --cl_policy_out)')
    p.add_argument('--cl_policy_out', default='', help='--cl_policy: npz with obs, actions, log-probs and the per-env outcome')
    p.add_argument('--cl_policy_det', action='store_true', help='--cl_policy: use the mean action (evaluation)')
    p.add_argument('--cl_dr_full', action='store_true', help='with --cl_dr: also the unobserved errors of align.dr: basket yaw '
                   '(marker fit) and a fixed in-hand plate bias (position mm / tilt deg) in the pose the planner uses')
    p.add_argument('--cl_dr', action='store_true', help='--closed_loop: (10/15) per env model error from config align.dr: the planner aims at '
                   'a shifted gap (tine model / marker error); the policy sees that shift with 2 mm noise (stand-in for wrist gap detection)')
    p.add_argument('--hold_view', action='store_true', help='replay with the viewer: keep the window open at the end until it is closed')
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


def drive(env, paths_deg, closed, speed, hold=15, trace=None, trace_phase='motion'):
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
    import env as basket_obs
    def record_step():
        if trace is None:
            return
        trace['phase'].append(np.full(env.num_envs, trace_phase, dtype=object))
        trace['open'].append(~np.asarray(closed, dtype=bool).copy())
        diag = g0_physical_grasp_diagnostics(env, pe)
        trace['physical_held'].append(diag['predicate'])
        trace['finger_ratio'].append(diag['finger_ratio'])
        trace['tcp_local'].append(diag['tcp_local'])
        trace['rim_distance'].append(diag['rim_distance'])
        trace['released_state'].append(trace['released'].copy())
        released = torch.as_tensor(trace['released'], dtype=torch.bool, device=env.device)
        obs, _ = basket_obs.observe(env, released)
        tp, tq = pe.tcp_pose(env)
        trace['obs'].append(obs.cpu().numpy())
        trace['pos'].append((tp - env.scene.env_origins).cpu().numpy())
        trace['quat'].append(tq.cpu().numpy())
        trace['joints'].append(torch.rad2deg(robot.data.joint_pos[:, aj]).cpu().numpy())
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
        record_step()
        sim_steps(env)
        if trace is not None:
            trace['released'] |= ~np.asarray(closed, dtype=bool)
            tp, tq = pe.tcp_pose(env)
            trace['next_pos'].append((tp - env.scene.env_origins).cpu().numpy())
            trace['next_quat'].append(tq.cpu().numpy())
            trace['next_joints'].append(torch.rad2deg(robot.data.joint_pos[:, aj]).cpu().numpy())
    for _ in range(hold):
        record_step()
        sim_steps(env)
        if trace is not None:
            tp, tq = pe.tcp_pose(env)
            trace['next_pos'].append((tp - env.scene.env_origins).cpu().numpy())
            trace['next_quat'].append(tq.cpu().numpy())
            trace['next_joints'].append(torch.rad2deg(robot.data.joint_pos[:, aj]).cpu().numpy())


def capture_g0_state(env, pe):
    """Exact same-process state needed to replay a recorded insertion segment."""
    robot, plate, basket = env.scene['robot'], env.scene['plate'], env.scene['basket']
    tcp, tcp_quat = pe.tcp_pose(env)
    return dict(q=robot.data.joint_pos.clone(), qd=robot.data.joint_vel.clone(),
                plate_pose=plate.data.root_pose_w.clone(), plate_vel=plate.data.root_vel_w.clone(),
                basket_pose=basket.data.root_pose_w.clone(), basket_vel=basket.data.root_vel_w.clone(),
                tcp_pos=(tcp - env.scene.env_origins).clone(), tcp_quat=tcp_quat.clone(),
                slot=env._slot.clone())


def restore_g0_state(env, saved):
    """Restore robot, plate, basket and SafeDiffIK controller to a recorded hover state."""
    import torch
    robot, plate, basket = env.scene['robot'], env.scene['plate'], env.scene['basket']
    ids = torch.arange(env.num_envs, device=env.device)
    robot.write_joint_state_to_sim(saved['q'], saved['qd'])
    robot.set_joint_position_target(saved['q'])
    plate.write_root_pose_to_sim(saved['plate_pose'])
    plate.write_root_velocity_to_sim(saved['plate_vel'])
    basket.write_root_pose_to_sim(saved['basket_pose'])
    basket.write_root_velocity_to_sim(saved['basket_vel'])
    env._slot.copy_(saved['slot'])
    env.scene.write_data_to_sim()
    env.sim.forward()
    arm = env.action_manager.get_term('arm_action')
    arm.reset(ids)
    arm._safe_target[:] = saved['q'][:, arm._joint_ids]
    arm.orientation_locked[:] = False
    env._pp = None


def g0_actions_from_trace(trace, slot_yaw, step_max):
    """Convert sampled teacher FK motion to bounded slot-frame TCP deltas + open labels."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    fields = ('obs', 'pos', 'quat', 'next_pos', 'next_quat', 'open')
    lengths = {name: len(trace[name]) for name in fields}
    if not lengths['pos'] or len(set(lengths.values())) != 1:
        raise ValueError(f'G0 trace fields must have the same nonzero frame count: {lengths}')
    obs = np.stack(trace['obs'], axis=1).astype(np.float32)
    pos = np.stack(trace['pos'], axis=1)
    quat = np.stack(trace['quat'], axis=1)
    nxt = np.stack(trace['next_pos'], axis=1)
    nq = np.stack(trace['next_quat'], axis=1)
    yaw = np.asarray(slot_yaw, dtype=float)
    n, t = pos.shape[:2]
    if pos.shape != (n, t, 3) or nxt.shape != (n, t, 3) or quat.shape != (n, t, 4) or nq.shape != (n, t, 4):
        raise ValueError(f'G0 TCP pose trace has invalid shapes: pos={pos.shape}, quat={quat.shape}, next={nxt.shape}/{nq.shape}')
    if obs.ndim != 3 or obs.shape[:2] != (n, t) or yaw.shape != (n,):
        raise ValueError(f'G0 observation/yaw shapes do not match pose trace: obs={obs.shape}, yaw={yaw.shape}, poses={(n, t)}')
    open_signal = np.stack(trace['open'], axis=1).astype(float)
    if open_signal.shape != (n, t) or not np.isfinite(open_signal).all() or not np.isin(open_signal, (0.0, 1.0)).all():
        raise ValueError('G0 open labels must be one binary value per environment and recorded frame')
    if not all(np.isfinite(x).all() for x in (obs, pos, quat, nxt, nq, yaw)):
        raise ValueError('G0 observation or TCP trace contains a non-finite value')
    if (np.linalg.norm(quat, axis=2) < 1e-8).any() or (np.linalg.norm(nq, axis=2) < 1e-8).any():
        raise ValueError('G0 TCP trace contains a zero quaternion')
    Rslot = np.zeros((len(yaw), 3, 3))
    Rslot[:, 0, 0], Rslot[:, 1, 0] = np.cos(yaw), np.sin(yaw)
    Rslot[:, 0, 1], Rslot[:, 1, 1] = -np.sin(yaw), np.cos(yaw)
    Rslot[:, 2, 2] = 1.0
    dp = np.einsum('nij,ntj->nti', Rslot.transpose(0, 2, 1), nxt - pos)
    q0 = Rotation.from_quat(quat[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(*quat.shape[:2], 3, 3)
    q1 = Rotation.from_quat(nq[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(*nq.shape[:2], 3, 3)
    # scipy Rotation accepts a stack shaped (N, 3, 3), not (env, time, 3, 3).
    # Flatten the batch/time axes for conversion and restore them for slot-frame rotation.
    relative = (q1 @ q0.transpose(0, 1, 3, 2)).reshape(-1, 3, 3)
    rvw = Rotation.from_matrix(relative).as_rotvec().reshape(n, t, 3)
    drv = np.einsum('nij,ntj->nti', Rslot.transpose(0, 2, 1), rvw)
    raw = np.concatenate([dp, drv], axis=2)
    # SafeDiffIKAction caps the translation and rotation *vector norms* per policy step.
    # Component-wise clipping alone can command up to sqrt(3) times the intended speed
    # on a diagonal, and would leave the saved BC labels different from the executed delta.
    caps = np.asarray(step_max, dtype=float)
    if caps.shape != (6,) or not np.isfinite(caps).all() or (caps <= 0).any():
        raise ValueError('step_max must be six finite positive per-axis limits')
    if not np.allclose(caps[:3], caps[0]) or not np.allclose(caps[3:], caps[3]):
        raise ValueError('G0 vector-norm clipping requires equal per-axis limits within translation/rotation groups')
    actions = raw.copy()
    clipped = np.zeros(raw.shape[:2], dtype=bool)
    for start in (0, 3):
        part = actions[..., start:start + 3]
        norm = np.linalg.norm(part, axis=2, keepdims=True)
        cap = float(caps[start])  # config uses equal limits for each component in each 3-vector
        scale = np.minimum(1.0, cap / np.maximum(norm, 1e-12))
        clipped |= scale[..., 0] < 1.0 - 1e-12
        actions[..., start:start + 3] = part * scale
    actions = actions.astype(np.float32)
    actions = np.concatenate([actions, open_signal.astype(np.float32)[..., None]], axis=2)
    return obs, actions, int(clipped.sum()), int(clipped.shape[0] * clipped.shape[1])


def g0_tracking_action(current_pos, current_quat, target_pos, target_quat, slot_yaw, step_max):
    """Bounded slot-frame action that tracks the teacher's next absolute TCP pose from the student's live pose."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    current_pos = np.asarray(current_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    current_quat = np.asarray(current_quat, dtype=float)
    target_quat = np.asarray(target_quat, dtype=float)
    yaw = np.asarray(slot_yaw, dtype=float)
    n = len(yaw)
    if current_pos.shape != (n, 3) or target_pos.shape != (n, 3) or current_quat.shape != (n, 4) or target_quat.shape != (n, 4):
        raise ValueError('tracking pose arrays must have shapes (env,3)/(env,4), aligned with slot_yaw')
    if not all(np.isfinite(x).all() for x in (current_pos, target_pos, current_quat, target_quat, yaw)):
        raise ValueError('tracking pose contains a non-finite value')
    if (np.linalg.norm(current_quat, axis=1) < 1e-8).any() or (np.linalg.norm(target_quat, axis=1) < 1e-8).any():
        raise ValueError('tracking pose contains a zero quaternion')
    Rslot = np.zeros((n, 3, 3))
    Rslot[:, 0, 0], Rslot[:, 1, 0] = np.cos(yaw), np.sin(yaw)
    Rslot[:, 0, 1], Rslot[:, 1, 1] = -np.sin(yaw), np.cos(yaw)
    Rslot[:, 2, 2] = 1.0
    dp = np.einsum('nij,nj->ni', Rslot.transpose(0, 2, 1), target_pos - current_pos)
    q0 = Rotation.from_quat(current_quat[:, [1, 2, 3, 0]]).as_matrix()
    q1 = Rotation.from_quat(target_quat[:, [1, 2, 3, 0]]).as_matrix()
    dr_world = Rotation.from_matrix(q1 @ q0.transpose(0, 2, 1)).as_rotvec()
    dr = np.einsum('nij,nj->ni', Rslot.transpose(0, 2, 1), dr_world)
    actions = np.concatenate([dp, dr], axis=1)
    caps = np.asarray(step_max, dtype=float)
    if caps.shape != (6,) or not np.isfinite(caps).all() or (caps <= 0).any():
        raise ValueError('step_max must be six finite positive per-axis limits')
    if not np.allclose(caps[:3], caps[0]) or not np.allclose(caps[3:], caps[3]):
        raise ValueError('tracking vector-norm clipping requires equal component limits per group')
    clipped = np.zeros(n, dtype=bool)
    for start in (0, 3):
        part = actions[:, start:start + 3]
        norm = np.linalg.norm(part, axis=1, keepdims=True)
        scale = np.minimum(1.0, caps[start] / np.maximum(norm, 1e-12))
        clipped |= scale[:, 0] < 1.0 - 1e-12
        actions[:, start:start + 3] = part * scale
    return actions.astype(np.float32), clipped


def g0_phase_gate_mask(phases):
    """Mark re-aim and insert phase ends; replay may cross them only after pose tracking catches up."""
    import numpy as np
    phases = np.asarray(phases, dtype=str)
    if phases.ndim != 1 or not phases.size:
        raise ValueError('G0 phase trace must be a nonempty 1D sequence')
    gates = np.zeros(phases.shape, dtype=bool)
    for i, phase in enumerate(phases):
        next_phase = phases[i + 1] if i + 1 < len(phases) else ''
        boundary = phase != next_phase
        gates[i] = boundary and (phase.startswith('reaim_') or phase == 'insert')
    return gates


def g0_gate_alignment_mask(position_error_m, orientation_error_rad):
    """Strictly enforce the approved <1 mm AND <0.5 degree phase-gate tolerance."""
    import numpy as np
    p, r = np.asarray(position_error_m, dtype=float), np.asarray(orientation_error_rad, dtype=float)
    if p.shape != r.shape or not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError('G0 phase-gate errors must be aligned finite arrays')
    return (p < 0.001) & (r < np.deg2rad(0.5))


def g0_gate_timeout_state(teacher_success, latched, wait_frames, timeout_frames):
    """A timed-out boundary is logged and latched so replay can continue; final success stays unchanged."""
    import numpy as np
    success, done, waited = (np.asarray(teacher_success, dtype=bool), np.asarray(latched, dtype=bool),
                             np.asarray(wait_frames, dtype=int))
    if success.ndim != 1 or done.shape != success.shape or waited.shape != success.shape:
        raise ValueError('G0 gate timeout arrays must be aligned 1D arrays')
    if (waited < 0).any() or int(timeout_frames) < 1:
        raise ValueError('G0 gate timeout frame counts must be nonnegative and timeout positive')
    expired = success & ~done & (waited >= int(timeout_frames))
    return done | expired, expired


def g0_replay_cursor_should_advance(is_gate, teacher_success, tracking_clipped, gate_pending, clip_timeout=None):
    """Do not outrun a clipped SafeDiffIK correction; gate/cursor timeout only logs and resumes."""
    import numpy as np
    success, clipped = np.asarray(teacher_success, dtype=bool), np.asarray(tracking_clipped, dtype=bool)
    if success.ndim != 1 or clipped.shape != success.shape:
        raise ValueError('G0 replay cursor arrays must be aligned 1D arrays')
    if is_gate:
        pending = np.asarray(gate_pending, dtype=bool)
        if pending.shape != success.shape:
            raise ValueError('G0 gate pending mask must align with replay environments')
        return not bool(pending.any())
    timed_out = np.zeros(success.shape, dtype=bool) if clip_timeout is None else np.asarray(clip_timeout, dtype=bool)
    if timed_out.shape != success.shape:
        raise ValueError('G0 clip-timeout mask must align with replay environments')
    return not bool((success & clipped & ~timed_out).any())


def g0_success_mask(seated, in_quadrant, exact):
    """G0's teacher/replay gate: the established seated, in-quadrant, exact-slot criterion."""
    import numpy as np
    return (np.asarray(seated, dtype=bool) & np.asarray(in_quadrant, dtype=bool)
            & np.asarray(exact, dtype=bool))


def g0_gate_status(teacher_success_count, reproduced_count, threshold=0.95):
    """A zero teacher-success denominator is unassessable, never a rate of zero."""
    teacher_success_count = int(teacher_success_count)
    reproduced_count = int(reproduced_count)
    if teacher_success_count < 0 or reproduced_count < 0 or reproduced_count > teacher_success_count:
        raise ValueError('G0 reproduced count must be within the nonnegative teacher-success denominator')
    if teacher_success_count == 0:
        return 'insufficient_teacher_successes', None
    ratio = reproduced_count / teacher_success_count
    return ('pass' if ratio >= float(threshold) else 'fail'), ratio


def g0_first_true(values):
    """First true index per episode; None means the signal never became true."""
    import numpy as np
    x = np.asarray(values, dtype=bool)
    if x.ndim != 2:
        raise ValueError('G0 boolean trace must have shape (episode, frame)')
    result = []
    for row in x:
        found = np.flatnonzero(row)
        result.append(int(found[0]) if found.size else None)
    return result


def g0_physical_grasp(finger_ratio, tcp_in_plate, lifted, finger_stall, rim_r, rim_dz, held_circle_tol):
    """Physical grasp qualification from the same finger-stall/rim-circle test as plate_env._state."""
    import numpy as np
    finger = np.asarray(finger_ratio, dtype=float)
    local = np.asarray(tcp_in_plate, dtype=float)
    lift = np.asarray(lifted, dtype=bool)
    if local.shape != (finger.size, 3) or lift.shape != finger.shape:
        raise ValueError('finger, local TCP and lift masks must have aligned batch dimensions')
    d_circle = np.sqrt((np.linalg.norm(local[:, :2], axis=1) - float(rim_r)) ** 2
                       + (local[:, 2] - float(rim_dz)) ** 2)
    lo, hi = map(float, finger_stall)
    return (lift & (finger > lo) & (finger < hi) & (d_circle < float(held_circle_tol)))


def g0_physical_hold_from_env(env, pe):
    """Diagnostic predicate only; not part of AO G0's seated_exact denominator."""
    return g0_physical_grasp_diagnostics(env, pe)['predicate']


def g0_physical_grasp_diagnostics(env, pe):
    """Raw finger/TCP-to-rim values and the current physical-grasp proxy for diagnosis."""
    import numpy as np
    from isaaclab.utils.math import quat_conjugate, quat_apply
    plate = env.scene['plate']
    tcp, _ = pe.tcp_pose(env)
    local = quat_apply(quat_conjugate(plate.data.root_quat_w), tcp - plate.data.root_pos_w)
    finger = pe.finger_closure(env).cpu().numpy()
    local = local.cpu().numpy()
    radial_error = np.linalg.norm(local[:, :2], axis=1) - float(pe.G['rim_r'])
    height_error = local[:, 2] - float(pe.G['rim_dz'])
    rim_distance = np.hypot(radial_error, height_error)
    lo, hi = map(float, pe.T['finger_stall'])
    finger_ok = (finger > lo) & (finger < hi)
    rim_ok = rim_distance < float(pe.T['held_circle_tol'])
    return dict(predicate=finger_ok & rim_ok, finger_ratio=finger, tcp_local=local,
                radial_error=radial_error, height_error=height_error,
                rim_distance=rim_distance, finger_ok=finger_ok, rim_ok=rim_ok,
                finger_stall=[lo, hi], rim_radius=float(pe.G['rim_r']),
                rim_dz=float(pe.G['rim_dz']), rim_tolerance=float(pe.T['held_circle_tol']))


def g0_trace_grasp_maintained(physical_held, released_state):
    """Every recorded pre-release action state must still satisfy the physical hold predicate."""
    import numpy as np
    held = np.stack(physical_held, axis=1).astype(bool)
    released = np.stack(released_state, axis=1).astype(bool)
    if held.size == 0 or held.shape != released.shape:
        raise ValueError('physical hold and release traces must have identical shapes')
    return np.all(held | released, axis=1)


def g0_seated_exact(env, rows, who, st, be, pe, lifted_at_lift):
    """Established replay seated_exact metric; physical finger/rim checks are diagnostics only."""
    import math
    import numpy as np
    import torch
    plate = env.scene['plate']
    up = pe.body_axis(plate.data.root_quat_w, 2)
    seated = (st['low'] < 0.060) & (st['low'] > 0.020)
    landed = np.full(env.num_envs, -1, dtype=int)
    pc = (plate.data.root_pos_w - env.scene.env_origins).cpu().numpy()
    pq = plate.data.root_quat_w.cpu().numpy()
    from scipy.spatial.transform import Rotation
    f = be.B['full']
    zh = f['floor_z'] + 0.10
    for k, i in enumerate(who):
        r = rows[i]
        qd = r.get('model_qd')
        if qd not in ('Q1', 'Q2', 'Q3', 'Q4'):
            continue
        Tb = np.asarray(r['Tb'])
        Rp = Rotation.from_quat(pq[k][[1, 2, 3, 0]]).as_matrix()
        Ti = np.linalg.inv(Tb)
        c = (Ti @ np.r_[pc[k] + 0.01 * Rp[:, 2], 1.0])[:3]
        m = Ti[:3, :3] @ Rp[:, 2]
        if abs(m[0]) < 0.5:
            continue
        x = c[0] - m[2] * (zh - c[2]) / m[0]
        far, left = {'Q1': (1, 0), 'Q2': (1, 1), 'Q3': (0, 1), 'Q4': (0, 0)}[qd]
        yc = f['pair_centres_y'][0 if left else 1]
        cf = f['clip_first_m']
        half = cf[qd] + f['tine_pitch'][qd] * np.arange(f['tines_from_clip'][qd])
        xs = np.sort(f.get('clip_x', 0.0) + (half if far else -half))
        xs -= np.tan(np.radians(f['tine_lean_deg'][qd])) * (zh - f['floor_z'])
        g = int(np.searchsorted(xs, x)) - 1
        if 0 <= g < len(xs) - 1 and abs(c[1] - yc) < 0.06:
            landed[k] = g
    target = np.asarray([int(rows[i].get('gap', -1)) for i in who])
    qd = [rows[i].get('model_qd') for i in who]
    up_np = up.cpu().numpy()
    yaw_err = np.full(env.num_envs, 180.0)
    lean = np.degrees(np.arcsin(np.abs(up_np[:, 2]).clip(max=1.0)))
    lean_limit = np.zeros(env.num_envs)
    for k, q in enumerate(qd):
        if q not in ('Q1', 'Q2', 'Q3', 'Q4'):
            continue
        nvec = np.asarray(rows[who[k]]['Tb'])[:3, 0]
        uh = up_np[k, :2]
        nh = nvec[:2]
        cosang = abs(float(uh @ nh)) / max(float(np.linalg.norm(uh) * np.linalg.norm(nh)), 1e-9)
        yaw_err[k] = math.degrees(math.acos(np.clip(cosang, -1.0, 1.0)))
        lean_limit[k] = abs(float(be.B['full']['tine_lean_deg'][q])) + 12.0
    quadrant_ok = (yaw_err < 15.0) & (lean < lean_limit) & (st['low'].cpu().numpy() < 0.095)
    in_quadrant = np.asarray(lifted_at_lift, dtype=bool) & quadrant_ok & (landed >= 0)
    exact = (landed == target) & (target >= 0)
    low = st['low'].cpu().numpy()
    result = g0_success_mask(seated.cpu().numpy(), in_quadrant, exact)
    details = dict(seated=seated.cpu().numpy(), in_quadrant=in_quadrant,
                   exact=exact, low_m=low, yaw_error_deg=yaw_err, lean_deg=lean,
                   quadrant_ok=quadrant_ok)
    return result, landed, details


def g0_pose_error(reference_pos, reference_quat_wxyz, actual_pos, actual_quat_wxyz):
    """Per-sample TCP translation [m] and shortest quaternion angle [rad]."""
    import numpy as np
    rp, ap = np.asarray(reference_pos, float), np.asarray(actual_pos, float)
    rq, aq = np.asarray(reference_quat_wxyz, float), np.asarray(actual_quat_wxyz, float)
    if rp.shape != ap.shape or rq.shape != aq.shape or rp.shape[:-1] != rq.shape[:-1] or rp.shape[-1] != 3 or rq.shape[-1] != 4:
        raise ValueError('G0 reference/actual TCP pose arrays must align as (...,3) and (...,4)')
    if not all(np.isfinite(a).all() for a in (rp, ap, rq, aq)):
        raise ValueError('G0 TCP pose comparison contains a non-finite value')
    rn = np.linalg.norm(rq, axis=-1, keepdims=True)
    an = np.linalg.norm(aq, axis=-1, keepdims=True)
    if (rn < 1e-9).any() or (an < 1e-9).any():
        raise ValueError('G0 TCP pose comparison contains a zero quaternion')
    dot = np.abs(np.sum((rq / rn) * (aq / an), axis=-1)).clip(0.0, 1.0)
    return np.linalg.norm(ap - rp, axis=-1), 2.0 * np.arccos(dot)


def g0_error_summary(values, scale=1.0):
    """Compact finite max/mean/p95 summary after applying a display scale."""
    import numpy as np
    x = np.asarray(values, dtype=float) * float(scale)
    x = x[np.isfinite(x)]
    return dict(count=int(x.size), mean=float(x.mean()) if x.size else None,
                p95=float(np.percentile(x, 95)) if x.size else None,
                max=float(x.max()) if x.size else None)


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


_CL_CALLS = [0]
_CL_POL = dict(policy=None, det=False, dr=False, dr_full=False, seed=0, cache={})   # (10/15) insert RL: loaded policy + per-env (obs, act, logp, params)

INSERT_OBS = 16      # quadrant one-hot 4, gap / 11, basket dx dy dyaw, in-hand plate offset (3) + rotvec (3), believed gap error (2)
INSERT_ACT = 5       # aim dx, aim dy, lean factor, release dz, yaw about vertical (tanh-squashed, scaled by INSERT_SCALE)
INSERT_SCALE = (0.03, 0.02, 0.5, 0.01, 4.0)     # m, m, lean factor, m, deg
INSERT_OBS_SCALE = (1, 1, 1, 1, 1, 10, 10, 5, 10, 10, 10, 0.5, 0.5, 0.5, 100, 100)   # (10/15 21:0x) policy input scale


def insert_policy_params(r, k, H, Tb):
    """(10/15) once per env: observation -> sampled (or mean) correction from the insert policy, cached by env index.
    Also draws the env's model error (align.dr) when --cl_dr: the planner aims at the gap shifted by it."""
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation as R
    c = _CL_POL['cache']
    if k in c:
        return c[k]
    import zlib
    rng = np.random.default_rng(zlib.crc32(f"dr|{r['name']}|{k}|{_CL_POL['seed']}".encode()))
    qd = r['model_qd']
    bias = np.zeros(2)
    if _CL_POL['dr']:
        dr = E_CFG()['align']['dr']
        bias = np.array([rng.uniform(-1, 1) * dr['gap_x_m'][qd], rng.uniform(-1, 1) * dr['gap_y_m']])
    seen = bias + rng.normal(0, 0.002, 2)
    yaw_bias, hand = 0.0, np.eye(4)
    if _CL_POL['dr'] and _CL_POL['dr_full']:                # unobserved: marker yaw error, in-hand plate pose bias
        dr = E_CFG()['align']['dr']
        yaw_bias = float(rng.uniform(-1, 1) * dr['yaw_deg'])
        hand[:3, :3] = R.from_rotvec(rng.normal(0, 1, 3) / np.sqrt(3) * np.radians(dr['in_hand_deg'])).as_matrix()
        hand[:3, 3] = rng.normal(0, 1, 3) / np.sqrt(3) * dr['in_hand_mm'] / 1000
    Tb = np.asarray(Tb)
    yaw_b = np.arctan2(Tb[1, 0], Tb[0, 0])
    obs = np.concatenate([[qd == q for q in ('Q1', 'Q2', 'Q3', 'Q4')], [r.get('gap', 0) / 11.0],
                          [Tb[0, 3] - 0.81, Tb[1, 3] + 0.19, yaw_b], H[:3, 3], R.from_matrix(H[:3, :3]).as_rotvec(), seen]).astype(np.float32)
    act, logp = np.zeros(INSERT_ACT, np.float32), 0.0
    pol = _CL_POL['policy']
    if pol is not None:
        with torch.no_grad():
            mu, std = pol(torch.as_tensor(obs)[None])
            a = mu if _CL_POL['det'] else mu + std * torch.randn_like(mu)
            logp = float(torch.distributions.Normal(mu, std).log_prob(a).sum())
            act = a[0].numpy()
    s = np.tanh(act) * np.array(INSERT_SCALE)
    c[k] = dict(obs=obs, act=act, logp=logp, bias=bias, dx=float(s[0] + bias[0]), dy=float(s[1] + bias[1]),
                lean=float(s[2]), dz=float(s[3]), yaw=float(s[4] + yaw_bias), hand=hand)
    return c[k]


def load_insert_policy(path):
    import torch
    from model import InsertPolicy
    pol = InsertPolicy(INSERT_OBS, INSERT_ACT)
    sd = torch.load(path, map_location='cpu')
    pol.load_state_dict(sd, strict='obs_scale' in sd)        # runs before 21:0x saved no input scale (= ones)
    return pol.eval()


def closed_loop_paths(env, rows, who, above_m=0.11, extra=0.005, dz=0.0, lean=1.0, dx=0.0, top=False, dx_q=None, lean_q=None, dz_q=None, edge_q=None, pin_dx=None, top_q=None, noise=None, blind_h=None, noise_fp=None, gate=None):
    """(10/14) per env (above, descend) joint paths (deg) from the plate actually in the hand now: the plate turns ~9 deg
    in the grip on the carry (Isaac), so the planned descent lands it 3-4 gaps off.  Keeps the plate's turn about the
    vertical small: shortest rotation of its normal onto the row axis, centre on the row's target_centre (base)."""
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    sys.path.insert(0, str(ROOT / 'calibration'))
    from robot_model import Kinematics
    kin, tool = Kinematics(), np.eye(4)
    tool[2, 3] = 0.165
    robot, plate = env.scene['robot'], env.scene['plate']
    qs = robot.data.joint_pos[:, :6].cpu().numpy()
    pp = (plate.data.root_pos_w - env.scene.env_origins).cpu().numpy()
    pq = plate.data.root_quat_w.cpu().numpy()
    _CL_CALLS[0] += 1                                       # re-aim call number: independent noise per iteration, same per env
    out, lean_used, gated = [], [], []
    for k, i in enumerate(who):
        r = rows[i]
        if 'target_centre' not in r:
            out.append((None, None))
            continue
        Rp = R.from_quat(pq[k][[1, 2, 3, 0]]).as_matrix()
        pk = pp[k].copy()
        if noise:                                            # (10/15) measurement noise of the in-hand plate pose (real: FP)
            import zlib                                      # reproducible across processes (Codex U6: hash() is salted)
            rng_ = np.random.default_rng(zlib.crc32(f"{r['name']}|{k}|{noise}|call{_CL_CALLS[0]}".encode()))
            pk = pk + rng_.normal(0, noise[0] / 1000 / np.sqrt(3), 3)
            Rp = R.from_rotvec(rng_.normal(0, np.radians(noise[1]) / np.sqrt(3), 3)).as_matrix() @ Rp
        if noise_fp:                                         # (10/15) Codex AC empirical FP model: basket-axis heavy-tailed position
            import zlib                                      # (inliers p50, 15 % outliers matching p90/p95) + small normal tilt
            rng_ = np.random.default_rng(zlib.crc32(f"fp|{r['name']}|{k}|call{_CL_CALLS[0]}".encode()))
            Rb = np.asarray(r['Tb'])[:3, :3]
            e = []
            for ax in ('x', 'y', 'z'):
                st = noise_fp['position_noise_model']['pooled'][ax]
                e.append(rng_.normal(0, st['p95_abs'] / 0.967) if rng_.random() < 0.15 else rng_.normal(0, st['p50_abs'] / 0.6745))
            pk = pk + Rb @ np.array(e) / 1000
            nn = noise_fp['plate_normal_noise_model']['pooled']
            rv = np.radians([rng_.normal(0, nn[f'normal_{a}_deg']['std_sample']) for a in ('x', 'y', 'z')])
            Rp = R.from_rotvec(Rb @ rv).as_matrix() @ Rp
        Tpl = np.eye(4)
        Tpl[:3, :3], Tpl[:3, 3] = Rp, pk + 0.01 * Rp[:, 2]
        Tt = kin.fk(qs[k]) @ tool
        H = np.linalg.inv(Tt) @ Tpl                          # plate in the TCP frame as it is now
        if blind_h and r['model_qd'] in blind_h:            # (10/15) no measurement: the quadrant's mean in-hand pose (tuning runs)
            Hm = np.asarray(blind_h[r['model_qd']])          # with a gate: keep the measurement unless it is an outlier
            if gate is None or np.linalg.norm(H[:3, 3] - Hm[:3, 3]) * 1000 > gate:
                H = Hm
                Rp = (Tt @ H)[:3, :3]
                gated.append(1)
        import basket_env as be
        Tb = np.asarray(r['Tb'])
        pol = insert_policy_params(r, k, H, Tb) if (_CL_POL['policy'] is not None or _CL_POL['dr']) else None
        path = []
        # (10/14 night) no IK at the full lean: first turn the plate more about its normal (a vertical plate jams in the
        # leaning tines - Q2 never seated below 60 mm with lean 0/0.5), only then lean less
        for lf, wide in ((1.0, False), (1.0, True), (0.5, False), (0.0, False)):
            ln_, dz_ = (lean_q or {}).get(r['model_qd'], lean), (dz_q or {}).get(r['model_qd'], dz)
            dx_ = (pin_dx or {}).get(r['model_qd'], {}).get(str(r.get('pin')), (dx_q or {}).get(r['model_qd'], dx))
            dy_ = yaw_ = 0.0
            Hp = H
            if pol is not None:                              # (10/15 insert RL) policy correction + the env's model error
                ln_, dz_, dx_, dy_, yaw_ = ln_ + pol['lean'], dz_ + pol['dz'], dx_ + pol['dx'], pol['dy'], pol['yaw']
                Hp = H @ pol['hand']                         # the planner's (biased) belief of the plate in the hand
            path = _cl_one(kin, tool, be, r, Tb, Rp, Hp, qs[k], ln_ * lf, above_m, extra, dz_, dx_, (top_q or {}).get(r['model_qd'], top), (edge_q or {}).get(r['model_qd']), wide=wide, dy=dy_, yaw_deg=yaw_)
            if len(path) == 13:
                lean_used.append(lf if not wide else 1.01)
                break
        out.append((np.array(path[:1]), np.array(path[1:])) if len(path) == 13 else (None, None))
        if len(path) < 13:
            print(f"[replay] closed loop: {r['name']} env {k} no IK at step {len(path)}", flush=True)
    import collections
    print(f"[replay] closed loop lean used (1.01 = full lean, wider turn): {sorted(collections.Counter(lean_used).items())}"
          f"{f'; mean in-hand pose used for {len(gated)}/{len(who)}' if blind_h else ''}", flush=True)
    return out


def _cl_one(kin, tool, be, r, Tb, Rp, H, q0, lean, above_m, extra, dz, dx, top, edge=None, wide=False, dy=0.0, yaw_deg=0.0):
    """One closed-loop (above + descent) joint path in deg for this tine lean; shorter than 13 = no IK."""
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    ln = np.radians(lean * be.B['full']['tine_lean_deg'][r['model_qd']])
    up = Tb[:3, :3] @ np.array([-np.sin(ln), 0.0, np.cos(ln)])   # tine axis (+ lean = top toward basket -x)
    nt = Tb[:3, :3] @ np.array([np.cos(ln), 0.0, np.sin(ln)])    # plate normal parallel to the tines
    nt = nt if nt @ Rp[:, 2] >= 0 else -nt
    if yaw_deg:                                              # (10/15 insert RL) turn the plate face about the vertical
        Rz = R.from_rotvec([0.0, 0.0, np.radians(yaw_deg)]).as_matrix()
        up, nt = Rz @ up, Rz @ nt
    f_ = be.B['full']                                       # (10/14 night) keep the shifted aim inside the quadrant's gaps:
    cf = f_['clip_first_m'][r['model_qd']] if isinstance(f_['clip_first_m'], dict) else f_['clip_first_m']
    hx = cf + f_['tine_pitch'][r['model_qd']] * np.arange(f_['tines_from_clip'][r['model_qd']])
    xs = np.sort(f_.get('clip_x', 0.0) + (hx if r['model_qd'] in ('Q1', 'Q2') else -hx))
    gx = (xs[:-1] + xs[1:]) / 2                              # gap centres (basket x); +3.5 cm put Q2 pins 1-2 past the last tine
    t_b = np.linalg.inv(Tb) @ np.r_[np.asarray(r['target_centre']), 1.0]
    if edge is not None:                                     # only where asked (Q2): the near rows need the bias shift past their
        dx = float(np.clip(t_b[0] + dx, gx.min(), gx.max() - edge) - t_b[0])   # end gap (+2 gaps), Q1 lost 12/90 to -1
    gap_b = np.asarray(r['target_centre']) - [0, 0, 0.165] + Tb[:3, 0] * dx + Tb[:3, 1] * dy   # gap centre on the floor (base)
    rot, _ = R.align_vectors([nt], [Rp[:, 2]])
    R0 = rot.as_matrix() @ Rp
    g = R0 @ np.linalg.inv(H)[:3, 3]                        # grip (TCP) from the plate centre, turned with the plate
    g -= (g @ nt) * nt
    a0 = np.arctan2(np.cross(g, up) @ nt, g @ up) if top else 0.0   # (10/14) turn the plate about its normal: grip at its top
    path = []                                              # (a side grip lets the plate spin 30-40 deg on release)
    for dphi in ((30, -30, 40, -40, 50, -50, 60, -60) if wide else (0, 10, -10, 20, -20) if top else (0,)):
        Td = np.eye(4)
        Td[:3, :3] = R.from_rotvec(nt * (a0 + np.radians(dphi))).as_matrix() @ R0
        path, prev = [], q0
        for lift in np.linspace(above_m, -extra, 13):
            Td[:3, 3] = gap_b + up * 0.165 + [0, 0, lift + dz]   # bottom over the gap, centre along the tine
            sol = None
            for sd in ([prev] if path else [prev, np.radians(r.get('above_deg', np.degrees(prev)))]):
                try:
                    cand = kin.ik(Td @ np.linalg.inv(H) @ np.linalg.inv(tool), sd)
                except RuntimeError:
                    continue
                if np.degrees(np.abs(cand - prev)).max() < (45 if not path else 8):
                    sol = cand
                    break
            if sol is None:
                break
            path.append(np.degrees(sol))
            prev = sol
        if len(path) == 13:
            break
    return path

# ---------------------------------------------------------------- P1 replay of the real runs
def run_replay(args):
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    import basket_env as be
    import plate_env as pe
    import env as E
    if args.g0 and (args.attach or not args.closed_loop or not args.plans or not args.out):
        raise SystemExit('--g0 requires --plans, --closed_loop, an explicit new --out path, and no --attach')
    if args.g0 and Path(args.out).exists():
        raise SystemExit(f'G0 refuses to overwrite {args.out}')
    runs = [] if args.plans else [json.loads(l) for l in open(args.labels)]
    rows = []
    for r in ([json.loads(l) for l in open(args.plans)] if args.plans else []):   # planned, not run: 'outcome' = planned
        legs = json.loads(Path(r['plan']).read_text())['legs']
        rows.append(dict(name=r['name'], outcome='planned', plate=r['plate'], Tb=r['Tb'], grasp=legs, slot=legs,
                         **{k: r[k] for k in ('model_qd', 'gap', 'pin', 'target_centre') if k in r}))
    for r in runs:
        d = Path(r['run']) if Path(r['run']).is_absolute() else ROOT / 'calibration/data/real_pipeline' / r['run']
        ev = [json.loads(x) for x in open(d / 'robot_state.jsonl') if '"event"' in x]
        grasp = json.loads(Path(next(e['plan'] for e in ev if e['event'] == 'plan')).read_text())['legs']
        slot = json.loads(Path(next(e['plan'] for e in ev if e['event'] == 'slot_plan')).read_text())['legs']
        photo = d / 'wrist_above.json' if (d / 'wrist_above.json').exists() else d / 'wrist_pre.json'
        rows.append(dict(name=d.name, outcome=r['outcome'], plate=next(e for e in ev if e['event'] == 'plan')['plate_root_fp'],
                         Tb=json.loads(photo.read_text())['T_base_basket'], grasp=grasp, slot=slot))
    n = len(rows) * args.per_run
    _CL_POL.update(policy=load_insert_policy(args.cl_policy) if args.cl_policy else None, det=args.cl_policy_det,
                   dr=args.cl_dr, dr_full=args.cl_dr_full, seed=args.seed, cache={})
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
    # Keep the old >5 cm lift metric for in-quadrant classification. Finger/rim checks are
    # serialized diagnostics only; they are not part of the established seated_exact gate.
    physical_lift_diagnostic = None
    if args.g0:
        physical_lift_diagnostic = g0_physical_grasp_diagnostics(env, pe)
    ti = robot.find_bodies('tcp')[0][0]                        # (10/13) the plate in the hand after the lift, world poses
    in_hand = dict(tcp_pos=(robot.data.body_pos_w[:, ti] - env.scene.env_origins).cpu().numpy().tolist(),
                   tcp_quat=robot.data.body_quat_w[:, ti].cpu().numpy().tolist(),
                   plate_pos=(env.scene['plate'].data.root_pos_w - env.scene.env_origins).cpu().numpy().tolist(),
                   plate_quat=env.scene['plate'].data.root_quat_w.cpu().numpy().tolist(),
                   joints=robot.data.joint_pos[:, :6].cpu().numpy().tolist())
    if args.attach:                                            # rigid in-hand plate (see attach), no extra noise
        attach(env, torch.as_tensor(held, device=dev))
    drive(env, L('to_above', 'slot'), ~open_, args.speed)
    # G0 starts at the first stop above the basket, before any closed-loop re-aim (stage 4).
    g0_record = None
    g0_start = None
    if args.g0:
        g0_record = dict(obs=[], pos=[], quat=[], next_pos=[], next_quat=[], next_joints=[], phase=[], open=[],
                         joints=[], physical_held=[], finger_ratio=[], tcp_local=[], rim_distance=[],
                         released_state=[], released=np.zeros(n, dtype=bool))
        g0_start = capture_g0_state(env, pe)
    if all('descend_pre' in rows[i]['slot'] for i in who):    # taught slot paths only (quadrant plans: to_above -> descend)
        drive(env, L('descend_pre', 'slot'), ~open_, args.speed / 4, trace=g0_record, trace_phase='preinsert')
    descend, ascend = L('descend', 'slot'), L('ascend', 'slot')
    if args.closed_loop:                                       # (10/14) re-aim from the plate actually in the hand
        fix = first_fix = None
        for it in range(args.cl_iter):                       # (10/14 night) measure + re-aim again at the above stop
            hs = [float(h) for h in args.cl_heights.split(',')] if args.cl_heights else [0.11]
            prev_fix, fix = fix, closed_loop_paths(env, rows, who, above_m=hs[min(it, len(hs) - 1)], dz=args.cl_dz, lean=args.cl_lean, dx=args.cl_dx, top=args.cl_top,
                                    dx_q={k: float(v) for k, v in (x.split(':') for x in args.cl_dx_q.split(',') if x)},
                                    lean_q={k: float(v) for k, v in (x.split(':') for x in args.cl_lean_q.split(',') if x)},
                                    dz_q={k: float(v) for k, v in (x.split(':') for x in args.cl_dz_q.split(',') if x)},
                                    edge_q={k: float(v) for k, v in (x.split(':') for x in args.cl_edge_q.split(',') if x)},
                                    pin_dx=json.loads(Path(args.cl_pin_dx).read_text()) if args.cl_pin_dx else None,
                                    top_q={k: bool(int(v)) for k, v in (x.split(':') for x in args.cl_top_q.split(',') if x)},
                                    noise=args.cl_noise, blind_h=json.loads(Path(args.cl_blind).read_text()) if args.cl_blind else None,
                                    noise_fp=json.loads(Path(args.cl_noise_fp).read_text()) if args.cl_noise_fp else None, gate=args.cl_gate)
            new_ok = sum(a is not None for a, _ in fix)
            if prev_fix is not None:                         # no IK now: keep the previous re-aim (the robot is there)
                fix = [f_ if f_[0] is not None else p_ for f_, p_ in zip(fix, prev_fix)]
            if it == 0:
                first_fix = fix
            print(f"[replay] closed loop {it + 1}/{args.cl_iter}: re-solved {new_ok}/{n} (with kept earlier aims {sum(a is not None for a, _ in fix)})", flush=True)
            drive(env, [a for a, _ in fix], ~open_, args.speed / 4, hold=30, trace=g0_record,
                  trace_phase=f'reaim_{it}')
        descend = [b if b is not None else d for (_, b), d in zip(fix, descend)]
        top_ = first_fix if first_fix is not None else fix   # (10/14 night) a lower re-aim: back up through it to the first stop
        ascend = [np.vstack([b[::-1], a] + ([t[0]] if t[0] is not None and t is not f_ else [])) if b is not None else u
                  for (a, b), u, t, f_ in zip(fix, L('ascend', 'slot'), top_, fix)]   # straight back up
    st_pre = be.basket_state(env)
    pre = dict(tcp_pos=(robot.data.body_pos_w[:, ti] - env.scene.env_origins).cpu().numpy().tolist(),   # (10/14) plate + TCP before the descent
               tcp_quat=robot.data.body_quat_w[:, ti].cpu().numpy().tolist(),
               plate_pos=(env.scene['plate'].data.root_pos_w - env.scene.env_origins).cpu().numpy().tolist(),
               plate_quat=env.scene['plate'].data.root_quat_w.cpu().numpy().tolist(),
               joints=robot.data.joint_pos[:, :6].cpu().numpy().tolist())
    drive(env, descend, ~open_, args.speed / 20, hold=30, trace=g0_record, trace_phase='insert')
    trace = dict(descended=dict(pos=(env.scene['plate'].data.root_pos_w - env.scene.env_origins).cpu().numpy().tolist(), quat=env.scene['plate'].data.root_quat_w.cpu().numpy().tolist()))   # (10/14) where the plate moves after the descent
    if args.attach:
        detach(env, torch.ones(n, dtype=torch.bool, device=dev))
    drive(env, [None] * n, open_, args.speed, hold=45, trace=g0_record, trace_phase='release')         # release
    trace['released'] = dict(pos=(env.scene['plate'].data.root_pos_w - env.scene.env_origins).cpu().numpy().tolist(), quat=env.scene['plate'].data.root_quat_w.cpu().numpy().tolist())
    if args.g0:
        teacher_strict_success, st_teacher = inserted(env)
        teacher_grasp_maintained = g0_trace_grasp_maintained(g0_record['physical_held'], g0_record['released_state'])
        teacher_exact, teacher_landed, teacher_detail = g0_seated_exact(
            env, rows, who, st_teacher, be, pe, held)
        slot_yaw = env._slot[:, 3].cpu().numpy()
        g0_obs, g0_act, clip_count, frame_count = g0_actions_from_trace(
            g0_record, slot_yaw, E_CFG()['action']['step_max'])
        restore_g0_state(env, g0_start)
        restored_tcp, restored_quat = pe.tcp_pose(env)
        start_pos_err, start_rot_err = g0_pose_error(
            g0_start['tcp_pos'].cpu().numpy(), g0_start['tcp_quat'].cpu().numpy(),
            (restored_tcp - env.scene.env_origins).cpu().numpy(), restored_quat.cpu().numpy())
        start_joint_err = (torch.rad2deg(env.scene['robot'].data.joint_pos - g0_start['q'])[:, :6]
                           .abs().amax(dim=1).cpu().numpy())
        student_released = np.zeros(n, dtype=bool)
        replay_action_trace, replay_ref_pos_trace, replay_ref_quat_trace, replay_ref_joint_trace = [], [], [], []
        replay_cursor_trace, replay_phase_trace = [], []
        replay_action_clipped = 0
        phases = np.stack(g0_record['phase'], axis=1).astype(str)
        if phases.shape != g0_act.shape[:2] or not np.all(phases == phases[:1]):
            raise RuntimeError('G0 phase trace is not aligned across vectorized environments')
        phase_1d = phases[0]
        if not np.any(phase_1d == 'insert'):
            raise RuntimeError('G0 trace has no insert-complete phase; refusing to open the gripper')
        gate_mask = g0_phase_gate_mask(phase_1d)
        phase_gate_wait_frames = np.zeros(n, dtype=np.int32)
        gate_episode_wait_frames = np.zeros(n, dtype=np.int32)
        cursor_clip_wait_frames = np.zeros(n, dtype=np.int32)
        cursor_clip_timeout_count = np.zeros(n, dtype=np.int32)
        phase_gate_timeout = np.zeros(n, dtype=bool)
        release_gate_passed = np.zeros(n, dtype=bool)
        student_pos_trace, student_quat_trace, student_joint_trace = [], [], []
        student_finger_trace, student_tcp_local_trace, student_rim_trace = [], [], []
        student_held_trace, student_released_trace, student_open_trace = [], [], []
        env._basket_tracking_mode = True
        cursor = 0
        gate_latched = np.zeros(n, dtype=bool)
        gate_timeout_frames = 60  # 2 s at 30 Hz; timeout is a replay failure, never a skipped success.
        while cursor < g0_act.shape[1]:
            tp, tq = pe.tcp_pose(env)
            current_pos = (tp - env.scene.env_origins).cpu().numpy()
            current_quat = tq.cpu().numpy()
            is_gate = bool(gate_mask[cursor])
            if is_gate:
                target_pos = np.asarray(g0_record['next_pos'][cursor])
                target_quat = np.asarray(g0_record['next_quat'][cursor])
                pos_err, rot_err = g0_pose_error(target_pos, target_quat, current_pos, current_quat)
                aligned = g0_gate_alignment_mask(pos_err, rot_err)
                gate_latched |= aligned & teacher_exact
                pending = teacher_exact & ~gate_latched
                if not pending.any():
                    if phase_1d[cursor] == 'insert':
                        # A boundary timeout is diagnostic only. The ordered insert phase
                        # completed (aligned or timed out), so the recorded release may run.
                        release_gate_passed[:] = True
                    cursor += 1
                    gate_latched[:] = False
                    gate_episode_wait_frames[:] = 0
                    continue
                phase_gate_wait_frames[pending] += 1
                gate_episode_wait_frames[pending] += 1
                gate_latched, expired = g0_gate_timeout_state(
                    teacher_exact, gate_latched, gate_episode_wait_frames, gate_timeout_frames)
                phase_gate_timeout |= expired
                pending = teacher_exact & ~gate_latched
                if not pending.any():
                    continue
            else:
                target_pos = np.asarray(g0_record['next_pos'][cursor])
                target_quat = np.asarray(g0_record['next_quat'][cursor])
                pending = np.ones(n, dtype=bool)
            # Recompute from the live student pose to the next absolute teacher pose.
            # During a phase gate, already aligned or timed-out successful episodes hold
            # position; lagging successes keep tracking the same boundary pose.
            track_action, track_clipped = g0_tracking_action(
                current_pos, current_quat, target_pos, target_quat,
                slot_yaw, E_CFG()['action']['step_max'])
            active = pending.copy() if is_gate else np.ones(n, dtype=bool)
            if is_gate:
                active &= ~gate_latched
            track_action[~active] = 0.0
            replay_action_clipped += int((track_clipped & active).sum())
            if not is_gate:
                cursor_clip_wait_frames = np.where(teacher_exact & track_clipped,
                                                   cursor_clip_wait_frames + 1, 0)
                clip_expired = teacher_exact & track_clipped & (cursor_clip_wait_frames >= gate_timeout_frames)
                cursor_clip_timeout_count += clip_expired.astype(np.int32)
            else:
                clip_expired = np.zeros(n, dtype=bool)
            requested_open = g0_act[:, cursor, 6] > 0.5
            allowed_open = requested_open & release_gate_passed
            replay_action = np.concatenate([track_action, allowed_open[:, None].astype(np.float32)], axis=1)
            replay_action_trace.append(replay_action.astype(np.float32))
            replay_ref_pos_trace.append(target_pos.copy())
            replay_ref_quat_trace.append(target_quat.copy())
            replay_ref_joint_trace.append(np.asarray(g0_record['next_joints'][cursor]).copy())
            replay_cursor_trace.append(np.full(n, cursor, dtype=np.int32))
            replay_phase_trace.append(np.full(n, phase_1d[cursor], dtype=object))
            student_pos_trace.append(current_pos)
            student_quat_trace.append(current_quat)
            student_joint_trace.append(torch.rad2deg(env.scene['robot'].data.joint_pos[:, :6]).cpu().numpy())
            diag = g0_physical_grasp_diagnostics(env, pe)
            student_finger_trace.append(diag['finger_ratio'])
            student_tcp_local_trace.append(diag['tcp_local'])
            student_rim_trace.append(diag['rim_distance'])
            student_held_trace.append(diag['predicate'])
            student_released_trace.append(student_released.copy())
            student_open_trace.append(allowed_open.copy())
            a = torch.as_tensor(track_action, dtype=torch.float32, device=dev)
            o = torch.as_tensor(allowed_open.astype(np.float32), dtype=torch.float32, device=dev)
            env.step(E.to_env_action(env, a, o))
            student_released |= allowed_open
            if not is_gate:
                # Time-dilate the recorded targets when SafeDiffIK saturates instead of
                # advancing the teacher stream faster than the capped student can follow.
                if g0_replay_cursor_should_advance(False, teacher_exact, track_clipped & active,
                                                   None, clip_expired):
                    cursor += 1
                    cursor_clip_wait_frames[:] = 0
            elif g0_replay_cursor_should_advance(True, teacher_exact, track_clipped & active,
                                                 teacher_exact & ~gate_latched):
                if phase_1d[cursor] == 'insert':
                    release_gate_passed[:] = True
                cursor += 1
                gate_latched[:] = False
                gate_episode_wait_frames[:] = 0
        env._basket_tracking_mode = False
        student_strict_success, st_student = inserted(env)
        student_exact, student_landed, student_detail = g0_seated_exact(
            env, rows, who, st_student, be, pe, held)
        teacher_count = int(teacher_exact.sum())
        reproduced = int((teacher_exact & student_exact).sum())
        gate, ratio = g0_gate_status(teacher_count, reproduced)
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        data_out = output.with_suffix('.npz')
        if data_out.exists():
            raise SystemExit(f'G0 refuses to overwrite {data_out}')
        teacher_open = np.stack(g0_record['open'], axis=1).astype(bool)
        student_open = np.stack(student_open_trace, axis=1).astype(bool)
        student_grasp_maintained = g0_trace_grasp_maintained(student_held_trace, student_released_trace)
        student_pos = np.stack(student_pos_trace, axis=1)
        student_quat = np.stack(student_quat_trace, axis=1)
        replay_ref_pos = np.stack(replay_ref_pos_trace, axis=1)
        replay_ref_quat = np.stack(replay_ref_quat_trace, axis=1)
        tcp_pos_err, tcp_rot_err = g0_pose_error(replay_ref_pos, replay_ref_quat, student_pos, student_quat)
        teacher_joints = np.stack(replay_ref_joint_trace, axis=1)
        student_joints = np.stack(student_joint_trace, axis=1)
        joint_err_deg = np.max(np.abs(teacher_joints - student_joints), axis=2)
        teacher_finger = np.stack(g0_record['finger_ratio'], axis=1)
        student_finger = np.stack(student_finger_trace, axis=1)
        teacher_tcp_local = np.stack(g0_record['tcp_local'], axis=1)
        student_tcp_local = np.stack(student_tcp_local_trace, axis=1)
        teacher_rim = np.stack(g0_record['rim_distance'], axis=1)
        student_rim = np.stack(student_rim_trace, axis=1)
        g0_episodes = []
        teacher_open_idx = g0_first_true(teacher_open)
        student_open_idx = g0_first_true(student_open)
        teacher_diag = physical_lift_diagnostic
        for k in range(n):
            g0_episodes.append(dict(
                quadrant=rows[who[k]].get('model_qd'), pin=int(rows[who[k]].get('pin', -1)),
                target_gap=int(rows[who[k]].get('gap', -1)),
                teacher=dict(seated=bool(teacher_detail['seated'][k]),
                             in_quadrant=bool(teacher_detail['in_quadrant'][k]),
                             exact=bool(teacher_detail['exact'][k]),
                             low_mm=float(teacher_detail['low_m'][k] * 1000),
                             landed_gap=int(teacher_landed[k]),
                             strict_inserted=bool(teacher_strict_success[k])),
                student=dict(seated=bool(student_detail['seated'][k]),
                             in_quadrant=bool(student_detail['in_quadrant'][k]),
                             exact=bool(student_detail['exact'][k]),
                             low_mm=float(student_detail['low_m'][k] * 1000),
                             landed_gap=int(student_landed[k]),
                             strict_inserted=bool(student_strict_success[k])),
                teacher_physical_lift_diagnostic=dict(
                    lifted_gt_50mm=bool(held[k]), finger_ratio=float(teacher_diag['finger_ratio'][k]),
                    finger_stall_ok=bool(teacher_diag['finger_ok'][k]),
                    tcp_plate_local_m=teacher_diag['tcp_local'][k].tolist(),
                    rim_radial_error_mm=float(teacher_diag['radial_error'][k] * 1000),
                    rim_height_error_mm=float(teacher_diag['height_error'][k] * 1000),
                    tcp_rim_distance_mm=float(teacher_diag['rim_distance'][k] * 1000),
                    rim_ok=bool(teacher_diag['rim_ok'][k]),
                    finger_stall_range=teacher_diag['finger_stall'],
                    rim_distance_tolerance_mm=teacher_diag['rim_tolerance'] * 1000),
                start_restore=dict(tcp_position_error_mm=float(start_pos_err[k] * 1000),
                                   tcp_orientation_error_deg=float(np.degrees(start_rot_err[k])),
                                   joint_max_error_deg=float(start_joint_err[k])),
                replay_alignment=dict(
                    tcp_position_error_mm=g0_error_summary(tcp_pos_err[k], 1000.0),
                    tcp_orientation_error_deg=g0_error_summary(tcp_rot_err[k], 180.0 / np.pi),
                    joint_max_error_deg=g0_error_summary(joint_err_deg[k]),
                    teacher_first_open_frame=teacher_open_idx[k],
                    student_first_open_command_frame=student_open_idx[k],
                    release_gate_passed=bool(release_gate_passed[k]),
                    phase_gate_timed_out=bool(phase_gate_timeout[k]),
                    phase_gate_wait_frames=int(phase_gate_wait_frames[k]),
                    teacher_grasp_maintained_diagnostic=bool(teacher_grasp_maintained[k]),
                    student_grasp_maintained_diagnostic=bool(student_grasp_maintained[k]))))
        replay_act = np.stack(replay_action_trace, axis=1)
        np.savez_compressed(data_out, obs=g0_obs, act=g0_act.astype(np.float32), replay_act=replay_act,
                            teacher_success=teacher_exact, student_success=student_exact,
                            lifted_at_lift=held,
                            teacher_grasp_maintained=teacher_grasp_maintained,
                            student_grasp_maintained_diagnostic=student_grasp_maintained,
                            teacher_landed_gap=teacher_landed, student_landed_gap=student_landed,
                            teacher_seated=teacher_detail['seated'], student_seated=student_detail['seated'],
                            teacher_in_quadrant=teacher_detail['in_quadrant'], student_in_quadrant=student_detail['in_quadrant'],
                            teacher_low_m=teacher_detail['low_m'], student_low_m=student_detail['low_m'],
                            teacher_physical_finger_ratio=teacher_finger, student_physical_finger_ratio=student_finger,
                            teacher_physical_tcp_plate_local_m=teacher_tcp_local, student_physical_tcp_plate_local_m=student_tcp_local,
                            teacher_physical_tcp_rim_distance_m=teacher_rim, student_physical_tcp_rim_distance_m=student_rim,
                            teacher_tcp_position_error_m=tcp_pos_err, teacher_tcp_orientation_error_rad=tcp_rot_err,
                            tcp_max_joint_error_deg=joint_err_deg, teacher_open_command=teacher_open,
                            student_open_command=student_open,
                            teacher_phase=phases, replay_phase=np.stack(replay_phase_trace, axis=1),
                            replay_teacher_cursor=np.stack(replay_cursor_trace, axis=1),
                            replay_reference_tcp_position_m=replay_ref_pos,
                            replay_reference_tcp_quaternion_wxyz=replay_ref_quat,
                            replay_release_gate_passed=release_gate_passed,
                            replay_phase_gate_timeout=phase_gate_timeout,
                            replay_phase_gate_wait_frames=phase_gate_wait_frames,
                            replay_target_clip_timeout_count=cursor_clip_timeout_count,
                            quadrant=np.asarray([rows[i].get('model_qd', '') for i in who]),
                            pin=np.asarray([rows[i].get('pin', -1) for i in who]),
                            target_gap=np.asarray([rows[i].get('gap', -1) for i in who]))
        per_q = {}
        for qd in ('Q1', 'Q2', 'Q3', 'Q4'):
            m = np.asarray([rows[i].get('model_qd') == qd for i in who])
            denom = int((teacher_exact & m).sum())
            num_ok = int((teacher_exact & student_exact & m).sum())
            per_q[qd] = dict(teacher_success=denom, reproduced=num_ok,
                             ratio=(num_ok / denom if denom else None))
        report = dict(gate=gate, criterion='same-start SafeDiffIK replay / teacher seated & in_quadrant & exact successes >= 0.95',
                      seated_exact_definition='0.020 m < low < 0.060 m AND in_quadrant AND landed_gap == target_gap; finger/rim metrics diagnostic only',
                      teacher_seated_exact=teacher_count, reproduced_seated_exact=reproduced,
                      reproduction_ratio=ratio, clipped_tcp_steps=clip_count, recorded_tcp_steps=frame_count,
                      physical_grasp_at_lift_diagnostic=int((teacher_diag['predicate'] & held).sum()),
                      lifted_at_lift=int(held.sum()), teacher_grasp_maintained_diagnostic=int(teacher_grasp_maintained.sum()),
                      per_quadrant=per_q, episodes=g0_episodes, actions=str(data_out), plans=args.plans,
                      attach=False, action_hz=E_CFG()['action']['hz'], step_max=E_CFG()['action']['step_max'],
                      replay_tracking_clipped_steps=replay_action_clipped,
                      phase_gates=dict(phases=['reaim_*', 'insert'], position_tolerance_mm=1.0,
                                       orientation_tolerance_deg=0.5, timeout_frames=gate_timeout_frames,
                                       timeout_seconds=gate_timeout_frames / E_CFG()['action']['hz'],
                                       timed_out_teacher_successes=int((teacher_exact & phase_gate_timeout).sum()),
                                       total_wait_frames=int(phase_gate_wait_frames.sum()),
                                       timeout_is_diagnostic_only=True, release_open_requires_insert_gate=True,
                                       target_clip_timeout_frames=gate_timeout_frames,
                                       target_clip_timeouts=int(cursor_clip_timeout_count.sum())),
                      replay_tracking_action_semantics='clip(teacher_next_TCP_pose - student_current_TCP_pose); hold each recorded target while a teacher-success SafeDiffIK correction saturates; act is clean teacher-state label, replay_act is executed tracking action')
        output.write_text(json.dumps(report, indent=2))
        print('[G0]', json.dumps(report), flush=True)
        env.close()
        return
    drive(env, ascend, open_, args.speed / 4)
    trace['ascended'] = dict(pos=(env.scene['plate'].data.root_pos_w - env.scene.env_origins).cpu().numpy().tolist(), quat=env.scene['plate'].data.root_quat_w.cpu().numpy().tolist())
    ok, st = inserted(env)
    ok = ok.cpu().numpy() & held
    # the real plates did not always go into the taught gap (tilt + 4 cm along the rows): judge "in A gap" too
    pitch = be.B['pins']['pitch']
    gap_err = ((st['d_bottom'] + pitch / 2) % pitch - pitch / 2).abs()
    sc = be.B['success']
    any_gap = ((gap_err < sc['bottom_tol_m']) & (st['low'] < sc['max_height_m']) & (st['normal_err'] < sc['max_normal_err_deg'])
               & (st['d_u'].abs() < sc['centre_tol_m'])).cpu().numpy() & held
    # (10/14) planned quadrant setups: judge the face against the rows of ITS quadrant (far tines lean the other way):
    # plate face across the rows (normal within 15 deg of n seen from above), standing (lean within the tine lean + 12 deg)
    up_, n_ = st['up'], st['n']
    yaw_err = torch.rad2deg(torch.acos((torch.abs((up_[:, :2] * n_[:, :2]).sum(1)) / up_[:, :2].norm(dim=1).clamp(min=1e-6)).clamp(max=1.0)))
    lean = torch.rad2deg(torch.asin(up_[:, 2].abs().clamp(max=1.0)))
    lean_ok = torch.tensor([abs(be.B['full'].get('tine_lean_deg', {}).get(next((qd for qd in ('Q1', 'Q2', 'Q3', 'Q4') if f'_{qd}_' in rows[i]['name']), 'Q3'), 12.5)) + 12.0
                            for i in who], device=dev)
    face_ok = ((yaw_err < 15.0) & (lean < lean_ok) & (st['low'] < 0.095) & (gap_err < sc['bottom_tol_m'])).cpu().numpy() & held
    # (10/14 보스) planned rows carrying their target gap (model quadrant + gap index): the plate must stand in THAT gap of
    # its quadrant (lowest rim point between the two tine bases, within the pair's rows), not just in some gap
    landed = np.full(n, -1)
    # which two tines the plate stands between: its mid-plane crossed with the row at mid-tine height, against the tines
    # there (they lean); the lowest point is on the foot ring, up to the dish depth (~1 gap) off the mid-plane
    from scipy.spatial.transform import Rotation as R_
    pc = (env.scene['plate'].data.root_pos_w - env.scene.env_origins).cpu().numpy()
    pq = env.scene['plate'].data.root_quat_w.cpu().numpy()
    f_ = be.B['full']
    zh = f_['floor_z'] + 0.10                               # near the tine tips (11.5 cm): where 보스 counts from above
    for k, i in enumerate(who):
        r = rows[i]
        if 'model_qd' not in r:
            continue
        T = np.asarray(r['Tb'])
        Rp = R_.from_quat(pq[k][[1, 2, 3, 0]]).as_matrix()
        Ti = np.linalg.inv(T)
        c = (Ti @ np.r_[pc[k] + 0.01 * Rp[:, 2], 1.0])[:3]
        m = Ti[:3, :3] @ Rp[:, 2]
        if abs(m[0]) < 0.5:                                    # face not across the rows (fallen / turned)
            continue
        x = c[0] - (m[2] * (zh - c[2])) / m[0]                # y = the plate centre's
        far, left = {'Q1': (1, 0), 'Q2': (1, 1), 'Q3': (0, 1), 'Q4': (0, 0)}[r['model_qd']]
        yc = f_['pair_centres_y'][0 if left else 1]
        cf = f_.get('clip_first_m', 0.01)
        half = (cf[r['model_qd']] if isinstance(cf, dict) else cf) + f_['tine_pitch'][r['model_qd']] * np.arange(f_['tines_from_clip'][r['model_qd']])
        xs = np.sort(f_.get('clip_x', 0.0) + (half if far else -half))      # tine bases, robot side -> far (basket_gaps order)
        xs = xs - np.tan(np.radians(f_['tine_lean_deg'][r['model_qd']])) * (zh - f_['floor_z'])   # + lean = top toward the robot
        g = int(np.searchsorted(xs, x)) - 1
        if 0 <= g < len(xs) - 1 and abs(c[1] - yc) < 0.06:
            landed[k] = g
    target = np.array([rows[i].get('gap', -1) if 'model_qd' in rows[i] else -1 for i in who])
    pin_ok = np.where(target >= 0, landed == target, True)
    loose = ((yaw_err < 15.0) & (lean < lean_ok) & (st['low'] < 0.095)).cpu().numpy() & held   # the any-gap error (2 cm pitch) does not hold per quadrant
    face_ok = np.where(target >= 0, loose & pin_ok, face_ok)
    in_quadrant = loose & (landed >= 0)                       # (10/14 night, stage 2) standing in SOME gap of its quadrant
    seated = ((st['low'] < 0.060) & (st['low'] > 0.020)).cpu().numpy()   # (10/14 night, Codex U) down on the 49 mm rail, not perched;
    # (10/15 Codex AJ) lower bound: a mesh point below 20 mm would be through the floor grid, not seated (none in tonight's finals: min 26 mm)
    np.set_printoptions(precision=1, suppress=True)
    print('[replay] final d_bottom mm', (st['d_bottom'] * 1000).cpu().numpy()[::args.per_run], flush=True)
    print('[replay] final gap err mm  ', (gap_err * 1000).cpu().numpy()[::args.per_run], flush=True)
    print('[replay] final low mm      ', (st['low'] * 1000).cpu().numpy()[::args.per_run], flush=True)
    print('[replay] final normal deg  ', st['normal_err'].cpu().numpy()[::args.per_run], ' d_u mm', (st['d_u'] * 1000).cpu().numpy()[::args.per_run], flush=True)
    print('\n[replay] run                              real      sim inserted  (held)  pre d_bottom mm  pre normal err deg', flush=True)
    res = []
    for i, r in enumerate(rows):
        m = who == i
        res.append(dict(run=r['name'], real=r['outcome'], sim_rate=float(ok[m].mean()), any_gap=float(any_gap[m].mean()), quadrant_ok=float(face_ok[m].mean()), in_quadrant=float(in_quadrant[m].mean()), seated_in_quadrant=float((in_quadrant & seated)[m].mean()),
                        seated_exact=float((face_ok & seated)[m].mean()), held=float(held[m].mean()),
                        landed_gaps=landed[m].tolist(), target_gap=int(target[m][0]),
                        pre_d_bottom_mm=float(st_pre['d_bottom'][torch.as_tensor(m)].mean() * 1000),
                        pre_normal_err=float(st_pre['normal_err'][torch.as_tensor(m)].mean())))
        print(f"[replay] {r['name']:32s} {r['outcome'][:8]:8s}  {ok[m].mean():5.0%} any gap {any_gap[m].mean():4.0%} quadrant ok {face_ok[m].mean():4.0%} in quadrant {in_quadrant[m].mean():4.0%} seated exact {(face_ok & seated)[m].mean():4.0%} ({held[m].mean():4.0%})  "
              f"{res[-1]['pre_d_bottom_mm']:+7.1f}        {res[-1]['pre_normal_err']:5.1f}  landed {landed[m].tolist()} target {int(target[m][0])}", flush=True)
    good = [x for x in res if x['real'] == 'success']
    gate = sum(x['sim_rate'] >= 0.8 for x in good)
    print(f'[replay] P1 gate: {gate}/{len(good)} real successes reproduced with sim rate >= 80 % (need >= 9)', flush=True)
    if args.cl_policy_out:                                     # (10/15 insert RL) per env transition for train.py insert_rl
        c = _CL_POL['cache']
        ks = sorted(c)
        np.savez_compressed(args.cl_policy_out, env=np.array(ks), obs=np.stack([c[k]['obs'] for k in ks]),
                            act=np.stack([c[k]['act'] for k in ks]), logp=np.array([c[k]['logp'] for k in ks]),
                            bias=np.stack([c[k]['bias'] for k in ks]), seated=seated[ks], exact=(face_ok & seated)[ks],
                            in_quadrant=in_quadrant[ks], landed=landed[ks], target=target[ks],
                            low=st['low'].cpu().numpy()[ks], quadrant=np.array([rows[who[k]]['model_qd'] for k in ks]))
    out = Path(args.out or RL / 'logs/basket/replay_20261012.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(results=res, gate=gate, need=9, names=[rows[i]['name'] for i in who], in_hand=in_hand, pre=pre,
                                   final_pose=dict(plate_pos=pc.tolist(), plate_quat=pq.tolist()), trace=trace,
                                   final={k: st[k].cpu().numpy().tolist() for k in ('d_bottom', 'low', 'normal_err', 'd_u')},
                                   gap_err=gap_err.cpu().numpy().tolist(), held=held.tolist(), yaw_err=yaw_err.cpu().numpy().tolist(),
                                   lean=lean.cpu().numpy().tolist(), quadrant_ok=face_ok.tolist(), in_quadrant=in_quadrant.tolist(), seated=seated.tolist(), landed_gap=landed.tolist(), target_gap=target.tolist()), indent=1))
    if args.hold_view:                                         # (10/14 보스) look at the result in the GUI
        print('[replay] done - close the window to quit', flush=True)
        import omni.kit.app
        while omni.kit.app.get_app().is_running():
            env.sim.render()
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
    env._basket_tracking_mode = True  # teacher and student actions are corrections from the live TCP state
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
            # Recomputed from this frame's observation: clean BC label is a tracking correction,
            # while DART perturbs only the executed copy below.
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
    env._basket_tracking_mode = True
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
