"""Goal 2 basket insertion runs (Isaac Lab python):  $ISAAC -p rl/basket_tasks.py <mode> --headless ...
  check   : build the basket scene, reset, print slot poses and that nothing collides at rest
  grasp   : S2 start state: plates at random (radius, azimuth) away from the basket, Codex library path replayed as
            joint targets (home -> pre -> grasp, close at the library close step -> lift); prints grasp success
  expert  : S3 grasp + scripted insertion: measure the plate in the hand, plate target = slot plane (grasp point on
            top), TCP target = plate target . in-hand^-1, joint moves lift end -> high above -> above -> straight
            down (dense IK) -> open -> straight up; prints insertion success (rl/basket.yaml success)
More modes (expert / collect / gate / export) are added step by step (collab/WEEKEND_CLAUDE_2026-10-09.md)."""
import argparse
import sys


def parse_args():
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("check", "grasp", "expert", "bc_train", "bc_eval"))
    p.add_argument("--data", default="/home/pan/pan/6D_pose/IL/data_basket_expert_20261008")
    p.add_argument("--bc_out", default="/home/pan/IsaacLab/logs/rsl_rl/basket_bc/model.pt")
    p.add_argument("--bc_ckpt", default="/home/pan/IsaacLab/logs/rsl_rl/basket_bc/model.pt")
    p.add_argument("--bc_steps", type=int, default=20000)
    p.add_argument("--bc_eval_s", type=float, default=120.0)
    p.add_argument("--bc_min_stage", type=int, default=1, help="bc_train: keep steps from this stage (0 = transport too)")
    p.add_argument("--bc_takeover", default=None, help="expert: after the planned transport, this insertion BC drives")
    p.add_argument("--z_safe", type=float, default=0.40, help="expert: transport TCP height (turned plate clears the rails)")
    p.add_argument("--snapshot", default=None, help="expert: save poses at the open to this .npz (offline section plots)")
    p.add_argument("--debug_cl", action="store_true")
    p.add_argument("--record", default=None, help="expert: save the demo (states + joint commands) to this .npz")
    p.add_argument("--retries", type=int, default=3, help="open-loop expert: back-up-and-shift re-insert attempts")
    p.add_argument("--align_height", type=float, default=0.075, help="closed loop: fine alignment below this lowest-point height")
    p.add_argument("--land_height", type=float, default=0.030, help="closed loop: open the gripper once the lowest point is below")
    p.add_argument("--closed_loop", action="store_true", help="expert: feedback descent / search / release (env.step)")
    p.add_argument("--remeasure", action="store_true", help="expert: re-measure the in-hand plate at `above`")
    p.add_argument("--sim_joint_speed", type=float, default=20.0, help="deg/s for the scripted joint moves (sim only)")
    p.add_argument("--library", default="reports/20261008_goal1/variant_library_combined/library.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_envs", type=int, default=4)
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


def run_check(args):
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    import basket_env as be
    env = ManagerBasedRLEnv(be.make_basket_cfg(args.num_envs, viewer=not args.headless))
    env.reset()
    zero = torch.zeros(env.num_envs, 7, device=env.device)
    zero[:, 6] = 1.0
    for _ in range(30):
        env.step(zero)
    b = env.scene["basket"].data.root_pos_w - env.scene.env_origins
    print("[check] slot (env frame):", env._slot.cpu().numpy().round(4).tolist(), flush=True)
    print("[check] basket root:", b.cpu().numpy().round(4).tolist(), flush=True)
    st = be.basket_state(env)
    print("[check] plate d_plane", st["d_plane"].cpu().numpy().round(3).tolist(), "low", st["low"].cpu().numpy().round(3).tolist())
    contact = env.scene["arm_contacts"].data.net_forces_w.norm(dim=-1).amax(dim=1)
    print("[check] arm contact N", contact.cpu().numpy().round(2).tolist(), flush=True)
    env.close()


LIB_CACHE = {}
APP = None


def sim_steps(env, k=None):
    for _ in range(k or env.cfg.decimation):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
    env.sim.render()


def grasped_start(env, args, rng):
    """S2: every env picks a successful library row (radius / phi / pitch) and an azimuth that keeps the plate away
    from the basket; the plate is placed there, the row's joint path (joint 1 + azimuth) is replayed as position
    targets with the gripper closing at the row's close step.  Returns dict(ok, row, azimuth, q_lift_end_deg)."""
    import math
    import numpy as np
    import torch
    import plate_env as pe
    lib = LIB_CACHE.setdefault(args.library, np.load(args.library))
    robot, plate, dev, n = env.scene["robot"], env.scene["plate"], env.device, env.num_envs
    arm = env.action_manager.get_term("arm_action")
    aj = arm._joint_ids
    fj = [robot.joint_names.index(j) for j in ("left_finger_joint", "right_finger_joint")]
    good = np.flatnonzero(lib["ok"] & (lib["radius_m"] >= 1.00) & (lib["radius_m"] <= 1.25))
    rows = rng.choice(good, n)
    az = np.radians(rng.uniform(10.0, 25.0, n))               # basket sits at about -7 deg: plates on the +y side
    r = lib["radius_m"][rows]
    env.reset()
    pose = torch.zeros(n, 7, device=dev)
    pose[:, 0] = torch.tensor(r * np.cos(az), dtype=torch.float32, device=dev)
    pose[:, 1] = torch.tensor(r * np.sin(az), dtype=torch.float32, device=dev)
    pose[:, 2], pose[:, 3] = pe.REST_Z, 1.0
    pose[:, :3] += env.scene.env_origins
    ids = torch.arange(n, device=dev)
    plate.write_root_pose_to_sim(pose)
    plate.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
    pe.box_under_plate(env, ids)
    Q = lib["q_deg"][:, rows].astype(np.float32).copy()       # (T, n, 6)
    Q[:, :, 0] += np.degrees(az)[None, :].astype(np.float32)
    close, end = lib["close_step"][rows], lib["success_step"][rows]
    home = robot.data.default_joint_pos.clone()
    q = home.clone()
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    q0 = torch.tensor(np.radians(Q[0]), device=dev)
    steps = 90                                                 # home -> pre: 3 s smoothstep (sim only)
    for k in range(steps + 1):
        s_ = k / steps
        q[:, aj] = home[:, aj] + (q0 - home[:, aj]) * (3 * s_ * s_ - 2 * s_ ** 3)
        robot.set_joint_position_target(q)
        sim_steps(env)
    T = int(end.max()) + 1
    for t in range(T):
        tt = np.minimum(t, end)
        q[:, aj] = torch.tensor(np.radians(Q[tt, np.arange(n)]), device=dev)
        shut = torch.tensor(t >= close, device=dev)
        q[:, fj] = torch.where(shut[:, None], torch.full_like(q[:, fj], pe.FINGER_CLOSED), torch.zeros_like(q[:, fj]))
        robot.set_joint_position_target(q)
        sim_steps(env)
    for _ in range(30):                                        # 1 s hold
        sim_steps(env)
    lift = plate.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - pe.REST_Z
    fingers = pe.finger_closure(env)
    ok = (lift > 0.05) & (fingers < 0.98)
    return dict(ok=ok, rows=rows, azimuth=az, q_lift_end=q[:, aj].clone(), lift=lift)


def run_grasp(args):
    import numpy as np
    from isaaclab.envs import ManagerBasedRLEnv
    import basket_env as be
    env = ManagerBasedRLEnv(be.make_basket_cfg(args.num_envs, viewer=not args.headless))
    s = grasped_start(env, args, np.random.default_rng(args.seed))
    ok = s["ok"].cpu().numpy()
    print(f"[grasp] {ok.sum()}/{len(ok)} lifted; lift mm median {float(s['lift'].median()) * 1000:.0f}", flush=True)
    contact = env.scene["arm_contacts"].data.net_forces_w.norm(dim=-1).amax(dim=1)
    print("[grasp] arm contact > 1 N:", int((contact > 1).sum()), flush=True)
    env.close()


def _quat_to_R(q):
    import numpy as np
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _T(R, p):
    import numpy as np
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def _plate_verts():
    import numpy as np
    if "verts" not in LIB_CACHE:
        p = __import__("pathlib").Path(__file__).resolve().parent / "assets" / "plate029_simplified.obj"
        LIB_CACHE["verts"] = np.array([list(map(float, l.split()[1:4])) for l in p.open() if l.startswith("v ")])
    return LIB_CACHE["verts"]


def insertion_targets(env, k, tcp_T, plate_T, flange_T):
    """Per env k (numpy, env frame): TCP poses for insert and above so the PLATE lands in the slot plane with the
    grasp point on top.  Returns (T_tcp_insert, T_tcp_above, up vector)."""
    import math
    import numpy as np
    import basket_env as be
    o = env._slot[k, :3].cpu().numpy().astype(float)
    yaw = float(env._slot[k, 3])
    tilt = math.radians(be.B["slot"].get("target_tilt_deg", math.degrees(be.slot_frame()[2])))
    n = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    Np = n * math.cos(tilt) + np.array([0, 0, 1.0]) * math.sin(tilt)
    v = -n * math.sin(tilt) + np.array([0, 0, 1.0]) * math.cos(tilt)
    h_c = be.B["slot"]["plate_centre"][2] - be.B["slot"]["floor_z"]
    c_t = o + v * (h_c / math.cos(tilt))
    Rp, pp = plate_T[:3, :3], plate_T[:3, 3]
    z_cur = Rp[:, 2]
    c_cur = pp + z_cur * 0.01
    a = tcp_T[:3, 3] - c_cur
    a -= z_cur * (a @ z_cur)
    a_loc = Rp.T @ (a / np.linalg.norm(a))
    z_t = Np if z_cur @ Np >= 0 else -Np
    zl = np.array([0, 0, 1.0])
    A = np.stack([a_loc, np.cross(zl, a_loc), zl], 1)
    Bm = np.stack([v, np.cross(z_t, v), z_t], 1)
    R_t0 = Bm @ A.T
    H = np.linalg.inv(tcp_T) @ plate_T                     # plate in the hand
    Hr_inv = np.linalg.inv(H)[:3, :3]
    # the plate is round: any turn about its normal fits the slot -> pick the one that points the tool most nearly
    # straight down (the taught insert; a sagged in-hand plate otherwise asks for a ~45 deg tilted, unreachable wrist)
    best = None
    for ang in np.radians(np.arange(0, 360, 2)):
        c_, s_ = math.cos(ang), math.sin(ang)
        K = np.array([[0, -z_t[2], z_t[1]], [z_t[2], 0, -z_t[0]], [-z_t[1], z_t[0], 0]])
        Rz = np.eye(3) + s_ * K + (1 - c_) * K @ K
        R = Rz @ R_t0
        tool_z = (R @ Hr_inv)[:, 2]
        if best is None or tool_z[2] < best[0]:
            best = (tool_z[2], R)
    R_t = best[1]
    plate_t = _T(R_t, c_t - z_t * 0.01)
    # place by the real mesh: its lowest point on the slot line, 10 mm above the floor (taught insert ~1 cm above)
    V = _plate_verts()
    W = V @ plate_t[:3, :3].T + plate_t[:3, 3]
    b = W[np.argmin(W[:, 2])]
    plate_t[:3, 3] += n * (be.B["slot"].get("target_bottom_n_m", 0.0) - (b - o) @ n) + np.array([0, 0, 1.0]) * (0.010 - (b[2] - o[2]))
    tcp_ins = plate_t @ np.linalg.inv(H)
    tcp_above = tcp_ins.copy()
    tcp_above[:3, 3] += v * 0.11
    return tcp_ins, tcp_above, v


def ik_robust(kin, target, seed, priors, tag, tries=12, rng=None):
    """IK from the chain seed; on failure from the golden-teaching priors and jittered seeds, keeping the solution
    closest to the chain seed (no wrist flips between neighbouring waypoints).  Raises if nothing converges."""
    import numpy as np
    rng = rng or np.random.default_rng(0)
    cands = []
    for s0 in [seed] + list(priors) + [seed + rng.normal(0, 0.35, 6) for _ in range(tries)]:
        try:
            q = kin.ik(target, s0)
        except RuntimeError:
            continue
        cands.append(q)
        if s0 is seed:
            break
    if not cands:
        raise RuntimeError(f"waypoint {tag} at {np.round(target[:3, 3], 3).tolist()} (flange)")
    return min(cands, key=lambda q: np.abs(q - seed).max())


N_TRANSPORT = 12


def transport_poses(start, above, z_safe):
    """12 TCP poses (the taught order): straight up to z_safe keeping the grasp orientation (3), across to above the
    slot at z_safe still flat (5), then down to `above` while turning to the insert orientation (4, slerp).  A flat
    plate hangs only ~5 cm under the TCP, so it clears the rails; turning happens inside the basket where the
    vertical plate (lowest point ~TCP - 0.24 m) stays above the pins; a vertical tool above ~0.38 m is out of reach."""
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp
    z = max(z_safe, start[2, 3])
    up = start.copy()
    up[2, 3] = z
    over = start.copy()
    over[:3, 3] = [above[0, 3], above[1, 3], z]
    out = []
    for t in (1 / 3, 2 / 3, 1.0):
        T = start.copy()
        T[:3, 3] = start[:3, 3] + (up[:3, 3] - start[:3, 3]) * t
        out.append(T)
    for t in np.linspace(0.2, 1.0, 5):
        T = up.copy()
        T[:3, 3] = up[:3, 3] + (over[:3, 3] - up[:3, 3]) * t
        out.append(T)
    sl = Slerp([0, 1], Rotation.from_matrix([over[:3, :3], above[:3, :3]]))
    for t in np.linspace(0.25, 1.0, 4):
        T = np.eye(4)
        T[:3, :3] = sl([t]).as_matrix()[0]
        T[:3, 3] = over[:3, 3] + (above[:3, 3] - over[:3, 3]) * t
        out.append(T)
    return out

def slot_obs(env, st, t_desc, low_hist):
    """Insertion observation (slot frame, translation/rotation of the basket factored out): plate bottom from the slot
    line (n, u), lowest point height, plate normal, TCP position and tool z / y axes in the slot frame, the lowest
    point's change over the last 0.5 s (stuck cue) and the time since the descent started."""
    import torch
    import plate_env as pe
    s = env._slot
    n, u = st["n"], st["u"]
    z = torch.zeros_like(n)
    z[:, 2] = 1.0
    R = torch.stack([n, u, z], 2)                                          # columns: slot axes in env frame
    tp, tq = pe.tcp_pose(env)
    tcp = tp - env.scene.env_origins - s[:, :3]
    up = pe.body_axis(env.scene["plate"].data.root_quat_w, 2)
    loc = lambda v: torch.einsum("nij,ni->nj", R, v)                       # noqa: E731
    return torch.cat([torch.stack([st["d_bottom"], st["d_u"], st["low"]], 1) * 20.0, loc(up), loc(tcp) * 5.0,
                      loc(pe.body_axis(tq, 2)), loc(pe.body_axis(tq, 1)),
                      ((st["low"] - low_hist) * 100.0)[:, None], t_desc[:, None] / 10.0], 1)


def closed_loop_insert(env, live, args, record=None):
    """Feedback expert (env.step, SafeDiffIK, slowest caps): keep the plate's lowest point on the slot line (n, u) while
    lowering it at 1.5 cm/s; if it stops sinking above the floor for 0.5 s it backs up 8 mm and retries with the
    target shifted +-2, +-4 mm along n; at <= 12 mm above the floor it opens, waits 1.5 s, and retreats 10 cm up.
    record: list -> (obs, action, phase) per step for BC.  Returns dict(before_open=basket_state at the open)."""
    import math
    import torch
    import basket_env as be
    import plate_env as pe
    from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul
    n_env, dev = env.num_envs, env.device
    arm = env.action_manager.get_term("arm_action")
    ids = torch.arange(n_env, device=dev)
    arm.reset(ids)
    scale, dt = pe.A["scale"], env.step_dt
    _, q0 = pe.tcp_pose(env)
    q_hold = q0.clone()                                                    # keep the turned orientation
    phase = torch.zeros(n_env, dtype=torch.long, device=dev)               # 0 descend, 1 release, 2 retreat, 3 done
    phase[~live] = 3
    t_desc = torch.zeros(n_env, device=dev)
    t_phase = torch.zeros(n_env, device=dev)
    low_hist = torch.zeros(n_env, device=dev)
    hist = []
    stuck = torch.zeros(n_env, device=dev)
    backoff = torch.zeros(n_env, device=dev)
    tries = torch.zeros(n_env, dtype=torch.long, device=dev)
    offsets = torch.tensor([0.0, 0.002, -0.002, 0.004, -0.004, 0.0], device=dev)
    before = None
    for step in range(int(25 / dt)):
        st = be.basket_state(env)
        hist.append(st["low"].clone())
        low_hist = hist[-16] if len(hist) >= 16 else hist[0]
        obs = slot_obs(env, st, t_desc, low_hist)
        tp, tq = pe.tcp_pose(env)
        v = torch.zeros(n_env, 3, device=dev)
        off = offsets[tries.clamp(max=len(offsets) - 1)]
        e_n = -(st["d_bottom"] - off)
        e_u = -st["d_u"]
        lateral = st["n"] * e_n[:, None] * 2.0 + st["u"] * e_u[:, None] * 1.0
        sink = (st["low"] - low_hist) < -0.001
        stuck = torch.where((phase == 0) & ~sink & (st["low"] > args.land_height) & (backoff <= 0), stuck + dt, torch.zeros_like(stuck))
        start_back = (phase == 0) & (stuck > 1.0)
        backoff = torch.where(start_back, torch.full_like(backoff, 0.5), backoff - dt)
        tries = tries + start_back.long()
        stuck[start_back] = 0
        # above the pin tops: sink freely; near them: sink only while the lowest point is within 2 mm of the gap centre
        near = st["low"] < args.align_height
        aligned = st["d_bottom"].sub(off).abs() < torch.where(near, torch.full_like(t_desc, 0.002), torch.full_like(t_desc, 0.008))
        vz = torch.where(backoff > 0, torch.full_like(t_desc, 0.016), torch.where(aligned, torch.full_like(t_desc, -0.012), torch.zeros_like(t_desc)))
        d = phase == 0
        v[d] = lateral[d] + torch.tensor([0, 0, 1.0], device=dev) * vz[d, None]
        r = phase == 2
        v[r, 2] = 0.03
        grip = torch.where(phase >= 1, torch.ones(n_env, device=dev), -torch.ones(n_env, device=dev))
        act = torch.zeros(n_env, 7, device=dev)
        act[:, :3] = v * dt / scale
        rot = axis_angle_from_quat(quat_mul(q_hold, quat_conjugate(tq)))
        act[:, 3:6] = torch.where((phase <= 1)[:, None], rot / scale, torch.zeros_like(rot))
        act[:, 6] = grip
        act[phase == 3] = 0.0
        act[phase == 3, 6] = 1.0
        if record is not None:
            record.append((obs.cpu(), act.cpu(), phase.cpu().clone(), live.cpu().clone()))
        env.step(act)
        if getattr(args, "debug_cl", False) and step % 15 == 0:
            tp2, _ = pe.tcp_pose(env)
            print(f"[cl] t {step * dt:5.1f} ph {phase[:4].tolist()} d_bot {(st['d_bottom'][:4] * 1000).round().tolist()} "
                  f"low {(st['low'][:4] * 1000).round().tolist()} vz {vz[:4].tolist()} tcpz {((tp2 - env.scene.env_origins)[:4, 2] * 1000).round().tolist()} "
                  f"tries {tries[:4].tolist()} act {act[0, :3].round(decimals=2).tolist()}", flush=True)
        t_desc += dt
        t_phase += dt
        land = (phase == 0) & (st["low"] < args.land_height)
        give_up = (phase == 0) & ((tries >= len(offsets)) | (t_desc > 15.0))
        if bool(land.any()) or bool(give_up.any()):
            if before is None:
                before = {k: v_.clone() for k, v_ in st.items()}
            else:
                for k_ in before:
                    before[k_][land | give_up] = st[k_][land | give_up]
        phase[land | give_up] = 1
        t_phase[land | give_up] = 0
        nxt = (phase == 1) & (t_phase > 1.5)
        phase[nxt] = 2
        t_phase[nxt] = 0
        done = (phase == 2) & (t_phase > 3.5)
        phase[done] = 3
        if bool((phase == 3).all()):
            break
    return dict(before_open=before if before is not None else be.basket_state(env))


def run_expert(args):
    import math
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "calibration"))
    from robot_model import Kinematics
    import basket_env as be
    import plate_env as pe
    env = ManagerBasedRLEnv(be.make_basket_cfg(args.num_envs, viewer=not args.headless))
    s = grasped_start(env, args, np.random.default_rng(args.seed))
    robot, plate, dev, n = env.scene["robot"], env.scene["plate"], env.device, env.num_envs
    arm = env.action_manager.get_term("arm_action")
    aj = arm._joint_ids
    fj = [robot.joint_names.index(j) for j in ("left_finger_joint", "right_finger_joint")]
    l6 = robot.body_names.index("link_6")
    org = env.scene.env_origins
    kin = Kinematics()
    golden = __import__("json").loads((be.RL_DIR / be.B["golden_teach"]).read_text())["poses"]
    seed_ins = np.radians(golden["insert"]["posj"])
    seed_above = np.radians(golden["above"]["posj"])

    def world_T(pos, quat, k):
        return _T(_quat_to_R(quat[k].cpu().numpy()), (pos[k] - org[k]).cpu().numpy())
    tp, tq = pe.tcp_pose(env)
    plans, fail = [], np.zeros(n, bool)
    for k in range(n):
        tcp_T = world_T(tp, tq, k)
        fl_T = world_T(robot.data.body_pos_w[:, l6], robot.data.body_quat_w[:, l6], k)
        flange_from_tcp = np.linalg.inv(tcp_T) @ fl_T           # constant mount transform, read from the sim
        plate_T = world_T(plate.data.root_pos_w, plate.data.root_quat_w, k)
        ins, above, v = insertion_targets(env, k, tcp_T, plate_T, flange_from_tcp)
        if k < 3:
            H = np.linalg.inv(tcp_T) @ plate_T
            print(f"[dbg] env {k} slot {env._slot[k].cpu().numpy().round(3).tolist()} ins tcp {ins[:3, 3].round(3).tolist()} "
                  f"tool z {ins[:3, 2].round(2).tolist()} | in-hand plate root in tcp {H[:3, 3].round(3).tolist()} "
                  f"plate z in tcp {H[:3, 2].round(2).tolist()}", flush=True)
        try:
            seed = np.radians(torch.rad2deg(robot.data.joint_pos[k, aj]).cpu().numpy())
            transport = transport_poses(tcp_T, above, args.z_safe)
            line = []
            for t in np.linspace(1.0, 0.0, 8):
                T_ = ins.copy()
                T_[:3, 3] = ins[:3, 3] + v * 0.11 * t
                line.append(T_)
            qs = []
            for wi, T_ in enumerate(transport + line[1:]):
                seed = ik_robust(kin, T_ @ flange_from_tcp, seed, [seed_above, seed_ins], wi)
                qs.append(seed.copy())
            plans.append(np.degrees(np.array(qs)))
        except RuntimeError as e:
            print(f"[expert] env {k} IK fail: {e}", flush=True)
            fail[k] = True
            plans.append(None)
    q = robot.data.joint_pos.clone()

    def drive(targets_deg, shut, speed=args.sim_joint_speed):
        """Joint-space straight moves to the per-env targets (None = stay) at <= speed deg/s, then settle 0.5 s."""
        start = torch.rad2deg(q[:, aj]).cpu().numpy()
        goal = np.array([t if t is not None else start[k] for k, t in enumerate(targets_deg)])
        steps = max(1, int(np.abs(goal - start).max() / speed * 30))
        for i in range(1, steps + 1):
            q[:, aj] = torch.tensor(np.radians(start + (goal - start) * i / steps), dtype=torch.float32, device=dev)
            q[:, fj] = pe.FINGER_CLOSED if shut else 0.0
            record_step(q, shut)
            robot.set_joint_position_target(q)
            sim_steps(env)
        for _ in range(15):
            record_step(q, shut)
            sim_steps(env)

    REC = []
    STAGE = [0]                                                 # 0 transport (planner), 1 insert + retries, 2 release, 3 retreat

    def record_step(q_target, shut):
        """Demo recording (--record): state before the step + the commanded joint target / gripper of this step."""
        if not args.record:
            return
        tp_, tq_ = pe.tcp_pose(env)
        pl = env.scene["plate"]
        REC.append(dict(q=torch.rad2deg(robot.data.joint_pos[:, aj]).cpu().numpy().astype(np.float32),
                        q_cmd=torch.rad2deg(q_target[:, aj]).cpu().numpy().astype(np.float32),
                        grip_close=np.full(n, bool(shut)),
                        tcp=torch.cat([tp_ - org, tq_], 1).cpu().numpy().astype(np.float32),
                        plate=torch.cat([pl.data.root_pos_w - org, pl.data.root_quat_w], 1).cpu().numpy().astype(np.float32),
                        slot=env._slot.cpu().numpy().astype(np.float32), stage=np.full(n, STAGE[0], np.int8)))
    live = s["ok"].cpu().numpy() & ~fail
    NT = N_TRANSPORT
    for i in range(NT):                                         # up to z_safe, over to the slot while turning, down to above
        drive([p[i] if (p is not None and live[k]) else None for k, p in enumerate(plans)], shut=True)
    if args.remeasure:                                          # re-measure the plate in the hand at `above` and re-plan
        tp, tq = pe.tcp_pose(env)
        for k in range(n):
            if not live[k]:
                continue
            tcp_T = world_T(tp, tq, k)
            fl_T = world_T(robot.data.body_pos_w[:, l6], robot.data.body_quat_w[:, l6], k)
            f_t = np.linalg.inv(tcp_T) @ fl_T
            ins, _, v = insertion_targets(env, k, tcp_T, world_T(plate.data.root_pos_w, plate.data.root_quat_w, k), f_t)
            seed = np.radians(plans[k][NT - 1])
            for j, t in enumerate(np.linspace(1.0, 0.0, 8)[1:]):
                T_ = ins.copy()
                T_[:3, 3] = ins[:3, 3] + v * 0.11 * t
                try:
                    seed = kin.ik(T_ @ f_t, seed)
                except RuntimeError:
                    live[k] = False
                    break
                plans[k][NT + j] = np.degrees(seed)
    st_above = be.basket_state(env)
    STAGE[0] = 1
    if args.bc_takeover:                                        # planner did the transport; the insertion BC takes over here
        still_closed = bc_control(env, args, args.bc_takeover, torch.tensor(live, device=dev), args.bc_eval_s)
        live = live & ~still_closed.cpu().numpy()                  # success also needs the BC to have let go
        st_before = be.basket_state(env)
    elif args.closed_loop:                                        # feedback descent / release / retreat via env.step
        rec = closed_loop_insert(env, torch.tensor(live, device=dev), args, record=getattr(args, "_record", None))
        st_before = rec["before_open"]
    else:
        for i in range(NT, NT + 7):                             # 7 line points down to insert
            drive([p[i] if (p is not None and live[k]) else None for k, p in enumerate(plans)], shut=True)
        for attempt in range(args.retries):                     # missed the gap? back up 4 cm, shift by the miss, re-insert
            st_r = be.basket_state(env)
            miss = live & ((st_r["d_bottom"].abs() > 0.005) | (st_r["low"] > be.B["success"]["max_height_m"])).cpu().numpy()
            if not miss.any():
                break
            tp_r, tq_r = pe.tcp_pose(env)
            q_now = torch.rad2deg(robot.data.joint_pos[:, aj]).cpu().numpy()
            retry = [None] * n
            for k in np.flatnonzero(miss):
                tcp_T = world_T(tp_r, tq_r, k)
                fl_T = world_T(robot.data.body_pos_w[:, l6], robot.data.body_quat_w[:, l6], k)
                f_t = np.linalg.inv(tcp_T) @ fl_T
                shift = -st_r["n"][k].cpu().numpy() * float(st_r["d_bottom"][k])
                seed, pts = np.radians(q_now[k]), []
                for dz, sh in ((0.04, 0.0), (0.04, 1.0), (0.0, 1.0), (-0.02, 1.0)):
                    T_ = tcp_T.copy()
                    T_[:3, 3] += np.array([0, 0, dz]) + shift * sh
                    try:
                        seed = ik_robust(kin, T_ @ f_t, seed, [], f"retry{attempt}")
                    except RuntimeError:
                        pts = []
                        break
                    pts.append(np.degrees(seed))
                retry[k] = pts or None
            for i in range(4):
                drive([r[i] if r is not None else None for r in retry], shut=True)
        st_before = be.basket_state(env)
        if args.snapshot:                                       # plate / basket / slot poses at the open, for offline plots
            np.savez(args.snapshot, plate_pos=(plate.data.root_pos_w - org).cpu().numpy(),
                     plate_quat=plate.data.root_quat_w.cpu().numpy(),
                     basket_pos=(env.scene["basket"].data.root_pos_w - org).cpu().numpy(),
                     basket_quat=env.scene["basket"].data.root_quat_w.cpu().numpy(), slot=env._slot.cpu().numpy(),
                     live=live, low=st_before["low"].cpu().numpy(), d_bottom=st_before["d_bottom"].cpu().numpy())
        q[:, fj] = 0.0                                          # open
        STAGE[0] = 2
        robot.set_joint_position_target(q)
        for _ in range(45):
            record_step(q, False)
            sim_steps(env)
        STAGE[0] = 3
        for i in list(range(NT + 5, NT - 2, -1)):               # straight back up the line to above
            drive([p[i] if (p is not None and live[k]) else None for k, p in enumerate(plans)], shut=False)
    p0 = plate.data.root_pos_w.clone()
    for _ in range(30):
        sim_steps(env)
    still = (plate.data.root_pos_w - p0).norm(dim=1) < 0.01
    st = be.basket_state(env)
    sc = be.B["success"]
    inside = ((st["d_bottom"].abs() < sc["bottom_tol_m"]) & (st["d_u"].abs() < sc["centre_tol_m"])
              & (st["low"] < sc["max_height_m"]) & (st["normal_err"] < sc["max_normal_err_deg"]))
    succ = inside & still & torch.tensor(live, device=dev)
    contact = env.scene["arm_contacts"].data.net_forces_w.norm(dim=-1).amax(dim=1) > 1.0
    print(f"[expert] grasped {int(s['ok'].sum())}/{n}, IK fail {int(fail.sum())}, inserted {int(succ.sum())}/{int(live.sum())}", flush=True)
    np.set_printoptions(precision=3, suppress=True)
    print("[expert] d_plane mm", (st["d_plane"] * 1000).cpu().numpy()[:16], flush=True)
    print("[expert] d_bottom mm", (st["d_bottom"] * 1000).cpu().numpy()[:16], " before-open", (st_before["d_bottom"] * 1000).cpu().numpy()[:16], flush=True)
    print("[expert] d_u mm", (st["d_u"] * 1000).cpu().numpy()[:16], flush=True)
    print("[expert] low mm", (st["low"] * 1000).cpu().numpy()[:16], flush=True)
    print("[expert] before-open d_plane mm", (st_before["d_plane"] * 1000).cpu().numpy()[:16], flush=True)
    print("[expert] before-open low mm", (st_before["low"] * 1000).cpu().numpy()[:16], " normal err", st_before["normal_err"].cpu().numpy()[:16], flush=True)
    print("[expert] above d_bottom mm", (st_above["d_bottom"] * 1000).cpu().numpy()[:16], flush=True)
    f = live & ~succ.cpu().numpy()
    print(f"[expert] failures {int(f.sum())}: before-open |d_bottom|>8.5 {int(((st_before['d_bottom'].abs() > 0.0085).cpu().numpy() & f).sum())}, "
          f"after normal>20 {int(((st['normal_err'] > 20).cpu().numpy() & f).sum())}, |d_u|>30 {int(((st['d_u'].abs() > 0.03).cpu().numpy() & f).sum())}, "
          f"not still {int(((~still).cpu().numpy() & f).sum())}", flush=True)
    print("[expert] normal err deg", st["normal_err"].cpu().numpy()[:16], " arm contact", int(contact.sum()), flush=True)
    if args.record:
        out = __import__("pathlib").Path(args.record)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **{k: np.stack([r[k] for r in REC]) for k in REC[0]}, success=succ.cpu().numpy(),
                            live=live, seed=args.seed, retries=args.retries)
        print(f"[expert] recorded {len(REC)} steps x {n} envs -> {out}  ({int(succ.sum())} successful demos)", flush=True)
    if not args.headless:                                       # GUI: keep the final scene open until the window closes
        print("[expert] GUI: close the window to finish", flush=True)
        while APP is not None and APP.is_running():
            env.sim.render()
    env.close()


# ---------------------------------------------------------------- BC on the insertion demos (S5)

def _quat_rot(qwxyz):
    """(..., 4) wxyz -> (..., 3, 3), numpy."""
    import numpy as np
    w, x, y, z = (qwxyz[..., i] for i in range(4))
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def basket_features(q, tcp, plate, slot, grip_close):
    """BC input (numpy, any leading shape), everything in the slot frame so the basket pose drops out:
    plate root (x5), plate normal, plate lowest mesh point (x10), TCP (x5), tool z / y axes, arm joints (deg/100),
    gripper closed flag.  25 dims."""
    import numpy as np
    V = _plate_verts()[::6]
    c, s_ = np.cos(slot[..., 3]), np.sin(slot[..., 3])
    Rs = np.stack([np.stack([c, -s_, np.zeros_like(c)], -1), np.stack([s_, c, np.zeros_like(c)], -1),
                   np.stack([np.zeros_like(c), np.zeros_like(c), np.ones_like(c)], -1)], -2)       # slot -> env
    loc = lambda v: np.einsum("...ji,...j->...i", Rs, v)                                           # noqa: E731
    Rp = _quat_rot(plate[..., 3:7])
    W = np.einsum("...ij,vj->...vi", Rp, V) + plate[..., None, :3]
    low = np.take_along_axis(W, W[..., 2:3].argmin(-2)[..., None], -2)[..., 0, :]
    Rt = _quat_rot(tcp[..., 3:7])
    return np.concatenate([loc(plate[..., :3] - slot[..., :3]) * 5, loc(Rp[..., :, 2]), loc(low - slot[..., :3]) * 10,
                           loc(tcp[..., :3] - slot[..., :3]) * 5, loc(Rt[..., :, 2]), loc(Rt[..., :, 1]), q / 100.0,
                           grip_close[..., None].astype(np.float32)], -1).astype(np.float32)


ACT_DEG = 2.0                                                    # action = (joint command - joint) / ACT_DEG, clipped to +-1


def run_bc_train(args):
    """Fit an MLP (obs 25 -> 6 joint deltas + gripper) on the successful demos of every batch in --data."""
    import glob
    import numpy as np
    import torch
    from torch import nn
    X, Y, G = [], [], []
    files = sorted(glob.glob(f"{args.data}/batch_*.npz"))
    for f in files:
        z = np.load(f)
        ok = np.flatnonzero(z["success"])
        if not len(ok):
            continue
        feats = basket_features(z["q"][:, ok], z["tcp"][:, ok], z["plate"][:, ok], z["slot"][:, ok], z["grip_close"][:, ok])
        dq = np.clip((z["q_cmd"][:, ok] - z["q"][:, ok]) / ACT_DEG, -1, 1)
        grip = np.where(z["grip_close"][:, ok], -1.0, 1.0)[..., None]
        act = np.concatenate([dq, grip], -1).astype(np.float32)
        stage = z["stage"][:, ok] if "stage" in z.files else np.ones(dq.shape[:2], np.int8)
        for j in range(len(ok)):                                     # one entry per episode (batches differ in length)
            keep = stage[:, j] >= args.bc_min_stage                  # insertion BC: from `above` on (planner transports)
            X.append(feats[keep, j])
            Y.append(act[keep, j])
    n_ep = len(X)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_ep)
    n_val = max(1, n_ep // 10)
    tr, va = perm[n_val:], perm[:n_val]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cat = lambda idx, A: torch.tensor(np.concatenate([A[i] for i in idx]), device=dev)  # noqa: E731
    xtr, ytr, xva, yva = cat(tr, X), cat(tr, Y), cat(va, X), cat(va, Y)
    torch.manual_seed(args.seed)
    net = nn.Sequential(nn.Linear(xtr.shape[-1], 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(),
                        nn.Linear(256, ytr.shape[-1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.bc_steps)
    for step in range(1, args.bc_steps + 1):
        idx = torch.randint(len(xtr), (4096,), device=dev)
        loss = nn.functional.mse_loss(net(xtr[idx]), ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
        if step % 2000 == 0:
            with torch.no_grad():
                v = nn.functional.mse_loss(net(xva), yva).item()
                gacc = ((net(xva)[:, 6] < 0) == (yva[:, 6] < 0)).float().mean().item()
            print(f"[bc] step {step} train {loss.item():.5f} val {v:.5f} grip acc {gacc:.4f}", flush=True)
    out = __import__("pathlib").Path(args.bc_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=net.state_dict(), in_dim=xtr.shape[-1], out_dim=ytr.shape[-1], act_deg=ACT_DEG, data=files,
                    episodes=int(n_ep), seed=args.seed), out)
    print(f"[bc] {n_ep} demo episodes from {len(files)} batches -> {out}", flush=True)


def bc_control(env, args, ckpt, live, seconds):
    """The insertion BC drives the arm joints + gripper (closed loop on the sim state) for `seconds`."""
    import numpy as np
    import torch
    from torch import nn
    import basket_env as be
    import plate_env as pe
    ck = torch.load(ckpt, map_location="cpu")
    net = nn.Sequential(nn.Linear(ck["in_dim"], 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(),
                        nn.Linear(256, ck["out_dim"]))
    net.load_state_dict(ck["state"])
    robot, plate, n = env.scene["robot"], env.scene["plate"], env.num_envs
    arm = env.action_manager.get_term("arm_action")
    aj = arm._joint_ids
    fj = [robot.joint_names.index(j) for j in ("left_finger_joint", "right_finger_joint")]
    org = env.scene.env_origins
    q = robot.data.joint_pos.clone()
    closed = torch.ones(n, dtype=torch.bool, device=env.device)
    max_step = args.sim_joint_speed / 30.0
    for t in range(int(seconds * 30)):
        tp, tq = pe.tcp_pose(env)
        if t % 150 == 0:
            st_ = be.basket_state(env)
            print(f"[bc] t {t / 30:5.0f}s  d_bottom median {float(st_['d_bottom'][live].abs().median()) * 1000:5.1f} mm  low median "
                  f"{float(st_['low'][live].median()) * 1000:5.0f} mm  closed {int(closed[live].sum())}/{int(live.sum())}", flush=True)
        obs = basket_features(torch.rad2deg(robot.data.joint_pos[:, aj]).cpu().numpy(), torch.cat([tp - org, tq], 1).cpu().numpy(),
                              torch.cat([plate.data.root_pos_w - org, plate.data.root_quat_w], 1).cpu().numpy(),
                              env._slot.cpu().numpy(), closed.cpu().numpy())
        with torch.no_grad():
            a = net(torch.tensor(obs)).numpy()
        dq = np.clip(a[:, :6] * ck["act_deg"], -max_step * 3, max_step * 3)
        tgt = torch.tensor(np.radians(torch.rad2deg(robot.data.joint_pos[:, aj]).cpu().numpy() + dq), dtype=torch.float32, device=env.device)
        q[:, aj] = torch.where(live[:, None], tgt, q[:, aj])
        closed = torch.where(live, torch.tensor(a[:, 6] < 0, device=env.device), closed)
        q[:, fj] = torch.where(closed[:, None], torch.full_like(q[:, fj], pe.FINGER_CLOSED), torch.zeros_like(q[:, fj]))
        robot.set_joint_position_target(q)
        sim_steps(env)
    return closed


def run_bc_eval(args):
    """Grasp (library replay) and then let the BC policy drive the arm joints + gripper (closed loop on the sim state)
    for up to 60 s; success test as the expert (rl/basket.yaml success)."""
    import numpy as np
    import torch
    from torch import nn
    from isaaclab.envs import ManagerBasedRLEnv
    import basket_env as be
    import plate_env as pe
    ck = torch.load(args.bc_ckpt, map_location="cpu")
    net = nn.Sequential(nn.Linear(ck["in_dim"], 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(),
                        nn.Linear(256, ck["out_dim"]))
    net.load_state_dict(ck["state"])
    env = ManagerBasedRLEnv(be.make_basket_cfg(args.num_envs, viewer=not args.headless))
    s = grasped_start(env, args, np.random.default_rng(args.seed))
    robot, plate, n = env.scene["robot"], env.scene["plate"], env.num_envs
    arm = env.action_manager.get_term("arm_action")
    aj = arm._joint_ids
    fj = [robot.joint_names.index(j) for j in ("left_finger_joint", "right_finger_joint")]
    org = env.scene.env_origins
    q = robot.data.joint_pos.clone()
    live = s["ok"].clone()
    closed = torch.ones(n, dtype=torch.bool, device=env.device)
    max_step = args.sim_joint_speed / 30.0
    for t in range(int(args.bc_eval_s * 30)):
        tp, tq = pe.tcp_pose(env)
        if t % 300 == 0:
            st_ = be.basket_state(env)
            d_tcp = ((tp - org)[:, :2] - env._slot[:, :2]).norm(dim=1)
            print(f"[bc_eval] t {t / 30:5.0f}s  tcp-slot xy median {float(d_tcp[live].median()) * 1000:5.0f} mm  tcp z median "
                  f"{float((tp - org)[live, 2].median()) * 1000:5.0f} mm  plate low median {float(st_['low'][live].median()) * 1000:5.0f} mm  "
                  f"closed {int(closed[live].sum())}", flush=True)
        obs = basket_features(torch.rad2deg(robot.data.joint_pos[:, aj]).cpu().numpy(), torch.cat([tp - org, tq], 1).cpu().numpy(),
                              torch.cat([plate.data.root_pos_w - org, plate.data.root_quat_w], 1).cpu().numpy(),
                              env._slot.cpu().numpy(), closed.cpu().numpy())
        with torch.no_grad():
            a = net(torch.tensor(obs)).numpy()
        dq = np.clip(a[:, :6] * ck["act_deg"], -max_step * 3, max_step * 3)
        q_meas = torch.rad2deg(robot.data.joint_pos[:, aj]).cpu().numpy()
        tgt = torch.tensor(np.radians(q_meas + dq), dtype=torch.float32, device=env.device)
        q[:, aj] = torch.where(live[:, None], tgt, q[:, aj])
        closed = torch.where(live, torch.tensor(a[:, 6] < 0, device=env.device), closed)
        q[:, fj] = torch.where(closed[:, None], torch.full_like(q[:, fj], pe.FINGER_CLOSED), torch.zeros_like(q[:, fj]))
        robot.set_joint_position_target(q)
        sim_steps(env)
    p0 = plate.data.root_pos_w.clone()
    for _ in range(30):
        sim_steps(env)
    still = (plate.data.root_pos_w - p0).norm(dim=1) < 0.01
    st = be.basket_state(env)
    sc = be.B["success"]
    inside = ((st["d_bottom"].abs() < sc["bottom_tol_m"]) & (st["d_u"].abs() < sc["centre_tol_m"])
              & (st["low"] < sc["max_height_m"]) & (st["normal_err"] < sc["max_normal_err_deg"]))
    succ = inside & still & live & ~closed
    print(f"[bc_eval] grasped {int(live.sum())}/{n}, inserted {int(succ.sum())}/{int(live.sum())} "
          f"(open {int((~closed & live).sum())})", flush=True)
    env.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if args.mode == "bc_train":                                 # torch only, no Isaac app
        run_bc_train(args)
        sys.exit(0)
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    APP = app
    try:
        {"check": run_check, "grasp": run_grasp, "expert": run_expert, "bc_eval": run_bc_eval}[args.mode](args)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        app.close()
