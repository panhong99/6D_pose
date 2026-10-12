"""Evaluate / watch / diagnose the plate rim-grasp policy (Isaac Lab python).

  ISAAC=~/IsaacLab/isaaclab.sh
  $ISAAC -p rl/eval.py gate --headless --checkpoint <model.pt> [--num_envs 2048]   # deterministic success rate (EVAL_JSON)
  $ISAAC -p rl/eval.py watch [--checkpoint <model.pt>]                            # GUI; default follows the newest checkpoint
  $ISAAC -p rl/eval.py demo --checkpoint <model.pt>                               # GUI: home -> ready MoveJ -> policy
  $ISAAC -p rl/eval.py export --headless --checkpoint <model.pt> --plate_x X --plate_y Y [--plate_yaw_deg D]  # real-robot plan
  $ISAAC -p rl/eval.py library --headless --checkpoint <model.pt> --approach ik --residual off   # plans per radius
  $ISAAC -p rl/eval.py serve --headless --checkpoint <model.pt> [--approach ik --residual off]   # warm export server
  $ISAAC -p rl/eval.py trace --headless --checkpoint <model.pt> [--bias_z 1.0]     # per-second state + reward terms
  $ISAAC -p rl/eval.py scripted --headless [--no_shield] [--lift_m 0.08]           # scripted grasp, no policy

  gate     : every env runs exactly one episode with the mean action; prints `EVAL_JSON {...}` (read by train.py)
  watch    : few envs in the GUI, first hold_start_s of each episode frozen at the start pose, reloads newer checkpoints
  demo     : the full real sequence in the GUI: home pose, smooth joint MoveJ to the ready joints (not learned), then the
             stage-2 policy grasps / lifts / holds; repeats with a new plate pose
  trace    : one deterministic episode; each second: held / TCP rise / plate lift / tilt / fingers / rim distance /
             actions / orientation lock / shield blocks, plus the mean of every reward term at a few instants.
             --bias_z adds an upward action in seconds 2-3 (does the plate follow? does the policy put it back down?)
  scripted : physics check without learning: from the ready start advance straight to the rim target, close, raise the
             TCP lift_m, hold hold_s with active position control, then zero_s with zero arm action (controller drift)
All take --stage (2 grasp, 3 place).  Env numbers: rl/config.yaml.
"""
import argparse
import json
import time

from rl_utils import load_config, newest_stage_checkpoint


DEFAULT_ENVS = {"gate": 2048, "watch": 4, "demo": 4, "trace": 32, "scripted": 8, "export": 16, "serve": 8, "library": 1}


def parse_args():
    # one flat parser: AppLauncher.add_app_launcher_args() pre-parses sys.argv, which breaks argparse sub-commands
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=tuple(DEFAULT_ENVS))
    p.add_argument("--stage", type=int, default=2, choices=(2, 3))
    p.add_argument("--checkpoint", default=None, help="gate/trace: required; watch: default follows the newest")
    p.add_argument("--num_envs", type=int, default=None, help=f"default per mode {DEFAULT_ENVS}")
    p.add_argument("--reload_s", type=float, default=30.0, help="watch: checkpoint poll period")
    p.add_argument("--hold_start_s", type=float, default=1.0, help="watch: freeze each episode at its start pose")
    p.add_argument("--residual", choices=("config", "on", "off"), default="config",
                   help="add the frozen BC base (config.yaml residual) regardless of residual.enabled")
    p.add_argument("--move_s", type=float, default=15.0, help="demo: home -> ready MoveJ duration (slow: J6 turns ~87 deg)")
    p.add_argument("--speed_cap", type=float, default=None, help="gate/demo/trace: override action.max_tcp_speed (m/s)")
    p.add_argument("--hold_windup_m",type=float,default=None,help="candidate: reduce maximum held TCP target lead [m]")
    p.add_argument("--episode_s",type=float,default=None,help="candidate: extend slow-task deadline, never shorten it")
    p.add_argument("--arm_joint_speed_deg_s",type=float,default=None,help="candidate arm drive/IK joint speed cap, <=6 deg/s")
    p.add_argument("--residual_base", default=None, help="override config residual.base (the BC actor this checkpoint was "
                   "trained on); implies residual on")
    p.add_argument("--plate_x", type=float, default=None, help="export: real plate root x [m] (camera_tools.py locate)")
    p.add_argument("--plate_y", type=float, default=None, help="export: real plate root y [m]")
    p.add_argument("--plate_yaw_deg", type=float, default=0.0, help="export: real plate yaw [deg]")
    p.add_argument("--wp_deg", type=float, default=1.5, help="export: keep a waypoint every this many joint degrees")
    p.add_argument("--bias_z", type=float, default=0.0, help="trace: extra +z action in seconds 2-3")
    p.add_argument("--no_shield", action="store_true", help="scripted: accept every IK target")
    p.add_argument("--lift_m", type=float, default=0.08, help="scripted: TCP raise after closing")
    p.add_argument("--hold_s", type=float, default=5.0, help="scripted: active hold after the raise")
    p.add_argument("--zero_s", type=float, default=5.0, help="scripted: zero arm action afterwards (drift check)")
    p.add_argument("--approach", choices=("policy", "ik"), default="policy",
                   help="ik: straight TCP line to the rim (no learning, valid anywhere reachable), the policy takes over "
                        "within --handoff_m of the rim (grasp / lift / hold)")
    p.add_argument("--rotate", action="store_true", help="export/serve: roll out on +x at the plate radius and turn "
                   "joint 1 by the plate azimuth (any direction, the policy only needs the radius)")
    p.add_argument("--approach_fast", type=float, default=1.0, help="export/serve with --approach ik: run the line this "
                   "many times faster in SIM while all envs are > 2 x handoff_m away (path only; real speed unchanged)")
    p.add_argument("--handoff_m", type=float, default=0.04, help="approach ik: rim distance where the policy takes over")
    p.add_argument("--export_hold_s", type=float, default=1.0, help="export/serve: end a rollout after this long a "
                   "stable lift (the 5 s task hold adds nothing to the plan; the real robot holds hold_s itself)")
    p.add_argument("--r_min", type=float, default=1.00, help="library: smallest plate radius from the base axis [m]")
    p.add_argument("--r_max", type=float, default=1.32, help="library: largest plate radius [m]")
    p.add_argument("--r_step", type=float, default=0.002, help="library: radius step [m]")
    p.add_argument("--pre_m", type=float, default=0.10, help="library: the direct joint move ends this far from the rim")
    p.add_argument("--plate_dz", type=float, default=0.0,
                   help="candidate sim only: plate/support height shift in metres; no real config change")
    p.add_argument("--output", default=None, help="library: new output directory (refuse existing library)")
    p.add_argument("--grasp_bank", default=None, help="opt-in IK start/goal bank (phi/pitch), sim candidates only")
    p.add_argument("--bank_fixed", action="store_true", help="gate: test every bank row exactly once (requires one env per row)")
    p.add_argument('--replay_library',default=None,help='gate candidate: stored joint paths; not live BC inference')
    p.add_argument('--confirmed_grasp_waypoint',action='store_true',help='library candidate: use physically settled grasp pose as close waypoint')
    p.add_argument("--latch_gripper", action="store_true", help="stage2 candidate: retain BC's valid near-rim close command until reset")
    p.add_argument("--grasp_wait",action="store_true",help="stage2 candidate: hold arm until measured jaws stall on a held plate, then BC lift")
    p.add_argument("--bank_azimuth_deg", type=float, default=0.0, help="gate: random joint-1 rotations of a candidate bank")
    p.add_argument("--seed", type=int, default=None, help="fixed simulation evaluation seed (does not change shared config)")
    p.add_argument("--port", type=int, default=5012, help="serve: TCP port on 127.0.0.1 (one JSON line per request)")
    p.add_argument("--dump", default=None, help="gate: save per-env start plate xy + outcome to this .npz (failure map)")
    p.add_argument("--plate_range", type=float, nargs=2, default=None, metavar=("DX", "DY"),
                   help="override the plate reset range to +-DX, +-DY m around plate_xy (wider test area)")
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    args.num_envs = args.num_envs or DEFAULT_ENVS[args.mode]
    if args.mode in ("gate", "trace", "demo", "export", "serve", "library") and not args.checkpoint:
        p.error(f"{args.mode} needs --checkpoint")
    if (args.latch_gripper or args.grasp_wait) and args.stage!=2:p.error('Grasp sequencer is only for stage2 grasp/lift/hold')
    if args.replay_library and (args.mode!='gate' or not args.bank_fixed or not args.grasp_bank or args.arm_joint_speed_deg_s is None):
        p.error('Library replay requires gate, fixed bank and explicit slow joint cap')
    return args


# ---------------------------------------------------------------- shared

def checkpoint_obs_dim(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]["actor.0.weight"].shape[1]


def build_env(args, viewer=False, terminations=True):
    import plate_env
    from isaaclab.envs import ManagerBasedRLEnv
    rim_rel = None
    if getattr(args, "checkpoint", None):                  # match the observation layout the checkpoint was trained with
        rim_rel = checkpoint_obs_dim(args.checkpoint) == 46
    cfg = plate_env.make_env_cfg(args.stage, args.num_envs, viewer=viewer, rim_rel=rim_rel)
    joint_cap=None
    if args.arm_joint_speed_deg_s is not None:
        import math
        if not 0<args.arm_joint_speed_deg_s<=6:raise ValueError('Arm joint cap must be <=6 deg/s')
        joint_cap=math.radians(args.arm_joint_speed_deg_s)
        for name,actuator in cfg.scene.robot.actuators.items():
            if name.startswith('arm_'):actuator.velocity_limit_sim=joint_cap
    if args.episode_s is not None:
        if args.episode_s<cfg.episode_length_s:raise ValueError('Slow-task deadline may only be extended')
        cfg.episode_length_s=args.episode_s
    if args.seed is not None:
        cfg.seed=args.seed
    dz = getattr(args, "plate_dz", 0.0)
    if dz:
        # One height per scene: resize the support so its bottom stays on the table.
        height = plate_env.G["box_h"] + dz
        if height <= 0:
            raise ValueError("plate_dz would put the support below the table")
        plate_env.G["box_h"] = height
        plate_env.REST_Z = plate_env.G["table_top_z"] + height
        cfg.scene.start_box.spawn.size = (plate_env.G["box_xy"], plate_env.G["box_xy"], height)
        cfg.scene.start_box.init_state.pos = (*plate_env.G["plate_xy"], plate_env.G["table_top_z"] + height / 2)
        cfg.scene.plate.init_state.pos = (*plate_env.G["plate_xy"], plate_env.REST_Z)
    if getattr(args, "plate_range", None):
        dx, dy = args.plate_range
        cfg.events.reset_plate.params["pose_range"].update(x=(-dx, dx), y=(-dy, dy))
    if getattr(args, "speed_cap", None):                    # read at every step by SafeDiffIKAction.process_actions
        plate_env.A["max_tcp_speed"] = args.speed_cap
    if args.hold_windup_m is not None:
        if not 0<args.hold_windup_m<=plate_env.A['hold_windup']:raise ValueError('Candidate windup may only be reduced')
        plate_env.A['hold_windup']=args.hold_windup_m
    if not terminations:                                    # diagnostics run past success / timeouts
        cfg.episode_length_s = 3600.0
        cfg.terminations.success = None
        cfg.terminations.plate_lost = None
    env = ManagerBasedRLEnv(cfg)
    if joint_cap is not None:env._arm_joint_speed_rad_s=joint_cap
    if getattr(args, "grasp_bank", None):
        from grasp_variants import install
        install(env, args.grasp_bank, fixed=args.mode == "library" or args.bank_fixed, azimuth_deg=args.bank_azimuth_deg)
        env.reset()
    return env


def tilt_deg(st):
    import torch
    return torch.rad2deg(torch.acos(st["plate_up_z"].clamp(-1, 1)))


def approach_wrapper(args, env, policy):
    """--approach ik: until an env's TCP first comes within handoff_m of the rim target, replace the policy with a
    straight-line TCP move to the rim (the IL expert's approach: translation only, gripper open, max_tcp_speed x 0.85);
    from then on the policy acts.  The latch resets with the episode.  Returns policy unchanged for --approach policy."""
    use_ik=getattr(args,"approach","policy")=="ik"
    wait=getattr(args,'grasp_wait',False)
    latch=getattr(args,'latch_gripper',False) or wait
    if not use_ik and not latch:
        return policy
    import torch
    import plate_env as pe
    cap = 0.85 * pe.A["max_tcp_speed"] * env.step_dt / pe.A["scale"]
    handed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    closed=torch.zeros_like(handed)
    released=torch.zeros_like(handed)
    stall=torch.zeros(env.num_envs,device=env.device)
    fingers_prev=torch.zeros_like(stall)

    base_speed, fast = pe.A["max_tcp_speed"], getattr(args, "approach_fast", 1.0)

    def act(obs):
        a = policy(obs)
        st = pe._state(env)
        handed[env.episode_length_buf <= 1] = False
        closed[env.episode_length_buf <= 1] = False
        released[env.episode_length_buf<=1]=False
        stall[env.episode_length_buf<=1]=0
        handed.logical_or_(st["d_rim"] < args.handoff_m)
        # --approach_fast (export / serve, SIM ONLY: the real robot replays the joint path at its own 6 deg/s): the
        # straight line runs k x faster while EVERY env is still > 2 x handoff_m from the rim, so the policy part
        # always runs at the trained speed cap
        k = fast if fast > 1.0 and bool((st["d_rim"] > 2 * args.handoff_m).all()) else 1.0
        pe.A["max_tcp_speed"] = base_speed * k
        err = st["rim"] - st["tcp"]
        line = torch.zeros_like(a)
        line[:, :3] = err / err.norm(dim=1, keepdim=True).clamp(min=1e-9) * cap * k
        line[:, 6] = 1.0                                    # open
        if use_ik:a=torch.where(handed[:, None],a,line)
        if latch:
            # BC still chooses when to close; no prescribed arm lift or invented contact/success signal.
            start=(~closed)&(a[:,6]<0)&(st['d_rim']<.010)
            closed.logical_or_(start)
            if wait:
                arm=env.action_manager.get_term('arm_action')
                arm.hold_pos[start]=arm._compute_frame_pose()[0][start]
                settled=(st['fingers']-fingers_prev).abs()<1e-3
                stall.copy_(torch.where(closed&~released&st['held']&settled,stall+1,torch.zeros_like(stall)))
                released.logical_or_(stall>=5)
                a[closed&~released,:6]=0
                fingers_prev.copy_(st['fingers'])
            a[:,6]=torch.where(closed,-torch.ones_like(a[:,6]),a[:,6])
        return a
    return act


# ---------------------------------------------------------------- gate

def run_gate(args, cfg):
    import torch
    from rl_utils import load_policy
    env = build_env(args)
    venv, _, policy = load_policy(env, cfg, args.stage, args.checkpoint)
    policy = approach_wrapper(args, env, policy)
    if args.replay_library:
        from grasp_variants import library_replay
        policy=library_replay(env,args.grasp_bank,args.replay_library)
    n, names = env.num_envs, ["success", "collision", "plate_lost", "time_out"]
    outcome = torch.full((n,), -1, dtype=torch.long, device=env.device)                # -1 running, else index in names
    shield_steps = torch.zeros(n, device=env.device)
    arm = env.action_manager.get_term("arm_action")
    import plate_env as pe
    speed, z_lo, z_hi = [], torch.full((n,), 9.0, device=env.device), torch.full((n,), -9.0, device=env.device)
    held_measured = torch.zeros(n, dtype=torch.bool, device=env.device)
    tcp_xy_anchor = torch.zeros(n, 2, device=env.device)
    plate_xy_anchor = torch.zeros_like(tcp_xy_anchor)
    tcp_xy_drift = torch.zeros(n, device=env.device)
    plate_xy_drift = torch.zeros_like(tcp_xy_drift)
    tcp_forward = torch.zeros_like(tcp_xy_drift)
    plate_forward = torch.zeros_like(tcp_xy_drift)
    obs = venv.get_observations()
    plate_xy0 = (env.scene["plate"].data.root_pos_w - env.scene.env_origins)[:, :2].clone()
    variant_start = {}
    if args.grasp_bank:
        idx=env._grasp_bank_index.clone()
        variant_start=dict(bank_index=idx.cpu().numpy(),
                           phi_deg=env._grasp_bank['phi_deg'][idx].cpu().numpy(),
                           pitch_deg=env._grasp_bank['pitch_deg'][idx].cpu().numpy(),
                           azimuth_deg=torch.rad2deg(env._grasp_azimuth).cpu().numpy())
    ever_held = torch.zeros(n, dtype=torch.bool, device=env.device)        # --dump diagnostics of the first episode
    max_lift = torch.full((n,), -1.0, device=env.device)
    min_drim = torch.full((n,), 9.0, device=env.device)
    max_tcp_speed = torch.zeros(n, device=env.device)
    max_arm_speed = torch.zeros_like(max_tcp_speed)
    max_tool_rotation_speed = torch.zeros_like(max_tcp_speed)
    max_tilt_held = torch.zeros(n, device=env.device)
    # ManagerBasedRLEnv resets terminated rows inside step(). Capture their physical
    # end state first so a collision's final movement is not omitted from speed metrics.
    terminal_reset=torch.zeros(n,dtype=torch.bool,device=env.device)
    terminal_tcp=torch.zeros(n,3,device=env.device)
    terminal_quat=torch.zeros(n,4,device=env.device)
    terminal_joint_speed=torch.zeros(n,device=env.device)
    original_reset=env._reset_idx
    def record_terminal(ids):
        pos,quat=pe.tcp_pose(env)
        terminal_reset[ids]=True;terminal_tcp[ids]=pos[ids];terminal_quat[ids]=quat[ids]
        velocity=env.scene['robot'].data.joint_vel[:,arm._joint_ids]
        terminal_joint_speed[ids]=torch.rad2deg(velocity[ids].abs().amax(dim=1))
        return original_reset(ids)
    env._reset_idx=record_terminal
    terminal_measured=0
    with torch.inference_mode():
        tcp_prev = pe._state(env)["tcp"].clone()
        quat_prev=pe.tcp_pose(env)[1].clone()
        for _ in range(int(env.cfg.episode_length_s * 30) + 5):
            terminal_reset.zero_()
            obs, _, _, _ = venv.step(policy(obs))
            running = outcome < 0
            shield_steps += arm.blocked.float() * running
            st = pe._state(env)                             # motion quality: TCP speed, plate bobbing while held up
            measured_tcp=torch.where(terminal_reset[:,None],terminal_tcp,st['tcp'])
            measured_quat=torch.where(terminal_reset[:,None],terminal_quat,pe.tcp_pose(env)[1])
            measured=running&((env.episode_length_buf>1)|terminal_reset)
            terminal_measured+=int((running&terminal_reset).sum())
            v = torch.norm(measured_tcp - tcp_prev, dim=1) / env.step_dt
            dots=(measured_quat*quat_prev).sum(1).abs()
            relative_vec=quat_prev[:,:1]*measured_quat[:,1:]-measured_quat[:,:1]*quat_prev[:,1:]-torch.cross(quat_prev[:,1:],measured_quat[:,1:],dim=1)
            rot_speed=torch.rad2deg(2*torch.atan2(relative_vec.norm(dim=1),dots))/env.step_dt
            max_tool_rotation_speed=torch.where(measured,torch.maximum(max_tool_rotation_speed,rot_speed),max_tool_rotation_speed)
            tcp_prev = st["tcp"].clone()
            quat_prev=pe.tcp_pose(env)[1].clone()
            speed.append(v[measured])
            max_tcp_speed = torch.where(measured,
                                        torch.maximum(max_tcp_speed, v), max_tcp_speed)
            joint_speed=torch.rad2deg(env.scene['robot'].data.joint_vel[:,arm._joint_ids].abs().amax(dim=1))
            joint_speed=torch.where(terminal_reset,terminal_joint_speed,joint_speed)
            max_arm_speed=torch.where(measured,torch.maximum(max_arm_speed,joint_speed),max_arm_speed)
            # Raw movement since the first valid held sample, accumulated per episode while held.
            # Auto-reset replaces a terminated episode's final physical state: exclude those samples
            # (episode_length_buf == 0), so these maxima can underestimate the terminal movement.
            held_sample = running & (env.episode_length_buf > 1) & st["held"]
            first_held = held_sample & ~held_measured
            tcp_xy_anchor[first_held] = st["tcp"][first_held, :2]
            plate_xy_anchor[first_held] = st["p"][first_held, :2]
            held_measured |= first_held
            tcp_delta = st["tcp"][:, :2] - tcp_xy_anchor
            plate_delta = st["p"][:, :2] - plate_xy_anchor
            tcp_xy_drift = torch.where(held_sample, torch.maximum(tcp_xy_drift, tcp_delta.norm(dim=1)), tcp_xy_drift)
            plate_xy_drift = torch.where(held_sample, torch.maximum(plate_xy_drift, plate_delta.norm(dim=1)), plate_xy_drift)
            tcp_forward = torch.where(held_sample, torch.maximum(tcp_forward, tcp_delta[:, 0]), tcp_forward)
            plate_forward = torch.where(held_sample, torch.maximum(plate_forward, plate_delta[:, 0]), plate_forward)
            live = running & (env.episode_length_buf > 1)
            ever_held |= live & st["held"]
            max_lift = torch.where(live, torch.maximum(max_lift, st["lift"]), max_lift)
            min_drim = torch.where(live, torch.minimum(min_drim, st["d_rim"]), min_drim)
            max_tilt_held = torch.where(live & st["held"], torch.maximum(max_tilt_held, tilt_deg(st)), max_tilt_held)
            up = running & st["stable_lift"]
            z_lo = torch.where(up, torch.minimum(z_lo, st["lift"]), z_lo)
            z_hi = torch.where(up, torch.maximum(z_hi, st["lift"]), z_hi)
            for k, name in enumerate(names):
                outcome[env.termination_manager.get_term(name) & running & (outcome < 0)] = k
            if (outcome >= 0).all():
                break
    res = {name: float((outcome == k).float().mean()) for k, name in enumerate(names)}
    res.update(stage=args.stage, episodes=n, unfinished=float((outcome < 0).float().mean()),
               mean_shield_steps=float(shield_steps.mean()), checkpoint=str(args.checkpoint),
               approach=args.approach, plate_range=args.plate_range,grasp_bank=args.grasp_bank,
               bank_azimuth_deg=args.bank_azimuth_deg,bank_fixed=args.bank_fixed,seed=args.seed,
               gripper_latched=args.latch_gripper or args.grasp_wait,grasp_wait=args.grasp_wait)
    res['hold_windup_m']=args.hold_windup_m
    res['episode_s']=env.cfg.episode_length_s
    res['execution_mode']='library_joint_replay' if args.replay_library else 'live_policy'
    res['replay_library']=args.replay_library
    res['arm_joint_speed_cap_deg_s']=args.arm_joint_speed_deg_s
    res['arm_joint_speed_max_deg_s']=float(max_arm_speed.max())
    res['arm_joint_over_6deg_episodes']=int((max_arm_speed>6.001).sum())
    res['terminal_speed_samples']=terminal_measured
    res['tool_rotation_speed_max_deg_s']=float(max_tool_rotation_speed.max())
    res['tool_rotation_over_10deg_episodes']=int((max_tool_rotation_speed>10.001).sum())
    sp, held_up = torch.cat(speed), z_hi > z_lo
    res.update(speed_cap=args.speed_cap, tcp_speed_p50=float(sp.median()), tcp_speed_p95=float(sp.quantile(0.95)),
               tcp_speed_max=float(sp.max()),
               speed_over_cap_episodes=int((max_tcp_speed > (args.speed_cap or cfg['action']['max_tcp_speed']) + 1e-5).sum()),
               speed_over_original_limit_episodes=int((max_tcp_speed > cfg['action']['max_tcp_speed']+1e-5).sum()),
               lift_range_during_hold_mm=float((z_hi - z_lo)[held_up].median() * 1000) if held_up.any() else None,
               lift_max_mm=float(z_hi[held_up].median() * 1000) if held_up.any() else None)
    def held_mm(values, quantile=None):
        measured = values[held_measured]
        if not measured.numel():
            return None
        return float((measured.max() if quantile is None else measured.quantile(quantile)) * 1000)

    res.update(held_tcp_xy_drift_p95_mm=held_mm(tcp_xy_drift, 0.95),
               held_tcp_xy_drift_max_mm=held_mm(tcp_xy_drift),
               held_tcp_forward_p95_mm=held_mm(tcp_forward, 0.95),
               held_plate_xy_drift_p95_mm=held_mm(plate_xy_drift, 0.95),
               held_plate_xy_drift_max_mm=held_mm(plate_xy_drift),
               held_plate_forward_p95_mm=held_mm(plate_forward, 0.95),
               held_measured_episodes=int(held_measured.sum()))
    if args.dump:
        import numpy as np
        np.savez(args.dump, plate_xy=plate_xy0.cpu().numpy(), outcome=outcome.cpu().numpy(), names=np.array(names),
                 ever_held=ever_held.cpu().numpy(), max_lift=max_lift.cpu().numpy(), min_drim=min_drim.cpu().numpy(),
                 max_tilt_held=max_tilt_held.cpu().numpy(),max_tcp_speed_mps=max_tcp_speed.cpu().numpy(),
                 max_arm_joint_speed_deg_s=max_arm_speed.cpu().numpy(),
                 max_tool_rotation_speed_deg_s=max_tool_rotation_speed.cpu().numpy(),**variant_start)
    print("EVAL_JSON " + json.dumps(res), flush=True)
    env.close()


# ---------------------------------------------------------------- watch (GUI)

def run_watch(args, cfg, app):
    import torch
    from rl_utils import load_policy
    env = build_env(args, viewer=True)
    fixed = args.checkpoint
    loaded, t_check, policy, venv, obs = None, 0.0, None, None, None
    while app.is_running():
        if fixed is None and time.time() - t_check > args.reload_s or loaded is None:
            t_check = time.time()
            ck = fixed or newest_stage_checkpoint(cfg, args.stage)
            if ck is not None and str(ck) != str(loaded):
                venv, _, policy = load_policy(env, cfg, args.stage, ck)
                policy = approach_wrapper(args, env, policy)
                obs, loaded = venv.get_observations(), ck
                print(f"[watch] stage {args.stage}: loaded {ck}", flush=True)
        if policy is None:
            app.update()
            time.sleep(1.0)
            continue
        with torch.inference_mode():
            act = policy(obs)
            frozen = env.episode_length_buf < int(args.hold_start_s * 30)                  # show the start pose first
            act[frozen] = 0.0
            act[frozen, 6] = 1.0                                                        # open (binary: < 0 closes)
            obs, _, _, _ = venv.step(act)
        for name in ("success", "collision", "plate_lost"):
            hit = env.termination_manager.get_term(name)
            if hit.any():
                print(f"[watch] env {hit.nonzero().flatten().tolist()} ended: {name}", flush=True)
    env.close()


# ---------------------------------------------------------------- demo (GUI): home -> ready MoveJ -> stage-2 policy

def run_demo(args, cfg, app):
    """The whole real sequence: hold at home, a smooth joint-space MoveJ to the ready joints (what the real robot does
    with the pendant / mover, not learned), then the stage-2 policy grasps and lifts.  Repeats with a new plate pose."""
    import torch
    import plate_env as pe
    from rl_utils import load_policy
    env = build_env(args, viewer=True, terminations=False)
    venv, _, policy = load_policy(env, cfg, args.stage, args.checkpoint)
    robot, arm, dev = env.scene["robot"], env.action_manager.get_term("arm_action"), env.device
    all_ids = torch.arange(env.num_envs, device=dev)
    home = robot.data.default_joint_pos.clone()                    # scene default = home_deg (config.yaml robot)
    dt = env.step_dt
    with torch.inference_mode():                                   # whole loop: reset writes into inference tensors
        while app.is_running():
            env.reset()
            ready = robot.data.joint_pos.clone()                       # reset_to_ready wrote the ready joints
            path = [home + (ready - home) * (3 * s * s - 2 * s ** 3)   # smoothstep: zero velocity at both ends
                    for s in [0.0] * int(1.0 / dt) + [k / int(args.move_s / dt) for k in range(int(args.move_s / dt) + 1)]]
            unsafe = sum(int((~arm.shield.safe(q[:, arm._joint_ids])).any()) for q in path)
            print(f"[demo] home -> ready MoveJ {args.move_s:g} s (shield-unsafe waypoints: {unsafe})", flush=True)
            for q in path:                                             # kinematic replay; physics keeps running for the plate
                robot.write_joint_state_to_sim(q, torch.zeros_like(q))
                robot.set_joint_position_target(q)
                for _ in range(env.cfg.decimation):
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(env.physics_dt)
                env.sim.render()
            env.episode_length_buf[:] = 0                              # the policy episode starts now, at ready
            pe.reset_task_state(env, all_ids)
            arm.reset(all_ids)
            obs = venv.get_observations()
            for t in range(int(cfg["task"]["episode_s"][2] / dt)):
                obs, _, _, _ = venv.step(policy(obs))
                if t % 30 == 29:
                    st = pe._state(env)
                    print(f"[demo] t={(t + 1) * dt:4.1f}s held {st['held'].float().mean():.2f} "
                          f"plate lift mm {float(st['lift'].median()) * 1000:5.1f} hold steps {float(env._pp['hold'].median()):.0f}",
                          flush=True)
                if not app.is_running():
                    break
    env.close()


# ---------------------------------------------------------------- export (sim rollout -> real supervised mover plan)

def run_export(args, cfg):
    """Roll the policy out in sim with the plate at the REAL measured pose (camera_tools.py locate) and write a plan for
    calibration/ros2_nodes.py mover (one human `go` per waypoint, <= 2 deg/s).  Open loop: the real robot replays the
    sim joint path, nothing reacts on the real robot.  Legs, in order (gripper by the Human with gripper.py between):
      to_ready (home -> ready MoveJ) | grasp (ready -> rim; then `gripper.py close`) | lift (raise + hold path)
      | lower (reverse of lift; then `gripper.py open`) | retreat (reverse of grasp) | to_home (reverse of to_ready)
      + to_grasp = to_ready + grasp, from_grasp = retreat + to_home (pipeline.py runs these with one go each)"""
    from rl_utils import load_policy
    env = build_env(args, terminations=False)
    venv, _, policy = load_policy(env, cfg, 2, args.checkpoint)
    policy = approach_wrapper(args, env, policy)
    try:
        path, accepted = export_once(args, cfg, env, venv, policy)
    finally:
        env.close()
    if path is None:
        raise SystemExit("no successful rollout at this plate pose: do not run it on the real robot")


def export_once(args, cfg, env, venv, policy, say=None):
    """One export rollout on an existing env (run_export, run_serve).  Returns (plan.json path | None, accepted)."""
    import math
    import time as _time
    import numpy as np
    import torch
    import plate_env as pe
    from rl_utils import rl_path

    def log(msg, flush=True):
        print(msg, flush=flush)
        if say:
            say(msg)
    robot, plate, arm, dev = env.scene["robot"], env.scene["plate"], env.action_manager.get_term("arm_action"), env.device
    n, ids = env.num_envs, torch.arange(env.num_envs, device=env.device)
    with torch.inference_mode():
        env.reset()
        pose = torch.zeros(n, 7, device=dev)
        # --rotate: the arm is symmetric about joint 1 (base z; FK check 2026-10-08), so roll out with the plate on the
        # +x axis at the same radius (inside the trained band) and turn joint 1 of the whole path by the plate azimuth
        theta = math.atan2(args.plate_y, args.plate_x) if getattr(args, "rotate", False) else 0.0
        sim_x, sim_y = (math.hypot(args.plate_x, args.plate_y), 0.0) if theta else (args.plate_x, args.plate_y)
        pose[:, :3] = torch.tensor([sim_x, sim_y, pe.REST_Z], device=dev) + env.scene.env_origins
        yaw = math.radians(args.plate_yaw_deg) - theta
        pose[:, 3], pose[:, 6] = math.cos(yaw / 2), math.sin(yaw / 2)
        plate.write_root_pose_to_sim(pose)
        plate.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
        pe.box_under_plate(env, ids)                            # the support box sits under the measured plate
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        env.episode_length_buf[:] = 0
        pe.reset_task_state(env, ids)
        arm.reset(ids)
        obs = venv.get_observations()
        q_log, grip_log, done_at = [], [], torch.full((n,), -1, dtype=torch.long, device=dev)
        bad = torch.zeros(n, dtype=torch.bool, device=dev)
        for t in range(int((args.episode_s or cfg["task"]["episode_s"][2]) * 30)):
            q_log.append(torch.rad2deg(robot.data.joint_pos[:, arm._joint_ids]).cpu().numpy())
            a = policy(obs)
            base = venv.base(venv._obs) + venv.scale * a if hasattr(venv, "base") else a
            grip_log.append((base[:, 6] < 0).cpu().numpy())
            obs, _, _, _ = venv.step(a)
            st = pe._state(env)
            bad |= env.termination_manager.get_term("collision") | arm.blocked
            held_ok = st["hold"] >= int(args.export_hold_s * 30)          # stable lift long enough for the plan
            done_at = torch.where((done_at < 0) & held_ok, torch.full_like(done_at, t + 1), done_at)
            first = int(done_at[done_at >= 0].min()) if (done_at >= 0).any() else None
            if (done_at >= 0).all() or bad.all() or (first is not None and t + 1 - first >= 60):
                break                                           # slower envs get 2 s after the first; plan = 1st success
        q_log.append(torch.rad2deg(robot.data.joint_pos[:, arm._joint_ids]).cpu().numpy())
    ok = ((done_at >= 0) & ~bad).cpu().numpy()
    if theta:
        log(f"[export] rotate: rollout at radius {sim_x:.3f} m on +x, joint 1 turned {math.degrees(theta):+.1f} deg")
    log(f"[export] plate ({args.plate_x:.3f}, {args.plate_y:.3f}) yaw {args.plate_yaw_deg:.1f}: sim success "
          f"{ok.sum()}/{n} (collision/shield {int(bad.sum())})", flush=True)
    if not ok.any():
        return None, False
    i = int(np.flatnonzero(ok)[0])
    q = np.stack(q_log)[:, i]                                   # (T+1, 6) deg
    q[:, 0] += math.degrees(theta)                              # --rotate: back to the real plate azimuth
    grip = np.stack(grip_log)[:, i]
    c = int(np.argmax(grip))                                    # first step with the close command
    end = int(done_at[i])

    def thin(path, step_deg):
        out = [path[0]]
        for p in path[1:]:
            if np.abs(p - out[-1]).max() >= step_deg:
                out.append(p)
        if np.abs(path[-1] - out[-1]).max() > 1e-3:
            out.append(path[-1])
        return np.array(out)

    home = np.array(cfg["robot"]["home_deg"])
    ready = q[0]
    k = max(1, int(np.ceil(np.abs(ready - home).max() / 40.0)))
    to_ready = np.array([home + (ready - home) * j / k for j in range(k + 1)])
    grasp = thin(q[:c + 1], args.wp_deg)
    lift = thin(q[c:end + 1], args.wp_deg)
    legs = dict(to_ready=to_ready, grasp=grasp, lift=lift, lower=lift[::-1], retreat=grasp[::-1], to_home=to_ready[::-1])
    legs["to_grasp"] = np.concatenate([to_ready, grasp[1:]])          # home -> ready -> rim (one confirmed sequence)
    legs["from_grasp"] = np.concatenate([grasp[::-1], to_ready[::-1][1:]])   # rim -> ready -> home

    def clearance(wp):                                          # dense straight-joint segments, shield geometry
        dense = np.concatenate([np.linspace(a, b, 20) for a, b in zip(wp[:-1], wp[1:])])
        qq = torch.tensor(np.radians(dense), dtype=torch.float32, device=dev)
        f, tb = arm.shield.clearance(qq)
        return float(f.min()), float(tb.min())

    margin = cfg["shield"]["frame_margin"]
    plan = dict(dry_run=True, source="rl_policy_rollout", checkpoint=str(args.checkpoint), residual=cfg["residual"],
                approach=args.approach, handoff_m=args.handoff_m,
                created=_time.strftime("%Y-%m-%d %H:%M:%S"), cube_centre_m=[args.plate_x, args.plate_y, pe.REST_Z],
                rotate_deg=round(math.degrees(theta), 3), sim_plate_xy=[sim_x, sim_y],
                plate_yaw_deg=args.plate_yaw_deg, margin_m=margin, table_margin_m=cfg["shield"]["table_margin"],
                sim_success=f"{int(ok.sum())}/{n}", gripper_close_after_leg="grasp", gripper_open_after_leg="lower",
                gripper_force=64, legs={}, reasons=[])
    for name, wp in legs.items():
        f, tb = clearance(wp)
        plan["legs"][name] = dict(waypoints_deg=np.round(wp, 3).tolist(), min_clearance_straight_m=f,
                                  min_table_clearance_m=tb, planner="rl rollout" if name not in ("to_ready", "to_home") else "joint line")
        if f < margin or tb < cfg["shield"]["table_margin"]:
            plan["reasons"].append(f"{name}: clearance frame {f * 1000:.0f} mm / table {tb * 1000:.0f} mm below the margin")
        log(f"[export] {name:8s}: {len(wp):3d} waypoints, largest move {np.abs(np.diff(wp, axis=0)).max():5.1f} deg, "
              f"clearance frame {f * 1000:4.0f} mm table {tb * 1000:4.0f} mm", flush=True)
    plan["accepted"] = not plan["reasons"]
    out = rl_path(cfg["paths"]["calibration_dir"]) / "data" / "rl_plans" / _time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    np.savez_compressed(out / "rollout.npz", q_deg=q, grip_close=grip, close_step=c, success_step=end)
    print(f"EXPORT_PLAN {out / 'plan.json'} accepted={plan['accepted']}", flush=True)
    return out / "plan.json", plan["accepted"]


# ---------------------------------------------------------------- library: plans for every radius (pipeline.py, no Isaac at run time)

def run_library(args, cfg):
    """One env per plate radius r (plate on +x, y = 0).  The arm is symmetric about joint 1, so a real plate at
    (r cos t, r sin t) uses the r path with joint 1 turned by t: pipeline.py only picks, rotates and collision-checks.
      pass 1: ready -> straight TCP line toward the rim; q_pre = joints when the TCP is pre_m from the rim
      pass 2: home -> q_pre as ONE joint-space move (kinematic sweep with physics on: did the arm touch the plate?),
              then the last pre_m on the straight line and the policy (grasp, lift) until export_hold_s of stable lift
    Writes <out>/library.npz + library.json.  Joint-space angles in degrees."""
    import math
    import time as _time
    import numpy as np
    import torch
    import plate_env as pe
    from rl_utils import load_policy
    bank = np.load(args.grasp_bank, allow_pickle=False) if args.grasp_bank else None
    radii = bank['radius_m'] if bank is not None else np.round(np.arange(args.r_min, args.r_max + 1e-9, args.r_step), 4)
    args.num_envs = len(radii)
    env = build_env(args, terminations=False)
    venv, _, policy = load_policy(env, cfg, 2, args.checkpoint)
    act = approach_wrapper(args, env, policy)
    robot, plate, arm, dev = env.scene["robot"], env.scene["plate"], env.action_manager.get_term("arm_action"), env.device
    n, ids = env.num_envs, torch.arange(env.num_envs, device=env.device)
    home = robot.data.default_joint_pos.clone()
    contact = env.scene["arm_contacts"]

    def place():
        env.reset()
        pose = torch.zeros(n, 7, device=dev)
        pose[:, 0] = torch.tensor(radii, dtype=torch.float32, device=dev)
        if bank is not None:
            pose[:, :3]=torch.tensor(bank['plate_xyz'],dtype=torch.float32,device=dev)
        pose[:, 2], pose[:, 3] = pe.REST_Z, 1.0
        pose[:, :3] += env.scene.env_origins
        plate.write_root_pose_to_sim(pose)
        plate.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
        pe.box_under_plate(env, ids)
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)

    def start_episode():
        env.episode_length_buf[:] = 0
        pe.reset_task_state(env, ids)
        arm.reset(ids)
        return venv.get_observations()

    def sim_steps(k=None):
        for _ in range(k or env.cfg.decimation):
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)

    with torch.inference_mode():
        # pass 1: q_pre on the approach line
        place()
        obs = start_episode()
        q_pre = torch.full_like(robot.data.joint_pos, float("nan"))
        for t in range(int(20 * 30)):
            st = pe._state(env)
            hit = torch.isnan(q_pre[:, 0]) & (st["d_rim"] < args.pre_m)
            q_pre[hit] = robot.data.joint_pos[hit]
            if not torch.isnan(q_pre[:, 0]).any():
                break
            obs, _, _, _ = venv.step(act(obs))
        have_pre = ~torch.isnan(q_pre[:, 0])
        q_pre[~have_pre] = home[~have_pre]
        # pass 2: home -> q_pre in one joint move (smoothstep), physics running, then line + policy
        place()
        p0 = plate.data.root_pos_w.clone()
        # Smoothstep's peak derivative is 1.5; keep the simulated joint sweep <= 6 deg/s.
        move_s = max(args.move_s, float(torch.rad2deg(q_pre - home).abs().max()) * 1.5 / 6.0)
        steps = max(1, int(math.ceil(move_s * 30)))
        touched = torch.zeros(n, dtype=torch.bool, device=dev)
        for k in range(steps + 1):
            s_ = k / steps
            q = home + (q_pre - home) * (3 * s_ * s_ - 2 * s_ ** 3)
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            robot.set_joint_position_target(q)
            sim_steps()
            touched |= contact.data.net_forces_w.norm(dim=-1).amax(dim=1) > 1.0
        moved = (plate.data.root_pos_w - p0).norm(dim=1) > 0.003
        obs = start_episode()
        q_log, grip_log = [], []
        done_at = torch.full((n,), -1, dtype=torch.long, device=dev)
        bad = touched | moved | ~have_pre
        shield_bad=torch.zeros_like(bad);collision_bad=torch.zeros_like(bad)
        grasp_step=torch.full((n,),-1,dtype=torch.long,device=dev)
        grasp_count=torch.zeros_like(grasp_step);fingers_prev=pe._state(env)['fingers'].clone()
        for t in range(int((args.episode_s or cfg["task"]["episode_s"][2]) / env.step_dt)):
            q_log.append(torch.rad2deg(robot.data.joint_pos[:, arm._joint_ids]).cpu().numpy())
            a = act(obs)
            grip_log.append((a[:, 6] < 0).cpu().numpy())
            obs, _, _, _ = venv.step(a)
            st = pe._state(env)
            settled_grasp=st['held']&((st['fingers']-fingers_prev).abs()<1e-3)&(st['lift']<.01)
            grasp_count=torch.where(settled_grasp,grasp_count+1,torch.zeros_like(grasp_count))
            grasp_step=torch.where((grasp_step<0)&(grasp_count>=5),torch.full_like(grasp_step,t+1),grasp_step)
            fingers_prev=st['fingers'].clone()
            live=done_at<0
            shield_bad |= arm.blocked & live
            collision_bad |= env.termination_manager.get_term("collision") & live
            bad |= (env.termination_manager.get_term("collision") | arm.blocked) & live
            held_ok = st["hold"] >= int(args.export_hold_s * 30)
            done_at = torch.where((done_at < 0) & held_ok, torch.full_like(done_at, t + 1), done_at)
            if ((done_at >= 0) | bad).all():
                break
        q_log.append(torch.rad2deg(robot.data.joint_pos[:, arm._joint_ids]).cpu().numpy())
    ok = ((done_at >= 0) & ~bad).cpu().numpy()
    grip = np.stack(grip_log)                                   # (T, n)
    close_step = np.where(grip.any(0), grip.argmax(0), -1)
    command_close_step=close_step.copy()
    if args.confirmed_grasp_waypoint:
        close_step=grasp_step.cpu().numpy();ok &= close_step>=0
    from pathlib import Path
    out = Path(args.output) if args.output else rl_path_lib(cfg) / (_time.strftime("%Y%m%d_%H%M%S") + f"_{_time.time_ns() % 1000000:06d}")
    if (out / "library.npz").exists():
        raise RuntimeError("Refusing to overwrite an existing library")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "library.npz", radius_m=radii, q_deg=np.stack(q_log),            # (T+1, n, 6)
                        home_deg=torch.rad2deg(home[:, arm._joint_ids]).cpu().numpy()[0],
                        q_pre_deg=torch.rad2deg(q_pre[:, arm._joint_ids]).cpu().numpy(),
                        close_step=close_step, success_step=done_at.cpu().numpy(), ok=ok,
                        close_command_step=command_close_step,grasp_step=grasp_step.cpu().numpy(),
                        gripper_close_command=grip,
                        touched_plate=(touched | moved).cpu().numpy(), plate_dz_m=float(args.plate_dz),
                        plate_root_z_m=float(pe.REST_Z),
                        shield_blocked=shield_bad.cpu().numpy(),arm_collision=collision_bad.cpu().numpy(),
                        final_lift_m=st['lift'].cpu().numpy(),final_held=st['held'].cpu().numpy(),
                        phi_deg=bank['phi_deg'] if bank is not None else np.zeros(n),
                        pitch_deg=bank['pitch_deg'] if bank is not None else np.zeros(n))
    meta = dict(created=_time.strftime("%Y-%m-%d %H:%M:%S"), checkpoint=str(args.checkpoint), approach=args.approach,
                handoff_m=args.handoff_m, pre_m=args.pre_m, move_s=move_s, export_hold_s=args.export_hold_s,
                speed_cap_mps=pe.A['max_tcp_speed'],hold_windup_m=pe.A['hold_windup'],
                arm_joint_speed_cap_deg_s=args.arm_joint_speed_deg_s,
                episode_deadline_s=args.episode_s or cfg['task']['episode_s'][2],
                gripper_latched=args.latch_gripper,grasp_wait=args.grasp_wait,
                confirmed_grasp_waypoint=args.confirmed_grasp_waypoint,
                plate_dz_m=float(args.plate_dz), plate_root_z_m=float(pe.REST_Z), joint_sweep_speed_cap_deg_s=6.0,
                grasp_bank=args.grasp_bank,
                plate_xy_frame="base, plate on +x at radius r", radii=[float(r) for r in radii],
                ok=[bool(v) for v in ok], n_ok=int(ok.sum()), n=n,
                shield_blocked=int(shield_bad.sum()),arm_collision=int(collision_bad.sum()),
                sweep_touched_plate=int((touched | moved).sum()), no_pre=int((~have_pre).sum()))
    (out / "library.json").write_text(json.dumps(meta, indent=1))
    for r, o, c, d in zip(radii, ok, close_step, done_at.cpu().numpy()):
        if not o:
            print(f"[library] r {r:.3f} m: FAILED (close {c}, done {d})", flush=True)
    print(f"[library] {int(ok.sum())}/{n} radii ok ({radii[0]:.3f}..{radii[-1]:.3f} m, step {args.r_step * 1000:.0f} mm)",
          flush=True)
    print(f"LIBRARY {out / 'library.npz'}", flush=True)
    env.close()


def rl_path_lib(cfg):
    from rl_utils import rl_path
    return rl_path(cfg["paths"]["calibration_dir"]) / "data" / "plan_library"


# ---------------------------------------------------------------- serve: warm export server (pipeline.py latency)

def run_serve(args, cfg):
    """Isaac + policy stay loaded; each request is one export_once.  Request (one JSON line on TCP 127.0.0.1:--port):
    {"plate_x", "plate_y", "plate_yaw_deg", "checkpoint", "approach", "residual_base"}.  The server only answers requests
    for the policy it was started with (else {"ok": false, "mismatch": ...} and pipeline.py falls back to a fresh export).
    Reply: {"ok", "plan", "accepted", "lines"}."""
    import socket
    from rl_utils import load_policy
    env = build_env(args, terminations=False)
    venv, _, policy = load_policy(env, cfg, 2, args.checkpoint)
    policy = approach_wrapper(args, env, policy)
    me = dict(checkpoint=str(args.checkpoint), approach=args.approach, residual_base=args.residual_base or None)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(1)
    print(f"SERVE_READY 127.0.0.1:{args.port} {json.dumps(me)}", flush=True)
    try:
        while True:
            conn, _ = srv.accept()
            with conn:
                try:
                    req = json.loads(conn.makefile().readline())
                    want = {k: req.get(k) for k in me}
                    if want != me:
                        reply = dict(ok=False, mismatch=dict(server=me, request=want))
                    else:
                        args.plate_x, args.plate_y = float(req["plate_x"]), float(req["plate_y"])
                        args.plate_yaw_deg = float(req.get("plate_yaw_deg", 0.0))
                        args.rotate = bool(req.get("rotate", False))
                        lines = []
                        path, accepted = export_once(args, cfg, env, venv, policy, say=lines.append)
                        reply = dict(ok=path is not None, plan=str(path) if path else None, accepted=bool(accepted),
                                     lines=lines)
                except Exception as e:                     # one bad request must not kill the warm server
                    import traceback
                    traceback.print_exc()
                    reply = dict(ok=False, error=repr(e))
                conn.sendall((json.dumps(reply) + "\n").encode())
    finally:
        srv.close()
        env.close()


# ---------------------------------------------------------------- trace

def run_trace(args, cfg):
    import torch
    import plate_env as pe
    from rl_utils import load_policy
    env = build_env(args, terminations=False)
    venv, _, policy = load_policy(env, cfg, args.stage, args.checkpoint)
    arm, rm = env.action_manager.get_term("arm_action"), env.reward_manager
    obs = venv.get_observations()
    tcp0, blocked = None, torch.zeros(env.num_envs, device=env.device)
    max_lift = torch.full((env.num_envs,), -1.0, device=env.device)
    with torch.inference_mode():
        for t in range(1, int(cfg["task"]["episode_s"][args.stage] * 30) + 1):    # env limit is 3600 s here
            act = policy(obs).clone()
            if 60 < t <= 90:
                act[:, 2] += args.bias_z
            obs, _, _, _ = venv.step(act)
            st = pe._state(env)
            blocked += arm.blocked.float()
            max_lift = torch.maximum(max_lift, st["lift"])
            tcp0 = st["tcp"].clone() if tcp0 is None else tcp0
            if t in (25, 55, 95, 180, 300, 420):
                vals = rm._step_reward.mean(0).tolist()
                print(f"  REW t={t / 30:4.1f}s " + " ".join(f"{n}={v:.2f}" for n, v in zip(rm.active_terms, vals)
                                                             if abs(v) > 0.005), flush=True)
            if t % 30 == 0:
                med = lambda x: float(x.median())                                            # noqa: E731
                print(f"t={t / 30:4.1f}s held {st['held'].float().mean():.2f} | TCP rise mm {med(st['tcp'][:, 2] - tcp0[:, 2]) * 1000:5.0f}"
                      f" | plate lift mm med {med(st['lift']) * 1000:5.1f} max {float(st['lift'].max()) * 1000:5.1f}"
                      f" | tilt {med(tilt_deg(st)):4.1f} | fingers {med(st['fingers']):.2f} | d_rim mm {med(st['d_rim']) * 1000:4.0f}"
                      f" d_circle mm {med(st['d_circle']) * 1000:3.0f} | act z {float(act[:, 2].mean()):+.2f} grip {float(act[:, 6].mean()):+.2f}"
                      f" | locked {arm.orientation_locked.float().mean():.2f} blocked {blocked.mean():.0f}", flush=True)
    print("max plate lift per env mm", [round(v) for v in (max_lift * 1000).tolist()], flush=True)
    env.close()


# ---------------------------------------------------------------- scripted grasp

def run_scripted(args, cfg):
    import torch
    import plate_env as pe
    env = build_env(args, terminations=False)
    env.reset()
    dev, arm = env.device, env.action_manager.get_term("arm_action")
    if args.no_shield:
        arm.shield.safe = lambda q: torch.ones(q.shape[0], dtype=torch.bool, device=dev)
    hold_run = torch.zeros(env.num_envs, device=dev)
    hold_best = torch.zeros(env.num_envs, device=dev)
    contact = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)

    def step(err=None, close=False):
        a = torch.zeros(env.num_envs, 7, device=dev)
        if err is not None:
            a[:, :3] = (err / cfg["action"]["scale"]).clamp(-1.0, 1.0)              # the env caps it at max_tcp_speed
        a[:, 6] = -1.0 if close else 1.0
        env.step(a)
        st = pe._state(env)
        hold_run.copy_(torch.where(st["stable_lift"], hold_run + 1, torch.zeros_like(hold_run)))
        hold_best.copy_(torch.maximum(hold_best, hold_run))
        contact.__ior__(env.scene["arm_contacts"].data.net_forces_w.norm(dim=-1).amax(1) > 1.0)
        return st

    def report(tag, z0):
        st = pe._state(env)
        print(f"[{tag}] d_rim mm {float(st['d_rim'].median()) * 1000:.1f} | at_grasp {st['at_grasp'].float().mean():.2f} "
              f"held {st['held'].float().mean():.2f} | fingers {float(st['fingers'].median()):.2f} | "
              f"TCP rise mm {float((st['tcp'][:, 2] - z0).median()) * 1000:.1f} | plate lift mm {float(st['lift'].median()) * 1000:.1f} | "
              f"tilt {float(tilt_deg(st).median()):.1f} | plate speed mm/s {[round(v) for v in (st['plate_speed'] * 1000).tolist()]} "
              f"| tilt per env {[round(v) for v in tilt_deg(st).tolist()]} | blocked {arm.blocked.float().mean():.2f} | contact ever {contact.float().mean():.2f}",
              flush=True)

    with torch.inference_mode():
        z0 = pe._state(env)["tcp"][:, 2].clone()
        for _ in range(30):
            step()
        for _ in range(120):                                                            # advance to the rim target
            st = pe._state(env)
            step(st["rim"] - st["tcp"])
        report("at rim", z0)
        target = pe._state(env)["tcp"].clone()
        for _ in range(60):
            step(target - pe._state(env)["tcp"], close=True)
        report("closed", z0)
        target[:, 2] += args.lift_m
        for _ in range(int((3 + args.hold_s) * 30)):                                    # raise, then actively hold
            step(target - pe._state(env)["tcp"], close=True)
        report(f"raised {args.lift_m * 100:.0f} cm + held {args.hold_s:g} s", z0)
        for k in range(1, int(args.zero_s * 30) + 1):                                   # zero arm action
            step(close=True)
            if k % 30 == 0:
                report(f"zero action {k / 30:.0f} s", z0)
    print(f"SCRIPT_RESULT success={(hold_best >= cfg['task']['hold_steps']).int().tolist()} "
          f"best_hold_steps={hold_best.int().tolist()} contact={contact.int().tolist()}", flush=True)
    env.close()


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)             # app.close() drops buffered output
    args = parse_args()
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    cfg = load_config()
    if args.residual != "config":
        cfg["residual"]["enabled"] = args.residual == "on"
    if args.residual_base:
        cfg["residual"].update(enabled=True, base=args.residual_base)
    try:
        if args.mode == "gate":
            run_gate(args, cfg)
        elif args.mode == "watch":
            run_watch(args, cfg, app)
        elif args.mode == "demo":
            run_demo(args, cfg, app)
        elif args.mode == "export":
            if args.plate_x is None or args.plate_y is None:
                raise SystemExit("export needs --plate_x --plate_y (from camera_tools.py locate)")
            run_export(args, cfg)
        elif args.mode == "serve":
            run_serve(args, cfg)
        elif args.mode == "library":
            run_library(args, cfg)
        elif args.mode == "trace":
            run_trace(args, cfg)
        else:
            run_scripted(args, cfg)
    except Exception:
        import traceback
        traceback.print_exc()                               # app.close() would swallow it
        raise
    finally:
        app.close()
