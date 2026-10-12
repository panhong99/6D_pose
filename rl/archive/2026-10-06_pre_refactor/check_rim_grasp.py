"""Scripted feasibility test of the stage-2 rim grasp (no learning): is the task physically possible in the sim?

The stage-1 policy drives each env to the ready pose; then a script takes over: advance the TCP straight to the rim target
(orientation held), close the gripper, lift 8 cm in +z and hold.  Prints per phase: TCP-rim distance, at_grasp, finger
joint closure, arm contact, plate displacement and lift.  If the scripted grasp cannot lift the plate, PPO cannot either.

  PP_STAGE=2 ./isaaclab.sh -p rl/check_rim_grasp.py --headless --checkpoint <stage1 model.pt> [--close 0.0425]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--num_envs", type=int, default=8)
p.add_argument("--close", type=float, default=None, help="finger close command (m per jaw); default env FINGER_CLOSED")
p.add_argument("--rim_dz", type=float, default=None, help="override grasp height above the plate bottom (m)")
p.add_argument("--rim_in", type=float, default=0.0, help="move the grasp target this far toward the plate centre (m)")
p.add_argument("--no_shield", action="store_true", help="accept every IK target (physics-only feasibility)")
p.add_argument("--lift_m", type=float, default=0.12, help="TCP upward displacement; verify plate lift separately")
p.add_argument("--keep_open", action="store_true", help="keep the GUI open at the final pose")
p.add_argument("--hold_s", type=float, default=5.0, help="duration of the scripted hold stability check")
p.add_argument("--lock_orientation", action="store_true", help="actively maintain the ready TCP orientation during scripted phases")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import torch
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul, axis_angle_from_quat
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import pick_place_env as pp
from pick_place_agent import PickPlacePPORunnerCfg

cfg = pp.PickPlaceEnvCfg()
cfg.seed = 123
cfg.scene.num_envs = args.num_envs
cfg.episode_length_s = 3600.0 if args.keep_open else 60.0
cfg.terminations.success = None
cfg.terminations.plate_lost = None
if args.close is not None:
    cfg.actions.gripper_action.close_command_expr = {".*_finger_joint": args.close}
env = ManagerBasedRLEnv(cfg)
venv = RslRlVecEnvWrapper(env)
runner = OnPolicyRunner(venv, PickPlacePPORunnerCfg().to_dict(), log_dir=None, device=env.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.device)
robot = env.scene["robot"]
fj = [robot.joint_names.index(n) for n in ("left_finger_joint", "right_finger_joint")]
dev = env.device
rim_off = torch.tensor(pp.RIM_OFFSET, device=dev)
if args.rim_dz is not None:
    rim_off[2] = args.rim_dz
if args.rim_in:
    rim_off[:2] -= args.rim_in * rim_off[:2] / torch.norm(rim_off[:2])
contact_any = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
arm = env.action_manager.get_term("arm_action")
print(f"SCRIPT_CONFIG version={pp.VERSION} stage={pp.STAGE} box_h={pp.BOX_H} rim_offset={pp.RIM_OFFSET}", flush=True)
if args.no_shield:
    arm.shield.safe = lambda q: torch.ones(q.shape[0], dtype=torch.bool, device=dev)
blocked_steps = torch.zeros(env.num_envs, device=dev)
hold_run = torch.zeros(env.num_envs, device=dev)
hold_best = torch.zeros(env.num_envs, device=dev)
script_orientation = None


def report(tag):
    st = pp._state(env)
    tcp = st["tcp"]
    pl = env.scene["plate"].data.root_pos_w
    rest = pp._const(env, (pp.PLATE_XY[0], pp.PLATE_XY[1], pp.REST_Z))
    d_rim = torch.norm(tcp - (pl + rim_off), dim=1)
    q = robot.data.joint_pos[:, fj]
    up = torch.zeros_like(pl)
    up[:, 2] = 1.0
    normal = quat_apply(env.scene['plate'].data.root_quat_w, up)
    tilt = torch.rad2deg(torch.acos(normal[:, 2].clamp(-1, 1)))
    local_tcp = quat_apply(quat_conjugate(env.scene['plate'].data.root_quat_w), tcp - pl)
    print(f"    plate tilt deg {[round(v, 1) for v in tilt.tolist()]} "
          f"local TCP mm {[[round(v, 1) for v in row] for row in (local_tcp * 1000).tolist()]}", flush=True)
    f = env.scene["arm_contacts"].data.net_forces_w.norm(dim=-1).amax(1)
    print(f"[{tag}] d_rim mm {[round(v, 1) for v in (d_rim * 1000).tolist()]}\n"
          f"    tool/finger angle deg {[round(v, 1) for v in torch.rad2deg(torch.acos(st['tool_match'].clamp(-1, 1))).tolist()]}"
          f" / {[round(v, 1) for v in torch.rad2deg(torch.acos(st['finger_match'].clamp(-1, 1))).tolist()]} (need < 10 / 15)\n"
          f"    at_grasp {st['at_grasp'].int().tolist()}  held {st['held'].int().tolist()}  lifted {st['lifted'].int().tolist()}\n"
          f"    finger q mm {[round(v, 1) for v in (q.mean(1) * 1000).tolist()]}  (cmd {cfg.actions.gripper_action.close_command_expr})\n"
          f"    arm/gripper-base contact N {[round(v, 1) for v in f.tolist()]}  ever {contact_any.int().tolist()}\n"
          f"    shield-blocked steps {blocked_steps.int().tolist()}  table clearance mm "
          f"{[round(v, 1) for v in (arm.shield.clearance(robot.data.joint_pos[:, arm._joint_ids])[1] * 1000).tolist()]} (shield needs >= 20)\n"
          f"    plate xy shift mm {[round(v, 1) for v in (torch.norm((pl - rest)[:, :2], dim=1) * 1000).tolist()]}"
          f"  lift mm {[round(v, 1) for v in ((pl[:, 2] - rest[:, 2]) * 1000).tolist()]}", flush=True)


def step(arm_xyz_err=None, close=False, policy_obs=None):
    global obs
    if policy_obs is not None:
        act = policy(obs).clone()
    else:
        act = torch.zeros(env.num_envs, 7, device=dev)
        act[:, :3] = (arm_xyz_err / 0.03).clamp(-0.3, 0.3)          # <= 9 mm per policy step
        if args.lock_orientation and script_orientation is not None:
            error = quat_mul(script_orientation, quat_conjugate(pp._tcp(env)[1]))
            act[:, 3:6] = (axis_angle_from_quat(error) / 0.03).clamp(-0.3, 0.3)
    act[:, 6] = -1.0 if close else 1.0                                 # Isaac Lab binary gripper: action < 0 = close
    obs, _, _, _ = venv.step(act)
    hold_run.copy_(torch.where(pp._state(env)["stable_lift"], hold_run + 1, torch.zeros_like(hold_run)))
    hold_best.copy_(torch.maximum(hold_best, hold_run))
    blocked_steps.__iadd__(arm.blocked.float())
    contact_any.__ior__(env.scene["arm_contacts"].data.net_forces_w.norm(dim=-1).amax(1) > 1.0)


obs = venv.get_observations()
with torch.inference_mode():
    for _ in range(int(6 * 30)):                                       # stage-1 policy: home -> ready
        step(policy_obs=True)
    report("ready (stage-1 policy, 6 s)")
    script_orientation = pp._tcp(env)[1].clone()
    for _ in range(int(4 * 30)):                                       # advance to the rim target
        tcp = pp._state(env)["tcp"]
        step(env.scene["plate"].data.root_pos_w * 0 + (env.scene["plate"].data.root_pos_w + rim_off) - tcp)
    report("at rim (scripted advance, 4 s)")
    target = pp._state(env)["tcp"].clone()
    for _ in range(int(2 * 30)):                                       # close
        step(target - pp._state(env)["tcp"], close=True)
    report("closed (2 s)")
    target[:, 2] += args.lift_m
    for _ in range(int(3 * 30)):                                       # lift 8 cm
        step(target - pp._state(env)["tcp"], close=True)
    report(f"lifted TCP {args.lift_m * 100:.1f} cm (3 s)")
    for _ in range(int(args.hold_s * 30)):
        step(target - pp._state(env)["tcp"], close=True)
    report(f"held {args.hold_s:g} s")
    passed = (hold_best >= pp.HOLD_STEPS) & ~contact_any
    print(f"SCRIPT_RESULT success={passed.int().tolist()} max_lift_hold_steps={hold_best.int().tolist()}", flush=True)
    print(f"ENV_ORIENTATION_LOCKED {arm.orientation_locked.int().tolist()}", flush=True)
    if args.keep_open:
        extra_steps = 0
        while app.is_running():
            step(target - pp._state(env)["tcp"], close=True)
            extra_steps += 1
            if extra_steps % 300 == 0:
                report(f"GUI hold {args.hold_s + extra_steps / 30:g} s")
    elif pp.V6 and pp.STAGE == 2:
        env.reset()
        assert not arm.orientation_locked.any(), 'Orientation lock persisted across reset'
        print('ENV_ORIENTATION_RESET passed', flush=True)
env.close()
app.close()
