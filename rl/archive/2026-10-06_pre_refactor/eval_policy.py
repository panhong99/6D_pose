"""Deterministic evaluation of a pick-and-place checkpoint: run every env for exactly one episode, report the success rate
and how the other episodes ended.  Stage from PP_STAGE.  Prints a final line `EVAL_JSON {...}` for rl/curriculum.py.

  PP_STAGE=1 ./isaaclab.sh -p rl/eval_policy.py --headless --checkpoint <model.pt> [--num_envs 1024]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--num_envs", type=int, default=1024)
p.add_argument("--trace", action="store_true", help="print first-env approach/action/shield measurements once per second")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import pick_place_env as pp
from pick_place_agent import PickPlacePPORunnerCfg

cfg = pp.PickPlaceEnvCfg()
cfg.scene.num_envs = args.num_envs
if args.trace and args.num_envs <= 16:
    # Small diagnostic runs do not need the contact buffers sized for 4096 environments.
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 65536
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 65536
env = ManagerBasedRLEnv(cfg)
venv = RslRlVecEnvWrapper(env)
runner = OnPolicyRunner(venv, PickPlacePPORunnerCfg().to_dict(), log_dir=None, device=env.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.device)

n = env.num_envs
outcome = torch.full((n,), -1, dtype=torch.long, device=env.device)      # -1 running, else index in names
names = ["success", "collision", "plate_lost", "time_out"]
shield_steps = torch.zeros(n, device=env.device)
obs = venv.get_observations()
with torch.inference_mode():
    for step in range(int(cfg.episode_length_s * 30) + 5):
        act = policy(obs)
        obs, _, dones, _ = venv.step(act)
        if args.trace and step % 30 == 0:
            st = pp._state(env)
            arm = env.action_manager.get_term("arm_action")
            frame, table = arm.shield.clearance(env.scene["robot"].data.joint_pos[:, arm._joint_ids])
            print("APPROACH_TRACE " + json.dumps(dict(step=step, rim_mm=float(st['d_rim'][0]*1000),
                ready_mm=float(st['d_ready'][0]*1000), action=act[0].tolist(), blocked=bool(arm.blocked[0]),
                frame_mm=float(frame[0]*1000), table_mm=float(table[0]*1000),
                orientation_locked=bool(arm.orientation_locked[0]), at_grasp=bool(st['at_grasp'][0]))), flush=True)
        running = outcome < 0
        shield_steps += env.action_manager.get_term("arm_action").blocked.float() * running
        for k, name in enumerate(names):
            hit = env.termination_manager.get_term(name) & running & (outcome < 0)
            outcome[hit] = k
        if (outcome >= 0).all():
            break
res = {name: float((outcome == k).float().mean()) for k, name in enumerate(names)}
res.update(stage=pp.STAGE, episodes=n, unfinished=float((outcome < 0).float().mean()),
           mean_shield_steps=float(shield_steps.mean()), checkpoint=args.checkpoint)
print("EVAL_JSON " + json.dumps(res), flush=True)
env.close()
app.close()
