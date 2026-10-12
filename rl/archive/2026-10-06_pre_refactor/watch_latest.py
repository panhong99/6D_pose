"""GUI viewer that follows training: runs a few envs with the newest checkpoint of the current stage and reloads it
whenever training saves a newer one (every 100 iterations).  Training keeps running headless in parallel.

  PP_STAGE=1 ./isaaclab.sh -p rl/watch_latest.py --num_envs 4
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=4)
p.add_argument("--reload_s", type=float, default=60.0)
p.add_argument("--hold_start_s", type=float, default=2.0, help="freeze each episode at the home pose this long")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import pick_place_env as pp
from experiment_settings import EXPERIMENT_NAME
from pick_place_agent import PickPlacePPORunnerCfg

RUNS = Path.home() / "IsaacLab" / "logs" / "rsl_rl" / EXPERIMENT_NAME


def newest():
    models = [m for d in RUNS.glob(f"*_stage{pp.STAGE}") for m in d.glob("model_*.pt")]
    return max(models, key=lambda m: m.stat().st_mtime) if models else None


cfg = pp.PickPlaceEnvCfg_PLAY()
cfg.scene.num_envs = args.num_envs
cfg.scene.env_spacing = 2.5
env = ManagerBasedRLEnv(cfg)
venv = RslRlVecEnvWrapper(env)
runner = OnPolicyRunner(venv, PickPlacePPORunnerCfg().to_dict(), log_dir=None, device=env.device)
loaded, t_check = None, 0.0
obs = venv.get_observations()
while app.is_running():
    if time.time() - t_check > args.reload_s or loaded is None:
        t_check = time.time()
        ck = newest()
        if ck is not None and ck != loaded:
            runner.load(str(ck))
            policy = runner.get_inference_policy(device=env.device)
            loaded = ck
            print(f"[watch] stage {pp.STAGE}: loaded {ck.parent.name}/{ck.name}", flush=True)
    if loaded is None:
        app.update()
        time.sleep(1.0)
        continue
    with torch.inference_mode():
        act = policy(obs)
        frozen = env.episode_length_buf < int(args.hold_start_s * 30)       # show the exact start pose first
        act[frozen] = 0.0
        act[frozen, 6] = 1.0                                                 # gripper open (Isaac Lab binary: < 0 = close)
        obs, _, dones, _ = venv.step(act)
    for name in ("success", "collision", "plate_lost"):
        hit = env.termination_manager.get_term(name)
        if hit.any():
            print(f"[watch] env {hit.nonzero().flatten().tolist()} ended: {name}", flush=True)
env.close()
app.close()
