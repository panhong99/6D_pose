"""Print the robot joints right after reset (no action yet) and after a few zero-action steps."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
app = AppLauncher(p.parse_args()).app
import torch
from isaaclab.envs import ManagerBasedRLEnv
import pick_place_env as pp
cfg = pp.PickPlaceEnvCfg_PLAY(); cfg.scene.num_envs = 4
env = ManagerBasedRLEnv(cfg); env.reset()
r = env.scene["robot"]
print("joint names", r.joint_names, flush=True)
print("default deg", (r.data.default_joint_pos[0, :6] * 57.2958).cpu().numpy().round(2).tolist(), flush=True)
for e in range(4):
    print("after reset env", e, (r.data.joint_pos[e, :6] * 57.2958).cpu().numpy().round(2).tolist(), flush=True)
act = torch.zeros(4, env.action_manager.total_action_dim, device=env.device); act[:, 6] = -1
for _ in range(30):
    env.step(act)
print("after 30 zero steps env0", (r.data.joint_pos[0, :6] * 57.2958).cpu().numpy().round(2).tolist(), flush=True)
env.close(); app.close()
