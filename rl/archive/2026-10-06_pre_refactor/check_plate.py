"""Zero-action rollout: does the scanned plate rest stably on its box, and is the pre-grasp pose collision-free?"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
app = AppLauncher(p.parse_args()).app
import torch
from isaaclab.envs import ManagerBasedRLEnv
import pick_place_env as pp

cfg = pp.PickPlaceEnvCfg_PLAY(); cfg.scene.num_envs = 8
env = ManagerBasedRLEnv(cfg); env.reset()
act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
plate = env.scene["plate"]
print("rest z target", pp.REST_Z, flush=True)
for t in range(90):
    env.step(act)
    if t in (0, 10, 30, 89):
        z = plate.data.root_pos_w[:, 2]
        print(f"t={t}: plate z mean {z.mean():.4f} min {z.min():.4f} max {z.max():.4f}  speed max {torch.norm(plate.data.root_lin_vel_w, dim=1).max():.4f}  "
              f"per-body |F| env0 {[round(float(v),1) for v in torch.norm(env.scene['arm_contacts'].data.net_forces_w[0], dim=-1)]}  "
              f"arm collision flags {pp.collision_force(env).sum().item():.0f}/8  tcp->rim {pp._state(env)['d_rim'].mean():.4f} m", flush=True)
env.close(); app.close()
