"""Trace one env step by step while sliding +y then moving down, to find what link_6 touches."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
app = AppLauncher(p.parse_args()).app
import torch
from isaaclab.envs import ManagerBasedRLEnv
import pick_place_env as pp
cfg = pp.PickPlaceEnvCfg_PLAY(); cfg.scene.num_envs = 2
env = ManagerBasedRLEnv(cfg); env.reset()
r = env.scene["robot"]; term = env.action_manager.get_term("arm_action")
act = torch.zeros(2, env.action_manager.total_action_dim, device=env.device); act[:, 6] = -1
o = env.scene.env_origins
for k in range(80):
    act[:, 1], act[:, 2] = (1.0, 0.0) if k < 12 else (0.0, -1.0)
    _, _, term_, trunc, _ = env.step(act)
    f = torch.norm(env.scene["arm_contacts"].data.net_forces_w[0], dim=-1)
    zmin = float((r.data.body_pos_w[0, :, 2]).min())
    if k % 5 == 0 or f.max() > 1 or term_[0]:
        print(k, 'tcp', (pp._tcp(env)[0] - o)[0].cpu().numpy().round(3), 'q', (r.data.joint_pos[0, :6] * 57.3).cpu().numpy().round(0),
              'F', f.cpu().numpy().round(1), 'blocked', bool(term.blocked[0]), 'done', bool(term_[0]),
              {n: bool(env.termination_manager.get_term(n)[0]) for n in env.termination_manager.active_terms}, flush=True)
env.close(); app.close()
