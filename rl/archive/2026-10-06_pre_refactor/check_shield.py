"""Check the voxel shield: torch FK vs Isaac link_6 pose, home clearance vs calibration CollisionChecker, and that a
policy pushing the TCP into the frame / table is stopped before contact."""
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
robot = env.scene["robot"]; term = env.action_manager.get_term("arm_action"); sh = term.shield
q = robot.data.joint_pos[:, :6]
T6 = sh.link_frames(q)[5]
i6 = robot.body_names.index("link_6")
p6 = robot.data.body_pos_w[:, i6] - env.scene.env_origins
print("FK link_6 err (mm):", (torch.norm(T6[:, :3, 3] - p6, dim=1) * 1000).tolist(), flush=True)
f, t = sh.clearance(q)
print("home clearance frame/table (m):", f.tolist(), t.tolist(), flush=True)
# drive: +x toward the far uprights at full speed, then straight down into the table
for name, a in (("toward frame (+x,+y)", (1.0, 1.0, 0.0)), ("down into table (-z)", (0.0, 0.0, -1.0))):
    env.reset()
    act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    act[:, 0], act[:, 1], act[:, 2] = a
    act[:, 6] = -1.0
    blocked = 0; contact = 0
    for k in range(200):
        if name.startswith("down") and k < 12:          # first slide to +y, away from the plate, then go down
            act[:, 1], act[:, 2] = 1.0, 0.0
        elif name.startswith("down"):
            act[:, 1], act[:, 2] = 0.0, -1.0
        env.step(act)
        blocked += int(term.blocked.sum()); contact += int((pp.collision_force(env) > 0).sum())
        fz = torch.norm(env.scene["arm_contacts"].data.net_forces_w, dim=-1)
        if (fz > 1).any() and contact < 4:
            e, b = [int(v) for v in (fz > 1).nonzero()[0]]
            print(f"   contact step {k}: env {e} body {env.scene['arm_contacts'].body_names[b]} |F| {float(fz[e, b]):.1f} "
                  f"tcp {(pp._tcp(env)[0] - env.scene.env_origins)[e].cpu().numpy().round(3)} plate {(env.scene['plate'].data.root_pos_w - env.scene.env_origins)[e].cpu().numpy().round(3)}", flush=True)
    f, t = sh.clearance(robot.data.joint_pos[:, :6])
    print(f"{name}: shield blocks {blocked}, arm contacts {contact}, final clearance frame {f.min():.3f} table {t.min():.3f}, "
          f"tcp {(pp._tcp(env)[0] - env.scene.env_origins)[0].cpu().numpy().round(3)}", flush=True)
env.close(); app.close()
