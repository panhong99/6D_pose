"""Check that the frame/table contact sensor really fires: drive the arm hard into the table / frame."""
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
env = ManagerBasedRLEnv(cfg)
env.reset()
sensor = env.scene["arm_contacts"]
print('sensor bodies', sensor.body_names, 'filters', sensor.num_filters if hasattr(sensor, 'num_filters') else '?', flush=True)
import math
robot = env.scene["robot"]
tcp_i = robot.body_names.index("tcp")
configs = {"pregrasp (should be free)": (18.5, 41.1, 67.3, 0, 71.6, 18.5),
           "shoulder pitched into table": (18.5, 85.0, 90.0, 0, 70.0, 18.5),
           "TCP on the upright (0.85, 0.56, 0.35)": (31.4, 48.5, 32.8, 0, 98.6, 31.4),
           "arm swung to +y rail/upright": (75.0, 30.0, 60.0, 0, 70.0, 0.0),
           "arm swung to +x far upright": (0.0, 20.0, 40.0, 0, 90.0, 0.0)}
act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
for name, q in configs.items():
    env.reset()
    qq = torch.tensor([math.radians(v) for v in q] + [0.0, 0.0], device=env.device).expand(env.num_envs, 8).clone()
    peak = 0.0; netpeak = 0.0
    for _ in range(60):
        robot.write_joint_state_to_sim(qq, torch.zeros_like(qq))
        robot.set_joint_position_target(qq)
        env.sim.step(render=False); env.scene.update(env.physics_dt)
        netpeak = max(netpeak, float(torch.norm(sensor.data.net_forces_w, dim=-1).max()))
    print(f"{name}: tcp {robot.data.body_pos_w[0, tcp_i].cpu().numpy().round(2)} peak net force {netpeak:.1f} N, flag {pp.collision_force(env).tolist()}", flush=True)
env.close(); app.close()
