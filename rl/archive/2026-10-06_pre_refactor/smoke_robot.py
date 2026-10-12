"""Headless smoke test: spawn M1013+2F-85 in Isaac Lab, close the gripper, print joint/TCP state."""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from m1013_2f85_cfg import M1013_2F85_CFG

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120))
sim_utils.GroundPlaneCfg().func('/World/ground', sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2000.0).func('/World/light', sim_utils.DomeLightCfg(intensity=2000.0))
robot = Articulation(M1013_2F85_CFG.replace(prim_path='/World/Robot'))
sim.reset()
robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
robot.reset()
print('joints', robot.joint_names, flush=True)
print('bodies', robot.body_names, flush=True)
tcp = robot.body_names.index('tcp')
fi = [robot.joint_names.index(n) for n in ('right_finger_joint', 'left_finger_joint')]
target = robot.data.default_joint_pos.clone()
for step in range(360):
    if step == 120:
        target[:, fi] = 0.0425
        print('closing', flush=True)
    robot.set_joint_position_target(target)
    robot.write_data_to_sim()
    sim.step()
    robot.update(1 / 120)
    if step in (0, 119, 239, 359):
        print(step, 'tcp', robot.data.body_pos_w[0, tcp].cpu().numpy().round(3), 'fingers',
              robot.data.joint_pos[0, fi].cpu().numpy().round(4),
              'arm err deg', (robot.data.joint_pos[0, :6] - robot.data.default_joint_pos[0, :6]).abs().max().item() * 57.3, flush=True)
app.close()
