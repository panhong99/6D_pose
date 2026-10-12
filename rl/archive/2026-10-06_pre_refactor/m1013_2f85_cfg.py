"""Isaac Lab articulation config: Doosan M1013 + Robotiq 2F-85 (see make_m1013_2f85_urdf.py)."""
import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

URDF = Path(__file__).resolve().parent / 'assets' / 'm1013_2f85.urdf'
HOME_DEG = (-5.98, -14.75, 117.74, -3.92, 7.46, -85.07)     # working home pose (calibration/kinematics.py HOME_JOINTS_DEG)

M1013_2F85_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(URDF),
        fix_base=True,
        activate_contact_sensors=True,       # robot-vs-frame/table contact sensing in the RL env
        merge_fixed_joints=False,            # keep gripper_base / tcp as separate bodies
        make_instanceable=False,
        convert_mimic_joints_to_normal_joints=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type='force', target_type='position',
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=1.0),   # relative-IK targets compound gravity sag otherwise,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=4),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={f'joint_{i + 1}': math.radians(q) for i, q in enumerate(HOME_DEG)}
        | {'right_finger_joint': 0.0, 'left_finger_joint': 0.0},
    ),
    actuators={
        'arm_big': ImplicitActuatorCfg(joint_names_expr=['joint_[1-3]'], effort_limit_sim=346.0, velocity_limit_sim=2.0,
                                       stiffness=10000.0, damping=1000.0),
        'arm_small': ImplicitActuatorCfg(joint_names_expr=['joint_[4-6]'], effort_limit_sim=50.0, velocity_limit_sim=3.9,
                                         stiffness=2500.0, damping=250.0),
        'gripper': ImplicitActuatorCfg(joint_names_expr=['.*_finger_joint'], effort_limit_sim=235.0, velocity_limit_sim=0.15,
                                       stiffness=2000.0, damping=100.0),
    },
)
