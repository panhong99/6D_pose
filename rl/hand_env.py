"""Isaac Lab task: Doosan M1013 + Tesollo DG-5F-S left grasps the flat plate on its box and lifts it (보스 10/26:
grasp success only, the arm path to the plate exists).

Not run directly: train.py hand / env_setup.py build it with `make_env_cfg(num_envs)` after the Isaac app started.
Every number comes from rl/hand.yaml.

Each episode starts at the real "closer" pose (fingertips ~8 cm above the near rim, hand open, back of the hand up).
  action : TCP (hand base) translation, orientation held (diff-IK, slow), + 20 finger targets (limits-mapped, EMA)
  success: plate root >= 5 cm above rest, tilt < 30 deg, for 2 s
Observations are simulator truth (no camera).

Sections: config -> robot / scene -> task state (_state, cached per step) -> observations -> rewards
-> terminations / events -> make_env_cfg.
"""
import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.manager_based.manipulation.lift import mdp

from rl_utils import load_config, rl_path

# ---------------------------------------------------------------- config (read once per process)
H = load_config(rl_path("hand.yaml"))
PLATE_USD = load_config()["paths"]["plate_usd"]
G, T, A = H["geometry"], H["task"], H["action"]
REST_Z = G["table_top_z"] + G["box_h"]                         # plate root (bottom centre) at rest
FINGERS = "joint_[1-5]_[1-4]"
TIPS = ["link_1_tip", "link_2_tip", "link_3_tip"]               # thumb, index, middle


# ---------------------------------------------------------------- robot / scene

def robot_cfg():
    r = H["robot"]
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(rl_path(H["paths"]["robot_urdf"])), fix_base=True, activate_contact_sensors=True,
            merge_fixed_joints=False, make_instanceable=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type="force", target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={f"joint_{i + 1}": math.radians(q) for i, q in enumerate(r["start_deg"])} | {FINGERS: 0.0}),
        actuators={"arm": ImplicitActuatorCfg(joint_names_expr=["joint_[1-6]"], stiffness=r["arm"]["stiffness"],
                                              damping=r["arm"]["damping"], effort_limit_sim=r["arm"]["effort"]),
                   "hand": ImplicitActuatorCfg(joint_names_expr=[FINGERS], stiffness=r["hand"]["stiffness"],
                                               damping=r["hand"]["damping"], effort_limit_sim=r["hand"]["effort"])})


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = robot_cfg()
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8, 0.0, G["table_top_z"] - 0.05)),
        spawn=sim_utils.CuboidCfg(size=(1.4, 1.2, 0.1), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.27, 0.3))))
    box = RigidObjectCfg(                                       # kinematic: follows the plate shift at reset
        prim_path="{ENV_REGEX_NS}/Box",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(G["plate_xy"][0], G["plate_xy"][1], G["table_top_z"] + G["box_h"] / 2)),
        spawn=sim_utils.CuboidCfg(size=(*G["box_size"], G["box_h"]), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                                  mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9))))
    plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(G["plate_xy"][0], G["plate_xy"][1], REST_Z)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(rl_path(PLATE_USD)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=2,
                                                         max_depenetration_velocity=2.0)))
    arm_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/link_[1-6]", history_length=1)
    plane = AssetBaseCfg(prim_path="/World/GroundPlane", init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.0)),
                         spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))


# ---------------------------------------------------------------- task state (computed once per env step)

def _state(env):
    if getattr(env, "_hand_state_step", None) == env.common_step_counter and hasattr(env, "_hand_state"):
        return env._hand_state
    robot, plate = env.scene["robot"], env.scene["plate"]
    o = env.scene.env_origins
    p = plate.data.root_pos_w - o
    up = quat_apply(plate.data.root_quat_w, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3))
    ids = [robot.body_names.index(n) for n in TIPS]
    tips = robot.data.body_pos_w[:, ids] - o[:, None]                         # (N, 3, 3) thumb, index, middle
    tcp = robot.data.body_pos_w[:, robot.body_names.index("link_base")] - o
    # rim point nearest the robot base (the near rim, where the hand starts), at rim height
    d = -p[:, :2] / torch.norm(p[:, :2], dim=1, keepdim=True)
    rim = torch.cat([p[:, :2] + G["rim_r"] * d, p[:, 2:3] + G["rim_dz"]], dim=1)
    dt_rim = torch.norm(tips - rim[:, None], dim=-1)                          # (N, 3)
    above = tips[..., 2] - rim[:, 2:3]
    opposite = (above[:, 0] * torch.maximum(above[:, 1], above[:, 2])) < 0.0  # thumb on one side, index/middle other
    near = (dt_rim[:, 0] < 0.04) & (torch.minimum(dt_rim[:, 1], dt_rim[:, 2]) < 0.04)
    lift = p[:, 2] - REST_Z
    tilt_ok = up[:, 2] > math.cos(math.radians(T["max_tilt_deg"]))
    held = near & opposite
    lifted = (lift > T["lift_ok"]) & tilt_ok & held
    if not hasattr(env, "_hold_count"):
        env._hold_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._plate_start = p.clone()
    env._hold_count = torch.where(lifted, env._hold_count + 1, torch.zeros_like(env._hold_count))
    moved = torch.norm(p[:, :2] - env._plate_start[:, :2], dim=1)
    st = dict(p=p, up=up, tips=tips, tcp=tcp, rim=rim, dt_rim=dt_rim, opposite=opposite, near=near, held=held,
              lift=lift, lifted=lifted, success=env._hold_count >= T["hold_steps"],
              knocked=(moved > T["knocked_m"]) & (lift < 0.02))
    env._hand_state, env._hand_state_step = st, env.common_step_counter
    return st


def reset_task_state(env, env_ids):
    if hasattr(env, "_hold_count"):
        env._hold_count[env_ids] = 0
        env._plate_start[env_ids] = env.scene["plate"].data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
    env._hand_state_step = None


def box_under_plate(env, env_ids):
    box, plate = env.scene["box"], env.scene["plate"]
    s = box.data.root_state_w[env_ids].clone()
    s[:, :2] = plate.data.root_pos_w[env_ids, :2]
    box.write_root_state_to_sim(s, env_ids)


# ---------------------------------------------------------------- observations

def obs_hand(env):
    r = env.scene["robot"]
    ids = r.find_joints(FINGERS)[0]
    return torch.cat([r.data.joint_pos[:, ids], 0.1 * r.data.joint_vel[:, ids]], dim=1)


def obs_geometry(env):
    st = _state(env)
    rel = (st["tips"] - st["rim"][:, None]).reshape(env.num_envs, -1)
    return torch.cat([st["tcp"] - st["rim"], rel, st["p"] - st["tcp"], st["up"], st["lift"][:, None]], dim=1)


def obs_plate_vel(env):
    return env.scene["plate"].data.root_lin_vel_w


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        hand = ObsTerm(func=obs_hand)
        geometry = ObsTerm(func=obs_geometry)
        plate_vel = ObsTerm(func=obs_plate_vel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------- actions

@configclass
class ActionsCfg:
    arm = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot", joint_names=["joint_[1-6]"], body_name="link_base",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=(A["arm_step_m"],) * 3 + (0.0,) * 3)                 # translation only, orientation held
    fingers = mdp.EMAJointPositionToLimitsActionCfg(asset_name="robot", joint_names=[FINGERS], alpha=A["finger_alpha"])


# ---------------------------------------------------------------- rewards (per step; Isaac Lab multiplies by dt)

def r_reach(env):
    d = _state(env)["dt_rim"]
    return 1.0 - torch.tanh((d[:, 0] + torch.minimum(d[:, 1], d[:, 2])) / 0.1)


def r_pinch(env):
    st = _state(env)
    return (st["near"] & st["opposite"]).float()


def r_lift(env):
    st = _state(env)
    return st["held"].float() * torch.clamp(st["lift"] / 0.10, 0.0, 1.0)


def r_hold(env):
    return _state(env)["lifted"].float()


def r_success(env):
    return _state(env)["success"].float()


def r_knocked(env):
    return _state(env)["knocked"].float()


def r_collision(env):
    return (torch.norm(env.scene["arm_contacts"].data.net_forces_w, dim=-1).amax(dim=1) > 1.0).float()


REWARD_FUNCS = {"reach": r_reach, "pinch": r_pinch, "lift": r_lift, "hold": r_hold, "success": r_success,
                "knocked": r_knocked, "collision": r_collision, "action_rate": mdp.action_rate_l2}


@configclass
class RewardsCfg:
    pass                                                        # filled from hand.yaml by make_env_cfg


# ---------------------------------------------------------------- terminations / events

def t_success(env):
    return _state(env)["success"]


def t_collision(env):
    return r_collision(env) > 0.5


def t_plate_lost(env):              # fell off the box or left the hand far below the rest height
    return _state(env)["p"][:, 2] < REST_Z - 0.03


def t_tcp_out(env):                 # hand wandered away from its start box
    st = _state(env)
    return torch.norm(st["tcp"] - env._tcp_start, dim=1) > A["arm_box_m"] if hasattr(env, "_tcp_start") else torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device)


def remember_tcp_start(env, env_ids):
    r = env.scene["robot"]
    tcp = r.data.body_pos_w[:, r.body_names.index("link_base")] - env.scene.env_origins
    if not hasattr(env, "_tcp_start"):
        env._tcp_start = tcp.clone()
    env._tcp_start[env_ids] = tcp[env_ids]


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=t_success)
    collision = DoneTerm(func=t_collision)
    plate_lost = DoneTerm(func=t_plate_lost)
    tcp_out = DoneTerm(func=t_tcp_out)


def _friction(asset_cfg):
    lo, hi = G["friction"]
    return EventTerm(func=mdp.randomize_rigid_body_material, mode="startup",
                     params={"asset_cfg": asset_cfg, "static_friction_range": (lo, hi),
                             "dynamic_friction_range": (0.9 * lo, 0.9 * hi), "restitution_range": (0.0, 0.0),
                             "num_buckets": 64})


@configclass
class EventCfg:                     # order matters: scene default -> plate shift -> box -> task state
    plate_friction = _friction(SceneEntityCfg("plate"))
    finger_friction = _friction(SceneEntityCfg("robot", body_names="link_[1-5]_.*"))
    plate_mass = EventTerm(func=mdp.randomize_rigid_body_mass, mode="startup",
                           params={"asset_cfg": SceneEntityCfg("plate"), "mass_distribution_params": tuple(G["plate_mass"]),
                                   "operation": "abs"})
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_plate = EventTerm(func=mdp.reset_root_state_uniform, mode="reset", params={
        "pose_range": {"x": (-G["plate_reset_xy"], G["plate_reset_xy"]), "y": (-G["plate_reset_xy"], G["plate_reset_xy"]),
                       "yaw": (-math.pi, math.pi)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("plate")})
    box_follow = EventTerm(func=box_under_plate, mode="reset")
    reset_state = EventTerm(func=reset_task_state, mode="reset")
    tcp_start = EventTerm(func=remember_tcp_start, mode="reset")


# ---------------------------------------------------------------- env config

@configclass
class HandEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=1024, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = T["decimation"]
        self.sim.dt = T["sim_dt"]
        self.sim.render_interval = self.decimation
        self.episode_length_s = T["episode_s"]
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1 << 24
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1 << 20


def make_env_cfg(num_envs, viewer=False):
    cfg = HandEnvCfg()
    cfg.scene.num_envs = num_envs
    for name, w in H["rewards"].items():
        setattr(cfg.rewards, name, RewTerm(func=REWARD_FUNCS[name], weight=float(w)))
    if num_envs <= 64:
        cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1 << 18
        cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 1 << 18
    if viewer:
        cfg.scene.env_spacing = 2.5
        cfg.viewer.origin_type = "env"
        cfg.viewer.env_index = 0
        cfg.viewer.eye = (1.6, -0.9, 0.7)
        cfg.viewer.lookat = (0.95, -0.1, 0.1)
    return cfg
