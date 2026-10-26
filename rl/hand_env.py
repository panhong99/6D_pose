"""Isaac Lab task: Doosan M1013 + Tesollo DG-5F-S left grasps the flat plate on its box and lifts it (보스 10/26:
grasp success only, the arm path to the plate exists).

Not run directly: train.py hand / env_setup.py build it with `make_env_cfg(num_envs)` after the Isaac app started.
Every number comes from rl/hand.yaml.

Each episode starts at the real "closer" pose (fingertips ~8 cm above the near rim, hand open, back of the hand up).
  action : TCP (hand base) move, slow, inside a box and a wrist-turn cone (diff-IK) + 20 finger targets (limits-mapped, EMA)
  held   : >= 2 distal finger links touching the plate (contact force, near the plate body)
  success: held, plate root >= 5 cm above rest, tilt < 30 deg, for 2 s
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
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_conjugate, quat_error_magnitude, quat_mul
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
                enabled_self_collisions=False, solver_position_iteration_count=r["solver"][0],
                solver_velocity_iteration_count=r["solver"][1])),
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
    distal_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/link_[1-5]_(4|tip)", history_length=1)
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
    tcp_q = robot.data.body_quat_w[:, robot.body_names.index("link_base")]
    # rim point nearest the robot base (the near rim, where the hand starts), at rim height
    d = -p[:, :2] / torch.norm(p[:, :2], dim=1, keepdim=True)
    rim = torch.cat([p[:, :2] + G["rim_r"] * d, p[:, 2:3] + G["rim_dz"]], dim=1)
    dt_rim = torch.norm(tips - rim[:, None], dim=-1)                          # (N, 3)
    above = tips[..., 2] - rim[:, 2:3]
    opposite = (above[:, 0] * torch.maximum(above[:, 1], above[:, 2])) < 0.0  # thumb on one side, index/middle other
    near = (dt_rim[:, 0] < 0.04) & (torch.minimum(dt_rim[:, 1], dt_rim[:, 2]) < 0.04)
    lift = p[:, 2] - REST_Z
    tilt_ok = up[:, 2] > math.cos(math.radians(T["max_tilt_deg"]))
    # touching = a finger's distal link or tip cap in contact (net force) while near the plate body; count fingers
    sensor = env.scene["distal_contacts"]
    force = torch.norm(sensor.data.net_forces_w, dim=-1)                                 # (N, 10)
    spos = robot.data.body_pos_w[:, [robot.body_names.index(n) for n in sensor.body_names]] - o[:, None]
    on_plate = (torch.norm(spos[..., :2] - p[:, None, :2], dim=-1) < G["rim_r"] + 0.03) & ((spos[..., 2] - p[:, None, 2]).abs() < 0.05)
    hit = (force > 0.5) & on_plate
    n_touch = sum((hit[:, [i for i, n in enumerate(sensor.body_names) if n.startswith(f"link_{f}_")]].any(dim=1)).long()
                  for f in range(1, 6))
    held = n_touch >= 2
    lifted = (lift > T["lift_ok"]) & tilt_ok & held
    if not hasattr(env, "_hold_count"):
        env._hold_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._plate_start, env._tcp_start = p.clone(), tcp.clone()
        env._q_start = tcp_q.clone()
    # start positions are taken on the first step of an episode: at reset the body buffers still hold the old episode
    fresh = env.episode_length_buf <= 1
    env._plate_start[fresh], env._tcp_start[fresh], env._q_start[fresh] = p[fresh], tcp[fresh], tcp_q[fresh]
    env._hold_count = torch.where(lifted, env._hold_count + 1, torch.zeros_like(env._hold_count))
    moved = torch.norm(p[:, :2] - env._plate_start[:, :2], dim=1)
    # physics glitch: a finger joint 0.1 rad past its limit (pilot 4 jammed fingers through their limits and flicked the plate)
    fid = robot.find_joints(FINGERS)[0]
    qf, lim = robot.data.joint_pos[:, fid], robot.data.soft_joint_pos_limits[:, fid]
    glitch = ((qf < lim[..., 0] - 0.1) | (qf > lim[..., 1] + 0.1)).any(dim=1)
    st = dict(p=p, up=up, tips=tips, tcp=tcp, rim=rim, dt_rim=dt_rim, opposite=opposite, near=near, held=held, glitch=glitch,
              n_touch=n_touch, tcp_q=tcp_q, turn=quat_error_magnitude(tcp_q, env._q_start),
              lift=lift, lifted=lifted, success=env._hold_count >= T["hold_steps"],
              knocked=(moved > T["knocked_m"]) & (lift < 0.02),
              tcp_out=((tcp - env._tcp_start).abs() > A["arm_box_m"] + 0.03).any(dim=1))     # walls hold it; safety only
    env._hand_state, env._hand_state_step = st, env.common_step_counter
    return st


def reset_task_state(env, env_ids):
    if hasattr(env, "_hold_count"):
        env._hold_count[env_ids] = 0
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

class BoxedDiffIKAction(DifferentialInverseKinematicsAction):
    """Diff-IK TCP move that cannot push the TCP out of its start box (arm_box_m per axis; z also not below the plate
    rim - 2 cm): the outward part of the command is dropped at the wall.  Pilot 2 (10/26) walked the palm down through
    the weak fingers and out of the box when that only ended the episode.  Wrist turns stop at turn_max_deg from the
    start (pilot 4: the fixed start orientation could not get the fingers under the rim)."""

    def process_actions(self, actions):
        env = self._env
        if hasattr(env, "_tcp_start"):
            st = _state(env)
            rel = st["tcp"] - env._tcp_start
            a = actions.clone()
            box = A["arm_box_m"]
            a[:, :3] = torch.where((rel >= box) & (a[:, :3] > 0), torch.zeros_like(a[:, :3]), a[:, :3])
            a[:, :3] = torch.where((rel <= -box) & (a[:, :3] < 0), torch.zeros_like(a[:, :3]), a[:, :3])
            low = st["tcp"][:, 2] <= st["rim"][:, 2] - 0.02
            a[:, 2] = torch.where(low & (a[:, 2] < 0), torch.zeros_like(a[:, 2]), a[:, 2])
            turned = st["turn"] > math.radians(A["turn_max_deg"])            # past the cone: turn back toward the start
            back = axis_angle_from_quat(quat_mul(env._q_start, quat_conjugate(st["tcp_q"])))
            back = back / torch.norm(back, dim=1, keepdim=True).clamp(min=1e-6)
            a[:, 3:] = torch.where(turned[:, None], back, a[:, 3:])
            actions = a
        super().process_actions(actions)

    def apply_actions(self):        # every physics step: no arm joint moves faster than arm_joint_deg_s (pilot 4 flipped
        ee_pos, ee_quat = self._compute_frame_pose()            # the wrist ~90 deg in 0.4 s through the IK)
        q = self._asset.data.joint_pos[:, self._joint_ids]
        q_des = self._ik_controller.compute(ee_pos, ee_quat, self._compute_frame_jacobian(), q)
        step = math.radians(A["arm_joint_deg_s"]) * self._env.physics_dt
        self._asset.set_joint_position_target(q + (q_des - q).clamp(-step, step), self._joint_ids)


@configclass
class BoxedDiffIKActionCfg(mdp.DifferentialInverseKinematicsActionCfg):
    class_type: type = BoxedDiffIKAction


@configclass
class ActionsCfg:
    arm = BoxedDiffIKActionCfg(
        asset_name="robot", joint_names=["joint_[1-6]"], body_name="link_base",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=(A["arm_step_m"],) * 3 + (math.radians(A["turn_step_deg"]),) * 3)
    fingers = mdp.EMAJointPositionToLimitsActionCfg(asset_name="robot", joint_names=[FINGERS], alpha=A["finger_alpha"])


# ---------------------------------------------------------------- rewards (per step; Isaac Lab multiplies by dt)

def r_reach(env):                   # thumb and the nearer of index / middle, each toward the rim (thumb starts ~25 cm off)
    d = _state(env)["dt_rim"]
    return 1.0 - 0.5 * (torch.tanh(d[:, 0] / 0.15) + torch.tanh(torch.minimum(d[:, 1], d[:, 2]) / 0.15))


def r_pinch(env):
    st = _state(env)
    return (st["near"] & st["opposite"]).float()


def r_touch(env):                   # distal finger links on the plate (0..1)
    return _state(env)["n_touch"].float() / 5.0


def r_lift(env):
    st = _state(env)
    return st["held"].float() * torch.clamp(st["lift"] / 0.10, 0.0, 1.0)


def r_level(env):                   # pilot 4 held the plate up for ~9.5 s but hanging ~44 deg: reward a level hold
    st = _state(env)
    tilt = torch.rad2deg(torch.acos(st["up"][:, 2].clamp(-1.0, 1.0)))
    return (st["held"] & (st["lift"] > 0.02)).float() * (1.0 - torch.tanh(tilt / T["max_tilt_deg"]))


def r_hold(env):
    return _state(env)["lifted"].float()


def r_success(env):
    return _state(env)["success"].float()


def r_knocked(env):
    return _state(env)["knocked"].float()


def r_collision(env):
    return (torch.norm(env.scene["arm_contacts"].data.net_forces_w, dim=-1).amax(dim=1) > 1.0).float()


def r_ended_early(env):             # leaving the box or losing the plate ends the episode: never a way out of penalties
    st = _state(env)
    return (st["tcp_out"] | st["glitch"] | (st["p"][:, 2] < REST_Z - 0.03)).float()


REWARD_FUNCS = {"reach": r_reach, "pinch": r_pinch, "touch": r_touch, "level": r_level, "lift": r_lift, "hold": r_hold, "success": r_success,
                "knocked": r_knocked, "collision": r_collision, "action_rate": mdp.action_rate_l2,
                "ended_early": r_ended_early}


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


def t_glitch(env):
    return _state(env)["glitch"]


def t_tcp_out(env):                 # hand wandered away from its start box
    return _state(env)["tcp_out"]


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=t_success)
    collision = DoneTerm(func=t_collision)
    plate_lost = DoneTerm(func=t_plate_lost)
    tcp_out = DoneTerm(func=t_tcp_out)
    glitch = DoneTerm(func=t_glitch)


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
