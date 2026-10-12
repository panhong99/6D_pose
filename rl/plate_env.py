"""Isaac Lab task: Doosan M1013 + Robotiq 2F-85 grasps the 029 plate by its rim (stage 2) and places it (stage 3).

Not run directly: train.py / eval.py / env_setup.py build it with `make_env_cfg(stage, num_envs)` after the Isaac app
started.  Every number comes from rl/config.yaml.

Every episode starts at the ready joints (assets/ready_joints.json): home -> ready is a slow joint MoveJ on the real robot,
not RL (Human decision 2026-10-06, option A).
  stage 2 grasp : advance to the rim, close, raise ~10 cm; success = held + plate >= 5 cm up + tilt < 25 deg
                  + plate speed < 3 cm/s for 5 s.  Arm capped at 5 cm/s and 10 deg/s, gripper 20 mm/s (speed first)
  stage 3 place : stage 2 terms + carry to the target box, lower, release, retreat; success = released for 1 s
Frames: world == robot base (env origin), +x toward the steel frame, +y robot-left, z up; table top z = -0.02.
Observations are simulator truth (+ 5 mm noise on the plate position as a FoundationPose stand-in), no camera.

Sections: config -> robot / scene -> task state (_state, cached per step) -> action (SafeDiffIKAction) -> observations
-> rewards -> terminations / events -> make_env_cfg.
History of each design choice: collab/DECISIONS.md (v5 .. v10, 2026-10-06).
"""
import json
import math
import sys

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
from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_conjugate, quat_mul, subtract_frame_transforms
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.manipulation.lift import mdp

from rl_utils import SafetyShield, load_config, rl_path

# ---------------------------------------------------------------- config (read once per process)
C = load_config()
G, T, A = C["geometry"], C["task"], C["action"]
REST_Z = G["table_top_z"] + G["box_h"]                                          # plate root (bottom centre) at rest
_rim_xy = torch.tensor(G["rim_offset"], dtype=torch.float32)
_rim_xy = _rim_xy * (1.0 - G["grasp_inset"] / _rim_xy.norm())                   # deeper rim target (Human GUI)
RIM_OFFSET = (float(_rim_xy[0]), float(_rim_xy[1]), G["rim_dz"])
FINGER_CLOSED = C["robot"]["finger_stroke"]                                     # close = full stroke (stalls on the rim)


def _cos(deg):
    return math.cos(math.radians(deg))


# ---------------------------------------------------------------- robot / scene

def robot_cfg():
    r = C["robot"]
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(rl_path(C["paths"]["robot_urdf"])), fix_base=True, activate_contact_sensors=True,
            merge_fixed_joints=False, make_instanceable=False, convert_mimic_joints_to_normal_joints=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type="force", target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={f"joint_{i + 1}": math.radians(q) for i, q in enumerate(r["home_deg"])}
            | {"right_finger_joint": 0.0, "left_finger_joint": 0.0}),
        actuators={name: ImplicitActuatorCfg(joint_names_expr=[a["joints"]], effort_limit_sim=a["effort"],
                                             velocity_limit_sim=a["velocity"], stiffness=a["stiffness"], damping=a["damping"])
                   for name, a in r["actuators"].items()})


def _box(name, xy, color):
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(xy[0], xy[1], G["table_top_z"] + G["box_h"] / 2)),
        spawn=sim_utils.CuboidCfg(size=(G["box_xy"], G["box_xy"], G["box_h"]), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)))


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = robot_cfg()
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.6, 0.0, G["table_top_z"] - 0.05)),
        spawn=sim_utils.CuboidCfg(size=(1.8, 1.2, 0.1), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.27, 0.3)),
                                  physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6)))
    # kinematic, so the plate support can follow the plate at reset (wide plate range: a fixed 8 cm box let the plate tip off)
    start_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/StartBox",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(G["plate_xy"][0], G["plate_xy"][1], G["table_top_z"] + G["box_h"] / 2)),
        spawn=sim_utils.CuboidCfg(size=(G["box_xy"], G["box_xy"], G["box_h"]), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                                  mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.40, 0.25))))
    # (10/8 Human) old stage-3 green target box removed: scenes keep only the plate, its support box and the basket
    plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(G["plate_xy"][0], G["plate_xy"][1], REST_Z)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(rl_path(C["paths"]["plate_usd"])),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=2,
                                                         max_depenetration_velocity=2.0)))
    # arm links + gripper body (fingers excluded): any contact ends the episode (unfiltered net force)
    arm_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/(link_[1-6]|gripper_base)", history_length=1)
    plane = AssetBaseCfg(prim_path="/World/GroundPlane", init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.0)),
                         spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))


def add_steel_frame(scene):
    """Measured frame uprights / rails (calibration/robot_model.frame_layout + data/real_cell_measured.json) as colliders."""
    sys.path.insert(0, str(rl_path(C["paths"]["calibration_dir"])))
    from robot_model import frame_layout, measured_cell
    for name, centre, size in frame_layout(measured_cell()):
        if name != "table":
            setattr(scene, f"Frame_{name}", AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Frame_{name}", init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(centre)),
                spawn=sim_utils.CuboidCfg(size=tuple(size), collision_props=sim_utils.CollisionPropertiesCfg(),
                                          visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.74, 0.77)))))


# ---------------------------------------------------------------- task state (computed once per env step)

def tcp_pose(env):
    r = env.scene["robot"]
    i = r.body_names.index("tcp")
    return r.data.body_pos_w[:, i], r.data.body_quat_w[:, i]


def body_axis(q, k):
    v = torch.zeros(q.shape[0], 3, device=q.device)
    v[:, k] = 1.0
    return quat_apply(q, v)


def env_point(env, xyz):
    return torch.tensor(xyz, device=env.device).unsqueeze(0) + env.scene.env_origins


def finger_closure(env):
    r = env.scene["robot"]
    ids = [r.joint_names.index(n) for n in ("left_finger_joint", "right_finger_joint")]
    return r.data.joint_pos[:, ids].mean(1) / FINGER_CLOSED                     # 0 open .. 1 fully shut


def _state(env):
    """All task quantities, cached per env step (terminations are evaluated before rewards, both read this)."""
    st = getattr(env, "_pp", None)
    if st is not None and st["step"] == env.common_step_counter:
        return st
    if st is None:
        st = reset_task_state(env, torch.arange(env.num_envs, device=env.device))
    dev, stage = env.device, env.cfg.stage
    tcp, tq = tcp_pose(env)
    plate = env.scene["plate"]
    p = plate.data.root_pos_w
    rim = p + getattr(env, "_grasp_rim_offset", torch.tensor(RIM_OFFSET, device=dev))
    ready = p + torch.tensor(G["ready_offset"], device=dev)
    start_rest = env_point(env, (G["plate_xy"][0], G["plate_xy"][1], REST_Z))
    target_rest = env_point(env, (G["target_xy"][0], G["target_xy"][1], REST_Z))
    tool_match = (body_axis(tq, 2) * getattr(env, "_grasp_tool_z", torch.tensor(G["ready_tool_z"], device=dev))).sum(1)
    finger_match = (body_axis(tq, 1) * getattr(env, "_grasp_finger_y", torch.tensor(G["ready_finger_axis"], device=dev))).sum(1)
    fingers = finger_closure(env)
    d_rim = torch.norm(tcp - rim, dim=1)
    gt = T["grasp_tol"]
    at_grasp = (d_rim < gt["dist"]) & (tool_match > _cos(gt["tool_deg"])) & (finger_match > _cos(gt["finger_deg"]))
    # held: fingers stalled on something AND the TCP on the rim circle in the PLATE frame (a plate held by one edge tilts)
    local = quat_apply(quat_conjugate(plate.data.root_quat_w), tcp - p)
    d_circle = torch.sqrt((torch.norm(local[:, :2], dim=1) - G["rim_r"]) ** 2 + (local[:, 2] - G["rim_dz"]) ** 2)
    held = (fingers > T["finger_stall"][0]) & (fingers < T["finger_stall"][1]) & (d_circle < T["held_circle_tol"])
    lift = p[:, 2] - start_rest[:, 2]
    lifted = held & (lift > T["lift_ok"])
    plate_up_z = body_axis(plate.data.root_quat_w, 2)[:, 2]
    # from the position change per policy step: PhysX reports ~30 mm/s on a plate pinched still in a stalled grip
    plate_speed = torch.norm(p - st["p_prev"], dim=1) / env.step_dt
    st["p_prev"] = p.clone()
    stable_lift = lifted & (plate_up_z > _cos(T["max_hold_tilt_deg"])) & (plate_speed < T["hold_max_plate_speed"])
    r = env.scene["robot"]
    tcp_vel = r.data.body_lin_vel_w[:, r.body_names.index("tcp")]
    tcp_acc = torch.norm(tcp_vel - st["tcp_vel"], dim=1) / env.step_dt
    st["tcp_vel"] = tcp_vel.clone()
    d_tgt_xy = torch.norm((p - target_rest)[:, :2], dim=1)
    placed = (d_tgt_xy < 0.025) & ((p[:, 2] - target_rest[:, 2]).abs() < 0.01) & (plate_speed < 0.05)
    plate_moved = torch.norm((p - st["p0"])[:, :2], dim=1) > T["plate_moved_tol"]
    released = placed & (fingers < 0.05) & (tcp[:, 2] > p[:, 2] + 0.075)
    first = held & torch.isnan(st["grasp_z"])
    st["grasp_z"] = torch.where(first, tcp[:, 2], st["grasp_z"])                                  # TCP height at first hold
    st["grasp_xy"] = torch.where(first[:, None], tcp[:, :2], st["grasp_xy"])                       # and where it was
    st["hold"] = torch.where(stable_lift, st["hold"] + 1, torch.zeros_like(st["hold"]))
    st["dwell"] = torch.where(released & (stage == 3), st["dwell"] + 1, torch.zeros_like(st["dwell"]))
    success = st["hold"] >= T["hold_steps"] if stage == 2 else st["dwell"] >= T["release_dwell_steps"]
    st.update(step=env.common_step_counter, tcp=tcp, rim=rim, p=p, lift=lift, d_rim=d_rim,
              d_ready=torch.norm(tcp - ready, dim=1), tool_match=tool_match, finger_match=finger_match, fingers=fingers,
              at_grasp=at_grasp, held=held, lifted=lifted, stable_lift=stable_lift, plate_up_z=plate_up_z,
              d_circle=d_circle, plate_speed=plate_speed, tcp_speed=torch.norm(tcp_vel, dim=1), tcp_acc=tcp_acc,
              tcp_rise=torch.nan_to_num(tcp[:, 2] - st["grasp_z"], nan=0.0),
              d_tgt_xy=d_tgt_xy, target_rest=target_rest, placed=placed, released=released, plate_moved=plate_moved,
              success=success)
    env._pp = st
    return st


def reset_task_state(env, env_ids):
    """Event (mode=reset, after the plate reset): clear counters, remember where the plate started."""
    st = getattr(env, "_pp", None)
    if st is None:
        n, dev = env.num_envs, env.device
        st = env._pp = {"dwell": torch.zeros(n, device=dev), "hold": torch.zeros(n, device=dev), "step": -1,
                        "grasp_z": torch.full((n,), float("nan"), device=dev), "tcp_vel": torch.zeros(n, 3, device=dev),
                        "grasp_xy": torch.zeros(n, 2, device=dev),
                        "p_prev": env.scene["plate"].data.root_pos_w.clone(),
                        "p0": env.scene["plate"].data.root_pos_w.clone()}
    st["dwell"][env_ids] = 0
    st["hold"][env_ids] = 0
    st["grasp_z"][env_ids] = float("nan")
    st["tcp_vel"][env_ids] = 0.0
    st["p_prev"][env_ids] = env.scene["plate"].data.root_pos_w[env_ids]
    st["p0"][env_ids] = env.scene["plate"].data.root_pos_w[env_ids]
    st["step"] = -1
    return st


def box_under_plate(env, env_ids):
    """Event (mode=reset, after the plate shift): the support box follows the plate xy (the Human moves both together)."""
    box, p = env.scene["start_box"], env.scene["plate"].data.root_pos_w[env_ids]
    pose = box.data.default_root_state[env_ids, :7].clone()
    pose[:, :2] = p[:, :2]
    pose[:, 2] = env.scene.env_origins[env_ids, 2] + G["table_top_z"] + G["box_h"] / 2
    box.write_root_pose_to_sim(pose, env_ids=env_ids)


def reset_to_ready(env, env_ids):
    """Event (mode=reset): arm at the ready joints, gripper open, zero velocity."""
    if hasattr(env, "_grasp_bank"):
        from grasp_variants import reset_from_bank
        reset_from_bank(env, env_ids)
        return
    r = env.scene["robot"]
    if not hasattr(env, "_ready_q"):
        q = r.data.default_joint_pos[0].clone()
        for name, v in json.loads(rl_path(C["paths"]["ready_joints"]).read_text())["joints"].items():
            q[r.joint_names.index(name)] = v
        env._ready_q = q
    q = env._ready_q.expand(len(env_ids), -1).clone()
    r.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
    r.set_joint_position_target(q, env_ids=env_ids)


# ---------------------------------------------------------------- action: shielded, position-holding differential IK

class SafeDiffIKAction(DifferentialInverseKinematicsAction):
    """Policy output [dx dy dz droll dpitch dyaw] x scale per step, solved by DLS differential IK, with four additions:
      speed  : the TCP target moves at most max_tcp_speed and rotates at most max_rot_speed_deg (Human: slowest first)
      hold   : position deltas are integrated into a held TCP target (zero action = stay; plain relative IK sank ~7 mm/s
               under the plate), the target leads the real TCP by at most hold_windup
      lock   : just before entering the rim (open gripper, close and aligned) the current TCP orientation is latched and
               rotation outputs are replaced by a correction toward it
      shield : a joint target closer than the shield margins to the frame/table is rejected, the last safe one is kept
               (`blocked` feeds the shield penalty)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self.shield = SafetyShield(dev, C)
        self._safe_target = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self.blocked = torch.zeros(n, dtype=torch.bool, device=dev)
        self.orientation_locked = torch.zeros(n, dtype=torch.bool, device=dev)
        self.orientation_target = self._asset.data.root_quat_w.clone()
        self.hold_pos = torch.zeros(n, 3, device=dev)
        self.hold_init = torch.full((n,), 2, dtype=torch.long, device=dev)

    def process_actions(self, actions):
        self.blocked[:] = False
        actions = actions.clone()
        ee_pos, ee_quat = self._compute_frame_pose()
        # hold: the first two steps after a reset re-read the (fresh) TCP pose, then integrate the deltas
        fresh = self.hold_init > 0
        self.hold_pos[fresh] = ee_pos[fresh]
        self.hold_init[fresh] -= 1
        step = actions[:, :3] * self.cfg.scale
        step = step * (A["max_tcp_speed"] * self._env.step_dt / step.norm(dim=1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0)
        lead = self.hold_pos + step - ee_pos
        lead = lead * (A["hold_windup"] / lead.norm(dim=1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0)
        moving = ~fresh
        # The first two reset steps use the same translation cap as all later steps.
        actions[fresh,:3]=step[fresh]/self.cfg.scale
        self.hold_pos[moving] = (ee_pos + lead)[moving]
        actions[moving, :3] = (self.hold_pos[moving] - ee_pos[moving]) / self.cfg.scale
        # lock: latch the approach orientation near the rim
        st, lk = _state(self._env), A["orientation_lock"]
        capture = (~self.orientation_locked & (st["d_rim"] < lk["dist"]) & (st["tool_match"] > _cos(lk["tool_deg"]))
                   & (st["finger_match"] > _cos(lk["finger_deg"])) & (st["fingers"] < 0.05))
        self.orientation_target[capture] = ee_quat[capture]
        self.orientation_locked |= capture
        err = axis_angle_from_quat(quat_mul(self.orientation_target, quat_conjugate(ee_quat)))
        locked = self.orientation_locked
        actions[locked, 3:6] = err[locked] / self.cfg.scale
        rot = actions[:, 3:6] * self.cfg.scale                                          # rotation speed cap (also the lock)
        rot_max = math.radians(A["max_rot_speed_deg"]) * self._env.step_dt
        actions[:, 3:6] = rot * (rot_max / rot.norm(dim=1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0) / self.cfg.scale
        super().process_actions(actions)

    def apply_actions(self):
        ee_pos, ee_quat = self._compute_frame_pose()
        q = self._asset.data.joint_pos[:, self._joint_ids]
        q_des = self._ik_controller.compute(ee_pos, ee_quat, self._compute_frame_jacobian(), q)
        replay=getattr(self._env,'_candidate_joint_replay_target',None)
        if replay is not None:q_des=replay
        joint_speed=getattr(self._env,'_arm_joint_speed_rad_s',None)
        if joint_speed is not None:
            dq=joint_speed*self._env.physics_dt
            # Rate-limit the commanded target. Limiting relative to measured q would repeatedly
            # erase the tracking error needed by the PD drive and make the arm stall.
            q_des=self._safe_target+(q_des-self._safe_target).clamp(-dq,dq)
        ok = self.shield.safe(q_des)
        self.blocked |= ~ok
        self._safe_target = torch.where(ok[:, None], q_des, self._safe_target)
        self._asset.set_joint_position_target(self._safe_target, self._joint_ids)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._safe_target[ids] = self._asset.data.joint_pos[ids][:, self._joint_ids]      # as written by reset_to_ready
        self.blocked[ids] = False
        self.orientation_locked[ids] = False
        self.hold_init[ids] = 2


@configclass
class SafeDiffIKActionCfg(mdp.DifferentialInverseKinematicsActionCfg):
    class_type: type = SafeDiffIKAction


@configclass
class ActionsCfg:
    arm_action = SafeDiffIKActionCfg(
        asset_name="robot", joint_names=["joint_[1-6]"], body_name="tcp", scale=A["scale"],
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"))
    # Isaac Lab binary action: < 0 closes (full stroke, stalls on the rim), >= 0 opens
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot", joint_names=[".*_finger_joint"],
        open_command_expr={".*_finger_joint": 0.0}, close_command_expr={".*_finger_joint": FINGER_CLOSED})


# ---------------------------------------------------------------- observations (43-dim, order fixed for old checkpoints)

def _to_base(env, pos_w):
    r = env.scene["robot"]
    return subtract_frame_transforms(r.data.root_pos_w, r.data.root_quat_w, pos_w)[0]


def obs_tcp(env):
    return _to_base(env, tcp_pose(env)[0])


def obs_tool_axes(env):
    q = tcp_pose(env)[1]
    return torch.cat([body_axis(q, 2), body_axis(q, 1)], dim=1)


def obs_plate(env):
    return _to_base(env, env.scene["plate"].data.root_pos_w)


def obs_plate_up(env):
    return body_axis(env.scene["plate"].data.root_quat_w, 2)


def obs_target(env):
    return _to_base(env, env_point(env, (G["target_xy"][0], G["target_xy"][1], REST_Z)))


def obs_rim_rel(env):               # rim grasp target - TCP: goal-relative input for the wide plate range (2026-10-07 night)
    st = _state(env)
    return st["rim"] - st["tcp"]


def obs_progress(env):
    st = _state(env)
    return torch.stack([st["hold"] / T["hold_steps"], st["dwell"] / 30.0], dim=1).clamp(max=1.0)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.5)
        tcp = ObsTerm(func=obs_tcp)
        tool_axes = ObsTerm(func=obs_tool_axes)
        plate = ObsTerm(func=obs_plate, noise=Unoise(n_min=-T["plate_obs_noise"], n_max=T["plate_obs_noise"]))
        plate_up = ObsTerm(func=obs_plate_up)
        target = ObsTerm(func=obs_target)
        progress = ObsTerm(func=obs_progress)
        actions = ObsTerm(func=mdp.last_action)
        rim_rel = ObsTerm(func=obs_rim_rel, noise=Unoise(n_min=-T["plate_obs_noise"], n_max=T["plate_obs_noise"]))   # 43:46

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------- rewards (per step; Isaac Lab multiplies by dt)

def r_reach(env, std):              # near the ready point
    return 1.0 - torch.tanh(_state(env)["d_ready"] / std)


def r_tool(env):
    return torch.clamp(_state(env)["tool_match"], min=0.0) ** 2


def r_finger(env):
    return torch.clamp(_state(env)["finger_match"], min=0.0) ** 2


def r_angle(env, key, std_deg):     # still has slope at 20+ deg where cos^2 is flat
    return 1.0 - torch.tanh(torch.rad2deg(torch.acos(_state(env)[key].clamp(-1.0, 1.0))) / std_deg)


def r_grasp_reach(env, std):
    st = _state(env)
    return (1.0 - torch.tanh(st["d_rim"] / std)) * torch.clamp(st["tool_match"], min=0.0)


def r_open_before_grasp(env):
    st = _state(env)
    return st["fingers"] * (~st["at_grasp"] & ~st["held"]).float()


def r_close_at_grasp(env):
    st = _state(env)
    return st["fingers"] * (st["at_grasp"] | st["held"]).float()


def r_lift(env):
    st = _state(env)
    return st["held"].float() * torch.clamp(st["lift"] / T["lift_target"], 0.0, 1.0)


def r_tcp_rise(env):
    st = _state(env)
    return st["held"].float() * torch.clamp(st["tcp_rise"] / T["lift_target"], 0.0, 1.0)


def r_over_lift(env):               # no reward for lifting higher than the band (the fast policy went to 45 cm)
    st = _state(env)
    return st["held"].float() * torch.clamp((st["lift"] - T["lift_target"] - T["lift_band"]) / 0.05, 0.0, 2.0)


def r_hold_still(env):              # once up, the plate should stay still (the fast policy bobbed it 39 cm)
    st = _state(env)
    up = st["lifted"] & (st["lift"] > T["lift_target"] - T["lift_band"])
    return up.float() * torch.clamp(st["plate_speed"] / T["hold_max_plate_speed"], 0.0, 3.0)


def r_held_drift(env, scale_m=0.01, max_scale=40.0):  # retain a gradient through 40 cm of post-grasp drift
    st = _state(env)
    d = torch.norm(st["tcp"][:, :2] - st["grasp_xy"], dim=1)
    return st["held"].float() * torch.clamp(d / scale_m, 0.0, max_scale)


def r_held_speed(env):
    st = _state(env)
    return st["held"].float() * (st["tcp_speed"] / A["max_tcp_speed"]).square()


def r_held_accel(env):
    st = _state(env)
    return st["held"].float() * st["tcp_acc"].square().clamp(max=4.0)


def r_level_hold(env):
    st = _state(env)
    return st["lifted"].float() * st["plate_up_z"].clamp(min=0.0).square()


def r_carry(env, std):
    st = _state(env)
    return st["lifted"].float() * (1.0 - torch.tanh(st["d_tgt_xy"] / std))


def r_lower(env):
    st = _state(env)
    over = (st["d_tgt_xy"] < 0.03) & st["held"]
    return over.float() * (1.0 - torch.tanh((st["p"][:, 2] - st["target_rest"][:, 2]).abs() / 0.03))


def r_release(env):
    st = _state(env)
    up = torch.clamp((st["tcp"][:, 2] - st["p"][:, 2] - 0.025) / 0.075, 0.0, 1.0)
    return st["placed"].float() * (1.0 - st["fingers"]) * (0.5 + 0.5 * up)


def r_success(env):
    return _state(env)["success"].float()


def r_hold_progress(env):
    return _state(env)["stable_lift"].float()


def r_dropped(env):
    return t_plate_lost(env).float()


def r_shield(env):
    return env.action_manager.get_term("arm_action").blocked.float()


def r_collision(env):
    return (torch.norm(env.scene["arm_contacts"].data.net_forces_w, dim=-1).amax(dim=1) > 1.0).float()


def r_knocked(env):
    st = _state(env)
    return (st["plate_moved"] & ~st["held"] & ~st["placed"]).float()


REWARD_FUNCS = {
    "reach": r_reach, "tool": r_tool, "finger": r_finger, "tool_angle": r_angle, "finger_angle": r_angle,
    "grasp_reach": r_grasp_reach, "grasp_reach_coarse": r_grasp_reach, "open_before_grasp": r_open_before_grasp,
    "close_at_grasp": r_close_at_grasp, "lift": r_lift, "tcp_rise": r_tcp_rise, "level_hold": r_level_hold,
    "over_lift": r_over_lift, "hold_still": r_hold_still, "held_speed": r_held_speed, "held_accel": r_held_accel,
    "held_drift": r_held_drift,
    "carry": r_carry, "carry_fine": r_carry, "lower": r_lower, "release": r_release, "success": r_success,
    "shield": r_shield, "collision": r_collision, "knocked": r_knocked, "hold_progress": r_hold_progress,
    "dropped": r_dropped,
    "action_rate": mdp.action_rate_l2, "joint_vel": mdp.joint_vel_l2,
}


@configclass
class RewardsCfg:
    pass                                                           # filled from config.yaml by make_env_cfg


def reward_terms(stage):
    base = "residual" if C.get("residual", {}).get("enabled", False) else "stage2"
    terms = dict(C["rewards"][base])
    if stage >= 3:
        terms.update(C["rewards"]["stage3"])
    out = {}
    for name, (weight, params) in terms.items():
        params = dict(params)
        if name == "joint_vel":
            params["asset_cfg"] = SceneEntityCfg("robot", joint_names=["joint_[1-6]"])
        out[name] = RewTerm(func=REWARD_FUNCS[name], weight=float(weight), params=params)
    return out


# ---------------------------------------------------------------- terminations / events

def t_success(env):
    return _state(env)["success"]


def t_collision(env):
    return r_collision(env) > 0.5


def t_plate_lost(env):              # dropped: below the box top and not held
    st = _state(env)
    return (st["p"][:, 2] < REST_Z - 0.03) & ~st["held"]


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=t_success)
    collision = DoneTerm(func=t_collision)
    plate_lost = DoneTerm(func=t_plate_lost)


def _friction(asset_cfg):
    s, d = C["physx"]["plate_friction"]
    return EventTerm(func=mdp.randomize_rigid_body_material, mode="startup",
                     params={"asset_cfg": asset_cfg, "static_friction_range": (s, s), "dynamic_friction_range": (d, d),
                             "restitution_range": (0.0, 0.0), "num_buckets": 1})


@configclass
class EventCfg:                     # order matters: scene default -> plate shift -> task state -> ready joints
    plate_friction = _friction(SceneEntityCfg("plate"))
    finger_friction = _friction(SceneEntityCfg("robot", body_names=".*_finger"))
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_plate = EventTerm(func=mdp.reset_root_state_uniform, mode="reset", params={
        "pose_range": {"x": tuple(G.get("plate_reset_x", (-G["plate_reset_xy"], G["plate_reset_xy"]))),
                       "y": tuple(G.get("plate_reset_y", (-G["plate_reset_xy"], G["plate_reset_xy"]))),
                       "yaw": (-math.pi, math.pi)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("plate")})
    box_follow = EventTerm(func=box_under_plate, mode="reset")
    reset_state = EventTerm(func=reset_task_state, mode="reset")
    ready_start = EventTerm(func=reset_to_ready, mode="reset")


# ---------------------------------------------------------------- env config

@configclass
class PlateEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=3.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    stage: int = 2

    def __post_init__(self):
        self.decimation = T["decimation"]
        self.sim.dt = T["sim_dt"]
        self.sim.render_interval = self.decimation
        px = C["physx"]
        self.sim.physx.bounce_threshold_velocity = px["bounce_threshold_velocity"]
        self.sim.physx.friction_correlation_distance = px["friction_correlation_distance"]
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = px["gpu_found_lost_aggregate_pairs_capacity"]
        self.sim.physx.gpu_total_aggregate_pairs_capacity = px["gpu_total_aggregate_pairs_capacity"]
        add_steel_frame(self.scene)


def make_env_cfg(stage, num_envs, viewer=False, rim_rel=None):
    """Env config for stage 2 (grasp) or 3 (place).  viewer=True frames env 0 for the GUI.
    rim_rel: append the 3-dim rim-target-minus-TCP observation (46-dim obs); None = config task.obs_rim_rel.
    Checkpoints before 2026-10-07 night use 43 dims (rim_rel=False)."""
    assert stage in (2, 3), stage
    cfg = PlateEnvCfg()
    if not (T.get("obs_rim_rel", False) if rim_rel is None else rim_rel):
        cfg.observations.policy.rim_rel = None
    cfg.stage = stage
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = T["episode_s"][stage]
    for name, term in reward_terms(stage).items():
        setattr(cfg.rewards, name, term)
    if num_envs <= 64:              # small diagnostic runs do not need contact buffers sized for 4096 envs
        cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1 << 18
        cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 1 << 18
    if viewer:
        cfg.scene.env_spacing = 2.5
        cfg.viewer.origin_type = "env"
        cfg.viewer.env_index = 0
        cfg.viewer.eye = (1.9, 1.5, 1.3)
        cfg.viewer.lookat = (0.9, -0.2, 0.1)
    return cfg
