"""Isaac Lab env (version from experiment_settings): M1013 + Robotiq 2F-85 side-rim pick and place.

Stages (Human 2026-10-06), selected with the environment variable PP_STAGE (1..3); each stage contains the previous ones
and starts exactly at the Human home joints (-5.98, -14.75, 117.74, -3.92, 7.46, -85.07 deg, no reset noise), so a stage-k policy is initialised from the stage-(k-1) policy:
  1 ready : from home, approach the robot-facing rim, roll wrist joint 6 +90 deg, align the open gripper centreline with the rim
  2 grasp : advance from the ready pose, close on the rim, lift >= 5 cm in base +z and hold 5 s
  3 place : carry to the target box, lower, release and retreat; plate still for 1 s
Success of a stage ends the episode with a bonus; the curriculum moves on only at >= 95 % success (rl/curriculum.py).

Frames: world == robot base (env origin), +x toward the steel frame, +y robot-left, z up.  Table top z = -0.02.
Safety: rl/safety_shield.py rejects any arm target closer than 5 cm to the steel frame / 2 cm to the table (voxel distance
fields); contact of an arm link with anything also ends the episode with a penalty.
Plate: the scanned 029 mesh (assets/plate029.usd, origin = bottom centre), placed at the table centre (Human: far from the
frame), target box 0.5 m away in -y.  Pose observations are simulator truth + 5 mm noise (stand-in for FoundationPose).
"""
import math
import os
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
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
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul, axis_angle_from_quat, subtract_frame_transforms
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.manipulation.lift import mdp

from m1013_2f85_cfg import M1013_2F85_CFG
from safety_shield import SafetyShield
from experiment_settings import VERSION

STAGE = int(os.environ.get("PP_STAGE", "1"))
assert STAGE in (1, 2, 3), STAGE
# v5 grasp fixes (Claude 2026-10-06, rl/check_rim_grasp.py): opt-in so the running v4 curriculum is unchanged.
# Full-stroke close, tilt-invariant `held`, looser at_grasp angles, no ready-pose pull in stage 2+, plate_moved vs reset pose.
_VN = int(VERSION[1:])
V5 = _VN >= 5
V6 = _VN >= 6
# v7 (Claude 2026-10-06): v6 task unchanged; stage 1 retrained at the 114.6 mm support (model_999 from 54.6 mm: 0 % there),
# and stage-2+ TRAINING may start a fraction of episodes at the ready pose (PP_READY_START_FRAC; eval/GUI always home).
V7 = _VN >= 7
# v8 (Human decision 2026-10-06, option A): home -> ready is a joint interpolation / slow MoveJ, not RL.  Stage 2+
# episodes (training, evaluation and GUI alike) start at the ready joints; RL learns advance -> close -> lift -> hold.
V8 = _VN >= 8
# v9 (Claude 2026-10-06): the arm holds an integrated TCP position target.  Relative diff-IK re-anchors on the current pose
# every policy step, so under the plate's weight a zero action sinks ~7 mm/s (scripted test: 68 -> 35 mm lift in 5 s) and
# the v8 policy learned to lift, sink back onto the box and just keep holding.
V9 = _VN >= 9
# v10 (Claude 2026-10-06): reward raising the TCP while holding.  v9 policy grasps and holds but keeps the TCP at grasp
# height (plate lift ~14 mm from the pinch tilt only); plate-root lift alone pays little because the weakly gripped plate
# pivots in the fingers as the TCP rises (forced +z test: plate 22 mm then back to 8 mm).
V10 = _VN >= 10
HOLD_WINDUP = 0.03                               # target may lead the actual TCP by at most 3 cm (shield/contact stops)

# ---------------------------------------------------------------- geometry (m), robot base frame
TABLE_TOP_Z = -0.02
BOX_H, BOX_XY = float(os.environ.get("PP_BOX_H", "0.1146" if V6 else "0.0546")), 0.08
PLATE_USD = str(Path(__file__).resolve().parent / "assets" / "plate029_textured.usda")
PLATE_MASS = 0.30                                 # ASSUMED (set in make_plate_usd.py)
PLATE_XY = (1.10, 0.0)                            # Human 2026-10-03: moved out from (0.60, 0) - too close to the base
TARGET_XY = (1.10, -0.45)                         # 0.45 m sideways: at -0.50 the rim grasp is 4.5 cm from an upright (< 5 cm shield)
REST_Z = TABLE_TOP_Z + BOX_H                      # plate root (bottom centre) resting on a box
RIM_R, RIM_DZ = 0.128, 0.016                      # grasp on the rim wall: radius, height above the plate bottom
# Human-confirmed GUI pose from calibration/data/rim_gripper_roll_preview_20261006.json.
# Offsets follow plate translation; the circular plate may reset with random yaw.
READY_OFFSET = (-0.160000, -0.080000, 0.041038094508)
RIM_OFFSET = (-0.093269058948, -0.087663462417, RIM_DZ)
GRASP_INSET = 0.015 if V6 and STAGE >= 2 else 0.0
if GRASP_INSET:
    radius = math.hypot(*RIM_OFFSET[:2])
    RIM_OFFSET = tuple(v * (1.0 - GRASP_INSET / radius) for v in RIM_OFFSET[:2]) + (RIM_DZ,)
READY_TOOL_Z = (0.930899055477, -0.106905579532, -0.349282329321)
READY_FINGER_AXIS = (-0.345298586772, 0.054353027066, -0.936917624139)  # TCP +y after Menagerie mount
# In the simplified RL gripper, the two collision pads start about 85.4 mm apart.
# Close each prismatic jaw by 30 mm so the target aperture is ~25.4 mm, matching
# the scanned plate thickness and the ~24 mm aperture used by the MJCF GUI demo.
FINGER_CLOSED = 0.0300
if V5:
    # 0.030 leaves a 25.4 mm gap and never touches the rim (scripted test: fingers reach 30.0 mm, plate not lifted).
    # Close the full stroke like the real gripper (close until contact): fingers stall at ~37 mm on the rim.
    FINGER_CLOSED = 0.0425
FINGER_STALL = (0.5, 0.97)                        # v5 `held`: closed on something (fully closed = nothing between pads)
LIFT_OK = 0.05                                    # "lifted" = bottom 5 cm above its rest height
CTRL_HZ = 30                                      # policy rate (sim 120 Hz, decimation 4)
HOLD_STEPS = 150                                  # stage 2: hold 5 s (Human)
DWELL_STEPS = {1: 15, 3: 30}                      # success must persist this many policy steps
EPISODE_S = {1: 10.0, 2: 14.0, 3: 24.0}
if V6:
    EPISODE_S[2] = 24.0  # home -> ready -> advance -> close -> lift -> 5 s hold
READY_START_FRAC = float(os.environ.get("PP_READY_START_FRAC", "0")) if V7 and STAGE >= 2 else 0.0
if V8 and STAGE >= 2:
    READY_START_FRAC = 1.0
    EPISODE_S[2] = 14.0                          # ready -> advance -> close -> lift -> 5 s hold (no home -> ready leg)
READY_JOINTS_FILE = Path(__file__).resolve().parent / "assets" / "ready_joints_v7.json"
MAX_HOLD_TILT_DEG = 25.0  # demo showed 9..16 deg; permit a modest margin, reject near-vertical holds


# ---------------------------------------------------------------- safe IK action
class SafeDiffIKAction(DifferentialInverseKinematicsAction):
    """Differential-IK TCP pose deltas whose joint targets must pass the voxel shield; a rejected target keeps the last
    safe target (the arm stops there), and the rejection is recorded for the penalty term."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.shield = SafetyShield(self.device)
        self._safe_target = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self.blocked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.orientation_locked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.orientation_target = self._asset.data.root_quat_w.clone()
        self.hold_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.hold_init = torch.full((self.num_envs,), 2, dtype=torch.long, device=self.device)

    def process_actions(self, actions):
        self.blocked[:] = False
        if V9:
            # integrate position deltas into a held target; the first steps after a reset re-read the (fresh) TCP pose
            ee_pos, _ = self._compute_frame_pose()
            fresh = self.hold_init > 0
            self.hold_pos[fresh] = ee_pos[fresh]
            self.hold_init[fresh] -= 1
            actions = actions.clone()
            moving = ~fresh
            target = self.hold_pos + actions[:, :3] * self.cfg.scale
            lead = target - ee_pos
            n = lead.norm(dim=1, keepdim=True).clamp(min=1e-9)
            target = ee_pos + lead * (HOLD_WINDUP / n).clamp(max=1.0)
            self.hold_pos[moving] = target[moving]
            actions[moving, :3] = (self.hold_pos[moving] - ee_pos[moving]) / self.cfg.scale
        if V6 and STAGE == 2:
            # Latch the achieved approach orientation just before entering the rim. This is the
            # same pose-hold controller as the approved demo, shared by train/eval/play.
            st = _state(self._env)
            capture = (~self.orientation_locked & (st['d_rim'] < 0.035)
                       & (st['tool_match'] > math.cos(math.radians(20)))
                       & (st['finger_match'] > math.cos(math.radians(25))) & (st['fingers'] < 0.05))
            _, current_quat = self._compute_frame_pose()
            self.orientation_target[capture] = current_quat[capture]
            self.orientation_locked |= capture
            error = axis_angle_from_quat(quat_mul(self.orientation_target, quat_conjugate(current_quat)))
            actions = actions.clone()
            actions[self.orientation_locked, 3:6] = (error[self.orientation_locked] / 0.03).clamp(-0.3, 0.3)
        super().process_actions(actions)

    def apply_actions(self):
        ee_pos, ee_quat = self._compute_frame_pose()
        q = self._asset.data.joint_pos[:, self._joint_ids]
        q_des = self._ik_controller.compute(ee_pos, ee_quat, self._compute_frame_jacobian(), q)
        ok = self.shield.safe(q_des)
        self.blocked |= ~ok
        self._safe_target = torch.where(ok[:, None], q_des, self._safe_target)
        self._asset.set_joint_position_target(self._safe_target, self._joint_ids)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        # joints as written by the reset events (home, or the v7 ready start); == default_joint_pos without a ready start
        self._safe_target[ids] = self._asset.data.joint_pos[ids][:, self._joint_ids]
        self.blocked[ids] = False
        self.orientation_locked[ids] = False
        self.hold_init[ids] = 2


@configclass
class SafeDiffIKActionCfg(mdp.DifferentialInverseKinematicsActionCfg):
    class_type: type = SafeDiffIKAction


# ---------------------------------------------------------------- scene
def _box(name, xy, color):
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(xy[0], xy[1], TABLE_TOP_Z + BOX_H / 2)),
        spawn=sim_utils.CuboidCfg(size=(BOX_XY, BOX_XY, BOX_H), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)))


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = M1013_2F85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.6, 0.0, TABLE_TOP_Z - 0.05)),
        spawn=sim_utils.CuboidCfg(size=(1.8, 1.2, 0.1), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.27, 0.3)),
                                  physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6)))
    start_box = _box("StartBox", PLATE_XY, (0.55, 0.40, 0.25))
    target_box = _box("TargetBox", TARGET_XY, (0.30, 0.55, 0.30))
    plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(PLATE_XY[0], PLATE_XY[1], REST_Z)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=PLATE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=2,
                                                         max_depenetration_velocity=2.0)))
    # arm links + gripper body (fingers excluded): any contact ends the episode (unfiltered net force, self-collision off)
    arm_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/(link_[1-6]|gripper_base)", history_length=1)
    plane = AssetBaseCfg(prim_path="/World/GroundPlane", init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.0)),
                         spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))


# ---------------------------------------------------------------- task state (computed once per env step)
def _tcp(env):
    r = env.scene["robot"]
    i = r.body_names.index("tcp")
    return r.data.body_pos_w[:, i], r.data.body_quat_w[:, i]


def _axis(q, k, n):
    v = torch.zeros(n, 3, device=q.device)
    v[:, k] = 1.0
    return quat_apply(q, v)


def _const(env, xyz):
    return torch.tensor(xyz, device=env.device).unsqueeze(0) + env.scene.env_origins


def _fingers(env):
    r = env.scene["robot"]
    ids = [r.joint_names.index(n) for n in ("left_finger_joint", "right_finger_joint")]
    return r.data.joint_pos[:, ids].mean(1) / FINGER_CLOSED                        # 0 open .. 1 closed


def _state(env):
    """All task quantities, cached per env step (ManagerBasedRLEnv computes terminations before rewards)."""
    st = getattr(env, "_pp", None)
    if st is not None and st["step"] == env.common_step_counter:
        return st
    n = env.num_envs
    if st is None:
        st = {"dwell": torch.zeros(n, device=env.device), "hold": torch.zeros(n, device=env.device), "step": -1}
    tcp, tq = _tcp(env)
    plate = env.scene["plate"]
    p = plate.data.root_pos_w
    rim = p + torch.tensor(RIM_OFFSET, device=env.device)
    ready = p + torch.tensor(READY_OFFSET, device=env.device)
    start_rest = _const(env, (PLATE_XY[0], PLATE_XY[1], REST_Z))
    target_rest = _const(env, (TARGET_XY[0], TARGET_XY[1], REST_Z))
    tool_z = _axis(tq, 2, n)
    finger_ax = _axis(tq, 1, n)                                                     # tcp y = finger closing axis
    fingers = _fingers(env)
    d_rim = torch.norm(tcp - rim, dim=1)
    d_ready = torch.norm(tcp - ready, dim=1)
    tool_match = (tool_z * torch.tensor(READY_TOOL_Z, device=env.device)).sum(1)
    finger_match = (finger_ax * torch.tensor(READY_FINGER_AXIS, device=env.device)).sum(1)
    rim_delta = rim - tcp
    axis_error = torch.norm(torch.cross(rim_delta, tool_z, dim=1), dim=1)
    forward = (rim_delta * tool_z).sum(1)
    oriented = (tool_match > math.cos(math.radians(10))) & (finger_match > math.cos(math.radians(15)))
    at_ready = (d_ready < 0.012) & oriented & (axis_error < 0.012) & (forward > 0.04) & (forward < 0.10)
    if V5:
        # stage-1 policy ends at ~10.1 deg tool error (its own gate is 10 deg), so a 10 deg at_grasp almost never fires
        grasp_oriented = (tool_match > math.cos(math.radians(20))) & (finger_match > math.cos(math.radians(25)))
        at_grasp = (d_rim < 0.015) & grasp_oriented
        # held: TCP on the rim circle in the PLATE frame (a plate held by one edge tilts ~20 deg when lifted, which the
        # world-frame test rejected) and the fingers stalled on something
        local = quat_apply(quat_conjugate(plate.data.root_quat_w), tcp - p)
        d_circle = torch.sqrt((torch.norm(local[:, :2], dim=1) - RIM_R) ** 2 + (local[:, 2] - RIM_DZ) ** 2)
        held = (fingers > FINGER_STALL[0]) & (fingers < FINGER_STALL[1]) & (d_circle < 0.03)
    else:
        at_grasp = (d_rim < 0.015) & oriented
        # held: closed fingers remain centred on the moving rim while the plate lifts
        d_centre_xy = torch.norm((tcp - p)[:, :2], dim=1)
        held = (fingers > 0.15) & (fingers < 1.01) & ((d_centre_xy - RIM_R).abs() < 0.03) & \
               ((tcp[:, 2] - p[:, 2] - RIM_DZ).abs() < 0.035)
    lift = p[:, 2] - start_rest[:, 2]
    lifted = held & (lift > LIFT_OK)
    plate_up_z = _axis(plate.data.root_quat_w, 2, n)[:, 2]
    stable_lift = lifted
    if V6:
        stable_lift = lifted & (plate_up_z > math.cos(math.radians(MAX_HOLD_TILT_DEG)))
    d_tgt_xy = torch.norm((p - target_rest)[:, :2], dim=1)
    speed = torch.norm(plate.data.root_lin_vel_w, dim=1)
    placed = (d_tgt_xy < 0.025) & ((p[:, 2] - target_rest[:, 2]).abs() < 0.01) & (speed < 0.05)
    # knocked off its place; v4 measured from the nominal spot although reset already shifts the plate up to 1.4 cm
    p0 = st["p0"] if V5 and "p0" in st else start_rest
    plate_moved = torch.norm((p - p0)[:, :2], dim=1) > 0.02
    released = placed & (fingers < 0.05) & (tcp[:, 2] > p[:, 2] + 0.025 + 0.05)

    if STAGE == 1:
        cond = at_ready & (fingers < 0.05) & ~plate_moved
    elif STAGE == 3:
        cond = released
    else:
        cond = torch.zeros_like(at_ready)
    st["dwell"] = torch.where(cond, st["dwell"] + 1, torch.zeros_like(st["dwell"]))
    st["hold"] = torch.where(stable_lift, st["hold"] + 1, torch.zeros_like(st["hold"]))
    if V10:
        if "grasp_z" not in st:
            st["grasp_z"] = torch.full((n,), float("nan"), device=env.device)
        st["grasp_z"] = torch.where(held & torch.isnan(st["grasp_z"]), tcp[:, 2], st["grasp_z"])
        st["tcp_rise"] = torch.nan_to_num(tcp[:, 2] - st["grasp_z"], nan=0.0)
    success = st["hold"] >= HOLD_STEPS if STAGE == 2 else st["dwell"] >= DWELL_STEPS[STAGE]
    st.update(step=env.common_step_counter, tcp=tcp, rim=rim, ready=ready, p=p, lift=lift, d_rim=d_rim,
              d_ready=d_ready, tool_match=tool_match, finger_match=finger_match, axis_error=axis_error,
              forward=forward, at_ready=at_ready, at_grasp=at_grasp, fingers=fingers, held=held, lifted=lifted, d_tgt_xy=d_tgt_xy,
              target_rest=target_rest, placed=placed, released=released, plate_moved=plate_moved, success=success,
              stable_lift=stable_lift, plate_up_z=plate_up_z)
    env._pp = st
    return st


def reset_ready_start(env, env_ids, frac: float):
    """v7 training aid: put a random `frac` of the reset envs at the ready pose (open gripper, zero velocity), so stage 2
    sees advance/close/lift from the first iterations instead of only after a 10 s approach."""
    import json
    if not hasattr(env, "_ready_q"):
        r = env.scene["robot"]
        names = json.loads(READY_JOINTS_FILE.read_text())["joints"]
        q = r.data.default_joint_pos[0].clone()
        for name, v in names.items():
            q[r.joint_names.index(name)] = v
        env._ready_q = q
    pick = env_ids[torch.rand(len(env_ids), device=env.device) < frac]
    if len(pick) == 0:
        return
    r = env.scene["robot"]
    q = env._ready_q.expand(len(pick), -1).clone()
    r.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=pick)
    r.set_joint_position_target(q, env_ids=pick)


def reset_stage_state(env, env_ids):
    st = getattr(env, "_pp", None)
    if st is None:
        st = env._pp = {"dwell": torch.zeros(env.num_envs, device=env.device),
                        "hold": torch.zeros(env.num_envs, device=env.device), "step": -1}
    st["dwell"][env_ids] = 0
    st["hold"][env_ids] = 0
    st["step"] = -1
    if "grasp_z" in st:
        st["grasp_z"][env_ids] = float("nan")
    if V5:                                                                          # runs after reset_plate
        if "p0" not in st:
            st["p0"] = env.scene["plate"].data.root_pos_w.clone()
        st["p0"][env_ids] = env.scene["plate"].data.root_pos_w[env_ids]


# ---------------------------------------------------------------- observations
def _to_base(env, pos_w):
    r = env.scene["robot"]
    return subtract_frame_transforms(r.data.root_pos_w, r.data.root_quat_w, pos_w)[0]


def obs_tcp(env):
    return _to_base(env, _tcp(env)[0])


def obs_tool_axes(env):
    q = _tcp(env)[1]
    return torch.cat([_axis(q, 2, env.num_envs), _axis(q, 1, env.num_envs)], dim=1)


def obs_plate(env):
    return _to_base(env, env.scene["plate"].data.root_pos_w)


def obs_plate_up(env):
    return _axis(env.scene["plate"].data.root_quat_w, 2, env.num_envs)


def obs_target(env):
    return _to_base(env, _const(env, (TARGET_XY[0], TARGET_XY[1], REST_Z)))


def obs_progress(env):
    st = _state(env)
    return torch.stack([st["hold"] / HOLD_STEPS, st["dwell"] / 30.0], dim=1).clamp(max=1.0)


# ---------------------------------------------------------------- rewards (per step; Isaac Lab multiplies by dt)
def r_reach(env, std: float):
    return 1.0 - torch.tanh(_state(env)["d_ready"] / std)


def r_tool(env):
    return torch.clamp(_state(env)["tool_match"], min=0.0) ** 2


def r_finger(env):
    return torch.clamp(_state(env)["finger_match"], min=0.0) ** 2


def r_angle(env, key: str, std_deg: float):
    """v7: angle-sensitive orientation shaping.  cos^2 already pays 85 % at 23 deg, and the retrained stage-1 policy
    settled there (tool 23 / finger 17 deg vs the 10 / 15 deg gate, 0 % success); this kernel still has slope there."""
    ang = torch.rad2deg(torch.acos(_state(env)[key].clamp(-1.0, 1.0)))
    return 1.0 - torch.tanh(ang / std_deg)


def r_axis(env):
    st = _state(env)
    return (1.0 - torch.tanh(st["axis_error"] / 0.025)) * (1.0 - torch.tanh(st["d_ready"] / 0.10))


def r_grasp_reach(env, std: float = 0.035):
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
    return st["held"].float() * torch.clamp(st["lift"] / 0.08, 0.0, 1.0)


def r_tcp_rise(env):
    st = _state(env)
    return st["held"].float() * torch.clamp(st["tcp_rise"] / 0.10, 0.0, 1.0)


def r_level_hold(env):
    st = _state(env)
    return st['lifted'].float() * st['plate_up_z'].clamp(min=0.0).square()


def r_carry(env, std: float):
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


def r_shield(env):
    return env.action_manager.get_term("arm_action").blocked.float()


def collision_force(env, threshold: float = 1.0):
    f = env.scene["arm_contacts"].data.net_forces_w
    return (torch.norm(f, dim=-1).amax(dim=1) > threshold).float()


def r_plate_knocked(env):
    st = _state(env)
    return (st["plate_moved"] & ~st["held"] & ~st["placed"]).float()


# ---------------------------------------------------------------- terminations
def t_success(env):
    return _state(env)["success"]


def t_collision(env):
    return collision_force(env) > 0.5


def t_plate_lost(env):
    """Stage 1: plate knocked off its place.  Later stages: plate below the box tops and not held (dropped)."""
    st = _state(env)
    if STAGE == 1:
        return st["plate_moved"]
    return (st["p"][:, 2] < REST_Z - 0.03) & ~st["held"]


# ---------------------------------------------------------------- MDP groups
@configclass
class ActionsCfg:
    # TCP pose deltas [dx, dy, dz, droll, dpitch, dyaw] x 0.03 (m / rad) per policy step, base frame, shielded
    arm_action = SafeDiffIKActionCfg(
        asset_name="robot", joint_names=["joint_[1-6]"], body_name="tcp", scale=0.03,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"))
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot", joint_names=[".*_finger_joint"],
        open_command_expr={".*_finger_joint": 0.0}, close_command_expr={".*_finger_joint": FINGER_CLOSED})


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.5)
        tcp = ObsTerm(func=obs_tcp)
        tool_axes = ObsTerm(func=obs_tool_axes)
        plate = ObsTerm(func=obs_plate, noise=Unoise(n_min=-0.005, n_max=0.005))     # FoundationPose stand-in
        plate_up = ObsTerm(func=obs_plate_up)
        target = ObsTerm(func=obs_target)
        progress = ObsTerm(func=obs_progress)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    plate_friction = EventTerm(func=mdp.randomize_rigid_body_material, mode="startup",
                               params={"asset_cfg": SceneEntityCfg("plate"), "static_friction_range": (1.0, 1.0),
                                       "dynamic_friction_range": (0.9, 0.9), "restitution_range": (0.0, 0.0), "num_buckets": 1})
    finger_friction = EventTerm(func=mdp.randomize_rigid_body_material, mode="startup",
                                params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_finger"), "static_friction_range": (1.0, 1.0),
                                        "dynamic_friction_range": (0.9, 0.9), "restitution_range": (0.0, 0.0), "num_buckets": 1})
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_plate = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
                            params={"pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "yaw": (-math.pi, math.pi)},
                                    "velocity_range": {}, "asset_cfg": SceneEntityCfg("plate")})
    reset_state = EventTerm(func=reset_stage_state, mode="reset")
    ready_start = (EventTerm(func=reset_ready_start, mode="reset", params={"frac": READY_START_FRAC})
                   if READY_START_FRAC > 0 else None)


@configclass
class RewardsCfg:
    # stage 1 (always on): reach the Human-confirmed open-gripper standoff pose
    reach = RewTerm(func=r_reach, params={"std": 0.2}, weight=1.0)
    reach_fine = RewTerm(func=r_reach, params={"std": 0.02}, weight=1.5)
    tool = RewTerm(func=r_tool, weight=0.5)
    finger = RewTerm(func=r_finger, weight=0.5)
    axis = RewTerm(func=r_axis, weight=0.5)
    open_before_grasp = RewTerm(func=r_open_before_grasp, weight=-1.0)
    success = RewTerm(func=r_success, weight=3000.0)          # x dt (1/30) = +100 at the success step
    # safety / smoothness (always on)
    shield = RewTerm(func=r_shield, weight=-1.0)
    collision = RewTerm(func=collision_force, weight=-1500.0)  # x dt = -50, episode ends
    knocked = RewTerm(func=r_plate_knocked, weight=-3.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001,
                        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-6]"])})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=t_success)
    collision = DoneTerm(func=t_collision)
    plate_lost = DoneTerm(func=t_plate_lost)


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=4096, env_spacing=3.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = EPISODE_S[STAGE]
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 16
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 1024   # 16k overflowed with 4096 envs (2026-10-03 hang)
        # stage-specific terms (stage k keeps everything from the earlier stages)
        if V7:
            self.rewards.tool_angle = RewTerm(func=r_angle, params={"key": "tool_match", "std_deg": 20.0}, weight=1.5)
            self.rewards.finger_angle = RewTerm(func=r_angle, params={"key": "finger_match", "std_deg": 20.0}, weight=1.5)
        if STAGE >= 2:
            self.rewards.reach.weight = 0.5
            self.rewards.reach_fine.weight = 0.5
            self.rewards.open_before_grasp.weight = -0.5
            self.rewards.grasp_reach = RewTerm(func=r_grasp_reach, weight=2.0)
            if V5:
                # v4: ready -> rim was a reward valley (~1.56 at ready, ~1.22 halfway, 2.5 at the rim per second) because
                # reach_fine/axis keep pulling to the ready point; drop them and add a wide rim term so every step forward pays
                self.rewards.reach_fine.weight = 0.0
                self.rewards.axis.weight = 0.0
                self.rewards.grasp_reach_coarse = RewTerm(func=r_grasp_reach, params={"std": 0.10}, weight=1.0)
            self.rewards.close_at_grasp = RewTerm(func=r_close_at_grasp, weight=1.0)
            self.rewards.lift = RewTerm(func=r_lift, weight=3.0)
            if V6:
                self.rewards.level_hold = RewTerm(func=r_level_hold, weight=1.0)
            if V10:
                self.rewards.tcp_rise = RewTerm(func=r_tcp_rise, weight=3.0)
            self.rewards.success.weight = 6000.0          # x dt = +200
        if STAGE >= 3:
            self.rewards.carry = RewTerm(func=r_carry, params={"std": 0.25}, weight=2.0)
            self.rewards.carry_fine = RewTerm(func=r_carry, params={"std": 0.03}, weight=2.0)
            self.rewards.lower = RewTerm(func=r_lower, weight=3.0)
            self.rewards.release = RewTerm(func=r_release, weight=3.0)
            self.rewards.success.weight = 9000.0          # x dt = +300
        # steel frame (measured uprights/rails, calibration/workcell.frame_layout): physical colliders in every env
        import json
        import sys
        calib = Path(__file__).resolve().parents[1] / "calibration"
        sys.path.insert(0, str(calib))
        import workcell as wc
        cell = dict(wc.FRAME_CELL)
        cell.update({k: v for k, v in json.loads((calib / "data" / "real_cell_measured.json").read_text()).items()
                     if not k.startswith("_")})
        for name, centre, size in wc.frame_layout(cell):
            if name == "table":
                continue
            setattr(self.scene, f"Frame_{name}", AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Frame_{name}", init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(centre)),
                spawn=sim_utils.CuboidCfg(size=tuple(size), collision_props=sim_utils.CollisionPropertiesCfg(),
                                          visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.74, 0.77)))))


@configclass
class PickPlaceEnvCfg_PLAY(PickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (1.9, 1.5, 1.3)
        self.viewer.lookat = (0.9, -0.2, 0.1)


gym.register(id="Pick-Place-M1013-2F85-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv",
             kwargs={"env_cfg_entry_point": f"{__name__}:PickPlaceEnvCfg",
                     "rsl_rl_cfg_entry_point": "pick_place_agent:PickPlacePPORunnerCfg"})
gym.register(id="Pick-Place-M1013-2F85-Play-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv",
             kwargs={"env_cfg_entry_point": f"{__name__}:PickPlaceEnvCfg_PLAY",
                     "rsl_rl_cfg_entry_point": "pick_place_agent:PickPlacePPORunnerCfg"})
