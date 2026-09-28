"""Isaac Sim standalone: capture RGB + flange poses for hand-eye calibration
end-to-end validation (see collab/CONTEXT.md, collab/LOG.md 2026-09-28).

Setup:
- Doosan m1013 imported from URDF (dsr_description2/urdf/m1013_isaac_sim.urdf),
  base fixed at world origin (T_base_world = identity).
- A generated ChArUco board is rigidly attached to the flange (tool0) -> this
  is T_ee_board (unknown to the calibration, solved as a byproduct).
- A fixed external camera (RtxCamera) is placed at a known ground-truth pose.
  USD camera convention is -Z forward / +Y up; OpenCV is +Z forward / +Y down.
  The two differ by a fixed 180deg rotation about local X (T_conv below).

Run (isaacsim's own python, not a conda env):
  /home/panhong/isaacsim/python.sh calibration/isaacsim_capture_kimm.py \
      --num-poses 20 --out-dir calibration/sim_capture

Output in --out-dir: board.png, board_config.json, gt.json (T_base_cam GT,
camera intrinsics K), poses.json (per-pose T_base_ee) and img_%03d.png.
isaacsim_verify_kimm.py (run separately, in an env with a recent OpenCV) then
detects the board, solves hand-eye, and compares to gt.json.
"""
import argparse
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as Rot

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "sim_capture"))
parser.add_argument("--num-poses", type=int, default=20)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--flip-v", action="store_true", help="flip board texture V (use if self-check fails)")
parser.add_argument("--max-samples", type=int, default=4000, help="rejection-sampling budget")
args, _ = parser.parse_known_args()

os.makedirs(args.out_dir, exist_ok=True)
rng = np.random.default_rng(args.seed)

# --- Board (generated with the isaacsim-bundled cv2, no conda env needed) ---
import cv2
import cv2.aruco as aruco

SQUARES_X, SQUARES_Y = 7, 5
SQUARE_LEN, MARKER_LEN = 0.035, 0.026  # meters
PX_PER_SQUARE = 120
board_w, board_h = SQUARES_X * SQUARE_LEN, SQUARES_Y * SQUARE_LEN

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
charuco_board = aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LEN, MARKER_LEN, aruco_dict)
board_img = charuco_board.generateImage((SQUARES_X * PX_PER_SQUARE, SQUARES_Y * PX_PER_SQUARE))
board_png_path = os.path.join(args.out_dir, "board.png")
cv2.imwrite(board_png_path, board_img)

with open(os.path.join(args.out_dir, "board_config.json"), "w") as f:
    json.dump({
        "squares_x": SQUARES_X, "squares_y": SQUARES_Y,
        "square_length": SQUARE_LEN, "marker_length": MARKER_LEN,
        "dictionary": "DICT_5X5_100",
    }, f, indent=2)

# --- Isaac Sim ---
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
from urdf_usd_converter import Converter

from isaacsim.core.experimental.prims import Articulation, XformPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
from pxr import UsdPhysics

stage.DefinePrim("/World", "Xform")
# We only need kinematics for this capture (no drives are authored on the imported
# joints, and the converter's own PhysicsScene isn't pulled in by AddReference since
# it lives outside the referenced defaultPrim) -> define our own scene with gravity
# off so set_dof_positions() teleports hold instead of the arm sagging/falling.
physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
physics_scene.CreateGravityMagnitudeAttr(0.0)

ground = stage.DefinePrim("/World/GroundPlane", "Xform")
UsdGeom.Plane.Define(stage, "/World/GroundPlane/Plane").GetAxisAttr().Set("Z")

dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.CreateIntensityAttr(800.0)
distant = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
distant.CreateIntensityAttr(1500.0)
distant.AddRotateXYZOp().Set(Gf.Vec3f(-40, 20, 0))

# --- Import Doosan m1013 from URDF ---
DSR_DESC_DIR = "/home/panhong/pan/doosan-robot2/dsr_description2"
URDF_PATH = os.path.join(DSR_DESC_DIR, "urdf", "m1013_isaac_sim.urdf")
USD_OUT_DIR = os.path.join(args.out_dir, "usd")
os.makedirs(USD_OUT_DIR, exist_ok=True)

converter = Converter(ros_packages=[{"name": "dsr_description2", "path": DSR_DESC_DIR}])
result_asset_path = converter.convert(URDF_PATH, USD_OUT_DIR)
if not result_asset_path:
    raise RuntimeError("URDF -> USD conversion failed")
print(f"[capture] converted robot USD: {result_asset_path.path}")

ROBOT_PATH = "/World/m1013"
robot_prim = stage.DefinePrim(ROBOT_PATH, "Xform")
robot_prim.GetReferences().AddReference(result_asset_path.path)
robot_xform = XformPrim(ROBOT_PATH, reset_xform_op_properties=True)
robot_xform.set_world_poses(positions=np.array([[0.0, 0.0, 0.0]]), orientations=np.array([[1.0, 0.0, 0.0, 0.0]]))

# tool0 may be nested at an arbitrary depth inside the referenced asset -> search for it.
tool0_path = None
for prim in Usd.PrimRange(robot_prim):
    if prim.GetName() == "tool0":
        tool0_path = str(prim.GetPath())
        break
if tool0_path is None:
    raise RuntimeError("could not find 'tool0' prim after import; check converted USD hierarchy")
print(f"[capture] tool0 prim: {tool0_path}")

# The URDF->USD converter tags almost every mesh (visual and collision alike) as
# purpose="guide", which RTX render skips by default -> force everything visible.
n_fixed = 0
for prim in Usd.PrimRange(robot_prim):
    imageable = UsdGeom.Imageable(prim)
    if imageable and imageable.GetPurposeAttr().Get() == UsdGeom.Tokens.guide:
        imageable.GetPurposeAttr().Set(UsdGeom.Tokens.default_)
        n_fixed += 1
print(f"[capture] reset purpose=guide -> default on {n_fixed} prims")

# --- ChArUco board rigidly attached to the flange ---
# NOTE: prims added at runtime as children of the *referenced* robot subtree (e.g.
# under tool0_path) silently fail to render here (confirmed empirically: a debug
# cube under /World rendered fine, an identical one under tool0 did not) -> the
# board is a separate top-level prim whose world pose we drive every frame to
# track tool0 (see update_board_pose()) instead of being parented under it.
BOARD_OFFSET_Z = 0.03  # stand-off from the flange face, meters
BOARD_PATH = "/World/CharucoBoard"


def make_board_prim(flip_v: bool):
    board_path = BOARD_PATH
    if stage.GetPrimAtPath(board_path):
        stage.RemovePrim(board_path)
    # A flat single-quad UsdGeom.Mesh silently failed to render here (even with no
    # material, even top-level) while a UsdGeom.Cube rendered fine -> use a thin
    # Cube as the board instead. Its 6 faces get the same default box-mapped UVs;
    # we only care about the one face facing the camera.
    w, h = board_w, board_h
    thickness = 0.002
    mesh = UsdGeom.Cube.Define(stage, board_path)
    mesh.CreateSizeAttr(1.0)
    mesh.CreateDoubleSidedAttr(True)
    # NOTE: scale is (re)applied in make_board_prim's caller, AFTER the XformPrim
    # wrapper is (re)created with reset_xform_op_properties=True, which wipes any
    # xformOps set here -> setting it here alone silently gets discarded.

    mat_path = f"{board_path}/Material"
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    tex = UsdShade.Shader.Define(stage, f"{mat_path}/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(os.path.abspath(board_png_path)))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    reader = UsdShade.Shader.Define(stage, f"{mat_path}/StReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return board_path


board_path = make_board_prim(args.flip_v)
board_xform = XformPrim(BOARD_PATH, reset_xform_op_properties=True)
board_xform.set_local_scales(np.array([[board_w, board_h, 0.002]]))


# --- Articulation ---
robot = Articulation(ROBOT_PATH)
timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(5):
    simulation_app.update()

joint_names = robot.dof_names
print(f"[capture] dof names: {joint_names}")
JOINT_LIMITS_DEG = None  # read back from USD instead of hardcoding, see below
dof_lo, dof_hi = robot.get_dof_limits()
dof_lo = np.asarray(dof_lo).reshape(-1)
dof_hi = np.asarray(dof_hi).reshape(-1)
print(f"[capture] dof limits (rad): lo={dof_lo} hi={dof_hi}")


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def get_tool0_pose():
    xf = XformPrim(tool0_path)
    pos, quat = xf.get_world_poses()
    pos = np.asarray(pos).reshape(3)
    quat = np.asarray(quat).reshape(4)  # (w, x, y, z)
    R = Rot.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    return make_T(R, pos)


def update_board_pose(T_base_ee):
    T_base_board = T_base_ee @ make_T(np.eye(3), [0, 0, BOARD_OFFSET_Z])
    quat_xyzw = Rot.from_matrix(T_base_board[:3, :3]).as_quat()
    quat_wxyz = np.array([[quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]])
    board_xform.set_world_poses(positions=T_base_board[:3, 3].reshape(1, 3), orientations=quat_wxyz)


def set_joint_positions(q):
    robot.set_dof_positions(np.asarray(q, dtype=np.float32).reshape(1, -1))
    for _ in range(4):
        simulation_app.update()
    update_board_pose(get_tool0_pose())
    simulation_app.update()


# Reference pose: joints 2/3 bent to bring the flange to a reachable position
# in front of the (yet to be placed) camera; joints 4/5/6 stay neutral here but
# are swept over their full range during sampling for rotation diversity.
Q_REF = np.array([0.0, -1.2, 1.4, 0.0, 0.3, 0.0])
set_joint_positions(Q_REF)
T_base_ee_ref = get_tool0_pose()
board_ref_pos = (T_base_ee_ref @ np.array([0, 0, 0.03, 1]))[:3]
print(f"[capture] reference board position: {board_ref_pos}")

# --- Fixed external camera at a known ground-truth pose (eye-to-hand) ---
CAM_POS = board_ref_pos + np.array([0.55, -0.35, 0.30])
LOOK_AT = board_ref_pos
UP = np.array([0.0, 0.0, 1.0])

fwd = LOOK_AT - CAM_POS
fwd /= np.linalg.norm(fwd)
right = np.cross(fwd, UP)
right /= np.linalg.norm(right)
up = np.cross(right, fwd)
# USD camera looks down -Z with +Y up -> columns are (right, up, -fwd)
R_base_camUSD = np.stack([right, up, -fwd], axis=1)
quat_xyzw = Rot.from_matrix(R_base_camUSD).as_quat()
quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

WIDTH, HEIGHT = 640, 480
RESOLUTION = (WIDTH, HEIGHT)  # kept in (width, height) order for our own K/margin math below
FOCAL_LEN_MM, HORIZ_APERTURE_MM = 10.5, 20.955
VERT_APERTURE_MM = HORIZ_APERTURE_MM * HEIGHT / WIDTH

cam_prim = RtxCamera.create(
    path="/World/EyeToHandCamera",
    tick_rate=30.0,
    positions=np.array([CAM_POS]),
    orientations=np.array([quat_wxyz]),
)
cam_prim.camera.set_focal_lengths([FOCAL_LEN_MM])
cam_prim.camera.set_apertures(horizontal_apertures=[HORIZ_APERTURE_MM], vertical_apertures=[VERT_APERTURE_MM])
camera = CameraSensor(cam_prim, resolution=(HEIGHT, WIDTH), annotators=["rgb", "distance_to_image_plane"])  # CameraSensor wants (H, W)

fx = FOCAL_LEN_MM / HORIZ_APERTURE_MM * RESOLUTION[0]
fy = FOCAL_LEN_MM / VERT_APERTURE_MM * RESOLUTION[1]
cx, cy = RESOLUTION[0] / 2.0, RESOLUTION[1] / 2.0
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

R_conv = Rot.from_euler("x", 180, degrees=True).as_matrix()  # USD cam axes -> OpenCV cam axes

T_base_camUSD = make_T(R_base_camUSD, CAM_POS)
T_base_cam_gt = T_base_camUSD @ make_T(R_conv, np.zeros(3))

with open(os.path.join(args.out_dir, "gt.json"), "w") as f:
    json.dump({"T_base_cam": T_base_cam_gt.tolist(), "K": K.tolist(), "resolution": RESOLUTION}, f, indent=2)

for _ in range(10):
    simulation_app.update()


def render_rgb():
    for _ in range(5):
        simulation_app.update()
    rgb, _ = camera.get_data("rgb")
    if hasattr(rgb, "numpy"):
        rgb = rgb.numpy()
    return np.asarray(rgb)


def board_visibility_score(T_base_ee):
    T_ee_board_nominal = make_T(np.eye(3), [0, 0, BOARD_OFFSET_Z])
    T_base_board = T_base_ee @ T_ee_board_nominal
    board_center = T_base_board[:3, 3]
    board_normal = T_base_board[:3, :3] @ np.array([0, 0, 1])
    to_cam = CAM_POS - board_center
    dist = np.linalg.norm(to_cam)
    if dist < 0.25 or dist > 1.6:
        return None
    cos_ang = np.dot(board_normal, to_cam) / dist
    facing_deg = np.degrees(np.arccos(np.clip(abs(cos_ang), -1, 1)))
    if facing_deg > 75:
        return None
    # project corners into the sim camera to check they land inside the image
    T_cam_board = np.linalg.inv(T_base_cam_gt) @ T_base_board
    hw, hh = board_w / 2, board_h / 2
    corners = np.array([[-hw, -hh, 0], [hw, -hh, 0], [hw, hh, 0], [-hw, hh, 0]])
    cam_pts = (T_cam_board[:3, :3] @ corners.T).T + T_cam_board[:3, 3]
    if np.any(cam_pts[:, 2] < 0.2):
        return None
    uv = (K @ cam_pts.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    margin = 0.08
    if np.any(uv[:, 0] < RESOLUTION[0] * margin) or np.any(uv[:, 0] > RESOLUTION[0] * (1 - margin)):
        return None
    if np.any(uv[:, 1] < RESOLUTION[1] * margin) or np.any(uv[:, 1] > RESOLUTION[1] * (1 - margin)):
        return None
    return facing_deg


# --- Self-check: does the board texture read correctly through the sim camera? ---
def self_check():
    set_joint_positions(Q_REF)
    T_base_ee_chk = get_tool0_pose()
    board_pos_chk = (T_base_ee_chk @ np.array([0, 0, BOARD_OFFSET_Z, 1]))[:3]
    dist_chk = np.linalg.norm(board_pos_chk - CAM_POS)
    print(f"[capture] self-check: tool0 pos={T_base_ee_chk[:3, 3]}, board pos={board_pos_chk}, cam-board dist={dist_chk:.3f} m")
    rgb = render_rgb()
    depth, _ = camera.get_data("distance_to_image_plane")
    if hasattr(depth, "numpy"):
        depth = depth.numpy()
    depth = np.asarray(depth).reshape(HEIGHT, WIDTH)
    finite = depth[np.isfinite(depth)]
    print(f"[capture] depth stats: min={finite.min() if finite.size else 'nan'} max={finite.max() if finite.size else 'nan'} n_finite={finite.size}/{depth.size}")
    depth_vis = np.clip((depth - 0.1) / (3.0 - 0.1), 0, 1)
    depth_vis = np.nan_to_num(depth_vis, nan=1.0)
    cv2.imwrite(os.path.join(args.out_dir, "self_check_depth.png"), (255 * (1 - depth_vis)).astype(np.uint8))
    bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
    detector = aruco.CharucoDetector(charuco_board)
    ch_corners, ch_ids, _, _ = detector.detectBoard(bgr)
    n = 0 if ch_ids is None else len(ch_ids)
    return n, bgr


n_detected, dbg_img = self_check()
print(f"[capture] self-check (flip_v={args.flip_v}): {n_detected} charuco corners detected")
if n_detected < 6:
    print("[capture] self-check weak/failed, retrying with flipped V ...")
    board_path = make_board_prim(not args.flip_v)
    board_xform = XformPrim(BOARD_PATH, reset_xform_op_properties=True)
    board_xform.set_local_scales(np.array([[board_w, board_h, 0.002]]))
    for _ in range(5):
        simulation_app.update()
    n_detected, dbg_img = self_check()
    print(f"[capture] self-check (flip_v={not args.flip_v}): {n_detected} charuco corners detected")
cv2.imwrite(os.path.join(args.out_dir, "self_check.png"), dbg_img)
if n_detected < 6:
    print(
        "[capture] board self-check failed under both texture orientations; "
        f"see {args.out_dir}/self_check.png for debugging"
    )
    if not args.headless:
        print("[capture] GUI kept open for manual inspection -> close the Isaac Sim window when done.")
        print("[capture] In the Stage panel, select /World/CharucoBoard, then press F to frame/snap the viewport to it.")
        while simulation_app.is_running():
            simulation_app.update()
    simulation_app.close()
    raise SystemExit(1)

# --- Rejection-sample poses and capture ---
# Joints 1-3 (position) only jitter around the reference pose so the board stays
# roughly in the camera's view; joints 4-6 (wrist/orientation) sweep their full
# range since sim_validate.py found rotation diversity matters most for accuracy.
poses = []
n_tried = 0
n_needed = args.num_poses
pos_jitter = np.array([0.4, 0.3, 0.3])
lo = np.concatenate([Q_REF[:3] - pos_jitter, dof_lo[3:]])
hi = np.concatenate([Q_REF[:3] + pos_jitter, dof_hi[3:]])
lo = np.clip(lo, dof_lo, dof_hi)
hi = np.clip(hi, dof_lo, dof_hi)

while len(poses) < n_needed and n_tried < args.max_samples:
    n_tried += 1
    q = rng.uniform(lo, hi)
    T_base_ee = None
    set_joint_positions(q)
    T_base_ee = get_tool0_pose()
    facing_deg = board_visibility_score(T_base_ee)
    if facing_deg is None:
        continue
    rgb = render_rgb()
    idx = len(poses)
    img_path = os.path.join(args.out_dir, f"img_{idx:03d}.png")
    cv2.imwrite(img_path, cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR))
    poses.append({"img": os.path.basename(img_path), "T_base_ee": T_base_ee.tolist(), "facing_deg": facing_deg})
    print(f"[capture] pose {idx + 1}/{n_needed} accepted (facing {facing_deg:.1f} deg, tried {n_tried})")

with open(os.path.join(args.out_dir, "poses.json"), "w") as f:
    json.dump(poses, f, indent=2)

print(f"[capture] done: {len(poses)}/{n_needed} poses saved to {args.out_dir} (sampled {n_tried} candidates)")
if len(poses) < n_needed:
    print("[capture] WARNING: could not fill the requested pose count within --max-samples")

simulation_app.close()
