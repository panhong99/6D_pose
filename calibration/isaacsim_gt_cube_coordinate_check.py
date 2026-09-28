"""Minimal Isaac Sim GT coordinate-transform check.

This is intentionally *not* a hand-eye calibration implementation. It places a
fixed camera and cube in a scene, reads their known world poses, converts the
cube pose into the camera and robot-base frames, and verifies:

    T_base_cube = T_base_camera @ T_camera_cube

Run with Isaac Sim's Python:
  /home/panhong/isaacsim/python.sh \
    calibration/isaacsim_gt_cube_coordinate_check.py
"""

import argparse
import json
import os

import numpy as np

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--log-path",
    default=os.path.join(os.path.dirname(__file__), "gt_cube_coordinate_log.json"),
)
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})

import omni.timeline
import omni.usd
from pxr import Gf, UsdGeom, UsdLux


def make_transform(position, rotation=np.eye(3)):
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(position, dtype=float)
    return transform


def add_xform(stage, path, transform):
    prim = UsdGeom.Xform.Define(stage, path)
    translate = prim.AddTranslateOp()
    translate.Set(Gf.Vec3d(*transform[:3, 3]))
    # The first check uses identity rotations deliberately. Rotation handling is
    # added after the translation/frame convention is confirmed.
    return prim


def main():
    stage = omni.usd.get_context().get_stage()
    stage.DefinePrim("/World", "Xform")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    UsdGeom.Plane.Define(stage, "/World/Ground").GetAxisAttr().Set("Z")
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(500.0)

    # The robot base is the world origin in this first test.
    base = make_transform([0.0, 0.0, 0.0])

    # Fixed head camera and cube. Change only these values for the first test.
    camera = make_transform([1.0, -1.0, 1.2])
    cube = make_transform([0.45, 0.10, 0.05])
    add_xform(stage, "/World/HeadCamera", camera)
    cube_prim = UsdGeom.Cube.Define(stage, "/World/Cube")
    cube_prim.CreateSizeAttr(0.10)
    cube_prim.AddTranslateOp().Set(Gf.Vec3d(*cube[:3, 3]))

    # HeadCamera above is just a plain Xform (position bookkeeping only, not an
    # actual UsdGeom.Camera) -> nothing would be visible there in the viewport.
    # Add a small visible marker at the same position so it can be seen in the GUI.
    cam_marker = UsdGeom.Cube.Define(stage, "/World/HeadCameraMarker")
    cam_marker.CreateSizeAttr(0.06)
    cam_marker.CreateDisplayColorAttr([Gf.Vec3f(1, 0, 0)])
    cam_marker.AddTranslateOp().Set(Gf.Vec3d(*camera[:3, 3]))

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(10):
        simulation_app.update()

    # Poses are written as T_parent_child: pose of child expressed in parent.
    T_world_base = base
    T_world_camera = camera
    T_world_cube = cube
    T_base_camera = np.linalg.inv(T_world_base) @ T_world_camera
    T_base_cube_gt = np.linalg.inv(T_world_base) @ T_world_cube
    T_camera_cube = np.linalg.inv(T_world_camera) @ T_world_cube
    T_base_cube_est = T_base_camera @ T_camera_cube

    position_error_m = np.linalg.norm(T_base_cube_est[:3, 3] - T_base_cube_gt[:3, 3])
    rotation_error_deg = 0.0  # Both test rotations are identity for this first check.

    print("\n=== Isaac Sim GT cube coordinate check ===")
    print("T_base_camera (camera pose in robot-base frame):")
    print(T_base_camera)
    print("T_camera_cube (cube pose in camera frame):")
    print(T_camera_cube)
    print("T_base_cube_gt (GT cube pose in robot-base frame):")
    print(T_base_cube_gt)
    print("T_base_cube_est (T_base_camera @ T_camera_cube):")
    print(T_base_cube_est)
    print(f"robot_base_position_m={T_base_cube_gt[:3, 3].tolist()}")
    print(f"camera_position_in_base_m={T_base_camera[:3, 3].tolist()}")
    print(f"cube_position_in_camera_m={T_camera_cube[:3, 3].tolist()}")
    print(f"cube_position_estimated_in_base_m={T_base_cube_est[:3, 3].tolist()}")
    print(f"position_error_mm={position_error_m * 1000.0:.6f}")
    print(f"rotation_error_deg={rotation_error_deg:.6f}")

    result = {
        "T_world_base": T_world_base.tolist(),
        "T_world_camera": T_world_camera.tolist(),
        "T_world_cube": T_world_cube.tolist(),
        "T_base_camera": T_base_camera.tolist(),
        "T_camera_cube": T_camera_cube.tolist(),
        "T_base_cube_gt": T_base_cube_gt.tolist(),
        "T_base_cube_est": T_base_cube_est.tolist(),
        "position_error_mm": float(position_error_m * 1000.0),
        "rotation_error_deg": rotation_error_deg,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.log_path)), exist_ok=True)
    with open(args.log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"log_path={os.path.abspath(args.log_path)}")

    if not args.headless:
        print("[check] GUI kept open for manual inspection -> close the Isaac Sim window when done.")
        while simulation_app.is_running():
            simulation_app.update()
    simulation_app.close()


if __name__ == "__main__":
    main()
