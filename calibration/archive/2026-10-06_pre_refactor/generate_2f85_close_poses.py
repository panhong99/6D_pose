"""Sample physically articulated Robotiq 2F-85 closing poses with Isaac Sim's MJCF importer.

Run with Isaac Sim's Python. The exported matrices are replayed by sim_home_scene.py while the arm remains
kinematic, so the hand's linkage rotations match the MJCF model instead of translating meshes by hand.
"""
from pathlib import Path

import numpy as np

from isaacsim import SimulationApp

HERE = Path(__file__).resolve().parent
MJCF = HERE.parent / 'assets' / 'robotiq_2f85' / '2f85.xml'
OUT = HERE / 'data' / 'gripper_close_poses_20261006.npz'
BODY_NAMES = (
    'base_mount', 'base',
    'right_driver', 'right_coupler', 'right_spring_link', 'right_follower', 'right_pad', 'right_silicone_pad',
    'left_driver', 'left_coupler', 'left_spring_link', 'left_follower', 'left_pad', 'left_silicone_pad',
)
MAX_DRIVER_DEG = 30.0  # MJCF articulation yields a 24 mm silicone-pad gap, matching the scanned plate thickness.
POSES = 31


def main():
    app = SimulationApp({'headless': True, 'width': 640, 'height': 480})
    try:
        import omni.kit.app
        import omni.kit.commands
        import omni.timeline
        import omni.usd
        from pxr import Sdf, UsdGeom, UsdPhysics

        manager = omni.kit.app.get_app().get_extension_manager()
        if not manager.is_extension_enabled('isaacsim.asset.importer.mjcf'):
            manager.set_extension_enabled_immediate('isaacsim.asset.importer.mjcf', True)
            for _ in range(5):
                app.update()

        context = omni.usd.get_context()
        context.new_stage()
        _, cfg = omni.kit.commands.execute('MJCFCreateImportConfig')
        cfg.set_fix_base(False)
        cfg.set_import_inertia_tensor(True)
        cfg.set_self_collision(False)
        cfg.set_create_physics_scene(True)
        omni.kit.commands.execute('MJCFCreateAsset', mjcf_path=str(MJCF), import_config=cfg, prim_path='/Gripper')
        stage = context.get_stage()
        physics = UsdPhysics.Scene.Get(stage, Sdf.Path('/physicsScene'))
        physics.GetGravityMagnitudeAttr().Set(0.0)

        # The custom viewer mounts the gripper 20 mm beyond link_6; express all sampled bodies in that frame.
        body_prims = {}
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() in BODY_NAMES:
                body_prims[prim.GetName()] = prim
        missing = sorted(set(BODY_NAMES) - body_prims.keys())
        if missing:
            raise RuntimeError(f'MJCF import is missing body links: {missing}')

        drives = []
        for joint_name in ('right_driver_joint', 'left_driver_joint'):
            drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(f'/Gripper/joints/{joint_name}'), 'angular')
            drive.GetStiffnessAttr().Set(50.0)
            drive.GetDampingAttr().Set(5.0)
            drive.GetMaxForceAttr().Set(100.0)
            drives.append(drive)

        def matrix_at(prim):
            return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T

        mount = body_prims['base_mount']
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(30):
            app.update()

        poses = []
        mount_world = matrix_at(mount)
        inv_mount = np.linalg.inv(mount_world)
        mount_in_link6 = np.eye(4)
        mount_in_link6[2, 3] = 0.027  # coupling 20 mm + MJCF base_mount offset 7 mm
        for pose_index, angle in enumerate(np.linspace(0.0, MAX_DRIVER_DEG, POSES)):
            for drive in drives:
                drive.GetTargetPositionAttr().Set(float(angle))  # angular drive target is in degrees
            for _ in range(4):
                app.update()
            mount_world = matrix_at(mount)
            inv_mount = np.linalg.inv(mount_world)
            mats = []
            for name in BODY_NAMES:
                relative_to_mount = inv_mount @ matrix_at(body_prims[name])
                mats.append(mount_in_link6 @ relative_to_mount)
            poses.append(mats)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT, body_names=np.asarray(BODY_NAMES), xforms=np.asarray(poses, dtype=np.float32),
                            driver_degrees=np.linspace(0.0, MAX_DRIVER_DEG, POSES),
                            silicone_gap_mm=np.asarray([76.6, 24.0], dtype=np.float32))
        print(f'SAVED {OUT} | poses={len(poses)} | bodies={len(BODY_NAMES)} | final_driver={MAX_DRIVER_DEG:.1f} deg | final_pad_gap≈24 mm',
              flush=True)
        timeline.stop()
    finally:
        app.close()


if __name__ == '__main__':
    main()
