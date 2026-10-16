"""Isaac Sim digital twin of the real cell: M1013 (home pose) + Robotiq 2F-85, the 029 plate on its box, the target box
and the fixed D455 placed by its calibration (T_base_cam_pnp.json).  Isaac Sim python, not conda:

  bash calibration/run.sh sim [--plate-xy X Y] [--free-viewport] [--render out.png] [--trajectory demo.npz]

Kinematic only (no physics): link poses come from robot_model.Kinematics exactly as the real planner uses them.  The
gripper is drawn from the menagerie STL meshes (open pose) on the 20 mm coupling; the TCP length is printed at start.
The viewport shows what the D455 sees unless --free-viewport; --render saves that 1280x720 view and exits (overlay it on a
real frame to check the calibration).  --trajectory replays a simulation-only joint trajectory (joints_rad, fps, ...).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calib_core import DATA, cpath, inv_T, load_config, make_T
from robot_model import DEFAULT_URDF, HOME_JOINTS_DEG, Kinematics, frame_layout, measured_cell

HERE = Path(__file__).resolve().parent
C = load_config()
PLATE_OBJ = cpath(C['paths']['plate_obj'])
D455_OPTICS = dict(focal_mm=1.88, h_aperture_mm=1280 * 1.88 / 647.7955, v_aperture_mm=720 * 1.88 / 646.8864)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--joints-deg', type=float, nargs=6, default=list(HOME_JOINTS_DEG))
    p.add_argument('--coupling-m', type=float, default=C['robot']['coupling_m'], help='AGC-DOO-KIT-85 coupling thickness (Human confirmed 20 mm, 2026-10-02)')
    p.add_argument('--plate-xy', type=float, nargs=2, default=[1.215, -0.009], help='plate centre in the base frame, m (2026-10-06 13:20, raised plate: circle fit to the rim edge in the fixed D455 depth with T_base_cam_pnp; before raising: (1.191, -0.033) from the wrist camera)')
    p.add_argument('--plate-lift-m', type=float, default=0.1146, help='support height = Human 14 cm plate-top height (2026-10-06) - 25.4 mm scanned plate; fixed-D455 depth: plate floor ~110 mm, rim top ~130-140 mm (before: 0.0546)')
    p.add_argument('--target-offset-xy', type=float, nargs=2, default=[0.0, -0.45], help='second (target) box offset from the plate box, base frame, m (positions swapped by Human: plate at +y, target at -y)')
    p.add_argument('--plate-box-size-m', type=float, nargs=2, default=[0.08, 0.08], help='support box X Y size (PLACEHOLDER)')
    p.add_argument('--t_base_cam', type=Path, default=None,
                   help='fixed-camera calibration json (default: newest data/real_handeye/*/T_base_cam_pnp.json, 2026-10-06 PnP result)')
    p.add_argument('--cam-xyz', type=float, nargs=3, default=None, help='override: D455 position in the base frame (disables --t_base_cam)')
    p.add_argument('--cam-target', type=float, nargs=3, default=None, help='override: point on the optical axis (with --cam-xyz)')
    p.add_argument('--cell-json', type=Path, default=cpath(C['paths']['cell_json']))
    p.add_argument('--render', type=Path, default=None, help='save the D455 colour view (1280x720, same FOV as the real camera) to this PNG and exit')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--free-viewport', action='store_true', help='keep the perspective viewport for inspecting and rotating scene prims')
    p.add_argument('--trajectory', type=Path, default=None, help='replay a simulation-only joint trajectory from an NPZ file')
    p.add_argument('--exit-after-run', action='store_true')
    return p.parse_args()


def world_T(stage, path):
    from pxr import UsdGeom
    return np.array(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)).T


def set_T(prim, T):
    from pxr import Gf, UsdGeom
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(T.T.tolist()))


def add_box(stage, path, centre, size, color, parent_T=None):
    from pxr import Gf, UsdGeom
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    T = make_T(np.eye(3), centre)
    set_T(cube.GetPrim(), T)
    UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(Gf.Vec3d(*size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])


GRIPPER_DIR = cpath(C['paths']['gripper_dir'])               # DeepMind mujoco_menagerie robotiq_2f85 (BSD-2, ROS-Industrial)
PINCH_Z = 0.145                                              # <site name="pinch"> in base frame


def _quat_R(q):
    w, x, y, z = (np.array(q, float) / np.linalg.norm(q))
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def read_stl(path):
    """Binary or ASCII STL -> (triangles (n,3,3))."""
    data = Path(path).read_bytes()
    n = int(np.frombuffer(data[80:84], '<u4')[0])
    if 84 + 50 * n == len(data):
        rec = np.frombuffer(data[84:], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), count=n)
        return rec['v'].astype(float)
    pts = [[float(t) for t in ln.split()[1:4]] for ln in data.decode().splitlines() if ln.strip().startswith('vertex')]
    return np.array(pts).reshape(-1, 3, 3)


def add_gripper(stage, link6_path, coupling_m):
    """Robotiq 2F-85 from the menagerie MJCF (open pose: all joints 0), kinematic meshes under link_6.
    The gripper origin sits coupling_m along flange +z (AGC-DOO-KIT-85 thickness, ASSUMED).  Returns the TCP
    (pinch point) distance from the flange."""
    import xml.etree.ElementTree as ET
    from pxr import Gf, UsdGeom
    root = ET.parse(GRIPPER_DIR / '2f85.xml').getroot()
    mats = {m.get('name'): [float(v) for v in m.get('rgba').split()[:3]] for m in root.iter('material')}
    meshes = {m.get('file')[:-4]: m.get('file') for m in root.iter('mesh') if m.get('file')}
    UsdGeom.Xform.Define(stage, link6_path + '/gripper_2F85')
    count = [0]

    def walk(body, T_parent, parent_path):
        pos = [float(v) for v in body.get('pos', '0 0 0').split()]
        quat = [float(v) for v in body.get('quat', '1 0 0 0').split()]
        T = T_parent @ make_T(_quat_R(quat), pos)                 # all joints at 0 (open)
        path = f'{parent_path}/{body.get("name")}'          # flat layout: T is already cumulative (relative to the mount)
        UsdGeom.Xform.Define(stage, path)
        set_T(stage.GetPrimAtPath(path), T)
        for g in body.findall('geom'):
            if g.get('class') != 'visual' or g.get('mesh') is None:
                continue
            tri = read_stl(GRIPPER_DIR / 'assets' / meshes[g.get('mesh')]) * 0.001
            mesh = UsdGeom.Mesh.Define(stage, f'{path}/{g.get("mesh")}_{count[0]}')
            count[0] += 1
            mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in tri.reshape(-1, 3)])
            mesh.CreateFaceVertexCountsAttr([3] * len(tri))
            mesh.CreateFaceVertexIndicesAttr(list(range(3 * len(tri))))
            mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*mats.get(g.get('material'), [0.15, 0.15, 0.15]))])
        for child in body.findall('body'):
            walk(child, T, parent_path)
        return

    base_mount = root.find('worldbody').find('body')
    walk(base_mount, make_T(np.eye(3), [0, 0, coupling_m]), link6_path + '/gripper_2F85')
    return coupling_m + 0.007 + 0.0038 + PINCH_Z


def add_plate(stage, path, obj_path, xy, table_z):
    from pxr import Gf, UsdGeom
    verts, faces = [], []
    for line in open(obj_path):
        if line.startswith('v '):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            idx = [int(tok.split('/')[0]) - 1 for tok in line.split()[1:]]
            for k in range(1, len(idx) - 1):                    # fan-triangulate
                faces.append((idx[0], idx[k], idx[k + 1]))
    v = np.array(verts)
    lo, hi = v.min(axis=0), v.max(axis=0)
    centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])     # bottom centre of the bounding box
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in (v - centre)])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([int(i) for f in faces for i in f])
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.75, 0.05, 0.08)])
    set_T(mesh.GetPrim(), make_T(np.eye(3), [xy[0], xy[1], table_z]))
    return hi - lo


def newest_t_base_cam():
    found = sorted(f for f in (DATA / 'real_handeye').glob('*/T_base_cam_pnp.json')
                   if not (f.parent / 'INVALID').exists())          # (10/16) same rule as calib_core.latest_t_base_cam
    if not found:
        raise SystemExit('no T_base_cam_pnp.json in data/real_handeye/; pass --t_base_cam or --cam-xyz/--cam-target')
    return found[-1]


def add_d455(stage, path, position, target, R_usd=None):
    """Fixed D455: USD camera looks along local -Z with +Y up; housing sits behind the lens."""
    from pxr import Gf, UsdGeom
    z_cam = np.asarray(position, float) - np.asarray(target, float)         # local +z points away from the target
    z_cam /= np.linalg.norm(z_cam)
    x_cam = np.cross([0, 0, 1.0], z_cam)
    x_cam /= np.linalg.norm(x_cam)
    R = np.column_stack([x_cam, np.cross(z_cam, x_cam), z_cam]) if R_usd is None else np.asarray(R_usd, float)
    root = UsdGeom.Xform.Define(stage, path)
    set_T(root.GetPrim(), make_T(R, position))
    camera = UsdGeom.Camera.Define(stage, path + '/color')
    camera.CreateFocalLengthAttr(D455_OPTICS['focal_mm'])
    camera.CreateHorizontalApertureAttr(D455_OPTICS['h_aperture_mm'])
    camera.CreateVerticalApertureAttr(D455_OPTICS['v_aperture_mm'])
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    add_box(stage, path + '/housing', [0, 0, 0.015], [0.124, 0.029, 0.026], (0.05, 0.05, 0.05))
    return R


def main():
    args = parse_args()
    cell = measured_cell(args.cell_json)
    kin = Kinematics(DEFAULT_URDF)
    q = np.radians(args.joints_deg)
    T_flange = kin.fk(q)
    tool_length = args.coupling_m + 0.007 + 0.0038 + PINCH_Z
    tcp = T_flange[:3, 3] + tool_length * T_flange[:3, 2]
    print(f'joints {args.joints_deg} deg -> flange {np.round(T_flange[:3, 3], 3).tolist()} m, '
          f'TCP/pinch ({tool_length:.3f} m) {np.round(tcp, 3).tolist()} m', flush=True)

    from isaacsim import SimulationApp
    app = SimulationApp({'headless': args.headless, 'width': 1600, 'height': 1000})
    try:
        import omni.kit.app
        import omni.kit.commands
        import omni.usd
        from isaacsim.core.utils.viewports import set_camera_view
        from pxr import Gf, UsdGeom, UsdLux, UsdPhysics
        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, '/World')
        table_z = -cell['base_plate_thickness']
        for name, centre, size in frame_layout(cell):
            color = (0.20, 0.22, 0.25) if name == 'table' else (0.72, 0.74, 0.77)
            add_box(stage, f'/World/Cell/{name}', centre, size, color)
        add_box(stage, '/World/Cell/base_plate', [0, 0, table_z / 2], [0.30, 0.30, -table_z], (0.5, 0.5, 0.5))
        UsdLux.DomeLight.Define(stage, '/World/DomeLight').CreateIntensityAttr(800.0)
        key = UsdLux.DistantLight.Define(stage, '/World/KeyLight')
        key.CreateIntensityAttr(1200.0)
        key.AddRotateXYZOp().Set(Gf.Vec3f(-45, 25, 25))

        ext = omni.kit.app.get_app().get_extension_manager()
        if not ext.is_extension_enabled('isaacsim.asset.importer.urdf'):
            ext.set_extension_enabled_immediate('isaacsim.asset.importer.urdf', True)
            for _ in range(5):
                app.update()
        ok, cfg = omni.kit.commands.execute('URDFCreateImportConfig')
        cfg.merge_fixed_joints, cfg.convex_decomp, cfg.fix_base = False, False, True
        cfg.import_inertia_tensor, cfg.collision_from_visuals = True, False
        ok, robot_path = omni.kit.commands.execute('URDFParseAndImportFile', urdf_path=str(DEFAULT_URDF), import_config=cfg)
        if not ok:
            raise RuntimeError('M1013 URDF import failed')
        for prim in stage.Traverse():                                   # kinematic replay only
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        names = set(kin.frames(np.zeros(6)))
        links = {p.GetName(): p for p in stage.Traverse() if str(p.GetPath()).startswith(robot_path + '/') and p.GetName() in names}
        if len(links) != 7:
            raise RuntimeError(f'expected 7 links, got {sorted(links)}')
        base = world_T(stage, str(links['base_link'].GetPath()))
        def show_joints(joints):
            for name, T in kin.frames(joints).items():
                prim = links[name]
                set_T(prim, inv_T(world_T(stage, str(prim.GetParent().GetPath()))) @ base @ T)

        show_joints(q)
        link6_path = str(links['link_6'].GetPath())
        add_gripper(stage, link6_path, args.coupling_m)
        gripper_root = link6_path + '/gripper_2F85'

        def local_matrix(path):
            return np.array(stage.GetPrimAtPath(path).GetAttribute('xformOp:transform').Get()).T

        lift = args.plate_lift_m
        if lift > 0:                                                    # small box under the plate
            add_box(stage, '/World/PlateSupport', [args.plate_xy[0], args.plate_xy[1], table_z + lift / 2],
                    [*args.plate_box_size_m, lift], (0.55, 0.40, 0.25))
        if lift > 0:                                                    # identical empty box = place target
            tx, ty = args.plate_xy[0] + args.target_offset_xy[0], args.plate_xy[1] + args.target_offset_xy[1]
            add_box(stage, '/World/TargetBox', [tx, ty, table_z + lift / 2], [*args.plate_box_size_m, lift], (0.30, 0.55, 0.30))
        size = add_plate(stage, '/World/Plate029', PLATE_OBJ, args.plate_xy, table_z + lift)
        plate_base_T = world_T(stage, '/World/Plate029')
        print(f'plate 029 {np.round(size, 3).tolist()} m at xy {args.plate_xy}', flush=True)
        if args.cam_xyz is None:                                        # calibrated fixed camera (optical frame -> USD camera frame)
            tpath = args.t_base_cam or newest_t_base_cam()
            T_bc = np.array(json.loads(Path(tpath).read_text())['T_base_cam'])
            cam_xyz, cam_R = T_bc[:3, 3], T_bc[:3, :3] @ np.diag([1.0, -1.0, -1.0])
            target = cam_xyz + 2.0 * T_bc[:3, 2]
            print(f'D455 from {tpath}', flush=True)
        else:
            cam_xyz, cam_R = args.cam_xyz, None
            target = args.cam_target or [args.plate_xy[0], args.plate_xy[1], table_z]
        add_d455(stage, '/World/D455', cam_xyz, target, cam_R)
        print(f'D455 at {np.round(cam_xyz, 3).tolist()} looking at {np.round(target, 3).tolist()}', flush=True)

        if args.trajectory:
            set_camera_view(eye=np.array([1.55, -0.75, 0.70]), target=np.array([0.95, -0.02, 0.14]))
        else:
            set_camera_view(eye=np.array([2.6, -1.8, 1.7]), target=np.array([0.8, 0.1, 0.4]))
        for _ in range(20):
            app.update()
        if args.render:
            import omni.replicator.core as rep
            from PIL import Image
            rp = rep.create.render_product('/World/D455/color', (1280, 720))
            ann = rep.AnnotatorRegistry.get_annotator('rgb')
            ann.attach([rp])
            for _ in range(15):
                rep.orchestrator.step()
            Image.fromarray(np.asarray(ann.get_data())[..., :3]).save(args.render)
            print(f'saved {args.render}', flush=True)
            return
        from omni.kit.viewport.utility import get_active_viewport
        if not args.free_viewport:
            get_active_viewport().camera_path = '/World/D455/color'  # the window shows what the D455 sees
        replay = np.load(args.trajectory) if args.trajectory else None
        if replay is not None:
            joints = replay['joints_rad']
            plate_lift = replay['plate_lift_m'] if 'plate_lift_m' in replay else np.zeros(len(joints))
            gripper_body_names = replay['gripper_body_names'].astype(str) if 'gripper_body_names' in replay else np.array([], dtype=str)
            gripper_xforms = replay['gripper_xforms'] if 'gripper_xforms' in replay else None
            gripper_pose_idx = replay['gripper_pose_idx'] if 'gripper_pose_idx' in replay else np.zeros(len(joints), dtype=int)
            phases = replay['phases'].astype(str) if 'phases' in replay else np.full(len(joints), 'replay')
            fps = int(replay['fps'])
            if joints.ndim != 2 or joints.shape[1] != 6 or fps <= 0:
                raise ValueError('trajectory must have joints_rad shape (N, 6) and positive fps')
            if len(plate_lift) != len(joints) or len(phases) != len(joints) or len(gripper_pose_idx) != len(joints):
                raise ValueError('plate_lift_m, gripper_pose_idx, and phases arrays must match joints_rad length')
            if gripper_xforms is not None:
                expected = [gripper_root + '/' + name for name in gripper_body_names]
                missing = [path for path in expected if not stage.GetPrimAtPath(path).IsValid()]
                if missing:
                    raise ValueError(f'trajectory gripper bodies do not match scene: {missing}')
                if gripper_xforms.ndim != 4 or gripper_xforms.shape[1:] != (len(expected), 4, 4):
                    raise ValueError(f'invalid gripper_xforms shape {gripper_xforms.shape}')
                if np.any(gripper_pose_idx < 0) or np.any(gripper_pose_idx >= len(gripper_xforms)):
                    raise ValueError('gripper_pose_idx references a missing articulation pose')
            elif np.any(gripper_pose_idx):
                raise ValueError('gripper_pose_idx requires gripper_xforms articulation data')
            print(f'replaying {len(joints)} joint frames at {fps} fps: {args.trajectory}', flush=True)
            start_time = time.monotonic()
            previous_frame = -1
            previous_phase = None
        if not args.headless and not args.exit_after_run:
            while app.is_running():
                if replay is not None:
                    frame = int((time.monotonic() - start_time) * fps) % len(joints)
                    if frame != previous_frame:
                        show_joints(joints[frame])
                        if gripper_xforms is not None:
                            pose = int(gripper_pose_idx[frame])
                            for i, name in enumerate(gripper_body_names):
                                set_T(stage.GetPrimAtPath(gripper_root + '/' + name), gripper_xforms[pose, i])
                        T_plate = plate_base_T.copy()
                        T_plate[2, 3] += float(plate_lift[frame])
                        set_T(stage.GetPrimAtPath('/World/Plate029'), T_plate)
                        if phases[frame] != previous_phase:
                            print(f'Stage 2 demo phase: {phases[frame]}', flush=True)
                            previous_phase = phases[frame]
                        previous_frame = frame
                app.update()
                time.sleep(0.025)
    except Exception:
        import traceback
        traceback.print_exc()                                   # app.close() may exit the process before Python prints it
        raise
    finally:
        app.close()


if __name__ == '__main__':
    main()
