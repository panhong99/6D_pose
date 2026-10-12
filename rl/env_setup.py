"""Build the assets the RL env needs (rl/assets/), run through Isaac Lab's python.

  ISAAC=~/IsaacLab/isaaclab.sh
  $ISAAC -p rl/env_setup.py urdf              # assets/m1013_2f85.urdf   M1013 + simplified 2F-85 (no Isaac app)
  $ISAAC -p rl/env_setup.py plate             # assets/plate029.usd (physics) + plate029_textured.usda (original texture)
  $ISAAC -p rl/env_setup.py sdf               # assets/cell_sdf.npz      shield distance fields (no Isaac app)
  $ISAAC -p rl/env_setup.py ready --headless  # assets/ready_joints.json  ready pose reached by the env's own diff-IK

urdf  : Robotiq geometry/inertia from mujoco_menagerie robotiq_2f85 (open pose).  The 4-bar linkage is NOT simulated:
        two prismatic finger links (follower + pad meshes, pad-box collisions), 0 = open (85 mm), stroke = closed.
        Gripper origin = coupling_m along link_6 +z; TCP = 0.145 m along the Robotiq base z.
plate : scanned 029 plate, simplified OBJ -> rigid body (plate_mass) with 16-hull convex decomposition (not filled in);
        the textured file references it, hides that mesh and shows demo_data/029_plate/tsdf/textured.obj instead.
sdf   : 1 cm Euclidean distance grids (frame / table) from calibration/robot_model.frame_layout + data/real_cell_measured.json and
        down-sampled M1013 link collision samples (calibration/data/m1013_collision_points.npz).
ready : from home, drive the TCP to the ready point (plate nominal, READY_TOOL_Z / READY_FINGER_AXIS) with the shielded
        IK action until position and angle errors vanish (about 20 s of sim), save the arm joints.
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np

from rl_utils import load_config, quat_to_R, rl_path


# ---------------------------------------------------------------- urdf

def _mk(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def _origin(T):
    R = T[:3, :3]
    sy = np.hypot(R[0, 0], R[1, 0])
    if sy > 1e-9:
        r, p, y = np.arctan2(R[2, 1], R[2, 2]), np.arctan2(-R[2, 0], sy), np.arctan2(R[1, 0], R[0, 0])
    else:
        r, p, y = np.arctan2(-R[1, 2], R[1, 1]), np.arctan2(-R[2, 0], sy), 0.0
    return f'<origin xyz="{T[0, 3]:.6f} {T[1, 3]:.6f} {T[2, 3]:.6f}" rpy="{r:.6f} {p:.6f} {y:.6f}"/>'


def build_urdf(cfg):
    gdir, share = rl_path(cfg["paths"]["gripper_mjcf_dir"]), rl_path(cfg["paths"]["m1013_share"])
    stroke, coupling = cfg["robot"]["finger_stroke"], cfg["robot"]["coupling_m"]
    root = ET.parse(gdir / '2f85.xml').getroot()
    meshfile = {m.get('file')[:-4]: m.get('file') for m in root.iter('mesh') if m.get('file')}
    bodies = {}                                                      # name -> (T in gripper-origin frame, element)

    def walk(body, T_parent):
        T = T_parent @ _mk(quat_to_R([float(v) for v in body.get('quat', '1 0 0 0').split()]),
                           [float(v) for v in body.get('pos', '0 0 0').split()])
        bodies[body.get('name')] = (T, body)
        for c in body.findall('body'):
            walk(c, T)
    walk(root.find('worldbody').find('body'), np.eye(4))

    def mesh_xml(tag, T_link, mesh, rgba=None):
        f = (gdir / 'assets' / meshfile[mesh]).as_posix()
        mat = f'<material name="m_{mesh}_{abs(hash((tag, str(T_link.round(4))))) % 10**6}"><color rgba="{rgba}"/></material>' if rgba else ''
        return (f'<{tag}>{_origin(T_link)}<geometry><mesh filename="{f}" scale="0.001 0.001 0.001"/></geometry>'
                f'{mat if tag == "visual" else ""}</{tag}>')

    black, grey = '0.15 0.15 0.15 1', '0.46 0.46 0.46 1'
    parts = ['<link name="gripper_base"><inertial><origin xyz="0 0 0.04"/><mass value="0.777"/>'
             '<inertia ixx="2.6e-4" ixy="0" ixz="0" iyy="2.3e-4" iyz="0" izz="1.5e-4"/></inertial>']
    for name, mesh, rgba, coll in (('base_mount', 'base_mount', black, True), ('base', 'base', black, True),
                                   ('right_driver', 'driver', grey, False), ('right_coupler', 'coupler', black, False),
                                   ('right_spring_link', 'spring_link', black, False),
                                   ('left_driver', 'driver', grey, False), ('left_coupler', 'coupler', black, False),
                                   ('left_spring_link', 'spring_link', black, False)):
        parts.append(mesh_xml('visual', bodies[name][0], mesh, rgba))
        if coll:
            parts.append(mesh_xml('collision', bodies[name][0], mesh))
    links, joints = [''.join(parts) + '</link>'], []
    for side in ('right', 'left'):                                   # prismatic fingers closing toward the centre line
        T_pad, pad = bodies[f'{side}_pad']
        d = np.array([0.0, 0.0, T_pad[2, 3]]) - T_pad[:3, 3]
        d[2] = 0.0
        axis = T_pad[:3, :3].T @ (d / np.linalg.norm(d))
        s = [f'<link name="{side}_finger"><inertial><origin xyz="0 0 0.02"/><mass value="0.05"/>'
             '<inertia ixx="2e-5" ixy="0" ixz="0" iyy="2e-5" iyz="0" izz="1e-5"/></inertial>',
             mesh_xml('visual', np.linalg.inv(T_pad) @ bodies[f'{side}_follower'][0], 'follower', black),
             mesh_xml('visual', np.eye(4), 'pad', grey), mesh_xml('visual', np.eye(4), 'silicone_pad', black)]
        for g in pad.findall('geom'):
            if g.get('class') in ('pad_box1', 'pad_box2'):
                dg = next(x for x in root.iter('default') if x.get('class') == g.get('class')).find('geom')
                pos = [float(v) for v in dg.get('pos').split()]
                size = [2 * float(v) for v in dg.get('size').split()]
                s.append(f'<collision><origin xyz="{pos[0]} {pos[1]} {pos[2]}" rpy="0 0 0"/>'
                         f'<geometry><box size="{size[0]} {size[1]} {size[2]}"/></geometry></collision>')
        links.append(''.join(s) + '</link>')
        joints.append(f'<joint name="{side}_finger_joint" type="prismatic"><parent link="gripper_base"/><child link="{side}_finger"/>'
                      f'{_origin(T_pad)}<axis xyz="{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f}"/>'
                      f'<limit lower="0" upper="{stroke}" effort="235" velocity="0.15"/><dynamics damping="1.0" friction="0.0"/></joint>')
    T_tcp = bodies['base'][0] @ _mk(np.eye(3), [0, 0, 0.145])
    links.append('<link name="tcp"/>')
    joints.append(f'<joint name="tcp_joint" type="fixed"><parent link="gripper_base"/><child link="tcp"/>{_origin(T_tcp)}</joint>')
    joints.append(f'<joint name="gripper_mount" type="fixed"><parent link="link_6"/><child link="gripper_base"/>'
                  f'<origin xyz="0 0 {coupling}" rpy="0 0 0"/></joint>')
    base = (share / 'urdf' / 'm1013.urdf').read_text().replace('package://dsr_description2', share.as_posix())
    out = rl_path(cfg["paths"]["robot_urdf"])
    out.write_text(base.replace('</robot>', '\n  ' + '\n  '.join(links + joints) + '\n</robot>'))
    print(f'wrote {out}  (coupling {coupling} m, TCP {coupling + T_tcp[2, 3]:.4f} m from link_6)', flush=True)


# ---------------------------------------------------------------- plate

def build_plate(cfg):
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app
    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
    from isaaclab.sim.schemas import schemas_cfg
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    phys = rl_path(cfg["paths"]["plate_physics_usd"])
    MeshConverter(MeshConverterCfg(
        asset_path=str(rl_path(cfg["paths"]["plate_obj_simplified"])), usd_dir=str(phys.parent), usd_file_name=phys.name,
        force_usd_conversion=True, make_instanceable=False,
        mass_props=schemas_cfg.MassPropertiesCfg(mass=cfg["geometry"]["plate_mass"]),
        rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
        collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
        mesh_collision_props=schemas_cfg.ConvexDecompositionPropertiesCfg(max_convex_hulls=16, hull_vertex_limit=64)))
    print(f'wrote {phys}', flush=True)

    obj, out = rl_path(cfg["paths"]["plate_obj_textured"]), rl_path(cfg["paths"]["plate_usd"])
    stage = Usd.Stage.CreateNew(str(out))
    root = stage.DefinePrim('/Plate', 'Xform')
    root.GetReferences().AddReference(str(phys))
    stage.SetDefaultPrim(root)
    for prim in list(stage.Traverse()):                              # hide the simplified collision mesh
        if prim.IsA(UsdGeom.Mesh):
            UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    verts, uv, faces, uv_idx, counts = [], [], [], [], []
    for line in obj.open():
        a = line.split()
        if not a:
            continue
        if a[0] == 'v':
            verts.append(list(map(float, a[1:4])))
        elif a[0] == 'vt':
            uv.append(list(map(float, a[1:3])))
        elif a[0] == 'f':
            counts.append(len(a) - 1)
            for token in a[1:]:
                i = token.split('/')
                faces.append(int(i[0]) - 1)
                uv_idx.append(int(i[1]) - 1)
    v = np.array(verts)
    lo, hi = v.min(0), v.max(0)
    centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])            # origin = bottom centre
    mesh = UsdGeom.Mesh.Define(stage, '/Plate/OriginalVisual')
    mesh.CreatePointsAttr([Gf.Vec3f(*x) for x in v - centre])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set([Gf.Vec2f(*x) for x in uv])
    st.SetIndices(uv_idx)
    mat = UsdShade.Material.Define(stage, '/Plate/OriginalMaterial')
    shader = UsdShade.Shader.Define(stage, '/Plate/OriginalMaterial/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.35)
    reader = UsdShade.Shader.Define(stage, '/Plate/OriginalMaterial/UV')
    reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    tex = UsdShade.Shader.Define(stage, '/Plate/OriginalMaterial/Texture')
    tex.CreateIdAttr('UsdUVTexture')
    tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(obj.with_suffix('.png'))))
    tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
    tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.CreateOutput('result', Sdf.ValueTypeNames.Float2))
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.CreateOutput('rgb', Sdf.ValueTypeNames.Float3))
    mat.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput('surface', Sdf.ValueTypeNames.Token))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.GetRootLayer().Save()
    print(f'wrote {out}  ({len(verts)} vertices, extent {np.round(hi - lo, 4).tolist()} m)', flush=True)
    app.close()


# ---------------------------------------------------------------- shield distance fields

def build_sdf(cfg):
    from scipy import ndimage
    calib = rl_path(cfg["paths"]["calibration_dir"])
    sys.path.insert(0, str(calib))
    from robot_model import frame_layout, measured_cell
    sh = cfg["shield"]
    res, lo, hi = sh["sdf_res"], np.array(sh["sdf_lo"]), np.array(sh["sdf_hi"])
    boxes = frame_layout(measured_cell())
    shape = np.ceil((hi - lo) / res).astype(int)
    centres = lo + (np.indices(shape).transpose(1, 2, 3, 0) + 0.5) * res

    def edt(selected):
        occ = np.zeros(shape, dtype=bool)
        for _, c, s in selected:
            occ |= np.all(np.abs(centres - np.asarray(c)) <= np.asarray(s) / 2, axis=-1)
        return ndimage.distance_transform_edt(~occ, sampling=res).astype(np.float32)

    pts = np.load(calib / 'data' / 'm1013_collision_points.npz')
    rng = np.random.default_rng(0)
    links = {f'link_{i}': pts[f'link_{i}'][rng.choice(len(pts[f'link_{i}']), sh["points_per_link"], replace=False)]
             for i in range(1, 7)}
    out = rl_path(cfg["paths"]["cell_sdf"])
    np.savez_compressed(out, lo=lo, res=res, frame=edt([b for b in boxes if b[0] != 'table']),
                        table=edt([b for b in boxes if b[0] == 'table']),
                        boxes=json.dumps([[n, list(map(float, c)), list(map(float, s))] for n, c, s in boxes]), **links)
    print(f'wrote {out}  grid {shape.tolist()}, frame boxes {[b[0] for b in boxes if b[0] != "table"]}', flush=True)


# ---------------------------------------------------------------- ready joints

def build_ready(args, cfg):
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_from_matrix, quat_mul
    import plate_env as pe

    env_cfg = pe.make_env_cfg(2, 4)
    env_cfg.events.ready_start = None                                # start from home
    env_cfg.events.reset_plate.params["pose_range"] = {"yaw": (-3.1416, 3.1416)}      # nominal plate position
    env_cfg.episode_length_s = 60.0
    env_cfg.terminations.success = None
    env = ManagerBasedRLEnv(env_cfg)
    dev, G = env.device, cfg["geometry"]
    z = torch.tensor(G["ready_tool_z"], device=dev)
    y = torch.tensor(G["ready_finger_axis"], device=dev)
    y = y - (y @ z) * z
    y = y / y.norm()
    q_t = quat_from_matrix(torch.stack([torch.linalg.cross(y, z), y, z], dim=1)).unsqueeze(0).expand(env.num_envs, -1)
    env.reset()
    with torch.inference_mode():
        for t in range(1200):                                        # rotation converges slowly near the wrist singularity
            tcp, tq = pe.tcp_pose(env)
            ready = env.scene["plate"].data.root_pos_w + torch.tensor(G["ready_offset"], device=dev)
            act = torch.zeros(env.num_envs, 7, device=dev)
            act[:, :3] = ((ready - tcp) / 0.03).clamp(-0.3, 0.3)
            act[:, 3:6] = (axis_angle_from_quat(quat_mul(q_t, quat_conjugate(tq))) / 0.03).clamp(-0.3, 0.3)
            act[:, 6] = 1.0
            env.step(act)
        st = pe._state(env)
        r = env.scene["robot"]
        angle = lambda m: torch.rad2deg(torch.acos(m.clamp(-1, 1))).max().item()       # noqa: E731
        print(f"ready error: {st['d_ready'].max().item() * 1000:.1f} mm, tool {angle(st['tool_match']):.2f} deg, "
              f"finger {angle(st['finger_match']):.2f} deg", flush=True)
        joints = {n: float(r.data.joint_pos[0, r.joint_names.index(n)]) for n in r.joint_names if n.startswith("joint_")}
    out = rl_path(cfg["paths"]["ready_joints"])
    out.write_text(json.dumps({"joints": joints, "note": f"ready TCP pose reached from home by the env diff-IK, "
                                                         f"box_h {G['box_h']}"}, indent=1))
    print(f"wrote {out}: {{{', '.join(f'{k}: {np.degrees(v):.2f} deg' for k, v in joints.items())}}}", flush=True)
    env.close()
    app.close()


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=("urdf", "plate", "sdf", "ready"))
    if "ready" in sys.argv:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    cfg = load_config()
    {"urdf": lambda: build_urdf(cfg), "plate": lambda: build_plate(cfg), "sdf": lambda: build_sdf(cfg),
     "ready": lambda: build_ready(args, cfg)}[args.what]()
