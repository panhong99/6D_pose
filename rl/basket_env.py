"""Goal 2 (2026-10-08): put the held plate into dish-basket slot 1 — Isaac Lab scene additions on top of plate_env.

  basket USD : one kinematic rigid body = thin floor + two rows of leaning pins around the taught slot (rl/basket.yaml)
  slot frame : origin on the floor where the taught plate plane meets it, x = n (horizontal slot axis, the pins of a row
               line up along it), y = u = z x n (the rows sit at +-4.75 cm), z up; nominal pose from the golden teaching
  make_basket_cfg(num_envs) : plate_env stage-2 env + the basket (pose randomised per reset) + no stage-2 success end
  basket_state(env)         : slot pose and plate-in-slot quantities for experts, observations and the success test
Pure helpers (slot_frame, pin_layout) need no Isaac app.  Used by rl/basket_tasks.py (collect / gate / export).
"""
import math
from pathlib import Path

import numpy as np
import yaml

RL_DIR = Path(__file__).resolve().parent
B = yaml.safe_load((RL_DIR / "basket.yaml").read_text())
BASKET_USD = RL_DIR / "assets" / "basket_slot1.usda"


# ---------------------------------------------------------------- geometry (numpy, no Isaac)

def slot_frame():
    """Nominal slot frame in base: (origin xyz, yaw rad, plate tilt rad).  origin = floor point of the taught plate
    plane under the plate centre; yaw = direction of n; tilt = plate angle from vertical (top toward -n)."""
    c, N = np.array(B["slot"]["plate_centre"], float), np.array(B["slot"]["plate_normal"], float)
    N /= np.linalg.norm(N)
    n_h = N[:2] / np.linalg.norm(N[:2])
    yaw = math.atan2(n_h[1], n_h[0])
    if B.get("full", {}).get("slot_axis_from_basket", False):    # pin rows run along the basket x axis (marker frame):
        yaw = math.radians(B["full"]["pose_base"][3])             # the taught-normal yaw (-8.7 deg) was ~10 deg off them
        n_h = np.array([math.cos(yaw), math.sin(yaw)])
    tilt = math.atan2(N[2], np.linalg.norm(N[:2]))
    h = c[2] - B["slot"]["floor_z"]
    o = np.array([c[0], c[1], B["slot"]["floor_z"]])
    o[:2] += n_h * h * math.tan(tilt)                      # the plane leans: its floor line lies +n of the centre
    return o, yaw, tilt


def pin_layout():
    """Pins in the slot frame: list of (base xyz, axis unit vector)."""
    p, (_, _, tilt) = B["pins"], slot_frame()
    lean = math.radians(p["lean_deg"]) if "lean_deg" in p else tilt    # (10/12 Human) measured tine lean toward -x
    axis = np.array([-math.sin(lean), 0.0, math.cos(lean)]) if p["lean_with_plate"] else np.array([0.0, 0.0, 1.0])
    k = p["per_row"] // 2
    xs = [(i + 0.5) * p["pitch"] for i in range(-k, k)]
    return [(np.array([x, u, 0.0]), axis) for u in p["rows_u"] for x in xs]


def basket_nominal_T():
    """Measured basket frame in base (rl/basket.yaml full.pose_base), 4x4."""
    x, y, z, yaw = B["full"]["pose_base"]
    c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    T = np.eye(4)
    T[:3, :3] = [[c, -s_, 0], [s_, c, 0], [0, 0, 1]]
    T[:3, 3] = [x, y, z]
    return T


def full_basket_parts():
    """Whole basket in the basket frame as simple shapes: ('cyl', p0, p1, r) and ('box', centre, size).
    Pins of both plate pairs line up along x (pitch 2 cm, phased so slot 1 sits between two of them) and lean like the
    taught plate; floor = wire grid; walls = vertical rods; gray side rails; blue handle."""
    f = B["full"]
    hx, hy = f["size_xy"][0] / 2, f["size_xy"][1] / 2
    z0, r = f["floor_z"], f["rod_r"]
    parts = []
    for y in np.arange(-hy, hy + 1e-9, f["floor_grid"]):                   # floor grid
        parts.append(("cyl", np.array([-hx, y, z0 - r]), np.array([hx, y, z0 - r]), r))
    for x in np.arange(-hx, hx + 1e-9, f["floor_grid"]):
        parts.append(("cyl", np.array([x, -hy, z0 - 3 * r]), np.array([x, hy, z0 - 3 * r]), r))
    for x in np.arange(-hx, hx + 1e-9, f["wall_rod_pitch"]):              # wall rods
        for y in (-hy, hy):
            parts.append(("cyl", np.array([x, y, z0]), np.array([x, y, 0.0]), r))
    for y in np.arange(-hy, hy + 1e-9, f["wall_rod_pitch"]):
        for x in (-hx, hx):
            parts.append(("cyl", np.array([x, y, z0]), np.array([x, y, 0.0]), r))
    parts.append(("cyl", np.array([-hx, -hy, -0.01]), np.array([hx, -hy, -0.01]), r * 1.5))   # back top wire
    rw, rh = f["rail_w_h"]
    for x in (-hx, hx):
        parts.append(("box", np.array([x, 0.0, -rh / 2]), np.array([rw, 2 * hy, rh])))
    hw, hh = f["handle_w_h"]
    parts.append(("box", np.array([0.0, hy, -hh / 2]), np.array([2 * hx, hw, hh])))
    # pins: slot-1 bottom line from the golden slot frame, expressed in the basket frame
    o, yaw, tilt = slot_frame()
    Tb = basket_nominal_T()
    o_b = (np.linalg.inv(Tb) @ np.r_[o, 1.0])[:3]
    p, h = B["pins"], B["pins"]["height"]
    lean = math.radians(p["lean_deg"]) if "lean_deg" in p else tilt    # (10/12 Human) measured tine lean toward -x
    axis = np.array([-math.sin(lean), 0.0, math.cos(lean)]) if p["lean_with_plate"] else np.array([0.0, 0.0, 1.0])
    x_ph = o_b[0] + p["pitch"] / 2
    xs = x_ph + p["pitch"] * np.arange(-40, 41)
    xs = xs[(xs >= f["pin_x_span"][0]) & (xs <= f["pin_x_span"][1]) & (np.abs(xs) >= f["centre_gap_x"])]
    count = f.get("tines_from_clip")                                     # (10/14 보스) tines per row in each quadrant, from the clip
    for k, yc in enumerate(f["pair_centres_y"]):                           # pair A (+y, left) then pair B (right)
        for far in (False, True):
            half = np.sort(np.abs(xs[(xs > 0) == far]))                    # nearest the clip first
            if count:
                half = half[:count[("Q2" if k == 0 else "Q1") if far else ("Q3" if k == 0 else "Q4")]]
            hx_ = half if far else -half
            if not len(hx_):
                continue
            for dy in p["rows_u"]:
                for x in hx_:
                    base = np.array([x, yc + dy, z0])
                    parts.append(("cyl", base, base + axis * h, f["pin_r"]))
                if "rail_z" in p:                                            # (10/12 Human) the comb rail at 4.9 cm
                    parts.append(("cyl", np.array([hx_.min(), yc + dy, z0 + p["rail_z"]]), np.array([hx_.max(), yc + dy, z0 + p["rail_z"]]), r))
    return parts


def build_full_basket_usd(path=RL_DIR / "assets" / "basket_full.usda"):
    """Whole basket (full_basket_parts) as one kinematic rigid body, basket frame = body frame."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Basket")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(3.0)
    for i, part in enumerate(full_basket_parts()):
        if part[0] == "cyl":
            _, a, b, r = part
            d = b - a
            g = UsdGeom.Cylinder.Define(stage, f"/Basket/c{i:04d}")
            g.CreateRadiusAttr(float(r))
            g.CreateHeightAttr(float(np.linalg.norm(d)))
            g.CreateAxisAttr("Z")
            rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*(d / np.linalg.norm(d)).tolist()))
            g.AddTranslateOp().Set(Gf.Vec3d(*((a + b) / 2).tolist()))
            g.AddOrientOp().Set(Gf.Quatf(rot.GetQuat()))
            g.CreateDisplayColorAttr([(0.2, 0.2, 0.22)])
        else:
            _, c, size = part
            g = UsdGeom.Cube.Define(stage, f"/Basket/b{i:04d}")
            g.CreateSizeAttr(1.0)
            g.AddTranslateOp().Set(Gf.Vec3d(*c.tolist()))
            g.AddScaleOp().Set(Gf.Vec3f(*size.tolist()))
            g.CreateDisplayColorAttr([(0.15, 0.45, 0.8)] if size[0] > size[1] else [(0.35, 0.35, 0.38)])
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    stage.GetRootLayer().Save()
    return path


def export_full_basket_obj(path=RL_DIR / "assets" / "basket_full.obj"):
    """Same shapes as a mesh (trimesh) for RViz / FK collision checks (conda python)."""
    import trimesh
    meshes = []
    for part in full_basket_parts():
        if part[0] == "cyl":
            _, a, b, r = part
            meshes.append(trimesh.creation.cylinder(radius=r, segment=np.stack([a, b]), sections=8))
        else:
            _, c, size = part
            meshes.append(trimesh.creation.box(extents=size, transform=trimesh.transformations.translation_matrix(c)))
    trimesh.util.concatenate(meshes).export(path)
    return path


def build_basket_usd(path=BASKET_USD):
    """Basket = kinematic rigid body: floor cube + pin cylinders (implicit PhysX shapes), slot frame as body frame."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Basket")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(2.0)
    fx, fy, fz = B["floor_size"]
    floor = UsdGeom.Cube.Define(stage, "/Basket/floor")
    floor.CreateSizeAttr(1.0)
    floor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -fz / 2))
    floor.AddScaleOp().Set(Gf.Vec3f(fx, fy, fz))
    floor.CreateDisplayColorAttr([(0.25, 0.25, 0.28)])
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    r, h = B["pins"]["radius"], B["pins"]["height"]
    for i, (base, axis) in enumerate(pin_layout()):
        pin = UsdGeom.Cylinder.Define(stage, f"/Basket/pin_{i:02d}")
        pin.CreateRadiusAttr(r)
        pin.CreateHeightAttr(h)
        pin.CreateAxisAttr("Z")
        centre = base + axis * h / 2
        rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*axis.tolist()))
        pin.AddTranslateOp().Set(Gf.Vec3d(*centre.tolist()))
        pin.AddOrientOp().Set(Gf.Quatf(rot.GetQuat()))
        pin.CreateDisplayColorAttr([(0.15, 0.15, 0.17)])
        UsdPhysics.CollisionAPI.Apply(pin.GetPrim())
    stage.GetRootLayer().Save()
    return path


# ---------------------------------------------------------------- Isaac Lab env

def make_basket_cfg(num_envs, viewer=False, full=True):
    """plate_env stage-2 config + the basket (full: whole parametric basket at the measured marker pose; else the
    slot-1 pin patch); stage-2 success no longer ends an episode (the plate goes on)."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.managers import EventTermCfg as EventTerm
    import plate_env as pe
    usd = RL_DIR / "assets" / ("basket_full.usda" if full else "basket_slot1.usda")
    if full:
        build_full_basket_usd(usd)
        Tn = basket_nominal_T()
        pos, yaw = Tn[:3, 3], math.radians(B["full"]["pose_base"][3])
    else:
        if not usd.exists():
            build_basket_usd(usd)
        pos, yaw, _ = slot_frame()
    cfg = pe.make_env_cfg(2, num_envs, viewer=viewer, rim_rel=True)
    cfg.scene.basket = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Basket",
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(np.asarray(pos).tolist()), rot=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))),
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd),
                                   rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)))
    cfg.terminations.success = None
    cfg.terminations.time_out = None
    cfg.terminations.plate_lost = None                     # a plate standing in the basket is "below the box" by design
    cfg.terminations.collision = None                      # measured by the task scripts, never auto-reset mid-insertion
    cfg.events.basket_pose = EventTerm(func=randomize_basket, mode="reset", params={"full": full})
    cfg.episode_length_s = 3600.0
    return cfg


def randomize_basket(env, env_ids, full=True):
    """Reset event: basket pose = nominal (full: measured marker frame; else the slot frame) moved by a uniform
    xy / yaw (rl/basket.yaml randomize) about the basket origin.  The slot-1 frame follows: env._slot (x, y, z, yaw)."""
    import torch
    dev, n = env.device, len(env_ids)
    rnd = B["randomize"]
    o_s, yaw_s, _ = slot_frame()
    if full:
        Tn = basket_nominal_T()
        o_b, yaw_b = Tn[:3, 3], math.radians(B["full"]["pose_base"][3])
    else:
        o_b, yaw_b = o_s, yaw_s
    dxy = (torch.rand(n, 2, device=dev) * 2 - 1) * rnd["xy_m"]
    dyaw = (torch.rand(n, device=dev) * 2 - 1) * math.radians(rnd["yaw_deg"])
    pose = torch.zeros(n, 7, device=dev)
    pose[:, :3] = torch.tensor(o_b, dtype=torch.float32, device=dev) + env.scene.env_origins[env_ids]
    pose[:, :2] += dxy
    yaw = yaw_b + dyaw
    pose[:, 3], pose[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
    env.scene["basket"].write_root_pose_to_sim(pose, env_ids=env_ids)
    if not hasattr(env, "_slot"):
        env._slot = torch.zeros(env.num_envs, 4, device=dev)          # x, y, z (env frame), yaw
    rel = torch.tensor(o_s[:2] - np.asarray(o_b)[:2], dtype=torch.float32, device=dev)   # slot origin from the basket origin
    c, s_ = torch.cos(dyaw), torch.sin(dyaw)
    env._slot[env_ids, 0] = float(o_b[0]) + dxy[:, 0] + c * rel[0] - s_ * rel[1]
    env._slot[env_ids, 1] = float(o_b[1]) + dxy[:, 1] + s_ * rel[0] + c * rel[1]
    env._slot[env_ids, 2] = float(o_s[2])
    env._slot[env_ids, 3] = yaw_s + dyaw


def basket_state(env):
    """Slot axes and the plate relative to the slot (env frame).  d_plane: plate centre to the slot plane along n
    (the taught plate plane, tilted), d_u: along the rows, low: plate lowest rim point above the floor."""
    import torch
    import plate_env as pe
    s = env._slot
    c, s_ = torch.cos(s[:, 3]), torch.sin(s[:, 3])
    n = torch.stack([c, s_, torch.zeros_like(c)], 1)
    u = torch.stack([-s_, c, torch.zeros_like(c)], 1)
    _, _, tilt = slot_frame()
    plate = env.scene["plate"]
    p = plate.data.root_pos_w - env.scene.env_origins
    up = pe.body_axis(plate.data.root_quat_w, 2)                       # plate normal (mesh z)
    centre = p + up * 0.01                                              # root = bottom centre of the dish; centre ~1 cm up
    rel = centre - s[:, :3]
    plane_dir = n * math.cos(tilt) + torch.tensor([0, 0, 1.0], device=env.device) * math.sin(tilt)
    d_plane = (rel * plane_dir).sum(1)                                  # the slot origin lies on the taught plate plane
    d_u = (rel * u).sum(1)
    normal_err = torch.rad2deg(torch.acos(torch.abs((up * plane_dir).sum(1)).clamp(max=1.0)))
    from isaaclab.utils.math import quat_apply
    if not hasattr(env, "_plate_verts"):                                 # collision mesh vertices in the plate root frame
        v = [list(map(float, l.split()[1:4])) for l in (RL_DIR / "assets" / "plate029_simplified.obj").open() if l.startswith("v ")]
        v = torch.tensor(v, device=env.device)
        env._plate_verts = v[torch.linspace(0, len(v) - 1, min(len(v), 600)).long()]
    V = env._plate_verts
    q = plate.data.root_quat_w[:, None, :].expand(-1, len(V), -1)
    W = p[:, None, :] + quat_apply(q.reshape(-1, 4), V.repeat(len(p), 1)).reshape(len(p), len(V), 3)
    k = W[:, :, 2].argmin(1)
    bottom = W[torch.arange(len(p)), k]
    low = bottom[:, 2] - B["slot"]["floor_z"]                           # lowest mesh point above the floor
    d_bottom = ((bottom - s[:, :3]) * n).sum(1)                         # ... from the slot line along n (pins at +-pitch/2)
    return dict(n=n, u=u, rel=rel, d_plane=d_plane, d_u=d_u, normal_err=normal_err, low=low, d_bottom=d_bottom)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["layout"]:
        o, yaw, tilt = slot_frame()
        print(f"slot origin {np.round(o, 4).tolist()}  yaw {math.degrees(yaw):.1f} deg  plate tilt {math.degrees(tilt):.1f} deg")
        for base, axis in pin_layout():
            print(f"  pin base (n, u) = ({base[0]:+.3f}, {base[1]:+.3f})  axis {np.round(axis, 3).tolist()}")
