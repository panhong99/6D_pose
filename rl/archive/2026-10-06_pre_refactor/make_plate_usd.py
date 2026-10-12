"""Convert the simplified 029 plate mesh to a rigid-body USD for Isaac Lab (run with isaaclab.sh -p).

Input : assets/plate029_simplified.obj  (open3d decimation of demo_data/029_plate/tsdf/textured.obj to 3000 triangles,
        origin = bottom-centre of the bounding box, z up; 266 x 268 x 25 mm dish, flat floor + rising rim)
Output: assets/plate029.usd  (rigid body, 0.3 kg, convex-decomposition collision so the dish is NOT filled in)
"""
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--hulls", type=int, default=16)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg

HERE = Path(__file__).resolve().parent
cfg = MeshConverterCfg(
    asset_path=str(HERE / "assets" / "plate029_simplified.obj"),
    usd_dir=str(HERE / "assets"), usd_file_name="plate029.usd", force_usd_conversion=True, make_instanceable=False,
    mass_props=schemas_cfg.MassPropertiesCfg(mass=0.30),
    rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
    collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
    mesh_collision_props=schemas_cfg.ConvexDecompositionPropertiesCfg(max_convex_hulls=args.hulls, hull_vertex_limit=64),
)
print("USD:", MeshConverter(cfg).usd_path, flush=True)
app.close()
