"""Preserve existing plate physics and display the original textured OBJ."""
from pathlib import Path
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
import numpy as np

here = Path(__file__).resolve().parent
source = here / 'assets/plate029.usd'
obj = here.parent / 'demo_data/029_plate/tsdf/textured.obj'
output = here / 'assets/plate029_textured.usda'
stage = Usd.Stage.CreateNew(str(output))
root = stage.DefinePrim('/Plate', 'Xform')
root.GetReferences().AddReference(str(source))
stage.SetDefaultPrim(root)
hidden = []
for prim in list(stage.Traverse()):
    if prim.IsA(UsdGeom.Mesh):
        UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        hidden.append(str(prim.GetPath()))
verts, uv, faces, uv_indices, counts = [], [], [], [], []
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
            indices = token.split('/')
            faces.append(int(indices[0]) - 1)
            uv_indices.append(int(indices[1]) - 1)
v = np.array(verts)
lo, hi = v.min(0), v.max(0)
centre = np.array([(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, lo[2]])
mesh = UsdGeom.Mesh.Define(stage, '/Plate/OriginalVisual')
mesh.CreatePointsAttr([Gf.Vec3f(*x) for x in v-centre])
mesh.CreateFaceVertexCountsAttr(counts)
mesh.CreateFaceVertexIndicesAttr(faces)
mesh.CreateSubdivisionSchemeAttr('none')
mesh.CreateDoubleSidedAttr(True)
st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
st.Set([Gf.Vec2f(*x) for x in uv])
st.SetIndices(uv_indices)
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
print(f'TEXTURED_PLATE {output} vertices={len(verts)} faces={len(counts)} hidden={hidden} extent_m={(hi-lo).tolist()}', flush=True)
app.close()
