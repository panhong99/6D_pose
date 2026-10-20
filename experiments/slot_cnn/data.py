"""(10/20 보스, BQ) wrist-camera slot CNN data: synthetic renders of the basket model with tine root/tip labels, and the
real test photos with Codex's v4 labels.

    ~/miniconda3/envs/foundationpose/bin/python experiments/slot_cnn/data.py render --n 20000 --out calibration/data/slot_cnn/sim_v1
    ~/miniconda3/envs/foundationpose/bin/python experiments/slot_cnn/data.py check     # model tines vs v4 labels, real photos
    ~/miniconda3/envs/foundationpose/bin/python experiments/slot_cnn/data.py sheet DIR # sim samples next to real photos

Labels (as v4): root = where the leaning tine crosses its comb rail (rail_z above the floor), tip = the free end.
Rendering: nvdiffrast, the basket of rl/basket_env.full_basket_parts (read only) with per-sample changes of tine pitch,
lean, wire radius, colour; a textured table (perforated board / white cloth); a held plate and the gripper fingers in the
camera frame; random light, blur, noise.  Camera K = the wrist D4xx (1280 x 720), poses around the real wrist looks."""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'rl'))
DATA = ROOT / 'calibration/data'
V4 = DATA / 'basket/wrist_look/az_retrace_20261019_v4'
W, H = 1280, 720                                     # wrist camera
OW, OH = 960, 544                                    # network input (stored samples)
K_WRIST = np.array([[643.6035, 0, 637.8715], [0, 642.8423, 364.3775], [0, 0, 1.0]])


# ----------------------------------------------------------------------------------------------------- basket geometry
def basket_parts():
    import basket_env as be
    return be.full_basket_parts(include_marker_blocks=False), be.B


def tine_table(parts, B):
    """Tines of the model: list of dict(q, row, num, base, top) in the basket frame (num 1 = next to the clip)."""
    f = B['full']
    out = []
    for p in parts:
        if p[0] != 'cyl' or abs(p[3] - f['pin_r']) > 1e-9:
            continue
        base, top = np.asarray(p[1], float), np.asarray(p[2], float)
        far = base[0] > f.get('clip_x', 0.0)
        pair = int(np.argmin([abs(base[1] - (yc + dy)) - 1e-6 * k
                              for k, yc in enumerate(f['pair_centres_y']) for dy in B['pins']['rows_u']]))
        k = pair // 2
        q = ('Q2' if k == 0 else 'Q1') if far else ('Q3' if k == 0 else 'Q4')
        out.append(dict(q=q, row=pair % 2, base=base, top=top))
    for q in ('Q1', 'Q2', 'Q3', 'Q4'):
        for row in (0, 1):
            ts = sorted([t for t in out if t['q'] == q and t['row'] == row],
                        key=lambda t: abs(t['base'][0] - f.get('clip_x', 0.0)))
            for i, t in enumerate(ts):
                t['num'] = i + 1
    return out


def root_tip(t, rail_z, tip_above_rail=None):
    """3D root (on the rail) and tip of a tine."""
    d = t['top'] - t['base']
    root = t['base'] + d * (rail_z / d[2])
    tip = t['top'] if tip_above_rail is None else root + d / np.linalg.norm(d) * tip_above_rail / (d[2] / np.linalg.norm(d))
    return root, tip


def project(P, T_cam_world, K=K_WRIST):
    """World points (n, 3) -> pixels (n, 2), depth (n)."""
    Pc = np.c_[P, np.ones(len(P))] @ T_cam_world.T
    z = Pc[:, 2]
    uv = (Pc[:, :3] @ K.T)[:, :2] / z[:, None]
    return uv, z


# -------------------------------------------------------------------------------------------------- real test photos
def real_tests():
    """Codex v4 labelled photos: [(image path, T_cam_basket (marker), [(id, root_uv|None, tip_uv|None)], label quadrant)]."""
    out = []
    for q in ('Q1', 'Q2', 'Q3', 'Q4'):
        lab = json.loads((V4 / f'{q}.json').read_text())
        meta = json.loads(Path(lab['capture_metadata']).read_text())
        Tbc = np.asarray(meta['T_base_flange']) @ np.asarray(meta['T_flange_cam'])
        Tcb = np.linalg.inv(Tbc) @ np.asarray(meta['T_base_basket'])
        pins = [(p['id'], p['root_uv'], p['tip_uv']) for p in lab['pins']]
        out.append((lab['image'], Tcb, pins, q))
    return out


def camera_poses():
    """T_basket_cam of every saved wrist photo with a basket pose (wrist looks + pre-insert photos): the view
    distribution the renders sample around."""
    out = []
    for f in sorted(glob.glob(str(DATA / 'basket/wrist_look/2026*.json'))) + \
            sorted(glob.glob(str(DATA / 'real_pipeline/*/wrist_pre.json'))) + \
            sorted(glob.glob(str(DATA / 'real_pipeline/*/wrist_above.json'))):
        if f.endswith('.req.json'):
            continue
        m = json.loads(Path(f).read_text())
        if not all(k in m for k in ('T_base_flange', 'T_flange_cam', 'T_base_basket')):
            continue
        name = Path(f).parent.name + '/' + Path(f).stem
        if Path(f).parent.name[:13] < '20261019_0900' and 'wrist_look' not in f:
            continue                                     # older runs: other basket frame definitions
        Tbc = np.asarray(m['T_base_flange']) @ np.asarray(m['T_flange_cam'])
        T = np.linalg.inv(np.asarray(m['T_base_basket'])) @ Tbc
        if Path(f).stem[:8] < '20261019' and 'wrist_look' in f:
            continue
        if T[2, 2] > -0.8:                               # optical axis must look down
            continue
        out.append((name, T))
    return out


def check(out):
    """Model tines projected with the marker pose onto the 4 labelled photos, next to the v4 labels."""
    parts, B = basket_parts()
    rail = B['pins']['rail_z']
    tines = tine_table(parts, B)
    out.mkdir(parents=True, exist_ok=True)
    for img_path, Tcb, pins, q in real_tests():
        img = cv2.imread(img_path)
        for t in tines:
            r, tp = root_tip(t, rail)
            uv, z = project(np.stack([r, tp]), Tcb)
            if (z > 0).all():
                cv2.line(img, tuple(uv[0].astype(int)), tuple(uv[1].astype(int)), (0, 0, 255), 1)
                cv2.putText(img, f"{t['q']}{t['row']}-{t['num']}", tuple(uv[1].astype(int)), 0, 0.3, (0, 0, 255), 1)
        for pid, ru, tu in pins:
            if ru and tu:
                cv2.line(img, tuple(np.int32(ru)), tuple(np.int32(tu)), (0, 220, 0), 2)
            for p in (ru, tu):
                if p:
                    cv2.circle(img, tuple(np.int32(p)), 3, (0, 220, 0), -1)
        cv2.imwrite(str(out / f'check_{q}.png'), img)
        print(q, img_path)


# ------------------------------------------------------------------------------------------------------------ renderer
def _unit_cyl(sections=8):
    a = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    ring = np.c_[np.cos(a), np.sin(a)]
    v = np.r_[np.c_[ring, np.zeros(sections)], np.c_[ring, np.ones(sections)]]
    n = np.r_[np.c_[ring, np.zeros(sections)], np.c_[ring, np.zeros(sections)]]
    f = []
    for i in range(sections):
        j = (i + 1) % sections
        f += [[i, j, sections + i], [j, sections + j, sections + i]]
    return v, n, np.array(f)


def _unit_box():
    import trimesh
    b = trimesh.creation.box(extents=(1, 1, 1))
    b = b.subdivide()                                      # flat faces, own normals per face
    v, f = b.vertices[b.faces].reshape(-1, 3), np.arange(len(b.faces) * 3).reshape(-1, 3)
    n = np.repeat(b.face_normals, 3, axis=0)
    return v, n, f


CYL, BOX = _unit_cyl(), None


def _frame_z(d):
    """Rotation taking +z to d."""
    z = d / np.linalg.norm(d)
    x = np.cross([0, 1.0, 0], z) if abs(z[1]) < 0.9 else np.cross([1.0, 0, 0], z)
    x /= np.linalg.norm(x)
    return np.c_[x, np.cross(z, x), z]


class Mesh:
    """Triangles with per-vertex normal, colour, uv (uv only for the textured table)."""
    def __init__(self):
        self.v, self.n, self.c, self.f, self.uv, self.tex = [], [], [], [], [], []
        self.count = 0

    def add(self, v, n, f, colour, uv=None, tex=0.0):
        self.v.append(v)
        self.n.append(n)
        self.f.append(f + self.count)
        self.c.append(np.broadcast_to(np.asarray(colour, float), (len(v), 3)))
        self.uv.append(np.zeros((len(v), 2)) if uv is None else uv)
        self.tex.append(np.full(len(v), tex))
        self.count += len(v)

    def cyl(self, a, b, r, colour):
        v0, n0, f0 = CYL
        R = _frame_z(b - a)
        L = np.linalg.norm(b - a)
        self.add((v0 * [r, r, L]) @ R.T + a, n0 @ R.T, f0, colour)

    def box(self, c, size, colour, R=np.eye(3)):
        global BOX
        if BOX is None:
            BOX = _unit_box()
        v0, n0, f0 = BOX
        self.add((v0 * size) @ R.T + c, n0 @ R.T, f0, colour)

    def arrays(self):
        return (np.concatenate(self.v), np.concatenate(self.n), np.concatenate(self.c), np.concatenate(self.f),
                np.concatenate(self.uv), np.concatenate(self.tex))


def random_tines(tines, B, rng):
    """Per-quadrant pitch scale about the clip, lean change and length change (covers the model-vs-real differences:
    Q1 pitch 30 vs 23.7 mm, Q4 1-2 cm, lean 10-15 deg)."""
    clip = B['full'].get('clip_x', 0.0)
    out = []
    qs = {q: (rng.uniform(0.75, 1.15), np.radians(rng.normal(0, 3.0)), rng.uniform(0.75, 1.05), rng.normal(0, 0.008))
          for q in ('Q1', 'Q2', 'Q3', 'Q4')}
    for t in tines:
        s, dl, ln, dx = qs[t['q']]
        base = t['base'].copy()
        base[0] = clip + (base[0] - clip) * s + dx
        d = (t['top'] - t['base']) * ln
        c, si = np.cos(dl), np.sin(dl)
        d = np.array([c * d[0] + si * d[2], d[1], -si * d[0] + c * d[2]])
        out.append(dict(t, base=base, top=base + d))
    return out


def table_texture(rng, size=1024):
    """Perforated board with a white cloth patch (texel = 2.5 mm; the patch sits under the basket)."""
    g = rng.uniform(0.5, 0.78)
    tex = np.full((size, size, 3), g) * rng.uniform(0.96, 1.04, 3)
    pitch = rng.uniform(8, 12)
    yy, xx = np.mgrid[0:size, 0:size]
    off = rng.uniform(0, pitch, 2)
    dots = (np.hypot((xx - off[0]) % pitch - pitch / 2, (yy - off[1]) % pitch - pitch / 2) < rng.uniform(1.0, 1.8))
    tex[dots] *= rng.uniform(0.15, 0.45)
    if rng.random() < 0.85:                                         # cloth under the basket (texture centre = basket)
        hw = rng.uniform(0.28, 0.42, 2) / 0.0025
        c = size / 2 + rng.normal(0, 15, 2)
        m = (np.abs(xx - c[0]) < hw[0]) & (np.abs(yy - c[1]) < hw[1])
        folds = cv2.GaussianBlur(rng.normal(0, 1, (size, size)), (0, 0), rng.uniform(6, 20))
        folds = folds / (np.abs(folds).max() + 1e-9)
        cloth = rng.uniform(0.82, 1.0) + 0.06 * folds
        tex[m] = (np.clip(cloth, 0, 1)[..., None] * rng.uniform(0.95, 1.0, 3))[m]
    tex = cv2.GaussianBlur(tex.astype(np.float32), (0, 0), 0.7)
    return np.clip(tex, 0, 1).astype(np.float32)


def build_scene(rng, parts, B, tines):
    """Basket (randomised tines) + table + distractor blocks in the basket frame -> Mesh, tines used."""
    f = B['full']
    m = Mesh()
    wire = np.clip(np.array([0.16, 0.16, 0.17]) * rng.uniform(0.6, 1.8) + rng.normal(0, 0.02, 3), 0.03, 0.5)
    tine = np.clip(np.full(3, rng.uniform(0.1, 0.6)) + rng.normal(0, 0.03, 3), 0.03, 0.7)   # (v2) real tines: thin, often
    r_rod, r_tine = rng.uniform(0.0015, 0.0024), rng.uniform(0.0011, 0.0019)               # lighter metal than the frame
    for p in parts:
        if p[0] == 'cyl':
            if abs(p[3] - f['pin_r']) < 1e-9:
                continue
            m.cyl(np.asarray(p[1], float), np.asarray(p[2], float), max(p[3], r_rod) if p[3] < 0.0025 else p[3], wire)
        else:
            c, size = np.asarray(p[1], float), np.asarray(p[2], float)
            blue = size[0] > size[1]
            col = np.array([0.1, 0.35, 0.75]) * rng.uniform(0.7, 1.2) if blue else np.full(3, rng.uniform(0.25, 0.5))
            m.box(c, size, np.clip(col, 0, 1))
    ts = random_tines(tines, B, rng)
    for t in ts:
        m.cyl(t['base'], t['top'], r_tine, tine * rng.uniform(0.9, 1.1))
    z_t = f['floor_z'] - 0.006                                       # table under the basket floor
    S = 1.28                                                         # 1024 texels * 2.5 mm / 2
    v = np.array([[-S, -S, z_t], [S, -S, z_t], [S, S, z_t], [-S, S, z_t]])
    m.add(v, np.tile([0, 0, 1.0], (4, 1)), np.array([[0, 1, 2], [0, 2, 3]]), (1, 1, 1),
          uv=np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]]), tex=1.0)
    for _ in range(rng.integers(0, 4)):                              # white blocks / boxes near the rim (markers, clutter)
        side = rng.choice([-1, 1])
        c = np.array([rng.uniform(-0.3, 0.3), side * rng.uniform(0.2, 0.32), rng.uniform(-0.06, 0.02)])
        if rng.random() < 0.5:
            c[:2] = c[[1, 0]]
        m.box(c, rng.uniform(0.04, 0.14, 3), np.full(3, rng.uniform(0.75, 0.98)))
    return m, ts


def add_held(m, rng, T_basket_cam):
    """Gripper fingers and (most of the time) the held plate, fixed in the camera frame."""
    def cam(p):
        return p @ T_basket_cam[:3, :3].T + T_basket_cam[:3, 3]
    Rc = T_basket_cam[:3, :3]
    finger = np.full(3, rng.uniform(0.04, 0.12))
    for sx in (-1, 1):                                               # two dark fingers at the image bottom
        c = np.array([sx * rng.uniform(0.03, 0.06), rng.uniform(0.065, 0.085), rng.uniform(0.09, 0.13)])
        m.box(cam(c[None])[0], [rng.uniform(0.04, 0.07), rng.uniform(0.025, 0.04), rng.uniform(0.04, 0.06)], finger, Rc)
    if rng.random() < 0.6:                                           # held plate seen edge-on (a disc in the camera y-z plane)
        colour = np.array([0.6, 0.05, 0.06]) * rng.uniform(0.7, 1.3) if rng.random() < 0.5 else np.full(3, rng.uniform(0.8, 0.98))
        ctr = np.array([rng.uniform(-0.02, 0.02), rng.uniform(0.07, 0.12), rng.uniform(0.17, 0.24)])
        a = np.radians(rng.normal(0, 3))
        Rp = Rc @ _frame_z(np.array([np.cos(a), 0, np.sin(a)]))         # disc axis ~ camera x
        R, th = 0.134, rng.uniform(0.008, 0.02)
        sec = _unit_cyl(48)
        v1, n1, f1 = sec
        m.add((v1 * [R, R, th] - [0, 0, th / 2]) @ Rp.T + cam(ctr[None])[0], n1 @ Rp.T, f1, np.clip(colour, 0, 1))
        cap = np.r_[[[0, 0, 0]], np.c_[np.cos(np.linspace(0, 2 * np.pi, 48, endpoint=False)),
                                      np.sin(np.linspace(0, 2 * np.pi, 48, endpoint=False)), np.zeros(48)]]
        for zz in (-th / 2, th / 2):
            fc = np.array([[0, 1 + i, 1 + (i + 1) % 48] for i in range(48)])
            m.add((cap * [R, R, 1] + [0, 0, zz]) @ Rp.T + cam(ctr[None])[0], np.tile(Rp[:, 2] * np.sign(zz), (49, 1)), fc,
                  np.clip(colour * 0.95, 0, 1))


def sample_camera(rng, poses):
    """A real wrist view (T_basket_cam) moved by a few cm and degrees."""
    from scipy.spatial.transform import Rotation
    looks = [i for i, (n, _) in enumerate(poses) if n.startswith('wrist_look')]
    i = rng.choice(looks) if looks and rng.random() < 0.6 else rng.integers(len(poses))   # straight-down looks (v3: 60 %, v1-v2: 40 %)
    T = poses[i][1].copy()
    T[:3, 3] += rng.normal(0, [0.04, 0.04, 0.04])
    yaw = Rotation.from_euler('z', rng.uniform(-30, 30), degrees=True).as_matrix()   # basket placed at another yaw
    T[:3, :3] = Rotation.from_rotvec(rng.normal(0, np.radians(5), 3)).as_matrix() @ yaw @ T[:3, :3]
    return T


class Renderer:
    def __init__(self, ss=2):
        import nvdiffrast.torch as dr
        import torch
        self.dr, self.torch, self.ss = dr, torch, ss
        self.ctx = dr.RasterizeCudaContext()

    def render(self, mesh, T_cam_world, tex, rng, K=K_WRIST, near=0.02, far=4.0):
        torch, dr = self.torch, self.dr
        v, n, c, f, uv, tf = mesh.arrays()
        Pc = v @ T_cam_world[:3, :3].T + T_cam_world[:3, 3]
        z = Pc[:, 2]
        u = K[0, 0] * Pc[:, 0] + K[0, 2] * z                            # u * z
        w = K[1, 1] * Pc[:, 1] + K[1, 2] * z
        clip = np.c_[2 * u / W - z, 2 * w / H - z, z * (far + near) / (far - near) - 2 * far * near / (far - near), z]
        dev = 'cuda'
        pos = torch.tensor(clip[None], dtype=torch.float32, device=dev)
        tri = torch.tensor(f, dtype=torch.int32, device=dev)
        rast, _ = dr.rasterize(self.ctx, pos, tri, (H * self.ss, W * self.ss))
        nc = n @ T_cam_world[:3, :3].T                                   # normals in the camera frame
        attr = torch.tensor(np.c_[c, nc, uv, tf, z][None], dtype=torch.float32, device=dev)
        A, _ = dr.interpolate(attr, rast, tri)
        A = A[0]
        hit = rast[0, ..., 3] > 0
        col, nrm, tuv, tfl, dep = A[..., :3], A[..., 3:6], A[..., 6:8], A[..., 8:9], A[..., 9]
        T = torch.tensor(tex[None], device=dev)
        tcol = dr.texture(T, tuv[None].contiguous(), filter_mode='linear')[0]
        col = torch.where(tfl > 0.5, tcol, col)
        nrm = torch.nn.functional.normalize(nrm, dim=-1)
        L = rng.normal(0, 1, 3)
        L[2] = -abs(L[2]) - 0.5                                          # light from behind the camera, roughly
        L = torch.tensor(L / np.linalg.norm(L), dtype=torch.float32, device=dev)
        lam = (nrm @ -L).abs()
        amb = float(rng.uniform(0.35, 0.7))
        spec = (lam ** float(rng.uniform(8, 40))) * float(rng.uniform(0.0, 0.8)) * (tfl[..., 0] < 0.5)
        img = col * (amb + (1 - amb) * lam[..., None]) + spec[..., None]
        img = torch.where(hit[..., None], img, torch.full_like(img, float(rng.uniform(0.5, 0.9))))
        img = torch.nn.functional.avg_pool2d(img.permute(2, 0, 1)[None], self.ss)[0].permute(1, 2, 0)
        depth = torch.where(hit, dep, torch.full_like(dep, 1e3))
        depth = -torch.nn.functional.max_pool2d(-depth[None, None], self.ss)[0, 0]   # nearest surface per pixel
        return img.clamp(0, 1).cpu().numpy(), depth.cpu().numpy()


def post(img, rng):
    """Camera-like changes: exposure, colour, shadow blobs, blur, noise; -> uint8 BGR at the network size."""
    img = img * rng.uniform(0.75, 1.25) * rng.uniform(0.9, 1.1, 3)
    for _ in range(rng.integers(0, 3)):                               # soft shadows (drawn at 1/8 size)
        mask = np.zeros((H // 8, W // 8), np.float32)
        cv2.ellipse(mask, (int(rng.uniform(0, W / 8)), int(rng.uniform(0, H / 8))),
                    (int(rng.uniform(6, 38)), int(rng.uniform(4, 25))), rng.uniform(0, 180), 0, 360, 1.0, -1)
        mask = cv2.resize(cv2.GaussianBlur(mask, (0, 0), 5), (W, H))
        img = img * (1 - rng.uniform(0.1, 0.4) * mask[..., None])
    s = rng.uniform(0.0, 1.4)
    if s > 0.3:
        img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), s)
    img = img + rng.normal(0, rng.uniform(0.005, 0.03), img.shape)
    img = cv2.resize(np.clip(img, 0, 1).astype(np.float32), (OW, OH - 4), interpolation=cv2.INTER_AREA)
    img = np.r_[img, np.zeros((4, OW, 3), np.float32)]
    out = (img[..., ::-1] * 255).astype(np.uint8)
    if rng.random() < 0.5:
        _, enc = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(60, 95))])
        out = cv2.imdecode(enc, 1)
    return out


SX, SY = OW / W, (OH - 4) / H                         # stored labels: network pixels


def labels(ts, rail, T_cam_world, depth):
    out = []
    for t in ts:
        r, tp = root_tip(t, rail)
        uv, z = project(np.stack([r, tp]), T_cam_world)
        rec = dict(q=t['q'], row=t['row'], num=t['num'])
        for name, p, zz in (('root', uv[0], z[0]), ('tip', uv[1], z[1])):
            inside = zz > 0.05 and 0 <= p[0] < W and 0 <= p[1] < H
            vis = False
            if inside:
                i, j = int(p[1]), int(p[0])
                patch = depth[max(0, i - 2):i + 3, max(0, j - 2):j + 3]
                vis = bool(np.median(patch) > zz - 0.012)                # nothing > 1.2 cm in front of it
            rec[name] = [float(p[0] * SX), float(p[1] * SY)] if inside else None
            rec[name + '_vis'] = vis
        out.append(rec)
    return out


def render_set(out, n, seed):
    out.mkdir(parents=True, exist_ok=False)
    parts, B = basket_parts()
    tines, rail = tine_table(parts, B), B['pins']['rail_z']
    poses = camera_poses()
    rd = Renderer()
    rng = np.random.default_rng(seed)
    meta = []
    for i in range(n):
        m, ts = build_scene(rng, parts, B, tines)
        Tbc = sample_camera(rng, poses)
        add_held(m, rng, Tbc)
        if i % 20 == 0:                                              # a new table texture every 20 samples
            tex = table_texture(rng)
        Tcw = np.linalg.inv(Tbc)
        img, depth = rd.render(m, Tcw, tex, rng)
        cv2.imwrite(str(out / f'{i:06d}.jpg'), post(img, rng), [cv2.IMWRITE_JPEG_QUALITY, 95])
        meta.append(dict(i=i, tines=labels(ts, rail, Tcw, depth)))
        if i % 500 == 0:
            print(f'{i}/{n}', flush=True)
    (out / 'labels.json').write_text(json.dumps(dict(size=[OW, OH], scale=[SX, SY], samples=meta)))


def sheet(d, out=None, k=8):
    """sim samples (labels drawn) beside real wrist photos at the network size."""
    d = Path(d)
    lab = json.loads((d / 'labels.json').read_text())['samples']
    tiles = []
    for s in lab[:k]:
        img = cv2.imread(str(d / f"{s['i']:06d}.jpg"))
        for t in s['tines']:
            for name, col in (('root', (255, 120, 0)), ('tip', (0, 200, 0))):
                if t[name] and t[name + '_vis']:
                    cv2.circle(img, tuple(np.int32(t[name])), 3, col, -1)
        tiles.append(img)
    reals = [p for p, *_ in real_tests()] + sorted(glob.glob(str(DATA / 'real_pipeline/*/wrist_pre.raw.png')))[-4:]
    for p in reals[:k]:
        r = cv2.resize(cv2.imread(p), (OW, OH - 4), interpolation=cv2.INTER_AREA)
        tiles.append(np.r_[r, np.zeros((4, OW, 3), np.uint8)])
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles) - len(tiles) % 4, 4)]
    big = np.vstack(rows)
    out = out or d / 'sheet.jpg'
    cv2.imwrite(str(out), cv2.resize(big, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    print(out)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('cmd', choices=('check', 'render', 'sheet'))
    a.add_argument('dir', nargs='?')
    a.add_argument('--out', default=str(DATA / 'slot_cnn/check'))
    a.add_argument('--n', type=int, default=100)
    a.add_argument('--seed', type=int, default=0)
    a = a.parse_args()
    if a.cmd == 'check':
        check(Path(a.out))
    elif a.cmd == 'render':
        render_set(Path(a.out), a.n, a.seed)
    else:
        sheet(a.dir)
