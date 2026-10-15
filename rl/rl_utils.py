"""Shared helpers for the plate rim-grasp RL code (no Isaac app needed; torch only for the shield).

  config      load_config, rl_path                       rl/config.yaml access
  shield      SafetyShield                               voxel keep-out check of IK joint targets (frame / table)
  runs        stage_run_dir, latest_checkpoint, newest_stage_checkpoint, train_success, import_checkpoint
  isaac       run_isaac                                  launch `isaaclab.sh -p <script>` with a log watchdog
  policy      ppo_runner_cfg, load_policy                rsl_rl runner config from config.yaml / load a checkpoint
"""
import json
import math
import os
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

RL_DIR = Path(__file__).resolve().parent
ROTATION_HOLD_DEFAULT = False


def integrate_spatial_quat(hold_quat, delta_rotvec):
    """Integrate an env-frame axis-angle delta on the left of scalar-first quaternions (torch tensors)."""
    import torch
    delta_rotvec = delta_rotvec.to(dtype=hold_quat.dtype, device=hold_quat.device)
    angle = torch.linalg.vector_norm(delta_rotvec, dim=-1, keepdim=True)
    axis = delta_rotvec / angle.clamp(min=1e-9)
    half = angle * 0.5
    delta = torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)
    w1, v1 = delta[..., :1], delta[..., 1:]
    w2, v2 = hold_quat[..., :1], hold_quat[..., 1:]
    return torch.cat([w1 * w2 - (v1 * v2).sum(-1, keepdim=True),
                      w1 * v2 + w2 * v1 + torch.cross(v1, v2, dim=-1)], dim=-1)


# ---------------------------------------------------------------- config

def load_config(path=RL_DIR / "config.yaml"):
    return yaml.safe_load(Path(path).read_text())


def rl_path(p):
    """config path -> absolute Path (relative to rl/, ~ expanded)."""
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else (RL_DIR / p).resolve()


# ---------------------------------------------------------------- safety shield

def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def quat_to_R(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_stl_vertices(path):
    data = Path(path).read_bytes()
    count = int(np.frombuffer(data[80:84], '<u4')[0])
    if 84 + 50 * count == len(data):
        rec = np.frombuffer(data[84:], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), count=count)
        return rec['v'].reshape(-1, 3).astype(np.float32) * 0.001
    return np.asarray([[float(v) for v in line.split()[1:4]] for line in data.decode().splitlines()
                       if line.strip().startswith('vertex')], dtype=np.float32) * 0.001


def gripper_surface_points(cfg):
    """Open Robotiq 2F-85 visual meshes (MJCF) + the 20 mm coupling, in the link_6 frame, voxel-decimated."""
    gdir = rl_path(cfg["paths"]["gripper_mjcf_dir"])
    root = ET.parse(gdir / '2f85.xml').getroot()
    mesh_files = {Path(m.get('file')).stem: m.get('file') for m in root.iter('mesh') if m.get('file')}
    points = []

    def walk(body, parent_T):
        T = parent_T.copy()
        T[:3, :3] = parent_T[:3, :3] @ quat_to_R(np.fromstring(body.get('quat', '1 0 0 0'), sep=' '))
        T[:3, 3] = parent_T[:3, :3] @ np.fromstring(body.get('pos', '0 0 0'), sep=' ') + parent_T[:3, 3]
        for geom in body.findall('geom'):
            name = geom.get('mesh')
            if geom.get('class') != 'visual' or name not in mesh_files:
                continue
            verts = read_stl_vertices(gdir / 'assets' / mesh_files[name])
            gR = quat_to_R(np.fromstring(geom.get('quat', '1 0 0 0'), sep=' '))
            gpos = np.fromstring(geom.get('pos', '0 0 0'), sep=' ')
            points.append(verts @ gR.T @ T[:3, :3].T + (T[:3, :3] @ gpos + T[:3, 3]))
        for child in body.findall('body'):
            walk(child, T)

    coupling = cfg["robot"]["coupling_m"]
    mount = np.eye(4)
    mount[:3, 3] = (0.0, 0.0, coupling)
    walk(root.find('worldbody').find('body'), mount)
    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)                 # coupling ~ 40 mm radius cylinder
    ring = np.column_stack([0.040 * np.cos(angles), 0.040 * np.sin(angles)])
    points.append(np.concatenate([np.column_stack([ring, np.full(len(ring), z)]) for z in (0.0, coupling / 2, coupling)]))
    points = np.concatenate(points).astype(np.float32)
    keys = np.floor(points / cfg["shield"]["gripper_surface_voxel"]).astype(np.int32)
    _, keep = np.unique(keys, axis=0, return_index=True)
    return points[keep]


class SafetyShield:
    """Batched torch FK of the M1013 (URDF revolute joints) + per-link surface samples looked up in 1 cm distance fields
    (env_setup.py sdf).  safe(q) is False when any point is closer than frame_margin to the steel frame or table_margin
    to the table; the env then keeps the last safe joint target."""

    def __init__(self, device, cfg=None):
        import torch
        import torch.nn.functional as F
        self.torch, self.F = torch, F
        cfg = cfg or load_config()
        self.frame_margin, self.table_margin = cfg["shield"]["frame_margin"], cfg["shield"]["table_margin"]
        self.device = device
        origins, axes = [], []
        urdf = Path(cfg["paths"]["m1013_share"]) / "urdf" / "m1013.urdf"
        for j in ET.parse(urdf).getroot().findall('joint'):
            if j.get('type') != 'revolute':
                continue
            o = j.find('origin')
            T = np.eye(4)
            T[:3, :3] = _rpy_to_R(*[float(v) for v in o.get('rpy', '0 0 0').split()])
            T[:3, 3] = [float(v) for v in o.get('xyz', '0 0 0').split()]
            origins.append(T)
            axes.append([float(v) for v in j.find('axis').get('xyz').split()])
        assert len(origins) == 6
        self.origins = torch.tensor(np.array(origins), dtype=torch.float32, device=device)       # (6, 4, 4)
        self.axes = torch.tensor(np.array(axes), dtype=torch.float32, device=device)            # (6, 3)
        d = np.load(rl_path(cfg["paths"]["cell_sdf"]))
        self.lo = torch.tensor(d['lo'], dtype=torch.float32, device=device)
        self.res = float(d['res'])
        # grid_sample wants (N, C, D, H, W) = (1, 1, z, y, x)
        self.sdf = {k: torch.tensor(d[k], dtype=torch.float32, device=device).permute(2, 1, 0)[None, None].contiguous()
                    for k in ('frame', 'table')}
        self.size = torch.tensor(d['frame'].shape, dtype=torch.float32, device=device)            # (nx, ny, nz)
        pts = [torch.tensor(d[f'link_{i}'], dtype=torch.float32, device=device) for i in range(1, 7)]
        pts[5] = torch.cat([pts[5], torch.tensor(gripper_surface_points(cfg), dtype=torch.float32, device=device)])
        self.link_pts = pts

    def link_frames(self, q):
        """q (N, 6) rad -> list of 6 link transforms (N, 4, 4) in the base frame (Rodrigues per revolute joint)."""
        torch = self.torch
        n = q.shape[0]
        T = torch.eye(4, device=self.device).expand(n, 4, 4)
        out = []
        for i in range(6):
            a, th = self.axes[i], q[:, i]
            K = torch.zeros(n, 3, 3, device=self.device)
            K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -a[2], a[1], -a[0]
            K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = a[2], -a[1], a[0]
            s, c = torch.sin(th)[:, None, None], torch.cos(th)[:, None, None]
            J = torch.eye(4, device=self.device).repeat(n, 1, 1)
            J[:, :3, :3] = torch.eye(3, device=self.device) + s * K + (1 - c) * (K @ K)
            T = T @ self.origins[i] @ J
            out.append(T)
        return out

    def _lookup(self, grid, p):
        """Trilinear distance at points p (N, M, 3); outside the grid = border value."""
        idx = (p - self.lo) / self.res - 0.5
        g = 2.0 * idx / (self.size - 1) - 1.0
        v = self.F.grid_sample(grid.expand(p.shape[0], -1, -1, -1, -1), g[:, :, None, None, :], mode='bilinear',
                               padding_mode='border', align_corners=True)
        return v[:, 0, :, 0, 0]

    def clearance(self, q):
        """(N, 6) joint angles -> (frame clearance, table clearance), each (N,) metres."""
        frames = self.link_frames(q)
        world = self.torch.cat([pts @ T[:, :3, :3].transpose(1, 2) + T[:, None, :3, 3]
                                for pts, T in zip(self.link_pts, frames)], dim=1)
        return self._lookup(self.sdf['frame'], world).amin(1), self._lookup(self.sdf['table'], world).amin(1)

    def safe(self, q):
        f, t = self.clearance(q)
        return (f >= self.frame_margin) & (t >= self.table_margin)


# ---------------------------------------------------------------- runs / checkpoints

def runs_root(cfg):
    return rl_path(cfg["paths"]["log_root"]) / cfg["paths"]["experiment"]


def new_run_dir(cfg, stage):
    return runs_root(cfg) / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_stage{stage}"


def checkpoint_iter(path):
    return int(re.findall(r"\d+", Path(path).stem)[0])


def latest_checkpoint(run_dir):
    models = sorted(Path(run_dir).glob("model_*.pt"), key=checkpoint_iter)
    return models[-1] if models else None


def newest_stage_checkpoint(cfg, stage):
    models = [m for r in runs_root(cfg).glob(f"*_stage{stage}") for m in r.glob("model_*.pt")]
    return max(models, key=lambda m: m.stat().st_mtime) if models else None


def import_checkpoint(cfg, ckpt):
    """Copy a checkpoint from another experiment folder into this one (keeps its run-folder name)."""
    ckpt = Path(ckpt)
    if ckpt.parent.parent == runs_root(cfg):
        return ckpt
    dst = runs_root(cfg) / ckpt.parent.name / ckpt.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(ckpt, dst)
    return dst


def strip_ansi(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def train_success(logfile, last=50):
    """Mean `Episode_Termination/success` over the last `last` iterations of an rsl_rl training log."""
    vals = [float(v) for v in re.findall(r"Episode_Termination/success:\s*([\d.]+)",
                                         strip_ansi(Path(logfile).read_text(errors="ignore")))]
    return sum(vals[-last:]) / max(1, len(vals[-last:])) if vals else 0.0


def success_by_iteration(logfile):
    """{iteration: Episode_Termination/success} from an rsl_rl training log."""
    out = {}
    for block in strip_ansi(Path(logfile).read_text(errors="ignore")).split("Learning iteration ")[1:]:
        it = re.match(r"\s*(\d+)/", block)
        m = re.search(r"Episode_Termination/success:\s*([\d.]+)", block)
        if it and m:
            out[int(it.group(1))] = float(m.group(1))
    return out


def best_saved_checkpoint(logfile, run_dir, window=50):
    """(checkpoint, mean training success over the `window` iterations before it) for the best model_*.pt of one run."""
    sr = success_by_iteration(logfile)
    best = (None, -1.0)
    for m in Path(run_dir).glob("model_*.pt"):
        k = checkpoint_iter(m)
        vals = [sr[i] for i in range(k - window + 1, k + 1) if i in sr]
        if len(vals) >= window // 2 and sum(vals) / len(vals) > best[1]:
            best = (m, sum(vals) / len(vals))
    return best


def read_tagged_json(logfile, tag):
    m = re.search(rf"{tag} (\{{.*\}})", Path(logfile).read_text(errors="ignore"))
    return json.loads(m.group(1)) if m else None


# ---------------------------------------------------------------- isaac subprocess

def run_isaac(cfg, args, logfile, watch_marker=None, watchdog_s=None, log=print):
    """Run `isaaclab.sh -p <args>` (own process group, log to file).  If watch_marker is given, the run is killed when the
    count of that marker in the log stops growing for watchdog_s seconds (2026-10-03: a chunk hung 2.8 days)."""
    lab = rl_path(cfg["paths"]["isaaclab"])
    env = dict(os.environ, TERM="xterm")
    for k in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV"):
        env.pop(k, None)
    watchdog_s = watchdog_s or cfg["curriculum"]["watchdog_s"]
    with open(logfile, "w") as f:
        proc = subprocess.Popen([str(lab / "isaaclab.sh"), "-p"] + [str(a) for a in args], cwd=lab, env=env,
                                stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        last, changed = -1, time.time()
        while proc.poll() is None:
            time.sleep(30)
            data = Path(logfile).read_bytes()
            progress = data.count(watch_marker.encode()) if watch_marker else len(data)
            if progress != last:
                last, changed = progress, time.time()
            if time.time() - changed > watchdog_s:
                log(f"watchdog: {Path(str(args[0])).name} silent for {watchdog_s // 60} min - killing it")
                os.killpg(proc.pid, 9)
                proc.wait()
                return -9
        return proc.returncode


# ---------------------------------------------------------------- policy (needs a running Isaac app)

def ppo_runner_cfg(cfg, stage):
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
    p = dict(cfg["ppo"])
    if cfg.get("residual", {}).get("enabled", False):
        p.update(cfg["residual"].get("ppo", {}))
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=p["num_steps_per_env"], max_iterations=cfg["curriculum"]["chunk"], save_interval=p["save_interval"],
        experiment_name=cfg["paths"]["experiment"], run_name=f"stage{stage}", seed=p["seed"],
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=p["init_noise_std"], actor_obs_normalization=False, critic_obs_normalization=False,
            actor_hidden_dims=p["actor_hidden_dims"], critic_hidden_dims=p["critic_hidden_dims"], activation="elu"),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=p["clip_param"], entropy_coef=p["entropy_coef"],
            num_learning_epochs=p["num_learning_epochs"], num_mini_batches=p["num_mini_batches"],
            learning_rate=p["learning_rate"], schedule=p.get("schedule", "adaptive"), gamma=p["gamma"], lam=p["lam"],
            desired_kl=p["desired_kl"], max_grad_norm=p["max_grad_norm"]))


def base_actor(cfg, checkpoint, device):
    """The frozen BC actor of a residual setup: MLP with the checkpoint's actor.* weights, no gradients."""
    import torch
    from torch import nn
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    dims = [sd["actor.0.weight"].shape[1]] + cfg["ppo"]["actor_hidden_dims"] + [sd["actor.6.weight"].shape[0]]
    layers = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1])] + ([nn.ELU()] if i < len(dims) - 2 else [])
    net = nn.Sequential(*layers)
    net.load_state_dict({k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")})
    return net.to(device).eval().requires_grad_(False)


def make_vecenv(env, cfg):
    """rsl_rl view of the env.  With config residual.enabled the PPO action is a correction on top of a frozen BC
    policy: executed = base(obs) + residual.scale * ppo_action (IL -> RL that PPO cannot erase, 2026-10-07)."""
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    res = cfg.get("residual", {})
    if not res.get("enabled", False):
        return RslRlVecEnvWrapper(env)

    class ResidualVecEnv(RslRlVecEnvWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.base = base_actor(cfg, rl_path(res["base"]), self.unwrapped.device)
            self.scale = float(res["scale"])
            self._obs = None

        def _keep(self, obs):
            self._obs = obs["policy"]
            return obs

        def reset(self):
            obs, extras = super().reset()
            return self._keep(obs), extras

        def get_observations(self):
            return self._keep(super().get_observations())

        def step(self, actions):
            if self._obs is None:
                self.get_observations()
            import torch
            with torch.no_grad():
                executed = self.base(self._obs) + self.scale * actions
            obs, rew, dones, extras = super().step(executed)
            return self._keep(obs), rew, dones, extras

    return ResidualVecEnv(env)


def load_policy(env, cfg, stage, checkpoint):
    """(wrapped vec env, runner, inference policy) for a checkpoint; env is a ManagerBasedRLEnv.
    With residual.enabled the returned policy is the correction; venv.step adds the BC base itself."""
    from rsl_rl.runners import OnPolicyRunner
    venv = make_vecenv(env, cfg)
    runner = OnPolicyRunner(venv, ppo_runner_cfg(cfg, stage).to_dict(), log_dir=None, device=env.device)
    runner.load(str(checkpoint))
    return venv, runner, runner.get_inference_policy(device=env.device)
