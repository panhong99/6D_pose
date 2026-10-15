"""Slot-frame observation / action for the basket insertion (2026-10-13), shared by expert.py, eval.py and the RL.

Slot frame: origin = the slot (gap centre line on the basket floor, env._slot), x = n (along the pin rows),
y = u (across), z up.  Observation (config.yaml obs.names, 17): plate lowest rim point (3), plate normal (3),
tool approach axis (3), TCP (3), lowest point above the tine rail (1), tool force (3), released (1).
Force: the sim has no flange force sensor yet -> zeros (documented; the real robot only streams it at rest).
Action: slot-frame TCP step (3 m + 3 rad axis-angle) + open flag, executed through the env's SafeDiffIK action
(position-holding differential IK with the 5 cm/s / 10 deg/s caps).
"""
import torch
import yaml
from pathlib import Path

CFG = yaml.safe_load((Path(__file__).resolve().parent / 'config.yaml').read_text())


def slot_R(env):
    """(n, 3, 3) columns = slot axes in the env frame."""
    s = env._slot
    c, si = torch.cos(s[:, 3]), torch.sin(s[:, 3])
    z = torch.zeros_like(c)
    n = torch.stack([c, si, z], 1)
    u = torch.stack([-si, c, z], 1)
    up = torch.stack([z, z, torch.ones_like(c)], 1)
    return torch.stack([n, u, up], 2)


def observe(env, released):
    """-> (n, 17) observation and the basket_state dict it came from."""
    import basket_env as be
    import plate_env as pe
    st = be.basket_state(env)
    R = slot_R(env)
    loc = lambda v: torch.einsum('nij,ni->nj', R, v)                       # noqa: E731  env -> slot frame
    s = env._slot
    plate = env.scene['plate']
    low_pt = torch.stack([st['d_bottom'], st['d_u'], st['low']], 1)   # lowest rim point (n, u) + its height
    nrm = loc(pe.body_axis(plate.data.root_quat_w, 2))
    tp, tq = pe.tcp_pose(env)
    tool = loc(pe.body_axis(tq, 2))
    tcp = loc(tp - env.scene.env_origins - s[:, :3])
    rail = be.B['pins'].get('rail_z', 0.049)
    above_rail = (st['low'] - rail)[:, None]
    force = torch.zeros(env.num_envs, 3, device=env.device)
    obs = torch.cat([low_pt, nrm, tool, tcp, above_rail, force, released.float()[:, None]], 1)
    return obs, st


def rim_row_crossings(centre, normal, rows_y, radius):
    """(10/15) Where the plate rim crosses each tine row, all in the slot frame (x along the rows, y across, z up).
    centre (n,3) rim-circle centre, normal (n,3) plate normal, rows_y: row offsets (k,) -> (x (n,k), z (n,k), ok (n,k)):
    the LOWER of the two rim points in the plane y = row_y; ok = False where the rim does not reach that row.
    x = 0 is the gap line, so x is the bottom's along-row alignment error at that row (L / R)."""
    down = torch.tensor([0.0, 0.0, -1.0], device=centre.device).expand_as(centre)
    a = down - normal * (down * normal).sum(1, keepdim=True)               # plate-plane direction toward the lowest rim
    a = a / a.norm(dim=1, keepdim=True).clamp(min=1e-9)
    b = torch.cross(normal, a, dim=1)
    rows = torch.as_tensor(rows_y, dtype=centre.dtype, device=centre.device)
    A, B = radius * a[:, 1:2], radius * b[:, 1:2]                          # y(theta) = cy + A cos + B sin
    C = rows[None, :] - centre[:, 1:2]
    rho = torch.sqrt(A ** 2 + B ** 2).clamp(min=1e-9)
    ok = C.abs() <= rho
    base, half = torch.atan2(B, A), torch.acos((C / rho).clamp(-1.0, 1.0))
    pts = []
    for th in (base + half, base - half):
        p = centre[:, None, :] + radius * (torch.cos(th)[..., None] * a[:, None, :] + torch.sin(th)[..., None] * b[:, None, :])
        pts.append(p)
    lower = torch.where((pts[0][..., 2] <= pts[1][..., 2])[..., None], pts[0], pts[1])
    return lower[..., 0], lower[..., 2], ok


def alignment(env, st, gap_bias=None):
    """(10/15) Stage-4 alignment observation (n, 6): rim x at row L, rim height above the rail at L, same for R (m),
    plate-plane error to the slot plane (rad), rows reached (0..1).  gap_bias (n, 2) [m] shifts the gap the policy
    BELIEVES in (tine-model / marker error, domain randomisation); rewards use the true geometry (bias None)."""
    import basket_env as be
    a = CFG['align']
    R = slot_R(env)
    loc = lambda v: torch.einsum('nij,ni->nj', R, v)                       # noqa: E731  env -> slot frame
    centre, normal = loc(st['rel']), loc(st['up'])
    if gap_bias is not None:
        centre = centre - torch.cat([gap_bias, torch.zeros_like(gap_bias[:, :1])], 1)
    x, z, ok = rim_row_crossings(centre, normal, a['rows_y'], a['rim_radius'])
    rail = be.B['pins'].get('rail_z', 0.049)
    z = z - rail
    return torch.stack([x[:, 0], z[:, 0], x[:, 1], z[:, 1], torch.deg2rad(st['normal_err']), ok.float().mean(1)], 1)


def to_env_action(env, step_slot, open_flag):
    """Slot-frame step (n, 6) [m, rad] + open (n,) -> env action (n, 7) for the SafeDiffIK arm term."""
    import plate_env as pe
    R = slot_R(env)
    dp = torch.einsum('nij,nj->ni', R, step_slot[:, :3])
    dr = torch.einsum('nij,nj->ni', R, step_slot[:, 3:6])
    act = torch.zeros(env.num_envs, 7, device=env.device)
    act[:, :3] = dp / pe.A['scale']
    act[:, 3:6] = dr / pe.A['scale']
    act[:, 6] = torch.where(open_flag > 0.5, torch.ones_like(open_flag), -torch.ones_like(open_flag))
    return act


def clip_step(step):
    m = torch.as_tensor(CFG['action']['step_max'], device=step.device)
    return torch.maximum(torch.minimum(step, m), -m)
