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
