"""Basket insertion training (2026-10-13).

    python rl/basket/train.py bc --data IL/data_basket_v2_<date> [--seed 1] [--preflight-only]
    python rl/basket/train.py rl --bc <bc.pt>           (residual PPO, Isaac; added after env.py / expert.py)

bc: behaviour cloning of the sim feedback expert (model.BCPolicy, MSE on the 6 TCP steps + BCE on 'open', AdamW,
cosine lr).  Safeguards (config.yaml preflight / bc): data check, tiny-batch overfit test, early stop on the
validation loss (patience), best-val checkpoint.  Curves: W&B (one run per seed, seeds grouped) + a PNG at the end.
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model as M  # noqa: E402

CFG = yaml.safe_load((HERE / 'config.yaml').read_text())
ROOT = HERE.parents[1]


# ---------------------------------------------------------------- data
def load_demos(data_dir):
    """batch_*.npz with obs [E, T, 17], act [E, T, 7] (m / rad per step + open), mask [E, T], success [E].
    -> list of episodes (obs [t, 17], act [t, 7]) of the successful demos only."""
    eps = []
    for f in sorted(glob.glob(str(Path(data_dir) / 'batch_*.npz'))):
        z = np.load(f)
        for e in range(len(z['success'])):
            if z['success'][e]:
                t = z['mask'][e].astype(bool)
                eps.append((z['obs'][e][t].astype(np.float32), z['act'][e][t].astype(np.float32)))
    return eps


def check_data(eps):
    """Preflight 1: refuse data that cannot train (wrong dims, NaN, actions beyond the speed caps, too few demos)."""
    n_obs, n_act = len(CFG['obs']['names']), len(CFG['action']['names'])
    step_max = np.array(CFG['action']['step_max'])
    problems = []
    if len(eps) < 100:
        problems.append(f'only {len(eps)} successful episodes')
    for i, (o, a) in enumerate(eps):
        if o.shape[1] != n_obs or a.shape[1] != n_act:
            problems.append(f'episode {i}: obs {o.shape} act {a.shape}')
            break
        if not (np.isfinite(o).all() and np.isfinite(a).all()):
            problems.append(f'episode {i}: NaN / inf')
            break
        if (np.abs(a[:, :6]) > step_max * 1.001).any():
            problems.append(f'episode {i}: action beyond the speed caps')
            break
    frames = sum(len(o) for o, _ in eps)
    print(f'[data] {len(eps)} episodes, {frames} frames' + (f'  PROBLEMS: {problems}' if problems else '  ok'), flush=True)
    if problems:
        raise SystemExit('data check failed: ' + '; '.join(problems))


def tensors(eps, idx, device):
    o = np.concatenate([eps[i][0] for i in idx])
    a = np.concatenate([eps[i][1] for i in idx])
    step_max = np.array(CFG['action']['step_max'], np.float32)
    cont = np.clip(a[:, :6] / step_max, -1, 1)                      # targets in step units
    above = o[:, CFG['obs']['names'].index('above_rail')]
    w = np.where(above <= CFG['bc']['near_pin_m'], CFG['bc']['near_pin_weight'], 1.0).astype(np.float32)
    T = lambda x: torch.as_tensor(x, device=device)
    return T(o), T(cont), T(a[:, 6]), T(w)


# ---------------------------------------------------------------- loss / metrics
def loss_fn(policy, o, cont, grip, w):
    pc, pg = policy(o)
    mse = ((pc - cont) ** 2).mean(-1)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pg, grip, reduction='none')
    return ((mse + CFG['bc']['grip_loss_weight'] * bce) * w).sum() / w.sum(), mse.mean(), bce.mean()


@torch.no_grad()
def metrics(policy, o, cont, grip, w, mean_baseline):
    loss, mse, bce = loss_fn(policy, o, cont, grip, w)
    pc, pg = policy(o)
    tol = torch.as_tensor(CFG['bc']['acc_tol'], device=o.device) / policy.step_max
    acc = ((pc - cont).abs() <= tol).all(-1).float().mean()
    grip_acc = ((pg > 0).float() == grip).float().mean()
    r2 = 1 - mse / mean_baseline                                     # vs "always predict the mean action"
    return dict(loss=loss.item(), mse=mse.item(), bce=bce.item(), acc=acc.item(), grip_acc=grip_acc.item(), r2=r2.item())


# ---------------------------------------------------------------- tracking
def start_wandb(name, group, config):
    """One W&B run per training job (offline when not logged in).  Returns None if wandb is missing."""
    try:
        import wandb
    except ImportError:
        print('[wandb] not installed: curves only as PNG', flush=True)
        return None
    mode = os.environ.get('WANDB_MODE', CFG['wandb']['mode'])
    if mode == 'auto':
        mode = 'online' if wandb.api.api_key else 'offline'
    return wandb.init(project=CFG['wandb']['project'], name=name, group=group, config=config, mode=mode,
                      dir=str(ROOT / 'rl' / 'logs' / 'basket'), reinit=True)


def plot_curves(hist, path):
    """Train vs val loss on one axis, accuracy / R2 and lr beside it (style: no grid, no top/right spines, no title)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    s = [h['step'] for h in hist]
    ax[0].plot(s, [h['train/loss'] for h in hist], label='train')
    ax[0].plot(s, [h['val/loss'] for h in hist], label='val')
    ax[0].set_yscale('log'); ax[0].set_xlabel('step'); ax[0].set_ylabel('loss'); ax[0].legend(frameon=False)
    ax[1].plot(s, [h['val/acc'] for h in hist], label='action acc')
    ax[1].plot(s, [h['val/grip_acc'] for h in hist], label='grip acc')
    ax[1].plot(s, [h['val/r2'] for h in hist], label='R2')
    ax[1].set_xlabel('step'); ax[1].set_ylim(0, 1.02); ax[1].legend(frameon=False)
    ax[2].plot(s, [h['lr'] for h in hist]); ax[2].set_xlabel('step'); ax[2].set_ylabel('lr'); ax[2].set_yscale('log')
    for a in ax:
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- BC
def overfit_test(eps, device, seed):
    """Preflight 2: a fresh net must memorise a tiny batch; if it cannot, the pipeline (not the data size) is wrong."""
    c = CFG['bc']
    torch.manual_seed(seed)
    p = M.BCPolicy(len(CFG['obs']['names']), CFG['action']['step_max'], c['hidden'], c['activation']).to(device)
    o, cont, grip, w = tensors(eps, range(len(eps)), device)
    sel = torch.randperm(len(o), device=device)[:CFG['preflight']['overfit_frames']]
    p.norm.fit(o)
    opt = torch.optim.AdamW(p.parameters(), lr=c['lr'])
    for _ in range(CFG['preflight']['overfit_steps']):
        loss, mse, _ = loss_fn(p, o[sel], cont[sel], grip[sel], w[sel])
        opt.zero_grad(); loss.backward(); opt.step()
    ok = mse.item() <= CFG['preflight']['overfit_max_loss']
    print(f'[preflight] tiny-batch overfit: mse {mse.item():.4f} (need <= {CFG["preflight"]["overfit_max_loss"]}) '
          f'-> {"ok" if ok else "FAILED"}', flush=True)
    if not ok:
        raise SystemExit('preflight failed: the net cannot memorise 256 frames (check obs / action / loss)')


def run_bc(args):
    c = CFG['bc']
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    eps = load_demos(args.data)
    check_data(eps)
    overfit_test(eps, dev, args.seed)
    if args.preflight_only:
        return
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(eps))
    n_val = max(1, int(len(eps) * c['val_fraction']))
    tr, va = order[n_val:], order[:n_val]
    o, cont, grip, w = tensors(eps, tr, dev)
    vo, vcont, vgrip, vw = tensors(eps, va, dev)
    base_val = ((vcont - cont.mean(0)) ** 2).mean()                  # R2 reference

    torch.manual_seed(args.seed)
    policy = M.BCPolicy(o.shape[1], CFG['action']['step_max'], c['hidden'], c['activation']).to(dev)
    policy.norm.fit(o)
    opt = torch.optim.AdamW(policy.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c['max_steps'], eta_min=c['lr_min'])
    out = ROOT / 'rl' / 'logs' / 'basket' / f'bc_{Path(args.data).name}' / f'seed{args.seed}_{time.strftime("%Y%m%d_%H%M%S")}'
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(obs_dim=o.shape[1], step_max=CFG['action']['step_max'], hidden=c['hidden'], activation=c['activation'],
                obs_names=CFG['obs']['names'], data=str(args.data), seed=args.seed)
    run = start_wandb(f'bc_seed{args.seed}', f'bc_{Path(args.data).name}', dict(bc=c, data=str(args.data), seed=args.seed,
                                                                                 train_eps=len(tr), val_eps=len(va)))
    hist, best, best_step, t0 = [], math.inf, 0, time.time()
    log_every = 250
    for step in range(1, c['max_steps'] + 1):
        idx = torch.randint(0, len(o), (c['batch'],), device=dev)
        loss, _, _ = loss_fn(policy, o[idx], cont[idx], grip[idx], w[idx])
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), c['grad_clip'])
        opt.step(); sched.step()
        if step % log_every == 0 or step == c['max_steps']:
            policy.eval()
            tm = metrics(policy, o[idx], cont[idx], grip[idx], w[idx], base_val)
            vm = metrics(policy, vo, vcont, vgrip, vw, base_val)
            policy.train()
            rec = {'step': step, 'epoch': step * c['batch'] / len(o), 'lr': sched.get_last_lr()[0], 'grad_norm': gn.item(),
                   **{f'train/{k}': v for k, v in tm.items()}, **{f'val/{k}': v for k, v in vm.items()}}
            hist.append(rec)
            if run:
                run.log(rec, step=step)
            if vm['loss'] < best * (1 - c['min_delta']):
                best, best_step = vm['loss'], step
                M.save(out / 'best.pt', policy, dict(meta, step=step, val=vm))
            if step % 2500 == 0:
                print(f'[bc] step {step} ep {rec["epoch"]:.1f} train {tm["loss"]:.4f} val {vm["loss"]:.4f} '
                      f'acc {vm["acc"]:.3f} grip {vm["grip_acc"]:.3f} R2 {vm["r2"]:.3f} lr {rec["lr"]:.1e} '
                      f'({time.time() - t0:.0f} s)', flush=True)
            if step - best_step >= c['patience_steps']:
                print(f'[bc] early stop: no val improvement since step {best_step}', flush=True)
                break
    M.save(out / 'last.pt', policy, dict(meta, step=step))
    plot_curves(hist, out / 'curves.png')
    (out / 'history.json').write_text(json.dumps(hist))
    last = hist[-1]
    verdict = ('underfit (R2 < 0.5)' if last['val/r2'] < 0.5 else
               'overfit (val rose > 20 % above its best)' if last['val/loss'] > best * 1.2 else 'ok')
    summary = dict(best_val=best, best_step=best_step, stopped=step, verdict=verdict, out=str(out))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(f'[bc] done: {summary}', flush=True)
    if run:
        run.summary.update(summary)
        run.finish()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=('bc', 'rl'))
    p.add_argument('--data', help='bc: demo folder with batch_*.npz (expert.py collect)')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--preflight-only', action='store_true', help='data check + tiny-batch overfit test, then exit')
    p.add_argument('--bc', help='rl: BC checkpoint (best.pt)')
    p.add_argument('--max_steps', type=int, default=None, help='bc: override config (tests)')
    return p.parse_args()


if __name__ == '__main__':
    a = parse_args()
    if a.max_steps:
        CFG['bc']['max_steps'] = a.max_steps
    if a.mode == 'bc':
        run_bc(a)
    else:
        raise SystemExit('rl: residual PPO is added together with env.py (needs the Isaac env)')
