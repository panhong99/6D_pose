"""PPO training of the plate rim grasp (rsl_rl), run through Isaac Lab's python.

  ISAAC=~/IsaacLab/isaaclab.sh
  $ISAAC -p rl/train.py curriculum --stage 2 --init <model.pt>     # chunks of PPO + deterministic >= 95 % gate
  $ISAAC -p rl/train.py curriculum --stage 2 --scratch              # same, random initial weights
  $ISAAC -p rl/train.py chunk --stage 2 --iters 500 [--resume <model.pt>]   # one Isaac process (used by curriculum)
  $ISAAC -p rl/train.py bc                                          # IL demos -> actor checkpoint (prints BC_CHECKPOINT)
  $ISAAC -p rl/train.py curriculum --stage 2 --init <bc model_0.pt> --critic_warmup 100 --experiment <name>

curriculum (no Isaac app in this process): trains `chunk` iterations per Isaac process (resume from the last checkpoint,
watchdog + retries).  After each chunk the saved checkpoint with the best 50-iteration training success is
gate-evaluated with `eval.py gate` (deterministic, every env one episode) if that success reaches proxy_gate, and the
next chunk rolls back to it when the last checkpoint collapsed by >= rollback_drop.  The stage passes at >= gate.  It stops after --last_stage
(default: the start stage) - the Human looks at the GUI (`eval.py watch`) before the next stage starts.
Progress: rl/logs/<experiment>/status.md and state.json.  Checkpoints: <log_root>/<experiment>/<date>_stage<k>/.
Numbers: rl/config.yaml (curriculum, ppo).
"""
import argparse
import json
import time
from pathlib import Path

from rl_utils import (RL_DIR, best_saved_checkpoint, import_checkpoint, latest_checkpoint, load_config, new_run_dir,
                      newest_stage_checkpoint, read_tagged_json, run_isaac, train_success)


def start_tracking(cfg, args, log_dir, job_type="train", *, role="train", tag=None, checkpoint=None):
    """Use a persistent ID for training chunks and separate descriptive IDs for each evaluation."""
    tracking = cfg.get("tracking", {})
    if not tracking.get("enabled", False):
        return None
    import os
    import sys
    local_packages = RL_DIR.parent / ".runtime" / "wandb"
    if local_packages.is_dir():
        sys.path.append(str(local_packages))
    os.environ.setdefault("WANDB_CACHE_DIR", str(RL_DIR.parent / ".runtime" / "wandb_cache"))
    import wandb

    mode = os.environ.get("WANDB_MODE", tracking.get("mode", "auto"))
    if mode == "auto":
        mode = "online" if wandb.api.api_key else "offline"
    exp = cfg["paths"]["experiment"]
    method = "BC" if job_type == "behaviour_cloning" or cfg.get("algorithm") == "BC" else (
        "BCresidual" if cfg.get("residual", {}).get("enabled", False) else "PPOonly")
    short_exp = exp.replace("m1013_2f85_il_residual_", "").replace("m1013_2f85_", "")
    group = f"robot_{'BC' if method == 'BC' else 'PPO'}__{method}__{short_exp}__stage{args.stage}"
    name = f"{group}__{role}" + (f"__{tag}" if tag else "")
    id_dir = RL_DIR / "logs" / exp / "wandb_ids"
    id_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    id_file = id_dir / f"{role}_stage{args.stage}{suffix}.txt"
    if not id_file.is_file():
        import uuid
        id_file.write_text(uuid.uuid4().hex[:12])
    kw = dict(project=tracking["project"], entity=tracking.get("entity") or None,
              id=id_file.read_text().strip(), resume="allow", name=name, group=group, job_type=job_type,
              dir=str(log_dir), settings=wandb.Settings(init_timeout=30))
    try:
        run = wandb.init(mode=mode, **kw)
    except Exception as e:                                  # e.g. no default entity: keep training, sync later
        print(f"[tracking] W&B {mode} init failed ({e}); logging offline - upload later with `wandb sync`", flush=True)
        mode = "offline"
        run = wandb.init(mode=mode, **kw)
    log_unit="bc_update" if job_type=="behaviour_cloning" else ("evaluation_set" if job_type=="evaluation" else "ppo_update")
    run.config.update({"task": cfg, "stage": args.stage, "num_envs": args.num_envs, "log_unit": log_unit,
                       "run_role": role, "run_tag": tag or "", "checkpoint": str(checkpoint) if checkpoint else ""},
                      allow_val_change=True)                # resumed runs: chunks may differ (e.g. warm-up)
    run.define_metric("Progress/iteration")
    run.define_metric("*", step_metric="Progress/iteration")
    print(f"[tracking] W&B mode={mode}, run={run.url or run.id}, directory={log_dir}", flush=True)
    return run


def tracked_runner(base_class, cfg, args, log_dir):
    """Keep native TensorBoard logs; batch W&B metrics once per PPO update, without uploading models/code."""
    run = start_tracking(cfg, args, log_dir, job_type="training", role="train")
    if run is None:
        return base_class, None

    class MetricWriter:
        def __init__(self, writer):
            self.writer, self.metrics = writer, {}

        def add_scalar(self, tag, value, global_step=None, *extra, **kwargs):
            self.writer.add_scalar(tag, value, global_step, *extra, **kwargs)
            # Native runner also emits reward vs wall time; omit those duplicate x-axis series from W&B.
            if not tag.endswith("/time"):
                self.metrics[tag] = float(value)

        def __getattr__(self, name):
            return getattr(self.writer, name)

    class TrackedRunner(base_class):
        def _prepare_logging_writer(self):
            super()._prepare_logging_writer()
            if self.writer is not None and not isinstance(self.writer, MetricWriter):
                self.writer = MetricWriter(self.writer)

        def log(self, locs, *extra, **kwargs):
            super().log(locs, *extra, **kwargs)
            metrics = dict(self.writer.metrics)
            self.writer.metrics.clear()
            metrics.update({"Progress/iteration": locs["it"], "Progress/chunk_env_steps": self.tot_timesteps,
                            "Progress/chunk_wall_s": self.tot_time,
                            "Progress/env_steps_per_update": self.num_steps_per_env * self.env.num_envs})
            run.log(metrics)

    return TrackedRunner, run


# ---------------------------------------------------------------- one Isaac training process

def run_chunk(args, cfg):
    import sys
    sys.stdout.reconfigure(line_buffering=True)             # watchdog reads the log; app.close() drops buffered output
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app

    import shutil
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.io import dump_yaml
    from rsl_rl.runners import OnPolicyRunner

    import plate_env
    from rl_utils import make_vecenv, ppo_runner_cfg

    agent = ppo_runner_cfg(cfg, args.stage)
    agent.max_iterations = args.iters
    env_cfg = plate_env.make_env_cfg(args.stage, args.num_envs)
    env_cfg.seed = agent.seed
    log_dir = new_run_dir(cfg, args.stage)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = ManagerBasedRLEnv(env_cfg)
    runner_class, tracking_run = tracked_runner(OnPolicyRunner, cfg, args, log_dir)
    runner = runner_class(make_vecenv(env, cfg), agent.to_dict(), log_dir=str(log_dir), device=env.device)
    if args.resume:
        print(f"[train] resume from {args.resume}", flush=True)
        runner.load(args.resume)
        # rsl_rl restores optimizer param groups from the checkpoint.  Pin resumed groups
        # back to the declared config LR before learning so stale/adaptive values cannot leak.
        configured_lr = float(agent.algorithm.learning_rate)
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = configured_lr
        runner.alg.learning_rate = configured_lr
        print(f"[train] optimizer LR reset to config: {configured_lr:g} "
              f"(schedule={agent.algorithm.schedule})", flush=True)
    elif cfg.get("residual", {}).get("enabled", False):     # residual starts at zero = exactly the BC policy
        import torch
        pol, res = runner.alg.policy, cfg["residual"]
        with torch.no_grad():
            pol.actor[-1].weight.zero_()
            pol.actor[-1].bias.zero_()
            if hasattr(pol, "std"):
                pol.std.fill_(res["init_std"])
        print(f"[train] residual on {res['base']} (scale {res['scale']}), zero-initialised", flush=True)
    if args.freeze_actor:                                   # critic warm-up: PPO updates only the value function
        pol = runner.alg.policy
        for p in list(pol.actor.parameters()) + [getattr(pol, "std", None), getattr(pol, "log_std", None)]:
            if p is not None:
                p.requires_grad_(False)
        print("[train] actor and action noise frozen (critic warm-up)", flush=True)
    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent)
    shutil.copy2(RL_DIR / "config.yaml", log_dir / "params" / "config.yaml")
    try:
        runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)
    finally:
        if runner.writer is not None:
            runner.writer.close()
        if tracking_run is not None:
            tracking_run.finish()
        sys.stdout.flush()
        env.close()
        app.close()


# ---------------------------------------------------------------- behaviour cloning (IL -> RL start point)

def run_bc(args, cfg):
    """Fit the PPO actor (same MLP) to the IL demonstrations (obs -> 7-dim action), write an rsl_rl checkpoint
    (fresh critic, action noise bc.init_std) that `curriculum --init` can resume.  No Isaac app: torch only."""
    import numpy as np
    import torch
    from torch import nn
    if args is None:  # Existing experiment drivers call run_bc(None, cfg).
        args = argparse.Namespace(stage=2,num_envs=0)
    b, p = cfg["bc"], cfg["ppo"]
    torch.manual_seed(p["seed"])
    split = {"train": [], "validation": []}
    data_selection=[]
    for d in b["data"]:
        d = (RL_DIR / d).resolve()
        if not (d / "manifest.json").is_file():
            print(f"[bc] skip {d} (no manifest)", flush=True)
            continue
        man = json.loads((d / "manifest.json").read_text())
        for e in man["episodes"]:
            with np.load(d / e["file"]) as z:
                if b.get('max_demo_tcp_speed_mps') is not None:
                    dt=man.get('policy_dt_s')
                    if not dt or not all(k in z.files for k in ('tcp_pose','next_tcp_pose')):
                        raise RuntimeError('Speed-filtered BC requires measured TCP transitions and policy dt')
                    speed=np.linalg.norm(z['next_tcp_pose'][:,:3]-z['tcp_pose'][:,:3],axis=1)/float(dt)
                    maximum=float(speed.max())
                    accepted=np.isfinite(maximum) and maximum<=b['max_demo_tcp_speed_mps']+b.get('demo_speed_tolerance_mps',1e-5)
                    selection=dict(directory=str(d),file=e['file'],split=e['split'],max_tcp_speed_mps=maximum)
                    if b.get('max_demo_arm_joint_speed_deg_s') is not None:
                        if 'arm_joint_speed_deg_s' in z.files:
                            joint_max=float(np.abs(z['arm_joint_speed_deg_s']).max())
                            selection['arm_speed_source']='recorded_joint_velocity'
                        else:
                            # This collector schema has 8 joint positions, then velocities scaled by 0.5.
                            if z['next_obs'].shape[1]!=46:raise RuntimeError('Joint speed fallback requires the known 46-dim schema')
                            joint_max=float(np.degrees(np.abs(z['next_obs'][:,8:14])*2).max())
                            selection['arm_speed_source']='next_obs_8_14_inverse_scale_0.5'
                        accepted=accepted and np.isfinite(joint_max) and joint_max<=b['max_demo_arm_joint_speed_deg_s']+b.get('demo_joint_speed_tolerance_deg_s',.01)
                        selection['max_arm_joint_speed_deg_s']=joint_max
                    selection['accepted']=bool(accepted);data_selection.append(selection)
                    if not accepted:continue
                ph = z["phase"].astype(np.float32) if "phase" in z.files else np.full(len(z["obs"]), 2.0, np.float32)
                split[e["split"]].append((z["obs"].astype(np.float32), z["actions"].astype(np.float32), ph))
        print(f"[bc] {d.name}: {len(man['episodes'])} episodes ({man['status']})", flush=True)
    if not split["train"]:
        raise SystemExit("no demonstrations found (config.yaml bc.data)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cat = lambda eps, i: torch.from_numpy(np.concatenate([e[i] for e in eps])).to(dev)  # noqa: E731
    xtr, ytr = cat(split["train"], 0), cat(split["train"], 1)
    xva, yva = (cat(split["validation"], 0), cat(split["validation"], 1)) if split["validation"] else (xtr, ytr)
    print(f"[bc] frames train {len(xtr)} validation {len(xva)}", flush=True)
    args.stage, args.num_envs = 2, 0
    out = new_run_dir(cfg, 2).with_name(new_run_dir(cfg, 2).name + "_bc")
    out.mkdir(parents=True, exist_ok=True)
    import yaml
    (out / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
    if data_selection:
        (out/'demo_selection.json').write_text(json.dumps(data_selection,indent=2))
        print(f'[bc] speed filter accepted {sum(x["accepted"] for x in data_selection)}/{len(data_selection)} episodes',flush=True)
    run = start_tracking(cfg,args,out,job_type="behaviour_cloning",role="train",tag=f"seed{p['seed']}")

    def mlp(out):
        layers, n = [], xtr.shape[1]
        for h in p["actor_hidden_dims"]:
            layers += [nn.Linear(n, h), nn.ELU()]
            n = h
        return nn.Sequential(*layers, nn.Linear(n, out))

    # copycat fix: the previous action (obs 36:43) and the joint velocities (obs 8:16) predict the expert's next command
    # almost perfectly, so BC copies "keep doing what you did" and overshoots the rim target with open fingers until the
    # gripper body hits the plate (2026-10-07: 97-100 % collisions).  Train with them zeroed and zero their first-layer
    # weights, so the BC policy ignores them (PPO may learn to use them later).
    for lo, hi in b["ignore_obs"]:
        xtr[:, lo:hi] = 0.0
        xva[:, lo:hi] = 0.0
    # small metre-scale goal signals barely move an unnormalised MLP: scale them while training, fold the scale into the
    # first layer afterwards (same function on raw observations)
    obs_scale = [(lo, hi, k) for lo, hi, k in b.get("obs_scale", []) if hi <= xtr.shape[1]]
    for lo, hi, k in obs_scale:
        xtr[:, lo:hi] *= k
        xva[:, lo:hi] *= k
    # approach / close frames are ~25 % of the data but decide the grasp: sample them phase_weight times more often
    ph = cat(split["train"], 2)
    w = torch.where(ph < 2, float(b.get("phase_weight", 1.0)), 1.0)
    actor = mlp(ytr.shape[1]).to(dev)
    opt = torch.optim.AdamW(actor.parameters(), lr=b["lr"], weight_decay=b["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, b["steps"])
    for step in range(1, b["steps"] + 1):
        idx = torch.multinomial(w, b["batch"], replacement=True)
        loss = nn.functional.mse_loss(actor(xtr[idx]), ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == b["steps"]:
            with torch.no_grad():
                pred = actor(xva)
                val = nn.functional.mse_loss(pred, yva).item()
                grip = (torch.sign(pred[:, 6]) == torch.sign(yva[:, 6])).float().mean().item()
            print(f"[bc] step {step} train mse {loss.item():.5f} val mse {val:.5f} grip sign acc {grip:.4f}", flush=True)
            if run:
                run.log({"Progress/iteration":step,"BC/train_mse":float(loss.item()),
                         "BC/validation_mse":val,"BC/gripper_sign_accuracy":grip})
    with torch.no_grad():
        for lo, hi in b["ignore_obs"]:
            actor[0].weight[:, lo:hi] = 0.0
        for lo, hi, k in obs_scale:
            actor[0].weight[:, lo:hi] *= k
    critic = mlp(1)
    state = {"std": torch.full((ytr.shape[1],), float(b["init_std"]))}
    state.update({f"actor.{k}": v.detach().cpu() for k, v in actor.state_dict().items()})
    state.update({f"critic.{k}": v.detach().cpu() for k, v in critic.state_dict().items()})
    template = torch.load(b.get("template_checkpoint") or newest_stage_checkpoint(cfg, 2), map_location="cpu", weights_only=False)
    assert list(template["model_state_dict"]) == list(state), "parameter names differ from an rsl_rl checkpoint"
    groups = template["optimizer_state_dict"]["param_groups"]                 # only the group layout is reused
    for g in groups:
        g["lr"] = p["learning_rate"]
    ckpt = out / "model_0.pt"
    torch.save({"model_state_dict": state, "optimizer_state_dict": {"state": {}, "param_groups": groups},
                "iter": 0, "infos": {"bc": {"data": b["data"], "frames": len(xtr), "val_mse": val}}}, ckpt)
    print(f"BC_CHECKPOINT {ckpt}", flush=True)
    if run:
        run.summary.update({"checkpoint":str(ckpt),"BC/validation_mse":val,"BC/gripper_sign_accuracy":grip})
        run.finish()


# ---------------------------------------------------------------- curriculum driver

class Curriculum:
    def __init__(self, args, cfg):
        self.args, self.cfg, self.cc = args, cfg, cfg["curriculum"]
        for k in ("chunk", "max_iters", "num_envs"):
            if getattr(args, k) is not None:
                self.cc[k] = getattr(args, k)
        if args.experiment:
            cfg["paths"]["experiment"] = args.experiment
        self.out = RL_DIR / "logs" / cfg["paths"]["experiment"]
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.is_file() else {"stages": {}}

    def log(self, msg):
        line = f"- {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(self.out / "status.md", "a") as f:
            f.write(line + "\n")

    def save(self):
        self.state_path.write_text(json.dumps(self.state, indent=1))

    def start_checkpoint(self):
        a = self.args
        if a.scratch:
            return None
        ckpt = Path(a.init) if a.init else newest_stage_checkpoint(self.cfg, a.stage)
        if ckpt is None:
            raise SystemExit(f"no checkpoint for stage {a.stage}: pass --init <model.pt> or --scratch")
        return import_checkpoint(self.cfg, ckpt)

    def train_chunk(self, stage, ckpt, tag, iters=None, freeze_actor=False):
        """One Isaac training run with retries; returns the new checkpoint (or None) and the log path."""
        tlog = self.out / f"train_{tag}.log"
        cmd = [RL_DIR / "train.py", "chunk", "--stage", stage, "--iters", iters or self.cc["chunk"],
               "--num_envs", self.cc["num_envs"], "--experiment", self.cfg["paths"]["experiment"]]
        cmd += ["--freeze_actor"] if freeze_actor else []
        for attempt in range(1, self.cc["retries"] + 1):
            rc = run_isaac(self.cfg, cmd + (["--resume", ckpt] if ckpt else []), tlog,
                           watch_marker="Learning iteration", log=self.log)
            new = newest_stage_checkpoint(self.cfg, stage)
            if rc == 0 and new is not None and new != ckpt:
                return new, tlog
            self.log(f"stage {stage}: training chunk failed (rc {rc}, attempt {attempt}/{self.cc['retries']}), see {tlog}")
            if new is not None and new != ckpt:                       # resume from whatever the crashed run saved
                ckpt = new
            time.sleep(60)
        return None, tlog

    def gate_eval(self, stage, ckpt, tag):
        elog = self.out / f"eval_{tag}.log"
        run_isaac(self.cfg, [RL_DIR / "eval.py", "gate", "--headless", "--stage", stage, "--checkpoint", ckpt,
                             "--num_envs", self.cc["eval_envs"]], elog, log=self.log)
        res = read_tagged_json(elog, "EVAL_JSON") or {"success": 0.0, "error": f"no EVAL_JSON in {elog}"}
        self.log(f"stage {stage}: deterministic eval {ckpt.parent.name}/{ckpt.name}: "
                 + json.dumps({k: res.get(k) for k in ("success", "collision", "plate_lost", "time_out", "mean_shield_steps")}))
        from types import SimpleNamespace
        eval_dir = self.out / f"wandb_eval_{tag}"
        eval_dir.mkdir(exist_ok=True)
        run = start_tracking(self.cfg, SimpleNamespace(stage=stage, resume=str(ckpt),
                                                      num_envs=self.cc["eval_envs"]), eval_dir,
                             job_type="deterministic_gate", role="gate", tag=tag, checkpoint=ckpt)
        if run is not None:
            try:
                metrics = {f"Eval/{k}": v for k, v in res.items() if isinstance(v, (int, float))}
                metrics["Progress/iteration"] = int(ckpt.stem.split("_")[-1])
                run.log(metrics)
            finally:
                run.finish()
        return res

    def run_stage(self, stage, ckpt):
        st = self.state["stages"].setdefault(str(stage), {"iters": 0, "evals": []})
        self.log(f"stage {stage} start (init {ckpt.parent.name + '/' + ckpt.name if ckpt else 'scratch'}, "
                 f"chunk {self.cc['chunk']}, envs {self.cc['num_envs']})")
        if self.args.eval_first and ckpt is not None:
            res = self.gate_eval(stage, ckpt, f"s{stage}_resume")
            st["evals"].append(res)
            st["baseline_success"] = res.get("success")
            st["baseline_checkpoint"] = str(ckpt)
            self.save()
            if res.get("success", 0.0) >= self.cc["gate"]:
                return self.passed(stage, st, ckpt)
        if self.args.critic_warmup and not st.get("critic_warmed"):            # also from a zero residual
            new, tlog = self.train_chunk(stage, ckpt, f"s{stage}_critic_warmup", self.args.critic_warmup, freeze_actor=True)
            if new is None:
                return None
            ckpt, st["critic_warmed"] = new, True
            self.log(f"stage {stage}: critic warm-up {self.args.critic_warmup} iters (actor frozen), training success "
                     f"{train_success(tlog):.3f}, ckpt {ckpt.parent.name}/{ckpt.name}")
            self.save()
        while st["iters"] < self.cc["max_iters"]:
            tag = f"s{stage}_{st['iters']:05d}"
            t0 = time.time()
            new, tlog = self.train_chunk(stage, ckpt, tag)
            if new is None:
                self.log(f"stage {stage}: giving up after {self.cc['retries']} failed attempts")
                return None
            ckpt = new
            st["iters"] += self.cc["chunk"]
            st["last_checkpoint"] = str(ckpt)
            sr = train_success(tlog)
            best, best_sr = best_saved_checkpoint(tlog, ckpt.parent)
            self.log(f"stage {stage}: +{self.cc['chunk']} iters ({st['iters']} total, {(time.time() - t0) / 60:.0f} min), "
                     f"training success (last 50 it) {sr:.3f}, ckpt {ckpt.parent.name}/{ckpt.name}; "
                     f"best saved {best.name if best else '-'} ({best_sr:.3f})")
            self.save()
            # PPO here rises then collapses within a chunk (2026-10-07): judge the chunk by its best saved checkpoint
            candidate = best if best is not None else ckpt
            # Always inspect the latest chunk when its training proxy collapsed; otherwise
            # the curriculum can miss a deterministic regression and spend the full budget.
            if best is not None and best_sr >= self.cc["proxy_gate"]:
                eval_ckpt = best
            else:
                eval_ckpt = ckpt
            if eval_ckpt is not None:
                res = self.gate_eval(stage, eval_ckpt, tag)
                st["evals"].append(res)
                self.save()
                baseline = st.get("baseline_success")
                floor = max(self.cc["min_success"], baseline - self.cc["max_regression"]) if baseline is not None else self.cc["min_success"]
                if res.get("collision", 0.0) > 0 or res.get("plate_lost", 0.0) > 0 or res.get("success", 0.0) < floor:
                    st["stopped_for_regression"] = {"checkpoint": str(eval_ckpt), "success": res.get("success"),
                                                     "floor": floor, "collision": res.get("collision"),
                                                     "plate_lost": res.get("plate_lost")}
                    self.save()
                    self.log(f"stage {stage}: stopping after deterministic regression/safety failure; "
                             f"success {res.get('success', 0):.3f}, floor {floor:.3f}, "
                             f"collision {res.get('collision', 0):.3f}, plate_lost {res.get('plate_lost', 0):.3f}")
                    return None
                if res.get("success", 0.0) >= self.cc["gate"]:
                    return self.passed(stage, st, eval_ckpt)
            if best is not None and best != ckpt and best_sr - sr >= self.cc["rollback_drop"]:
                self.log(f"stage {stage}: rollback - next chunk resumes from {best.parent.name}/{best.name} "
                         f"({best_sr:.3f}) instead of the last checkpoint ({sr:.3f})")
                ckpt = best
        self.log(f"stage {stage} NOT passed within {st['iters']} iterations (gate {self.cc['gate']})")
        return None

    def passed(self, stage, st, ckpt):
        st["passed_checkpoint"] = str(ckpt)
        self.save()
        self.log(f"stage {stage} PASSED with {ckpt.parent.name}/{ckpt.name} - show the Human the GUI before the next "
                 f"stage: isaaclab.sh -p rl/eval.py watch --stage {stage} --checkpoint {ckpt}")
        return ckpt

    def run(self):
        ckpt = self.start_checkpoint()
        last = self.args.last_stage or self.args.stage
        for stage in range(self.args.stage, last + 1):
            ckpt = self.run_stage(stage, ckpt)
            if ckpt is None:
                break


# ---------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",default=None,help="candidate config file; do not modify shared config.yaml")
    sub = p.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("curriculum", help="chunked training + deterministic gate (driver, no Isaac app)")
    c.add_argument("--stage", type=int, default=2, choices=(2, 3))
    c.add_argument("--last_stage", type=int, default=None, help="default: same as --stage (Human GUI check between stages)")
    c.add_argument("--init", default=None, help="checkpoint to start from (default: newest checkpoint of this stage)")
    c.add_argument("--scratch", action="store_true", help="random initial weights")
    c.add_argument("--eval_first", action="store_true", help="gate-evaluate the start checkpoint before training")
    c.add_argument("--critic_warmup", type=int, default=0, help="first train only the critic this many iterations "
                   "(actor frozen; use after `bc`)")
    c.add_argument("--experiment", default=None, help="override config.yaml paths.experiment (separate runs / status)")
    for k in ("chunk", "max_iters", "num_envs"):
        c.add_argument(f"--{k}", type=int, default=None, help="override config.yaml curriculum." + k)
    b=sub.add_parser("bc", help="behaviour cloning from the IL demonstrations (config.yaml bc) -> rsl_rl checkpoint")
    b.add_argument("--experiment",default=None)
    b.add_argument("--seed",type=int,default=None)
    k = sub.add_parser("chunk", help="one Isaac training process (headless)")
    k.add_argument("--stage", type=int, default=2, choices=(2, 3))
    k.add_argument("--iters", type=int, required=True)
    k.add_argument("--num_envs", type=int, default=4096)
    k.add_argument("--resume", default=None)
    k.add_argument("--freeze_actor", action="store_true", help="critic warm-up: only the value function learns")
    k.add_argument("--experiment", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    if getattr(args,"seed",None) is not None:
        cfg["ppo"]["seed"]=args.seed
    if getattr(args, "experiment", None):
        cfg["paths"]["experiment"] = args.experiment
    if args.mode == "chunk":
        run_chunk(args, cfg)
    elif args.mode == "bc":
        run_bc(args, cfg)
    else:
        Curriculum(args, cfg).run()
