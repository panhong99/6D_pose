"""Checked, simulation-only overnight wide BC -> residual PPO pipeline.

Run with Isaac Sim's Python. Reuses complete demos only after fresh verification.
BC qualification is a training-entry check, not approval for a physical robot.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from rl_utils import RL_DIR, load_config

ROOT = RL_DIR.parent
CHECKS = RL_DIR / "night_checks.py"
ACTIVE = None
RUN_TOKEN = None


def stop_active():
    global ACTIVE
    # run_isaac creates independent process groups. An inherited job token also
    # identifies orphaned GPU children if their curriculum parent already exited.
    pids = set()
    if RUN_TOKEN:
        marker = f"NIGHT_PIPELINE_RUN={RUN_TOKEN}".encode()
        for path in Path("/proc").glob("[0-9]*/environ"):
            try:
                if marker in path.read_bytes().split(b"\0"):
                    pids.add(int(path.parent.name))
            except (OSError, ValueError):
                pass
    if ACTIVE is not None and ACTIVE.poll() is None:
        pids.add(ACTIVE.pid)
    groups = set()
    for pid in pids:
        try:
            group = os.getpgid(pid)
            if group != os.getpgrp():
                groups.add(group)
        except ProcessLookupError:
            pass
    for sig, wait in ((signal.SIGINT, 15), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        for group in groups:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            alive = set()
            for pid in pids:
                try:
                    if Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z":
                        alive.add(pid)
                except OSError:
                    pass
            pids = alive
            if not pids:
                return
            time.sleep(0.25)


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def checked_json(*args):
    result = subprocess.run([sys.executable, str(CHECKS), *map(str, args)],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def main():
    global ACTIVE, RUN_TOKEN
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", default="m1013_2f85_il_residual_wide_stable")
    p.add_argument("--min-bc-success", type=float, default=0.85)
    p.add_argument("--max-bc-collision", type=float, default=0.02)
    args = p.parse_args()
    if not (0 < args.min_bc_success <= 1 and 0 <= args.max_bc_collision < 1):
        p.error("invalid BC qualification thresholds")
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    lock = (RL_DIR / "logs" / "wide_night.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = RL_DIR / "logs" / f"night_codex_{stamp}"
    out.mkdir(parents=True)
    RUN_TOKEN = f"{out.name}_{os.getpid()}"
    (RL_DIR / "logs" / "night_codex_latest.json").write_text(json.dumps({"directory": str(out), "pid": os.getpid()}))
    config_path = RL_DIR / "config.yaml"
    initial_config = config_path.read_bytes()
    cfg = load_config()
    cfg["paths"]["experiment"] = args.experiment
    state_path = RL_DIR / "logs" / args.experiment / "state.json"
    if state_path.exists():
        raise RuntimeError(f"refusing scratch run with existing curriculum state: {state_path}")
    protected = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                 (RL_DIR / "train.py", RL_DIR / "plate_env.py", RL_DIR / "rl_utils.py", ROOT / "IL/collect.py",
                  ROOT / "calibration/config.yaml", ROOT / "calibration/pipeline.py")}
    (out / "config_before.yaml").write_bytes(initial_config)
    (out / "protected_hashes.json").write_text(json.dumps(protected, indent=2))
    status = {"experiment": args.experiment, "started": stamp, "status": "starting", "pid": os.getpid(),
              "bc_min_success": args.min_bc_success, "bc_max_collision": args.max_bc_collision}
    run = None

    def say(message):
        line = f"- {time.strftime('%F %T')} {message}"
        print(line, flush=True)
        with (out / "pipeline.log").open("a") as f:
            f.write(line + "\n")

    def save_status(**values):
        status.update(values)
        (out / "status.json").write_text(json.dumps(status, indent=2))

    env = os.environ.copy()
    env.pop("CONDA_PREFIX", None)
    env.pop("CONDA_DEFAULT_ENV", None)
    env.update(TERM="xterm", WANDB_MODE="online", NIGHT_PIPELINE_RUN=RUN_TOKEN)
    # W&B SDK subprocesses and PPO children use the same authenticated online mode.
    os.environ["WANDB_MODE"] = "online"
    isaac = Path(cfg["paths"]["isaaclab"]).expanduser() / "isaaclab.sh"

    def execute(command, log, timeout=None):
        global ACTIVE
        say(f"start {log.name}")
        with log.open("w") as f:
            ACTIVE = subprocess.Popen(list(map(str, command)), cwd=isaac.parent, env=env,
                                      stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = ACTIVE.wait(timeout=timeout)
                if rc:
                    stop_active()
            except BaseException:
                stop_active()
                raise
            finally:
                ACTIVE = None
        if rc:
            raise subprocess.CalledProcessError(rc, command)
        say(f"complete {log.name}")

    try:
        # Existing scalar tracking is retained. Stop before training if online preflight fails.
        from train import start_tracking
        tracking_dir = out / "wandb_preflight"
        tracking_dir.mkdir()
        run = start_tracking(cfg, SimpleNamespace(stage=2, num_envs=512), tracking_dir,
                             job_type="preflight", role="preflight")
        if run is None or run.settings.mode != "online":
            raise RuntimeError("online W&B preflight failed; no training started")
        run.define_metric("BC/seed")
        run.define_metric("BC/*", step_metric="BC/seed")
        save_status(status="verifying_data", wandb_url=run.url, wandb_id=run.id)
        say(f"W&B online: {run.url}")
        qualified = []
        for name, seed, dart in (("data_wide2_claude", 3001, 0.0), ("data_wide2_claude_dart", 3002, 0.2)):
            data = ROOT / "IL" / name
            if not (data / "manifest.json").exists():
                execute([isaac, "-p", ROOT / "IL/collect.py", "--headless", "--episodes", 200,
                         "--num_envs", 32, "--seed", seed, "--max_batches", 12, "--dart_noise", dart,
                         "--slow_zone_m", 0.02, "--output", data], out / f"collect_{name}.log", timeout=1200)
            checked_json("data", data, "--seed", seed, "--dart", dart, "--episodes", 200)
            verify_log = out / f"verify_{name}.log"
            execute([isaac, "-p", ROOT / "IL/collect.py", "--verify", "--headless", "--output", data],
                    verify_log, timeout=300)
            report = checked_json("data", data, "--seed", seed, "--dart", dart, "--episodes", 200,
                                  "--verify-log", verify_log)
            (out / f"data_{name}.json").write_text(json.dumps(report, indent=2))
            run.log({f"Data/{name}/episodes": 200, "Progress/iteration": 0})
            say(f"verified complete 200 episodes: {name}")
        save_status(status="bc_training")
        for seed in (1, 2, 3):
            bc_log = out / f"bc_seed{seed}.log"
            code = (
                f"import sys; sys.path.insert(0,{str(RL_DIR)!r}); sys.argv=['train.py','bc']; "
                "import train; from rl_utils import load_config; cfg=load_config(); "
                "cfg['paths']['experiment']='m1013_2f85_rim_slow'; "
                "cfg['bc']['data']=['../IL/data_wide2_claude','../IL/data_wide2_claude_dart']; "
                f"cfg['bc']['steps']=4000; cfg['ppo']['seed']={seed}; train.run_bc(None,cfg)"
            )
            try:
                execute([sys.executable, "-c", code], bc_log, timeout=1200)
                ckpt = checked_json("checkpoint", bc_log)["checkpoint"]
                gate_log = out / f"bc_gate_seed{seed}.log"
                execute([isaac, "-p", RL_DIR / "eval.py", "gate", "--headless", "--stage", 2,
                         "--residual", "off", "--num_envs", 512, "--checkpoint", ckpt], gate_log, timeout=900)
                # Record all evaluable seed results, even candidates rejected for their numerical score.
                lines = [s.split("EVAL_JSON ", 1)[1] for s in gate_log.read_text().splitlines() if "EVAL_JSON " in s]
                if len(lines) == 1:
                    measured = json.loads(lines[0])
                    metrics = {f"BC/{k}": v for k, v in measured.items() if type(v) in (int, float)}
                    import torch
                    val_mse = torch.load(ckpt, map_location="cpu", weights_only=False).get("infos", {}).get("bc", {}).get("val_mse")
                    if val_mse is not None:
                        metrics["BC/val_mse"] = val_mse
                    run.log({"BC/seed": seed, **metrics})
                result = checked_json("gate", gate_log, "--checkpoint", ckpt, "--episodes", 512,
                                      "--min-success", args.min_bc_success, "--max-collision", args.max_bc_collision)
                candidate = out / f"qualified_seed{seed}.json"
                candidate.write_text(json.dumps(result, indent=2))
                qualified.append(candidate)
                run.log({"BC/seed": seed, "BC/qualified": 1})
                say(f"qualified BC seed {seed}: success={result['success']:.4f}, collision={result['collision']:.4f}")
            except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
                run.log({"BC/seed": seed, "BC/qualified": 0})
                say(f"BC seed {seed} rejected: {e}; stderr={getattr(e, 'stderr', None)}")
        if not qualified:
            raise RuntimeError("no BC passed qualification; stopping before config update and PPO")
        selected = checked_json("select", *qualified, "--out", out / "selected_bc.json")
        if config_path.read_bytes() != initial_config or state_path.exists():
            raise RuntimeError("config or curriculum state changed concurrently; stopping")
        for path, sha in protected.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
                raise RuntimeError(f"protected source changed concurrently: {path}")
        # Update only the qualified BC checkpoint and fresh experiment name. Keep the reviewed
        # PPO stability settings and the separated old run intact.
        text = initial_config.decode()
        text, nbase = re.subn(r"^  base:.*$", f"  base: {json.dumps(selected['checkpoint'])}   # qualified wide BC", text, flags=re.M)
        text, nexp = re.subn(r"^  experiment:.*$", f"  experiment: {args.experiment}      # checked overnight wide run", text, flags=re.M)
        if nbase != 1 or nexp != 1:
            raise RuntimeError("config field update is ambiguous")
        import yaml
        expected = yaml.safe_load(initial_config)
        expected["residual"]["base"] = selected["checkpoint"]
        expected["paths"]["experiment"] = args.experiment
        if yaml.safe_load(text) != expected:
            raise RuntimeError("config update changed additional settings")
        temporary = config_path.with_suffix(".night.tmp")
        temporary.write_text(text)
        os.replace(temporary, config_path)
        (out / "config_training.yaml").write_text(text)
        save_status(status="residual_training", selected_bc=selected)
        run.log({"Night/selected_bc_success": selected["success"], "Night/selected_bc_collision": selected["collision"]})
        run.finish()
        run = None
        say(f"start residual PPO: {selected['checkpoint']}; warm-up150 + maximum3000 iterations")
        execute([isaac, "-p", RL_DIR / "train.py", "curriculum", "--stage", 2, "--scratch",
                 "--critic_warmup", 150, "--max_iters", 3000, "--experiment", args.experiment],
                out / "curriculum_driver.log")
        state = json.loads(state_path.read_text())
        passed = state.get("stages", {}).get("2", {}).get("passed_checkpoint")
        save_status(status="passed" if passed else "not_passed", passed_checkpoint=passed)
        say(f"curriculum {'PASSED' if passed else 'NOT PASSED'}; physical deployment still requires motion review")
        return 0 if passed else 1
    except BaseException as e:
        save_status(status="stopped" if isinstance(e, KeyboardInterrupt) else "failed", error=str(e))
        say(f"STOP: {type(e).__name__}: {e}")
        return 130 if isinstance(e, KeyboardInterrupt) else 1
    finally:
        stop_active()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
