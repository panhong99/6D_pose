"""Staged curriculum driver (plain python3, no Isaac imports): train stage 1..3 in chunks, gate on >= 95 % success.

For each stage: train CHUNK iterations (4096 envs, headless), resuming from the latest checkpoint (stage k > 1 starts from
the stage k-1 checkpoint that passed).  When the training success rate over the last 50 iterations is >= PROXY_GATE, run
rl/eval_policy.py (deterministic, 2048 episodes); the stage passes only if that success rate is >= 0.95.  A stage that
does not pass within MAX_ITERS stops the whole curriculum (later stages never start).  After each pass a play video is
recorded.  Progress: rl/logs/curriculum_v4/status.md and state.json.

  python3 rl/curriculum.py [--start_stage 1] [--chunk 1000] [--max_iters 15000]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from experiment_settings import VERSION, EXPERIMENT_NAME

HERE = Path(__file__).resolve().parent
LAB = Path(os.environ.get("ISAACLAB_PATH", Path.home() / "IsaacLab"))
VER = VERSION
RUNS = LAB / "logs" / "rsl_rl" / EXPERIMENT_NAME
OUT = HERE / "logs" / f"curriculum_{VER}"
GATE = 0.95
PROXY_GATE = 0.93
WATCHDOG_S = 900                  # kill an Isaac run whose log has not grown for 15 min

ap = argparse.ArgumentParser()
ap.add_argument("--start_stage", type=int, default=1)
ap.add_argument("--init_checkpoint", default=None, help="checkpoint to start --start_stage from (default: none / state.json)")
ap.add_argument("--chunk", type=int, default=1000)
ap.add_argument("--max_iters", type=int, default=15000)
ap.add_argument("--num_envs", type=int, default=4096)
ap.add_argument("--scratch", action="store_true", help="start --start_stage from random weights (no resume/init)")
ap.add_argument("--last_stage", type=int, default=3, help="stop after this stage passes")
ap.add_argument("--eval_first", action="store_true", help="evaluate the resume checkpoint before training more")
args = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
state_path = OUT / "state.json"
state = json.loads(state_path.read_text()) if state_path.is_file() else {"stages": {}}


def log(msg):
    line = f"- {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "status.md", "a") as f:
        f.write(line + "\n")


def save():
    state_path.write_text(json.dumps(state, indent=1))


def isaac(script, stage, extra, logfile):
    env = dict(os.environ, PP_STAGE=str(stage), TERM="xterm")
    for k in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV"):
        env.pop(k, None)
    if not (script.startswith("train") and "--play" not in extra):
        env.pop("PP_READY_START_FRAC", None)          # v7 ready starts are a training aid; gate/video start at home
    cmd = [str(LAB / "isaaclab.sh"), "-p", str(HERE / script)] + extra
    with open(logfile, "w") as f:
        proc = subprocess.Popen(cmd, cwd=LAB, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        training = script.startswith("train") and "--play" not in extra
        last_progress, last_change = -1, time.time()
        while proc.poll() is None:                    # watchdog: 2026-10-03 a chunk hung for 2.8 days at 100 % CPU
            time.sleep(30)
            data = Path(logfile).read_bytes()
            progress = data.count(b"Learning iteration") if training else len(data)   # PhysX error spam is not progress
            if progress != last_progress:
                last_progress, last_change = progress, time.time()
            if time.time() - last_change > WATCHDOG_S:
                log(f"watchdog: {script} silent for {WATCHDOG_S // 60} min - killing it")
                os.killpg(proc.pid, 9)
                proc.wait()
                return -9
        return proc.returncode


def latest_checkpoint(run_dir):
    models = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    return models[-1] if models else None


def train_success(logfile, last=50):
    text = re.sub(r"\x1b\[[0-9;]*m", "", Path(logfile).read_text(errors="ignore"))
    vals = [float(v) for v in re.findall(r"Episode_Termination/success:\s*([\d.]+)", text)]
    return sum(vals[-last:]) / max(1, len(vals[-last:])) if vals else 0.0


def newest_stage_checkpoint(stage):
    runs = list(RUNS.glob(f"*_stage{stage}"))
    models = [m for r in runs for m in r.glob("model_*.pt")]
    return max(models, key=lambda m: m.stat().st_mtime) if models else None


ckpt = Path(args.init_checkpoint) if args.init_checkpoint else newest_stage_checkpoint(args.start_stage)   # resume
if args.scratch:
    ckpt = None
if ckpt is not None and ckpt.parent.parent != RUNS:   # rsl_rl --load_run only searches this experiment: copy it in
    dst = RUNS / ckpt.parent.name / ckpt.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(ckpt, dst)
    ckpt = dst
if ckpt is None and args.start_stage > 1 and not args.scratch:
    prev = state["stages"].get(str(args.start_stage - 1), {})
    ckpt = Path(prev["passed_checkpoint"]) if prev.get("passed_checkpoint") else None
    if ckpt is None:
        raise SystemExit(f"stage {args.start_stage - 1} has no passed checkpoint; pass --init_checkpoint")

for stage in range(args.start_stage, args.last_stage + 1):
    st = state["stages"].setdefault(str(stage), {"iters": 0, "evals": []})
    log(f"stage {stage} start (init {ckpt.name if ckpt else 'scratch'})")
    passed = False
    if args.eval_first and stage == args.start_stage and ckpt is not None:
        elog = OUT / f"eval_s{stage}_resume.log"
        isaac("eval_policy.py", stage, ["--headless", "--num_envs", "2048", "--checkpoint", str(ckpt)], elog)
        m = re.search(r"EVAL_JSON (\{.*\})", elog.read_text(errors="ignore"))
        res = json.loads(m.group(1)) if m else {"success": 0.0, "error": "no EVAL_JSON"}
        st["evals"].append(res)
        log(f"stage {stage}: deterministic eval of resume ckpt {ckpt.name}: {json.dumps({k: res.get(k) for k in ('success', 'collision', 'plate_lost', 'time_out')})}")
        save()
        if res.get("success", 0.0) >= GATE:
            st["passed_checkpoint"] = str(ckpt)
            save()
            log(f"stage {stage} PASSED ({res['success']:.3f} >= {GATE})")
            isaac("train_pick_place.py", stage, ["--play", "--num_envs", "4", "--headless", "--enable_cameras", "--video",
                                                 "--video_length", str(int(30 * 25)), "--checkpoint", str(ckpt)],
                  OUT / f"video_s{stage}.log")
            log(f"stage {stage}: video in {ckpt.parent}/videos/play/")
            passed = True
    while not passed and st["iters"] < args.max_iters:
        tag = f"s{stage}_{st['iters']:05d}"
        extra = ["--num_envs", str(args.num_envs), "--headless", "--max_iterations", str(args.chunk), "--run_name", f"stage{stage}"]
        if ckpt is not None:
            extra += ["--resume", "--load_run", ckpt.parent.name, "--checkpoint", ckpt.name]
        tlog = OUT / f"train_{tag}.log"
        t0 = time.time()
        for attempt in range(1, 4):                                   # unattended: retry a crashed chunk
            rc = isaac("train_pick_place.py", stage, extra, tlog)
            run_dir = max(RUNS.glob(f"*_stage{stage}"), key=lambda p: p.stat().st_mtime)
            new = latest_checkpoint(run_dir)
            if rc == 0 and new is not None and new != ckpt:
                break
            log(f"stage {stage}: training chunk failed (rc {rc}, attempt {attempt}/3), see {tlog}")
            new = newest_stage_checkpoint(stage) or ckpt             # resume from whatever was saved
            if new is not None and new != ckpt:
                ckpt = new
                extra = [a for a in extra if a not in ("--resume",)]
                extra = extra[:extra.index("--load_run")] if "--load_run" in extra else extra
                extra += ["--resume", "--load_run", ckpt.parent.name, "--checkpoint", ckpt.name]
            time.sleep(60)
        else:
            log(f"stage {stage}: giving up after 3 failed attempts")
            break
        ckpt = new
        st["iters"] += args.chunk
        sr = train_success(tlog)
        st["last_checkpoint"] = str(ckpt)
        log(f"stage {stage}: +{args.chunk} iters ({st['iters']} total, {(time.time() - t0) / 60:.0f} min), "
            f"training success (last 50 it) {sr:.3f}, ckpt {ckpt.parent.name}/{ckpt.name}")
        save()
        if sr < PROXY_GATE:
            continue
        elog = OUT / f"eval_{tag}.log"
        isaac("eval_policy.py", stage, ["--headless", "--num_envs", "2048", "--checkpoint", str(ckpt)], elog)
        m = re.search(r"EVAL_JSON (\{.*\})", elog.read_text(errors="ignore"))
        res = json.loads(m.group(1)) if m else {"success": 0.0, "error": "no EVAL_JSON"}
        st["evals"].append(res)
        log(f"stage {stage}: deterministic eval {json.dumps({k: res.get(k) for k in ('success', 'collision', 'plate_lost', 'time_out', 'mean_shield_steps')})}")
        save()
        if res.get("success", 0.0) >= GATE:
            st["passed_checkpoint"] = str(ckpt)
            save()
            log(f"stage {stage} PASSED ({res['success']:.3f} >= {GATE})")
            isaac("train_pick_place.py", stage, ["--play", "--num_envs", "4", "--headless", "--enable_cameras", "--video",
                                                 "--video_length", str(int(30 * 25)), "--checkpoint", str(ckpt)],
                  OUT / f"video_s{stage}.log")
            log(f"stage {stage}: video in {ckpt.parent}/videos/play/")
            passed = True
            break
    if not passed:
        log(f"stage {stage} NOT passed within {st['iters']} iterations - curriculum stopped (gate {GATE})")
        break
else:
    log("all 3 stages passed")
