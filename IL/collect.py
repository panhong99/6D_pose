"""Collect physical stage-2 demonstrations with the existing RL controller and safety shield.

Run with Isaac Lab's Python:
  ~/IsaacLab/isaaclab.sh -p IL/collect.py --headless --episodes 50 --num_envs 8
Only successful, unblocked, collision-free episodes enter data/episodes. Rejections go into manifest.json.
No policy is trained and no real robot is contacted.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))


def parse_args():
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max_batches", type=int, default=20)
    p.add_argument("--output", type=Path, default=ROOT / "IL" / "data")
    p.add_argument("--lift_m", type=float, default=0.10, help="TCP raise after closing (task lift target 10 cm)")
    p.add_argument("--total_s", type=float, default=17.0, help="recorded seconds per episode (env episode is 20 s)")
    p.add_argument('--speed_cap',type=float,default=None,help='candidate: lower TCP command cap')
    p.add_argument('--hold_windup_m',type=float,default=None,help='candidate: lower held target lead')
    p.add_argument('--arm_joint_speed_deg_s',type=float,default=None,help='candidate: joint drive/target cap <=6deg/s')
    p.add_argument("--close_dist", type=float, default=0.003, help="approach -> close when the TCP is this close to the rim target")
    p.add_argument("--slow_zone_m", type=float, default=0.02, help="proportional expert: full speed beyond this distance to "
                   "the target, linearly slower inside it (smooth, self-correcting BC labels; 0 = old saturated step labels)")
    p.add_argument("--dart_noise", type=float, default=0.0, help="DART: execute expert xyz command + N(0, this) "
                   "(policy units), record the clean expert command as the label (recovery data against BC drift)")
    p.add_argument("--stall_steps", type=int, default=5, help="close -> raise after held with still fingers this many steps")
    p.add_argument("--speed_lo", type=float, default=0.7, help="expert speed factor range x max_tcp_speed: [lo, 1.0]")
    p.add_argument("--plate_range", type=float, nargs=2, default=None, metavar=("DX", "DY"),
                   help="override the plate reset range to +-DX, +-DY m around plate_xy (IK+BC: any table position)")
    p.add_argument("--grasp_bank", type=Path, default=None, help="opt-in phi/pitch IK candidate bank (sim only)")
    p.add_argument("--bank_azimuth_deg", type=float, default=0.0, help="randomly rotate bank starts/goals about joint 1")
    p.add_argument("--plate_dz",type=float,default=0.0,help="one support/plate height per simulation, metres")
    p.add_argument("--verify", action="store_true", help="audit saved episodes without starting Isaac Sim")
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    if a.episodes < 1 or a.num_envs < 1 or a.max_batches < 1:
        p.error("episodes, num_envs and max_batches must be positive")
    return a


def verify(args):
    """Re-read saved transitions and recompute success/safety against the saved deadline."""
    import numpy as np
    import yaml
    out = args.output.resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    rcfg = yaml.safe_load((out / "rl_config_snapshot.yaml").read_text())
    deadline = float(rcfg["task"]["episode_s"][2])
    v_max = rcfg["task"].get("hold_max_plate_speed")
    if manifest["status"] != "complete":
        raise RuntimeError("collection is not complete")
    results = []
    for e in manifest["episodes"]:
        path = out / e["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == e["sha256"]
        with np.load(path, allow_pickle=False) as d:
            n = e["frames"]
            assert d["obs"].shape[0] == n and d["obs"].shape[1] in (43, 46) and d["next_obs"].shape == d["obs"].shape
            assert d["actions"].shape == (n, 7)
            assert all(np.isfinite(d[k]).all() for k in d.files)
            assert not d["shield_blocked"].any() and d["arm_contact_n"].max() <= 0.5
            assert d["frame_clearance_m"].min() >= 0.049
            assert d["table_clearance_m"].min() >= 0.019
            # Observations after step t must be the observations before step t+1, not re-sampled noisy inputs.
            np.testing.assert_array_equal(d["next_obs"][:-1], d["obs"][1:])
            if 'arm_joint_speed_deg_s' in d.files:
                # Check joint order and the observation scale used by legacy speed audits.
                np.testing.assert_allclose(np.degrees(d['next_obs'][:,8:14]*2),d['arm_joint_speed_deg_s'],atol=1e-3)
            stable = d["held"] & (d["lift_m"] > 0.05) & (d["tilt_deg"] < 25.0)
            if v_max is not None and "plate_speed_mps" in d.files:              # slow env (2026-10-07): plate still
                stable &= d["plate_speed_mps"] < v_max
            np.testing.assert_array_equal(stable, d["stable_lift"])
            count, best, first = 0, 0, None
            for t, hit in enumerate(stable):
                count = count + 1 if hit else 0
                best = max(count, best)
                if count >= 150 and first is None:
                    first = (t + 1) * manifest["policy_dt_s"]
            assert best >= 150 and stable[-1] and first <= deadline
            assert d["terminal"].sum() == 1 and d["terminal"][-1]
            assert set(np.unique(d["actions"][:, 6])) == {-1.0, 1.0}
            results.append(dict(file=e["file"], success_time_s=first, best_hold_steps=best))
    report = dict(episodes=len(results), frames=sum(e["frames"] for e in manifest["episodes"]),
                  train=sum(e["split"] == "train" for e in manifest["episodes"]),
                  validation=sum(e["split"] == "validation" for e in manifest["episodes"]),
                  max_success_time_s=max(r["success_time_s"] for r in results),
                  minimum_best_hold_steps=min(r["best_hold_steps"] for r in results),
                  all_passed=True, episodes_checked=results)
    (out / "verification.json").write_text(json.dumps(report, indent=2))
    print("VERIFY_RESULT " + json.dumps({k: v for k, v in report.items() if k != "episodes_checked"}), flush=True)


def collect(args):
    import numpy as np
    import torch
    import plate_env as pe
    from isaaclab.envs import ManagerBasedRLEnv
    from rl_utils import load_config

    out = args.output.resolve()
    episodes_dir = out / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise RuntimeError("manifest already exists; choose a new --output to preserve existing data")
    config = load_config()
    for key,value in [('max_tcp_speed',args.speed_cap),('hold_windup',args.hold_windup_m)]:
        if value is not None:
            if not 0<value<=config['action'][key]:raise ValueError('Candidate may only reduce '+key)
            config['action'][key]=value;pe.A[key]=value
    if args.total_s>config['task']['episode_s'][2]:
        config['task']['episode_s'][2]=args.total_s
    source_files = [ROOT / "rl" / n for n in ("plate_env.py", "rl_utils.py", "config.yaml")]
    if args.grasp_bank:
        source_files.append(ROOT / "rl/grasp_variants.py")
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    snapshot_dir = out / "source_snapshot"
    snapshot_dir.mkdir(exist_ok=True)
    for p in source_files:
        (snapshot_dir / p.name).write_bytes(p.read_bytes())
    import yaml
    (out / "rl_config_snapshot.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
    cfg = pe.make_env_cfg(2, args.num_envs)
    joint_cap=None
    if args.arm_joint_speed_deg_s is not None:
        if not 0<args.arm_joint_speed_deg_s<=6:raise ValueError('Joint cap must be <=6deg/s')
        joint_cap=np.radians(args.arm_joint_speed_deg_s)
        for name,actuator in cfg.scene.robot.actuators.items():
            if name.startswith('arm_'):actuator.velocity_limit_sim=float(joint_cap)
    if args.plate_dz:
        height=pe.G['box_h']+args.plate_dz
        if height<=0:raise ValueError('Support height must be positive')
        pe.G['box_h']=height;pe.REST_Z=pe.G['table_top_z']+height
        cfg.scene.start_box.spawn.size=(pe.G['box_xy'],pe.G['box_xy'],height)
        cfg.scene.start_box.init_state.pos=(*pe.G['plate_xy'],pe.G['table_top_z']+height/2)
        cfg.scene.plate.init_state.pos=(*pe.G['plate_xy'],pe.REST_Z)
    if args.plate_range:
        dx, dy = args.plate_range
        cfg.events.reset_plate.params["pose_range"].update(x=(-dx, dx), y=(-dy, dy))
    cfg.seed = args.seed
    cfg.scene.arm_contacts.history_length = config["task"]["decimation"]
    # Keep physical state continuous while recording: never auto-reset in the middle of a demonstration.
    # We independently enforce the original task success and collision conditions below.
    cfg.episode_length_s = 3600.0
    for name in ("time_out", "success", "collision", "plate_lost"):
        setattr(cfg.terminations, name, None)
    env = ManagerBasedRLEnv(cfg)
    if joint_cap is not None:env._arm_joint_speed_rad_s=float(joint_cap)
    if args.grasp_bank:
        from grasp_variants import install
        install(env,args.grasp_bank,azimuth_deg=args.bank_azimuth_deg)
    dev = env.device
    arm = env.action_manager.get_term("arm_action")
    robot, plate = env.scene["robot"], env.scene["plate"]
    dt = float(env.step_dt)
    manifest = dict(schema_version=1, task="stage2_ready_grasp_lift_hold", simulation_only=True,
                    requested=args.episodes, seed=args.seed, num_envs=args.num_envs, policy_dt_s=dt,
                    observations="43-dimensional current RL policy observations, with configured pose noise",
                    action_semantics="7-dimensional pre-controller commands; xyz/rotation scaled by RL config, grip sign",
                    controller="current RL integrated-position IK + orientation latch + voxel safety shield",
                    plate_range=args.plate_range, grasp_bank=str(args.grasp_bank) if args.grasp_bank else None,
                    bank_azimuth_deg=args.bank_azimuth_deg,
                    plate_dz_m=args.plate_dz,
                    arm_joint_speed_cap_deg_s=args.arm_joint_speed_deg_s,
                    grasp_bank_sha256=hashlib.sha256(args.grasp_bank.read_bytes()).hexdigest() if args.grasp_bank else None,
                    sources_sha256=sources, phase_names=["approach", "close", "raise_hold"], phase_switching="state-based per env", dart_noise=args.dart_noise, slow_zone_m=args.slow_zone_m,
                    success_conditions=dict(plate_lift_m=config["task"]["lift_ok"],
                                            tilt_deg=config["task"]["max_hold_tilt_deg"],
                                            consecutive_steps=config["task"]["hold_steps"],
                                            no_shield_blocks=True, max_arm_contact_n=0.5,
                                            actual_clearance_tolerance_m=0.001),
                    episodes=[], rejected=[], status="collecting")
    start = time.monotonic()

    def persist():
        manifest["elapsed_s"] = round(time.monotonic() - start, 2)
        temp = out / "manifest.tmp"
        temp.write_text(json.dumps(manifest, indent=2))
        temp.replace(out / "manifest.json")

    def cpu(t):
        return t.detach().cpu().numpy().copy()

    try:
        with torch.inference_mode():
            for batch in range(args.max_batches):
                obs, _ = env.reset()
                initial_plate = cpu(plate.data.root_pos_w - env.scene.env_origins)
                initial_quat = cpu(plate.data.root_quat_w)
                bank_index = cpu(env._grasp_bank_index) if args.grasp_bank else None
                bank_azimuth = cpu(env._grasp_azimuth) if args.grasp_bank else None
                # Per-episode speed and lift amplitude complement the environment's +-1 cm xy and random yaw resets.
                # command cap = max_tcp_speed per policy step, scaled by a per-episode factor in [speed_lo, 1]
                cap = config["action"]["max_tcp_speed"] * dt / config["action"]["scale"]
                speed = cap * (args.speed_lo + (1.0 - args.speed_lo) * torch.rand(args.num_envs, 1, device=dev))
                lift_m = args.lift_m + (torch.rand(args.num_envs, device=dev) - 0.5) * 0.02
                # STATE-based phases per env (time-based switches teach BC contradictory labels for the same state):
                #   0 approach the rim target (open) -> 1 when the TCP is within close_dist of it
                #   1 close and hold the TCP still   -> 2 when held and the fingers stopped for stall_steps
                #   2 raise lift_m above the grasp point and hold there (same controller, no further switch)
                ph = torch.zeros(args.num_envs, dtype=torch.long, device=dev)
                target = torch.zeros(args.num_envs, 3, device=dev)
                stall = torch.zeros(args.num_envs, device=dev)
                fingers_prev = pe._state(env)["fingers"].clone()
                records = []
                best_hold = torch.zeros(args.num_envs, device=dev)
                run_hold = torch.zeros_like(best_hold)
                for tick in range(round(args.total_s / dt)):
                    st = pe._state(env)
                    start_close = (ph == 0) & (st["d_rim"] < args.close_dist)
                    target[start_close] = st["tcp"][start_close]
                    ph[start_close] = 1
                    stopped = (st["fingers"] - fingers_prev).abs() < 1e-3
                    stall = torch.where((ph == 1) & st["held"] & stopped, stall + 1, torch.zeros_like(stall))
                    start_raise = (ph == 1) & (stall >= args.stall_steps)
                    target[start_raise, 2] += lift_m[start_raise]
                    ph[start_raise] = 2
                    fingers_prev = st["fingers"].clone()
                    err = torch.where((ph == 0)[:, None], st["rim"] - st["tcp"], target - st["tcp"])
                    act = torch.zeros(args.num_envs, 7, device=dev)
                    if args.slow_zone_m > 0:                   # proportional: |cmd| = speed * min(1, |err| / slow_zone)
                        dist = err.norm(dim=1, keepdim=True).clamp(min=1e-9)
                        act[:, :3] = err / dist * speed * torch.clamp(dist / args.slow_zone_m, max=1.0)
                    else:
                        act[:, :3] = torch.maximum(torch.minimum(err / config["action"]["scale"], speed), -speed)
                    act[:, 6] = torch.where(ph >= 1, -1.0, 1.0)
                    before = dict(obs=cpu(obs["policy"]), actions=cpu(act),
                                  joint_pos=cpu(robot.data.joint_pos), joint_vel=cpu(robot.data.joint_vel),
                                  tcp_pose=cpu(torch.cat(pe.tcp_pose(env), dim=1)),
                                  plate_pose=cpu(torch.cat((plate.data.root_pos_w, plate.data.root_quat_w), dim=1)),
                                  hold_target=cpu(arm.hold_pos), orientation_target=cpu(arm.orientation_target),
                                  orientation_locked=cpu(arm.orientation_locked), phase=cpu(ph).astype(np.int8))
                    executed = act.clone()
                    if args.dart_noise > 0:
                        executed[:, :3] += args.dart_noise * torch.randn(args.num_envs, 3, device=dev)
                    before["executed_actions"] = cpu(executed)
                    obs, reward, _, _, _ = env.step(executed)
                    st = pe._state(env)
                    contact = env.scene["arm_contacts"].data.net_forces_w_history.norm(dim=-1).amax(dim=(1, 2))
                    frame, table = arm.shield.clearance(robot.data.joint_pos[:, arm._joint_ids])
                    run_hold = torch.where(st["stable_lift"], run_hold + 1, torch.zeros_like(run_hold))
                    best_hold = torch.maximum(best_hold, run_hold)
                    before.update(next_obs=cpu(obs["policy"]), rewards=cpu(reward),
                                  controller_actions=cpu(arm._raw_actions),
                                  processed_arm_actions=cpu(arm._processed_actions),
                                  arm_joint_target=cpu(arm._safe_target),
                                  arm_joint_speed_deg_s=cpu(torch.rad2deg(robot.data.joint_vel[:,arm._joint_ids])),
                                  next_tcp_pose=cpu(torch.cat(pe.tcp_pose(env), dim=1)),
                                  next_plate_pose=cpu(torch.cat((plate.data.root_pos_w, plate.data.root_quat_w), dim=1)),
                                  shield_blocked=cpu(arm.blocked), arm_contact_n=cpu(contact),
                                  frame_clearance_m=cpu(frame), table_clearance_m=cpu(table),
                                  held=cpu(st["held"]), stable_lift=cpu(st["stable_lift"]),
                                  lift_m=cpu(st["lift"]), plate_speed_mps=cpu(st["plate_speed"]),
                                  tilt_deg=cpu(torch.rad2deg(torch.acos(st["plate_up_z"].clamp(-1, 1)))))
                    records.append(before)
                    if tick % 60 == 59:
                        print(f"COLLECT_TICK batch={batch} t={(tick + 1) * dt:.0f}s phases={torch.bincount(ph, minlength=3).tolist()} "
                              f"held={float(st['held'].float().mean()):.3f} lift_mm={float(st['lift'].median()) * 1000:.1f} "
                              f"best_hold={int(best_hold.max())}", flush=True)
                arrays = {key: np.stack([r[key] for r in records]) for key in records[0]}
                for i in range(args.num_envs):
                    data = {k: v[:, i] for k, v in arrays.items()}
                    reason = []
                    if best_hold[i] < config["task"]["hold_steps"] or not bool(st["stable_lift"][i]):
                        reason.append("no_stable_5s_hold")
                    if data["shield_blocked"].any():
                        reason.append("shield_blocked")
                    if data["arm_contact_n"].max() > 0.5:
                        reason.append("arm_collision")
                    if data["frame_clearance_m"].min() < config["shield"]["frame_margin"] - 0.001:
                        reason.append("frame_clearance")
                    if data["table_clearance_m"].min() < config["shield"]["table_margin"] - 0.001:
                        reason.append("table_clearance")
                    if any(not np.isfinite(v).all() for v in data.values()):
                        reason.append("nonfinite")
                    metrics = dict(batch=batch, env=i, best_hold_steps=int(best_hold[i]),
                                   final_lift_m=float(st["lift"][i]), final_tilt_deg=float(data["tilt_deg"][-1]),
                                   max_arm_contact_n=float(data["arm_contact_n"].max()),
                                   shield_blocked_steps=int(data["shield_blocked"].sum()),
                                   min_frame_clearance_m=float(data["frame_clearance_m"].min()),
                                   min_table_clearance_m=float(data["table_clearance_m"].min()),
                                   initial_plate_xyz=initial_plate[i].tolist(), initial_plate_quat_wxyz=initial_quat[i].tolist(),
                                   expert_cmd_per_step=float(speed[i, 0]), commanded_lift_m=float(lift_m[i]))
                    if args.grasp_bank:
                        j=int(bank_index[i])
                        metrics.update(bank_index=j,phi_deg=float(env._grasp_bank['phi_deg'][j]),
                                       pitch_deg=float(env._grasp_bank['pitch_deg'][j]),azimuth_deg=float(np.degrees(bank_azimuth[i])))
                    if reason:
                        metrics["reasons"] = reason
                        manifest["rejected"].append(metrics)
                    elif len(manifest["episodes"]) < args.episodes:
                        idx = len(manifest["episodes"])
                        filename = f"episode_{idx:04d}.npz"
                        # Split by whole episode, never randomly split correlated frames.
                        split = "validation" if idx % 5 == 4 else "train"
                        data["terminal"] = np.zeros(len(records), dtype=np.bool_)
                        data["terminal"][-1] = True
                        data["success"] = np.asarray(True)
                        data["env_origin"] = cpu(env.scene.env_origins[i])
                        data["initial_plate_pose"] = np.concatenate((initial_plate[i], initial_quat[i]))
                        np.savez_compressed(episodes_dir / filename, **data)
                        metrics.update(file=f"episodes/{filename}", split=split, frames=len(records),
                                       sha256=hashlib.sha256((episodes_dir / filename).read_bytes()).hexdigest())
                        manifest["episodes"].append(metrics)
                persist()
                print(f"COLLECT_BATCH {batch} accepted={len(manifest['episodes'])}/{args.episodes} "
                      f"rejected={len(manifest['rejected'])}", flush=True)
                if len(manifest["episodes"]) >= args.episodes:
                    break
        manifest["status"] = "complete" if len(manifest["episodes"]) >= args.episodes else "incomplete"
        manifest["total_frames"] = sum(e["frames"] for e in manifest["episodes"])
        persist()
        print("COLLECT_RESULT " + json.dumps({k: manifest[k] for k in ("status", "total_frames", "elapsed_s")}) +
              f" accepted={len(manifest['episodes'])} rejected={len(manifest['rejected'])}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if args.verify:
        verify(args)
        sys.exit(0)
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    try:
        collect(args)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        app.close()
