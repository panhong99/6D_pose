"""Offline safety screening for recorded basket-teaching demos.

Reads saved joint/wrench telemetry only. It never connects to a robot or camera.
The result is a screening report, not a robot safety certification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEACH_DIR = ROOT / "calibration" / "data" / "teach"
_CHECKERS = None
_CHECKER_META = {}


def _read_state(path):
    rows = []
    with path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: malformed JSON") from exc
            if row.get("kind") == "state":
                rows.append(row)
    if len(rows) < 2:
        raise ValueError(f"{path}: need at least two state samples")
    return rows


def _checker_set():
    global _CHECKERS, _CHECKER_META
    if _CHECKERS is None:
        import sys

        sys.path.insert(0, str(ROOT / "calibration"))
        from calib_core import cpath, latest_t_base_cam, load_config, load_json
        from plate_plans import library_checkers

        config = load_config()
        camera_path = Path(latest_t_base_cam())
        cell_path = Path(cpath(config["paths"]["cell_json"]))
        Tbc = np.asarray(load_json(camera_path)["T_base_cam"], dtype=float)
        _CHECKERS = library_checkers(Tbc, config["plate_ik_bc"])
        _CHECKER_META = {
            "checker": "plate_plans.library_checkers",
            "camera_transform_source": str(camera_path),
            "cell_boxes_source": str(cell_path if cell_path.is_file() else "robot_model.FRAME_CELL defaults"),
            "voxel_resolution_m": 0.01,
            "trajectory_sample_period_s": 0.1,
            "included": "static workcell frame/table, fixed D455 body, approximate gripper cylinder",
        }
    return _CHECKERS


def _segments(mask, times, max_gap_s=0.15):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    groups = [[int(indices[0])]]
    for idx in indices[1:]:
        if times[idx] - times[groups[-1][-1]] <= max_gap_s:
            groups[-1].append(int(idx))
        else:
            groups.append([int(idx)])
    return groups


def _peak_event(rows, index, magnitude, threshold, kind, baseline):
    row = rows[index]
    return {
        "kind": kind,
        "time_from_start_s": float(row["t"] - rows[0]["t"]),
        "peak_deviation": float(magnitude[index]),
        "threshold": float(threshold),
        "baseline_vector": np.asarray(baseline, dtype=float).tolist(),
        "measured_vector": np.asarray(row[kind], dtype=float).tolist(),
        "joint_deg": np.asarray(row["posj"], dtype=float).tolist(),
        "flange_xyz_mm": np.asarray(row["posx"], dtype=float)[:3].tolist(),
    }


def _wrench_audit(rows, t, joint_speed):
    force = np.asarray([r["tool_force"] for r in rows], dtype=float)
    torque = np.asarray([r["ext_torque"] for r in rows], dtype=float)
    if force.shape != (len(rows), 6) or torque.shape != (len(rows), 6):
        raise ValueError("tool_force/ext_torque must be six-axis vectors")
    stationary = joint_speed <= 0.25  # deg/s; baseline only, not a motion limit
    if stationary.sum() < 20:
        stationary = np.arange(len(rows)) < max(20, int(0.05 * len(rows)))

    result = {}
    signals = (("tool_force", force, 50.0), ("ext_torque", torque, 40.0))
    for name, signal, screening_floor in signals:
        baseline = np.median(signal[stationary], axis=0)
        dev = np.linalg.norm(signal - baseline, axis=1)
        stationary_dev = dev[stationary]
        med = float(np.median(stationary_dev))
        mad = float(np.median(np.abs(stationary_dev - med)))
        threshold = max(screening_floor, med + 8.0 * 1.4826 * mad)
        event_mask = dev > threshold
        events = []
        for group in _segments(event_mask, t):
            peak = max(group, key=lambda i: dev[i])
            ev = _peak_event(rows, peak, dev, threshold, name, baseline)
            ev["start_s"] = float(t[group[0]])
            ev["end_s"] = float(t[group[-1]])
            events.append(ev)
        peak = int(np.argmax(dev))
        result[name] = {
            "stationary_baseline_vector": baseline.tolist(),
            "stationary_sample_count": int(stationary.sum()),
            "stationary_residual_median": med,
            "stationary_residual_mad": mad,
            "screening_threshold": float(threshold),
            "screening_floor": screening_floor,
            "peak_deviation": float(dev[peak]),
            "peak_event": _peak_event(rows, peak, dev, threshold, name, baseline),
            "events": events,
        }
    return result


def _fk_kinematics(rows):
    import sys

    sys.path.insert(0, str(ROOT / "calibration"))
    from calib_core import load_config
    from robot_model import Kinematics

    kin = Kinematics()
    tool_length = float(load_config()["plate_ik_bc"].get("tool_length_m", 0.165))
    q = np.radians(np.asarray([r["posj"] for r in rows], dtype=float))
    flange = np.asarray([kin.fk(j) for j in q])
    tcp = flange.copy()
    tcp[:, :3, 3] += flange[:, :3, 2] * tool_length
    return kin, q, flange, tcp, tool_length


def _collision_audit(rows, t, kin, q, keyposes):
    checkers = _checker_set()
    dt = float(np.median(np.diff(t)))
    stride = max(1, int(round(0.1 / dt)))
    indices = set(range(0, len(rows), stride)) | {len(rows) - 1}
    key_times = {}
    for name, pose in keyposes.items():
        time_s = pose.get("t")
        if time_s is not None:
            idx = int(np.argmin(np.abs(t - float(time_s))))
            indices.add(idx)
            key_times[idx] = name
    indices = np.asarray(sorted(indices), dtype=int)
    measurements = {name: [] for name in checkers}
    for idx in indices:
        for name, (checker, margin) in checkers.items():
            measurements[name].append((idx, float(checker.clearance(q[idx])), float(margin)))

    result = {}
    for name, values in measurements.items():
        bad = [v for v in values if v[1] < v[2]]
        worst = min(values, key=lambda v: v[1])
        sample_t = np.asarray([t[idx] - t[0] for idx, _, _ in values])
        bad_mask = np.asarray([clearance < margin for _, clearance, margin in values])
        violation_events = []
        for group in _segments(bad_mask, sample_t, max_gap_s=0.25):
            group_values = [values[j] for j in group]
            idx, clearance, margin = min(group_values, key=lambda v: v[1])
            violation_events.append({
                "start_s": float(sample_t[group[0]]),
                "end_s": float(sample_t[group[-1]]),
                "minimum_clearance_m": clearance,
                "margin_m": margin,
                "minimum_time_from_start_s": float(t[idx] - t[0]),
                "joint_deg": np.degrees(q[idx]).tolist(),
                "flange_xyz_mm": np.asarray(rows[idx]["posx"], dtype=float)[:3].tolist(),
                "keypose": key_times.get(idx),
                "samples": len(group),
            })
        result[name] = {
            "margin_m": float(worst[2]),
            "sample_count": len(values),
            "sample_period_s": float(stride * dt),
            "minimum_clearance_m": float(worst[1]),
            "minimum_time_from_start_s": float(t[worst[0]] - t[0]),
            "minimum_joint_deg": np.degrees(q[worst[0]]).tolist(),
            "minimum_keypose": key_times.get(worst[0]),
            "violation_events": violation_events,
            "violation_sample_count": len(bad),
        }
    return result


def demo_audit(full_json) -> dict:
    """Audit one saved full-demo JSON and its robot_state.jsonl, without hardware I/O."""
    path = Path(full_json) if not isinstance(full_json, dict) else None
    record = json.loads(path.read_text()) if path is not None else full_json
    if path is not None and path.name != Path(record.get("file", path.name)).name:
        raise ValueError("demo JSON file identity mismatch")
    state_path = Path(record.get("run", "")) / "robot_state.jsonl"
    rows = _read_state(state_path)
    t = np.asarray([float(r["t"]) for r in rows], dtype=float)
    qdeg = np.asarray([r["posj"] for r in rows], dtype=float)
    posx = np.asarray([r["posx"][:3] for r in rows], dtype=float)
    if qdeg.shape != (len(rows), 6) or posx.shape != (len(rows), 3) or not np.isfinite(qdeg).all():
        raise ValueError("invalid recorded joint/TCP states")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("state timestamps must be strictly increasing")
    joint_step = np.diff(np.radians(qdeg), axis=0) / dt[:, None]
    joint_speed = np.max(np.abs(np.degrees(joint_step)), axis=1)
    joint_speed_state = np.r_[joint_speed, joint_speed[-1]]
    kin, q, flange, tcp, tool_length = _fk_kinematics(rows)
    tcp_speed = np.linalg.norm(np.diff(tcp[:, :3, 3], axis=0), axis=1) / dt
    tcp_speed_state = np.r_[tcp_speed, tcp_speed[-1]]
    wrench = _wrench_audit(rows, t - t[0], joint_speed_state)
    poses = record.get("poses", {})
    key_q = {name: np.radians(np.asarray(pose["posj"], dtype=float)) for name, pose in poses.items()
             if isinstance(pose, dict) and pose.get("posj") is not None}
    collision = _collision_audit(rows, t, kin, q, poses)

    speed_idx = int(np.argmax(joint_speed_state))
    tcp_idx = int(np.argmax(tcp_speed_state))
    speed = {
        "max_joint_axis_deg_s": float(joint_speed_state[speed_idx]),
        "joint_axis_index_1_based": int(np.argmax(np.abs(joint_step[speed_idx if speed_idx < len(joint_step) else -1])) + 1),
        "joint_peak_time_from_start_s": float(t[speed_idx] - t[0]),
        "joint_peak_pose_deg": qdeg[speed_idx].tolist(),
        "max_tcp_speed_mm_s_fk": float(tcp_speed_state[tcp_idx] * 1000.0),
        "tcp_peak_time_from_start_s": float(t[tcp_idx] - t[0]),
        "tcp_peak_flange_xyz_mm": (flange[tcp_idx, :3, 3] * 1000.0).tolist(),
        "max_recorded_flange_speed_mm_s": float(np.max(np.linalg.norm(np.diff(posx, axis=0) / dt[:, None], axis=1))),
    }
    wrench_bad = any(a["events"] for a in wrench.values())
    collision_bad = any(a["violation_sample_count"] for a in collision.values())
    manifest_entry = _manifest_entry(path.name if path is not None else "")
    return {
        "demo": path.name if path is not None else record.get("file"),
        "quadrant": record.get("quadrant"),
        "slot": record.get("slot"),
        "source_unsafe_tag": manifest_entry.get("unsafe", record.get("unsafe")) if manifest_entry else record.get("unsafe"),
        "state_path": str(state_path),
        "state_count": len(rows),
        "duration_s": float(t[-1] - t[0]),
        "wrench": wrench,
        "collision": collision,
        "collision_model": _CHECKER_META,
        "speed": speed,
        "tcp_tool_length_m": tool_length,
        "safe": not (wrench_bad or collision_bad),
        "unsafe_reasons": (["wrench_threshold"] if wrench_bad else []) + (["frame_or_table_clearance"] if collision_bad else []),
        "scope_note": "Offline screening only: force/torque cutoffs are dataset-specific baseline-relative heuristics; collision uses sampled robot mesh points, configured static frame/table and gripper cylinder, not continuous swept-volume or carried-plate proof.",
    }


def _manifest_entry(filename):
    manifest = TEACH_DIR / "bc_demos.json"
    if not filename or not manifest.is_file():
        return {}
    data = json.loads(manifest.read_text())
    return next((d for d in data.get("demos", []) if Path(d.get("file", "")).name == filename), {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", help="one full_NN.json")
    group.add_argument("--all", action="store_true", help="audit full demos listed in bc_demos.json")
    parser.add_argument("--out", type=Path, help="new JSON output file; existing files are never overwritten")
    args = parser.parse_args()
    if args.all:
        manifest = json.loads((TEACH_DIR / "bc_demos.json").read_text())
        reports = [demo_audit(TEACH_DIR / d["file"]) for d in manifest["demos"]]
    else:
        reports = [demo_audit(Path(args.demo))]
    output = {"demos": reports}
    text = json.dumps(output, indent=2, allow_nan=False)
    if args.out:
        if args.out.exists():
            raise FileExistsError(args.out)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    for report in reports:
        print(f"{report['demo']}: {'SAFE' if report['safe'] else 'UNSAFE'}; "
              f"reasons={report['unsafe_reasons']}; samples={report['state_count']}")


if __name__ == "__main__":
    main()
