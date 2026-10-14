"""Offline conversion of taught basket demos into basket-frame BC records.

No camera, ROS, Isaac, or robot connection is used. Source demonstrations are
read-only; every output is written to a new directory under calibration/data.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from robot_model import Kinematics


ROOT = Path(__file__).resolve().parents[1]
TEACH_DIR = ROOT / "calibration" / "data" / "teach"
KEYPOSES = ("pre_grasp", "grasp", "lift", "carry", "above", "insert", "retreat")
DEFAULT_TARGET_LAYOUT = "new: basket centred y~0, plate in front"
DEFAULT_TOOL_LENGTH_M = 0.165


def _transform(value, name):
    T = np.asarray(value, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"{name}: expected finite 4x4 transform")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{name}: invalid homogeneous last row")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(R), 1., atol=1e-3):
        raise ValueError(f"{name}: rotation is not proper orthonormal")
    return T


def _tcp_from_flange(T_base_flange, tool_length_m):
    T = np.asarray(T_base_flange, dtype=float).copy()
    T[:3, 3] += T[:3, 2] * float(tool_length_m)
    return T


def transform_keyposes(poses, T_base_basket, kin, tool_length_m=DEFAULT_TOOL_LENGTH_M):
    """Return validated keypose arrays; posx in source JSON is flange xyz in mm."""
    Tbb = _transform(T_base_basket, "T_base_basket")
    if not np.isfinite(tool_length_m) or tool_length_m <= 0:
        raise ValueError("tool_length_m must be finite and positive")
    absent = [name for name in KEYPOSES if name not in poses]
    if absent:
        raise ValueError(f"missing keyposes: {absent}")

    q_all, flange_all, tcp_all, basket_tcp_all, reported_all, errors = [], [], [], [], [], []
    T_basket_base = np.linalg.inv(Tbb)
    for name in KEYPOSES:
        rec = poses[name]
        q_deg = np.asarray(rec.get("posj"), dtype=float)
        posx = np.asarray(rec.get("posx"), dtype=float)
        if q_deg.shape != (6,) or posx.shape != (6,) or not np.isfinite(q_deg).all() or not np.isfinite(posx).all():
            raise ValueError(f"{name}: invalid posj/posx vector")
        Tbf = kin.fk(np.radians(q_deg))
        err = float(np.linalg.norm(Tbf[:3, 3] * 1000. - posx[:3]))
        if err >= 1.0:
            raise ValueError(f"{name}: FK flange xyz differs from recorded posx by {err:.3f} mm (limit 1 mm)")
        Ttcp = _tcp_from_flange(Tbf, tool_length_m)
        q_all.append(q_deg)
        flange_all.append(Tbf)
        tcp_all.append(Ttcp)
        basket_tcp_all.append(T_basket_base @ Ttcp)
        reported_all.append(posx)
        errors.append(err)
    return {
        "keypose_names": np.asarray(KEYPOSES),
        "keypose_joint_deg": np.asarray(q_all, dtype=np.float64),
        "keypose_T_base_flange": np.asarray(flange_all, dtype=np.float64),
        "keypose_T_base_tcp": np.asarray(tcp_all, dtype=np.float64),
        "keypose_T_basket_tcp": np.asarray(basket_tcp_all, dtype=np.float64),
        "keypose_recorded_posx_mm_deg": np.asarray(reported_all, dtype=np.float64),
        "keypose_fk_flange_xyz_error_mm": np.asarray(errors, dtype=np.float64),
    }


def _plate_geometry():
    import yaml

    config = yaml.safe_load((ROOT / "rl" / "config.yaml").read_text())
    g = config["geometry"]
    rim_to_centre = float(g["rim_r"]) - float(g["grasp_inset"])
    return rim_to_centre, {"rim_r_m": float(g["rim_r"]), "grasp_inset_m": float(g["grasp_inset"])}


def _plate_centre_estimate(T_base_tcp, T_base_basket, rim_to_centre_m):
    """Apply the same flat-plate radial grasp model used by plate_plans._plate_in_hand."""
    axis = np.asarray(T_base_tcp[:3, 2], dtype=float).copy()
    axis[2] = 0.
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        raise ValueError("grasp TCP local z has no horizontal component; plate centre cannot be estimated")
    centre_base = T_base_tcp[:3, 3] + axis / norm * rim_to_centre_m
    centre_basket = np.linalg.inv(T_base_basket) @ np.r_[centre_base, 1.]
    return centre_base, centre_basket[:3]


def default_exclusion_reasons(demo, target_layout):
    reasons = []
    if bool(demo.get("unsafe", False)):
        reasons.append("manifest_unsafe")
    layout = demo.get("layout")
    if not layout or layout != target_layout:
        reasons.append("layout_mismatch_or_unknown")
    if not demo.get("quadrant"):
        reasons.append("missing_quadrant")
    if demo.get("landed_pin") is None:
        reasons.append("missing_landed_pin")
    return reasons


def _read_jsonl(path):
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: malformed JSONL: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def convert_demo(demo, target_layout, kin, tool_length_m=DEFAULT_TOOL_LENGTH_M):
    filename = Path(demo["file"]).name
    if filename != demo["file"]:
        raise ValueError(f"manifest file must be a basename, got {demo['file']!r}")
    src = TEACH_DIR / filename
    if not src.is_file():
        raise FileNotFoundError(src)
    record = json.loads(src.read_text())
    run = Path(record.get("run", ""))
    state_path = run / "robot_state.jsonl"
    if not run.is_dir() or not state_path.is_file():
        raise FileNotFoundError(f"missing recorded trajectory: {state_path}")

    Tbb = _transform(record.get("T_base_basket"), f"{filename}:T_base_basket")
    keys = transform_keyposes(record.get("poses", {}), Tbb, kin, tool_length_m)
    rows = _read_jsonl(state_path)
    states = [r for r in rows if r.get("kind") == "state"]
    if not states:
        raise ValueError(f"{state_path}: no state samples")

    t = np.asarray([float(r["t"]) for r in states], dtype=np.float64)
    q_deg = np.asarray([r["posj"] for r in states], dtype=np.float64)
    posx = np.asarray([r["posx"] for r in states], dtype=np.float64)
    if q_deg.ndim != 2 or q_deg.shape[1] != 6 or posx.ndim != 2 or posx.shape[1] < 3:
        raise ValueError(f"{state_path}: bad state vector dimensions")
    if not np.isfinite(t).all() or not np.isfinite(q_deg).all() or not np.isfinite(posx).all():
        raise ValueError(f"{state_path}: non-finite trajectory values")
    dt = np.diff(t)
    if len(dt) and np.any(dt <= 0):
        raise ValueError(f"{state_path}: timestamps are not strictly increasing")

    flange = np.asarray([kin.fk(np.radians(q)) for q in q_deg])
    tcp = flange.copy()
    tcp[:, :3, 3] += flange[:, :3, 2] * tool_length_m
    T_basket_base = np.linalg.inv(Tbb)
    basket_tcp = np.einsum("ij,njk->nik", T_basket_base, tcp)
    fk_errors = np.linalg.norm(flange[:, :3, 3] * 1000. - posx[:, :3], axis=1)

    key_events = sorted((float(r["t"]), str(r.get("pose"))) for r in rows
                        if r.get("kind") == "event" and r.get("event") == "teach_pose" and r.get("pose"))
    event_times = [e[0] for e in key_events]
    phases = []
    for stamp in t:
        idx = bisect.bisect_right(event_times, float(stamp)) - 1
        phases.append("before_first_keypose" if idx < 0 else f"after_{key_events[idx][1]}")

    key_idx = KEYPOSES.index("grasp")
    centre_base, centre_basket = _plate_centre_estimate(
        keys["keypose_T_base_tcp"][key_idx], Tbb, _plate_geometry()[0])
    layout = demo.get("layout")
    unsafe = bool(demo.get("unsafe", False))
    reasons = default_exclusion_reasons(demo, target_layout)

    finite_events = [r for r in rows if r.get("kind") == "event"]
    speed = np.linalg.norm(np.diff(tcp[:, :3, 3], axis=0), axis=1) / dt if len(dt) else np.zeros(0)
    joint_speed = np.max(np.abs(np.diff(q_deg, axis=0)) / dt[:, None], axis=1) if len(dt) else np.zeros(0)
    metadata = {
        "schema_version": 1,
        "demo_id": Path(filename).stem,
        "source_demo": str(src.relative_to(ROOT)),
        "source_state_log": str(state_path.relative_to(ROOT)),
        "quadrant": demo.get("quadrant"),
        "file_quadrant": record.get("quadrant"),
        "file_quadrant_differs_from_manifest": record.get("quadrant") != demo.get("quadrant"),
        "landed_pin": demo.get("landed_pin"),
        "layout": layout,
        "target_layout": target_layout,
        "unsafe": unsafe,
        "included_by_default": not reasons,
        "default_exclusion_reasons": reasons,
        "T_base_basket": Tbb.tolist(),
        "basket_fit_mm": record.get("basket_fit_mm"),
        "tool_length_m": float(tool_length_m),
        "keypose_count": len(KEYPOSES),
        "keypose_fk_flange_xyz_error_mm": keys["keypose_fk_flange_xyz_error_mm"].tolist(),
        "trajectory_sample_count": int(len(states)),
        "trajectory_duration_s": float(t[-1] - t[0]),
        "median_sample_hz": float(1. / np.median(dt)) if len(dt) else None,
        "trajectory_fk_flange_xyz_error_mm": {
            "median": float(np.median(fk_errors)), "p95": float(np.percentile(fk_errors, 95)),
            "max": float(np.max(fk_errors)),
        },
        "trajectory_fk_flange_xyz_error_over_1mm_samples": int(np.count_nonzero(fk_errors > 1.0)),
        "trajectory_fk_posx_alignment_note": (
            "Some continuous samples exceed 1 mm FK-vs-recorded flange posx; cause/timing alignment not isolated. "
            "Derived TCP is computed from joint FK; recorded posx is preserved for audit."
            if np.any(fk_errors > 1.0) else "All continuous FK-vs-recorded flange posx samples are within 1 mm."
        ),
        "recorded_motion_peak_joint_axis_deg_s": float(np.max(joint_speed)) if len(joint_speed) else 0.,
        "derived_tcp_peak_speed_m_s": float(np.max(speed)) if len(speed) else 0.,
        "events": finite_events,
        "plate_centre_estimate": {
            "base_xyz_m": centre_base.tolist(),
            "basket_xyz_m": centre_basket.tolist(),
            "status": "model-based estimate, not measured ground truth",
            "method": "grasp TCP + projected local-z axis * (rl/config.yaml rim_r - grasp_inset)",
        },
    }
    trajectory = {
        "t_unix_s": t,
        "t_rel_s": t - t[0],
        "joint_deg": q_deg,
        "recorded_flange_posx_mm_deg": posx,
        "recorded_leg": np.asarray([str(r.get("leg") or "") for r in states]),
        "phase_after_keypose": np.asarray(phases),
        "T_base_flange": flange,
        "T_base_tcp": tcp,
        "T_basket_tcp": basket_tcp,
        "fk_flange_xyz_error_mm": fk_errors,
    }
    return metadata, keys, trajectory


def convert_manifest(manifest_path, output_dir, target_layout=DEFAULT_TARGET_LAYOUT):
    manifest_path = Path(manifest_path)
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(f"output directory already exists; refusing to overwrite: {out}")
    manifest = json.loads(manifest_path.read_text())
    demos = manifest.get("demos")
    if not isinstance(demos, list) or not demos:
        raise ValueError(f"{manifest_path}: expected non-empty demos list")
    out.parent.mkdir(parents=True, exist_ok=True)
    kin = Kinematics()
    rim_to_centre, geom = _plate_geometry()
    summary = []
    with tempfile.TemporaryDirectory(prefix=f".{out.name}.", dir=out.parent) as tmp:
        tmp = Path(tmp)
        for demo in demos:
            metadata, keys, trajectory = convert_demo(demo, target_layout, kin)
            stem = metadata["demo_id"]
            npz_name, json_name = f"{stem}.npz", f"{stem}.json"
            np.savez_compressed(tmp / npz_name, **keys, **trajectory)
            detail = dict(metadata, dataset=npz_name, summary=f"{stem} summary -> {json_name}")
            (tmp / json_name).write_text(json.dumps(detail, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            summary.append(detail)

        run_summary = {
            "schema_version": 1,
            "manifest": str(manifest_path),
            "target_layout": target_layout,
            "tool_length_m": DEFAULT_TOOL_LENGTH_M,
            "plate_geometry": {**geom, "rim_to_centre_m": rim_to_centre},
            "demo_count": len(summary),
            "included_by_default_count": sum(bool(d["included_by_default"]) for d in summary),
            "demos": summary,
        }
        (tmp / "summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        with (tmp / "summary.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["demo", "quadrant", "pin", "layout", "unsafe", "included_by_default", "samples",
                        "duration_s", "keypose_fk_max_mm", "trajectory_fk_p95_mm", "exclude_reasons"])
            for d in summary:
                w.writerow([d["demo_id"], d["quadrant"], d["landed_pin"], d["layout"], d["unsafe"],
                            d["included_by_default"], d["trajectory_sample_count"], f'{d["trajectory_duration_s"]:.3f}',
                            f'{max(d["keypose_fk_flange_xyz_error_mm"]):.4f}',
                            f'{d["trajectory_fk_flange_xyz_error_mm"]["p95"]:.4f}',
                            ";".join(d["default_exclusion_reasons"])])
        if out.exists():
            raise FileExistsError(f"output directory appeared during conversion; refusing to overwrite: {out}")
        tmp.rename(out)
    return run_summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=TEACH_DIR / "bc_demos.json")
    p.add_argument("--out", type=Path, required=True, help="new output directory; existing paths are refused")
    p.add_argument("--target-layout", default=DEFAULT_TARGET_LAYOUT)
    args = p.parse_args(argv)
    summary = convert_manifest(args.manifest, args.out, args.target_layout)
    print(json.dumps({"out": str(args.out), "demo_count": summary["demo_count"],
                      "included_by_default_count": summary["included_by_default_count"],
                      "demos": [{"demo": d["demo_id"], "included": d["included_by_default"],
                                 "samples": d["trajectory_sample_count"],
                                 "keypose_fk_max_mm": max(d["keypose_fk_flange_xyz_error_mm"]),
                                 "exclude": d["default_exclusion_reasons"]} for d in summary["demos"]]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
