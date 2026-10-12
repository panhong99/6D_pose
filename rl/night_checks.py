"""Fail-closed checks for the overnight IL pipeline; standard library only.

Commands print one JSON object on success and return nonzero on failed checks.
These checks qualify simulation training inputs, not physical robot deployment.
"""
import argparse
import ast
import hashlib
import json
import math
import re
import struct
import sys
import zipfile
from pathlib import Path

RL_DIR = Path(__file__).resolve().parent
SOURCES = ("plate_env.py", "rl_utils.py", "config.yaml")


class CheckError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise CheckError(message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"invalid {label}: {value!r}")
    return value


def number(value, label, lower=None, upper=None):
    require(type(value) in (int, float) and math.isfinite(value), f"nonfinite/invalid {label}: {value!r}")
    require(lower is None or value >= lower, f"{label} below {lower}: {value}")
    require(upper is None or value <= upper, f"{label} above {upper}: {value}")
    return value


def tagged(log, tag):
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", Path(log).read_text())
    matches = [line.strip()[len(tag) + 1:] for line in text.splitlines()
               if line.strip().startswith(tag + " ")]
    require(len(matches) == 1, f"expected one {tag} in {log}, found {len(matches)}")
    return matches[0]


def npz_shape(path, key):
    """Read only the NPY header, without importing NumPy or loading array data."""
    with zipfile.ZipFile(path) as archive, archive.open(key + ".npy") as stream:
        require(stream.read(6) == b"\x93NUMPY", f"invalid NPY magic: {path}:{key}")
        version = tuple(stream.read(2))
        require(version in ((1, 0), (2, 0), (3, 0)), f"unsupported NPY version: {version}")
        width = 2 if version == (1, 0) else 4
        size = struct.unpack("<H" if width == 2 else "<I", stream.read(width))[0]
        require(size <= 65536, f"NPY header too large: {path}:{key}")
        raw = stream.read(size)
        require(len(raw) == size, f"truncated NPY header: {path}:{key}")
        header = ast.literal_eval(raw.decode("utf-8" if version == (3, 0) else "latin1"))
        shape = header.get("shape")
        require(isinstance(shape, tuple) and all(type(n) is int and n >= 0 for n in shape),
                f"invalid NPY shape: {path}:{key}")
        return shape


def data_check(directory, seed, dart, episodes=200, verify_log=None, obs_dim=46, source_dir=RL_DIR):
    out = Path(directory).resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    integer(episodes, "required episodes", 1)
    number(dart, "required DART", 0)
    require(manifest.get("status") == "complete", "collection is not complete")
    require(manifest.get("task") == "stage2_ready_grasp_lift_hold", "unexpected task")
    require(manifest.get("phase_switching") == "state-based per env", "unexpected expert phase switching")
    require(integer(manifest.get("requested"), "requested episodes", 1) == episodes, "requested episode count differs")
    require(integer(manifest.get("seed"), "seed") == seed, "seed differs")
    require(abs(number(manifest.get("dart_noise"), "DART", 0) - dart) < 1e-12, "DART differs")
    require(abs(number(manifest.get("slow_zone_m"), "slow zone", 0) - 0.02) < 1e-12,
            "expected proportional expert slow_zone_m=0.02")
    source_hashes = manifest.get("sources_sha256", {})
    for name in SOURCES:
        expected = hashlib.sha256((Path(source_dir) / name).read_bytes()).hexdigest()
        require(source_hashes.get(name) == expected, f"collector source/config differs: {name}")
        require(hashlib.sha256((out / "source_snapshot" / name).read_bytes()).hexdigest() == expected,
                f"source snapshot differs: {name}")
    entries = manifest.get("episodes")
    require(isinstance(entries, list) and len(entries) == episodes, "saved episode count differs")
    seen, frames, splits = set(), 0, {"train": 0, "validation": 0}
    for entry in entries:
        path = (out / entry["file"]).resolve()
        require(path.is_relative_to(out) and path not in seen, "duplicate or escaping episode path")
        seen.add(path)
        count = integer(entry.get("frames"), "episode frames", 1)
        require(entry.get("split") in splits, "invalid dataset split")
        splits[entry["split"]] += 1
        frames += count
        require(npz_shape(path, "obs") == (count, obs_dim), f"unexpected observation shape: {path}")
        require(npz_shape(path, "next_obs") == (count, obs_dim), f"unexpected next observation shape: {path}")
    require(integer(manifest.get("total_frames"), "total frames", 1) == frames, "total frame count differs")
    require(all(splits.values()), "train and validation must both be nonempty")
    if verify_log is not None:
        report = json.loads(tagged(verify_log, "VERIFY_RESULT"))
        require(report.get("all_passed") is True, "verification did not pass")
        expected = {"episodes": episodes, "frames": frames, **splits}
        for key, value in expected.items():
            require(integer(report.get(key), "verification " + key) == value, f"verification {key} differs")
    return {"directory": str(out), "episodes": episodes, "frames": frames, **splits,
            "obs_dim": obs_dim, "verified": verify_log is not None}


def checkpoint_check(log):
    path = Path(tagged(log, "BC_CHECKPOINT")).resolve()
    require(path.is_file(), f"BC checkpoint missing: {path}")
    return {"checkpoint": str(path)}


def validate_gate(result, checkpoint, episodes, min_success, max_collision):
    number(min_success, "minimum success", 0, 1)
    number(max_collision, "maximum collision", 0, 1)
    checkpoint = Path(checkpoint).resolve()
    require(checkpoint.is_file(), f"checkpoint missing: {checkpoint}")
    require(integer(result.get("stage"), "stage") == 2, "gate is not stage 2")
    require(integer(result.get("episodes"), "gate episodes", 1) == episodes, "gate episode count differs")
    require(Path(result.get("checkpoint", "")).resolve() == checkpoint, "gate checkpoint differs")
    rates = {name: number(result.get(name), name, 0, 1)
             for name in ("success", "collision", "plate_lost", "time_out", "unfinished")}
    require(rates["unfinished"] == 0, "gate has unfinished episodes")
    require(abs(sum(rates.values()) - 1) < 1e-5, "gate outcomes do not sum to one")
    require(rates["success"] >= min_success, f"BC success below qualification: {rates['success']} < {min_success}")
    require(rates["collision"] <= max_collision,
            f"BC collision above qualification: {rates['collision']} > {max_collision}")
    return {**result, "checkpoint": str(checkpoint), "qualified": True,
            "min_success": min_success, "max_collision": max_collision}


def gate_check(log, checkpoint, episodes=512, min_success=0.85, max_collision=0.02):
    return validate_gate(json.loads(tagged(log, "EVAL_JSON")), checkpoint, episodes, min_success, max_collision)


def select_check(paths, out):
    candidates = []
    for path in paths:
        result = json.loads(Path(path).read_text())
        require(result.get("qualified") is True, f"candidate not qualified: {path}")
        candidates.append(validate_gate(result, result["checkpoint"], 512, result["min_success"], result["max_collision"]))
    require(candidates, "no qualified BC candidates")
    selected = max(candidates, key=lambda result: (result["success"], -result["collision"]))
    target = Path(out)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(selected, indent=2) + "\n")
    temporary.replace(target)
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data")
    data.add_argument("directory")
    data.add_argument("--seed", type=int, required=True)
    data.add_argument("--dart", type=float, required=True)
    data.add_argument("--episodes", type=int, default=200)
    data.add_argument("--obs-dim", type=int, default=46)
    data.add_argument("--verify-log")
    gate = commands.add_parser("gate")
    gate.add_argument("log")
    gate.add_argument("--checkpoint", required=True)
    gate.add_argument("--episodes", type=int, default=512)
    gate.add_argument("--min-success", type=float, default=0.85)
    gate.add_argument("--max-collision", type=float, default=0.02)
    checkpoint = commands.add_parser("checkpoint")
    checkpoint.add_argument("log")
    select = commands.add_parser("select")
    select.add_argument("paths", nargs="+")
    select.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "data":
            result = data_check(args.directory, args.seed, args.dart, args.episodes, args.verify_log, args.obs_dim)
        elif args.command == "gate":
            result = gate_check(args.log, args.checkpoint, args.episodes, args.min_success, args.max_collision)
        elif args.command == "checkpoint":
            result = checkpoint_check(args.log)
        else:
            result = select_check(args.paths, args.out)
        print(json.dumps(result, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, struct.error, zipfile.BadZipFile) as error:
        print(f"night check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
