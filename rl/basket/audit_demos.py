"""Read-only basket NPZ audit. Counts recorded successes; never infers physical qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(data):
    report = {"data": str(data.resolve()), "streams": {}, "errors": [], "files": [],
              "qualification": "unverified", "missing_provenance": []}
    for path in sorted(data.glob("batch_*.npz")):
        try:
            with np.load(path, allow_pickle=False) as z:
                required = {"obs", "act", "mask", "success", "dart", "live"}
                missing = sorted(required - set(z.files))
                if missing:
                    report["errors"].append({"file": path.name, "missing": missing})
                    continue
                obs, act, mask, success = z["obs"], z["act"], z["mask"], z["success"]
                stream = "dart" if bool(z["dart"]) else "clean"
                counts = report["streams"].setdefault(stream, {"batches": 0, "episodes": 0, "recorded_success": 0, "frames": 0})
                counts["batches"] += 1
                counts["episodes"] += len(success)
                counts["recorded_success"] += int(success.sum())
                counts["frames"] += int((mask & success[:, None]).sum())
                valid = obs.ndim == act.ndim == 3 and obs.shape[:2] == act.shape[:2] == mask.shape
                valid = valid and obs.shape[-1] == 17 and act.shape[-1] == 7 and len(success) == len(obs)
                finite = bool(np.isfinite(obs).all() and np.isfinite(act).all())
                if not valid or not finite:
                    report["errors"].append({"file": path.name, "shape_ok": valid, "finite": finite})
                absent = [k for k in ("executed_action", "tcp_speed_mps", "joint_speed_deg_s", "shield_blocked", "collector_config", "source_hash") if k not in z.files]
                if absent:
                    report["missing_provenance"].append({"file": path.name, "fields": absent})
                report["files"].append({"file": path.name, "success": int(success.sum()), "dart": stream == "dart", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        except Exception as exc:
            report["errors"].append({"file": path.name, "exception": str(exc)})
    report["target_per_stream"] = 1500
    report["shortfall"] = {k: max(0, 1500 - report["streams"].get(k, {}).get("recorded_success", 0)) for k in ("clean", "dart")}
    report["qualified_physical_success"] = None
    report["reason"] = "Physical/safety validity cannot be established from these fields; current v2 collection is documented as forced attachment and disabled shield. Recorded success is not qualified physical success."
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = audit(a.data)
    if a.out.exists():
        raise SystemExit("Refusing to overwrite an existing audit")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("files", "missing_provenance")}, indent=2))
