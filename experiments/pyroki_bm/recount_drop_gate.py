"""Recount saved PyRoki sweep JSONs with the planner's drop gate.

The original table applied the carry-only wrist excursion limit to the
shared vertical drop. This reads each existing per-job JSON without changing
it and prints a corrected aggregate; it does not rerun either planner.
"""
import argparse
import json
from pathlib import Path


def recount(folder):
    files = sorted(p for p in folder.glob("*.json") if p.name != "table.json")
    if not files:
        raise ValueError(f"no per-job JSON files in {folder}")
    counts = {"jobs": 0, "drop_valid": 0, "legacy_full_valid": 0,
              "pyroki_full_valid": 0, "both_full_valid": 0}
    for path in files:
        row = json.loads(path.read_text())
        required = ("legacy", "pyroki", "shared_vertical_drop")
        if not all(key in row for key in required):
            raise ValueError(f"missing benchmark fields: {path}")
        drop = row["shared_vertical_drop"]
        # Match benchmark_saved.insertion_gate: wrist is diagnostic on drop.
        drop_valid = (bool(drop["finite"]) and bool(drop["within_limits"])
                      and float(drop["min_insert_clearance"]) >= 1.0)
        legacy = bool(row["legacy"]["valid"]) and drop_valid
        pyroki = bool(row["pyroki"]["valid"]) and drop_valid
        counts["jobs"] += 1
        counts["drop_valid"] += drop_valid
        counts["legacy_full_valid"] += legacy
        counts["pyroki_full_valid"] += pyroki
        counts["both_full_valid"] += legacy and pyroki
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    print(json.dumps(recount(args.folder), sort_keys=True))
