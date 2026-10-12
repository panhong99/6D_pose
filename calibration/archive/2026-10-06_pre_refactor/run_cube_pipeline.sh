#!/usr/bin/env bash
# Terminal 2: one command owns perception, planning, ROS publication and supervised motion.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
exec conda run --no-capture-output -n foundationpose python -u \
  "$SCRIPT_DIR/cube_pipeline.py" "$@"
