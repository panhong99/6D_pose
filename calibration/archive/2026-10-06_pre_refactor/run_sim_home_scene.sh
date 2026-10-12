#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_ROOT="${ISAAC_ROOT:-/home/pan/isaacsim-5.1.0}"
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH LD_LIBRARY_PATH
cd "$SCRIPT_DIR"
exec "$ISAAC_ROOT/python.sh" -u "$SCRIPT_DIR/sim_home_scene.py" "$@"
