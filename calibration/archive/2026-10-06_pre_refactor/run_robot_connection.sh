#!/usr/bin/env bash
# Terminal 1: connection only. The pipeline is started in terminal 2.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOOSAN_SETUP="${DOOSAN_SETUP:-$SCRIPT_DIR/../../doosan_ws/install/setup.bash}"
exec env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH PATH=/usr/bin:/bin \
  bash -c '
    set -e
    source /opt/ros/jazzy/setup.bash
    source "$1"
    shift
    exec ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py \
      mode:=real host:=192.168.127.100 port:=12345 model:=m1013 "$@"
  ' robot-connection "$DOOSAN_SETUP" "$@"
