#!/usr/bin/env bash
# Launchers that set the right Python / ROS environment for each calibration/ tool.
#   bash calibration/run.sh connect            terminal 1: Doosan bringup (ROS 2 Jazzy, system Python)  [Human]
#   bash calibration/run.sh stream             read-only flange pose stream -> UDP 5006 (needed by calibrate.py)
#   bash calibration/run.sh viz [--plan P]     read-only RViz: real robot, FoundationPose plate, plan preview (UDP 5011)
#   bash calibration/run.sh pipeline [args]    terminal 2: cube pipeline in conda foundationpose (see pipeline.py)
#   bash calibration/run.sh sim [args]         Isaac Sim digital twin of the cell (see sim_scene.py)
#   bash calibration/run.sh mover --plan P --leg L [--execute]   one plan leg, Human-run
#   bash calibration/run.sh wrist_look         read-only: wrist photo at the current pose + model pins (starts its own stream)
#   bash calibration/run.sh tests              hardware-free regression tests
# Other tools run directly in conda:  python calibration/{calibrate,camera_tools,robot_model}.py <mode> ...
# After a reboot this PC needs the controller subnet:  sudo ip addr add 192.168.127.5/24 dev enp5s0
set -eo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOOSAN_SETUP="${DOOSAN_SETUP:-$DIR/../../doosan_ws/install/setup.bash}"
ISAAC_ROOT="${ISAAC_ROOT:-/home/pan/isaacsim-5.1.0}"
mode="${1:-}"; shift || true

ros() {   # clean system-Python environment with ROS + Doosan sourced (conda's Python breaks rclpy)
  exec env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u CONDA_PREFIX -u CONDA_DEFAULT_ENV PATH=/usr/bin:/bin \
    bash -c 'set -e; source /opt/ros/jazzy/setup.bash; source "$1"; shift; exec "$@"' ros-env "$DOOSAN_SETUP" "$@"
}

case "$mode" in
  connect)
    # gui:=false: no bringup RViz (its /dsr01/joint_states stay 0 in this build); our view is `run.sh viz`.
    # Pass gui:=true to get it back (later arguments win).
    ros ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py mode:=real host:=192.168.127.100 port:=12345 model:=m1013 gui:=false "$@" ;;
  stream)
    ros /usr/bin/python3 -u "$DIR/ros2_nodes.py" flange "$@" ;;
  viz)
    ros bash -c 'python3 -u "$0/ros2_nodes.py" viz "${@}" & node=$!; trap "kill $node 2>/dev/null" EXIT;
                 rviz2 -d "$0/viz.rviz"' "$DIR" "$@" ;;
  pipeline)
    cd "$DIR/.." && exec conda run --no-capture-output -n foundationpose python -u "$DIR/pipeline.py" "$@" ;;
  sim)
    unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH LD_LIBRARY_PATH
    cd "$DIR" && exec "$ISAAC_ROOT/python.sh" -u "$DIR/sim_scene.py" "$@" ;;
  mover)      # one leg of a plan.json, Human-run (asks the opening phrase + go; --execute to really move)
    ros /usr/bin/python3 -u "$DIR/ros2_nodes.py" mover "$@" ;;
  wrist_look)
    mkdir -p "$DIR/data/basket/wrist_look"
    ( ros /usr/bin/python3 -u "$DIR/ros2_nodes.py" flange ) > "$DIR/data/basket/wrist_look/stream.log" 2>&1 & sp=$!
    trap 'kill $sp 2>/dev/null' EXIT
    cd "$DIR/.." && conda run --no-capture-output -n foundationpose python -u "$DIR/camera_tools.py" wrist_look "$@" ;;
  tests)
    cd "$DIR" && exec conda run --no-capture-output -n foundationpose python -m unittest tests "$@" ;;
  *)
    sed -n '2,11p' "$0"; exit 2 ;;
esac
