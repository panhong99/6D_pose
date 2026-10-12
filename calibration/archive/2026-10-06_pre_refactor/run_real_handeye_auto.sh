#!/usr/bin/env bash
# Hands-free real hand-eye capture.  The robot is moved ONLY by the human; nothing here commands it.
# Prerequisite (separate terminal, started by the human): Doosan bringup connected to the controller, e.g.
#   source /opt/ros/jazzy/setup.bash && source ~/pan/doosan_ws/install/setup.bash
#   ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py mode:=real host:=<robot_ip> port:=12345 model:=m1013
# Usage: ./calibration/run_real_handeye_auto.sh --marker_size 0.10 [other auto_handeye_capture.py args]
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_ID="${ROBOT_ID:-dsr01}"
CTRL_NS="${CTRL_NS:-dsr_controller2}"

# 1) read-only pose stream with system Python + ROS (conda python breaks rclpy)
env -i HOME="$HOME" USER="$USER" PATH=/usr/bin:/bin DISPLAY="${DISPLAY:-}" bash -c "
  source /opt/ros/jazzy/setup.bash && source $HOME/pan/doosan_ws/install/setup.bash
  if ! ros2 service list 2>/dev/null | grep -q "/$ROBOT_ID/$CTRL_NS/aux_control/get_current_tool_flange_posx"; then
    echo 'Doosan bringup not running (service /$ROBOT_ID/$CTRL_NS/aux_control/get_current_tool_flange_posx missing)'; exit 1; fi
  exec /usr/bin/python3 '$SCRIPT_DIR/ros2_flange_udp.py' --robot_id '$ROBOT_ID' --controller_ns '$CTRL_NS'" &
STREAM_PID=$!
trap 'kill $STREAM_PID 2>/dev/null || true' EXIT
sleep 3
kill -0 $STREAM_PID 2>/dev/null || { echo 'pose stream failed to start'; exit 1; }

# 2) camera + automatic capture (conda env)
cd "$SCRIPT_DIR"
conda run --no-capture-output -n foundationpose python auto_handeye_capture.py "$@"
