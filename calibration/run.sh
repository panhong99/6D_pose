#!/usr/bin/env bash
# Launchers that set the right Python / ROS environment for each calibration/ tool.
#   bash calibration/run.sh connect            terminal 1: Doosan bringup (ROS 2 Jazzy, system Python)  [Human]
#   bash calibration/run.sh stream             read-only flange pose stream -> UDP 5006 (needed by calibrate.py)
#   bash calibration/run.sh viz [--plan P]     read-only RViz: real robot, FoundationPose plate, plan preview (UDP 5011)
#   bash calibration/run.sh pipeline [args]    terminal 2: cube pipeline in conda foundationpose (see pipeline.py)
#   bash calibration/run.sh demo4 [--execute]  meeting demo: Q4, Q3, Q1, Q2 one --auto --sim_q run each; DEMO_PINS="Q3:9 Q3:4:white" picks pins/colour
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
  demo4)      # (10/15 보스) meeting demo: Q4 -> Q3 -> Q1 -> Q2, each one --auto run (--sim_q settings); Human puts the plate
              # back on the stand between runs.  Extra args go to every run (e.g. --execute).  Pins: DEMO_PINS="Q4:6 Q3:6 Q1:6 Q2:5"
    cd "$DIR/.."
    # (10/16) optional colour: Q3:9:white (default red).  (10/17) optional landed pins of earlier inserts aimed at this
    # pin, comma separated (Q4:10:red:7.5,8), used with DEMO_INSERT_POLICY=<seen-gap policy .pt>
    # (10/18 보스) same quadrant: always the larger pin first -> the list is reordered inside each quadrant (the quadrant
    # order stays), and every plate put earlier in this demo (all quadrants) is passed on (--basket_plates Q:pin) so the
    # carry goes over it.  After a failed run Human says whether a plate stayed in the basket.
    pins_in="${DEMO_PINS:-Q4:6 Q3:6 Q1:6 Q2:5}"
    pins_run=$(python3 -c 'import sys; e = sys.argv[1:]; q = [x.split(":")[0] for x in e]
s = {k: iter(sorted((x for x in e if x.split(":")[0] == k), key=lambda x: -int(x.split(":")[1]))) for k in set(q)}
print(" ".join(next(s[k]) for k in q))' $pins_in)
    [ "$pins_run" != "$(echo $pins_in)" ] && echo "### 같은 칸은 큰 번호부터: $pins_in -> $pins_run"
    placed="${DEMO_BASKET_PLATES:-}"   # plates already in the basket before this demo, e.g. DEMO_BASKET_PLATES="Q3:10 Q4:10"
    for qp in $pins_run; do
      IFS=: read -r q pin color landed <<< "$qp"; color=${color:-red}
      pol=()
      [ "${DEMO_PHOTOS:-1}" = 1 ] && pol+=(--photo_after_release)   # (10/19 보스) wrist photo after each release, DEMO_PHOTOS=0 = off
      [ -n "${placed// /}" ] && pol+=(--basket_plates $placed)
      if [ -n "${DEMO_INSERT_POLICY:-}" ]; then
        pol+=(--insert_policy "$DEMO_INSERT_POLICY")
        [ -n "$landed" ] && pol+=(--insert_landed_pins ${landed//,/ })
      elif [ -n "$landed" ]; then
        echo "!!! ${qp}: 착지 기록은 DEMO_INSERT_POLICY 와 함께만 씁니다"; continue
      fi
      echo; echo "=== ${q} 핀 ${pin} (${color}${landed:+, 지난 착지 $landed}${DEMO_INSERT_POLICY:+, 정책}): 접시를 받침 위에 올려 두고 Enter (건너뛰기: s, 끝내기: q) ==="
      read -r ans
      [ "$ans" = q ] && break
      [ "$ans" = s ] && continue
      if conda run --no-capture-output -n foundationpose python -u "$DIR/pipeline.py" --task plate --plate_policy ik_bc \
        --place basket --insert quadrant --quadrant "$q" --pin "$pin" --plate_color "$color" --sim_q --auto "${pol[@]}" "$@"; then
        placed="$placed $q:$pin"
      else
        echo "!!! ${q} 핀 ${pin} 실패 - 로그 확인. 이 접시가 바구니에 꽂혀 남았으면 y, 아니면 Enter"
        read -r left
        [ "$left" = y ] && placed="$placed $q:$pin"
      fi
    done ;;
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
