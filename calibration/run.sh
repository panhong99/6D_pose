#!/usr/bin/env bash
# Launchers that set the right Python / ROS environment for each calibration/ tool.
#   bash calibration/run.sh connect            terminal 1: Doosan bringup (ROS 2 Jazzy, system Python)  [Human]
#   bash calibration/run.sh stream             read-only flange pose stream -> UDP 5006 (needed by calibrate.py)
#   bash calibration/run.sh viz [--plan P]     read-only RViz: real robot, FoundationPose plate, plan preview (UDP 5011)
#   bash calibration/run.sh pipeline [args]    terminal 2: cube pipeline in conda foundationpose (see pipeline.py)
#   bash calibration/run.sh demo4 [--execute]  meeting demo: Q4, Q3, Q1, Q2 one --auto --sim_q run each; DEMO_PINS="Q3:back Q3:front:white" picks pins/colour
#   bash calibration/run.sh demo1022 Q3|Q4 [--execute]   10/22 demo: two plates in that quadrant, fixed order (collab/DEMO_20261022.md)
#   bash calibration/run.sh markers            read-only: live fixed-camera view of the 4 basket markers (q quits)
#   bash calibration/run.sh basket_check       read-only: marker blocks vs the stored layout (which block moved)
#   bash calibration/run.sh basket_relayout    store moved marker blocks where they are now (typed y; new layout file)
#   bash calibration/run.sh look4 [--execute]  plate-free wrist photo per quadrant (look pose -> photo -> home); LOOK_QUADRANTS="Q1 Q3"; live marker view first (LOOK_SHOW=0 off), marker-free basket pose saved (LOOK_VISION=0 off), 30 deg/s
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
  demo1022)   # (10/19 보스) the 10/22 demo, one word per quadrant: two plates, the order the gripper side needs
    qd="${1:-}"; shift || true
    for a in "$@"; do              # (10/19) two pasted lines ran as '--executebash calibration/run.sh demo1022 Q3': stop here
      case "$a" in --execute) ;; *) echo "!!! demo1022: 알 수 없는 인자 '$a' (허용: --execute). 명령은 한 줄씩 실행"; exit 2 ;; esac
    done
    case "$qd" in
      Q3) pins="Q3:back:red Q3:front:white" ;;      # (10/19 보스) always red first, then white; Q3: robot-side (large pins) first (gripper on the far side)
      Q4) pins="Q4:front:red Q4:back:white" ;;      # (10/19 보스) always red first, then white; Q4: far (small pins) first (gripper on the robot side)
      # (10/23 보스: front = small pin numbers, back = large; the labels were swapped, the order of the ends is unchanged)
      *) echo "usage: bash calibration/run.sh demo1022 Q3|Q4 [--execute]   (collab/DEMO_20261022.md)"; exit 2 ;;
    esac
    ( cd "$DIR/.." && conda run --no-capture-output -n foundationpose python -u "$DIR/camera_tools.py" basket_check ) || {
      echo "!!! 바구니 마커 점검 실패 - 위 메시지 확인. 바구니가 비어 있을 때만 블록 판정이 맞음"
      echo "    (빈 바구니에서 블록이 밀렸으면: bash calibration/run.sh basket_relayout)"; exit 1; }
    DEMO_PINS="$pins" DEMO_BASKET_PLATES="" exec bash "$DIR/run.sh" demo4 "$@" ;;
  basket_check)     # (10/19 보스) read-only: which marker block moved (exit 0 ok, 3 moved, 4 unsure)
    cd "$DIR/.." && exec conda run --no-capture-output -n foundationpose python -u "$DIR/camera_tools.py" basket_check "$@" ;;
  basket_relayout)  # (10/19 보스) the moved blocks stored where they are now, in the frame of the unmoved ones (asks y)
    cd "$DIR/.." && exec conda run --no-capture-output -n foundationpose python -u "$DIR/camera_tools.py" basket_check --relayout "$@" ;;
  markers)    # (10/19) pre-demo check: all 4 basket markers seen by the fixed camera (close with q before the demo)
    cd "$DIR/.." && exec conda run --no-capture-output -n foundationpose python -u "$DIR/camera_tools.py" markers_live "$@" ;;
  demo4)      # (10/15 보스) meeting demo: Q4 -> Q3 -> Q1 -> Q2, each one --auto run (--sim_q settings); Human puts the plate
              # back on the stand between runs.  Extra args go to every run (e.g. --execute).  Pins: DEMO_PINS="Q4:6 Q3:6 Q1:6 Q2:5"
    cd "$DIR/.."
    # (10/16) optional colour: Q3:9:white (default red).  (10/17) optional landed pins of earlier inserts aimed at this
    # pin, comma separated (Q4:10:red:7.5,8), used with DEMO_INSERT_POLICY=<seen-gap policy .pt>
    # (10/18 보스) same quadrant: always the larger pin first -> the list is reordered inside each quadrant (the quadrant
    # order stays), and every plate put earlier in this demo (all quadrants) is passed on (--basket_plates Q:pin) so the
    # carry goes over it.  After a failed run Human says whether a plate stayed in the basket.
    pins_in="${DEMO_PINS:-Q4:6 Q3:6 Q1:6 Q2:5}"
    # (10/19 보스) the pin may be front / back instead of a number: the first pin of that half with a plan.
    # (10/19 15:1x 보스 A) the order typed: the gripper is on the robot side of the plate in Q4 (back pin first there: white
    # into 7 touched the red at 11) but on the far side in Q3 (front first).  The planner also checks the gripper on standing plates.
    pins_run="$(echo $pins_in)"
    echo "### 순서 그대로: $pins_run (Q3는 back(큰 번호) 먼저, Q4는 front(작은 번호) 먼저 - 그리퍼 쪽)"
    placed="${DEMO_BASKET_PLATES:-}"   # plates already in the basket before this demo, e.g. DEMO_BASKET_PLATES="Q3:10 Q4:10"
    for qp in $pins_run; do
      IFS=: read -r q pin color landed <<< "$qp"; color=${color:-auto}   # (10/19 보스) auto = the plate on the stand
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
      chosen="$(mktemp)"; rm -f "$chosen"
      if [ "$pin" = front ] || [ "$pin" = back ]; then where=(--region "$pin" --chosen_pin_out "$chosen"); else where=(--pin "$pin"); fi
      if conda run --no-capture-output -n foundationpose python -u "$DIR/pipeline.py" --task plate --plate_policy ik_bc \
        --place basket --insert quadrant --quadrant "$q" "${where[@]}" --plate_color "$color" --sim_q --auto "${pol[@]}" "$@"; then
        [ -s "$chosen" ] && got=$(cat "$chosen") || got="$q:$pin"
        placed="$placed $got"; echo "--- ${got} 에 넣음"
      else
        [ -s "$chosen" ] && got=$(cat "$chosen") || got="$q:$pin"
        echo "!!! ${q} ${pin} 실패 - 로그 확인. 이 접시가 바구니(${got})에 꽂혀 남았으면 y, 아니면 Enter"
        read -r left
        if [ "$left" = y ]; then
          case "$got" in *:front|*:back) echo "!!! 고른 핀을 모름: 다음 실행 전에 DEMO_BASKET_PLATES로 알려 주세요";; *) placed="$placed $got";; esac
        fi
      fi
      rm -f "$chosen"
    done ;;
  look4)      # (10/19 보스) plate-free wrist photo of each quadrant (Codex tine labels): per quadrant the basket is measured
              # from the markers, a look pose is planned, then look leg -> photo -> home leg.  Human: empty gripper, no plate
              # in the basket view; types the opening statement once and `go` per leg.  --execute to really move (else checks only).
              # Quadrants: LOOK_QUADRANTS="Q1 Q2 Q3 Q4".  Speed: mover default (slowest).
    cd "$DIR/.."
    exe=0; for a in "$@"; do [ "$a" = --execute ] && exe=1; done
    open=()
    if [ $exe = 1 ]; then
      read -r -p 'Type exactly "area clear, e-stop in hand" to continue: ' ans
      [ "$ans" = "area clear, e-stop in hand" ] || { echo "중단"; exit 1; }
      open=(--opening_done)
    fi
    # (10/21 보스 "7배 빨라도 돼", then "더 빨라도") look legs at 30 deg/s; a later --vel_deg_s in "$@" wins
    mover() { ( ros /usr/bin/python3 -u "$DIR/ros2_nodes.py" mover --vel_deg_s 30 --acc_deg_s2 30 "$@" ); }
    if [ "${LOOK_SHOW:-1}" = 1 ]; then  # (10/21 보스) live fixed-camera marker view first: check the basket, q closes it (frees the camera)
      echo "=== 고정 카메라 화면: 마커 4개(초록) 보이는지 확인 후 창에서 q ==="
      conda run --no-capture-output -n foundationpose python -u "$DIR/camera_tools.py" markers_live || true
    fi
    if [ "${LOOK_VISION:-1}" = 1 ]; then  # (10/21 보스) marker-free basket pose (YOLOE + depth) from the fixed D455, robot at home
      stamp=$(date +%Y%m%d_%H%M%S); vdir="$DIR/data/basket/wrist_look"
      conda run --no-capture-output -n foundationpose python -u "$DIR/basket_vision.py" --camera fixed \
        --out "$vdir/basket_vision_${stamp}.json" --save "$vdir/basket_vision_${stamp}.npz" || echo "!!! 마커 없는 바구니 위치 실패 (촬영은 계속)"
    fi
    for qd in ${LOOK_QUADRANTS:-Q1 Q2 Q3 Q4}; do
      echo; echo "=== ${qd}: 손목 사진. 그리퍼 빈손, 바구니에 접시 없는지 확인 후 Enter (건너뛰기: s, 끝내기: q) ==="
      read -r ans; [ "$ans" = q ] && break; [ "$ans" = s ] && continue
      log="$DIR/data/basket/wrist_look/look4_$(date +%Y%m%d_%H%M%S)_${qd}.log"
      conda run --no-capture-output -n foundationpose python -u "$DIR/plate_plans.py" look_plans --measure --quadrants "$qd" 2>&1 | tee "$log" || true
      plan=$(sed -n 's/^plan: //p' "$log" | tail -1) || true
      req=$(grep -o "[^ ]*_${qd}\.req\.json" "$log" | tail -1) || true
      if [ -z "$plan" ] || [ -z "$req" ]; then echo "!!! ${qd}: 계획 없음 (위 로그) - 다음 분면"; continue; fi
      mover --plan "$plan" --leg "${qd}_look" --smooth --go_per leg "${open[@]}" "$@" || { echo "!!! ${qd}: 이동 안 함/실패 - 상태 확인 전 정지"; break; }
      photo_status=not_requested
      if [ $exe = 1 ]; then
        if conda run --no-capture-output -n foundationpose python -u "$DIR/camera_tools.py" pre_photo --req "$req"; then
          photo_status=ok
        else
          echo "!!! ${qd}: 사진 실패 (로봇은 홈으로 돌아갑니다)"
          photo_status=failed
        fi
      fi
      mover --plan "$plan" --leg "${qd}_home" --smooth --go_per leg "${open[@]}" "$@" || { echo "!!! ${qd}: 홈 복귀 실패 - 멈춤"; break; }
      if [ $photo_status = ok ]; then
        echo "--- ${qd} 이동/촬영 완료: ${req%.req.json}.raw.png / .json"
      elif [ $photo_status = failed ]; then
        echo "--- ${qd} 이동 완료, 사진 저장 실패: AZ 입력 미완료"
      else
        echo "--- ${qd} 계획/이동 검사 완료 (--execute 없음; 사진은 저장되지 않음)"
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
    sed -n '2,12p' "$0"; exit 2 ;;
esac
