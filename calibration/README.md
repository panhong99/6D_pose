# calibration/ — 실기 M1013 + RealSense D455 (eye-to-hand) 파이프라인

고정 D455가 큐브를 보고 → FoundationPose로 `T_cam_object` → 로봇 base 좌표 → IK → 프레임 충돌 검사 → 관절 경유점 표 →
(사람이 승인하며) 최저속으로 한 점씩 이동. 시뮬레이션 코드는 `~/pan/6D_pose_archive/calibration_2026-10-01/`로 옮겼다(삭제 아님).

```
T_base_object = T_base_cam @ T_cam_object            # T_a_b: b 좌표를 a로. 단위 m, 카메라는 OpenCV 축(x 오른쪽, y 아래, z 앞)
```

## 구성

| 구분 | 파일 | 역할 |
|---|---|---|
| 인식 | `../real_time_project/main.py` | D455 → YOLOE-seg → FoundationPose, 카메라 기준 4x4 pose를 UDP 5005로 송신 (`../pose_sender_kimm.py`) |
| hand-eye | `auto_handeye_capture.py`, `run_real_handeye_auto.sh` | flange ArUco를 보며 `T_base_cam` 수집·계산 (사람이 로봇을 움직이고, pose는 ROS2로 **읽기만**) |
| | `calib_utils.py` | 변환·ArUco 검출·Park-Martin+최소제곱 해·leave-one-out |
| | `check_marker.py`, `verify_handeye_marker.py`, `analyze_handeye_samples.py`, `real_d455_preview.py` | 마커 확인, 마커로 변환 검증, 샘플 진단, 카메라 미리보기 |
| ROS2 | `ros2_flange_udp.py` | `get_current_tool_flange_posx`/`get_current_posj` 읽어 UDP 5006으로 송신 (읽기 전용) |
| | `ros2_dryrun_publisher.py` | 계획 결과를 `/sixd/dryrun/*` 토픽으로 발행 (제어 토픽 아님) |
| | `ros2_pendant_mover.py` | **감독형 실기 이동**: 경유점마다 사람이 `go` 입력, 기본 2 deg/s (하드 한도 5) |
| 계획 | `real_pose_dryrun.py` | pose 수신 → base 변환 → IK → 충돌 검사 → 경유점 표 (`data/real_dryrun/`). 로봇에 아무것도 안 보냄 |
| | `predict_base_pose.py` | FoundationPose 예측 base 좌표를 직접 맞춘 flange 위치와 비교 |
| | `ik_reach_check.py` | 큐브 위치 ±범위에서 IK/관절한계/충돌 도달성 지도 |
| 모델 | `kinematics.py` | M1013 URDF 순/역기구학, top-down 파지 자세 |
| | `workcell.py` | 테이블·프레임 박스, 1 cm voxel 거리장, RRT+경유점 단순화 |
| | `trajectory.py` | 부드러운 경로·속도/가속 제한 시간 배분 |
| 시험 | `test_core.py` | 오프라인 12개 (`python -m unittest test_core`) |

데이터: `data/real_handeye/<세션>/`(샘플·`T_base_cam.json`·`T_base_cam_zfix.json`), `data/real_cell_measured.json`(프레임 실측),
`data/real_d455_intrinsics.json`, `data/m1013_collision_points.npz`(충돌 메시 샘플 캐시), `data/real_dryrun/`(최근 성공 계획).

## 사용 순서

0. 로봇 연결(별도 터미널, 연결만 함):
   ```bash
   sudo ip addr add 192.168.127.5/24 dev enp5s0      # 재부팅 시 1회 (컨트롤러 192.168.127.100)
   source /opt/ros/jazzy/setup.bash && source ~/pan/doosan_ws/install/setup.bash
   ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py mode:=real host:=192.168.127.100 port:=12345 model:=m1013
   ```
1. hand-eye (카메라를 고정한 뒤): `./calibration/run_real_handeye_auto.sh --marker_size 0.10` — 자세 25개, 결과 `data/real_handeye/<시간>/`.
   카메라가 움직이면 이전 결과는 무효이므로 반드시 재캘리브레이션.
2. 인식 (conda, 640x480; 1280x720은 FoundationPose OOM): `python real_time_project/main.py --serial 338122300585 --width 640 --height 480`
3. 계획 (conda): `python calibration/real_pose_dryrun.py --once` → `data/real_dryrun/<시간>/plan_001.{json,csv}`
   (`--cell_json`으로 프레임 값 덮어쓰기. 기본 `T_base_cam`은 최신 세션의 `T_base_cam_zfix.json`)
4. 이동 (시스템 python3, 로봇 제어권을 펜던트에서 이전, 비상정지 대기):
   ```bash
   python3 calibration/ros2_pendant_mover.py --plan <plan.json> --leg approach            # 검사만
   python3 calibration/ros2_pendant_mover.py --plan <plan.json> --leg approach --execute  # 점마다 go 입력
   python3 calibration/ros2_pendant_mover.py --plan <plan.json> --leg return --execute
   ```

## 안전 규칙

- 이동은 `ros2_pendant_mover.py`만, **항상 최저속**, 한 점씩 사람이 승인. 연속·자동 이동 코드는 없다.
- 계획이 `accepted`가 아니거나 30분 지났거나 한 관절 45° 초과, 관절 한계 2° 이내, 현재 관절이 출발점과 3° 이상 다르면 거부.
- 프레임 위치(기둥 4개)는 실측했지만 프로파일 굵기·테이블 가장자리는 가정. 큐브 위치 오차는 현재 1~2 cm 수준(실측 대조 필요).
- `calibration/data/real_handeye/*`의 샘플은 보존. 카메라 이동·마커 교체 후에는 새 세션을 만든다.
