# LOG

날짜별 진행을 누적한다. 최신 항목을 아래에 추가한다. 실험 결과는 metric 수치와 log 경로를 함께 남긴다.

포맷:

```
## YYYY-MM-DD
- 한 일:
- 결과(metric, log 경로):
- 다음 할 일:
- 작성자: Claude | Codex | Human
```

## 2026-09-28
- 한 일: hand-eye calibration 방향 논의 (Builder: Claude). eye-to-hand(카메라 고정) vs eye-in-hand 구분, OpenCV `calibrateHandEye` 기반 절차, sim(Isaac Sim) 활용 방안 정리
- 결과(metric, log 경로): 실행 결과 없음 (논의 단계). calibration 정확도 기준(예: reprojection/AX=XB residual) 미정
- 다음 할 일: Human 답변 필요 - 카메라 마운트 방식, Doosan 모델, calibration target 보유 여부, 로봇 pose 취득 방법(ROS2 vs DRFL). 답변 후 (1) sim에서 synthetic 데이터로 calibration 코드 검증, (2) 실기 데이터 수집 스크립트 작성
- 작성자: Claude

## 2026-09-28 (3)
- 한 일: 현재 정상 동작하는 `m1013.usd` articulation을 ROS 2 `trajectory_msgs/JointTrajectory`로 제어하는 시뮬레이션 전용 브리지 `isaacsim_ros2_m1013_bridge.py` 추가. Doosan M1013 설정의 `joint_1`~`joint_6` 및 `/dsr_moveit_controller/joint_trajectory` topic을 사용하고, `/joint_states`를 발행하도록 구성.
- 결과(metric, log 경로): AST 문법 검증 통과. Isaac Sim/ROS 2 실제 실행 검증은 아직 하지 않음.
- 다음 할 일: Isaac Sim에서 브리지를 실행한 뒤 ROS 2 `JointTrajectory` 1회 발행으로 관절 이름·단위(rad)·topic 연결을 확인. 이후 MoveIt 또는 FoundationPose pose-to-joint 변환 노드 연결.
- 작성자: Codex

## 2026-09-28 (4)
- 한 일: Isaac Sim ROS 2 M1013 브리지의 기본 환경을 수정해 흰색 기본 바닥 대신 어두운 회색 ground box를 사용하고 DomeLight 강도를 낮춤.
- 결과(metric, log 경로): 코드 수정 완료. Isaac Sim 재실행 후 시각 확인 필요.
- 다음 할 일: 실제 실행 화면에서 대비를 확인하고, 여전히 과노출이면 색관리/robot material 또는 추가 key light를 조정.
- 작성자: Codex

## 2026-09-28 (5)
- 한 일: `calibration/isaacsim_gt_cube_coordinate_check.py` 추가. Isaac Sim에 고정 head camera와 cube를 배치하고 `T_base_camera`, `T_camera_cube`, `T_base_cube_gt`, `T_base_cube_est` 및 위치 오차를 터미널과 JSON 로그로 출력하도록 구성.
- 결과(metric, log 경로): AST 문법 검증 통과. Isaac Sim 실행 결과는 아직 없음.
- 다음 할 일: Isaac Sim Python으로 스크립트를 실행해 position error가 0 mm에 가까운지 확인.
- 작성자: Codex

## 2026-09-28 (6)
- 한 일: `calibration/ros2_m1013_joint_test.py` 추가. ROS 2 `JointTrajectory`를 `/dsr_moveit_controller/joint_trajectory`로 1회 발행해 Isaac Sim M1013 브리지의 관절 제어를 시험하도록 구성.
- 결과(metric, log 경로): AST 문법 검증 통과. Isaac Sim과 ROS 2를 함께 실행한 결과는 아직 없음.
- 다음 할 일: Isaac Sim 브리지 실행 후 publisher를 실행하고 `/joint_states` 수신 및 화면상 관절 이동 확인.
- 작성자: Codex

## 2026-09-28 (7)
- 한 일: 실제 Doosan M1013 ROS 2 제어용 `calibration/real_doosan_ros2_joint_test.py` 추가. Isaac Sim 의존성을 제거하고 Doosan 공식 `DR_init`/`DSR_ROBOT2.movej` API를 사용하도록 구성. 기본은 dry-run이며 `--execute`와 명시적 6개 joint target이 있어야 실제 명령을 보냄.
- 결과(metric, log 경로): AST 문법 검증 통과. 실제 로봇 명령은 실행하지 않음.
- 다음 할 일: Doosan ROS 2 workspace를 source한 터미널에서 dry-run 실행 후, 로봇 모드·속도·목표 joint를 Human이 확인하고 실제 실행 여부 결정.
- 작성자: Codex

## 2026-09-28 (8)
- 한 일: 실제 Doosan 테스트 코드의 기본 joint velocity/acceleration을 `1.0`으로 낮추고, 명령줄에서 `1.0` 초과 값을 거부하는 보수적 안전 제한 추가.
- 결과(metric, log 경로): AST 검증 필요; 실제 로봇 명령은 실행하지 않음.
- 다음 할 일: dry-run으로 제한 동작을 확인한 뒤 Human이 실제 실행을 별도로 승인.
- 작성자: Codex

## 2026-09-28 (9)
- 한 일: 실제 Doosan 테스트 코드의 보수적 velocity/acceleration 제한을 `0.2`로 낮춤.
- 결과(metric, log 경로): 실제 로봇 명령은 실행하지 않음.
- 다음 할 일: dry-run 확인 후 Human이 실제 실행 여부 결정.
- 작성자: Codex

## 2026-09-28 (10)
- 한 일: 실제 Doosan ROS 2 실행 절차를 확인하고 `calibration/real_doosan_flange_pose_publisher.py` 추가. Doosan `get_current_tool_flange_posx()` 결과를 `/dsr01/flange_pose`의 `Float64MultiArray`로 10 Hz 발행하도록 구성.
- 결과(metric, log 경로): 실제 로봇 명령 및 pose publisher는 실행하지 않음.
- 다음 할 일: Doosan ROS 2 bringup이 실행된 상태에서 publisher를 실행하고 `/dsr01/flange_pose`를 echo하여 `[x,y,z,a,b,c]`를 확인.
- 작성자: Codex

## 2026-09-28 (11)
- 한 일: ROS RealSense topic이 없는 환경을 확인하고 `calibration/realsense_aruco_pose.py` 추가. `pyrealsense2`로 color stream과 카메라 내부 파라미터를 직접 읽고 OpenCV ArUco 검출 및 `solvePnP`로 `T_camera_marker`의 translation/rotation vector를 출력하도록 구성.
- 결과(metric, log 경로): 코드 추가 완료. 실제 카메라 실행은 아직 하지 않음.
- 다음 할 일: marker dictionary/ID/실제 변 길이를 지정해 RealSense 영상에서 marker 검출 확인.
- 작성자: Codex

## 2026-09-28 (12)
- 한 일: Doosan flange publisher가 최신 `[x,y,z,a,b,c]`를 `/tmp/doosan_flange_pose.json`에 저장하고, RealSense ArUco 검출기가 이를 읽어 `T_camera_marker`와 `T_base_flange` 후보를 한 줄에 함께 출력하도록 연결.
- 결과(metric, log 경로): 두 Python 환경의 rclpy 충돌을 피하는 프로세스 분리 구조로 코드 수정. 실제 실행 결과는 아직 없음.
- 다음 할 일: Doosan pose publisher와 RealSense ArUco detector를 동시에 실행해 한 화면에서 marker pose와 flange pose가 출력되는지 확인. 이후 다중 pose 샘플 저장 기능 추가.
- 작성자: Codex

## 2026-09-28 (2)
- 한 일 (Builder: Claude): Isaac Sim 6.0.1 standalone 스크립트로 hand-eye calibration end-to-end 검증 착수.
  `calibration/isaacsim_capture_kimm.py` 작성 — Doosan m1013을 URDF(`m1013_isaac_sim.urdf`)에서
  `urdf_usd_converter.Converter`로 임포트, 알려진 T_base_cam에 RtxCamera 배치, ChArUco 보드를
  flange(tool0)에 부착해 N개 자세에서 RGB+flange pose 캡처하는 구조. `calibration/isaacsim_verify_kimm.py`
  (OpenCV로 detect+solvePnP+handeye_core 호출, GT와 비교)도 작성 완료.
- 과정에서 실제로 겪은 버그 4개 (모두 Isaac Sim 6.0.1 API 관련, 문서화 안 된 함정):
  1. `CameraSensor(resolution=...)`는 (width,height)가 아니라 (height,width) 순서 — 해상도 반대로 나옴
  2. URDF 임포트된 로봇 관절에 PhysX drive가 없어서 `set_dof_positions()`로 텔레포트해도 중력으로 sag/fall함
     → `/World/PhysicsScene`을 직접 만들어 gravity=0으로 해결 (converter가 만드는 PhysicsScene은 defaultPrim
     밖에 있어서 AddReference로 안 딸려옴)
  3. URDF→USD 변환된 mesh 18개 중 10개가 `purpose="guide"`로 태깅되어 기본 렌더에서 안 보임 → 순회하며
     `purpose`를 `default`로 강제 리셋해서 해결
- 결과(미해결): ChArUco 보드 prim(Mesh/Plane/thin Cube 모두 시도, tool0 자식으로도 /World 최상위로도 시도,
  material 있이/없이도 시도)이 카메라 렌더(RGB+depth 둘 다)에 전혀 안 잡힘 — depth 통계가 보드 유무/위치/크기와
  무관하게 완전히 동일(byte-identical)하게 나옴. `/World` 최상위에 만든 디버그용 Cube(displayColor 지정)는
  한 번 렌더에 나타난 것으로 보였으나 material 없는 board Cube로는 재현 안 됨 — 그 디버그 결과도 완전히
  신뢰하기 어려움. 근본 원인 미파악 상태로 중단 (디버깅 시도 여러 회, 개별 원인 4개는 해결했지만 핵심 blocker
  남음 — Human 지정 종료조건 중 "디버깅 시도 3회 이상 실패 시 중단 후 보고"에 해당한다고 판단해 보고).
- 다음 할 일: 이 특정 Isaac Sim 6.0.1 렌더링 이슈(런타임에 추가한 커스텀 geometry prim이 RTX 카메라에
  안 잡히는 문제)를 usdview 등으로 독립 재현하거나, NVIDIA 포럼/릴리스노트 확인 필요. 대안으로 보드를
  URDF 자체에 링크로 포함시켜(로봇과 함께 import) 우회하는 방법도 고려 가능.
- 작성자: Claude
