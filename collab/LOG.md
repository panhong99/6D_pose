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

## 2026-10-01 (Codex: ROS2 read-only pose bridge 진단)
- Human 실행에서 `DSR_ROBOT2.py:104 NameError: SetSingularityHandlingForce` 확인. `dsr_common2` 코드가 존재하지 않는 선택 `dsr_msgs2` service를 import 중 생성하는 패키지 버전 불일치였다.
- source와 현재 install의 `DSR_ROBOT2.py` 3개 복사본에서 해당 client를 조건부 생성하도록 수정했다. `/usr/bin/python3 -m py_compile`로 bridge와 package 구문 검증 완료. 실제 robot/ROS2 service 호출 및 motion command는 실행하지 않았다.
- 작성자: Codex

## 2026-10-01 (Codex: 실기 수동 hand-eye 31개 샘플 결과)
- Human이 `handeye_samples.json`에 #0–#30(31개)을 수집하고 solve를 실행했다. 출력 `T_base_cam.json`의 내부 일관성 residual은 position mean **184.08 mm**, max **584.46 mm**, rotation mean **22.30 deg**, max **108.18 deg**였다. (앞선 로그 기입 148.08 mm는 screenshot을 잘못 읽은 오기.)
- 판정: 이 결과는 robot-base 변환 또는 manipulation에 사용 불가. 입력 샘플은 보존하며, 재수집 전 marker physical edge 길이/입력 `--marker_size`, pendant의 BASE frame·현재 pose 항목, 각 C 캡처 시 완전 정지 및 marker plate의 flange 고정 상태를 확인해야 한다. 실제 robot/ROS2 motion command는 실행하지 않았다.
- Human이 PC와 controller 사이 IP socket이 공유된다고 알려줌. 기존 통신이 pose telemetry도 지원하는지 API/protocol을 아직 확인하지 않았다.
- 작성자: Codex

## 2026-10-01 (Codex: 31개 샘플 forensic 재검사)
- 원본 `calibration/data/handeye_samples.json`은 변경하지 않았고 SHA256이 기존 기록과 동일함. 저장된 실패 transform으로 per-sample `evaluate()`를 재계산해 JSON 결과와 일치 확인: position mean/median/max **184.08/123.55/584.46 mm**, rotation **22.30/11.28/108.18°**.
- 핵심 구조적 단서: 고정된 flange-marker와 camera-base transform이라면 각 샘플 쌍의 상대 회전각은 좌표계 conjugation 불변으로 일치해야 한다. #0–13의 91쌍에서 차이 median **0.98°**, mean **1.72°**, 92.3%가 5° 미만. #14–30의 136쌍은 median **20.80°**, mean **28.19°**, 15.4%만 5° 미만. 데이터가 대략 #14 이후에 크게 불일치한다. 이는 변환 규약/기록 시점/마커 고정 중 하나가 바뀌었거나 그 구간 입력이 잘못됐을 가능성을 보여주지만 원인은 미확정.
- 예시: #20→21은 5초 차이로 기록됐고 로봇 위치가 336.9 mm 변했지만 카메라 관측 마커는 0.51 mm / 0.14°만 변했다. #11→12도 robot position 113.2 mm 변화에 camera marker 0.33 mm 변화. 이 sample pair들의 회전각 차이는 각각 2.38°/7.03°.
- 가장 큰 per-sample residual: #27 **584.5 mm / 55.3°**, #23 **502.5 mm / 37.9°**, #17 **479.9 mm / 55.0°**, #15 **392.5 mm / 108.2°**. 저장된 transform 하에서 #0–13은 상대적으로 작은 편이나 이를 subset calibration으로 검증한 것은 아님.
- 비교 진단 자료 `calibration/data/handeye_forensics/report_20261001.json`도 확인. 대체 Euler convention ZYX median pair-angle mismatch 30.81°로 ZYZ 9.64°보다 악화. 현재 ZYZ로 저장된 행렬은 posx 재변환 결과와 정확히 일치하지만 pendant의 실제 field/frame/TCP semantics를 증명하진 않는다. Planar ArUco alternate-pose flips도 큰 residual을 보편적으로 해결하지 못함.
- 해석/다음 단계: 전체 31개에서 손실함수만 바꾸거나 무작정 outlier 제거해 사용할 행렬을 만들지 않는다. 우선 #0–13과 #14 이후 사이에 marker mount/pose 입력/coordinate mode 변화가 있었는지 확인하고, 실제 pendant의 선택한 pose 항목(BASE vs USER/TOOL, flange vs TCP)을 대조한다. 기록된 `posx`의 현재 convention만으로 실물 정의를 확정할 수 없다.
- 작성자: Codex

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

## 2026-09-29
- 한 일 (Builder: Claude): 폴더명 변경(`FoundationPose`→`6D_pose`) 후 YOLOE/DINO 데모 실행 환경 복구.
  - `real_time_v1_kimm.py`의 `FoundationPose.real_time_project.` import 접두사 제거(YOLO 스크립트 방식).
  - `pose_tracker_kimm.py`의 Cutie 경로를 `/home/pan/pan/6D_pose/FoundationPose-plus-plus/src`로 수정. README 경로 `panhong/pan/FoundationPose`→`pan/6D_pose`.
  - `mycpp/build/*.so`를 이전 폴더에서 복사(`mycpp.cluster_poses` AttributeError 해결). `transformers<5` 설치.
  - `textured_57mm.obj`를 `textured.obj`에서 0.978배 균일 축소해 재생성(원본 Codex 파일과 동일한지는 미확인).
- 결과(metric, log 경로): 정지 큐브 진단(scratchpad `diag.py`): YOLOE conf 0.26, mask 2215px, depth 0.587m, 60프레임 회전변화 평균 0.7°/최대 1.9°, score 82~88, Cutie+Kalman enabled. 움직임/회전 상황은 미재현.
  - 5/2 vs 10/5 iteration 비교는 mask 유효 depth 부족으로 실패(큐브가 화면 밖).
- 다음 할 일: (1) 큐브를 정면 0.4~0.6m에 두고 iteration 5/2 vs 10/5, `--roi_patience 0`, `--conf`/`--imgsz` 비교. (2) 복구 relock 비활성화 옵션 추가 여부 결정. (3) README에 plus-plus/Cutie 설치, 폴더명(`6D_pose`) 정정. (4) ROS2(Jazzy) 없음 → bridge 확인 불가. (5) 미커밋 변경 다수(README, pose_tracker, real_time_v1_kimm).
- 작성자: Claude

## 2026-09-30
- 한 일 (Builder: Claude): (1) yoloe-v8l-seg.pt 다운로드(루트, 107MB), 로드/추론 확인(latency 321ms는 첫 호출 포함). (2) Install.md 대조: FP 가중치, Cutie(-e 설치, 가중치 2개), Cutie() 로드 OK. 환경은 RTX 5070 / torch 2.11+cu128 / numpy 2.4 / opencv 5.0로 원본 docker(cu121)와 차이. (3) 코드 정리: pose_tracker_kimm.py의 Cutie 절대경로를 __file__ 기준으로, 중복 kf_mean 대입 제거. real_time_v1_yolo.py에 --max_depth, --auto_recovery 추가, 기본 checkpoint 경로를 cwd 비의존으로 변경. (4) CAD 확인: textured.obj 실제 큐브 약 58mm(회전으로 AABB만 75mm), textured_57mm.obj는 0.978배 균일축소본, 원점은 큐브 중심에서 ~3cm 벗어남. README가 말한 textured_57mm_transform.json은 없음.
- 결과: py_compile OK. 성능 비교(11s vs v8l, iteration, mesh)는 아직 미실행 - 카메라/큐브 필요. 로그 경로 없음.
- 다음 할 일: (1) 정지 큐브에서 tracking 프레임별 시간 profiling(Cutie/track_one/_score). (2) 11s vs v8l, 5/2 vs 10/5, mesh 원본 vs 축소 비교. (3) mesh를 큐브 중심 원점으로 재정렬(로봇 연동 전). 주의: YOLOE text encoder(mobileclip_blt.ts)는 실행 cwd에서 찾으므로 repo 루트에서 실행할 것.
- 작성자: Claude
- (2026-09-30 추가) YOLOE 스크립트 자동 재탐지 기본 활성화 및 detect_bbox 추가. py_compile/--help/tests.test_yoloe(11) OK. 실물 카메라 실행은 미검증. 화면 관찰: 큐브 약 0.38m(추정)에서 박스가 4배 크게 틀어짐 -> D455 최소 거리 미만 depth 무효 의심(미검증).
- (2026-09-30 추가) 합성 프레임 profiling(RTX 5070, 640x480, mask 80x80): register iter5 1840ms / iter10 2966ms. tracking 1프레임: track_one iter2 ~20ms, iter5 ~49ms, _score ~10ms, Cutie ~17ms, 전체 _track iter2 48ms / iter5 80ms. (합성 입력이라 절대값은 참고용, 카메라/detector 시간 제외.) 원본 기본값 복원 후 iter10/5 조건 tracking은 이론상 ~12FPS 수준으로 예상되어 속도 최적화가 다음 과제.
- (2026-09-30 추가) 속도 조정 적용 및 합성 프레임 실측(RTX 5070, 640x480): _track 77ms(10/5,score1) / 61ms(10/3,score1) / 51ms(10/3,score3). 카메라/detector/화면 제외 값. 다음: 실카메라로 FPS와 안정성 확인, bbox 어긋남 판정(검사 횟수 기준) 수정 여부 결정.
- (2026-09-30 추가) 파일 정리: main.py(YOLOE), real_time_utils.py 신설. 실행: `python real_time_project/main.py --mesh_file ... --yoloe_checkpoint yoloe-v8l-seg.pt`. py_compile/--help/import OK, 카메라 실행 미검증. main.py에 30프레임마다 [TIMING] 로그(read/process/draw) 추가.
- (2026-09-30 추가) real_time_project = main.py / real_time_utils.py / real_time_v1_dino.py / detector_comparison/ 4개로 정리. DINO README 삭제(내용은 루트 README에 요약).
- (2026-09-30 추가) real_time_project = main.py / real_time_utils.py / detector_comparison/. 실행: `python real_time_project/main.py --detector yoloe --yoloe_checkpoint yoloe-v8l-seg.pt --prompt "toy cube"`.
- (2026-09-30 추가) 최종 구성: real_time_project/{main.py, real_time_utils.py, detectors.py}. 실행: `python real_time_project/main.py --detector yoloe --prompt "toy cube"` (기본 checkpoint 루트 yoloe-v8l-seg.pt).

## 2026-09-30 (calibration)
- 한 일 (Builder: Claude): ROS2 Jazzy(Ubuntu 24.04 네이티브, 도커 불필요 - 두산 jazzy 브랜치 확인) 설치 스크립트 ~/pan/doosan_ws/setup_ros2_jazzy.sh 작성, doosan-robot2(jazzy) 클론(~/pan/doosan_ws/src). calibration/{calib_utils.py, check_marker.py, capture_handeye.py, ros2_flange_udp.py} 작성.
- 결과: calib_utils 자체테스트 통과(posx 왕복, 합성 마커 검출 1.5mm/0.11도(100mm 마커), 합성 hand-eye 풀이 오차 0.0~0.5mm). D455 1280x720@31fps 프레임 읽기 OK, 마커는 시야에 없어 검출 0/30(정상). UDP 수신 시험 OK. 로봇 연결과 ROS2 설치는 Human이 진행 중.
- 다음 할 일: (1) check_marker.py로 마커 검출/jitter 확인 (2) ROS2 설치 후 ros2_flange_udp.py로 flange pose 스트림, 펜던트 값과 ZYZ 규약 대조 (3) capture_handeye.py로 15~25자세 수집 후 풀이, residual 확인 (4) 검증 후 T_base_cam으로 FoundationPose 결과를 base 좌표로 변환.
- 작성자: Claude
- (2026-09-30 추가) 자동 수집: ros2_run_waypoints.py 신설, ros2_flange_udp.py에 posj 추가, capture_handeye.py에 W(자세 저장)/--auto 추가. 러너-캡처 UDP 프로토콜은 목 테스트 통과(hello/arrived 재시도/skip). 이동시간 예: 30도 @3deg/s = 11초. 실제 로봇 미검증.
- (2026-09-30 추가) ROS2 Jazzy 설치와 doosan-robot2 빌드 완료(5개 패키지: dsr_bringup2/msgs2/hardware2/controller2/description2, DRCF_VER=2). D455 프레임 확인: flange 마커가 검은 판 위에 흰 여백 없이 붙어 있고 크기도 작아(안쪽 무늬 ~20-30px) DICT 전체에서 검출 0. 조치: 흰 종이 여백(5-10mm 이상)과 더 큰 마커 필요. 네트워크: PC는 enp5s0(10.114.1.55) 하나뿐, 로봇 IP(192.168.137.100) ping 실패 -> 로봇 IP/연결 방식 확인 필요. 컨트롤러 버전은 미확인(펜던트 시스템 정보에서 확인).

## 2026-09-30 (calibration, Codex: simulation feasibility)
- 한 일: 기존 5005 UDP 송신 및 과거 Isaac Sim 스크립트 이력을 점검했다.
- 결과(metric, log 경로): `pose_sender_kimm.py`는 `T_cam_obj` 4x4를 JSON UDP 5005으로 송신한다. 수신하여 `T_base_obj`와 안전 flange 목표를 산출하는 노드는 아직 없다. 과거 Isaac Sim GT coordinate check와 M1013 ROS2 bridge는 Git 이력에 있으나 현재 작업 트리에서는 삭제되어 있으며, `/home/pan` 아래 Isaac Sim `python.sh`/설치 디렉터리는 발견되지 않았다. 실제 로봇 연결/이동 명령 실행 0회; 새 실험 로그 없음.
- 다음 할 일: simulation-only UDP receiver/transform/GT test를 만들고 수치 오차 0에 가까운지 검증한 뒤, Human이 Isaac Sim 설치 위치 또는 설치 의사를 정하면 M1013 visual bridge를 복원·현행 경로에 맞춰 별도 검증한다.
- 작성자: Codex

## 2026-09-30 (calibration, Codex: UDP transform simulation)
- 한 일: `calibration/sim_udp_pose_pipeline.py`를 추가했다. 고정 카메라/marker/flange/cube의 결정론적 GT를 만들어 hand-eye 풀이, 실제 `pose_sender_kimm.PoseSender` JSON UDP packet 송수신, `T_base_cube = T_base_cam @ T_cam_cube`와 base Z +100 mm 접근점 생성을 한 프로세스에서 검증한다. 실제 ROS2·카메라·Isaac·로봇 명령은 사용하지 않는다.
- 결과(metric, log 경로): `conda run -n foundationpose python calibration/sim_udp_pose_pipeline.py` 통과. `T_base_cam` 위치 오차 `6.28e-13 mm`, 회전 `0 deg`; UDP 이후 cube 위치 오차 `4.83e-13 mm`, 회전 `0 deg`; hand-eye residual 평균 `4.75e-13 mm`/`6.48e-07 deg`. 접근 위치 `[0.460, 0.120, 0.185] m`. 결과 JSON: `calibration/data/sim_udp_pose_pipeline.json` (gitignore 대상).
- 다음 할 일: Isaac Sim 5.1.0 설치 완료 후 동일 frame chain으로 M1013 URDF, 고정 D455, cube, flange marker를 시각화하고, GT·계산 cube·접근점을 scene에서 비교한다. flange orientation/IK/실제 motion은 미검증 상태로 유지한다.
- 작성자: Codex

## 2026-09-30 (Isaac Sim installation, Codex)
- 한 일: Isaac Sim 5.1.0 workstation binary 설치를 시작했다. Ubuntu 24.04.5, RTX 5070, NVIDIA driver 580.178.04, RAM 31 GiB, root free 535 GiB를 확인했고, 공식 5.1 요구사항(24.04 지원, Linux driver 580.65.06)을 대조했다.
- 결과(metric, log 경로): GPU/OS/driver/storage는 충족. VRAM 12 GiB는 공식 16 GiB 권장보다 작으므로 M1013+D455+cube의 소규모 저해상도 scene부터 검증한다. 공식 ZIP 8,768,419,777 bytes를 `/home/pan/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64.zip`에 재개 가능하게 다운로드 중이며, 관측 처리량 약 0.3–0.4 MiB/s, 예상 약 6–7 h. 실제 로봇 연결/이동 명령 0회.
- 다음 할 일: 다운로드 완료 후 `/home/pan/isaacsim-5.1.0`에 압축 해제, `post_install.sh`, compatibility checker로 GUI/headless Kit 기동을 확인한 뒤 M1013+D455+cube scene을 구현한다.
- 작성자: Codex
- (2026-09-30 추가) Human 요청으로 Isaac Sim 설치를 중단하고, 부분 installer ZIP 69MB, 빈 `/home/pan/isaacsim-5.1.0`, 임시 `/tmp/isaac-download-tools` 7.1MB를 휴지통으로 이동했다. 활성 다운로드 프로세스와 기존 Isaac/Omniverse cache는 없었다. 휴지통에 있어 복구 가능하며, 실제 로봇 연결/이동 명령은 0회.
- (2026-09-30 추가) Human 요청으로 다운로드 병목을 시험했다. 단일 wget 약 0.3–0.4 MiB/s 대신, 임시 aria2 8개 Range connection에서 평균 4.6 MiB/s(최대 약 4.9 MiB/s)를 확인해 예상 약 28–30분으로 단축했다. 30초 시험 뒤 중단했으며 `/home/pan/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64.zip`의 약 191 MiB partial과 `.aria2` control file은 재개용으로 남겼다. 실제 로봇 연결/이동 명령은 0회.
- (2026-09-30 추가) Human과 sim-first 계획 합의: Isaac Sim에서 hand-eye `T_base_cam`/GT calibration error와 target motion reachability를 먼저 검증하고, 같은 변환 코드·UDP 형식을 실기로 재현한다. 실기 motion은 Human 명시 승인 전 dry-run만 허용.
- (2026-09-30 추가) Isaac Sim 5.1.0 standalone 설치 완료: 공식 ZIP MD5 `139fce058c4994021cdea0772c29ecd3`, `unzip -tq` PASS, `/home/pan/isaacsim-5.1.0`에 압축 해제 및 `post_install.sh` 완료. Compatibility Checker는 RTX 5070/driver 580.178.04/Ubuntu 24.04.5를 인식했고 IOMMU 및 CPU powersave 경고만 표시. GUI 기동 확인. `calibration/isaacsim_scene_m1013_d455_cube.py` 추가(visual-only ground/table, URDF M1013 `fix_base=True`, rigid-body 없는 fixed D455, static 60 mm cube; ArUco/IK/motion 없음), `python3 -m py_compile` PASS. 현재 Human GUI가 열려 있어 standalone runtime 실행은 보류.
- (2026-09-30 추가) Isaac GUI 종료 시 native segmentation fault 관찰(Kit shutdown thread). 최초 standalone headless crash도 IOMMU popup 대기 상태로 남은 Compatibility Checker GPU process와 동시 실행된 상태였다. stale checker를 종료한 후, conda 변수를 제거한 단일 headless 실행이 exit 0으로 통과했다. 장면 USD `calibration/data/m1013_d455_cube_scene.usda` 47MB 생성, `m1013` 6 joints/link_6, `/World/D455_Mount`, `/World/RubiksCube`, physics scene 존재 확인. URDF importer의 in-memory-stage/link_6 fabric warning은 있었으나 생성 결과 정상. 실제 로봇 연결/이동 0회.
- (2026-09-30 추가) Human 화면의 viewport label이 `Perspective`임을 확인해 D455 view가 아님을 판별했다. `calibration/isaacsim_scene_m1013_d455_cube.py`의 D455 pose를 computed look-at quaternion으로 변경했다(local `-Z` -> cube centre). 독립 numpy 검증: quaternion `(w,x,y,z)=(0.8947935859, 0.4109815868, 0.0728193589, 0.1585431011)`, optical-axis alignment `0.000000 deg`; `python3 -m py_compile` 및 `git diff --check` PASS. 현재 열려 있는 Isaac instance는 이전 script이므로, 종료 후 script를 재실행해 GUI에서 `/World/D455_Mount/color` camera viewport를 확인해야 한다. 실제 robot 연결/이동 0회.
- (2026-09-30 추가) 수정 scene 재실행은 `M1013 imported at: /m1013` 직후 종료/크래시했다. 재현 headless test(`--headless --steps 3`)도 native SIGSEGV; log `/home/pan/isaacsim-5.1.0/kit/logs/Kit/Isaac-Sim Python/5.1/kit_20260930_174253.log`, dump `.../kit/data/Kit/Isaac-Sim Python/5.1/7c02ae1d-41ae-48f8-ce2c4fbf-6d8c72cb.dmp`. Backtrace는 `libnvidia-glvkspirv.so.580.178.04`/`libcarb.graphics-vulkan.plugin.so` shader/Vulkan thread이며 Python traceback 없음. 따라서 현 시점에는 scene Python logic보다 Isaac Sim 5.1.0 + RTX 5070/driver Vulkan runtime 불안정으로 분류(원인 미확정). 실제 robot 연결/이동 0회.

## 2026-09-30 (Codex: standalone eye-to-hand 전체 연결 및 GUI 실행)
- 구현: calibration/isaacsim_scene_m1013_d455_cube.py를 visual-only에서 자동 전체 실험으로 확장. sim_eye_to_hand.py, run_sim_eye_to_hand.sh, test_sim_eye_to_hand.py, README_sim_eye_to_hand.md 추가. 기존 실기 hand-eye 코드 및 사용자 다른 변경은 유지.
- 실측 최종 결과: headless=False, RTX RGB 1920×1080, 18/18 ArUco 관측. camera extrinsic error 3.6782018 mm / 0.1169900°, cube base pose error 3.8912507 mm / 0.1867830°, IK/USD flange approach error 2.006e-13 mm / 0°. PASS. 마지막 도달 오차는 운동학 재생의 수치 정확도이며 실기/PhysX 정확도 의미 없음.
- 결과/로그: calibration/data/eye_to_hand/{result.json,status.json,T_base_cam.txt,run.log,calibration_*.png,cube.png,cube_depth.npy,final.png}, scene calibration/data/m1013_d455_cube_scene.usda. status.json passed (2026-09-30 18:03:10 KST).
- 검증: 실제 GUI 프로세스 유지, desktop screenshot으로 PASS 결과 창/버튼/로봇 씬 표시 확인. independent numerical unittest 3개 PASS(FK/IK reachable+unreachable, held-out hand-eye/좌표 곱 순서, tilted RGB-D plane exact recovery+invalid depth). py_compile, bash -n PASS. git diff --check는 기존 다른 작업의 DECISIONS.md 60/99행 trailing whitespace만 보고했으며 기존 내용은 수정하지 않음.
- 다음 확인: Human 복귀 후 GUI의 Replay calibration poses + cube approach로 재생하고 D455 image 버튼으로 관측 시점을 확인. metric 정의/통과 임계값은 DECISIONS.md 검토 요청. 실기 적용과 충돌/동역학/파지 검증은 별도 범위.
- 작성자: Codex

## 2026-09-30 (Codex: replay 오류 수정 및 calibration 재사용 cube approach)
- 원인: GUI replay 버튼 후 `RuntimeError: Accessed invalid expired 'Xform' prim </m1013/base_link>` 발생. calibration 결과 저장 직후 `context.save_as_stage()`가 in-memory URDF의 cached prim handle을 만료시켰고, replay callback이 이 handle을 사용했다. robot/행렬 오차가 아니다.
- 수정: scene USD 저장을 GUI/replay 종료 이후로 이동. `--object-approach` 모드를 추가해 flange ArUco와 calibration pose 수집 없이 기존 `data/eye_to_hand/result.json`의 `T_base_cam`을 읽고 cube 측정·base 변환·IK 접근 재생만 수행. 별도 output: `data/eye_to_hand/object_approach_result.json`.
- 검증: `./calibration/run_sim_eye_to_hand.sh --headless --exit-after-run --object-approach` exit 0. camera GT comparison 3.575 mm/0.120°, cube 3.834 mm/0.260°, reach 3.20e-13 mm/0°, PASS. GUI replay save-order 수정은 standalone event loop 구조를 코드 검토로 확인했고, 다음 GUI 실행에서 버튼으로 확인 가능.
- 작성자: Codex

## 2026-09-30 (Codex: GUI/terminal coordinate comparison 출력)
- 구현: object-approach 및 full calibration 결과에 `comparisons`를 추가했다. 각 항목은 estimated/GT 4x4, xyz[m], `estimated - GT` 축별 delta[mm], translation norm[mm], rotation[deg]를 저장한다. GUI에도 cube base 좌표의 estimated/GT/delta를 표시하고 terminal에는 base_camera, base_cube, flange_target 비교와 두 cube 4x4를 출력한다.
- 검증: `./calibration/run_sim_eye_to_hand.sh --headless --exit-after-run --object-approach` PASS. base cube estimate `[0.420721, 0.096391, 0.031054] m`, GT `[0.420000, 0.100000, 0.030000] m`, estimate-GT `[+0.721, -3.609, +1.054] mm`, norm `3.828 mm`, rotation `0.197 deg`. 로그는 terminal stdout, 상세 JSON은 `calibration/data/eye_to_hand/object_approach_result.json`.
- 작성자: Codex

## 2026-09-30 (Claude: YOLOE + FoundationPose GUI 전체 실행 및 frame 점검)
- 한 일: `sim_eye_to_hand.py`에 `cube_rotations`/`cube_symmetry_error`와 FP 모드 `foundationpose_frame_check`(terminal·GUI·JSON) 추가, `position_approach_passed` 추가. GUI에 경로 문구(`D455 RGB-D -> YOLOE-seg mask -> FoundationPose T_cam_object`, `T_base_object = saved T_base_cam @ T_cam_object (no ArUco)`), T_cam_object/T_base_object xyz, FP-only/hand-eye 분리, 회전 대칭 분리 표시. `foundationpose_sim_pose.py`가 `mesh_bbox_center_m`/`mesh_extents_m` 저장. `test_sim_eye_to_hand.py`에 대칭 분리 테스트 추가(4/4 PASS), py_compile/bash -n PASS.
- 실행: `DISPLAY=:1 ./calibration/run_yoloe_foundationpose_sim.sh` 3단계 모두 완료(headless capture → conda foundationpose YOLOE prompt `cube` conf .01 imgsz 640 + FP max-side 640 → Isaac GUI approach). GUI 창 열린 상태로 유지, 스크린샷으로 패널 문구/수치 확인. log `calibration/data/eye_to_hand/logs/yoloe_foundationpose_gui_20260930_185720.log`, 스크린샷 `.../logs/gui_foundationpose_approach_20260930_1858.png`, 결과 `calibration/data/eye_to_hand/foundationpose_approach_result.json`.
- 결과: base cube estimate `[0.422113, 0.097195, 0.026493] m`, GT `[0.42, 0.10, 0.03]`, delta `[+2.113, -2.805, -3.507] mm`, norm 4.963 mm. 분해: hand-eye-only `[+1.10, -3.60, +0.75] mm`(3.83 mm), FP-only camera frame 4.45 mm(ray 방향 +3.08 mm 더 멀리, 횡 3.21 mm). 회전 raw 119.048° = cube 대칭 120°(body diagonal `[-1,1,1]/√3`) + residual 2.375°. 이전 run(118.880°)도 동일 대칭 120° + residual 2.696°. flange reach 3e-13 mm. `position_approach_passed=True`, `passed=False`(residual 2.38° > 2°).
- frame 결론: mesh-origin offset 0 mm(CAD bbox centre 원점) → 118~119°는 origin/frame 불일치가 아니라 단색 cube의 120° 대칭 선택. 90°/180° 대칭이 아니라 120° 대각 대칭. 57 vs 60 mm 크기 불일치는 centre를 카메라 쪽으로 ~1.5 mm 당기는 방향이지만 관측은 +3.1 mm 더 멀리 → 크기로는 설명되지 않음(640 px 다운스케일로 cube ~40 px, depth nearest 리사이즈 추정, 미확정).
- 확인: `result.json`, `T_base_cam.txt` sha256 실행 전후 동일(덮어쓰지 않음). 실기 robot/ROS2 명령 0회.
- 미검증: GUI "Replay cube approach" 버튼 클릭(xdotool 없음) — Human이 열린 창에서 클릭 확인 필요. VS Code 터미널의 `expired prim </m1013/base_link>` traceback은 줄번호(main:202/tick:267/place:151)가 현 코드(207/404/178)와 달라 수정 전 실행분으로 판단.
- 남은 리스크: (1) grasp orientation 미검증 — FP 회전은 24-fold 모호, flange에 적용 금지 유지. 실제 Rubik's cube 색면 텍스처로 Isaac cube를 바꾸면 raw 회전 검증 가능. (2) residual 2.4°/FP 위치 4.5 mm는 640 px 입력 한계일 가능성 — ROI crop 후 원해상도 FP로 분리 필요. (3) CAD 57 mm vs scene 60 mm. (4) 단일 프레임·단일 cube pose 결과, 다른 위치/yaw 미검증. (5) 충돌/PhysX/파지 미검증.
- 다음 할 일: Replay 버튼 확인, Isaac cube를 60→57 mm 또는 Rubik 텍스처로 맞춰 재측정, cube yaw/위치 여러 개로 반복 통계.
- 작성자: Claude

## 2026-09-30 (Claude: Rubik CAD 렌더 + cube 1.65 m + all-zero 시작 자세 실행)
- 한 일: 위 DECISIONS 항목 구현. `isaacsim_scene_m1013_d455_cube.py`(add_rubiks_cube, CUBE_MESH, D455 고정 상수, cube 위치 계산, `--start-joints-deg`), `sim_eye_to_hand.py`(args 기반 cube_gt, q0=start joints, q_nominal IK seed, frame_check 크기/문구 갱신, report `start_joints_deg`). py_compile, unittest 4/4 PASS.
- 1차 시도 실패(기록): cube만 이동 + 기존 IK 대기자세 → flange가 cube 시야 가림, YOLOE conf 0.025로 flange 검출, FP-only 810 mm / 회전 90°. log `calibration/data/eye_to_hand/logs/yoloe_foundationpose_gui_realcube_20260930_192309.log`. 2차(카메라 후퇴안)는 T_base_cam 무효 문제로 실행 전 중단(exit 143은 의도적 종료).
- 최종 실행: `DISPLAY=:1 ./calibration/run_yoloe_foundationpose_sim.sh` 3단계 exit 정상, GUI 유지. log `calibration/data/eye_to_hand/logs/yoloe_foundationpose_gui_realcube_far_home0_20260930_193003.log`, screenshot `.../logs/gui_realcube_far_home0.png`, 결과 `foundationpose_approach_result.json`.
- 결과: YOLOE `cube` conf 0.054, bbox ~28×35 px(640 스케일), mask 794 px. base cube estimate `[0.213541, 0.657873, 0.028551]` vs GT `[0.214524, 0.661635, 0.0285]`, delta `[-0.983, -3.761, +0.051] mm`, norm **3.888 mm**, 회전 **0.765°(raw, 대칭 0°)**. 분해: FP-only camera frame `[-1.217, -0.492, +0.362] mm`(**1.36 mm**, ray +0.40 / 횡 1.30), hand-eye-only `[+0.354, -3.872, -0.182] mm`(3.89 mm). flange reach 1e-12 mm. `passed=True`, `position_approach_passed=True`, start_joints `[0]*6`.
- 해석: 텍스처 cube에서 FP가 올바른 방향(대칭 없음)을 찾았고 FP 오차 4.45 → 1.36 mm, 회전 residual 2.4° → 0.77°. 거리 +0.5 m에도 개선 → 이전 오차의 주원인은 단색 cube/60 vs 57 mm 불일치로 판단. 이제 전체 오차 3.9 mm의 대부분(3.87 mm, 주로 base −Y)은 hand-eye 고정 편향.
- 확인: `result.json`, `T_base_cam.txt` sha256 불변. 실기 robot/ROS2 명령 0회.
- 남은 리스크/다음: (1) 남은 오차 대부분이 hand-eye 편향 → 캘리브레이션 자세 수/마커 품질 개선이 다음 레버(단 result.json 덮어쓰기는 Human 승인 필요, 별도 파일로 저장 옵션 추가 권장). (2) 단일 프레임·단일 cube pose. (3) cube가 table prop(y≤0.325) 밖 바닥 위 — 시각적 문제일 뿐 기능 무관. (4) FP 회전을 flange에 적용하는 옵션은 여전히 미구현. (5) YOLOE conf 0.054로 낮음 — 실기 환경 오검출 위험, prompt/threshold 재검토 필요.
- 작성자: Claude

## 2026-09-30 (Claude: side 카메라 배치 재캘리브레이션 + FP 접근, 30°/s)
- side calibration: `./calibration/run_sim_eye_to_hand.sh --setup side --headless --exit-after-run`, 33회 시도 중 18개 채택(reproj 0.02–0.27 px). 카메라 GT 대비 **1.936 mm / 0.025°**(v1 3.575 mm / 0.120°), held-out `[0.28, 0.39, 0.16, 0.19, 0.21] mm` / `[0.09, 0.30, 0.16, 0.38, 0.13]°`, cube ArUco 1.99 mm / 0.024°. PASS. 결과 `calibration/data/eye_to_hand_side/result.json`, log `calibration/data/eye_to_hand/logs/calib_side_*.log`.
- side FP 접근(GUI): YOLOE `cube` conf 0.11, mask 3760 px, cube–D455 0.605 m. base cube estimate `[0.649837, 0.011290, 0.028956]` vs GT `[0.647959, 0.011686, 0.0285]`, delta `[+1.878, -0.397, +0.456] mm`, **1.972 mm / 0.316°**(대칭 0°). 분해: FP-only camera 0.27 mm(ray +0.07, 횡 0.26), hand-eye-only `[+1.97, -0.15, +0.41] mm`. flange reach 1e-11 mm, 30°/s, 시작 all-zero. PASS. log `calibration/data/eye_to_hand_side/yoloe_foundationpose_gui_20260930_194132.log`, screenshot `.../gui_side_setup.png`, 결과 `.../foundationpose_approach_result.json`.
- 확인: v1 `result.json`/`T_base_cam.txt` sha256 불변. py_compile, bash -n, unittest 4/4. 실기 robot/ROS2 명령 0회.
- 미검증: GUI Replay 버튼(자동 클릭 도구 없음). 30°/s 체감 속도는 Human 확인 필요.
- 다음 할 일: 캘리브레이션 개선안(DECISIONS 참조) 중 선택, top-down 접근 옵션 여부, 여러 cube 위치 반복 통계.
- 작성자: Claude

## 2026-09-30 (Claude: grasp-ready top-down 접근)
- 구현: `sim_eye_to_hand.py` `cube_yaw_deg`, `grasp_frames`, `TOP_DOWN`, grasp report/terminal/GUI. scene `--approach-orientation`, `--tool-length-m`, `--cube-yaw-deg`. unittest 5/5(신규: yaw 0/30/−20°, FP가 다른 면을 up으로 잡아도 yaw 복원, TCP 위치, 도구 수직, j4≈0).
- 검증 1 (headless, cube yaw 30°): yaw 측정 +29.681° (오차 −0.319°), tilt 0.116°, flange yaw −150.3°, q_pregrasp `[-2.0, 24, 110, 0.03, 46, -32]°`, q_grasp(IK만) `[-2.0, 38, 110, 0.05, 29, -32]°`, cube 1.796 mm / 0.341°, PASS. log `calibration/data/eye_to_hand_side/grasp_yaw30_headless_*.log`, 결과 `foundationpose_approach_result_yaw30.json`.
- 검증 2 (GUI, Human 배치 yaw 0°): yaw 측정 −0.253°, flange yaw 179.75°, q_pregrasp `[-2.0, 24, 110, 0.03, 46, -1.8]°`, cube 1.976 mm / 0.318°, reach 0 mm, PASS. log `calibration/data/eye_to_hand_side/grasp_gui_*.log`, screenshot `gui_grasp_ready.png`.
- 확인: v1 result.json/T_base_cam.txt sha256 불변. 실기 robot/ROS2 0회.
- 다음: 그리퍼 TCP 길이 입력 후 재실행, 필요 시 pregrasp→grasp 하강 재생 옵션(현재 미구현, 파지/충돌 미검증).
- 작성자: Claude

## 2026-09-30 (Claude: pregrasp 100 mm)
- GUI 실행 PASS: flange z 0.129 m(목표=실측), q_pregrasp `[-2.0, 30, 110, 0.04, 38, -1.8]°`, cube 1.968 mm / 0.325°, yaw 오차 −0.258°. log `calibration/data/eye_to_hand_side/grasp_gui_pregrasp100_*.log`, screenshot `gui_pregrasp100.png`. unittest 5/5, v1 파일 sha256 불변, 실기 명령 0회.
- 작성자: Claude

## 2026-09-30 (Claude: 세션 마무리 / 내일 시작점)
- 현재 상태: 기본 setup = `side`(Human 배치). 캘리브레이션 `data/eye_to_hand_side/result.json`(카메라 1.94 mm/0.025°), FP 접근 PASS(cube 1.97 mm/0.32°), top-down grasp-ready(j4≈0, j6=cube yaw), pregrasp 100 mm, 관절 30°/s, 시작 자세 all-zero. v1 `data/eye_to_hand/result.json` 불변. 실기 명령 0회.
- 내일 논의 주제(Human 요청: sim 환경 구성 공부·논의 후 real로 가져갈 수 있게): (1) 시뮬 노이즈 주입(D455 depth 노이즈, 내부 파라미터·인코더 오차)으로 실제 예상 오차 추정 및 파지 허용 오차 역산, (2) 핸드 정보(flange→파지 중심 길이, 닫힘 축, 최대 개폭)로 `--tool-length-m` 설정, (3) 하강·파지·후퇴 동작 및 충돌 검사(현재 pregrasp 정지, 충돌 미검증), (4) 실기 이전 항목: 두산 TCP 등록, posx ZYZ 규약 펜던트 대조, 실제 D455 depth/내부 파라미터, 캘리브레이션 자세는 테이블 위 15 cm 이상 권장, ChArUco·자세 25–40개, 독립 기준 검증, (5) YOLOE conf 낮음(0.05–0.11) 대응.
- 정리 사항: Isaac GUI 창이 열려 있을 수 있음(닫으면 scene USD가 저장되며 표시된 traceback 없음). 토큰·VRAM 12 GB이므로 GUI 여러 개는 닫고 시작 권장.
- 작성자: Claude

## 2026-10-01 (Claude: 실기 셀 sim 구성, voxel 충돌, 2-카메라 캘리브레이션)
- 실기 확인: D455(serial 338122300585, USB 3.2, fw 5.17.3.10) 미리보기 GUI `calibration/real_d455_preview.py`(s 저장, q 종료; 카메라 읽기 전용) 추가, Human이 촬영 → `calibration/data/real_env/`. 실기 depth 유효율 57–75%, 광학 테이블 반사로 테이블면 depth 결손. depth 점군으로 프레임 위치 자동 측정 시도는 벽 점이 지배해 신뢰 불가 → Human 실측값 사용.
- 구현: `workcell.py`(형상, CollisionChecker, RRT, export), scene `frame_L/frame_R`, `add_d455`, 실기 광학, pipeline 충돌 연동/리플레이(실제 재생 경로)/GUI·터미널 clearance 출력. unittest 6/6(신규: 레이아웃 0.75 m·45°, 충돌 판정, RRT 경로 clearance, voxel export).
- 결과 frame_L 캘리브레이션: 18자세·좁은 범위 9.641 mm/0.274°(백업 `result_18poses_narrow.json`) → 30자세·넓은 범위 **7.968 mm/0.064°**, 카메라 좌표 delta `[0.33, -0.55, -7.94] mm`(광축 편향), held-out ≤3.92 mm/0.58°. log `calibration/data/eye_to_hand_frame_L/calib30_*.log`.
- 결과 frame_R 캘리브레이션: **6.116 mm/0.063°**, delta `[1.17, 0.54, -5.98] mm`, 30/30, 충돌 거부 1회. log `calibration/data/eye_to_hand_frame_R/calib30_*.log`.
- 결과 frame_L FP 접근(GUI): YOLOE conf 0.555, mask 754 px(입력 0.5배). FP-only camera 1.85 mm, cube 회전 0.58°, yaw 오차 +0.07°. base cube **8.514 mm**(대부분 hand-eye). 접근 RRT 3 waypoints, 최소 clearance 22.4 mm(margin 20), pregrasp 60 mm, grasp 40 mm. PASS. log `calibration/data/eye_to_hand_frame_L/fp_gui_*.log`, voxels `.../workcell_voxels.npz`.
- 확인: v1 `result.json`/`T_base_cam.txt` 불변(sha256 앞자리 847d6d32…, c8b0ff63… 일치 확인). 실기 robot/ROS2 명령 0회.
- 다음: (1) 가정값 실측(프레임 깊이, 로봇 base 위치, 카메라 바 높이/면, 카메라 각도), (2) hand-eye 광축 편향 대책: 큰 마커(≥150 mm) 또는 ChArUco, RGB-D 보정 검토, (3) 두 카메라 결과 융합, (4) 실기 이전 시 voxel/박스로 경로 dry-run 검사.
- 작성자: Claude

## 2026-10-01 (Claude: 움직임/충돌/GUI 개선, cone 카메라)
- 구현: `trajectory.py`(PCHIP 경로, 곡률·가감속 제한 시간배분), `workcell.CollisionChecker.tidy/plan_margin/STATIC_LINKS`, `add_voxel_overlay`, `find_calib_centre.py`, GUI 슬라이더/스크롤/voxel 토글, 옵션 `--max-joint-accel-deg`, `--show-voxels`, `--flange-marker-m`. 기본 속도 8°/s, 가속 10°/s². unittest 8/8. 중간에 옵션 4개(`--approach-orientation/--pregrasp-height-m/--tool-radius-m/--tool-length-m`)가 편집 실수로 빠졌다가 복구됨(그 사이 실행은 기본값 사용).
- Human 배치 cone: scene 저장본 백업 `/tmp/.../scratchpad/user_scene_with_cones.usda`(임시), 좌표는 workcell.FRAME_CELL에 반영.
- 결과 frame_L(GUI, 150 mm 마커 캘리브레이션): 카메라 9.419 mm/0.135°, cube 8.176 mm/2.139°, FP-only 1.78 mm, 접근 rrt+tidy 13 waypoints 최소 clearance 63.2 mm, position approach PASS / 회전 2.14°로 full FAIL. log `calibration/data/eye_to_hand_frame_L/fp_gui_fixed_*.log`.
- 결과 frame_R 캘리브레이션: 4.581 mm/0.067°, held-out ≤3.10 mm/0.70°. log `calibration/data/eye_to_hand_frame_R/calib_marker150_*.log`.
- 환경 확인: `real_time_project/main.py --help`, `capture_handeye.py --help` 로드 OK, `sim_udp_pose_pipeline.py` PASS(UDP 포맷+변환 오차 ~1e-13).
- 다음: 실제 프레임 실측 반영, 핸드 TCP/반경 입력, hand-eye 광축 편향 대책(ChArUco/RGB-D), 실기 UDP 수신→base 변환→dry-run 경로 검사 노드.
- 작성자: Claude

## 2026-10-01 (Claude: Codex 인수인계)
- Human 요청으로 작업 주체를 Codex로 전환. 인수인계 문서 `collab/HANDOFF_CODEX_2026-10-01.md`(목표, 하드룰, 환경, 명령어, 코드 지도, 오늘 한 일, 현재 수치, 위험/다음 할 일) 작성. 상태: unittest 8/8, v1 result.json sha256 불변, Isaac GUI 창 1개가 열려 있을 수 있음. 실기 robot/ROS2 명령 0회.
- 작성자: Claude

## 2026-10-01 (Codex: 인수 테스트 및 Isaac 상태 확인)
- 검증: `cd calibration && conda run -n foundationpose python -m unittest test_sim_eye_to_hand` 8/8 PASS (7.970 s).
- Isaac 상태: `frame_L` setup의 `--foundationpose-approach` GUI process가 열려 있음(PID 27958); `--output-usd /tmp/.../scratch_gui2.usda`를 사용 중이다. GUI를 닫으면 live scene 저장이 수행되므로 닫지 않았다.
- 해석: frame_L/frame_R 수치는 D455 내부 left/right stereo intrinsic calibration error가 아니라, 고정 D455 두 배치 각각의 robot-base 대비 eye-to-hand extrinsic `T_base_cam` error이다. 다음 작업은 실기 프레임/robot-base 실측을 받아 FRAME_CELL 가정을 교체하는 것.
- 작성자: Codex

## 2026-10-01 (Claude: 실기 hand-eye 31샘플 원인 분석 + 자동 캡처 도구)
- 분석(오프라인, read-only): `calibration/analyze_handeye_samples.py` → `calibration/data/handeye_forensics/report_20261001.{json,log}`. 원본 `handeye_samples.json` sha256 `3fa7e471…` 불변.
- 결론: **원인은 카메라 영상과 수기 입력 robot pose의 짝(인덱스) 어긋남**. 근거: (1) #11/#12, #20/#21은 카메라 marker pose가 0.3/0.5 mm로 같은데 입력 pose는 113/337 mm 다름(강체 marker로 불가능). (2) #0–#13(#6, #12 제외 12개)은 한 해로 일관: fit 2.6 mm/1.86°. (3) #14부터 camera #i ↔ robot #(i+1)(#17–#20은 +2)로 다시 짝지으면 각각 0.9–16 mm 일치. #16, #30 영상에 맞는 pose 없음, robot #12/#14/#17/#18은 어느 영상에도 안 맞음.
- 배제된 가설: Euler 규약(ZYZ가 최적, 두산 `DRFS.h`에도 ZYZ 명시), TCP/user frame 오프셋(상수 변환이므로 T_flange_marker/T_base_cam에 흡수되어 큰 residual을 만들 수 없음), IPPE 평면 marker 뒤집힘(뒤집은 해도 맞지 않음).
- 재짝지은 27쌍 탐색 해(사용 금지, 검증 안 됨): fit 4.1 mm/1.92°, leave-one-out 평균 4.8 mm(중앙 3.4, 최대 21.5), marker scale 추정 0.977(→ marker 실제 변 길이 재측정 필요). T_base_cam 위치 ≈ [1.498, 0.338, 0.108] m, 카메라 광축 ≈ base −x 방향 수평 — Human이 실제 카메라 위치와 대조 필요.
- 구현: `calibration/auto_handeye_capture.py`(사람이 로봇을 움직이고, 스크립트는 ROS2 read-only pose를 받아 정지 1.5 s + 새 자세 + marker 기울기/jitter + 짝 검사 통과 시 자동 캡처, 목표 개수에서 자동 solve + leave-one-out), `calibration/run_real_handeye_auto.sh`(시스템 python으로 `ros2_flange_udp.py` 실행 후 캡처 시작; bringup은 Human이 별도 실행). 출력은 `data/real_handeye/<time>/`만 사용하고 보호 파일 쓰기 거부. `calib_utils.py`: Park–Martin 회전 SO(3) 투영(불일치 데이터에서 det −1 → crash 수정), `pair_angle_mismatch_deg`, `leave_one_out` 추가. 기존 `capture_handeye.py`는 수정하지 않음.
- 검증: py_compile, `bash -n`, `calib_utils.py` self-test PASS, unittest 8/8, UDP PoseStream 정지 판정 테스트 OK. 기존 31샘플을 새 짝 검사에 순서대로 재생 → 짝 어긋난 샘플 전부 거부, 채택 12개 fit 4.2 mm, LOO 평균 5.1 mm(log `data/handeye_forensics/guard_replay_20261001.log`). 카메라·로봇 실기 실행은 안 함. 실기 robot/ROS2 명령 0회, bringup 실행 안 함.
- 다음(Human): marker 검은 사각 변 길이 실측, bringup 실행 후 `./calibration/run_real_handeye_auto.sh --marker_size <실측 m>`로 재수집(25자세), 결과 LOO와 독립 측정(카메라 위치 줄자 등) 비교.
- 작성자: Claude

## 2026-10-01 (Claude: 자동 캡처 알림 방식 변경)
- `auto_handeye_capture.py`: 캡처 알림을 터미널 벨(`\a`, 스피커 없으면 무음)에서 화면 표시로 변경 — 카메라 창에 2초간 초록 테두리 + "CAPTURED #n" 배너, 터미널/`capture.log`에도 로그. py_compile만 확인, 카메라 실행은 안 함. 실기 robot/ROS2 명령 0회.
- 작성자: Claude

## 2026-10-01 (Claude: 실기 컨트롤러 연결 + pose 스트림 수정)
- 연결: PC는 사내망(10.114.1.55), 컨트롤러는 192.168.127.100이라 대역이 달라 ping 100% 손실. 같은 스위치에 있으므로 `sudo ip addr add 192.168.127.5/24 dev enp5s0`(임시, 재부팅 시 소멸; Human이 실행)로 보조 주소 추가 → ping 0.5 ms 0% 손실. bringup 재실행 후 `dsr_controller2 configured and activated`, 실제 로봇 값 수신 확인(posj ≈ 0, flange posx [0.02, 34.5, 1452.5, 180, 0, 180] — 영점 자세로 일치). 이 빌드의 서비스 이름은 `/dsr01/dsr_controller2/aux_control/...`.
- 수정: `ros2_flange_udp.py`를 `DSR_ROBOT2` 래퍼(첫 호출에서 응답 대기로 멈춤) 대신 `GetCurrentToolFlangePosx`/`GetCurrentPosj` 서비스 직접 호출로 재작성(read-only, 20 Hz 검증). `run_real_handeye_auto.sh`의 서비스 확인을 `dsr_controller2` 경로로 수정(`CTRL_NS`).
- Human 지시 반영: 로봇 속도는 항상 가장 느리게. sim 기본 관절 속도 8 → **3°/s**, 가속 10 → **3°/s²**(`isaacsim_scene_m1013_d455_cube.py`, `sim_eye_to_hand.py` 기본값; 옵션으로 명시하면 변경 가능, 기본은 느림 유지). 실기 이동 명령은 여전히 0회(로봇은 사람이 움직임; 본 작업은 pose 읽기만).
- 다음: D455 미리보기 창 종료 후 `./calibration/run_real_handeye_auto.sh --marker_size <실측 m>` 실행, 25자세 수집, leave-one-out 확인.
- 작성자: Claude

## 2026-10-01 (Claude: 실기 hand-eye 재수집 결과)
- 세션 `calibration/data/real_handeye/20261001_160439/`(자동 캡처, ROS2 read-only pose, marker 100 mm 실측 10 cm). 25샘플 solve: fit pos 5.70 mm(max 17.1)/rot 1.89°, leave-one-out pos 6.31 mm(중앙 4.79, max 18.1)/rot 1.98°. 26샘플: fit 5.8 mm, **marker scale 0.980 자유 보정 시 2.7 mm(max 7.3)**. 이전 수기 입력(184 mm)과 달리 pose 짝 문제는 해소됨(pair 검사 통과).
- 관찰: 잔차가 거리에 비례(0.5–0.6 m: 2.5–6 mm, 1.1–1.25 m: 8–17 mm) → marker 크기/내부 파라미터 스케일 편향(sim과 같은 광축 편향 패턴). T_base_cam 위치 ≈ [1.496, 0.388, 0.097] m, 광축 ≈ [−0.98, −0.11, 0.17].
- 미검증: 독립 측정(카메라 실제 위치 줄자 등) 없음, 큰 marker/보정 비교 필요. `T_base_cam.json`은 사용 승인 전. 실기 이동 명령 0회.
- 작성자: Claude

## 2026-10-01 (Claude: real_cam 시뮬 시도)
- real_cam 캘리브레이션: 9.183 mm/0.245°(30자세, 100 mm 여유 불가→margin 0.05로 실행). log `/tmp/.../realcam_calib.log`, 결과 `calibration/data/eye_to_hand_real_cam/result.json`.
- real_cam FP 접근: **무효**. 큐브가 카메라 시야 밖/기둥에 가려져 FoundationPose 오검출(큐브 오차 1077 mm). 복귀·펜던트 경유점 코드는 동작(경유점 CSV 생성)하나 목표가 틀려 수치(최소 clearance 500 mm)는 의미 없음. 큐브 위치/프레임 위치 재설정 후 재실행 필요.
- 작성자: Claude

## 2026-10-01 (Claude: real_cam 시뮬 접근+복귀 성공)
- 배치: 단일 D455를 실기 `T_base_cam` 위치에(프레임 바깥 모서리, 낮게, −x 방향), 카메라 바로 옆 기둥 `upright_11`은 시야를 가려 시뮬에서 제외(`omit_boxes`), 큐브 (1.05, 0.10). 큐브 y=0.30/0.20에서는 접근 목표가 기둥 `upright_01`(0.77, 0.355)과 각각 10/64 mm로 margin 100 mm 위반 → y=0.10으로 이동. 접근 높이 100 mm(flange z 0.128 m).
- 결과(실행 `realcam_fp5.log`, 결과 `calibration/data/eye_to_hand_real_cam/foundationpose_approach_result.json`): YOLOE conf 0.55, 큐브 base 추정 `[1.0567, 0.1032, 0.0258]` vs GT `[1.05, 0.10, 0.0285]`, delta `[+6.7, +3.2, −2.7] mm`, **7.92 mm / 0.795°**(대부분 hand-eye: 6.5/3.0/−3.4 mm, FP-only 0.78 mm). 접근 최소 clearance **127.3 mm**, 복귀 **128.1 mm**(margin 100 mm), 직선 경유점 기준 접근 120.4 mm / 복귀 125.3 mm. 전체 PASS. 속도 3°/s. 경유점 표 `pendant_waypoints.csv`(접근 13, 복귀 9 경유점).
- 한계: 프레임/기둥 위치·카메라 옆 기둥 제외는 가정. sim 깊이는 이상적. 실기 이동 명령 0회.
- 작성자: Claude

## 2026-10-01 (Claude: 드라이런 파이프라인)
- `real_pose_dryrun.py`, `ros2_dryrun_publisher.py`, `data/real_cell_template.json` 추가. 합성 pose 시험 통과(위 DECISIONS 참조). 실기 이동 명령 0회, 로봇/카메라 미연결 시험.
- 작성자: Claude

## 2026-10-01 (Claude: 실기 D455 + FoundationPose 드라이런 첫 실행)
- 실제 D455(640x480; 1280x720은 FoundationPose OOM)로 `main.py` → `real_pose_dryrun.py` 동작. 결과 `calibration/data/real_dryrun/20261001_172931/plan_001.{json,csv}`: 큐브 base `[1.0792, 0.0238, 0.0097]` m, yaw −9.4°, tilt 1.2°, 접근 6 moves / 복귀 5 moves, pregrasp·직선 구간 최소 clearance 110 mm, 시작 자세는 관절 스트림 없어 영점. 판정 ACCEPTED(드라이런, 전송 없음).
- 의심: 큐브 중심 z 0.0097 m (예상 ≈0.0285, 약 19 mm 낮음) → hand-eye 높이 편향 또는 반사 테이블 depth. Human 대략 거리 ≈100 cm(base 판 가장자리 기준일 가능성); 줄자로 base 중심축 기준 x/y 실측 필요. 실기 이동 명령 0회.
- 작성자: Claude

## 2026-10-01 (Claude: 감독형 이동 스크립트)
- `ros2_pendant_mover.py` 추가(위 DECISIONS). 검사 모드 시험만 수행. 실제 로봇 이동 0회, 로봇 서비스 호출 0회.
- 작성자: Claude

## 2026-10-01 (Claude: 카메라 이동 발견 + IK 도달성 점검)
- 실기 비교: ROS(`get_current_tool_flange_posx`, Base/World, TCP) = 펜던트 값 `967.74 90.96 173.28 4.66 168.52 1.57` (좌표계 불일치 가설 기각). 카메라 사진에 `T_base_cam`으로 투영: 큐브 (563,567) vs 실제 (560,570) 일치(FP 기준이라 당연), flange 판 (371,408) vs 약 (545,405), base 발 (555,518) vs 약 (720,500) → 로봇 쪽 점이 ~170 px 왼쪽 = 약 15° yaw 불일치. 큐브 예측 (917,246) vs flange(큐브 바로 위, 968,91) 수평 163 mm. Human 확인: **캘리브레이션(16:04) 이후 카메라를 움직였음** → 현재 `T_base_cam` 무효, 재캘리브레이션 필요(카메라 고정 후).
- flange 마커 검증(`verify_handeye_marker.py`): 이 자세에서는 마커가 카메라에 거의 옆면이라 40프레임 중 0 검출. 마커가 카메라를 향하는 자세에서 재시도 필요.
- IK 도달성: `ik_reach_check.py` 추가(예측 큐브 ±20 cm 격자, top-down pregrasp 100/150 mm). IK+관절한계만: 81점 중 79점 가능(실패 2점은 가장자리 모서리). 가정 프레임 포함 시 대부분 'C'(프레임 기둥과 margin 100 mm 미만) — 프레임이 가정값이라 의미 제한. 결과 `calibration/data/real_dryrun/ik_reach_*.csv`.
- 상태: Human이 로봇을 default(영점)에 두고 제어권 회수 후 bringup을 끄고 식사. 실기 이동 명령 0회(이 단계에서도).
- 작성자: Claude

## 2026-10-01 (Claude: 재캘리브레이션 + 큐브-위 flange 검증)
- 재캘리브레이션 `calibration/data/real_handeye/20261001_191859/`(25샘플, 카메라 고정): fit 5.31 mm/1.31°, LOO 5.88 mm/1.38°; 이전(160439) 대비 카메라 yaw **14.1° 변화**(−173.5→−159.7°) = 사이에 카메라를 움직였기 때문(Human 확인). marker scale 자유 보정 시 0.9765, fit 3.2 mm(이전 세션도 0.98 — 반복).
- 검증(Human이 flange 중심을 큐브 중심 바로 위에 둠, flange posx `967.79 89.01 136.51 3.22 168.81 -0.0015`, ROS=펜던트 일치): 카메라→큐브 `[-72.6, 178.9, 569.3] mm`. 기본 T_base_cam 예측 큐브 `[992.3, 98.2, 6.4]` → 수평 차이 **26.2 mm**(dx +24.5, dy +9.2). marker-scale 보정 T_base_cam 예측 `[968.6, 90.4, 16.2]` → 수평 차이 **1.7 mm**(dx +0.8, dy +1.4). 보정본 저장 `T_base_cam_markerscale.json`(탐색용, 기본 아님; `--t_base_cam`로 지정). 단 1점 검증 → 다른 위치에서 추가 확인 필요. 큐브 중심 z는 보정본 16.2 mm(예상 28.5 mm, 약 12 mm 낮음 — 받침 판/테이블 높이 확인 필요).
- 이전 이전 세션 카메라 이동으로 무효였던 값에서 나온 결론(163 mm 불일치)은 해결됨. 실기 이동 명령 0회.
- 작성자: Claude

## 2026-10-01 (Claude: 두 번째 위치 검증, 테이블 두께)
- 두 번째 위치(flange `910.23 286.95 164.4 15.42 173.32 -0.003`, 눈 맞춤): 기본 T_base_cam 예측 큐브 `[920.8, 305.0, 7.3]` 수평 차 20.9 mm; marker-scale 보정본 `[897.2, 297.4, 16.6]` 수평 차 16.7 mm. 첫 위치(26.2 / 1.7 mm)와 합치면 보정본이 둘 다 더 가깝지만 확정 근거 부족, 방향이 위치마다 달라 눈 맞춤 오차(~1–2 cm) 가능. 전 오차 ≤ ~2 cm 수준(카메라 이동 전 163 mm). 결론 보류, 포인터/핀 또는 카메라 위치 줄자 실측으로 확인 필요.
- Human 실측: 테이블 두께 **10 cm**, 로봇은 받침 없이 테이블 위에 직접 설치 → base 설치면 = 테이블 상면(z=0), 큐브 중심 예상 z=28.5 mm. 예측 z(7–17 mm)가 낮은 것은 받침 때문이 아니라 카메라 z/FP depth 편향. `FRAME_CELL['table_thickness']` 0.05→0.10. unittest 8/8.
- 작성자: Claude

## 2026-10-01 (Claude: 줄자 실측으로 카메라 위치 대조)
- Human 줄자: 렌즈 높이 **11.5 cm**, base 중심~렌즈 수평거리 **약 156 cm**. 계산값: 기본 z 9.3 cm / 거리 156.4 cm, 보정본 z 10.3 cm / 거리 153.9 cm. 거리는 기본 쪽, 높이는 보정본 쪽에 가까워 둘 중 하나를 가릴 수 없음(모두 1–2 cm 이내 → 변환이 줄자와 대체로 일치).
- 단 높이는 두 변환 모두 12–22 mm 낮고, 큐브 중심 z 예측도 12–21 mm 낮음(7 vs 28.5 mm) — 서로 독립적인 두 단서가 같은 방향·크기 → T_base_cam의 카메라 높이 z 편향 가능성이 큼. 탐색용 `T_base_cam_zfix.json`(기본 해에서 z만 0.115 m로 교체) 저장; 이 변환으로 큐브 z 예측이 ≈28–29 mm(예상 28.5 mm)로 맞춰짐. 기본으로 채택하지 않음(Human 확인 요청).
- 작성자: Claude

## 2026-10-01 (Claude: calibration/ 정리)
- sim/시험/수기·자동이동 코드와 데이터를 `~/pan/6D_pose_archive/calibration_2026-10-01/`로 이동(삭제 없음, 약 360 MB → 저장소 `calibration/` 432 KB). 핵심 18개 파일 + README.md. `test_core.py` 12/12, 구문·`--help` 확인. 실기 이동 후 결과: 접근 3구간 오차 ≤0.01°, flange 목표와 0.3 mm 일치, 복귀 성공(Human 확인). push 대기.
- 작성자: Claude

## 2026-10-01 (Claude: git용 README 갱신)
- 루트 `README.md`에 7장(실기 로봇 연동 현재 진행 상황: 확인된 결과 표, 미해결/주의)과 8장(검증 명령어 1~7)을 추가, 폴더 구조에 `calibration/`·`collab/` 반영. 1~3번 명령(오프라인 시험 12개, 스크립트 `--help`, D455 인식 수)을 문서 그대로 실행해 확인. 기존 1~6장은 수정하지 않음.
- 작성자: Claude

## 2026-10-01 (Claude: 퇴근 정리)
- `collab/NEXT_2026-10-02.md` 작성(현재 상태, 변환별 오차, 결정 필요 사항, 내일 2대 카메라 작업, 재시작 명령, 규칙). 내가 켠 프로세스(main.py 등) 종료, bringup은 이미 꺼져 있음 확인.
- 작성자: Claude

## 2026-10-02 (Claude: 큐브 위치 변경 후 재실행 + 도착 비교)
- 큐브 위치를 바꾼 뒤 `main.py`(640x480) + `real_pose_dryrun.py`로 새 plan 생성(`calibration/data/real_dryrun/20261002_091915/plan_001.json`, 기본 변환 `T_base_cam_zfix`). FP 큐브 base `[1040.5, 47.6, 28.6] mm`, 접근 3구간, 2 deg/s, 점마다 `go`(감독형). 첫 시도는 ROS2 환경(`source /opt/ros/jazzy`, `doosan_ws`)을 source하지 않아 `No module named 'rclpy'`로 실패 → source 후 재실행해 이동 완료(Human).
- 도착 후 ROS(`ros2_flange_udp.py`, UDP 5006) flange `[1040.60, 47.84, 128.60] mm` vs plan 목표 `[1040.55, 47.64, 128.62]`: 차이 0.21 mm / 0.03°, 관절 차이 ≤0.006°. **이것은 이동 정확도이지 캘리브레이션/GT 오차가 아님**: 목표가 FP×T_base_cam→IK 결과라 도착값과 일치는 당연하고 T_base_cam이 틀려도 같은 값이 나옴. GT 오차는 로봇 쪽과 독립적인 기준이 있어야 관측 가능.
- 계획(미실행): GT 기준 확보 방법 3가지 — (1) 핀/포인터를 flange에 달고 큐브 중심을 직접 찍어 펜던트 좌표를 GT로 사용, (2) flange 아래 큐브 수평 어긋남을 자로 측정(오차 1–2 cm 포함), (3) `verify_handeye_marker.py`로 flange ArUco 두 경로(로봇 경로 vs 카메라 경로) 비교 — 마커가 카메라를 향하는 자세 필요, `main.py`는 끄고 D455 비우기, **640x480으로 실행**(1280x720은 FoundationPose OOM; 스크립트 기본값이 1280x720이므로 `--width 640 --height 480` 명시, 캘리브레이션 해상도와 intrinsics 일치 여부 확인 필요), `--t_base_cam`으로 기본/markerscale/zfix 비교 권장.
- 실기 이동은 Human이 승인·실행(감독형·최저속). 나는 로봇에 이동 명령 0회, 읽기 전용 UDP 수신만 수행.
- 작성자: Claude

## 2026-10-02 (Codex: 두 터미널 통합 실행과 README 재작성)
- 추가: `calibration/run_robot_connection.sh`, `run_cube_pipeline.sh`, `cube_pipeline.py`, `test_cube_pipeline.py`. 통합 실행에서 현재 관절 읽기 → YOLOE/FP 연속 인식 → 안정성/변환/IK/충돌/관절 검사 → JSON/CSV 저장 → ROS 관찰용 토픽 발행 → 선택적 기존 감독형 mover 호출. 인식 창 유지, plan 경로 자동 전달, 종료 시 소유 자식만 정리.
- 보완: publisher의 준비/처리 확인 파일과 마지막 결과 QoS, 중복 UDP 포트 거부, mover의 오프라인 결과 거부·승인 후 현재 관절/plan 재검사·실패 시 정지 요청. 루트 및 calibration README를 실제 사용 순서로 교체.
- 검증: 기존 12개+통합 7개 = 19/19 PASS(12.019 s), 승인 대기 중 자세 변경 회귀시험 추가 1/1 PASS(0.005 s). 전체 20개 모두 하드웨어 없는 시험이며 ROS/프로세스/소켓 동작은 모의. 쉘 구문·Python 구문·CLI 도움말 및 시스템 Python ROS 의존성 import 확인(노드 생성/서비스 호출 없음).
- 통합 셸 명령 `bash calibration/run_cube_pipeline.sh --offline-test` PASS. 합성 큐브 `[1.04, 0.05, 0.0285]` m, 접근/복귀 각 2 moves, 직선 최소 clearance 각각 약 120 mm(기준 100 mm). 결과 `calibration/data/pipeline_offline/20261002_102913_vekiuelr/plan_001.{json,csv}`; 실제 이동에 사용 금지 표시.
- 이번 작업에서는 카메라/GPU 인식/bringup/실제 ROS 노드를 실행하지 않았고 로봇 서비스 호출·이동 명령 0회. 실제 하드웨어 통합 실행은 Human 실행 전 미검증 상태. GT 작업은 보류, 기존 calibration 결과 보존.
- 작성자: Codex

## 2026-10-02 (Claude: wrist 카메라 hand-eye 캡처 스크립트)
- Human 제안: 큐브 중앙에 ArUco(DICT_4X4_50, id 1, 45 mm) 부착, wrist D455(시리얼 338122303684, 고정 카메라는 338122300585)로 마커를 가까이서 읽어 base 기준 GT로 사용 → FoundationPose×T_base_cam과 비교.
- 전제: wrist 카메라 `T_flange_cam`이 없음 → `calibration/wrist_handeye.py` 추가(eye-in-hand, 고정 마커, Human이 펜던트로 이동·스크립트는 읽기 전용; 로봇 pose는 ros2_flange_udp UDP 5006). 식 `T_base_flange @ T_flange_cam @ T_cam_marker = T_base_marker(상수)`, Park-Martin + 정제. 합성 데이터 자체 시험: X 위치 오차 ~1e-13 mm. 실기 캡처는 아직 안 함. 결과 `data/wrist_handeye/<시간>/{samples.json,result.json}`.
- 남은 일: 캡처 20+자세 → LOO 오차 확인 → 큐브 위 마커의 `T_cube_marker`(큐브 크기·면 중심) 입력 → FoundationPose×T_base_cam 과 비교 스크립트.
- 작성자: Claude
- (추가) `wrist_handeye.py`: 첫 세션 `data/wrist_handeye/20261002_105957/` 20샘플 전체 fit 5.5 mm(최대 22) / LOO 6.1 mm. 마커 거리 0.9–1.0 m 샘플(16,18,19)이 15–22 mm로 튐(45 mm 마커는 먼 거리에서 부정확) → 0.7 m 이하 16샘플만 오프라인 재계산 fit 2.4 mm(최대 5.9) / LOO 2.9 mm, T_base_marker ≈ [989.1, 45.1, 34.0] mm. `--max_dist`(기본 0.7)/`--min_dist`(기본 0.3) 옵션과 `--session` 이어찍기 추가. 큐브는 마커 부착 때문에 위치가 바뀐 상태(카메라는 그대로) → 이전 plan의 큐브 위치는 무효. 작성자: Claude

## 2026-10-02 (Claude: Codex 두 터미널 통합 파이프라인 검토)
- 검토 대상: `cube_pipeline.py`, `run_robot_connection.sh`, `run_cube_pipeline.sh`, `test_cube_pipeline.py`, `ros2_pendant_mover.py`/`ros2_dryrun_publisher.py` 변경. 하드웨어 없이 `test_cube_pipeline` + `test_core` 20개 통과(로봇·카메라·ROS 호출 0회).
- 수정 1건: `--serial` 기본값이 빈 문자열이었음 → D455가 2대(고정 338122300585, wrist 338122303684)라 인식이 wrist 카메라를 잡을 수 있음. 기본값을 고정 카메라 시리얼로 변경, README 문구 갱신.
- 주의(코드 이상 아님): ① 통합 파이프라인은 자체적으로 `ros2_flange_udp.py`(UDP 5006)와 publisher(5010)를 띄우고 5005/5006을 독점 bind → 기존 터미널의 `ros2_flange_udp.py`·`wrist_handeye.py`가 떠 있으면 충돌하거나 중복. ② mover 속도 한도가 5→2 deg/s로 낮아짐(의도적). ③ plan 나이 한도 30분: 접근 후 return을 30분 넘게 미루면 거부됨(안전쪽 동작).
- 작성자: Claude

## 2026-10-02 (Claude: wrist GT 검증 + 큐브 오차 측정 스크립트)
- wrist 최종 캘리브레이션: 30샘플 중 마커 거리 0.3–0.65 m 20샘플. marker_scale 1.0이면 fit 2.3 mm / LOO 2.6 mm, T_base_marker ≈ [987, 46, 37] mm. Human 확인: 큐브는 테이블 위에 직접, 마커는 **윗면**에 부착 → 마커 윗면 z는 약 57 mm여야 하는데 측정은 34–37 mm(약 20 mm 낮음).
- 오프라인 점검(`--marker_scale`로 카메라→마커 거리 배율 스캔): 배율 0.982에서 fit 0.77 mm / LOO 0.89 mm로 급감(1.0일 때 2.3 mm). 고정 카메라에서도 0.977 반복됨 → 인쇄 마커가 실제로 45 mm보다 약 1.8% 작을 가능성(≈44.2 mm) 또는 intrinsics 스케일. 실측 필요. 이 배율에서도 마커 윗면 z=44 mm로 57 mm보다 약 13 mm 낮음 → 가설: base z=0이 실제 테이블 면보다 약 13 mm 위(설치 플레이트/간격?). 어제 모든 카메라 기반 z 예측이 12–23 mm 낮았던 것과 같은 방향. **미검증**; zfix가 카메라 z 오차가 아니라 base 높이 오프셋을 보정했을 수도 있음. 확인: 테이블 면의 base z를 실측(플랜지/핀 접촉 또는 줄자).
- `wrist_handeye.py`에 `--marker_scale` 추가(solve 시 배율 적용, result.json meta에 기록). 현재 result.json은 0.982 적용본(fit 0.77 / LOO 0.89 mm). 인쇄 마커 변 길이를 캘리퍼/자로 재서 확정 필요.
- `calibration/cube_gt_compare.py` 추가: wrist로 큐브 윗면 마커 GT(큐브 중심 = 마커 원점에서 −z로 28.5 mm) vs FoundationPose×T_base_cam(변환별). 위치 오차 dx/dy/dz, 윗면 법선 각도, yaw(90° 주기), 평균·편향·산포. 합성 데이터로 계산 검증(위치·tilt·yaw·축 재배치·90° wrap). 실기 측정은 아직 안 함.
- 작성자: Claude

## 2026-10-02 (Claude: 5점 오차 측정 결과와 기본 변환 변경)
- `cube_gt_compare.py`로 5곳 측정. 수평 오차 평균: 기본/zfix 19.4 mm, markerscale 6.9 mm. 평균 편향(FP−GT) markerscale x −4.6, y −5.1, z +0.9 mm, 산포 ~1 mm. 상수 xy 오프셋 보정 후 leave-one-out 수평 오차 평균 2.1 mm(xy만, z·회전 제외). 자세: 법선 1.5–4°, yaw ±3°.
- Human 결정으로 보정본을 기본으로 채택(DECISIONS 참조). z는 GT 문제로 보류(마커 실측 길이와 테이블 면 base z 필요).
- 작성자: Claude

## 2026-10-02 (Claude: 마커 배율 원인 확정, z 오차 해석)
- Human 확인: 고정 카메라용 100 mm 출력 마커를 줄자로 재면 약 98 mm → 출력 배율 약 2% 축소(캘리브레이션이 찾은 0.977과 일치). wrist용 45 mm 마커도 같은 방식으로 출력했으므로 약 44.2 mm(0.982)로 판단(줄자로는 45 vs 44.2 구분 불가, 간접 근거). 따라서 wrist `marker_scale` 0.982 유지, 기본 변환 `T_base_cam_markerscale_xyoffset.json` 유지. 다음 출력은 인쇄 배율 100%로 하고 실측 변 길이를 `--marker_size`에 사용.
- z 해석: GT 마커 윗면 z ≈ 43 mm(배율 0.982) / 39 mm(1.0). 큐브가 테이블 위에 있으므로 테이블 면의 base z ≈ −14 mm(배율 1.0이면 −18 mm). Human 눈대중 플레이트 두께 ≈ 20 mm와 같은 크기 → base z=0이 테이블 면이 아니라 플레이트 윗면(베이스 바닥면). 어제 줄자 렌즈 높이 11.5 cm − 1.3 cm ≈ 10.2 cm가 markerscale 카메라 z(10.3 cm)와 일치, zfix(11.5 cm)는 이 오프셋을 반대로 보정한 것. FoundationPose z는 로봇 좌표계에서 이미 일관(FP−GT dz +0.9 mm). 충돌 모델의 테이블 z=0 가정은 실제보다 높아 보수적(안전한 쪽). 플레이트 두께 정밀 측정 후 `workcell.py`/`real_cell_measured.json` 테이블 높이 수정 예정.
- 작성자: Claude

## 2026-10-02 (Claude: 테이블 높이 반영)
- 플레이트 20 mm 측정 반영: `workcell.py` `base_plate_thickness`, `real_cell_measured.json`. 테이블 상면 z=−0.020, 프레임도 같은 기준. 단위시험 20개 통과(로봇·카메라·ROS 호출 0회). 자세한 내용은 DECISIONS 참조.
- 작성자: Claude

## 2026-10-02 (Claude: README 정리, ArUco 거리-오차)
- 루트 `README.md`를 환경 세팅 / 평소 실행(터미널 2개) / 캘리브레이션·오차 측정 명령 / 현재 설정·한계 / 문제 해결 중심으로 재작성, gif 표 삭제(`assets/videos/*.gif` 파일은 그대로 둠). `calibration/README.md`에 `wrist_handeye.py`, `cube_gt_compare.py`와 기본 변환 설명 갱신.
- wrist ArUco 거리-오차(30샘플, 1280x720, fx 647, 45 mm 마커, 기준 해는 0.3–0.65 m 샘플): 변 길이 ≥50 px(≤0.58 m) 평균 0.6–0.8 mm, 40–50 px 평균 1.5 mm(최대 5.9 mm는 43 px), 30–40 px(0.9 m) 13.5 mm, <30 px(1.0 m 이상) 11.6 mm. 이 표본에서 임계는 약 45 px. 단, 먼 샘플은 tilt 37°로 겹쳐 있고 표본 수가 적음.
- 작성자: Claude
