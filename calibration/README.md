# 실기 파이프라인 구성

평소 사용 순서는 루트 [README.md](../README.md)의 **터미널 두 개 실행**을 참고합니다.

```bash
# 터미널 1 (연결)
bash calibration/run_robot_connection.sh

# 터미널 2 (인식·계획·결과 발행, 이동 없음)
bash calibration/run_cube_pipeline.sh --serial 338122300585

# 사람이 실제 이동을 진행할 때의 터미널 2 명령
bash calibration/run_cube_pipeline.sh --serial 338122300585 --execute
```

통합 실행은 plan 경로를 자동 전달하며 인식 창을 계속 켜둡니다. 실제 이동은 기존 mover의 시작 선언문·점마다 `go` 승인을 유지합니다. 도착 후 `return` 입력으로 복귀를 선택하고, `quit`이면 현재 위치에서 종료합니다.

## 구성

| 파일 | 역할 |
|---|---|
| `run_robot_connection.sh` | 터미널 1: ROS 환경 설정 + Doosan bringup |
| `run_cube_pipeline.sh` | 터미널 2: foundationpose 환경으로 통합 실행 |
| `cube_pipeline.py` | 자식 프로세스 관리, pose 안정성, 현재 관절 수신, 계획·검사·ROS 발행·감독형 이동 연결 |
| `../real_time_project/main.py` | D455 → YOLOE → FoundationPose, UDP 5005 |
| `real_pose_dryrun.py` | 공통 변환 선택·IK·작업 셀 충돌 계획 함수, 기존 개별 드라이런 CLI |
| `ros2_flange_udp.py` | 시스템 Python: 로봇 flange·관절 읽기 → UDP 5006 |
| `ros2_dryrun_publisher.py` | 시스템 Python: UDP 5010 → `/sixd/dryrun/*` 결과 토픽 |
| `ros2_pendant_mover.py` | 시스템 Python: 경유점 검사 및 사람의 `go` 후 MoveJoint |
| `kinematics.py` | M1013 URDF FK/IK·top-down 접근 자세 |
| `workcell.py` | 테이블·프레임·카메라 충돌 모델, RRT, 직선 경유점 단순화 |
| `trajectory.py` | 경로 보간·시간 배분 |
| `auto_handeye_capture.py`, `run_real_handeye_auto.sh` | 사람의 로봇 조작 + ROS pose 읽기로 hand-eye 수집 |
| `calib_utils.py` | 변환·ArUco·eye-to-hand solver |
| `verify_handeye_marker.py`, `predict_base_pose.py` | 변환 검증·base 예측 비교 |
| `wrist_handeye.py` | wrist D455 eye-in-hand 캘리브레이션(고정 마커, 읽기 전용). `data/wrist_handeye/` |
| `cube_gt_compare.py` | 큐브 윗면 마커(wrist GT)와 FoundationPose×`T_base_cam` 위치 오차 측정. `data/gt_compare/` |
| `check_marker.py`, `real_d455_preview.py` | 마커 확인·D455 미리보기 |
| `test_core.py`, `test_cube_pipeline.py` | 로봇·카메라 없는 회귀 시험 |

## 통합 실행의 경계

- 기본 실행은 상태 읽기·계획·관찰용 ROS 결과 발행까지만 합니다.
- `--execute`에서만 기존 감독형 mover를 실행합니다. 자동 연속 이동이나 자동 복귀는 없습니다.
- 실제 관절값이 1초 이내로 수신되지 않으면 계획을 만들지 않습니다. 영점으로 대체하지 않습니다.
- 카메라 pose의 시각·행렬·연속 안정성, 양방향 경로의 충돌 여유·관절 한계·구간 크기를 확인합니다.
- 인식/ROS 자식 프로세스가 종료되면 파이프라인을 중단합니다. 실행 중 mover가 있으면 SIGINT로 정지를 요청합니다.
- UDP 수신 포트는 한 실행이 독점합니다. 기존 bridge/planner와 중복 실행하면 실패합니다.
- 한 번 실행에 큐브 목표 하나를 고정합니다. 물체를 옮겼으면 새 실행으로 계획을 만듭니다.
- `--offline-test`는 합성 pose만 사용하며, 저장 결과는 실제 이동용으로 거부합니다.

```text
D455 → main.py → UDP 5005 → cube_pipeline.py
ROS 관절 읽기 → UDP 5006 → 현재 출발 자세
T_base_cam × T_cam_object → IK/충돌 검사 → plan_001.json
계획 → UDP 5010 → ROS /sixd/dryrun/* (관찰)
사람 승인 → ros2_pendant_mover.py → MoveJoint (실제 이동)
```

변환 파일과 캘리브레이션 샘플은 읽기만 합니다. 기본 변환은 최신 세션의 `T_base_cam_markerscale_xyoffset.json`(없으면 zfix, 그 다음 `T_base_cam.json`)입니다. 수평 오차 5점 비교는 루트 README 4장을 참고하세요.

## 결과와 진단

매 실행 결과와 프로세스별 로그는 `data/real_pipeline/<시간_고유번호>/`에 저장됩니다.
오프라인 결과는 `data/pipeline_offline/`에 저장됩니다. 프레임 기본값은 `data/real_cell_measured.json`입니다.

프로파일 굵기와 테이블 가장자리에는 가정이 남아 있습니다. 카메라 이동 후 재캘리브레이션, 툴 장착 후 TCP 길이·반경 반영이 필요합니다. 현재 통합 동작은 큐브 중심 위 접근점 정지와 선택적 복귀이며 파지는 구현하지 않았습니다.

```bash
# 하드웨어 없는 검사
bash calibration/run_cube_pipeline.sh --offline-test
conda run --no-capture-output -n foundationpose \
  python -m unittest discover -s calibration -p 'test_*.py'
```
