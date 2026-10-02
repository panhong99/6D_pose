# D455 + YOLOE + FoundationPose → Doosan M1013

고정 D455 한 대로 큐브의 6D pose를 구하고, 로봇 base 좌표로 변환해 큐브 위 접근점까지 감독형으로 이동합니다.
이동은 항상 가장 느린 속도(2 deg/s)이고, 경유점마다 사람이 `go`를 입력합니다.

```text
D455 → YOLOE → FoundationPose → T_base_cam → IK·충돌 검사 → 경로 → (사람 승인) → MoveJoint
```

## 1. 환경 세팅 (처음 한 번)

| 구성 | 값 |
|---|---|
| OS / ROS | Ubuntu 24.04, ROS 2 Jazzy, Doosan `doosan_ws` |
| conda 환경 | `foundationpose` (인식, 캘리브레이션 스크립트) |
| 로봇 | M1013, 컨트롤러 `192.168.127.100`, PC `192.168.127.5/24` (인터페이스 `enp5s0`) |
| 카메라 | 고정 D455 `338122300585`, wrist D455 `338122303684` |

```bash
cd /home/pan/pan/6D_pose

# 1) 인식 환경
conda env create -f environment.yml
conda activate foundationpose
bash build_all.sh
pip install -U pyrealsense2 ultralytics
#   weights/ (FoundationPose), yoloe-v8l-seg.pt, mobileclip_blt.ts 를 저장소 루트에 준비

# 2) ROS 2 Jazzy + Doosan workspace (conda를 쓰지 않는 일반 터미널)
bash ~/pan/doosan_ws/setup_ros2_jazzy.sh        # 컨트롤러 v3.x 는 DRCF_VER=3 을 앞에 붙임

# 3) 로봇 네트워크 (재부팅 후 192.168.127.5/24 가 없을 때만)
ip -4 addr show dev enp5s0
sudo ip addr add 192.168.127.5/24 dev enp5s0
```

ROS 2는 시스템 Python을 씁니다. ROS 터미널에서는 `conda deactivate` 후 `source /opt/ros/jazzy/setup.bash && source ~/pan/doosan_ws/install/setup.bash`를 합니다.
인식(`main.py`)은 `foundationpose` 환경, **640×480만** 사용합니다(1280×720은 FoundationPose OOM).

## 2. 평소 실행 (터미널 2개)

**터미널 1 — 로봇 연결 (계속 켜 둠)**

```bash
cd /home/pan/pan/6D_pose
bash calibration/run_robot_connection.sh
```

**터미널 2 — 인식 → 변환 → 계획 → ROS 발행 (이동 없음)**

```bash
cd /home/pan/pan/6D_pose
bash calibration/run_cube_pipeline.sh --serial 338122300585
```

**실제 접근 이동 (사람이 승인하며 진행)**

```bash
bash calibration/run_cube_pipeline.sh --serial 338122300585 --execute
```

`--execute`에서의 순서: 펜던트에서 PC로 제어권 이전 → `area clear, e-stop in hand` 입력 → 경유점마다 `go` → 도착 후 `return`(복귀 경로 재승인) 또는 `quit`.
이동은 큐브 중심 위 100 mm 접근점까지입니다(하강·파지는 없음). 종료는 `Ctrl+C`(터미널 1의 연결은 따로 `Ctrl+C`).
결과는 `calibration/data/real_pipeline/<시간>/`(`plan_001.json/csv`, 각 프로세스 로그)에 저장됩니다.

| 옵션 | 용도 |
|---|---|
| `--serial <값>` | 카메라 선택. 기본 고정 카메라 `338122300585` |
| `--t_base_cam <파일>` | 사용할 `T_base_cam` 지정 |
| `--cell_json <파일>` | 작업 셀. 기본 `calibration/data/real_cell_measured.json` |
| `--pregrasp_height_m 0.10`, `--tool_length_m`, `--tool_radius_m` | 접근 높이, 툴 길이·반경 |
| `--conf 0.15`, `--prompt "rubik's cube"` | YOLOE 검출 설정 |
| `--offline-test` | 하드웨어 없이 계획·검사 확인(실제 이동 불가) |

## 3. 캘리브레이션과 오차 확인

카메라를 움직이면 `T_base_cam`이 무효가 됩니다. 아래 순서로 다시 합니다. 로봇은 항상 사람이 펜던트로 움직이고, 스크립트는 읽기만 합니다.

**3-1. 고정 카메라 hand-eye** (터미널 1 연결 후, 인식 프로그램은 끔)

```bash
bash calibration/run_real_handeye_auto.sh --marker_size <검은_사각형_실측_변길이_m> --serial 338122300585
```

결과: `calibration/data/real_handeye/<시간>/T_base_cam*.json`. 자세 25개 이상, 마커가 카메라를 향하게.
**마커는 출력 배율 100%로 뽑고 실측 변 길이를 넣습니다**(배율 축소 시 거리 편향 발생: 100 mm → 약 98 mm 사례).

**3-2. wrist 카메라 hand-eye** (큐브 윗면 마커를 GT로 쓰기 위한 준비)

```bash
# 터미널 A: bringup  (bash calibration/run_robot_connection.sh)
# 터미널 B: 로봇 pose 읽기
source /opt/ros/jazzy/setup.bash && source ~/pan/doosan_ws/install/setup.bash
python3 calibration/ros2_flange_udp.py
# 터미널 C: 캡처 (conda activate foundationpose)
python calibration/wrist_handeye.py --marker_size 0.045
```

마커를 0.3~0.6 m에서 비스듬히(tilt 15° 이상) 보게 하고, flange 회전을 크게 바꾸며 20자세 이상 2초씩 정지합니다. 마커(큐브)는 고정합니다.
다시 계산: `python calibration/wrist_handeye.py --solve <samples.json> [--marker_scale 0.982]` (기본 거리 필터 0.3~0.7 m).

**3-3. 큐브 위치 오차 측정** (FoundationPose × `T_base_cam` vs wrist 마커 GT)

```bash
# 터미널 A: bringup, 터미널 B: ros2_flange_udp.py (위와 동일)
# 터미널 C: 고정 카메라 인식
python real_time_project/main.py --serial 338122300585 --width 640 --height 480
# 터미널 D: 측정
python calibration/cube_gt_compare.py
```

위치마다 wrist 창에서 `G`(GT) → 로봇을 시야 밖으로 치움 → `F`(FoundationPose) → `N`(저장). `U` 취소, `S` 요약, `Q` 종료. 결과: `calibration/data/gt_compare/<시간>/`.
`run_cube_pipeline.sh`·`wrist_handeye.py`와 동시에 켜지 않습니다(UDP 5005/5006 충돌).

## 4. 현재 설정과 알려진 한계 (2026-10-02)

- **기본 `T_base_cam`**: 최신 세션의 `T_base_cam_markerscale_xyoffset.json`(마커 배율 보정 + base 기준 x +4.56, y +5.07 mm). 없으면 `zfix`, 그것도 없으면 `T_base_cam.json`. 시작 시 사용 파일을 출력합니다. **카메라를 옮기면 재캘리브레이션과 재보정이 필요합니다.**
- **정확도**: 큐브 5곳 비교에서 수평 오차 평균 — 기본/zfix 19.4 mm, markerscale 6.9 mm, 상수 오프셋 보정 후 약 2 mm(한 점씩 빼고 확인). GT 자체 오차 2~3 mm 포함. 윗면 법선 1.5~4°, yaw ±3°(보정 안 함). 5점·한 카메라 배치 기준입니다.
- **높이(z)**: base 좌표 z=0은 베이스 바닥면이고 테이블과 사이에 20 mm 철판이 있어 **테이블 상면은 z=−0.020 m**입니다(`workcell.py`의 `base_plate_thickness`). 마커 GT의 z도 이 오프셋과 일치합니다. 하강·파지 구간은 아직 없습니다.
- 작업 셀의 프로파일 굵기·테이블 가장자리 위치는 가정값이고, 그리퍼 TCP는 장착 후 `--tool_length_m`, `--tool_radius_m`로 넣어야 합니다.
- ArUco 거리 한계: 45 mm 마커는 화면에서 변 길이 약 45 px(약 0.65 m) 이상에서 오차 1 mm대, 약 32 px 이하(약 0.9 m 이상)에서 10 mm 이상으로 커집니다(1280×720 기준).

### 캘리브레이션 오차 표 (6D pose, 2026-10-02)

고정 D455 + FoundationPose + `T_base_cam`(기본: markerscale + xy 오프셋)으로 구한 큐브 pose를, 큐브 윗면 ArUco(45 mm)를 wrist D455로 본 값(GT)과 비교했습니다.
큐브를 놓은 5곳(측정 위치 5개, 윗면 높이 약 57 mm, 고정 카메라 약 1.5 m), 오차 = FoundationPose − GT, base 좌표(mm). 회전 오차는 큐브의 90° 대칭을 제거한 뒤 GT 마커 좌표계의 rx, ry, rz(°)입니다.
GT 자체 오차(위치 약 2~3 mm, 각도 약 1°)가 포함되어 있습니다. 원본: `calibration/data/gt_compare/20261002_112500/` (`positions.json`, `error_table.md`).

측정 위치별 오차 (기본 변환). `#`은 큐브를 놓은 위치 번호(0~4)이고, `GT x / y`는 그 위치에서 GT가 알려 준 큐브 중심의 base 좌표(mm)입니다:

| # | GT x / y (mm) | dx | dy | dz | rx (°) | ry (°) | rz (°) | 수평 (mm) | 회전 합 (°) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 983 / 45 | +0.1 | +0.2 | +0.6 | +1.2 | −1.0 | −0.5 | 0.2 | 1.7 |
| 1 | 1032 / −141 | +1.2 | −1.6 | −0.4 | +2.7 | −2.9 | −0.5 | 2.0 | 4.0 |
| 2 | 1137 / 169 | +1.9 | +1.2 | +1.9 | +1.1 | −1.1 | +1.1 | 2.2 | 1.9 |
| 3 | 897 / 209 | −1.1 | +1.2 | +2.1 | +1.1 | −3.2 | +2.7 | 1.6 | 4.3 |
| 4 | 772 / 89 | −2.0 | −1.0 | +0.5 | +1.7 | −1.1 | −0.5 | 2.3 | 2.0 |
| 평균 | | +0.0 | +0.0 | +0.9 | +1.6 | −1.9 | +0.5 | 1.7 | 2.8 |
| 표준편차 | | 1.4 | 1.1 | 0.9 | 0.6 | 1.0 | 1.3 | | |
| 최대 \|값\| | | 2.0 | 1.6 | 2.1 | 2.7 | 3.2 | 2.7 | 2.3 | 4.3 |

변환별 요약 (5점 평균):

| 변환 | 수평 (mm) | dx | dy | dz | dz 산포 | 회전 합 (°) | rx / ry / rz 평균 (°) |
|---|---|---|---|---|---|---|---|
| **기본 (markerscale + xy 오프셋)** | **1.7** | +0.0 | +0.0 | +0.9 | 0.9 | 2.8 | +1.6 / −1.9 / +0.5 |
| markerscale (오프셋 없음) | 6.9 | −4.6 | −5.1 | +0.9 | 0.9 | 2.8 | +1.6 / −1.9 / +0.5 |
| zfix (이전 기본) | 19.4 | +19.1 | +2.6 | +14.0 | 1.1 | 2.9 | +1.5 / −2.0 / +0.5 |
| 원래 `T_base_cam` | 19.4 | +19.1 | +2.6 | −8.9 | 1.1 | 2.9 | +1.5 / −2.0 / +0.5 |

표 읽는 법:
- **오차 = (FoundationPose 결과 × `T_base_cam`) − GT.** FoundationPose는 큐브 pose를 카메라 기준으로 주고(`T_cam_cube`), `T_base_cam`을 곱하면 로봇 base 기준이 됩니다(`T_base_cube = T_base_cam × T_cam_cube`). 이 값을 wrist 마커로 구한 GT(base 기준)와 비교한 차이입니다.
- **부호:** `dx +`는 FoundationPose가 GT보다 x(로봇에서 물체 쪽)로 더 멀게 측정했다는 뜻이고, `−`는 더 가깝게입니다. y는 왼쪽이 +, z는 위가 +입니다. `rx, ry, rz`는 GT 마커 좌표계 기준 회전 오차(°)입니다.
- **수평 (mm)** = √(dx² + dy²), **회전 합 (°)** = rx, ry, rz를 합친 전체 회전 각도 오차입니다(부호 없는 크기).
- **평균 / 표준편차 / 최대 \|값\|** 행은 5개 측정 위치의 오차를 모아 계산한 값입니다. 평균은 한쪽으로 치우친 편향, 표준편차는 위치에 따라 흔들리는 정도입니다.
- **변환별 요약**은 `T_base_cam` 후보를 바꿔 가며, **같은 5곳의 FoundationPose 결과에 각각 곱해서** 오차를 비교한 표입니다. 행마다 FoundationPose 결과는 같고 곱한 행렬만 다릅니다. 각 열은 5곳 평균이고, `dz 산포`는 5곳에서 dz의 표준편차입니다.

| 행 이름 | 의미 |
|---|---|
| 기본 (markerscale + xy 오프셋) | 지금 쓰는 변환. 아래 markerscale 행의 변환에 base 좌표 x +4.56 mm, y +5.07 mm를 더한 것 |
| markerscale (오프셋 없음) | 캘리브레이션 마커의 출력 배율 축소(약 2%)를 보정해서 다시 푼 변환 |
| zfix (이전 기본) | 어제까지 기본이던 변환. 원래 변환에서 카메라 높이(z)만 줄자로 잰 값(11.5 cm)으로 바꾼 것 |
| 원래 `T_base_cam` | 어제 hand-eye 캘리브레이션 결과를 아무 보정 없이 그대로 쓴 것 |

회전 오차가 네 행에서 거의 같은 이유는 네 변환이 평행이동만 다르고 회전 성분은 거의 같기 때문입니다.

해석 시 주의:
- 수평 1.7 mm는 오프셋을 구한 같은 5점에서의 값(in-sample)입니다. 한 점씩 빼고 확인한 값은 **평균 2.1 mm, 최대 2.8 mm**입니다.
- z는 GT의 마커 배율(0.982)과 테이블 높이(−20 mm)에 의존해서, FoundationPose와 GT 사이의 일관성으로만 읽습니다.
- 회전은 보정하지 않았습니다. 합 평균 2.8°(최대 4.3°)이고 rx +1.6°, ry −1.9°의 일정한 편향이 있습니다(FoundationPose 자세 기울기인지 `T_base_cam` 회전 오차인지는 이 데이터로 구분 못 함).
- 5점, 윗면 높이 한 가지, 고정 카메라 한 배치 기준입니다. 카메라를 옮기면 다시 측정합니다.

## 5. 자주 막히는 경우

| 증상 | 확인 |
|---|---|
| `No module named 'rclpy'` / `'yaml'` | ROS 터미널에서 `conda deactivate` 후 두 개의 `source`를 했는지 |
| `no fresh robot joints` | 터미널 1 연결, `robot_state.log` |
| `Access control deny` | 제어권이 펜던트에 있음. 이동 전 PC로 이전(자동 재시도 없음) |
| `Address already in use` | 다른 `ros2_flange_udp.py`·`wrist_handeye.py`·planner가 UDP 5005/5006/5010 사용 중 |
| D455 사용 중 | 다른 인식·미리보기·검증 프로그램 종료. 카메라 2대는 시리얼로 구분 |
| CUDA/OOM | 다른 GPU 프로그램 종료, 640×480 사용 |
| 경로 `REFUSED` | 터미널의 거리·충돌·관절 한계 사유 확인(다른 경로로 자동 이동하지 않음) |

로봇·카메라 없는 검증:

```bash
bash calibration/run_cube_pipeline.sh --offline-test
conda run --no-capture-output -n foundationpose python -m unittest discover -s calibration -p 'test_*.py'
```

구현 파일 지도는 [calibration/README.md](calibration/README.md), 작업 기록은 `collab/LOG.md`, 결정은 `collab/DECISIONS.md`에 있습니다.
