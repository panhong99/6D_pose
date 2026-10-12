# 접시 잡기 → 식기 바구니 넣기 (D455 + FoundationPose + Doosan M1013)

테이블 위 접시를 FoundationPose로 찾아 잡고, 마커로 위치를 잰 식기 바구니의 슬롯에 넣습니다.

```text
고정 D455 → YOLOE + FoundationPose → 접시 위치 → 잡기 경로 라이브러리(IK 접근 + BC)
          → 마커 4개 → 바구니 위치 → 베스트 티칭 경로를 바구니·잡은 자세에 맞춰 IK 보정
          → 운반 → 넣기(기울여 천천히) → 놓기 → 홈        (손목 D455: 바구니 위·넣기 직전 사진)
```

현재(2026-10-12): 접시 위치 무관 잡기, 바구니 위치를 바꿔도 넣기 성공, `--auto`로 사람 입력 없이 연속 동작.
속도는 항상 느리게: 운반 6°/s, 바구니 위→넣기 직전 1.5°/s, 핀 안 0.25°/s.

## 1. 환경 (처음 한 번)

| 구성 | 값 |
|---|---|
| OS / ROS | Ubuntu 24.04, ROS 2 Jazzy, Doosan `~/pan/doosan_ws` |
| conda | `foundationpose` (인식, 계획, 카메라 도구) |
| 로봇 | M1013, 컨트롤러 `192.168.127.100`, PC `192.168.127.5/24` (`enp5s0`) |
| 카메라 | 고정 D455 `338122300585`, 손목 D455 `338122303684` |
| 그리퍼 | Robotiq 2F-85, PC USB-RS485 직접 제어(두산과 통신 안 함), 힘 64/255 |

```bash
conda env create -f environment.yml && conda activate foundationpose && bash build_all.sh
bash ~/pan/doosan_ws/setup_ros2_jazzy.sh                  # ROS 2 + Doosan (conda 밖)
sudo ip addr add 192.168.127.5/24 dev enp5s0              # 재부팅 후 주소가 없을 때만
```

## 2. 실행 (`~/pan/6D_pose`에서)

```bash
# 터미널 1: 로봇 연결 (계속 켜 둠, 두산 RViz는 안 뜸)
bash calibration/run.sh connect

# 터미널 2 (선택): 경로 미리보기 RViz — 잡기부터 바구니까지 전체 궤적, 바구니 모델
bash calibration/run.sh viz

# 터미널 3: 접시 → 바구니 (펜던트에서 PC로 제어권 이전 후)
bash calibration/run.sh pipeline --task plate --plate_policy ik_bc --execute          # 정지점에서 확인
bash calibration/run.sh pipeline --task plate --plate_policy ik_bc --auto --execute   # 연속 자동
```

| 옵션 | 뜻 |
|---|---|
| `--execute` | 실제 이동. 빼면 계획·검사만 |
| `--auto` | 시작 문구 1번 뒤 사람 입력 없음. 그리퍼 자동(잡기 직전 FP로 접시 이동 확인, 물체 감지 확인) |
| `--place basket` | (기본) 마커로 바구니 위치 측정 후 넣기 |
| `--place golden` / `lower` | 마커 없이 고정 위치(`basket_shift_m`) / 제자리에 내려놓기 |
| `--stop_at_above` | 바구니 위에서 멈추고 끝 |

`--auto` 없이 실행하면: 바구니 위 `d`/`h`, 넣기 직전 `d`/`u`, 그리퍼 `c`/`o`/`s`를 묻습니다.
세부 값(깊이, 기울기, 높이, 속도, 보정)은 모두 `calibration/config.yaml`의 `plate_ik_bc`에 있습니다.

**바구니를 옮길 때**: 몸체를 잡고 옮기고 마커 4개가 고정 카메라에 보이게 둡니다. 멀리(+x)는 거의 못 가고 로봇 쪽으로 옮기는 게 안전합니다(넣기 자세는 그리퍼를 세워야 해서 팔이 덜 닿음). 닿지 않으면 움직이기 전에 거부합니다.
**마커를 다시 붙였을 때**: 바구니를 알려진 자리에 두고 `conda run -n foundationpose python calibration/camera_tools.py basket --define_current`.

## 3. 도구

| 명령 | 용도 |
|---|---|
| `bash calibration/run.sh wrist_look` | 지금 자세의 손목 사진 + 모델 핀(초록), 로봇 안 움직임 |
| `bash calibration/run.sh mover --plan P --leg L [--execute]` | 계획의 한 구간만 실행(수동 이어 가기) |
| `python calibration/camera_tools.py basket` | 마커로 바구니 위치·정합 오차 확인 |
| `/usr/bin/python3 calibration/gripper.py open\|close\|goto --pos N\|status` | 그리퍼 직접 조작 |
| `bash calibration/run.sh tests` | 하드웨어 없는 회귀 테스트 |

결과: `calibration/data/real_pipeline/<시각>_plate_*/` — `robot_state.jsonl`(관절·FP·외력·단계 이벤트), `wrist_above.png`/`wrist_pre.png`, `preview_plan.json`, 비정상 종료 시 `faulthandler.log`.

## 4. 자주 막히는 경우

| 증상 | 확인 |
|---|---|
| `basket pose unreliable … residual` | 마커가 움직였거나 가려짐(8 mm 초과). 접시·팔이 가리는지, 마커 고정 확인 |
| `unreachable … 로봇 쪽(-x)으로 옮기세요` | 바구니가 팔이 닿는 범위 밖. 로봇 쪽으로 당기기 |
| `Access control deny` | 제어권이 펜던트에 있음 → PC로 이전 |
| D455 `Device or resource busy` | 이전 실행의 인식 창이 남아 있음 → 닫기 |
| 펜던트 조그가 계속 눌린 것처럼 움직임 | 조그 중엔 `run.sh viz`를 끄고, `connect`를 다시 시작하지 않기 |
| `No module named 'rclpy'` | ROS 명령은 conda 밖(시스템 Python)에서 |

구현 파일은 [calibration/README.md](calibration/README.md), 진행 기록은 `collab/LOG.md`, 결정은 `collab/DECISIONS.md`.
