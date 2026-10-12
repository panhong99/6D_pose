# CONTEXT: FoundationPose (real-time 6D pose)

> repo 위치: `/home/panhong/pan/FoundationPose`
> "(초안)"은 repo를 읽고 추정한 내용이므로 Human이 확인해야 한다. 비어 있는 항목은 Human에게 질문 후 채운다.

## 연구 목표
- **Human 목표 (2026-10-08, 다음 주까지, 원문 요지):**
  - **목표 1. 임의의 위치·자세의 접시 파지.**
  - **목표 2. 잡은 접시를 식기세척기 바구니 슬롯에 넣기.**
  - 잘 넣은 상태 = 슬롯에 들어가 있고, 넣은 뒤 접시가 움직이지 않는다.
    - 비전으로 더 확실히 하려면 슬롯 철제 프레임을 인식해 그 사이로 넣는다(인식이 어려울 것).
    - 보완: F/T(힘) 측정. 슬롯에 넣은 뒤 접시를 좌우로 움직여 반력으로 갇혀 있는지 확인한다.
  - 바구니를 고정해 두고 행동만 학습하는 것은 티칭보다 나을 게 없다. **바구니의 위치·자세가 바뀌어도 정리할 수 있어야 학습의 의미가 있다.**
  - 방법 2(학습):
    - 바구니 CAD를 쓰고, 비전으로 바구니 6D pose를 구한다. 슬롯 형상은 바구니 기준으로 고정이므로 슬롯의 6D pose가 나온다.
    - 바구니 기준으로 정해진 위치·자세로 넣으면 된다. 넣는 동작은 학습한다.
    - 비전이 슬롯 위치를 학습 모델에 전달하고, 학습 모델이 로봇 모션을 제어한다.
  - 티칭(데모용):
    - 잡은 뒤 항상 티칭 경로로 바구니에 담는다. 슬롯 1..n마다 경로를 하나씩 만든다.
    - 첫 번째 잡은 접시는 1번 슬롯 경로, 두 번째 접시는 2번 슬롯 경로, … 순서로 실행한다.
- **실증 최종 목표(Human 확인, 2026-10-07):** 접시를 파지·들어 올린 뒤 세워서 dish tray/rack의 슬롯 사이에 삽입하고, 트레이가 접시를 지지하는 상태에서 그리퍼를 열어 사진처럼 거치한다. 일반 평면/상자 위 place는 최종 목표가 아니다. 목표 자세 참고: `calibration/data/tray_capture/20261007_100957_fixed_color.png`(고정 D455). 슬롯 치수·삽입 방향·그리퍼 여유·트레이 좌표는 아직 계측하지 않았다. 현재 저속 파지/들기 Stage2를 기반으로 운반→세우기/정렬→저속 삽입→지지 확인/놓기의 후속 단계가 필요하다.
- (초안) D455 RGB-D에서 FoundationPose + detector(Grounding DINO+SAM2 / YOLOE) + Cutie/Kalman tracking으로 real-time 6D pose를 추정하고, ROS2 `/foundationpose/pose`(PoseStamped, 카메라 좌표계)로 Doosan 로봇 쪽에 전달하는 파이프라인의 inference 속도 개선
- (초안) 참고 repo: teal024/FoundationPose-plus-plus

## 현재 가설
- (비어 있음) bottleneck을 profiling으로 먼저 확인하기로 한 상태 (init vs tracking, hypothesis 개수, refine iteration, render 비용, detector/tracker 비용). 확정된 가설 없음

## 성공 기준
- (비어 있음) 목표 FPS / latency / pose 정확도 threshold 필요

## 제약 조건
- **Human 최우선 조건(2026-10-07):** 2주 안에 실기 pick-and-place 실증. 성공률보다 사고 방지를 우선하며 로봇/그리퍼는 항상 가장 느린 속도로 설계·학습·평가·실행한다. 실기 최저 설정값은 장비에서 확인하며 sim 목표 속도 제한을 실제 속도 보장과 혼동하지 않는다. 시간 부족은 제한시간 연장으로 대응하고 속도를 올리지 않는다. 실제 TCP/관절 속도·가속도와 파지 후 흔들림을 검증한다.
- (초안) 환경: conda `foundationpose`, ROS2 jazzy, D455는 PC에만 연결, 로봇은 PC와 IP 랜선 연결
- (비어 있음) 사용 GPU 및 VRAM 한도

## 평가 방식
- (비어 있음) 속도: 어떤 구간을 어떻게 측정할지
- (비어 있음) 정확도: camera calibration으로 6D pose GT를 확보할 예정이라고만 알려져 있음. 지표(ADD/ADD-S 등)와 GT 절차 미정

## 건드리면 안 되는 것
- (초안, .gitignore/README 기준) `weights/`, `mobileclip_blt.ts`, `*.pt`, Hugging Face cache, debug 결과는 Git에 올리지 않음
- (비어 있음) 수정 금지 파일/디렉터리 (예: `estimater.py`, `Utils.py`, `mycpp/`, `learning/`가 upstream 원본 그대로여야 하는지)

## 현재 상태 (2026-10-07 10:00, Claude 갱신)
- 프로젝트: FP 실시간 6D pose → 실제 Doosan M1013 + Robotiq 2F-85. 접시 pick-and-place RL은 **Isaac Lab 시뮬레이션 전용**(실로봇에 RL 명령 없음, 실기 이동은 Human이 펜던트로).
- **최우선 원칙(Human 2026-10-07): 로봇 속도는 무조건 가장 느리게. 성공률보다 우선.** RL도 학습 env 안에 속도 상한을 둔다(TCP 5 cm/s, 회전 10°/s, 그리퍼 20 mm/s).
- 과제(Human 확정, A안): 홈 → ready는 RL이 아니라 관절 보간(실기 MoveJ). RL은 ready 자세에서 시작해 **2단계 파지**(림 측면 15 mm 깊이 진입 → 닫기 → 약 10 cm 들기, 5 cm 이상·기울기 <25°·접시 속도 <3 cm/s로 5초 유지) → **3단계 place**. 단계 통과 = 결정적 평가 ≥95%, **다음 단계 전에 Human에게 GUI로 보고 후 승인**.
- 학습 상태(10/7 14:30): 순수 RL slow 최고 55.7%(기준선) → **IL+RL로 전환**: 상태 기반 전문가 데모 + DART → BC(83%) → BC 고정 + PPO residual(실험 `m1013_2f85_il_residual`). BC를 PPO로 직접 미세조정하면 붕괴했다. DECISIONS 2026-10-07 IL 항목들 참조. 3단계는 그 뒤로 미룸.
- W&B: Codex tracking 코드 사용. 계정에 기본 entity가 없어 지금은 offline으로 기록(Human이 entity를 정하면 sync).
- 최종 목표(10/7): 접시를 식기 트레이 칸(핀 간격 20 mm)에 거의 수직으로 세워 넣기, 시연 약 2주 뒤(10/21경). 놓는 자세는 펜던트로 가르친다(DECISIONS 10/7 트레이 항목들).
- 실제 그리퍼: PC USB-RS485로만 제어(`calibration/gripper.py`), **파지력 64/255(1/4) 확정, 미끄러짐 없음**, 속도는 항상 최저.
- 그래프 스타일(Human): grid 없음, 위·오른쪽 축선 없음, 전체 제목 없음(축 제목은 허용).
- 셀 기하: 받침 114.6 mm(접시 윗면 약 14 cm), sim 접시 (1.10, 0) — 실제 (1.215, -0.009)는 미반영. 고정 D455 캘리브레이션 `calibration/data/real_handeye/20261006_122330_fromwrist/T_base_cam_pnp.json`(빼고 예측 평균 5 mm/최대 10 mm).
- **RL 코드 구조(2026-10-06 정리, `rl/`)**: `config.yaml`(모든 수치) / `plate_env.py`(과제) / `rl_utils.py`(설정·실드·체크포인트·서브프로세스) / `train.py`(curriculum·chunk) / `eval.py`(gate·watch·trace·scripted) / `env_setup.py`(urdf·plate·sdf·ready). 예전 파일은 `rl/archive/2026-10-06_pre_refactor/`. 실행은 모두 `~/IsaacLab/isaaclab.sh -p rl/<file>.py ...`. 코드 정리 원칙: 역할별 파일 몇 개, 기능은 함수/모드로, 옵션 많으면 YAML, 1000줄·스파게티 금지.
- **calibration 코드 구조(2026-10-06 정리)**: `config.yaml` / `calib_core.py` / `robot_model.py` / `sensors.py` / `calibrate.py`(wrist·fixed·gt) / `camera_tools.py` / `pipeline.py` / `ros2_nodes.py`(시스템 Python) / `sim_scene.py`(Isaac Sim) / `tests.py` / `run.sh`(connect·stream·pipeline·sim·tests). 예전 파일은 `calibration/archive/2026-10-06_pre_refactor/`. 실행 환경별 주의는 `calibration/README.md`.
- IL: Codex가 `IL/data/`에 스크립트 전문가 데모 50개 생성(검증 통과, LOG 19:55). IL → RL 의견은 DECISIONS 마지막 항목.
- 상세 인수인계: `HANDOFF_CODEX_2026-10-06.md` 맨 위 "현재 RL 코드" 절.

## 현재 상태 (2026-10-08 14:00, Claude 갱신)
- **목표 1 진행:** 실기 `pipeline.py --task plate --plate_policy ik_bc --execute`로 5회 시연. Human 평가: "파지 높이가 조금씩 어긋나는 것 말고는 전반적으로 괜찮다".
- **학습 정책:** BC(IL)만 쓴다. 손가락·절대 xy를 숨긴 입력, 데모 1200개. sim 98~100%(반경 1.04~1.32 m). RL은 쓰지 않는다(10/7 야간 붕괴, DECISIONS 10/8).
- **경로:** 반경별 sim 경로 라이브러리 + 1축 회전(어느 방향이든). 홈 → 접시 앞 10 cm는 관절 1회 이동, 이후 직선 + BC. 실행 중 계산 수 ms. 프레임·테이블 간섭은 FK로 검사.
- **실기 운용:** 6°/s, 그리퍼는 Human 선택, 힘 64. offset (0.025, 0.040). RViz 미리보기 `run.sh viz`. 퇴근 시 `collab/eod_push.sh`(코드·yaml만).
- **실행 계획표: `collab/PLAN_2026-10-08.md`** (담당·완료 기준·일정)
- **남은 것:** 파지 높이 보정, 범위를 테이블 전체로(반경 < 1.04 m, 프레임 근처), 경로 최적화, 가려짐 기반 자동 파지, 목표 2 전체.

## 현재 상태 (2026-10-12 16:xx, Claude 갱신)
- **목표 1 완료(실기):** 테이블 위 접시를 위치 무관하게 잡음(union BC 라이브러리, 반경 0.85~1.32 m, ±60°). 잡기 보정 `offset_m` [0.0025, 0.055].
- **목표 2 데모 수준 완료(실기, 학습 없음):** 마커 4개로 바구니 위치 측정 → 베스트 티칭 경로를 바구니 이동·손 안 접시(FP 중심)·회전 14.7°·기울임 15°·높이·깊이로 보정해 IK → 운반 → 넣기 직전 정지점 → 0.25°/s 넣기 → 놓기 → 홈. 바구니를 옮겨도 성공, `--auto`로 사람 입력 없이 연속 동작(5회 연속 완주). 모든 값은 `calibration/config.yaml plate_ik_bc`.
- **알려진 한계:** 바구니는 지금 자리보다 멀리(+x) 못 감(넣기 자세는 그리퍼가 서야 해서 팔 도달 한계), 운반 경유점이 티칭에서 와서 바구니·접시 자리를 바꾸면 실패, 충돌 검사에 바구니 없음, 파이프라인 가끔 segfault(놓은 뒤 상승 중, `faulthandler.log`로 추적 중), 마커는 종이라 닿으면 움직임.
- **다음(Human 10/12):** ① 바구니 위치 자유도(운반 경로 계획 + 넣는 방법 후보 탐색, 학습은 넣기 직전 mm 보정에), ② 바구니가 옮겨지면 바구니 위에서 손목 카메라로 재측정 후 재계획, ③ 여러 접시(다른 모양·흰색 접시, 2칸 간격, 3개), ④ 받침 없이 바닥 접시 잡기(테두리 끝 집기는 들 때 빠짐 → 위에서 테두리 벽 집기 검토, 티칭 진행 중).
