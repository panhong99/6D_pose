# 경로 생성 다루기: RRT와 경로 라이브러리 (PyRoki는 아침에 추가)

10/19 기준이다. 접시 → 바구니 경로가 어떻게 만들어지는지, 그리고 어떤 손잡이를 돌리면 무엇이 바뀌는지를 정리했다. 손잡이는 대부분 `calibration/config.yaml`의 `plate_ik_bc:` 아래에 있다.

## 1. 한 번의 접시 넣기 = 경로 4조각

| 순서 | 구간 | 만드는 방법 | 코드 |
|---|---|---|---|
| 1 | 받침 → 잡기 → 들기 | **잡기 라이브러리**: 미리 계산·sim 검증한 1,645개 경로 중에서 감지된 접시 위치(거리 r, 방향 az)에 가장 가까운 것을 골라 첫 관절만 돌린다 | `plate_plans.plan_from_library` |
| 2 | 들기 끝 → 슬롯 바로 위 | **운반**: ① 경로 라이브러리(전에 성공한 경로) → ② RRT 순서로 찾는다. 규칙 경로(수직 상승 → 이동 → 하강)는 `rule` 모드일 때 쓴다 | `plate_plans.quadrant_plan`, `rrt_carry` |
| 3 | 슬롯 위 → 꽂기 | **수직 하강**: IK로 1 cm마다 풀어서 곧게 내린다 | `quadrant_plan` 안 `down` |
| 4 | 놓은 뒤 → 홈 | 직선, 막히면 RRT | `CollisionChecker.plan` |

IK와 FK는 numpy/scipy로 직접 푼다(`robot_model.Kinematics`). FK는 URDF 관절을 곱하고, IK는 `scipy.optimize.least_squares`가 시작 자세에서 출발해 위치·자세 오차를 줄인다.

## 2. RRT가 하는 일 (10초 설명)

1. 출발 자세와 목표 자세에서 나무 두 개를 키운다(양방향 RRT).
2. 관절 공간에서 무작위 점을 뽑고(출발·목표 사이 범위 ± `span_deg`), 가까운 가지를 그쪽으로 최대 10° 늘린다.
3. 늘린 가지는 매번 충돌 검사를 한다. 두 나무가 이어지면 경로가 된다.
4. 지름길 찾기(shortcut)와 다듬기(tidy)로 경로를 부드럽게 만든다.
5. 손목(J4/J6)이 출발 자세보다 `max_wrist_deg` 넘게 돌면 버리고 다음 시드로 다시 찾는다.

**충돌 검사에 들어가는 환경 정보(prior)**: 셀 프레임, 테이블, 고정 카메라, 바구니 벽, 마커 블록, 이미 서 있는 접시, 손에 든 접시 점구름. 여기에 **바구니 위 높이 제한**(`FloorCheck`)이 더해진다. 접시 맨 아래가 마커 블록 꼭대기 + 5 cm 아래로 못 내려가고, Q4는 서 있는 접시 꼭대기 + 5 cm 아래로도 못 내려간다.

**같은 입력 + 같은 시드 = 같은 경로.** 입력(접시 자세, 바구니 위치)이 1 mm만 달라도 경로 모양은 달라질 수 있다. 그래서 경로 라이브러리를 먼저 쓴다(4절).

## 3. 손잡이 (config `plate_ik_bc:`)

| 키 | 지금 값 | 바꾸면 |
|---|---|---|
| `basket_carry_mode` | `rrt` | `rule`: 수직 상승 30 cm → 이동 → 하강 고정 규칙(10/18~19 오후 방식) |
| `basket_carry_rrt_fallback` | `true` | `rule` 모드에서 규칙이 실패하면 RRT로 다시 찾을지 |
| `basket_rrt.seeds` | `[0, 1]` | 시도할 난수 시드. 바꾸면 다른 경로가 나온다. 늘리면 더 끈질기게 찾지만 느려진다 |
| `basket_rrt.max_nodes` | `1500` | 시드 하나당 탐색 횟수. 어려운 배치에서 늘린다(4000이면 실패 후보 하나에 수십 초) |
| `basket_rrt.step_deg` | `2.0` | 가지를 따라 충돌을 검사하는 간격(°). 작을수록 촘촘하고 느리다 |
| `basket_rrt.span_deg` | `90` | 무작위로 뽑는 관절 범위(출발·목표 ± 이 값). 줄이면 덜 돌아가는 경로, 너무 줄이면 못 찾는다 |
| `basket_rrt.max_wrist_deg` | `100` | 손목 회전 허용치(°). 줄이면 손목을 덜 돌린다 |
| `basket_carry_clear_m` | `0.05` | 바구니 위 높이 제한 = 마커 블록 꼭대기 + 이 값 |
| `basket_carry_rrt_over_plates` | `[Q4]` | 이 분면은 서 있는 접시 꼭대기 + 5 cm도 높이 제한에 넣는다 |
| `basket_carry_rrt_extra_m` | `{}` | 접시가 서 있을 때 분면별로 높이 제한을 더 올린다(예: `{Q4: 0.1}`) |
| `basket_carry_library` | `data/carry_library` | 전에 성공한 운반 경로 폴더. 빈 값이면 쓰지 않는다 |
| `basket_wall_margin_m` / `basket_insert_wall_margin_m` | 0.01 / 0.005 | 운반 / 수직 꽂기에서 벽·마커와 유지할 최소 거리 |
| `basket_region_min_spacing` | `4` | 두 접시 사이 최소 칸 수 |
| `vel_deg_s` / `basket_quadrant_vel_deg_s` | 60 / 10 | 이동 / 꽂기 관절 속도(°/s). 가속도도 같은 값 |

## 4. 경로 라이브러리 (운반 경로 재사용)

- 파일 위치: `calibration/data/carry_library/<분면>_<색>.json`. 예: `Q4_white.json`.
- 오늘 실기에서 성공한 경로 4개를 넣어 두었다: Q3_red 12번, Q3_white 8번, Q4_red 6번, Q4_white 10번.
- 실행할 때는 저장된 경로를 먼저 써 본다. 시작은 지금 들어 올린 자세에, 끝은 이번 슬롯 위 자세에 다시 잇는다. 충돌 여유, 높이 제한, 손목 검사를 모두 통과해야 쓴다. 통과하면 화면에 `prior ...`가 나오고, 아니면 RRT로 넘어간다.
- 라이브러리에 없는 조합은 처음 성공한 실행이 자동으로 저장한다. 이미 있는 파일은 덮어쓰지 않는다.
- 경로를 바꾸고 싶으면 그 json 파일을 지운다. 다음 성공한 실행이 새로 저장한다.

## 5. 화면에서 읽기

| 표시 | 뜻 |
|---|---|
| `carry high 0.368` | 규칙 경로 (수직 상승 → 이동 → 하강) |
| `carry rrt-floor rrt+tidy` | RRT로 찾은 경로 |
| `carry prior ...plan.json` | 경로 라이브러리에서 가져온 경로 |
| `ik/limit` | 꽂는 자세에 팔이 안 닿거나 관절 한계. **RRT로 풀리지 않는다** (접시 자리나 칸 문제) |
| `insert clearance` | 수직 꽂기가 벽·마커·서 있는 접시에 너무 가깝다 |
| `rrt carry ... RRT budget exhausted` | RRT가 탐색 한도 안에 못 찾음 → `max_nodes`나 `seeds`를 늘린다 |
| `rrt carry ... wrist N deg` | 찾은 경로가 손목을 너무 돌림 → `max_wrist_deg`를 조정한다 |

## 6. 파이썬에서 직접 써 보기 (로봇 없이)

```python
import sys, numpy as np
sys.path.insert(0, 'calibration')
from robot_model import Kinematics, CollisionChecker
kin = Kinematics()
T = kin.fk(np.radians([0, 0, 90, 0, 90, 0]))          # FK: 4x4 flange pose in base
q = kin.ik(T, np.radians([5, 5, 85, 0, 85, 0]))      # IK: seed 근처의 해 (rad)

# RRT: 장애물 상자 (이름, 중심, 크기) 목록으로 검사기를 만들고 두 자세를 잇는다.
# 상자는 J1을 -30 -> +30 deg 돌릴 때 손목이 지나가는 자리(TCP 높이)에 둔다.
chk = CollisionChecker(kin, [('box', [0.559, 0.034, 0.651], [0.12, 0.12, 0.08])], margin=0.01)
a, b = np.radians([-30, 0, 90, 0, 90, 0]), np.radians([30, 0, 90, 0, 90, 0])
for seed in (0, 0, 1):
    path, how = chk.plan(a, b, seed=seed, max_nodes=1500, step_deg=2.0, span_deg=90)
    print(seed, how, len(path), np.round(np.degrees(path[len(path) // 2])))
# 0 rrt+tidy 9 [-8 -2 86 -4 92 3]    <- 같은 시드 = 같은 경로
# 0 rrt+tidy 9 [-8 -2 86 -4 92 3]
# 1 rrt+tidy 9 [ 1 -3 87 -1 91 1]    <- 시드를 바꾸면 다른 경로
```

`'straight'`가 나오면 두 자세 사이에 장애물이 없어서 직선으로 이은 것이다. (10/19 실행 결과)

받침 위 접시 위치를 바꿔 가며 시연 4단계 경로가 나오는지 한 번에 보려면 이렇게 한다(로봇·카메라 없음):

```bash
~/miniconda3/envs/foundationpose/bin/python experiments/stand_free/sweep.py --out <새 폴더> --radii 0.95,1.05 --az 20,30
```

## 7. PyRoki (Codex BN + Claude, 10/19 밤)

- **무엇**: Python 라이브러리이고 계산은 JAX로 한다(CPU 가능, MIT). FK, IK, **궤적 최적화**를 제공한다. 궤적 최적화는 충돌·매끄러움을 비용으로 넣어 한 번에 경로를 만든다. RRT 같은 무작위 탐색은 없다.
- **환경**: `.envs/pyroki-bm-20261019` (시연 conda 환경과 분리). 코드: `experiments/pyroki_bm/` (시연 코드 `calibration/`은 건드리지 않음).
- **결과**: FK는 지금 코드와 1e-12 mm까지 같다. IK는 20/20 성공했다. 받침 지도 111건에서 RRT와 같은 길이로 더 매끄러운 경로를 만들었고(가속도 약 25%, 가가속도 약 40% 낮음), 경로 하나에 약 4초 걸렸다. 최종 합격은 지금 RRT와 같은 충돌 검사(`MultiChecker` + `FloorCheck`)로 정한다.
- **손잡이** (`trajopt_backend.optimize`): `n`(경로 점 수, 기본 20), `maxiter`(최적화 반복, 120), `max_wrist_deg`(100), `seed_paths`(초기 경로, 예: RRT 경로를 주면 다듬기), `robot_penalty_weight`(근사 로봇 충돌 비용, 기본 0: 정확한 검사와 안 맞아서 끔).
- **써 보기** (로봇 없음):
  ```bash
  JAX_PLATFORM_NAME=cpu .envs/pyroki-bm-20261019/bin/python experiments/pyroki_bm/benchmark_saved.py \
      <저장된 plan.json> --standing Q4:6 --output /tmp/out.json      # 같은 출발·도착에서 RRT vs PyRoki
  ~/miniconda3/envs/foundationpose/bin/python experiments/pyroki_bm/bench_sweep.py <sweep 폴더> <새 폴더>   # 받침 지도 전체
  ```
- **아직 안 된 것**: pipeline 연결(옵션). 시연 뒤에 "PyRoki 먼저, 실패하면 RRT" 또는 "RRT 경로를 PyRoki로 다듬기" 중 하나를 옵션으로 붙이는 것이 다음 단계다.
