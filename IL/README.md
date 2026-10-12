# 접시 파지 모방학습 데이터

`data/episodes/`에는 현재 Isaac Lab 환경에서 물리적으로 접시를 잡고 들어 올린 **성공 궤적만** 저장한다.
범위는 Stage 2: ready → 테두리 15 mm 추가 진입 → 닫기 → 상승 → 유지다. 홈에서 ready로 이동하거나
Stage 3 이동/place를 포함하지 않는다. 실제 로봇 명령, 접시를 강제로 부착하거나 위치를 덮어쓰는 동작은 없다.

## 생성

```bash
~/IsaacLab/isaaclab.sh -p IL/collect.py --headless --episodes 50 --num_envs 8
```

기존 manifest를 덮어쓰지 않는다. 추가 수집은 `--output IL/data/<새 이름>`을 사용한다.
기본 seed 123, 최대 20배치 후 종료. 접시 xy ±10 mm·yaw 무작위(현재 환경), expert 속도 0.85~1.15배,
TCP 상승 150~170 mm. 재현 설정과 소스 SHA256은 데이터에 기록한다.

수집 후 원본 파일을 다시 읽어 검증:

```bash
~/IsaacLab/isaaclab.sh -p IL/collect.py --verify --headless
```

검증은 시뮬레이터를 실행하지 않는다. 배열 크기·유한성·SHA256·연속 관측 연결·충돌·실드·성공 유지와
원래 14초 제한 안의 성공 도달을 재계산하고 `data/verification.json`에 저장한다.

## 성공·안전 기준

- 환경의 `held` 판정 + 접시 원점 50 mm 초과 상승 + 기울기 25° 미만, 연속 150스텝(5초).
- 궤적 마지막에도 위 상태를 유지.
- 모든 IK 목표는 기존 복셀 실드를 통과. 차단된 시도는 성공 데이터에서 제외.
- 프레임 50 mm, 테이블 20 mm 안전 여유. 실제 관절 위치도 매 정책 스텝 검사(1 mm 수치 여유).
- 팔/그리퍼 몸체 접촉 힘 0.5 N 초과 시 제외. 센서 이력으로 정책 스텝 안의 4개 물리 스텝을 확인.
- NaN/Inf 없는 데이터만 저장. 실패 원인과 수치는 manifest에 기록하고 실패 궤적은 저장하지 않는다.

실드는 프레임/테이블에 대한 근사 거리장이다. 손가락과 접시의 접촉은 파지에 필요하며 팔 충돌 센서에서
손가락은 제외되어 있다. 실제 시스템의 충돌·파손 안전을 인증하는 데이터는 아니다.
접시 질량·마찰과 손가락/접시 충돌 형상의 기존 가정은 유지한다.

## 파일

- `data/manifest.json`: 성공/제외 목록, 설정·지표·SHA256, episode 단위 train/validation 구분.
- `data/episodes/episode_XXXX.npz`: `allow_pickle=False`로 읽을 수 있는 NumPy 배열.
- `data/rl_config_snapshot.yaml`: RL 설정 복사본.
- `data/reproduction.log`: 수집 전 성공 동작 재현.
- `data/collection.log`: 데이터 생성 진행.
- `data/verification.json`, `data/summary.json`: 파일 전체 검사 및 통계.
- `data/source_snapshot/`: 수집 관련 코드·설정 사본. 이번 첫 수집에서 `plate_env.py`는 수집 중 다른 작업으로
  수정되어, 수집 시 원본 SHA256과 수정 후 SHA256을 manifest에 구분해 기록했다. 해당 파일의 수집 시점
  정확한 소스 사본은 없으며 수정 후 사본은 `plate_env_after_collection.py`다. 수집 프로세스는 이미 읽은
  환경 코드를 계속 사용했고 설정 YAML은 바뀌지 않았다. 향후 수집기는 시작 시 소스 사본을 먼저 저장한다.

주요 배열:

| 배열 | 의미 |
|---|---|
| `obs`, `next_obs` | 현재 RL과 같은 43차원 정책 관측. 설정된 접시 좌표 노이즈 포함 |
| `actions` | **BC 학습 정답**: 제어기 보정 전 7차원 명령 |
| `controller_actions` | 위치 유지·방향 고정 이후 6차원 arm 명령. BC 정답으로 대체하지 않음 |
| `processed_arm_actions` | 위 명령을 scale로 변환한 값 |
| `joint_pos`, `joint_vel`, `arm_joint_target` | 실제 관절 상태와 실드를 통과한 관절 목표 |
| `tcp_pose`, `plate_pose`, `next_*_pose` | 실제 물리 상태, xyz + quaternion(wxyz), 세계 좌표 |
| `hold_target`, `orientation_target`, `orientation_locked` | 제어기 내부 상태(분석/재현용) |
| `held`, `stable_lift`, `lift_m`, `tilt_deg` | 프레임별 파지·들기 상태 |
| `shield_blocked`, `arm_contact_n`, `*_clearance_m` | 안전 검사 결과 |
| `phase`, `terminal`, `rewards` | 시연 단계, 기록 끝, 원래 환경 보상 |
| `env_origin`, `initial_plate_pose` | 병렬 환경 원점, 시작 접시 상태(로봇 base 기준) |

현재 상태에서 action을 준 다음 `next_obs`를 받는 순서다. 정책 주기는 약 30 Hz이며 manifest의 정확한 dt를 사용한다.
`terminal`은 기록 끝이며 환경은 수집 중 자동 리셋하지 않는다. 성공 판정은 기존 조건을 별도로 검사한다.
기록된 제어기와 같은 구조로 BC를 실행해야 행동 의미가 일치한다.

## 학습 시 주의

프레임을 무작위로 나누면 같은 episode가 양쪽에 섞인다. manifest의 episode 단위 분할을 사용한다.
장시간 정지/유지 프레임이 이동·닫기보다 많으므로 phase별 샘플 균형을 검토한다.
검증 데모를 잘 맞추는 것과 새 환경에서 실제로 성공하는 것은 다르다. BC 후 별도 위치·seed의 물리 rollout 평가가 필요하다.
이 데이터는 제한된 xy 범위의 시작 자세 고정 데모다. 광범위한 FoundationPose 입력이나 실제 로봇에 대한 일반화는 아직 검증하지 않았다.

## 2026-10-07 Codex 저속 데이터

[저속시연설명](data/slow_codex_20261007/README.md):50개,30000프레임,검증통과. 호환경로`IL/data_slow_codex`(symlink). 기존빠른시연과분리,source snapshot 및수집gripper속도override명시. BC입력은해당폴더의manifest와episodes만사용하고diagnostics는제외한다.
