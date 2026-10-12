# 확인된 데모를 반영한 Stage 2 v6

Human이 2026-10-06 확인한 파지 데모를 학습 환경에 반영한 설정.

- `PP_VERSION=v6` (기본값). 과거 실행은 `PP_VERSION=v4` 또는 `PP_VERSION=v5`로 선택. 기존 `PP_V5=1`도 v5를 선택하며 명시적인 `PP_VERSION`이 우선한다.
- 접시 테두리까지 테이블 기준 약 140 mm. 받침 높이 114.6 mm.
- Stage 1의 ready 목표는 유지. Stage 2 이후의 림 TCP 목표는 기존보다 접시 중심 쪽으로 15 mm 진입.
- Stage 2에서 열린 그리퍼가 파지 목표 35 mm 이내에 들어오고 tool/finger 방향 오차가 20°/25° 이내이면 그 시점의 TCP 방향을 저장. 이후 환경의 IK action controller가 회전 오차를 보정한다. RL은 전진·상승 등 위치와 그리퍼 열기/닫기를 제어한다. 회전 출력은 방향 유지가 켜진 뒤 보정 명령으로 교체한다. train/eval/watch 모두 같은 controller를 사용하며 reset 때 방향 저장을 해제한다.
- 철제 프레임 50 mm / 테이블 20 mm 실드는 모든 IK 목표에 적용.
- 성공: 파지 상태에서 접시 원점(바닥 중심)이 시작보다 50 mm 이상 상승, 기울기 <25° 상태를 연속 5초 유지. 25°는 데모 측정 9~16°보다 여유를 둔 설정이며 실기 기준으로 별도 검토가 필요하다.
- 접시 수평 유지 보상 추가. Stage 2 제한 시간 24초. 손가락 힘/마찰 변경 없음.
- 렌더링은 원본 OBJ/텍스처, 충돌은 기존 근사 형상. 질량/마찰은 가정값이다.
- 새 실험 및 로그: `m1013_2f85_rim_v6`, `rl/logs/curriculum_v6/`. 관측·정책 출력 차원을 유지해 기존 Stage 1 checkpoint를 재사용한다.

학습 시작 명령 (코드 작성 요청만으로 실행하지 않음):

```bash
bash rl/run_stage2_v6.sh --num_envs 4096 --chunk 1000 --max_iters 15000
```

학습 환경의 실제 방향 유지 controller로 스크립트 검증:

```bash
PP_VERSION=v6 PP_STAGE=2 ~/IsaacLab/isaaclab.sh -p rl/check_rim_grasp.py --headless --num_envs 8 --lift_m 0.16 --hold_s 30 --checkpoint /home/pan/IsaacLab/logs/rsl_rl/m1013_2f85_rim_v4/2026-10-06_11-18-23_stage1/model_999.pt
```

v6 검증에서 `--rim_in 0.015`와 `--lock_orientation`을 추가하지 않는다. 목표 깊이와 방향 보정이 환경 자체에 들어 있다. 스크립트 데모 성공은 학습 정책의 성공률을 의미하지 않으며 curriculum은 결정적 평가 ≥95%를 요구한다.
