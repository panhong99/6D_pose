# 접시 PPO 학습 로그

수치는 `config.yaml`, 학습은 `train.py`, 평가/GUI는 `eval.py`.
이번 W&B 연동은 loss/reward/action/success 정의와 학습 종료 기준을 바꾸지 않는다.

## W&B

`tracking.enabled: true`, project `m1013-plate-pick-place`, `mode: auto`.
인증이 있으면 online, 없으면 offline으로 로컬에 기록한다. `WANDB_MODE` 환경변수가 mode보다 우선한다.
TensorBoard 로그도 계속 저장하며 모델/원본 코드/시연 데이터는 W&B에 업로드하지 않는다.
청크마다 별도 run, 같은 실험/stage는 같은 group이며 `Progress/iteration`을 공통 x축으로 사용한다.
resume checkpoint를 run config에 남긴다. Eval gate는 별도 `deterministic_gate` run으로 분리한다.

### 로그인 및 온라인 보기

프로젝트 루트(`/home/pan/pan/6D_pose`)에서 실행:

```bash
PYTHONPATH="$PWD/.runtime/wandb" ~/isaacsim-5.1.0/python.sh -m wandb login
```

그 후 기존 curriculum/chunk 명령으로 학습하면 run URL이 로그에 출력된다.
키는 로그인 도구에서 입력하고 코드/YAML/공유문서에 넣지 않는다.
프로젝트 폴더를 옮길 경우 W&B 의존성 설치:

```bash
~/isaacsim-5.1.0/python.sh -m pip install --target .runtime/wandb wandb==0.30.0
```

오프라인 로그는 checkpoint run 폴더 안 `wandb/offline-run-*`에 저장된다.
나중에 로그인 후 `python.sh -m wandb sync <offline-run-folder>`로 동기화한다(같은 PYTHONPATH 사용).
공식 참고: https://docs.wandb.ai/support/models/articles/how-do-i-deal-with-network-issues

### 기록 단위와 지표

매 PPO update(24 policy steps×4096 envs =98,304 transitions)를 기록한다.
이전 실행에서는 약4초/update였고, update 내부5epoch×4minibatches 손실은 평균으로 기록한다.
물리 스텝마다/미니배치마다 원격 로그를 보내지 않는다.

| 지표 | 용도 |
|---|---|
| Loss/value_function, surrogate, entropy | critic/actor/탐색 손실 |
| Loss/learning_rate, Policy/mean_noise_std | 학습률과 탐색 강도 |
| Train/mean_reward, mean_episode_length | 총 보상, 에피소드 길이 |
| Episode_Termination/success, collision, plate_lost, time_out | 학습 중 종료 유형별 통계 |
| Episode_Reward/* | reach/close/lift/tcp_rise/level_hold/실드 등 각 보상 기여 |
| Perf/* | FPS, rollout/update 소요 시간 |
| Progress/* | checkpoint iteration, 청크별 환경 transition 수/누적시간 |
| Eval/* | 별도 결정적 gate 성공/충돌/놓침/timeout/평균 실드 차단 |

훈련 success와 Eval/success를 혼동하지 않는다. Progress/chunk_*는 청크 재시작마다 0부터 시작한다.
학습 iteration은 checkpoint 번호로 이어지며 과거 runner 특성상 청크 경계에1개 겹칠 수 있다.

## 확인된 결과

`reports/20261007_v10_audit/`: v9→v10 비교, 실제 학습 곡선 PNG/CSV, CPU PPO 2update W&B offline 검증 로그.
과거 학습 곡선은 로컬 TensorBoard에서 복원했다. 기존 학습/GUI를 재시작하거나 Stage3 학습을 시작하지 않았다.
