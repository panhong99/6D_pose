# D455 + Grounding DINO + SAM2 + FoundationPose

## 실행

기존 D455 테스트 창을 `q`로 종료한 뒤 실행합니다. 카메라를 사용하는 RealSense Viewer/ROS 드라이버도 종료합니다.

```bash
conda activate foundationpose
cd ~/pan/FoundationPose
python real_time_project/real_time_v1_kimm.py
```

기본 대상은 `rubik's cube`, DINO-tiny + SAM2.1-tiny입니다.
기본 mesh는 저장소의 `demo_data/077_rubiks_cube/google_16k/textured_57mm.obj`입니다.
로컬 `weights/sam2.1_hiera_tiny_kimm.pt`가 있으면 사용하고, 없으면 Hugging Face에서 SAM2를 로드합니다.
첫 실행은 모델 다운로드가 필요할 수 있습니다. FoundationPose CUDA 환경과 자체 가중치/확장도 필요합니다.

```bash
python real_time_project/real_time_v1_kimm.py \
  --mesh_file ./demo_data/077_rubiks_cube/google_16k/textured_57mm.obj \
  --mesh_scale 1.0 \
  --prompt "rubik's cube" \
  --sam_checkpoint ./weights/sam2.1_hiera_tiny_kimm.pt \
  --retry_interval 0.5
```

`--mesh_scale`은 CAD가 미터이면 1.0, mm이면 0.001입니다.
출력되는 `CAD extent (meters)`가 실제 큐브 크기와 맞는지 확인합니다.
`--serial`, `--width`, `--height`, `--fps`로 카메라를 설정합니다.
`--box_threshold`, `--text_threshold`, `--dino_model`, `--sam2_config`로 검출/분할 모델을 설정할 수 있습니다.
`--sam2_checkpoint`는 `--sam_checkpoint`의 별칭입니다.

## 자동 처리 흐름

1. 시작하면 DINO bbox → SAM2 mask → FoundationPose `register()`를 자동 실행합니다.
2. 정상 tracking에서는 매 프레임 pose/depth 일치 여부를 확인하고, 기본 0.5초 간격으로 DINO bbox만 확인합니다. SAM2는 초기화/복구 때만 호출합니다.
3. tracking score 급락, 유효하지 않은 pose/depth 등의 검증 실패, frame gap 발생 시 재탐지합니다.
4. 검출/등록에 실패하면 `--retry_interval`초(기본 0.1초) 후 새 RGB-D 프레임으로 재시도합니다.
5. `s`로 강제 재탐지, `q` 또는 Esc로 종료합니다. 수기 ROI 선택/마스크 승인 단계는 없습니다.

마스크와 pose 초기화는 반드시 동일한 RGB·정렬된 depth·K 프레임을 사용합니다.
초기화/복구 중에는 큐브를 잠시 멈추는 것이 좋습니다. DINO/SAM2 및 FoundationPose 등록은
동기 실행하므로 그동안 화면 갱신이 멈출 수 있습니다. 복구 시간은 두 단계 시간의 합입니다.
등록 완료 다음 프레임은 등록 소요 시간 때문에 frame-gap lost로 처리하지 않습니다.
따라서 등록 중 빠르게 이동한 대상은 다음 tracking에서 실패해 다시 복구할 수 있습니다.

`--drift_score_ratio` 기본 0.6: 현재 score를 최근 score의 이동 평균과 비교하는 기존 휴리스틱입니다.
0은 score 기반 판정을 끕니다. 이 점수는 정확도/확률이 아니며, 모든 pose drift나 큐브 대칭으로 인한
잘못된 방향을 검출한다고 보장할 수 없습니다. 필요하면 `s`로 재탐지를 강제합니다.
`--max_frame_gap` 기본 1초는 정상 추적 프레임 간격에 적용됩니다.
CUDA OOM/모델 오류/카메라 오류는 숨기지 않고 종료합니다.

## 기록과 파일 구조

- `real_time_v1_kimm.py`: 모델·카메라 생성, 시각화 및 선택적 UDP.
- `recovery_tracker_kimm.py`: SEARCHING → REGISTERED → TRACKING → LOST 상태 처리.
- `detector_comparison/pipelines/pipeline_grounding_dino_sam2.py`: 기존 검증한 DINO+SAM2 파이프라인 재사용.
- `pose_tracker_kimm.py`: CAD/포즈 검증, score 기반 lost 판정.
- `d455_source_kimm.py`: depth 정렬/미터 변환/왜곡 보정, RGB-D 입력.
- `sam_mask_kimm.py`: 이전 수기 SAM 유틸리티. 이번 실행에서는 사용하지 않습니다.

`--debug_dir` 아래에 등록 시도별 RGB/mask/depth/K/bbox와 초기 pose 또는 등록 오류를 저장합니다.
자동 채택된 마스크의 `accepted=true`는 정답 판정이 아닙니다. `--verbose_pose`를 주면 매 pose를 출력합니다.
반복 등록이 많으면 캡처 파일도 늘어납니다.

## ROS2 (기본 활성화)

`--publish_pose`를 생략해도 검증된 pose를 기본으로 UDP 송신합니다. 기본 목적지는 `127.0.0.1:5005`이며 `--udp_host`, `--udp_port`로 변경합니다. ROS2 토픽 발행을 위해서는 `ros2 run foundationpose_bridge udp_pose_bridge`를 별도로 실행해야 합니다.
탐색/등록 실패 동안 새 pose를 보내지 않습니다. 수신 측에서는 마지막 pose의 timestamp가 오래되면
유효하지 않은 것으로 처리해야 합니다. 이번 변경에는 UDP lost 상태 메시지를 추가하지 않았습니다.
`--frame_id` 기본값은 `camera_color_optical_frame`이며 원본 CAD 원점의 카메라 기준 pose입니다.
타임스탬프는 호스트 수신 시각으로, 하드웨어 촬영 시각은 아닙니다.

## 코드 검증

작업 공간(`FoundationPose`의 상위 폴더)에서:

```bash
python -m unittest FoundationPose.real_time_project.test_recovery_tracker_kimm -v
```

모델을 모사한 recovery 전환 테스트와 CLI 로딩을 확인했습니다.
실제 D455 + DINO/SAM2 + FoundationPose 통합 실행, GPU 메모리 사용 및 복구 성능은 아직 검증하지 않았습니다.

### 5.7cm CAD

기본 모델 `textured_57mm.obj`는 원본을 면 방향에 맞춰 정렬한 뒤 각 축 0.057m로 맞춘 텍스처 모델입니다. 원점은 큐브 중심입니다. `--mesh_scale 1.0`을 사용합니다. 원본 OBJ는 보존하며 변환 행렬은 같은 폴더의 `textured_57mm_transform.json`에 저장했습니다. 기존 CAD 원점 기준으로 설정한 외부 변환은 새 원점에 맞게 조정해야 합니다.

### 추적 위치 검증

`--validation_interval 0.5`(기본값): pose 중심이 화면 밖이거나, 주변 depth가 큐브 반대각선+2.5cm보다 크게 다르거나, 주기적 DINO bbox와 어긋나면 "의심(SUSPECT)" 신호로 기록합니다. score 저하도 같은 취급입니다.

**연속 실패 유예(`--loss_patience`, 기본 5)**: 위 신호 중 하나라도 걸린 프레임은 즉시 lost가 아니라 `SUSPECT (n/patience)` 상태로 표시되며 추적은 계속됩니다 — depth 구멍이나 모션 블러처럼 한두 프레임짜리 노이즈에 매번 몇 초짜리 재탐지를 돌리지 않기 위함입니다. 같은 신호가 patience회 연속으로 발생해야 실제 lost로 확정하고 재탐지를 시작합니다. 단, DINO가 완전히 놓친 게 연속 2회면(즉, 물체 자체를 못 찾음) patience를 기다리지 않고 즉시 lost 처리합니다 — 이건 노이즈라기엔 너무 명확한 신호라서요. SUSPECT 상태의 pose는 화면엔 계속 그려지지만 UDP로는 전송하지 않습니다(로봇에 검증 안 된 pose를 보내지 않기 위함).

이 검증은 회전 정확도나 동일 물체임을 보장하지 않습니다. DINO 점검 프레임은 추가 추론 시간 때문에 느려집니다. `--validation_interval 0`은 추가 검증을 끄고 이전 score 기준만 사용합니다(이 경우 `--loss_patience`는 score 저하에만 적용됨). 새 검증/유예 로직은 모사 테스트만 수행했으며 실물 임계값·patience 값 조정이 필요합니다.

### 추적/복구 진단 로그

기존 실행 명령에서 터미널로 기본 출력하며 추가 로그 파일을 자동 생성하지 않습니다.
- `[SCORE]`: 매 추적 프레임의 score, 등록 당시 고정 `registration_baseline`, 판정 전 `ema_before`, 실제 `threshold`, 판정 후 `ema_after`, score 판정 PASS/LOST. PASS는 score 검사만 통과했다는 뜻이며 이후 기하 검사에서 LOST가 날 수 있습니다. 거부한 프레임은 EMA를 갱신하지 않습니다.
- `[SEARCH_BEGIN/END]`: 해당 탐색 구간의 `attempt`(1부터), `retry`(0부터), DINO/SAM 개별 시간, 검출 총시간, pose estimate 총시간, 사이클 시간, 탐색 누적시간, 결과/실패 이유. 실패한 재등록도 같은 구간의 횟수를 누적합니다. 새 loss나 수동 reset은 횟수를 초기화합니다.
- `[REGISTER]`: 같은 frame의 FP `register_ms`와 검증/score까지 포함한 `estimate_total_ms`. SEARCH_END의 estimate 시간과 중복되는 구간이므로 합산하지 않습니다. `None`은 미실행/측정값 없음입니다.
- `[LOST]`: score/기하 검증 실패 또는 frame gap 원인.

DINO/SAM 시간은 CUDA 동기화를 포함하며 SAM은 이미지 embedding 계산도 포함합니다.
`cycle_ms`는 한 번의 탐색 처리이고, `search_elapsed_ms`는 loss 인지(초기 시작은 첫 시도)부터 현재 시도 종료까지로 재시도 대기·프레임 획득·이전 사이클의 저장/화면 처리도 포함합니다. 현재 사이클 이후의 저장/화면 처리와 아직 감지하지 못한 drift 시간은 포함하지 않습니다.
프레임별 출력과 단계별 동기화에는 계측 오버헤드가 있습니다. 이 로그는 원인 구분용이며 실제 drift 순간은 화면 관찰과 함께 비교해야 합니다.
