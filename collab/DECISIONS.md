# DECISIONS

결정과 이유를 누적한다. 최신 항목을 아래에 추가한다. 핵심 로직(loss, data split, metric 계산) 변경은 반드시 여기에 기록하고 Human 검토를 요청한다.

포맷:

```
## YYYY-MM-DD 제목
- 결정:
- 이유:
- 대안:
- 작성자: Claude | Codex | Human
```

## 2026-09-29 정확도 우선 시 옵션 조정 방침(코드 미변경)
- 결정: 핵심 로직은 바꾸지 않고 CLI 옵션(`--register_iter 10 --track_iter 5 --roi_patience 0 --conf/--imgsz`)으로 정확도를 먼저 비교한다. relock 비활성화 옵션 추가는 Human 확인 후 진행.
- 이유: `FoundationPose-plus-plus` 원본 기본값이 10/5이고, 9/28 커밋(`9f706c6`)의 ROI crop/relock이 속도 개선용이라 정확도에 영향을 줬을 가능성. 단 실측 근거는 아직 없음.
- 대안: 원본과 동일한 Kalman 갱신 순서로 맞추기, `--force_apply_color` 유사 옵션 추가.
- 작성자: Claude

## 2026-09-30 pose 전달 검증은 시뮬레이션 변환 검증부터 분리해 수행
- 결정: 실제 Doosan 연결·이동 전에 FoundationPose UDP packet의 `T_cam_obj`를 수신해 `T_base_obj = T_base_cam @ T_cam_obj` 및 안전 오프셋 flange 목표로 변환하는 시뮬레이션 전용 검증을 먼저 만든다. Isaac Sim M1013 시각화는 이 수치 검증 통과 뒤의 선택 단계로 둔다.
- 이유: 현재 5005 송신기는 존재하지만 수신/변환/목표 생성 노드는 없으며, 로컬 Isaac Sim 설치 경로도 발견되지 않았다. 수치 GT 기반 검증은 Isaac 설치나 실제 로봇 없이 frame 방향·m/mm 오류를 확정적으로 잡을 수 있다.
- 대안: Isaac Sim에서만 바로 검증. 단 설치/robot USD/articulation API 문제와 변환 오류가 섞여 원인 분리가 어렵다.
- 작성자: Codex

## 2026-09-30 simulation-only 접근 목표는 base Z +100 mm, flange 자세는 미검증으로 유지
- 결정: `sim_udp_pose_pipeline.py`는 `T_base_obj`의 translation에 base-frame `[0, 0, +0.100] m`를 더한 position-only 접근점을 생성한다. cube에서 복사한 orientation은 `UNVERIFIED`로 기록하고 실기 로봇 명령에 사용하지 않는다.
- 이유: 고정 카메라 변환과 UDP 전달을 검증하는 현 단계에서는 base 수직 안전 오프셋을 분명히 검증할 수 있다. 반면 M1013 flange 축/도구 좌표계 및 IK·충돌은 아직 실기 검증 전이다.
- 대안: object-local offset 또는 고정 flange orientation을 즉시 확정. 이들은 object orientation·그리퍼/툴 길이·Doosan convention 검증이 선행되어야 한다.
- 작성자: Codex

## 2026-09-30 sim-first hand-eye → target movement → real-world 재현 계획
- 결정: (1) Isaac Sim에서 고정 D455/ArUco-on-flange/M1013/cube로 `T_base_cam`을 hand-eye 방식으로 산출하고 GT 대비 calibration error를 측정한다. (2) 동일 변환 코드와 UDP packet으로 `T_base_obj` 및 접근 flange target을 만들고, 시뮬레이션에서 IK/도달을 검증한다. (3) 검증된 동일 알고리즘을 실제 D455/Doosan으로 옮긴다.
- 이유: simulation에는 GT가 있어 frame order, 단위, calibration error, IK reachability를 실제 로봇 위험 없이 분리 검증할 수 있다. 실기에서는 GT가 없으므로 펜던트/실측으로 별도 교차 검증한다.
- 안전: 실기 이동은 sim에서 수치/도달 검증 후에도 dry-run이 먼저이며, Human의 명시적 승인 전에는 어떠한 robot motion command도 실행하지 않는다.
- 작성자: Codex

## 2026-09-30 Isaac D455 optical axis는 계산된 look-at으로 cube centre를 향하게 함
- 결정: Stage-1 scene의 `/World/D455_Mount/color`는 손으로 정한 Euler angle 대신, USD camera의 local `-Z` optical axis가 cube centre를 정확히 향하도록 계산한 quaternion을 사용한다. GUI의 `Perspective` 뷰는 검증 대상 camera view가 아니므로, 검증 시 해당 camera prim을 viewport camera로 선택한다.
- 이유: 기존 `(66, 0, 43)` Euler 배치는 cube를 실제로 바라보는지 보장하지 않았고, frame/optical-axis convention 혼동을 초기에 제거해야 한다. 코드 내부 방향 검증은 0.000000 deg이다.
- 범위: 이는 simulation scene 배치값일 뿐 실제 D455 외부파라미터나 hand-eye `T_base_cam`의 추정값이 아니다. 실기 캘리브레이션은 별도로 수행한다.
- 작성자: Codex

## 2026-09-30 YOLOE 11s -> v8l 비교 및 코드 정리 (핵심 로직 미변경)
- 결정: `yoloe-v8l-seg.pt`를 repo 루트에 받아(.gitignore 등록됨) `--yoloe_checkpoint`로 11s와 비교한다. 기본값은 비교 결과 확인 후 Human이 결정. `_score()`(매 tracking 프레임 scorer forward)는 수동 재탐지 모드에서 쓰이지 않는 오버헤드로 보이나, 핵심 로직이라 profiling 후 Human 검토를 받고 변경한다.
- 이유: Human이 11s가 이전(v8l 사용) 대비 눈에 띄게 나쁘다고 관찰(시각 판단, 벤치마크 없음). 11s 약 10M / v8l 약 45M params(공식 자료 기억 기반, 미검증; 로컬 측정 53.5M은 모델 전체 포함). validation_interval/loss_patience/drift는 manual 모드에서 미사용이라 DINO/YOLOE 스크립트 차이는 원인 아님으로 정정.
- 대안: 원본 plus-plus 루프와 동일한 순수 모드 비교, mesh 원본(textured.obj) 비교.
- 작성자: Claude

## 2026-09-30 YOLOE 스크립트 자동 재탐지 기본 활성화 (Human 요청)
- 결정: real_time_v1_yolo.py의 --auto_recovery 기본 True(--no-auto_recovery로 수동). validation_interval 0.5, loss_patience 5, drift_score_ratio 0.3을 DINO 스크립트와 동일 기본값으로 옵션화. YOLOEPipeline.detect_bbox() 추가(infer와 동일 비용, mask는 버림). SUSPECT pose는 UDP 미전송.
- 이유: 수동 모드는 틀어진 pose를 lost로 판정하지 않아 화면에 큰 오차 pose가 계속 남음. 기존 YOLOE 스크립트는 validation/drift가 0으로 고정이라 auto를 켜도 감지 불가였음.
- 대안: 수동 유지. 위험: 0.5초마다 YOLOE 추가 추론(v8l이면 FPS 하락 가능), 임계값(0.3/5)은 실물 튜닝 필요.
- 작성자: Claude

## 2026-09-30 속도 shortcut 기본값을 원본(FoundationPose++)으로 복원 (Human 요청)
- 결정: iteration 기본값 10/5(원본), relock_iterations 기본 0(비활성, 항상 full register), roi_patience 기본 0(비활성, 항상 full-frame 검출). PoseTracker는 relock_iterations<=0이면 relock 및 Cutie tier-0 relock을 건너뜀. 옵션으로 다시 켤 수 있음(--relock_iterations 6 --roi_patience 3, --register_iter 5 --track_iter 2).
- 이유: 9/28에 속도 개선용으로 넣은 5/2, relock, ROI crop이 정확도를 해쳤을 가능성(실측 근거 없음). 원본 조건으로 되돌려 기준선 확보.
- 대안: 속도 우선 설정 유지. 
- 작성자: Claude

## 2026-09-30 YOLOE 스크립트 속도 조정: track_iter 3, score_interval 3 (Human 요청)
- 결정: real_time_v1_yolo.py 기본 track_iter 5->3(원본 5에서 완화), 신규 --score_interval 3(drift 검사용 scorer를 3프레임에 1회). PoseTracker.score_interval 추가(기본 1=기존 동작, drift_score_ratio<=0이면 score 계산 생략). 건너뛴 프레임은 직전 판정(last_score_ok)을 유지해 연속 실패 카운트가 초기화되지 않게 함. register_iter 10, relock/ROI off는 유지. DINO 스크립트 기본값은 변경 없음(track 5, score 매 프레임).
- 이유: 합성 프레임 실측 _track 전체 77ms(iter5, score 매프레임) -> 61ms(iter3) -> 51ms(iter3 + score 3프레임당 1회). 정확도 영향은 미측정; track_iter는 정확도와 직접 교환, score_interval은 drift 감지 지연(최대 N프레임) 증가.
- 대안: track_iter 5 유지 + score_interval만 적용(약 67ms 예상).
- 작성자: Claude

## 2026-09-30 real_time_project 파일 간소화 (Human 요청, 로직 변경 없음)
- 결정: real_time_v1_yolo.py -> main.py(주력). camera_rectify_kimm.py + d455_source_kimm.py -> real_time_utils.py로 통합. pose_tracker_kimm.py, recovery_tracker_kimm.py(핵심 로직)와 pose_sender_kimm.py(ros2_ws 테스트가 직접 import)는 유지. real_time_v1_dino.py는 import만 real_time_utils로 변경.
- 이유: 파일 수 축소. 옛 경로 `FoundationPose.real_time_project...` 폴백 import 제거.
- 작성자: Claude

## 2026-09-30 real_time_project 추가 통합 (Human 요청, 로직 변경 없음)
- 결정: pose_tracker_kimm.py(PoseTracker), recovery_tracker_kimm.py(RecoveryTracker)를 real_time_utils.py로 통합(703줄). README_real_time_v1_kimm.md는 루트 README.md 5장(동작 흐름/옵션)으로 요약 이전 후 삭제. pose_sender_kimm.py는 ros2_ws 테스트 import 때문에 유지.
- 검증: 합성 프레임으로 PoseTracker(Cutie+KF enabled) 등록/tracking 실행 확인(등록 2968ms, tracking 51.5ms, 통합 전과 동일). RecoveryTracker 상태 머신은 import/컴파일만 확인, 카메라 실행은 미검증.
- 작성자: Claude

## 2026-09-30 main.py 통합 (--detector yoloe|dino) (Human 요청)
- 결정: main_yoloe.py + real_time_v1_dino.py -> main.py 하나. 기본값은 (a) 통일: register 10, track 3, score_interval 3, validation 0.5, auto_recovery on, relock/ROI off. 검출기 전용 옵션은 argparse 그룹으로 분리, 선택한 검출기만 import. --est_refine_iter/--track_refine_iter는 별칭 유지. DINO 스크립트 대비 변경: publish_pose 옵션 제거(UDP 항상 송신), 기본 mesh는 textured_57mm.obj(필수 아님), DINO 기본 동작이 YOLOE 기본값으로 통일됨(track 5->3, 수동->자동 재탐지).
- 검증: parse_args/--help, YOLOE(v8l) 및 DINO+SAM2 detector 생성과 infer 확인. 카메라 실행 미검증.
- 작성자: Claude

## 2026-09-30 detector_comparison 삭제 및 detectors.py 통합 (Human 요청, 로직 변경 없음)
- 결정: real_time_project/detector_comparison/ 전체 삭제(benchmark.py, live_realsense.py, tests/, install_sam2.sh, requirements.txt, README.md, artifacts.py, yoloe-11s-seg.pt). base.py + pipeline_yoloe.py + pipeline_grounding_dino_sam2.py의 클래스만 real_time_project/detectors.py(400줄)로 통합(각 CLI 예제 main과 패키지 import 분기 제거). main.py 기본 yoloe_checkpoint를 루트 yoloe-v8l-seg.pt로 변경(11s는 이름만 주면 자동 다운로드).
- 이유: 파일 수 최소화. 벤치마크/라이브 비교 도구는 비교가 끝나 불필요하다는 Human 판단.
- 검증: 통합 후 YOLOE(v8l)/DINO+SAM2 생성, infer, detect_bbox 확인. 카메라 실행 미검증.
- 주의: 삭제한 파일은 git에 커밋된 것이면 git으로 복구 가능(미커밋 수정본인 pipeline_yoloe.py의 detect_bbox 추가는 detectors.py에 반영됨). SAM2 설치는 `pip install git+https://github.com/facebookresearch/sam2.git`.
- 작성자: Claude

## 2026-09-30 hand-eye 캘리브레이션 새 구현 (eye-to-hand, 카메라 고정)
- 결정: calibration/ 새로 작성(이전 코드는 Human이 삭제). 체인 T_cam_marker = inv(T_base_cam) @ T_base_flange @ T_flange_marker. OpenCV 5.0에는 cv2.calibrateHandEye가 없어 Park-Martin 닫힌해 + scipy least_squares(soft_l1) 정밀화를 직접 구현. 두산 posx는 [mm, deg, 내재 ZYZ]로 가정. 마커: DICT_4X4_50, ID 1, 25mm(변경 가능). flange pose는 get_current_tool_flange_posx(base, tcp=0)로 읽어 UDP(5006)로 conda 쪽에 전달(ROS2 Jazzy 시스템 python과 conda 3.11 분리).
- 이유: ROS2와 conda의 rclpy 충돌 회피. 합성 검증: 노이즈 0.3도/1mm에서 T_base_cam 오차 0.46mm/0.05도. 마커 크기별 검출 오차(0.5m, 합성): 25mm 평균 2~3.5mm(정면에 가까우면 회전 최대 15도), 50mm 약 1.4mm, 100mm 약 0.7mm -> 25mm는 정확도 한계, 큰 마커 권장.
- 미검증(위험): 두산 ZYZ 오일러 규약을 펜던트 표시값과 대조하지 않음. 실제 로봇/마커 데이터 없음. Human 검토 필요.
- 작성자: Claude

## 2026-09-30 캘리브레이션 자세 수집 자동화 (teach & replay, 저속)
- 결정: Human이 손으로 티칭한 관절 자세(waypoints.json, capture_handeye.py에서 마커가 검출될 때만 W로 저장)를 ros2_run_waypoints.py가 movej로 재생. 도착 시 UDP(5007/5008)로 capture_handeye.py --auto에 캡처를 요청하고 마지막에 자동 풀이. 속도는 기본 3 deg/s, 3 deg/s^2, 하드 상한 10. 기본은 dry-run, --execute + 'MOVE' 입력 필요, 관절 스텝 60도 초과/standby 아님이면 거부, Ctrl+C는 MoveStop(quick stop).
- 이유: 카메라 위치를 모르는 상태에서 마커가 항상 보이는 자세를 자동 생성하기 어렵고 충돌 위험이 있어, 사람이 안전한 자세를 한 번 티칭하는 방식이 안전. 
- 미검증(위험): 실제 로봇 동작 전부. amovej/check_motion 완료 판정, 서비스 이름(motion/move_stop), robot_state==1(STANDBY) 가정, 관절 경로 충돌 여부는 Human이 dry-run과 저속으로 확인해야 함. 핵심 안전 로직이라 Human 검토 요청.
- 작성자: Claude

## 2026-09-30 standalone rendered eye-to-hand + cube approach (Human 요청)
- 결정: 기존 stage-1 승인 대기를 현재 Human의 전체 구현 요청으로 대체했다. 기본 headless=False인 standalone에서 실제 RTX ArUco 영상→기존 solve_eye_to_hand→RGB-D 큐브 자세→base 변환→IK 관절 재생→GT 오차 계산을 자동 실행한다.
- 측정: flange에 DICT_4X4_50 ID 1(100 mm), cube 상면에 ID 2(45 mm). 작은 cube marker의 RGB-only 거리 오차(960×540에서 약 44 mm, 1920×1080에서 약 20 mm)를 해결하기 위해 이상적인 simulated depth 평면과 marker corner ray 교점을 rigid alignment한다. FoundationPose 추론 자체는 실행하지 않는다.
- 핵심 metric: T_a_b는 b→a, metres. translation은 두 pose 원점의 Euclidean 거리×1000, rotation은 R_gt.T @ R_est의 geodesic 각도(deg). 18 sample 중 매 네 번째 5개는 solver에서 제외하고 held-out residual을 계산한다. GT camera/cube는 평가에만 사용한다. depth 평면은 mean-centred SVD(초기 componentwise median이 평면 밖일 수 있음을 독립 수치 테스트로 찾아 수정).
- 종료/통과: 최대 36 자세, 목표 유효 관측 18개, 최소 12개. camera<10 mm/2°, cube<10 mm/2°, reach<1 mm, held-out 최대<5 mm/2°. 이 metric 정의와 임계값은 Human 검토 대상으로 기록하며 작업을 중단하는 승인 단계는 추가하지 않는다.
- 이동 범위: M1013 URDF FK/IK로 USD link를 관절 보간 재생한다. PhysX rigid body/joint를 비활성화한 운동학 검증이며 충돌/동역학/그리퍼 파지는 미검증이다. flange 목표는 측정 cube 중심에서 base Z +180 mm. 실제 ROS2/Doosan 연결·이동 없음.
- GUI: 결과 수치, D455/overview 전환, calibration+approach 재생 버튼. 측정 후 sensor render product 갱신 중지, idle sleep 25 ms. 출력과 사용법은 calibration/README_sim_eye_to_hand.md.
- 작성자: Codex

## 2026-09-30 저장된 eye-to-hand 행렬 재사용 object-approach 모드
- 결정: `--object-approach`는 기존 `calibration/data/eye_to_hand/result.json`의 `T_base_cam`만 읽고, flange ArUco를 장면에 추가하거나 hand-eye를 다시 풀지 않는다. cube 상면 ArUco ID 2와 simulated RGB-D에서 `T_cam_object`를 측정한 뒤 `T_base_object = T_base_cam @ T_cam_object`와 cube centre 위 base Z +180 mm의 flange 접근점을 계산·재생한다.
- 이유: hand-eye 단계와 object localization/motion 단계를 분리해 Human이 요구한 다음 단계를 독립적으로 확인할 수 있어야 한다. 결과는 기존 calibration 결과를 덮지 않고 `object_approach_result.json`에 저장한다.
- replay 수정: `context.save_as_stage()`가 in-memory URDF stage의 cached `Usd.Prim`을 만료시켜 replay callback이 `/m1013/base_link`에서 죽었다. USD 저장을 GUI loop 종료 후로 옮겼다.
- 작성자: Codex

## 2026-09-30 FoundationPose frame 점검 및 cube 대칭 분리 metric (Claude, Human 검토 요청)
- 점검 결과(frame): CAD `demo_data/077_rubiks_cube/google_16k/textured_57mm.obj`은 bbox centre `[0,0,0]`, extents 57 mm×3, 면이 축 정렬(각 축 area frac ≈0.28). Isaac `/World/RubiksCube`는 centre 원점, base축 정렬, 60 mm, 단색(displayColor)이다. 따라서 mesh-origin offset = 0 mm, frame 원점 정의는 동일(둘 다 cube centre). 남는 차이는 크기(half-size 1.5 mm)와 회전 라벨링뿐이다. (`textured.obj`는 centre `[-16.1,-0.5,28.7] mm`로 원점이 다르므로 쓰면 안 된다.)
- 결정: FoundationPose approach report에 `foundationpose_frame_check`를 추가. 회전 오차를 24개 cube 고유회전(signed permutation, det +1) 중 최근접 S와 residual로 분리(`R_gt.T @ R_est = residual @ S`). 위치 오차는 camera frame FP-only(`T_cam_object - inv(T_base_cam_GT) @ T_base_cube_GT`, ray 방향/횡방향)와 hand-eye-only(`T_base_cam_est @ T_cam_object_GT - GT`)로 분리.
- 결정(pass metric 변경): `passed`의 cube 회전 조건을 FP 모드에서만 raw가 아니라 symmetry-reduced 각도 < 2°로 판정. 새 필드 `position_approach_passed`는 회전을 제외한 기존 조건(calibration<10 mm/2°, cube 위치<10 mm, reach<1 mm, held-out)만 본다. 이유: Isaac cube가 단색이라 24-fold 회전은 관측으로 구분 불가하며, 현재 approach는 위치만 사용한다. ArUco 경로(`--object-approach`, full calibration)는 기존 판정 그대로.
- 유지: approach는 position-only(`cube 위치 + base Z 180 mm`, nominal safe flange orientation). FoundationPose 회전을 flange target에 적용하는 옵션은 만들지 않았다(별도 옵션·검증 전 적용 금지).
- Human 검토 요청: symmetry-reduced 회전을 pass에 쓰는 것이 타당한지, 실제 Rubik's cube(색 면 구분 가능)에서는 raw 회전을 써야 하는지.
- 작성자: Claude

## 2026-09-30 실제 Rubik CAD 렌더, cube +50 cm, 시작 자세 all-zero (Claude, Human 요청)
- 결정(cube 렌더): 단색 60 mm `UsdGeom.Cube` placeholder를 `demo_data/077_rubiks_cube/google_16k/textured_57mm.obj` 렌더로 교체(`add_rubiks_cube`). mtl은 이미지 텍스처 없이 7개 flat Kd 색이라 색별 `UsdGeom.Mesh` 7개(16384 face 전부, 누락/중복 0)로 만들고 translate만 적용. 크기 57 mm로 CAD와 정확히 일치(half-size mismatch 0 mm).
- 결정(GT 단일 출처): cube 중심/크기를 scene main에서 `args.cube_center_m`/`args.cube_size_m`로 넘겨 `sim_eye_to_hand.py`의 `cube_gt`/`cube_marker`가 하드코딩(.42,.10,.03 / .0305)이 아니라 같은 값을 쓰도록 변경. cube ArUco 높이 = size/2 + 0.5 mm.
- 결정(거리): D455 pose는 캘리브레이션 당시와 완전히 동일하게 고정(position `[0.72,-0.72,0.78]`, aim `[0.42,0.10,0.25]` 상수화). 이유: 카메라를 움직이면 저장된 `result.json`의 `T_base_cam`이 무효가 되고, 덮어쓰기는 금지. 대신 cube를 카메라 수평 방위를 따라 바닥 높이 유지하며 밀어 카메라 거리 1.152 → 1.652 m(+0.50 m), 위치 `[0.2145, 0.6616, 0.0285]`. 로봇 base 평면거리 0.70 m, IK 도달 확인.
  - 기각안 1: cube를 3D ray로 연장 → 바닥 아래로 내려가 불가. 기각안 2(1차 시도): 기존 IK nominal 대기자세에서 cube 이동 → flange가 시선을 가려 YOLOE가 flange를 cube로 검출(conf 0.025), FP 810 mm 오차. 기각안 3: 카메라를 0.5 m 뒤로 → T_base_cam 무효.
- 결정(시작 자세): 기본 시작/복귀 자세를 모든 joint 0°(FK: flange z=1.45 m, +Z 방향 = 하늘)로 변경. `--start-joints-deg J1..J6` 옵션, report `start_joints_deg`. IK seed는 기존 nominal IK 해(`q_nominal`)를 유지해 elbow-up branch 선택을 보존(0 자세 seed는 다른 branch로 수렴할 위험). 0 자세 팔은 camera-cube 시선에서 0.435 m 떨어져 가림 없음.
- 주의: 이 cube 이동은 full calibration 모드(result.json 작성)의 cube ArUco 위치에도 적용된다. full calibration을 다시 돌리면 result.json이 덮어써지므로 실행 전 Human 확인 필요.
- 작성자: Claude

## 2026-09-30 카메라 배치별 setup 분리, side 배치 재캘리브레이션, 관절 속도 제한 (Claude, Human 요청)
- 배경: Human이 Isaac GUI에서 cube와 D455를 직접 옮김(USD 저장본에서 읽음). cube `[0.64796, 0.01169, 0.0285]`, D455 `[1.13712, 0.07371, 0.37973]`, quat wxyz `(0.6527842, 0.3673936, 0.32492983, 0.57733464)`. 카메라가 옮겨졌으므로 기존 `T_base_cam`은 무효.
- 결정(setup 분리): `isaacsim_scene_m1013_d455_cube.py`에 `SETUPS`(v1 = 기존 카메라, `data/eye_to_hand`; side = Human 배치, `data/eye_to_hand_side`), `--setup`(기본 side). hand-eye 결과는 해당 카메라 pose에만 유효하므로 setup마다 data dir 분리. `foundationpose_sim_pose.py --setup`, runner는 `SETUP` env(기본 side). full calibration은 기존 result.json이 있으면 `--overwrite-calibration` 없이는 중단(v1 보호).
- 결정(캘리브레이션 중심): v1의 flange 중심 `[.48,-.05,.43]`은 side 카메라 화면 밖(v=-603 px)이라 setup별 `calib_flange_centre`로 분리. side = 마커가 렌즈 앞 0.5 m, 이미지 (960,300)에 오도록 `[0.6061, 0.1391, 0.1435]`(오프라인 검사: 36개 중 26개 화면 안, 링크 원점 최저 z 0.006 m). 샘플링 범위(±9–10 cm, ±25°)와 18 목표/12 최소는 동일, **최대 시도 36 → 60**(side는 거부율이 높음; v1은 18개에서 조기 종료하므로 결과 불변).
- 결정(속도): 운동학 재생을 프레임 수 고정(approach 90, calib 12, replay 25)에서 **관절 최대 속도 제한**으로 변경. `joint_path`가 가장 많이 도는 관절이 `--max-joint-speed-deg`(기본 30°/s)를 넘지 않게 dt=1/30 s 간격으로 보간, GUI에서는 벽시계로 속도 유지, headless는 대기 없음. replay도 벽시계 인덱싱. report에 `max_joint_speed_deg_s`.
- 기타: `--cube-xy X Y` 옵션(setup 기본값 덮어쓰기), runner는 추가 인자를 두 Isaac 실행에 전달.
- Human 검토 요청: side 카메라가 낮아(z 0.38 m) 캘리브레이션 자세가 바닥 근처(링크 원점 z≈0)까지 내려감. sim에서는 충돌 검사가 없지만 실기에서는 충돌 위험 → 실기 배치 시 카메라를 높이거나 캘리브레이션 볼륨을 테이블 위 15 cm 이상으로 두는 것을 권장. approach의 flange 방향은 여전히 "카메라를 향하는 nominal"이라 side 배치에서는 수평에 가까움; top-down(flange z = -Z) 접근 IK도 도달 가능 확인됨 — 별도 옵션으로 만들지 결정 필요.
- 작성자: Claude

## 2026-09-30 grasp-ready top-down 접근 자세 (Claude, Human 요청 "조인트 5번까지 그리퍼 있는 것처럼")
- 결정: 기본 접근 자세를 `--approach-orientation grasp`로 변경. j1–j5는 top-down 손목(j4≈0°, flange z = base −Z, IK seed `[atan2(y,x), 0, 90, 0, 90, 0]`), j6은 cube 옆면에 맞춤. cube yaw는 pose 추정(FP 또는 ArUco)의 가장 수직인 축을 뺀 나머지 축을 base XY에 투영해 mod 90°로 구하고, 90°마다 반복되는 후보 4개 중 IK 가능하고 j6 비틀림이 가장 작은 것을 선택. 이것은 FoundationPose 회전(yaw만)을 flange 목표에 적용하는 첫 번째 옵션이며, 90°/180° 대칭은 평행 그리퍼에 무관하므로 yaw mod 90만 사용.
- 옵션: `topdown`(yaw 고정 180°), `camera`(기존 카메라 향함 자세). `--tool-length-m`(기본 0): TCP를 flange +z로 tool 길이만큼 두고, pregrasp = TCP가 cube 중심 + 180 mm, grasp = TCP가 cube 중심. grasp 자세는 **IK 도달성만 확인**하고 재생은 pregrasp에서 멈춤(하강/파지 미실행 유지). `--cube-yaw-deg`로 scene cube를 회전시켜 검증 가능(GT도 같이 회전).
- report `grasp`: yaw 측정/GT/오차, tilt, flange yaw, q_pregrasp/q_grasp, pregrasp/grasp flange T.
- Human 확인 필요: 실제 그리퍼 TCP 길이와 finger 닫힘 축(flange x 또는 y). 현재는 flange x축을 cube 면 법선과 평행하게 둠. 닫힘 축이 y라도 90° 대칭이라 cube에는 동일하게 유효.
- 작성자: Claude

## 2026-09-30 pregrasp 높이 180 → 100 mm (Claude, Human 요청 "너무 멀다, 핸드로 할 것")
- 결정: `--pregrasp-height-m`(기본 0.10) 추가, TCP를 cube 중심 위 100 mm(cube 윗면 위 71.5 mm)에 정지. 0.0285 m(cube 윗면) 이하는 argparse에서 거부. tool 길이 0이면 flange 기준. GUI 문구도 높이 값을 표시.
- Human 확인 필요: 핸드 장착 시 TCP(손바닥/파지 중심) 길이를 `--tool-length-m`로 넣어야 flange가 아니라 핸드 파지점이 100 mm에 옴. 넣지 않으면 핸드가 cube와 겹칠 수 있음.
- 작성자: Claude

## 2026-10-01 실기 셀(테이블+알루미늄 프레임+D455 2대) sim 구성 및 voxel 충돌 검사 (Claude, Human 요청)
- 입력(Human 실측): 테이블 1.80×1.20 m, 프레임 높이 1.20 m, 카메라 바 양쪽 수직 프레임 간격 0.75 m, 그 바에 D455 2대(양쪽, 약 45° 기울임), 로봇–물체 1.00–1.10 m. 실기 D455 사진 `calibration/data/real_env/real_color_20261001_0917*.png`, intrinsics `real_intrinsics_*.json`(1280×720, fx 647.8, fy 646.9 → 89°×58°).
- 결정(형상, 단일 출처): `calibration/workcell.py` `FRAME_CELL` + `frame_layout()`이 테이블·프레임·D455 박스와 카메라 pose를 만들고, 같은 박스 목록으로 Isaac visual과 충돌 모델을 생성. **가정(실측 필요)**: 로봇 base는 테이블 짧은 변에서 30 cm·중앙선, 물체 x=1.05 m, 프레임은 물체 중심 0.75×0.75 m 정사각(깊이 미측정), 프로파일 40 mm, 카메라 바 = 로봇 반대편 윗 레일(1.20 m), 카메라는 기둥에서 6 cm 안쪽·바 아래 5 cm, pitch 45°·yaw는 물체 방향.
- 결정(카메라 광학): frame 셋업은 실기 D455 intrinsics로 렌더(1280×720, focal 1.88 mm, aperture 역산). 기존 v1/side는 원래 광학(1920×1080, 62° HFOV) 유지 — 저장된 캘리브레이션 유효성 보존.
- 결정(셋업): `frame_L`, `frame_R`(두 카메라 모두 렌더, setup이 측정 카메라 선택, data dir `eye_to_hand_frame_L/R` 분리). 캘리브레이션 중심은 오프라인 탐색(IK·충돌·시야)으로 `[0.882, ±0.0635, 0.6138]`.
- 결정(충돌): M1013 URDF collision mesh(.dae, mm→m) 표면 샘플 링크당 750점(캐시 `calibration/data/m1013_collision_points.npz`; Isaac Python에 pycollada가 없어 conda env에서 1회 생성, conda env에 `pycollada==0.9.3`를 `--no-deps`로 설치)을 FK로 변환, 1 cm voxel occupancy의 거리장(EDT)으로 clearance 계산. margin 20 mm. 캘리브레이션 중에는 ArUco 판(0.085 m, r 0.072 m)을, 접근 중에는 `--tool-length-m/--tool-radius-m` 원통을 툴로 포함. 이동은 직선 관절 보간이 충돌이면 bidirectional RRT(2° 간격 검사)+shortcut. 충돌 자세의 캘리브레이션 샘플은 거부. voxel/박스 목록은 `data/<setup>/workcell_voxels.npz/.json`로 export(실기에서 경로 사전 검사용).
- 결정(pass metric 변경, Human 검토 요청): frame 셋업에서 `collision_free`(재생된 접근 경로 최소 clearance ≥ margin)를 `position_approach_passed`에 포함. 다른 셋업은 영향 없음.
- 결정(캘리브레이션 샘플링, Human 검토 요청): setup별 `calib_samples/attempts/spread_m/spread_deg`. 기존 기본값(18, 60, ±9/10 cm·z −6..+9 cm, ±25°)은 난수 시퀀스까지 동일 재현(검증). frame 셋업은 30개, 최대 120회, ±15/15/12 cm, ±35°.
- 결정(속도): 기본 관절 속도 30 → 15°/s (Human 요청).
- 발견: 실기 FOV(89°)에서는 100 mm ArUco가 0.6–0.8 m 거리에서 84–115 px에 불과하고, hand-eye 오차가 카메라 광축 방향 스케일 편향(−6~−8 mm)으로 지배됨. 45° pitch에서는 물체가 이미지 아래쪽(v≈640/720). 카메라 바가 1.2 m 윗 레일이라면 물체를 화면 중앙에 두려면 pitch ≈ 68° 필요.
- 작성자: Claude

## 2026-10-01 움직임 부드럽게(경로+시간 배분), 충돌 여유 50 mm, voxel 시각화, 카메라 위치=Human cone (Claude, Human 요청)
- Human 지적: 로봇 속도 아직 빠름, 움직임이 불안, 실기에서는 프레임에 부딪힐 것 같음, "voxel이 맞나?", 카메라는 GUI의 cone 위치에 달 것.
- 원인 분석(코드로 확인): (1) 경로 계획기(RRT)가 무작위 중간 자세(j5 76°, j6 −29°)를 남기고, (2) 그 점들을 일정 속도 직선으로 이어 각 점에서 급정지·급출발, (3) GUI Replay가 현재 자세에서 시작 자세로 순간이동(teleport), (4) 충돌 여유가 20 mm로 작았음(최소 22 mm). 직선 경로는 near 상단 레일(x≈0.68, z≈1.19)에 닿아 막힘.
- 결정(경로): `workcell.CollisionChecker.tidy()` — 경유점을 이웃 점의 중점 쪽으로 당기되 양쪽 edge가 충돌 여유를 유지하는 만큼만 이동(elastic band). RRT+shortcut 뒤에 적용(`rrt+tidy`). 이후 PCHIP로 코너 없는 C1 경로(`trajectory.smooth_path`)를 만들고, 부드러운 경로가 여유의 90%를 침범하면 polyline을 세분(1/2/4/8)해 재시도, 마지막은 코너 유지.
- 결정(시간 배분): `trajectory.time_parameterize` — 구간별 곡률/관절 속도 한도, 가감속 한도(forward/backward pass), 0.4 s 이동평균(저크 제한), 시작·끝 속도 0. 이전: 구간마다 등속+급정지. 측정(5 waypoint 경로): peak a 171 → 10.8°/s², peak jerk 5143 → 낮음, 시작 첫 프레임 속도 6.5 → 0.1°/s.
- 기본값: 관절 최대 속도 15 → **8°/s**, 가속 한도 **10°/s²**(`--max-joint-speed-deg`, `--max-joint-accel-deg`). GUI에 속도 슬라이더(1–40°/s, Replay에 적용). Replay는 현재 자세에서 시작 자세로 이어서 이동한 뒤 재생(순간이동 제거).
- 결정(충돌 여유): 기본 margin 20 → **50 mm**(`FRAME_CELL['collision_margin_m']`). 계획 경로 최소 clearance 50.0 mm 확인. 샘플 간 이산화로 실제 최소는 ≈36 mm까지 내려갈 수 있어 테스트는 0.6×margin 기준. 툴 길이 0이면 핸드는 충돌 모델에 없음 → 핸드 장착 시 `--tool-length-m/--tool-radius-m` 필수.
- 결정(voxel 표시): 이전 GUI의 흰 프레임은 렌더용 박스이고 voxel(거리장)은 눈에 보이지 않는 내부 자료였음. 이제 GUI에서 점유 voxel 표면층(1 cm)을 빨간 반투명 큐브로 그림(`--show-voxels`, 버튼으로 토글). 센서 촬영이 끝난 뒤에 생성하므로 RGB-D/YOLOE/FoundationPose 입력에 섞이지 않음.
- 결정(카메라 위치): Human이 GUI에 놓은 `Cone_01 (1.0722, 0.5706, 1.1654)`, `Cone_02 (1.0688, −0.1324, 1.1043)`을 `frame_L`, `frame_R` 카메라 위치로 사용(`FRAME_CELL['camera_positions']`). 방향은 큐브를 바라보게(`camera_aim='object'`): pitch L 63.3°, R 82.9°(수평 아래). Human이 말한 "45°"를 유지하면 R은 큐브가 시야 밖(위치에서 38° 벗어남)이므로 적용하지 않음 — Human 확인 필요. 새 위치마다 캘리브레이션 중심 재탐색 필요 → `find_calib_centre.py` 추가.
- 가정 주의: 프레임 기둥/레일 배치는 여전히 내 가정(0.75 m 정사각, 로봇 base 30 cm 안쪽)이며 cone 위치와 맞지 않음(L cone은 가정 프레임 바깥). 실제 프레임 배치 확인 필요.
- 작성자: Claude

## 2026-10-01 (후속) 계획 여유, GUI 수정, 150 mm 마커 (Claude)
- 충돌: 검증 margin 50 mm는 유지하되 계획은 1.3배(65 mm)로 여유를 두고(edge 끝점은 50 mm), 회전만 하는 `link_1`(테이블과 60 mm 고정)은 검사에서 제외(`STATIC_LINKS`). 결과: 접근 경로 최소 clearance 42.4 → **63.2 mm**, 코너 없이 부드러운 경로 유지.
- GUI: 패널을 ScrollingFrame으로 바꾸고 버튼(D455 image, overview, 속도 슬라이더, Replay, voxel 토글)을 맨 위로(슬라이더 추가 후 Replay 버튼이 창 아래로 밀려 안 보였음). voxel 표시에서 D455 렌즈 주변 12 cm는 그리지 않음(카메라 시점이 자기 몸체 voxel에 갇혀 검게 보였음; 충돌 검사에는 포함).
- 카메라 방향: Human 선택 = 큐브 중심을 바라보기 유지(L 63.3°, R 82.9°). 프레임 형상: Human 선택 = 현재 가정(0.75 m 정사각) 유지 — cone L은 가정 프레임 밖이라 실기 충돌 모델로는 부정확, 실측 필요.
- 캘리브레이션 마커: frame 셋업 기본 150 mm(`flange_marker_m`, `--flange-marker-m`). cone 위치에서 100 mm → 150 mm: L 16.49 → **9.42 mm**(0.319 → 0.135°), R 7.42 → **4.58 mm**(0.309 → 0.067°). 여전히 카메라 광축 방향 편향이 대부분. frame 셋업은 cube ArUco(45 mm)가 안 보여도 hand-eye 결과만 저장하도록 함(`mode: hand-eye calibration only`).
- 작성자: Claude

## 2026-10-01 (Codex: 실기 flange pose 자동 수집 경로)
- 결정: 수동으로 펜던트 값을 입력하는 대신, 사람이 자세를 바꾼 뒤 `ros2_flange_udp.py`가 Doosan ROS2의 `get_current_tool_flange_posx(DR_BASE)`와 `get_current_posj()`를 읽어 localhost UDP 5006으로 보낸다. `capture_handeye.py --robot udp`는 C 키 시 최신 pose를 자동으로 샘플에 기록한다. 두 스크립트 모두 이동/jog/waypoint 실행을 하지 않는다.
- 호환: 설치된 `DSR_ROBOT2.py`가 현재 `dsr_msgs2`에 없는 선택 서비스 `SetSingularityHandlingForce`를 import 단계에서 생성해 `NameError`가 났다. 해당 선택 서비스 client만 조건부 생성하게 source 및 install 복사본을 수정했다. pose 조회 API에는 영향이 없다. `py_compile` 통과. Doosan workspace를 재빌드하면 install 복사본이 source 변경으로 다시 생성된다.
- 환경: ROS Jazzy는 시스템 `/usr/bin/python3`으로 실행해야 한다. Miniconda Python이 PATH를 우선하면 ROS Python ABI/패키지 오류가 날 수 있다.
- 작성자: Codex

## 2026-10-01 (Codex: Claude용 실기 hand-eye 인수인계)
- Human은 펜던트 수동입력으로 31개 marker/flange 샘플을 수집했다. solve residual **184.08 mm mean / 584.46 mm max**, **22.30° mean / 108.18° max**이므로 산출 transform은 사용하지 않는다. 샘플은 보존하고 우선 좌표계/Euler/TCP/마커 치수 가정을 오프라인 분석한다.
- 후속 handoff는 `collab/HANDOFF_REAL_HANDEYE_2026-10-01.md`. Human은 PC와 Doosan controller가 IP socket으로 연결되어 있다고 알렸으나 pose state API/protocol은 확인되지 않음. 추정 protocol을 만들지 말고 read-only 지원 API를 조사한다.
- 결과 파일 보호: `data/eye_to_hand/result.json`, `T_base_cam.txt` 금지된 overwrite 유지. 실패한 fresh solve는 `data/T_base_cam.json`에 별도 저장되어 있으며 실사용 금지.
- 작성자: Codex

## 2026-10-01 (Codex: 31개 수동 샘플 재검사)
- 관찰: 저장된 all-sample 해의 residual이 높고 #0–13 vs #14–30 상대 회전 일관성이 크게 달라진다. 각 flange relative rotation `R_fiᵀ R_fj`와 marker-camera relative rotation `R_miᵀ R_mj`의 geodesic angle은 고정 eye-to-hand/rigid mount에서 동일해야 하므로 이 검사는 `T_base_cam` 추정치와 무관한 데이터 정합성 진단이다.
- #0–13 pair median angle mismatch 0.98° (92.3% <5°); #14–30 median 20.80° (15.4% <5°). 따라서 문제를 단순히 “몇 개의 isolated outlier”로 간주해 삭제하지 않는다. #14 이후 구간의 pose pairing/convention/rigidity 변화를 확인한다.
- 결정: 31개 원본 보존. 신규 transform 생성이나 use 권장하지 않음. pendant field/frame/TCP semantics를 확인하기 전에 solver/loss/threshold를 변경하지 않는다.
- 작성자: Codex

## 2026-10-01 실기 hand-eye: 실패 원인 = pose 짝 어긋남, 자동 캡처로 전환 (Claude)
- 결정: 31샘플 solve 실패(184 mm)의 원인은 수기 입력 pose와 영상의 인덱스 어긋남으로 판단(LOG 참조). 기존 샘플은 보존하고 재짝지은 해는 탐색용으로만 기록, 사용하지 않음. 재수집 권장.
- 결정: 실기 수집은 `auto_handeye_capture.py`로 — 로봇은 Human이 움직이고, pose는 ROS2 `get_current_tool_flange_posx(DR_BASE)` read-only. 로봇 이동 명령 경로(`ros2_run_waypoints.py`, `capture_handeye.py --auto`)는 사용하지 않음.
- 새 캡처 게이트(Human 검토 요청, 진단용 기본값): 정지 1.5 s(0.3 mm/0.1°), 저장 샘플과 ≥40 mm 또는 ≥10° 차이, 회전각 짝 검사 중앙값 ≤8°(저장 3개 이상), 저장 샘플 해로 예측한 marker 위치 오차 ≤25 mm(저장 6개 이상), 기존 tilt ≥15°·jitter ≤1.5 mm 유지. 리포트에 leave-one-out 추가.
- 리스크: 25 mm 예측 게이트는 초기 샘플이 나쁘면 이후 좋은 샘플을 거부할 수 있음(U로 되돌리기/세션 새로 시작). 100 mm marker는 0.8–1.4 m에서 회전 잡음 ~2°라 150 mm 이상 권장(sim 결과와 같은 결론).
- 작성자: Claude

## 2026-10-01 real_cam 시뮬, 복귀 경로, 펜던트 경유점, link_2 제외 (Claude)
- 결정(Human 요청 반영): 실기 hand-eye `T_base_cam`(data/real_handeye/20261001_160439)의 위치·방향으로 단일 카메라 setup `real_cam`(data_dir `eye_to_hand_real_cam`) 추가. 프레임은 카메라와 바깥 면이 맞도록 x로 +7 cm 이동(해석 — Human 확인 필요). 검증 margin 0.10 m(`cell_overrides`), 환경변수 `CELL_MARGIN_M`로 덮어쓰기(캘리브레이션 자세는 0.10에서 불가해 0.05로 실행). sim 캘리브레이션 9.18 mm / 0.245°.
- 결정(안전 검사 변경, Human 검토 요청): `STATIC_LINKS`에 `link_2` 추가. 근거: 영점·접근 자세 모두 어깨 하우징이 테이블과 80 mm로 일정하고 프레임에서 0.7 m 이상 떨어져 있어, margin 0.10 m는 로봇 자체 구조로 충족 불가였음. 영향: 모든 frame setup의 clearance 수치가 달라질 수 있음(link_2 제외). unittest 8/8.
- 결정: 접근 후 시작 자세로의 **복귀 경로를 별도 계획·검증**(`return_min_clearance_m`), `collision_free`에 포함(Human 검토 요청). 펜던트 경유점 표(`pendant_waypoints.csv`)와 GUI 재생 버튼 추가 — 경유점 사이는 관절 공간 직선.
- 기본 관절 속도 3°/s, 가속 3°/s²(Human: 항상 가장 느리게).
- 미해결: real_cam에서 큐브가 보이지 않음 — 카메라가 프레임 기둥 바로 옆이라 시야 왼쪽 절반이 기둥에 가려지고 큐브(1.05, 0)가 시야 가장자리(축에서 ~41°)임. FoundationPose가 엉뚱한 pose를 반환해 접근 결과 무효(큐브 오차 1077 mm). 실제 프레임/큐브 위치 실측 필요.
- 작성자: Claude

## 2026-10-01 6D pose → 로봇 ROS2 파이프라인: 드라이런까지 (Claude, Human 선택)
- Human 질문: "6D pose 좌표값을 로봇에 ROS2로 보내는 파이프라인". 초기 규칙(실기 이동/ROS2 명령 금지)이 있어 범위를 확인 → Human이 **드라이런까지** 선택(이동 명령은 만들지 않음).
- 구현: `calibration/real_pose_dryrun.py`(conda) — `real_time_project/main.py`가 UDP 5005로 보내는 `T_cam_object` 수신 → `T_base_object = T_base_cam @ T_cam_object`(실기 hand-eye 결과 기본) → 안정성 판정(8개 연속 3 mm/3°) → 상식 검사(거리·높이·기울기) → top-down 접근 자세 IK → work-cell 충돌 검사(margin 0.10 m, 카메라 몸체 박스 포함, 실측 `--cell_json`) → RRT 직선 경유점 표(접근+복귀) → `data/real_dryrun/<time>/plan_NNN.{json,csv}` 저장, UDP 5010 전달. `calibration/ros2_dryrun_publisher.py`(시스템 python) — `/sixd/dryrun/{target_pose,target_joints_deg,waypoints_deg,status}`만 발행. **로봇 제어 토픽/서비스/`/dsr01`은 사용하지 않음.** 현재 관절은 `ros2_flange_udp.py`(UDP 5006, 읽기 전용)로 받아 시작 자세로 사용, 없으면 all-zero.
- 검증: 합성 pose(큐브 (1.05, 0.10) 기준)로 end-to-end 시험 — ACCEPTED, 접근/복귀 직선 최소 clearance 125 mm, ROS2 `/sixd/dryrun/status`와 `target_joints_deg` 수신 확인. 실제 D455/FoundationPose/로봇은 연결하지 않음.
- 리스크/가정: 프레임·기둥 위치는 아직 가정(`data/real_cell_template.json`에 실측 기입 필요). `T_base_cam`은 줄자 대조 전. 실기 depth/FoundationPose 정확도 미검증. 관절 한계 검사는 아직 없음(경유점 표를 사람이 확인).
- 작성자: Claude

## 2026-10-01 펜던트 경유점 단순화 (Claude, Human 요청)
- Human 요청: 경로를 point-to-point 몇 구간으로 나눠 안전하게. 결정: `CollisionChecker.simplify` — 플래너 경유점(13~17개)을 그리디로 줄여, 각 구간이 직선 관절 이동으로 **margin×1.15(115 mm) 이상** 유지하고 **한 관절 최대 45° 이하**가 되게 함(건너뛸 수 없는 점은 유지). 적용: `real_pose_dryrun.py`(`--max_segment_deg`), sim 펜던트 표/재생 버튼.
- 측정(큐브 (1.05, 0.10), 시작 영점): 구간 수 vs 직선 최소 clearance — margin 100 mm 요구: 6/5 moves 103 mm(45°/…); 115 mm 요구: 접근 6 moves 117 mm, 복귀 5 moves 125 mm. 3 moves(60°+)는 108 mm로 여유가 줄고 구간 하나가 57° 이상 돌아 육안 확인이 어려움 → 채택하지 않음. 이 기준(45°, 1.15배)은 내가 정한 값이며 Human 검토 요청.
- 작성자: Claude

## 2026-10-01 규칙 부분 해제: 감독형 실기 point-to-point 이동 스크립트 (Claude, Human 선택)
- Human이 "point-to-point로 무빙 실행, 무조건 가장 느리게"를 요청, 실행 대상 질문에 **실제 로봇(규칙 해제 필요)**을 선택 → 초기 규칙(실기 이동/ROS2 명령 금지)을 이 범위에 한해 해제한 것으로 기록. 범위: 감독형·최저속·한 점씩 확인 이동. 그 외 자동/연속 이동은 여전히 금지.
- 구현: `calibration/ros2_pendant_mover.py`(시스템 python) — 드라이런 plan의 한 구간(approach/return)을 `MoveJoint`로 **경유점마다 사람이 `go`를 입력해야만** 이동. `--execute` 없으면 검사만(ROS 불필요). 안전장치: 속도·가속 기본 3, 하드 한도 5 deg/s; plan accepted·최신(기본 30분)·구간당 한 관절 ≤45°·URDF 관절 한계 2° 여유·현재 관절이 직전 경유점과 ≤3°·도착 오차 ≤1°; 제어권 거부 시 중단(재시도 없음); Ctrl+C = quick stop 요청; 시작 선언문 입력. **어시스턴트는 이 스크립트를 로봇에 대해 실행하지 않음(사람이 실행).**
- 검증: 로봇 연결 없이 검사 모드만 시험 — 정상 plan 승인 표(접근 6 moves) 출력, 속도 10 deg/s·미승인 plan·누락 구간 거부 확인. 실제 이동은 시험하지 않음. 메모: 시작 자세는 plan 기준(영점)과 현재 관절이 맞아야 함.
- 리스크(실행 전 Human 확인 필요): (1) 큐브 pose가 실행 중 크게 변동(17:29–17:32 사이 x 1.08→0.92, y 0.02→0.25, yaw −9→−26°) — 추적 불안정 또는 큐브 이동. 줄자 실측과 대조 전 실행 금지. (2) 프레임·기둥은 가정값, 실측 필요. (3) 큐브 높이 z 0.01 m(예상 0.0285). (4) 제어권(`Access control deny`)은 펜던트 쪽 — Human이 이전해야 함. (5) 비상정지 손 닿는 곳, 주변 인원/물체 확인.
- 작성자: Claude

## 2026-10-01 높이 보정 변환을 기본값으로, 이동 속도 2 deg/s (Claude, Human 지시)
- Human 지시: 보정변환(높이 보정)을 기본값으로 저장. `real_pose_dryrun.latest_t_base_cam()`이 최신 세션의 `T_base_cam_zfix.json`(기본 해에서 카메라 z만 줄자값 0.115 m로 교체; 원본 `T_base_cam.json`은 보존)을 우선 사용. 근거: 줄자 렌즈 높이 11.5 cm vs 계산 9.2 cm, 큐브 중심 z 예측 7 mm vs 예상 28.5 mm가 같은 방향·크기로 어긋남 → zfix로 큐브 z 30.4 mm. 단 줄자 한 값(±5–10 mm) 의존, 수평 오차는 눈 맞춤 한계로 1–2 cm 미해결.
- Human 지시 "속도 무조건 가장 느리게" → `ros2_pendant_mover.py` 기본 속도/가속 3 → **2 deg/s, 2 deg/s²**(하드 한도 5 유지).
- 새 plan 시도(큐브 `[0.921, 0.305, 0.030]`): 가정 프레임과 0 mm로 **거부**(실제 프레임 미실측). 큐브를 중앙(y≈0.1)으로 옮기거나 프레임 실측 필요.
- 작성자: Claude

## 2026-10-01 프레임 실측 반영 (Claude, Human 실측)
- Human 실측(로봇 base 중심 기준): 기둥 x = 0.85 m(가까운 쪽), 1.65 m(먼 쪽), y = ±0.56 m(좌우 대칭, Human 확인). `data/real_cell_measured.json` — `frame_centre_xy [1.25, 0]`, `frame_size [0.84, 1.16]`(기둥 중심선 간격 + 프로파일 40 mm), 높이 1.20 m, 테이블 두께 0.10 m. `workcell.frame_layout`에 `frame_centre_xy` 옵션 추가. **가정 유지**: 프로파일 40 mm, 로봇-테이블 가장자리 거리 0.30 m(원 가정; 먼 쪽 기둥 x=1.65 m는 이 테이블 모델 가장자리 1.50 m 밖), 레일 배치.
- 결과: `real_pose_dryrun.py --cell_json data/real_cell_measured.json` — 큐브 `[1.0406, 0.0474, 0.0289]`(높이 보정으로 z 28.9 mm), 접근 3 moves/복귀 3 moves, 직선 구간 최소 clearance 120 mm, ACCEPTED. plan `calibration/data/real_dryrun/20261001_195225/plan_001.json`. 로봇 현재 관절 영점(읽기 전용 확인). `ros2_pendant_mover.py` 검사 모드 접근/복귀 모두 CHECK OK(속도 2 deg/s). 이동 명령 미실행.
- 작성자: Claude

## 2026-10-01 calibration/ 정리 (Claude, Human 요청)
- 요청: 핵심(ROS 통신, YOLOE pose, 동작 명령)만 남기고 sim/테스트 자료 정리. 결정: **삭제하지 않고** `~/pan/6D_pose_archive/calibration_2026-10-01/`(저장소 밖)로 이동. 보호 파일 sha256(`eye_to_hand/result.json` 847d6d32…, `T_base_cam.txt` c8b0ff63…, `handeye_samples.json` 3fa7e471…) 이동 후 동일 확인.
- 유지: 실기 파이프라인 18개 파일(인식 `real_time_project/main.py` + `calibration/`: hand-eye 수집·검증, ROS2 읽기/발행/감독형 이동, 드라이런 계획, 기구학, 작업 셀, 궤적, `test_core.py`), 데이터 `real_handeye/`(2 세션), `real_cell_*.json`, `real_d455_intrinsics.json`, 충돌 샘플 캐시, 성공 계획 1개.
- 이동: Isaac scene/sim 파이프라인·FoundationPose sim·시험, 수기 입력 캡처(`capture_handeye.py`)와 **자동 이동 스크립트 `ros2_run_waypoints.py`**, sim 캘리브레이션/USD(약 360 MB), 실패 샘플/분석, 사진, 이전 계획.
- 코드 변경: 순수 기구학을 `kinematics.py`로 분리(`Kinematics`, `cube_yaw_deg`, `grasp_frames`, `DEFAULT_URDF`), `workcell.py`에서 Isaac/카메라 배치 코드 제거(`frame_layout`은 박스 목록만 반환), `FRAME_CELL` 기본값을 실측값으로(여유 0.10 m). 시험 `test_core.py` 12개 통과(영점 FK=실기 flange, IK 왕복, top-down 자세, hand-eye 복원, 실측 프레임 배치, 계획 수락/거부, mover 거부 규칙·기본 속도 ≤2 deg/s, 궤적 한계). 각 스크립트 `--help`/구문 확인. 커밋/푸시는 하지 않음(Human 확인 후).
- 작성자: Claude
