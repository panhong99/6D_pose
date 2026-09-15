# FoundationPose 데모 → ROS2 디버깅

## 구성과 입력 계약

`run_demo.py` → `pose_sender_kimm.py` → UDP localhost:5005 →
`foundationpose_bridge/udp_pose_bridge.py` → `/foundationpose/pose`

- FoundationPose는 conda의 Python에서, ROS2 브리지는 ROS Jazzy의 시스템 Python에서 실행한다.
- 기존 `udp_pose_bridge` 실행 엔트리와 메시지 검증을 재사용한다.
- 발행 메시지는 `geometry_msgs/msg/PoseStamped`, QoS는 reliable/volatile, depth 10이다.
- 위치는 미터, 회전은 quaternion `(x, y, z, w)`이다.
- pose는 원본 CAD mesh 좌표 → 카메라 optical 좌표 변환이다. 카메라 optical 좌표는 x 오른쪽, y 아래, z 전방이다.
- 기본 frame_id는 `camera_color_optical_frame`; 데모에서는 이 이름으로 데이터셋 카메라 좌표를 표시하는 것이며 실제 D455 TF와 자동 연결되는 것은 아니다.
- `--frame_id`는 좌표계 이름만 바꾼다. 로봇 base 좌표로 변환하려면 별도 외부 보정과 TF 변환이 필요하다.
- 데모 timestamp는 프레임을 읽은 시점의 wall time이며 원본 촬영 시간이 아니다. 실시간 연결 시에는 해당 RGB-D 프레임의 촬영 시간을 전달해야 한다.
- UDP는 유실될 수 있다. frame_identifier는 UDP 패킷에는 있지만 PoseStamped 필드에는 포함되지 않는다.

## 실행

ROS 터미널은 conda를 활성화하지 않은 터미널을 사용한다.

터미널 1: 브리지 실행

```bash
source /opt/ros/jazzy/setup.bash
cd /home/panhong/pan/ros2_ws
colcon build --packages-select foundationpose_bridge --symlink-install
source install/setup.bash
ros2 run foundationpose_bridge udp_pose_bridge
```

터미널 2: 수신 확인 (데모 실행 전에 시작)

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic echo /foundationpose/pose
```

터미널 3: 실제 데모 추론 및 발행

```bash
conda activate foundationpose
cd /home/panhong/pan/FoundationPose
python run_demo.py --publish_pose --debug 0 --max_frames 10 --debug_dir /tmp/foundationpose_demo_ros_kimm
```

`--max_frames 0`은 전체 프레임이다. `--debug 1`은 화면도 표시한다.
기존 데모는 debug_dir 내부 파일을 실행 시작 시 지우므로 전용 출력 경로를 사용한다.

GPU 없이 송신기→브리지→실제 ROS 구독자 확인:

```bash
source /opt/ros/jazzy/setup.bash
source /home/panhong/pan/ros2_ws/install/setup.bash
python3 /home/panhong/pan/ros2_ws/src/foundationpose_bridge/test/udp_pose_smoke.py
```

이 테스트는 알려진 행렬로 위치, 회전, 시간, frame_id 전달과 잘못된 패킷 거부를 확인한다. 추론 정확도 테스트는 아니다.

## 중단점과 코드 대응

현재 수정된 `FoundationPose/run_demo.py` 기준:

| 위치 | 확인할 값과 역할 |
|---|---|
| 39행 | CAD 파일 → trimesh |
| 56행 | mesh를 보관하는 FoundationPose 생성 |
| 76행 | 프레임의 RGB 및 depth 읽기 |
| 83행 | 첫 프레임의 mask 읽기 |
| 86행 | register 결과: 첫 pose |
| 101행 | track_one 결과: 다음 pose |
| 105행 | 두 분기 공통의 pose를 UDP로 전달 (추가) |
| 108행 | 같은 pose를 기존 방식으로 txt 저장 |
| 111행 | 시각화용 center_pose; ROS로 보내는 원본 pose와 구분 |

함수 호출 결과를 보려면 호출 다음 실행 줄에 중단점을 둔다.
첫 프레임에서는 87행, 추적 프레임에서는 104행에서 `pose`, `pose[:3, 3]`를 확인한다.
송신기 `PoseSender.send()`에서 packet을 보고, 별도의 ROS Python 디버그 세션에서
`UdpPoseBridge.receive()`의 `publisher.publish(msg)`에 중단점을 두고 msg를 확인한다.

알고리즘 내부는 `estimater.py`의 `register()` → `refiner.predict()` → `scorer.predict()` →
최고 점수 pose 반환 순으로 읽는다. `track_one()`은 이전 pose와 refiner를 사용한다.
DL forward는 `learning/training/predict_pose_refine.py:191` 및 `predict_score.py:194`이다.

이번 변경은 데모의 추론 분기를 유지하고 공통 반환값 직후 송신 호출을 추가했다.
txt 저장이나 시각화 자체를 ROS 메시지로 감싼 것이 아니다.
등록 실패 시 pose_last가 없는 상태로 발행·추적하지 않도록 확인을 추가했다.

## D455 실시간 경로

D455에서도 RGB, RGB에 정렬된 depth, 그 RGB 해상도의 K, 초기 타깃 mask,
실물 크기와 일치하는 미터 단위 CAD가 필요하다. CAD는 로딩 후 추적 중에도 렌더링에 사용한다.
mask를 매 프레임 새로 입력하지는 않지만 추적 실패 후 재등록에는 다시 필요하다.

데모와 실시간의 출력 정의는 같은 4x4 pose이다. 실제 수치는 카메라 위치와 물체 위치에 따라 달라지며,
깊이 노이즈, 동기화, 분할 품질, 캘리브레이션, 프레임 누락과 처리 지연에 따라 정확도도 달라진다.

기존 `foundationpose_bridge/node.py`의 `pose_node`는 RGB-D 구독 기반의 별도 경로다.
현재 이 파일이 import하는 `FoundationPose/run_live_d455.py`가 디스크에 없어 실행할 수 없다.
이번 데모→UDP→ROS2 경로는 그 파일에 의존하지 않는다. 실시간 카메라 통합을 완료한 상태는 아니다.
