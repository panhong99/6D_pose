Reference link: https://github.com/NVlabs/FoundationPose/tree/main

## 머스타드 데모 실행 및 ROS2 Pose 통신

FoundationPose의 머스타드 데모 데이터에서 물체의 포즈를 추정하고 ROS2 토픽으로 발행합니다. D455 카메라 없이 실행가능

### 실행 환경

- FoundationPose 실행 환경 및 모델 가중치
- 머스타드 데모 데이터
- ROS2 Jazzy
- 빌드된 `foundationpose_bridge` 패키지

아래 명령의 `/home/panhong/pan`은 자신의 프로젝트 경로로 변경합니다. ROS2 터미널 1·2는 conda 환경을 활성화하지 않은 상태에서 실행가능

### 1. ROS2 브리지 실행 — 터미널 1

```bash
source /opt/ros/jazzy/setup.bash
source /home/panhong/pan/ros2_ws/install/setup.bash

ros2 run foundationpose_bridge udp_pose_bridge
```

브리지는 UDP로 전달받은 포즈를 `/foundationpose/pose` 토픽으로 발행합니다.

### 2. Pose 수신 확인 — 터미널 2

```bash
source /opt/ros/jazzy/setup.bash

ros2 topic echo /foundationpose/pose
```

데모 실행 후 위치와 회전 정보가 출력됩니다.

### 3. 머스타드 데모 실행 — 터미널 3

```bash
conda activate foundationpose
cd /home/panhong/pan/FoundationPose

python run_demo.py --publish_pose --debug 1 \
  --debug_dir /tmp/foundationpose_demo_ros_kimm
```

첫 프레임에서는 mask를 사용해 초기 포즈를 추정하고, 이후 프레임에서는 이전 포즈를 기반으로 추적합니다. 처리된 각 프레임의 포즈를 ROS2 브리지로 전달합니다.

| 옵션 | 설명 |
|---|---|
| `--publish_pose` | UDP를 통해 ROS2 브리지로 포즈 전달 |
| `--debug 1` | 추정 결과를 영상에 표시 |
| `--debug 0` | 영상 창 없이 실행 |
| `--max_frames 10` | 처음 10개 프레임만 처리. 기본값 `0`은 전체 처리 |
| `--debug_dir` | 결과 저장 경로. 기존 데모 동작에 따라 실행 시 내부 파일이 삭제되므로 전용 경로 사용 |

### 출력 메시지

- 토픽: `/foundationpose/pose`
- 타입: `geometry_msgs/msg/PoseStamped`
- 기본 좌표계: `camera_color_optical_frame`
- `pose.position`: 카메라 좌표계 기준 CAD 원점의 위치, 단위는 미터
- `pose.orientation`: CAD 좌표계의 회전, 쿼터니언 `(x, y, z, w)`
- `header.stamp`: 데모 프레임을 읽은 시각이며 원본 촬영 시각은 아님

이 포즈는 로봇 베이스 좌표계 기준이 아닙니다. 로봇 제어에 사용하려면 카메라와 로봇 사이의 좌표 변환이 추가로 필요합니다.

### 데이터 흐름

```text
머스타드 데모 RGB + Depth + K + 초기 Mask + CAD
    ↓
run_demo.py: register() / track_one()
    ↓
pose_sender_kimm.py: 4×4 포즈 행렬 송신
    ↓ UDP
udp_pose_bridge.py: PoseStamped 변환 및 발행

## Demo

https://github.com/user-attachments/assets/https://github.com/NVlabs/FoundationPose/issues/415#issue-5394696207
    ↓
/foundationpose/pose
```

실행을 종료하려면 각 터미널에서 `Ctrl+C`를 누릅니다.
