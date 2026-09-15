# FoundationPose 실시간 사용 가이드

이 저장소는 FoundationPose를 학습하지 않고, 준비된 모델과 CAD를 이용해 D455 RGB-D 카메라에서 6D pose를 추정하고 ROS2로 전달하기 위한 실행 코드입니다.

## 1. 환경 준비

```bash
cd /home/panhong/pan/FoundationPose
conda env create -f environment.yml
conda activate foundationpose
bash build_all.sh
pip install -U pyrealsense2 ultralytics
```

이미 `foundationpose` 환경이 있으면 환경 생성은 생략합니다. D455 연결 확인:

```bash
python -c "import pyrealsense2 as rs; print(rs.context().query_devices().size())"
```

`1` 이상이면 카메라가 인식된 것입니다.

## 2. ROS2 bridge 빌드

```bash
cd /home/panhong/pan/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select foundationpose_bridge
```

## 3. ROS2 pose 확인

터미널 1:

```bash
source /opt/ros/jazzy/setup.bash
source /home/panhong/pan/ros2_ws/install/setup.bash
ros2 run foundationpose_bridge udp_pose_bridge
```

터미널 2:

```bash
source /opt/ros/jazzy/setup.bash
source /home/panhong/pan/ros2_ws/install/setup.bash
ros2 topic echo /foundationpose/pose
```

## 4. D455 + DINO + SAM2 + FoundationPose

터미널 3:

```bash
conda activate foundationpose
cd /home/panhong/pan/FoundationPose
python real_time_project/real_time_v1_kimm.py \
  --mesh_file /home/panhong/pan/FoundationPose/demo_data/077_rubiks_cube/google_16k/textured_57mm.obj \
  --mesh_scale 1.0 \
  --prompt "rubik's cube" \
  --validation_interval 0
```

처음에는 DINO가 bbox를 찾고 SAM2가 mask를 생성합니다. 이후에는 FoundationPose와 Cutie/Kalman이 추적합니다. `s`는 수동 재탐색, `q` 또는 `ESC`는 종료입니다. 자동 재탐색은 `--auto_recovery`를 추가합니다.

## 5. D455 + YOLOE + FoundationPose

YOLOE-seg는 text prompt로 bbox와 mask를 한 번에 생성합니다.

```bash
conda activate foundationpose
cd /home/panhong/pan/FoundationPose
python real_time_project/real_time_v1_yolo.py \
  --mesh_file /home/panhong/pan/FoundationPose/demo_data/077_rubiks_cube/google_16k/textured_57mm.obj \
  --mesh_scale 1.0 \
  --prompt "rubik's cube" \
  --yoloe_checkpoint /home/panhong/pan/FoundationPose/real_time_project/detector_comparison/yoloe-11s-seg.pt
```

ROS2 bridge와 `ros2 topic echo /foundationpose/pose`는 위와 동일합니다.

## 6. 모델과 폴더 구조

Rubik’s Cube CAD는 `demo_data/077_rubiks_cube/google_16k/textured_57mm.obj`이며 `--mesh_scale 1.0`을 사용합니다. FoundationPose weights는 `weights/` 아래에 준비합니다. 모델 weights, Hugging Face cache, debug 결과는 Git에 올리지 않습니다.

```text
FoundationPose/
├── real_time_project/
│   ├── real_time_v1_kimm.py             # DINO + SAM2 실행
│   ├── real_time_v1_yolo.py             # YOLOE 실행
│   ├── d455_source_kimm.py              # D455 RGB-D 입력/정렬
│   ├── camera_rectify_kimm.py           # 왜곡 보정
│   ├── pose_tracker_kimm.py             # FoundationPose + Cutie/Kalman
│   ├── recovery_tracker_kimm.py         # 탐색/추적 상태 관리
│   └── detector_comparison/
│       ├── pipelines/pipeline_yoloe.py
│       ├── pipelines/pipeline_grounding_dino_sam2.py
│       └── yoloe-11s-seg.pt
├── demo_data/
├── weights/
├── estimater.py
├── pose_sender_kimm.py
└── environment.yml
```

ROS2 bridge:

```text
/home/panhong/pan/ros2_ws/src/foundationpose_bridge/
```

발행 토픽은 `/foundationpose/pose`이며 타입은 `geometry_msgs/msg/PoseStamped`입니다. Pose는 카메라 좌표계 기준입니다.
