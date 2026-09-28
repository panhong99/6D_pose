# CONTEXT: FoundationPose (real-time 6D pose)

> repo 위치: `/home/panhong/pan/FoundationPose`
> "(초안)"은 repo를 읽고 추정한 내용이므로 Human이 확인해야 한다. 비어 있는 항목은 Human에게 질문 후 채운다.

## 연구 목표
- (초안) D455 RGB-D에서 FoundationPose + detector(Grounding DINO+SAM2 / YOLOE) + Cutie/Kalman tracking으로 real-time 6D pose를 추정하고, ROS2 `/foundationpose/pose`(PoseStamped, 카메라 좌표계)로 Doosan 로봇 쪽에 전달하는 파이프라인의 inference 속도 개선
- (초안) 참고 repo: teal024/FoundationPose-plus-plus

## 현재 가설
- (비어 있음) bottleneck을 profiling으로 먼저 확인하기로 한 상태 (init vs tracking, hypothesis 개수, refine iteration, render 비용, detector/tracker 비용). 확정된 가설 없음

## 성공 기준
- (비어 있음) 목표 FPS / latency / pose 정확도 threshold 필요

## 제약 조건
- (초안) 환경: conda `foundationpose`, ROS2 jazzy, D455는 PC에만 연결, 로봇은 PC와 IP 랜선 연결
- (비어 있음) 사용 GPU 및 VRAM 한도

## 평가 방식
- (비어 있음) 속도: 어떤 구간을 어떻게 측정할지
- (비어 있음) 정확도: camera calibration으로 6D pose GT를 확보할 예정이라고만 알려져 있음. 지표(ADD/ADD-S 등)와 GT 절차 미정

## 건드리면 안 되는 것
- (초안, .gitignore/README 기준) `weights/`, `mobileclip_blt.ts`, `*.pt`, Hugging Face cache, debug 결과는 Git에 올리지 않음
- (비어 있음) 수정 금지 파일/디렉터리 (예: `estimater.py`, `Utils.py`, `mycpp/`, `learning/`가 upstream 원본 그대로여야 하는지)
