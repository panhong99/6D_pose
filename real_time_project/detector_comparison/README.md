# 루빅큐브 detector + segmenter 비교

같은 RGB 프레임과 text prompt로 `YOLOE`와 `Grounding DINO + SAM2`를 순차 실행합니다. YOLOE는 segmentation checkpoint의 한 번의 이미지 forward로 bbox와 mask를 얻고, Grounding DINO는 bbox를 검출한 뒤 Meta 공식 SAM2 image predictor에 box prompt로 전달합니다. 영상도 프레임마다 독립 추론하며 FoundationPose 연동과 tracking은 포함하지 않습니다.

```text
detector_comparison/
├── pipelines/
│   ├── base.py                         # 공통 인터페이스, 좌표/입력 검증, 동기화 타이머
│   ├── pipeline_yoloe.py
│   ├── pipeline_grounding_dino_sam2.py
│   └── artifacts.py                    # 결과와 시각화 저장
├── benchmark.py
├── install_sam2.sh
└── requirements.txt
```

## 설치

아래 명령은 이 디렉터리에서 실행합니다. 기존 `ultralytics`, `transformers`가 설치된 환경을 활성화해 사용할 수 있습니다. 이 작업공간의 `foundationpose` conda 환경을 사용하는 경우 먼저 `conda activate foundationpose`를 실행합니다. 별도 환경은 Python 3.10 이상으로 만듭니다.

```bash
cd FoundationPose/real_time_project/detector_comparison

# 새 환경을 만들 경우에만 실행
python3.11 -m venv .venv
source .venv/bin/activate

# torch/torchvision은 먼저 장비에 맞는 조합으로 설치한 뒤 나머지를 설치
python -m pip install -r requirements.txt

# YOLOE text prompt의 CLIP tokenizer를 명시적으로 준비
python -m pip install git+https://github.com/ultralytics/CLIP.git

# SAM2: 이미 정상 설치되어 있으면 재사용
bash install_sam2.sh

python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

SAM2는 Python ≥3.10, PyTorch ≥2.5.1 및 이에 맞는 torchvision ≥0.20.1을 요구합니다. CPU/CUDA용 PyTorch 설치 명령은 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에서 선택합니다. `requirements.txt`는 최소 버전 목록이며 기존 환경의 버전을 완전히 고정하지 않습니다. SAM2 설치도 필요한 경우 PyTorch를 업그레이드할 수 있으므로 기존 FoundationPose 환경을 유지해야 한다면 별도 환경을 사용합니다. [SAM2 공식 설치 안내](https://github.com/facebookresearch/sam2#installation)

SAM2 설치 스크립트는 공식 Git URL을 통한 pip 설치를 먼저 시도하고, 실패하면 `git clone`과 `pip install -e`로 재시도합니다. 설치 방법, Python 실행 파일, clone 경로, Git revision을 직접 지정할 수도 있습니다.

```bash
bash install_sam2.sh --help
bash install_sam2.sh --method git
bash install_sam2.sh --method editable --repo_dir ./sam2_src
bash install_sam2.sh --python ./.venv/bin/python --method editable \
  --repo_dir ./sam2_src --revision main

# 선택: CUDA toolkit/nvcc와 현재 PyTorch의 CUDA 버전이 맞는 장비
bash install_sam2.sh --method editable --repo_dir ./sam2_src --build_cuda
```

기본값 `SAM2_BUILD_CUDA=0`은 선택적 CUDA 후처리 extension 빌드를 생략합니다. 이 설정에서도 CUDA로 모델 추론이 가능하며, extension을 이용하는 작은 구멍/작은 영역 후처리는 생략됩니다. `--build_cuda`는 빌드 오류를 표시하고 실패 처리합니다. 기존 clone은 자동 reset/pull하지 않으며, `--revision` 변경 전 로컬 수정이 있으면 중단합니다. 재현 실험에서는 `main` 대신 특정 commit SHA를 지정할 수 있습니다. [SAM2 설치 FAQ](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md)

처음 실행할 때 YOLOE checkpoint와 `mobileclip_blt.ts` text encoder, Grounding DINO 모델/processor, SAM2 checkpoint가 내려받아집니다. YOLOE text encoder는 실행 디렉터리에 저장될 수 있습니다. 따라서 첫 준비 실행에는 네트워크가 필요하고, offline 실행 전에는 사용할 환경과 실행 디렉터리에서 한 번 추론해 캐시를 준비합니다. [Ultralytics YOLOE 설치 안내](https://docs.ultralytics.com/models/yoloe/#installation-and-requirements)

## 빠른 비교 실행

```bash
# 이미지/영상이 있는 폴더를 재귀 탐색; 기본 prompt 3개를 각각 비교
python benchmark.py --input_dir ./test_images --output_dir ./results

# 직접 지정한 prompt만 비교; 각 --prompt는 별도의 실험
python benchmark.py --input_dir ./test_images --output_dir ./results_prompts \
  --prompt "rubik's cube" --prompt "colorful cube" \
  --prompt "a colorful Rubik's cube" --device auto

# 단일 영상: 원본 frame index 0, 5, 10, ...을 샘플링
python benchmark.py --video ./test_videos/cube.mp4 --output_dir ./results_video \
  --video_stride 5 --max_frames 100 --prompt "rubik's cube"

# 한 파이프라인만 실행하거나 CPU 강제 지정
python benchmark.py --input_dir ./test_images --output_dir ./results_yoloe \
  --pipelines yoloe --device cpu --prompt "rubik's cube"
```

기본 prompt는 `rubik's cube`, `colorful cube`, `toy cube`입니다. `--prompt`를 하나라도 주면 기본 목록을 대체합니다. 여러 prompt를 합쳐 한 번 추론하거나 결과를 ensemble하지 않으므로 prompt별 검출 특성을 비교할 수 있습니다. 객체가 여러 개면 각 파이프라인은 threshold를 통과한 유효 bbox 중 detector confidence가 가장 높은 하나를 선택합니다.

| 옵션 | 기본값 / 의미 |
|---|---|
| `--input_dir`, `--video` | 둘 중 하나 지정; 폴더는 이미지와 영상을 재귀 탐색 |
| `--output_dir` | 새 디렉터리 또는 비어 있는 디렉터리; 이전 실험과 섞이지 않도록 기존 결과가 있으면 중단 |
| `--pipelines` | `yoloe grounding_dino_sam2`; 하나만 선택 가능 |
| `--device` | `auto`: CUDA가 있으면 CUDA, 없으면 CPU; `cpu`, `cuda:0` 등 직접 지정 가능 |
| `--warmup` | `1`: 파이프라인/prompt별 첫 프레임으로 측정 전 warmup |
| `--video_stride` | `1`: 영상의 샘플링 간격 |
| `--max_frames` | 전체 입력에서 처리할 프레임 수 상한; `0` 또는 생략하면 제한 없음 |
| `--blur_tag` | `blur`: 파일명에서 찾을 대소문자 무관 문자열; 여러 번 지정 가능 |
| `--tags_csv` | 영상 특정 프레임 등에 붙일 수동 태그 CSV |
| `--yoloe_checkpoint` | `yoloe-11s-seg.pt`; text prompt 지원 `yoloe-*-seg.pt` |
| `--yoloe_conf`, `--imgsz` | `0.15`, `640` |
| `--dino_model` | `IDEA-Research/grounding-dino-tiny`; `.../grounding-dino-base`로 변경 가능 |
| `--box_threshold`, `--text_threshold` | `0.3`, `0.25` |
| `--sam2_model` | `facebook/sam2.1-hiera-tiny`; 로컬 checkpoint 미지정 시 사용 |
| `--sam2_checkpoint` | 선택: 로컬 SAM2 checkpoint 파일 |
| `--sam2_config` | `configs/sam2.1/sam2.1_hiera_t.yaml`; 로컬 checkpoint와 맞는 config |

YOLOE는 text prompt용 segmentation checkpoint를 사용합니다. `*-seg-pf.pt`는 prompt-free 모델이므로 이 비교에 사용할 수 없습니다. SAM2 config는 설치된 SAM2 패키지 안의 config 이름이며, checkpoint 크기/세대와 일치해야 합니다. `--sam2_config`를 바꾸려면 `--sam2_checkpoint`도 지정합니다. [YOLOE prompting 모드](https://docs.ultralytics.com/models/yoloe/#choosing-a-prompting-mode), [SAM2 모델 및 config](https://github.com/facebookresearch/sam2#model-description)

```bash
# DINO-base + 직접 내려받은 SAM2.1 tiny checkpoint 예시
mkdir -p checkpoints
curl -L https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt \
  -o ./checkpoints/sam2.1_hiera_tiny.pt
python benchmark.py --input_dir ./test_images --output_dir ./results_base \
  --dino_model IDEA-Research/grounding-dino-base \
  --sam2_checkpoint ./checkpoints/sam2.1_hiera_tiny.pt \
  --sam2_config configs/sam2.1/sam2.1_hiera_t.yaml
```

## 모션 블러 태그

기본적으로 파일명에 `blur`가 있으면 `blur`, 나머지는 `normal` 그룹입니다. `CUBE_BLUR_001.jpg`도 포함하며 디렉터리 이름 자체를 blur 판정에 쓰지 않습니다. 영상 파일명이 `cube_blur.mp4`이면 그 영상의 모든 샘플 프레임이 blur로 분류됩니다. 아래처럼 태그 문자열을 추가할 수 있습니다.

```bash
python benchmark.py --input_dir ./test_images --output_dir ./results_tags \
  --blur_tag blur --blur_tag motion --tags_csv ./tags.csv
```

수동 태그 파일은 `source,frame_index,tag` 열을 갖습니다. `source`는 `--input_dir` 기준 상대 경로이며, 단일 `--video` 입력은 해당 영상 파일명을 사용합니다. `frame_index`는 stride 적용 전 0부터 시작하는 원본 영상 프레임 번호입니다. 빈 값이면 해당 파일 전체에 태그를 적용합니다. `tag`는 `blur` 또는 `normal`입니다. 개별 프레임 태그가 파일 전체 태그보다 우선하며, 수동 태그가 파일명 판정보다 우선합니다.

```csv
source,frame_index,tag
cube_01.jpg,,blur
videos/cube.mp4,15,blur
videos/fast_cube.mp4,,blur
videos/fast_cube.mp4,20,normal
```

blur 문자열/태그는 사용자가 지정한 그룹 표시이며 실제 흐림 정도를 자동으로 측정하지 않습니다. 따라서 태그 없는 프레임은 `normal` 그룹에 집계되지만 선명함이 검증되었다는 의미는 아닙니다.

## 출력과 지표 해석

- `per_frame.csv`, `per_frame.json`: 프레임/pipeline/prompt별 `success`, `detection_success`, `has_bbox`, `has_mask`, `bbox`, `confidence`, `latency_ms`, 태그, 결과 파일 경로 및 `error`.
- `summary.csv`, `summary.json`: pipeline/prompt별 `all`, `blur`, `normal` 그룹의 프레임 수, 성공률, 오류 수, latency 평균/중앙값/p95. 성공한 프레임만의 latency도 별도로 집계합니다.
- `run_config.json`: 실제 실행 옵션, 장비/패키지 정보 및 모델 로딩·prompt 준비·warmup 시간 등 측정 조건.
- `input_errors.json`: 이미지/영상 읽기 오류. 읽기에 실패한 입력을 검출 실패로 바꾸어 집계하지 않습니다.
- `frames/`, `frames.json`: 샘플링한 원본 PNG와 원본 파일/프레임 번호/태그 대응표.
- `masks/`, `overlays/`, `comparisons/`: mask PNG, bbox+mask overlay, prompt별 두 파이프라인 side-by-side 비교 이미지. mask 파일은 원본 크기의 0/255 단일 채널 PNG이며, Python API의 mask 값은 0/1 binary입니다.

`success=True`는 **유효 bbox와 비어 있지 않은 mask를 둘 다 얻었다**는 뜻입니다. detection success는 bbox가 있는지를 따로 기록하므로 bbox만 검출된 경우와 mask까지 생성된 경우를 구별할 수 있습니다. 정답 annotation과 비교하지 않으므로 성공률은 Rubik's cube를 정확히 찾은 비율이나 mask IoU/mAP가 아닙니다. Overlay를 확인하고, 정확도 평가가 필요하면 별도 GT bbox/mask를 준비해야 합니다. 두 detector의 confidence는 서로 보정된 확률이 아니며 같은 숫자를 같은 신뢰도로 해석할 수 없습니다. DINO+SAM2의 `confidence`는 DINO 검출 점수입니다.

`summary`의 `success_rate`와 `detection_success_rate`는 0~1 비율이며, 분모 `num_frames`에는 실행 오류가 난 프레임도 포함합니다. `error_count`를 함께 확인합니다. `latency_*`는 오류 없이 끝난 추론(검출 miss 포함)을 집계하고 `successful_latency_*`는 bbox+mask에 성공한 추론만 집계합니다. 표본이 없는 집계는 JSON에서 `null`입니다.

모델은 한 파이프라인씩 메모리에 올려 순차 실행합니다. 입력 영상은 한 번 샘플링해 원본 PNG를 저장한 뒤 두 파이프라인이 같은 프레임을 읽습니다. 디코딩 차이를 줄이고 긴 영상을 모두 RAM에 담지 않지만, 저장 공간은 샘플 수에 비례해 늘어납니다.

`latency_ms`는 CUDA 동기화를 포함한 `infer()` wall time입니다. 입력 전처리, 모델 forward, bbox/mask 후처리와 결과 CPU 변환을 포함하고 파일 읽기/저장, overlay, 모델 로딩은 제외합니다. 기본 warmup이 켜져 있으면 prompt 준비와 첫 추론도 측정 전에 수행하고 별도 기록합니다. `--warmup 0`에서는 첫 추론에 lazy 준비 비용이 포함될 수 있습니다. 영상은 SAM2 temporal propagation 없이 각 프레임에서 DINO와 SAM2를 새로 수행합니다.

DINO가 bbox를 찾지 못하면 SAM2 호출을 생략하므로 miss가 많은 실험의 전체 평균 latency가 낮아질 수 있습니다. 전체 정상 실행 latency와 성공 프레임 latency를 함께 비교합니다. 프레임 추론 예외는 `error`와 `latency_ms=null`로 남기며 일반적인 검출 miss와 구분합니다. 모델 로딩 오류가 나면 해당 파이프라인의 행을 오류로 기록하고 다른 파이프라인은 계속 실행합니다. Warmup 오류도 별도 기록하며 이후 프레임 추론은 시도합니다. 종료 코드는 정상 완료 `0`, 입력 읽기/설정/추론/warmup 오류가 있는 결과 완료 `1`, 유효 입력 없음 또는 인자/출력 경로 오류 `2`입니다. 실제 장비의 성공률/latency 수치는 아직 측정하지 않았으며 테스트 이미지/영상에서 실행해야 합니다.

## 독립 파이프라인 실행과 Python API

```bash
python -m pipelines.pipeline_yoloe --image ./test_images/cube.jpg \
  --prompt "rubik's cube" --prompt "colorful cube" \
  --checkpoint yoloe-11s-seg.pt --conf 0.15 --imgsz 640 \
  --device auto --output_dir ./results_single_yoloe

python -m pipelines.pipeline_grounding_dino_sam2 --image ./test_images/cube.jpg \
  --prompt "rubik's cube" --prompt "colorful cube" \
  --dino_model IDEA-Research/grounding-dino-tiny \
  --sam2_model facebook/sam2.1-hiera-tiny \
  --device auto --output_dir ./results_single_dino
```

공통 입력은 RGB 순서의 `(H, W, 3)`, `np.uint8` 이미지입니다. OpenCV로 읽은 BGR 이미지는 먼저 RGB로 변환합니다. bbox는 원본 이미지 좌표의 `[x1, y1, x2, y2]`이며 mask는 원본 `(H, W)` 크기입니다.

```python
import cv2
from pipelines.base import DetectorPipeline
from pipelines.pipeline_yoloe import YOLOEPipeline
from pipelines.pipeline_grounding_dino_sam2 import GroundingDINOSAM2Pipeline

pipeline: DetectorPipeline = YOLOEPipeline(device="auto")
# 교체 시: pipeline = GroundingDINOSAM2Pipeline(device="auto")

bgr = cv2.imread("./test_images/cube.jpg")
if bgr is None:
    raise FileNotFoundError("./test_images/cube.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

prompt = "rubik's cube"
pipeline.prepare_prompt(prompt)  # 선택: 반복 사용할 prompt의 준비 비용을 먼저 지불
result = pipeline.infer(rgb, prompt)
# {
#   "success": bool,
#   "bbox": [x1, y1, x2, y2] 또는 None,
#   "mask": np.ndarray((H, W), binary) 또는 None,
#   "confidence": float,
#   "latency_ms": float,
# }
```

Grounding DINO의 transformers API는 [공식 모델 문서](https://huggingface.co/docs/transformers/model_doc/grounding-dino)를, SAM2 box prompt API는 [공식 image predictor 예제](https://github.com/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb)를 사용합니다.


## D455 실시간 실행

검출 구현은 `pipelines/pipeline_yoloe.py`와
`pipelines/pipeline_grounding_dino_sam2.py`로 분리되어 있습니다.
`live_realsense.py`는 D455 RGB 입력과 화면 표시를 공통으로 처리하고,
`--pipeline`으로 선택한 모델 하나만 로드합니다. 기존 `benchmark.py`는 저장된 이미지/영상 비교용입니다.

```bash
conda activate foundationpose  # 해당 환경을 사용하는 경우
cd ~/pan/FoundationPose/real_time_project/detector_comparison

# 1. YOLOE 실행 → q 또는 Esc로 종료
python live_realsense.py --pipeline yoloe --prompt "rubik's cube"

# 2. 위 실행을 종료한 후 Grounding DINO + SAM2 실행
python live_realsense.py --pipeline dino --prompt "rubik's cube"
```

- 왼쪽: 현재 RGB 영상. 오른쪽: 마지막 추론에 사용한 프레임과 그 프레임의 bbox/mask.
- 추론 중에는 다음 입력을 최신 프레임으로 교체하므로 오래된 입력이 계속 쌓이지 않습니다.
- 카메라는 기본 640×480, 30 FPS이며, `--width`, `--height`, `--fps`로 변경합니다.
  설정은 카메라가 지원하는 조합이어야 합니다. depth 스트림은 사용하지 않습니다.
- `infer FPS`는 완료된 추론 사이의 처리 빈도입니다. 카메라 FPS와 다릅니다.
  `age`는 해당 프레임을 프로그램에서 받은 뒤 경과한 시간이며 센서 timestamp 기반 지연은 아닙니다.
  첫 추론은 모델 워밍업 때문에 느릴 수 있습니다.
- `s`: 마지막 추론 결과의 overlay PNG, binary mask PNG(마스크가 있을 때), JSON을
  `live_captures/`에 저장합니다. 경로는 `--output_dir`로 변경합니다.
- `q`, Esc 또는 창 닫기: 종료. 다른 실행이 D455를 점유하고 있다면 먼저 종료하세요.
- `--device auto`가 기본이며 CUDA를 사용할 수 없으면 CPU로 실행합니다.
  GPU를 명시하려면 `--device cuda:0`, 특정 카메라는 `--serial SERIAL`을 사용합니다.
- YOLOE 옵션: `--yoloe_checkpoint`, `--yoloe_conf`, `--imgsz`.
  DINO/SAM2 옵션: `--dino_model`, `--box_threshold`, `--text_threshold`,
  `--sam2_model` 또는 `--sam2_checkpoint`와 `--sam2_config`.
- RealSense Python 바인딩(`pyrealsense2`)과 OpenCV GUI를 사용할 수 있는 데스크톱에서 실행합니다.
  가중치가 없다면 첫 실행에 다운로드가 필요합니다.

코드 작성 후 CLI 로딩만 확인했으며, 실제 카메라 실행 및 실시간 성능 측정은 아직 진행하지 않았습니다.
