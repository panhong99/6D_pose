"""Text-prompted detector + segmenter pipelines used by main.py.

Both return the same dict from infer(): success, bbox [x1,y1,x2,y2], mask (H,W bool),
confidence, latency_ms; detect_bbox() is the box-only check used for pose validation.

  YOLOEPipeline             YOLOE-seg checkpoint: box and mask in one forward pass
  GroundingDINOSAM2Pipeline Grounding DINO box, then SAM2 box-prompted mask
                            (SAM2: pip install git+https://github.com/facebookresearch/sam2.git)
"""
import inspect
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch


class DetectorPipeline(ABC):
    def prepare_prompt(self, text_prompt: str) -> None:
        """Optional prompt preparation for a benchmark's untimed warmup phase."""

    @abstractmethod
    def infer(self, rgb_image: np.ndarray, text_prompt: str) -> dict:
        """Detect+segment the object named by text_prompt in one RGB frame.

        Input is uint8 RGB (H, W, 3). Output coordinates are original-image
        pixel xyxy, and mask is bool (H, W). Select one highest-scoring valid
        detection. success means a box AND a nonempty mask, not GT correctness.
        confidence is the detector score (not SAM's predicted mask IoU).
        latency_ms is synchronized wall time for infer, including preprocessing,
        prompt setup on a cache miss, inference, and CPU output conversion.
        Model construction/downloads and caller disk I/O are excluded.
        Runtime errors propagate; ordinary no-detection returns success=False.

        Returns:
            {
                "success": bool,
                "bbox": [x1, y1, x2, y2] or None,
                "mask": np.ndarray (H, W) bool/binary or None,
                "confidence": float,
                "latency_ms": float,
            }
        """
        raise NotImplementedError


def validate_input(rgb_image: np.ndarray, text_prompt: str) -> str:
    if not isinstance(rgb_image, np.ndarray):
        raise TypeError('rgb_image must be a numpy.ndarray')
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3 or min(rgb_image.shape[:2]) < 1:
        raise ValueError('rgb_image must have shape (H, W, 3) with H,W > 0')
    if rgb_image.dtype != np.uint8:
        raise ValueError('rgb_image must be uint8 RGB with values in [0, 255]')
    if not isinstance(text_prompt, str) or not text_prompt.strip().strip('.'):
        raise ValueError('text_prompt must be a nonempty object description')
    return text_prompt.strip()


def resolve_device(device=None) -> torch.device:
    if device is None or str(device) == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    selected = torch.device(device)
    if selected.type not in ('cuda', 'cpu'):
        raise ValueError('Supported devices: auto, cpu, cuda, cuda:N')
    if selected.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA was requested but torch.cuda.is_available() is False')
        if selected.index is not None and selected.index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device index does not exist: {selected}')
    return selected


def synchronize(device) -> None:
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


class InferenceTimer:
    def __init__(self, device):
        self.device = device
        self.elapsed_ms = 0.0

    def __enter__(self):
        synchronize(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        synchronize(self.device)
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        return False


def clip_box(box, height: int, width: int):
    xyxy = np.asarray(box, dtype=np.float64).reshape(-1)
    if xyxy.size != 4 or not np.isfinite(xyxy).all():
        return None
    xyxy[[0, 2]] = np.clip(xyxy[[0, 2]], 0, width)
    xyxy[[1, 3]] = np.clip(xyxy[[1, 3]], 0, height)
    if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
        return None
    return xyxy.tolist()


def make_result(bbox=None, mask=None, confidence=0.0, latency_ms=0.0) -> dict:
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError('Output mask must be two-dimensional')
        if not mask.any():
            mask = None
    return dict(success=bbox is not None and mask is not None,
                bbox=bbox, mask=mask, confidence=float(confidence),
                latency_ms=float(latency_ms))


# ---------------------------------------------------------------- YOLOE


class YOLOEPipeline(DetectorPipeline):
    """Single-target zero-shot segmentation with an official YOLOE checkpoint.

    The highest-confidence finite, nondegenerate box is returned. success
    additionally requires a nonempty mask in original image coordinates.
    Latency includes prompt setup when needed, preprocessing, model inference,
    and CPU result extraction; checkpoint loading in __init__ is excluded.
    One instance should be used sequentially because prompts mutate model state.
    """

    def __init__(
        self,
        checkpoint: str | Path = "yoloe-11s-seg.pt",
        device: str | torch.device | None = None,
        conf: float = 0.15,
        imgsz: int = 640,
    ):
        checkpoint = str(checkpoint)
        name = Path(checkpoint).name.lower()
        if "-pf" in name:
            raise ValueError(
                "Prompt-free YOLOE checkpoints cannot accept text prompts. "
                "Use a yoloe-*-seg.pt checkpoint, such as yoloe-11s-seg.pt."
            )
        if not name.startswith("yoloe-") or not name.endswith("-seg.pt"):
            raise ValueError(
                "Expected a text-prompted segmentation checkpoint named "
                "yoloe-*-seg.pt (or a path to one)."
            )
        if not np.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError("conf must be a finite number between 0 and 1.")
        if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0:
            raise ValueError("imgsz must be a positive integer.")

        self.device = resolve_device(device)
        self.checkpoint = checkpoint
        self.conf = float(conf)
        self.imgsz = imgsz
        try:
            from ultralytics import YOLOE
        except ImportError as exc:
            raise ImportError(
                "YOLOE requires a recent ultralytics release. "
                "Install it with: python -m pip install -U ultralytics"
            ) from exc

        self.model = YOLOE(checkpoint)
        if self.model.task != "segment":
            raise ValueError("The loaded YOLOE checkpoint does not support segmentation.")
        self.model.to(self.device)
        self._prompt: str | None = None
        self._prompt_embeddings: dict[str, torch.Tensor] = {}

    def prepare_prompt(self, text_prompt: str) -> None:
        """Cache text embeddings and select this prompt without running an image."""
        if not isinstance(text_prompt, str) or not text_prompt.strip():
            raise ValueError("text_prompt must be a nonempty string.")
        text_prompt = text_prompt.strip()
        if self._prompt == text_prompt:
            return
        with torch.inference_mode():
            if text_prompt not in self._prompt_embeddings:
                self._prompt_embeddings[text_prompt] = self.model.get_text_pe([text_prompt])
            self.model.set_classes([text_prompt], self._prompt_embeddings[text_prompt])
        self._prompt = text_prompt

    def detect_bbox(self, rgb_image: np.ndarray, text_prompt: str) -> dict:
        """Box-only watchdog check used by RecoveryTracker to validate the tracked pose.

        YOLOE produces the box and mask in one forward pass, so this costs the same
        as infer(); the mask is simply dropped.
        """
        result = self.infer(rgb_image, text_prompt)
        return dict(bbox=result["bbox"], confidence=result["confidence"],
                    latency_ms=result["latency_ms"])

    def infer(self, rgb_image: np.ndarray, text_prompt: str) -> dict:
        text_prompt = validate_input(rgb_image, text_prompt)
        height, width = rgb_image.shape[:2]
        bbox, mask, confidence = None, None, 0.0
        with InferenceTimer(self.device) as timer, torch.inference_mode():
            self.prepare_prompt(text_prompt)
            # Ultralytics assumes BGR for numpy sources, while our interface is RGB.
            bgr_image = np.ascontiguousarray(rgb_image[..., ::-1])
            predictions = self.model.predict(
                source=bgr_image,
                conf=self.conf,
                imgsz=self.imgsz,
                device=str(self.device),
                retina_masks=True,
                verbose=False,
                save=False,
            )
            if predictions:
                result = predictions[0]
                if result.boxes is not None and len(result.boxes):
                    scores = result.boxes.conf.detach().cpu().numpy()
                    boxes = result.boxes.xyxy.detach().cpu().numpy()
                    for index in np.argsort(-scores, kind="stable"):
                        score = float(scores[index])
                        if not np.isfinite(score) or not self.conf <= score <= 1.0:
                            continue
                        candidate = clip_box(boxes[index], height, width)
                        if candidate is None:
                            continue
                        bbox, confidence = candidate, score
                        if result.masks is not None and len(result.masks.data) > index:
                            raw_mask = result.masks.data[index].detach().cpu().numpy()
                            if raw_mask.shape != (height, width):
                                # Direct resizing would stretch letterbox padding into
                                # the scene. Require retina_masks' documented geometry.
                                raise RuntimeError(
                                    "YOLOE returned mask shape "
                                    f"{raw_mask.shape}, expected {(height, width)} with "
                                    "retina_masks=True. Upgrade ultralytics; padded "
                                    "masks cannot be resized directly."
                                )
                            if np.isfinite(raw_mask).all():
                                binary_mask = np.ascontiguousarray(raw_mask > 0.5)
                                if binary_mask.any():
                                    mask = binary_mask
                        break
        return make_result(
            bbox=bbox,
            mask=mask,
            confidence=confidence,
            latency_ms=timer.elapsed_ms,
        )


# ---------------------------------------------------------------- Grounding DINO + SAM2


DEFAULT_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
DEFAULT_SAM2_MODEL = "facebook/sam2.1-hiera-tiny"
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


class GroundingDINOSAM2Pipeline(DetectorPipeline):
    """Return the highest-confidence valid DINO box and its SAM2 mask.

    ``confidence`` is the DINO detection score, not SAM2's predicted mask IoU.
    ``latency_ms`` includes RGB preprocessing, both models, box/mask
    postprocessing and CPU transfer, with CUDA synchronization at both ends.
    Model construction and weight downloads are outside this interval.
    """

    def __init__(
        self, sam2_checkpoint=None, dino_model=DEFAULT_DINO_MODEL,
        sam2_config=DEFAULT_SAM2_CONFIG, device=None, box_threshold=0.3,
        text_threshold=0.25, sam2_model=DEFAULT_SAM2_MODEL,
    ):
        self.device = resolve_device(device)
        for name, value in (("box_threshold", box_threshold),
                            ("text_threshold", text_threshold)):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite number between 0 and 1")
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)

        checkpoint = None
        if sam2_checkpoint is not None:
            checkpoint = Path(sam2_checkpoint).expanduser()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"SAM2 checkpoint does not exist: {checkpoint}")
            if not sam2_config:
                raise ValueError("sam2_config is required with sam2_checkpoint")
        elif sam2_config not in (None, DEFAULT_SAM2_CONFIG):
            raise ValueError("A custom sam2_config requires sam2_checkpoint")

        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from sam2.build_sam import build_sam2
        except ImportError as exc:
            raise ImportError(
                "The official Meta SAM2 package or one of its dependencies is "
                "missing. From detector_comparison, run `bash install_sam2.sh` "
                "in this Python environment, then retry. Original error: "
                f"{exc}"
            ) from exc
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.dino_processor = AutoProcessor.from_pretrained(dino_model)
        self.dino_model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(dino_model)
            .to(self.device).float().eval()
        )
        if checkpoint is None:
            self.sam2_predictor = SAM2ImagePredictor.from_pretrained(
                sam2_model, device=str(self.device), apply_postprocessing=False,
            )
        else:
            sam_model = build_sam2(
                sam2_config, str(checkpoint), device=str(self.device),
                apply_postprocessing=False,
            )
            self.sam2_predictor = SAM2ImagePredictor(sam_model)
        self.sam2_predictor.model.float().eval()

    def _detect_box(self, rgb_image, text_prompt):
        # Period-separated phrases are supported by both old and new processors.
        text = text_prompt.lower().rstrip(".").strip() + "."
        inputs = self.dino_processor(
            images=rgb_image, text=text, return_tensors="pt",
        ).to(self.device)
        outputs = self.dino_model(**inputs)
        height, width = rgb_image.shape[:2]

        postprocess = self.dino_processor.post_process_grounded_object_detection
        parameters = inspect.signature(postprocess).parameters
        if "threshold" in parameters:
            threshold_name = "threshold"
        elif "box_threshold" in parameters:
            threshold_name = "box_threshold"
        else:
            raise RuntimeError(
                "Unsupported transformers Grounding DINO postprocessor: "
                "expected a threshold or box_threshold parameter."
            )
        detections = postprocess(
            outputs, inputs["input_ids"],
            **{threshold_name: self.box_threshold},
            text_threshold=self.text_threshold, target_sizes=[(height, width)],
        )[0]
        boxes = detections["boxes"].detach().float().cpu().numpy()
        scores = detections["scores"].detach().float().cpu().numpy()
        # Invalid or entirely off-image predictions must not hide a valid box.
        for index in np.argsort(-scores):
            score = float(scores[index])
            if not np.isfinite(score) or not self.box_threshold <= score <= 1.0:
                continue
            box = clip_box(boxes[index], height, width)
            if box is not None:
                return box, score
        return None, 0.0

    def detect_bbox(self, rgb_image: np.ndarray, text_prompt: str) -> dict:
        """DINO-only watchdog check; SAM2 is not called."""
        with InferenceTimer(self.device) as timer, torch.inference_mode():
            prompt = validate_input(rgb_image, text_prompt)
            box, score = self._detect_box(rgb_image, prompt)
        return dict(bbox=box, confidence=score, latency_ms=timer.elapsed_ms)

    def infer(self, rgb_image: np.ndarray, text_prompt: str) -> dict:
        box, score, mask = None, 0.0, None
        dino_timer, sam_timer = InferenceTimer(self.device), InferenceTimer(self.device)
        with InferenceTimer(self.device) as timer:
            prompt = validate_input(rgb_image, text_prompt)
            with torch.inference_mode():
                with dino_timer:
                    box, score = self._detect_box(rgb_image, prompt)
                if box is not None:
                    with sam_timer:
                        # Recompute the image embedding for every frame; no tracking.
                        self.sam2_predictor.set_image(rgb_image)
                        masks, _, _ = self.sam2_predictor.predict(
                            box=np.asarray(box, dtype=np.float32),
                            multimask_output=False,
                        )
                        if masks is not None:
                            # SAM2 predict returns CPU NumPy binary masks in C,H,W.
                            masks = np.asarray(masks)
                            if masks.size:
                                expected = (1, *rgb_image.shape[:2])
                                if masks.shape != expected:
                                    raise RuntimeError(
                                        f"SAM2 returned mask shape {masks.shape}; "
                                        f"expected {expected}"
                                    )
                                if not np.isfinite(masks).all():
                                    raise RuntimeError("SAM2 returned a non-finite mask")
                                candidate = np.ascontiguousarray(masks[0] > 0.5)
                                if candidate.any():
                                    mask = candidate
        result = make_result(
            bbox=box, mask=mask, confidence=score, latency_ms=timer.elapsed_ms,
        )
        result.update(dino_ms=dino_timer.elapsed_ms,
                      sam_ms=sam_timer.elapsed_ms if box is not None else None)
        return result
