"""Text-prompted Grounding DINO boxes followed by official SAM2 image masks.

Install SAM2 separately with ``bash install_sam2.sh`` from the project directory.
Weights download on first use; use ``sam2_checkpoint`` and ``sam2_config`` for
local SAM2 weights. Both models run in FP32, including on CPU.

Standalone example (from detector_comparison):
    python -m pipelines.pipeline_grounding_dino_sam2 --image cube.jpg \
        --prompt "rubik's cube" --prompt "colorful cube" --output_dir results/dino
"""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .base import (
        DetectorPipeline, InferenceTimer, clip_box, make_result, resolve_device,
        validate_input,
    )
else:  # Also support: python pipelines/pipeline_grounding_dino_sam2.py ...
    from base import (
        DetectorPipeline, InferenceTimer, clip_box, make_result, resolve_device,
        validate_input,
    )


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


def main():
    import cv2

    if __package__:
        from .artifacts import save_single_result
    else:
        from artifacts import save_single_result

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", action="append", help="Repeat to compare prompts")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--dino_model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--sam2_model", default=DEFAULT_SAM2_MODEL,
                        help="Hugging Face SAM2 ID; used without --sam2_checkpoint")
    parser.add_argument("--sam2_checkpoint", type=Path)
    parser.add_argument("--sam2_config", default=DEFAULT_SAM2_CONFIG,
                        help="SAM2 package Hydra config matching the local checkpoint")
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--output_dir", type=Path, default=Path("results/dino_sam2"))
    args = parser.parse_args()

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        parser.error(f"Cannot read image: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pipeline = GroundingDINOSAM2Pipeline(
        sam2_checkpoint=args.sam2_checkpoint, sam2_config=args.sam2_config,
        dino_model=args.dino_model, sam2_model=args.sam2_model, device=args.device,
        box_threshold=args.box_threshold, text_threshold=args.text_threshold,
    )
    for index, prompt in enumerate(args.prompt or ["rubik's cube"]):
        result = pipeline.infer(rgb, prompt)
        paths = save_single_result(
            rgb, {**result, "prompt": prompt}, args.output_dir, f"prompt_{index:03d}",
            title=f"Grounding DINO + SAM2 | {prompt}",
        )
        print(json.dumps({
            "prompt": prompt, **{k: v for k, v in result.items() if k != "mask"},
            "has_mask": result["mask"] is not None, "artifacts": paths,
        }, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
