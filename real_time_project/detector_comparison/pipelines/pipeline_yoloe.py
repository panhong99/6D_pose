"""YOLOE text-prompted boxes and masks from one image forward pass (no SAM).

Install the common requirements from the comparison directory. Text prompting
also downloads a MobileCLIP encoder on first use; see README.md for setup.

Run from the comparison directory::

    python -m pipelines.pipeline_yoloe --image ./cube.jpg \
        --prompt "rubik's cube" --prompt "colorful cube" --output_dir ./results/yoloe
    python pipelines/pipeline_yoloe.py --image ./cube.jpg --device cpu
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

if __package__:
    from .base import (
        DetectorPipeline,
        InferenceTimer,
        clip_box,
        make_result,
        resolve_device,
        validate_input,
    )
else:  # python pipelines/pipeline_yoloe.py
    from base import (
        DetectorPipeline,
        InferenceTimer,
        clip_box,
        make_result,
        resolve_device,
        validate_input,
    )


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--prompt", action="append", dest="prompts",
        help="One target description; repeat to compare prompts independently.",
    )
    parser.add_argument("--checkpoint", default="yoloe-11s-seg.pt")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Default: auto CUDA/CPU; e.g. cpu, cuda:0.")
    parser.add_argument("--output_dir", type=Path, default=Path("results/yoloe"))
    args = parser.parse_args()

    if __package__:
        from .artifacts import save_single_result
    else:
        from artifacts import save_single_result

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        parser.error(f"Cannot read image: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pipeline = YOLOEPipeline(
        checkpoint=args.checkpoint, device=args.device, conf=args.conf, imgsz=args.imgsz,
    )
    prompts = args.prompts or ["rubik's cube", "colorful cube", "toy cube"]
    for index, prompt in enumerate(prompts):
        result = pipeline.infer(rgb, prompt)
        paths = save_single_result(
            rgb, {**result, "prompt": prompt}, args.output_dir,
            f"prompt_{index:03d}", title=f"YOLOE | {prompt}",
        )
        print(json.dumps({
            "prompt": prompt,
            **{key: value for key, value in result.items() if key != "mask"},
            "artifacts": paths,
        }, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
