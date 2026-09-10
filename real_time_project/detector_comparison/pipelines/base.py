"""Shared RGB input, original-image coordinates, success and timing contract."""
from abc import ABC, abstractmethod
import time

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
