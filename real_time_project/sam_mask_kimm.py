"""SAM 2.1 Tiny box-prompted initialization; no camera or pose logic."""
import numpy as np
import torch


class SamMask:
    def __init__(self, checkpoint):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',
                           str(checkpoint), device='cuda', apply_postprocessing=False)
        self.predictor = SAM2ImagePredictor(model)

    def predict(self, rgb, box):
        box = np.asarray(box, dtype=np.float32)
        h, w = rgb.shape[:2]
        if (box.shape != (4,) or not np.isfinite(box).all()
                or not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h)):
            raise ValueError('Box must be pixel coordinates x1 y1 x2 y2 within image')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            self.predictor.set_image(rgb)
            masks, _, _ = self.predictor.predict(box=box, multimask_output=False)
        mask = np.asarray(masks[0], dtype=bool)
        if mask.shape != (h, w) or not mask.any():
            raise ValueError('SAM returned an empty or incorrectly sized mask')
        return mask
