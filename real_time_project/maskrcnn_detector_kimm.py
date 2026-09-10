"""COCO-pretrained instance detection, independent of camera and FoundationPose."""
import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights


class MaskRCNNDetector:
    def __init__(self, device='cuda', threshold=0.5):
        self.device = torch.device(device)
        self.threshold = threshold
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.categories = weights.meta['categories']
        self.model = maskrcnn_resnet50_fpn_v2(weights=weights).to(self.device).eval()

    def predict(self, rgb):
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).to(self.device).float() / 255
        with torch.inference_mode():
            output = self.model([tensor])[0]
        keep = output['scores'] >= self.threshold
        boxes = output['boxes'][keep].cpu().numpy()
        scores = output['scores'][keep].cpu().numpy()
        labels = output['labels'][keep].cpu().numpy()
        masks = (output['masks'][keep, 0] >= 0.5).cpu().numpy()
        return [dict(box=box.tolist(), score=float(score), label_id=int(label),
                     label=self.categories[int(label)], mask=mask)
                for box, score, label, mask in zip(boxes, scores, labels, masks)]


def draw_detections(rgb, detections):
    image = rgb.copy()
    for index, detection in enumerate(detections):
        color = ((53 * index + 255) % 256, (97 * index) % 256, (151 * index + 255) % 256)
        contours, _ = cv2.findContours(detection['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, 2)
        x1, y1, x2, y2 = map(int, detection['box'])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        cv2.putText(image, f'{index}: {detection["label"]} {detection["score"]:.2f}',
                    (max(0, x1), max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image
