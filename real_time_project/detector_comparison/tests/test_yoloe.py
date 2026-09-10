"""YOLOE adapter contract tests; no network or model weights are required."""

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from pipelines.pipeline_yoloe import YOLOEPipeline


class Boxes:
    def __init__(self, boxes, scores):
        self.xyxy = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        self.conf = torch.tensor(scores, dtype=torch.float32)

    def __len__(self):
        return len(self.conf)


def prediction(boxes=(), scores=(), masks=None):
    return SimpleNamespace(
        boxes=Boxes(boxes, scores),
        masks=None if masks is None else SimpleNamespace(data=torch.tensor(masks)),
    )


class YOLOEPipelineTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock(task="segment")
        self.model.get_text_pe.side_effect = lambda texts: torch.ones((1, len(texts), 8))
        self.model.predict.return_value = [prediction()]
        self.factory = Mock(return_value=self.model)
        module_patch = patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLOE=self.factory)})
        module_patch.start()
        self.addCleanup(module_patch.stop)
        self.pipeline = YOLOEPipeline(device="cpu")
        self.rgb = np.zeros((4, 8, 3), dtype=np.uint8)
        self.rgb[0, 0] = [255, 41, 9]

    def test_rgb_conversion_and_original_mask_geometry(self):
        mask = np.zeros((1, 4, 8), dtype=np.float32)
        mask[0, 1:3, 2:7] = 1
        self.model.predict.return_value = [prediction([[2, 1, 7, 3]], [0.8], mask)]

        result = self.pipeline.infer(self.rgb, "rubik's cube")

        self.assertTrue(result["success"])
        np.testing.assert_array_equal(result["mask"], mask[0] > 0.5)
        self.assertEqual(result["mask"].dtype, np.bool_)
        self.assertEqual(result["bbox"], [2.0, 1.0, 7.0, 3.0])
        kwargs = self.model.predict.call_args.kwargs
        np.testing.assert_array_equal(kwargs["source"][0, 0], [9, 41, 255])
        self.assertTrue(kwargs["source"].flags.c_contiguous)
        self.assertTrue(kwargs["retina_masks"])
        self.assertEqual(kwargs["imgsz"], 640)
        self.assertGreaterEqual(result["latency_ms"], 0)
        np.testing.assert_array_equal(self.rgb[0, 0], [255, 41, 9])

    def test_highest_valid_score_preserves_box_mask_pairing(self):
        masks = np.zeros((4, 4, 8), dtype=np.float32)
        masks[2, 2, 3] = 1
        self.model.predict.return_value = [prediction(
            [[1, 1, 2, 2], [1, 1, 1, 2], [-2, -1, 15, 10], [0, 0, 2, 2]],
            [0.2, 0.95, 0.7, float("nan")], masks,
        )]

        result = self.pipeline.infer(self.rgb, "cube")

        self.assertTrue(result["success"])
        self.assertEqual(result["bbox"], [0.0, 0.0, 8.0, 4.0])
        self.assertAlmostEqual(result["confidence"], 0.7)
        np.testing.assert_array_equal(result["mask"], masks[2].astype(bool))

    def test_no_detection_and_missing_boxes_return_failure(self):
        for detection in (prediction(), SimpleNamespace(boxes=None, masks=None)):
            with self.subTest(detection=detection):
                self.model.predict.return_value = [detection]
                result = self.pipeline.infer(self.rgb, "cube")
                self.assertFalse(result["success"])
                self.assertIsNone(result["bbox"])
                self.assertIsNone(result["mask"])
                self.assertEqual(result["confidence"], 0)

    def test_missing_empty_and_nonfinite_masks_fail_but_keep_detection(self):
        for masks in (None, np.zeros((1, 4, 8)), np.full((1, 4, 8), np.nan)):
            with self.subTest(masks=masks):
                self.model.predict.return_value = [prediction([[1, 1, 3, 3]], [0.9], masks)]
                result = self.pipeline.infer(self.rgb, "cube")
                self.assertFalse(result["success"])
                self.assertEqual(result["bbox"], [1.0, 1.0, 3.0, 3.0])
                self.assertIsNone(result["mask"])
                self.assertAlmostEqual(result["confidence"], 0.9)

    def test_padded_mask_is_not_stretched(self):
        self.model.predict.return_value = [prediction(
            [[1, 1, 3, 3]], [0.9], np.ones((1, 8, 8)),
        )]
        with self.assertRaisesRegex(RuntimeError, "retina_masks=True"):
            self.pipeline.infer(self.rgb, "cube")

    def test_invalid_coordinates_and_low_scores_are_rejected(self):
        self.model.predict.return_value = [prediction(
            [[float("nan"), 0, 2, 3], [20, 20, 25, 25], [1, 1, 3, 3]],
            [0.9, 0.8, 0.1], np.ones((3, 4, 8)),
        )]
        result = self.pipeline.infer(self.rgb, "cube")
        self.assertFalse(result["success"])
        self.assertIsNone(result["bbox"])

    def test_prompt_embeddings_are_cached_when_prompts_change(self):
        self.pipeline.prepare_prompt("  rubik's cube  ")
        self.pipeline.infer(self.rgb, "rubik's cube")
        self.pipeline.infer(self.rgb, "colorful cube")
        self.pipeline.infer(self.rgb, "rubik's cube")
        self.assertEqual(self.model.get_text_pe.call_count, 2)
        self.assertEqual(self.model.set_classes.call_count, 3)
        self.assertEqual(self.model.get_text_pe.call_args_list[0].args[0], ["rubik's cube"])

    def test_prompt_preparation_occurs_inside_latency_timer(self):
        events = []

        class Timer:
            elapsed_ms = 12.5

            def __init__(self, device):
                pass

            def __enter__(self):
                events.append("start")
                return self

            def __exit__(self, *args):
                events.append("end")

        self.model.get_text_pe.side_effect = lambda texts: events.append("text")
        self.model.predict.side_effect = lambda **kwargs: events.append("image") or [prediction()]
        with patch("pipelines.pipeline_yoloe.InferenceTimer", Timer):
            result = self.pipeline.infer(self.rgb, "cube")
        self.assertEqual(events, ["start", "text", "image", "end"])
        self.assertEqual(result["latency_ms"], 12.5)

    def test_invalid_input_never_reaches_prediction(self):
        for image, prompt in ((self.rgb.astype(float), "cube"), (self.rgb, "  ")):
            with self.subTest(prompt=prompt, dtype=image.dtype):
                with self.assertRaises(ValueError):
                    self.pipeline.infer(image, prompt)
        self.model.predict.assert_not_called()

    def test_invalid_checkpoints_rejected_before_model_loading(self):
        self.factory.reset_mock()
        for checkpoint in ("yoloe-11s-seg-pf.pt", "yoloe-11s.pt", "yolo11s-seg.pt"):
            with self.subTest(checkpoint=checkpoint):
                with self.assertRaises(ValueError):
                    YOLOEPipeline(checkpoint, device="cpu")
        self.factory.assert_not_called()
        self.model.task = "detect"
        with self.assertRaisesRegex(ValueError, "does not support segmentation"):
            YOLOEPipeline(device="cpu")

    def test_invalid_threshold_and_image_size_rejected(self):
        for conf in (float("nan"), -0.1, 1.1):
            with self.subTest(conf=conf), self.assertRaises(ValueError):
                YOLOEPipeline(conf=conf, device="cpu")
        for imgsz in (0, -1, 1.2, True):
            with self.subTest(imgsz=imgsz), self.assertRaises(ValueError):
                YOLOEPipeline(imgsz=imgsz, device="cpu")


if __name__ == "__main__":
    unittest.main()
