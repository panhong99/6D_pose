"""Offline contract tests: fake pretrained models, real tensor/NumPy processing."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from pipelines import pipeline_grounding_dino_sam2 as pipeline_module


class FakeBatch(dict):
    def to(self, device):
        return self


class Processor:
    def __init__(self, legacy=False):
        self.boxes = torch.tensor([[1.0, 2.0, 7.0, 5.0]])
        self.scores = torch.tensor([0.8])
        self.last_text = None
        self.last_thresholds = None
        if legacy:
            self.post_process_grounded_object_detection = self.legacy_postprocess

    def __call__(self, *, images, text, return_tensors):
        self.last_text = text
        self.last_image = images
        return FakeBatch(input_ids=torch.tensor([[1, 2]]),
                         pixel_values=torch.zeros((1, 3, 6, 8)))

    def post_process_grounded_object_detection(
        self, outputs, input_ids, threshold, text_threshold, target_sizes,
    ):
        self.last_thresholds = (threshold, text_threshold, target_sizes)
        return [{"boxes": self.boxes, "scores": self.scores}]

    def legacy_postprocess(
        self, outputs, input_ids, box_threshold, text_threshold, target_sizes,
    ):
        return Processor.post_process_grounded_object_detection(
            self, outputs, input_ids, box_threshold, text_threshold, target_sizes,
        )


class GroundingDINOSAM2Tests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((6, 8, 3), dtype=np.uint8)
        self.rgb[..., 0] = 230  # Makes unintended RGB/BGR conversion observable.
        self.processor = Processor()
        self.dino = MagicMock()
        self.dino.to.return_value = self.dino
        self.dino.float.return_value = self.dino
        self.dino.eval.return_value = self.dino
        self.predictor = MagicMock()
        self.predictor.model.float.return_value = self.predictor.model
        self.predictor.model.eval.return_value = self.predictor.model
        mask = np.zeros((1, 6, 8), dtype=np.float32)
        mask[:, 2:5, 1:7] = 1
        self.predictor.predict.return_value = (mask, np.array([0.1]), None)
        self.predictor_cls = MagicMock(return_value=self.predictor)
        self.predictor_cls.from_pretrained.return_value = self.predictor
        self.builder = MagicMock(return_value=self.predictor.model)
        self.processor_factory = MagicMock()
        self.processor_factory.from_pretrained.return_value = self.processor
        self.dino_factory = MagicMock()
        self.dino_factory.from_pretrained.return_value = self.dino
        modules = {
            "transformers": types.SimpleNamespace(
                AutoProcessor=self.processor_factory,
                AutoModelForZeroShotObjectDetection=self.dino_factory,
            ),
            "sam2.sam2_image_predictor": types.SimpleNamespace(
                SAM2ImagePredictor=self.predictor_cls,
            ),
            "sam2.build_sam": types.SimpleNamespace(build_sam2=self.builder),
        }
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)

    def create_pipeline(self, **kwargs):
        return pipeline_module.GroundingDINOSAM2Pipeline(device="cpu", **kwargs)

    def test_hf_defaults_and_rgb_binary_output(self):
        pipeline = self.create_pipeline()
        result = pipeline.infer(self.rgb, "  RUBIK'S CUBE  ")
        self.predictor_cls.from_pretrained.assert_called_once_with(
            "facebook/sam2.1-hiera-tiny", device="cpu", apply_postprocessing=False,
        )
        self.builder.assert_not_called()
        self.assertTrue(result["success"])
        self.assertEqual(result["bbox"], [1.0, 2.0, 7.0, 5.0])
        self.assertAlmostEqual(result["confidence"], 0.8)
        self.assertGreaterEqual(result["latency_ms"], 0)
        self.assertGreaterEqual(result["dino_ms"], 0)
        self.assertGreaterEqual(result["sam_ms"], 0)
        self.assertEqual(result["mask"].shape, (6, 8))
        self.assertEqual(result["mask"].dtype, np.bool_)
        self.assertTrue(result["mask"][2, 1])
        self.assertFalse(result["mask"][0, 0])
        self.assertEqual(self.processor.last_text, "rubik's cube.")
        np.testing.assert_array_equal(self.processor.last_image, self.rgb)
        np.testing.assert_array_equal(self.predictor.set_image.call_args.args[0], self.rgb)
        np.testing.assert_array_equal(self.predictor.predict.call_args.kwargs["box"],
                                      [1, 2, 7, 5])
        self.assertIs(self.predictor.predict.call_args.kwargs["multimask_output"], False)
        self.dino.float.assert_called_once()
        self.predictor.model.float.assert_called_once()

    def test_current_and_legacy_threshold_signatures(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                self.processor = Processor(legacy=legacy)
                self.processor_factory.from_pretrained.return_value = self.processor
                pipeline = self.create_pipeline(box_threshold=0.4, text_threshold=0.2)
                self.assertTrue(pipeline.infer(self.rgb, "cube")["success"])
                self.assertEqual(self.processor.last_thresholds, (0.4, 0.2, [(6, 8)]))

    def test_select_highest_valid_box_and_clip_to_original_image(self):
        self.processor.boxes = torch.tensor([
            [20.0, 20.0, 30.0, 30.0],  # Highest score, entirely off-image.
            [float("nan"), 1.0, 2.0, 3.0],
            [-2.0, -3.0, 11.0, 10.0],
            [1.0, 1.0, 3.0, 3.0],
        ])
        self.processor.scores = torch.tensor([0.99, 0.98, 0.8, 0.5])
        result = self.create_pipeline().infer(self.rgb, "cube")
        self.assertEqual(result["bbox"], [0.0, 0.0, 8.0, 6.0])
        self.assertAlmostEqual(result["confidence"], 0.8)
        np.testing.assert_array_equal(self.predictor.predict.call_args.kwargs["box"],
                                      [0, 0, 8, 6])

    def test_no_detection_skips_sam2(self):
        self.processor.boxes = torch.empty((0, 4))
        self.processor.scores = torch.empty(0)
        result = self.create_pipeline().infer(self.rgb, "cube")
        self.assertFalse(result["success"])
        self.assertIsNone(result["bbox"])
        self.assertIsNone(result["mask"])
        self.assertEqual(result["confidence"], 0.0)
        self.assertIsNone(result["sam_ms"])
        self.predictor.set_image.assert_not_called()
        self.predictor.predict.assert_not_called()

    def test_invalid_score_and_degenerate_box_are_not_selected(self):
        self.processor.boxes = torch.tensor([
            [1, 1, 3, 3], [1, 1, 3, 3], [1, 1, 1, 3],
        ])
        self.processor.scores = torch.tensor([float("nan"), 1.2, 0.9])
        result = self.create_pipeline().infer(self.rgb, "cube")
        self.assertFalse(result["success"])
        self.predictor.predict.assert_not_called()

    def test_empty_mask_preserves_detection_and_score(self):
        self.predictor.predict.return_value = (np.zeros((1, 6, 8)), np.array([0.9]), None)
        result = self.create_pipeline().infer(self.rgb, "cube")
        self.assertFalse(result["success"])
        self.assertIsNone(result["mask"])
        self.assertEqual(result["bbox"], [1.0, 2.0, 7.0, 5.0])
        self.assertAlmostEqual(result["confidence"], 0.8)

    def test_wrong_size_mask_raises_instead_of_saving_misaligned_mask(self):
        self.predictor.predict.return_value = (np.ones((1, 3, 4)), np.array([0.9]), None)
        with self.assertRaisesRegex(RuntimeError, "mask shape"):
            self.create_pipeline().infer(self.rgb, "cube")

    def test_local_checkpoint_and_matching_config(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sam2.pt"
            checkpoint.touch()
            self.create_pipeline(
                sam2_checkpoint=checkpoint,
                sam2_config="configs/sam2.1/sam2.1_hiera_s.yaml",
                dino_model="IDEA-Research/grounding-dino-base",
            )
            self.builder.assert_called_once_with(
                "configs/sam2.1/sam2.1_hiera_s.yaml", str(checkpoint),
                device="cpu", apply_postprocessing=False,
            )
            self.predictor_cls.from_pretrained.assert_not_called()
            self.dino_factory.from_pretrained.assert_called_once_with(
                "IDEA-Research/grounding-dino-base",
            )

    def test_invalid_local_configuration_fails_before_weight_loading(self):
        with self.assertRaises(FileNotFoundError):
            self.create_pipeline(sam2_checkpoint="no_such_checkpoint.pt")
        with self.assertRaisesRegex(ValueError, "requires sam2_checkpoint"):
            self.create_pipeline(sam2_config="custom.yaml")
        self.dino_factory.from_pretrained.assert_not_called()

    def test_missing_sam2_reports_install_command_before_loading_dino(self):
        with patch.dict(sys.modules, {"sam2.sam2_image_predictor": None}):
            with self.assertRaisesRegex(ImportError, "bash install_sam2.sh"):
                self.create_pipeline()
        self.dino_factory.from_pretrained.assert_not_called()

    def test_invalid_input_is_rejected_before_model_forward(self):
        pipeline = self.create_pipeline()
        with self.assertRaises((ValueError, TypeError)):
            pipeline.infer(self.rgb.astype(np.float32), "cube")
        with self.assertRaises((ValueError, TypeError)):
            pipeline.infer(self.rgb, "   ")
        self.dino.assert_not_called()

    def test_inference_and_cpu_mask_conversion_are_inside_timing_scope(self):
        events = []

        class Timer:
            elapsed_ms = 12.5

            def __init__(self, device):
                pass

            def __enter__(self):
                events.append("start")
                return self

            def __exit__(self, *exc_info):
                events.append("stop")

        class ReturnedMasks:
            def __array__(self, dtype=None, copy=None):
                events.append("cpu_mask")
                return np.ones((1, 6, 8), dtype=np.float32)

        def dino_forward(**inputs):
            self.assertTrue(torch.is_inference_mode_enabled())
            events.append("dino")
            return object()

        def segment(**kwargs):
            self.assertTrue(torch.is_inference_mode_enabled())
            events.append("sam2")
            return ReturnedMasks(), np.array([0.9]), None

        self.dino.side_effect = dino_forward
        self.predictor.predict.side_effect = segment
        with patch.object(pipeline_module, "InferenceTimer", Timer):
            result = self.create_pipeline().infer(self.rgb, "cube")
        # Outer total timer contains separate DINO and SAM timers.
        self.assertEqual(events, ["start", "start", "dino", "stop",
                                  "start", "sam2", "cpu_mask", "stop", "stop"])
        self.assertEqual(result["latency_ms"], 12.5)


if __name__ == "__main__":
    unittest.main()
