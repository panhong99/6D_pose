"""Exercise real decoding, exports and aggregation with deterministic fake models."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import benchmark
from pipelines.artifacts import draw_result, write_rgb
from pipelines.base import DetectorPipeline, InferenceTimer, clip_box, make_result, resolve_device, validate_input


class FakePipeline(DetectorPipeline):
    def __init__(self, name, events, fail=False):
        self.name, self.events, self.fail = name, events, fail

    def prepare_prompt(self, text_prompt):
        self.events.append(('prepare', self.name, text_prompt))

    def infer(self, rgb_image, text_prompt):
        self.events.append(('infer', self.name, text_prompt, tuple(rgb_image[0, 0])))
        if self.fail:
            raise RuntimeError('test model failure')
        # Red is a detected cube; blue represents a normal no-detection outcome.
        if rgb_image[0, 0, 0] < 100:
            return make_result(latency_ms=2.0)
        mask = np.zeros(rgb_image.shape[:2], dtype=bool)
        mask[2:8, 3:11] = True
        return make_result([3.0, 2.0, 11.0, 8.0], mask, 0.8, 10.0)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs = self.root / 'input'
        self.inputs.mkdir()
        write_rgb(self.inputs / 'cube.png', np.full((24, 40, 3), [210, 20, 5], dtype=np.uint8))
        write_rgb(self.inputs / 'nested' / 'cube_BLUR.png', np.full((24, 40, 3), [5, 20, 210], dtype=np.uint8))
        self.events = []

    def tearDown(self):
        self.temp.cleanup()

    def args(self, *extra):
        return benchmark.parse_args(['--input_dir', str(self.inputs), '--output_dir', str(self.root/'results'),
                                     '--device', 'cpu', '--prompt', "rubik's cube", *extra])

    def factory(self, name, args):
        self.events.append(('load', name))
        return FakePipeline(name, self.events)

    def run_quiet(self, args, factory=None):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return benchmark.run_benchmark(args, factory or self.factory)

    def test_complete_export_and_blur_aggregation(self):
        args = self.args('--prompt', 'colorful cube')
        self.assertEqual(self.run_quiet(args), 0)
        rows = json.loads((args.output_dir/'per_frame.json').read_text())
        self.assertEqual(len(rows), 8)
        self.assertEqual(sum(row['success'] for row in rows), 4)
        self.assertEqual(len(list((args.output_dir/'comparisons').rglob('*.png'))), 4)
        summary = json.loads((args.output_dir/'summary.json').read_text())
        all_group = next(row for row in summary if row['pipeline'] == 'yoloe' and
                         row['prompt_id'] == 'prompt_000' and row['tag'] == 'all')
        self.assertEqual(all_group['success_rate'], 0.5)
        self.assertEqual(all_group['latency_mean_ms'], 6.0)
        self.assertEqual(all_group['successful_latency_mean_ms'], 10.0)
        self.assertTrue(all(row['success_rate'] == 0 for row in summary if row['tag'] == 'blur'))
        for row in rows:
            if row['mask_path']:
                mask = cv2.imread(str(args.output_dir/row['mask_path']), cv2.IMREAD_GRAYSCALE)
                self.assertEqual(mask.shape, (24, 40))
                self.assertEqual(set(np.unique(mask)), {0, 255})
                self.assertEqual(int((mask > 0).sum()), 48)
        with (args.output_dir/'per_frame.csv').open(newline='') as stream:
            csv_rows = list(csv.DictReader(stream))
        self.assertEqual(json.loads(csv_rows[0]['bbox']), [3.0, 2.0, 11.0, 8.0])
        config = json.loads((args.output_dir/'run_config.json').read_text())
        self.assertEqual(config['status'], 'completed')
        # Entire first pipeline finishes before the second is loaded.
        second_load = self.events.index(('load', 'grounding_dino_sam2'))
        self.assertFalse(any(event[1] == 'yoloe' for event in self.events[second_load:]))
        self.assertEqual(sum(event[0] == 'prepare' for event in self.events), 4)
        self.assertEqual(sum(event[0] == 'infer' for event in self.events), 12)

    def test_runtime_errors_are_not_misses_or_fast_samples(self):
        args = self.args('--warmup', '0')
        def factory(name, config):
            return FakePipeline(name, self.events, fail=name == 'yoloe')
        self.assertEqual(self.run_quiet(args, factory), 1)
        rows = json.loads((args.output_dir/'per_frame.json').read_text())
        errors = [row for row in rows if row['pipeline'] == 'yoloe']
        self.assertTrue(all(row['error'] and row['latency_ms'] is None for row in errors))
        summary = benchmark.summarize(rows)
        group = next(row for row in summary if row['pipeline'] == 'yoloe' and row['tag'] == 'all')
        self.assertEqual(group['error_count'], 2)
        self.assertEqual(group['latency_count'], 0)
        self.assertIsNone(group['latency_mean_ms'])
        self.assertTrue(any(row['success'] for row in rows))

    def test_setup_failure_still_exports_other_pipeline(self):
        args = self.args('--warmup', '0')
        def factory(name, config):
            if name == 'grounding_dino_sam2':
                raise ImportError('SAM2 missing: bash install_sam2.sh')
            return self.factory(name, config)
        self.assertEqual(self.run_quiet(args, factory), 1)
        rows = json.loads((args.output_dir/'per_frame.json').read_text())
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(bool(row['error']) for row in rows), 2)
        self.assertEqual(len(list((args.output_dir/'comparisons').rglob('*.png'))), 2)

    def test_bad_input_recorded_and_output_reuse_rejected(self):
        (self.inputs/'broken.png').write_bytes(b'not an image')
        args = self.args('--max_frames', '1', '--warmup', '0')
        self.assertEqual(self.run_quiet(args), 1)
        errors = json.loads((args.output_dir/'input_errors.json').read_text())
        self.assertEqual(errors[0]['source'], 'broken.png')
        frames = json.loads((args.output_dir/'frames.json').read_text())
        self.assertEqual(len(frames), 1)
        with self.assertRaisesRegex(ValueError, 'new or empty'):
            self.run_quiet(args)

    def test_manual_tag_precedence(self):
        path = self.root/'tags.csv'
        path.write_text('source,frame_index,tag\nclip.mp4,,normal\nclip.mp4,3,blur\n'
                        'nested/cube_BLUR.png,,normal\n')
        tags = benchmark.load_tags(path)
        self.assertEqual(benchmark.frame_tag('clip.mp4', 3, ['blur'], tags), 'blur')
        self.assertEqual(benchmark.frame_tag('clip.mp4', 4, ['blur'], tags), 'normal')
        self.assertEqual(benchmark.frame_tag('nested/cube_BLUR.png', None, ['blur'], tags), 'normal')
        self.assertEqual(benchmark.frame_tag('other_BLUR.mp4', 2, ['blur'], tags), 'blur')

    def test_video_stride_and_global_frame_cap(self):
        path = self.root/'clip_BLUR.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (40, 24))
        if not writer.isOpened():
            self.skipTest('MJPG video encoder unavailable')
        try:
            for index in range(5):
                writer.write(np.full((24, 40, 3), [index*20, 10, 210], dtype=np.uint8))
        finally:
            writer.release()
        args = benchmark.parse_args(['--video', str(path), '--output_dir', str(self.root/'video_results'),
                                     '--video_stride', '2', '--max_frames', '2', '--warmup', '0',
                                     '--prompt', 'cube', '--pipelines', 'yoloe', '--device', 'cpu'])
        self.assertEqual(self.run_quiet(args), 0)
        frames = json.loads((args.output_dir/'frames.json').read_text())
        self.assertEqual([frame['frame_index'] for frame in frames], [0, 2])
        self.assertEqual([frame['timestamp_ms'] for frame in frames], [0.0, 200.0])
        self.assertTrue(all(frame['tag'] == 'blur' for frame in frames))


class ContractTests(unittest.TestCase):
    def test_rgb_uint8_and_nonempty_prompt(self):
        image = np.zeros((12, 20, 3), dtype=np.uint8)
        self.assertEqual(validate_input(image, ' cube '), 'cube')
        for bad in (image.astype(float), image[..., 0], image[:0]):
            with self.assertRaises(ValueError):
                validate_input(bad, 'cube')
        with self.assertRaises(ValueError):
            validate_input(image, '...')

    def test_box_and_mask_contract(self):
        self.assertEqual(clip_box([-5, -8, 100, 80], 24, 40), [0, 0, 40, 24])
        self.assertIsNone(clip_box([50, 0, 60, 12], 24, 40))
        self.assertIsNone(clip_box([0, 0, float('nan'), 12], 24, 40))
        result = make_result([0, 0, 10, 10], np.zeros((24, 40), dtype=bool), 0.7, 2)
        self.assertFalse(result['success'])
        self.assertIsNone(result['mask'])
        self.assertIsNotNone(result['bbox'])

    def test_cuda_timer_synchronizes_both_ends(self):
        with patch('pipelines.base.torch.cuda.synchronize') as sync:
            with InferenceTimer('cuda:1') as timer:
                pass
            self.assertEqual(sync.call_count, 2)
            self.assertGreaterEqual(timer.elapsed_ms, 0)
        with patch('pipelines.base.torch.cuda.is_available', return_value=False):
            self.assertEqual(str(resolve_device('auto')), 'cpu')
            with self.assertRaises(ValueError):
                resolve_device('cuda')

    def test_overlay_keeps_original_pixels_outside_mask_and_header(self):
        rgb = np.full((24, 40, 3), [100, 80, 60], dtype=np.uint8)
        mask = np.zeros((24, 40), dtype=bool)
        mask[8:12, 10:16] = True
        result = make_result(mask=mask, latency_ms=2)
        drawn = draw_result(rgb, result, 'test')
        source_region = drawn[-24:, :40]
        np.testing.assert_array_equal(source_region[~mask], rgb[~mask])
        self.assertFalse(np.array_equal(source_region[mask], rgb[mask]))


if __name__ == '__main__':
    unittest.main()
