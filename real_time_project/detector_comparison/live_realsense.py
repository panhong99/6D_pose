"""D455 RGB preview and selectable zero-shot detector (no FoundationPose).

python live_realsense.py --pipeline yoloe
python live_realsense.py --pipeline dino
q / Esc: quit; s: save the last completed inference image, mask and JSON.
"""
import argparse
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pipelines.artifacts import draw_result, save_single_result, side_by_side


def build_detector(args):
    if args.pipeline == 'yoloe':
        from pipelines.pipeline_yoloe import YOLOEPipeline
        return YOLOEPipeline(args.yoloe_checkpoint, device=args.device,
                             conf=args.yoloe_conf, imgsz=args.imgsz)

    from pipelines.pipeline_grounding_dino_sam2 import GroundingDINOSAM2Pipeline
    return GroundingDINOSAM2Pipeline(
        sam2_checkpoint=args.sam2_checkpoint, dino_model=args.dino_model,
        sam2_model=args.sam2_model, sam2_config=args.sam2_config,
        device=args.device, box_threshold=args.box_threshold,
        text_threshold=args.text_threshold)

class InferenceWorker:
    """One pending frame, overwritten by newer RGB frames while inference runs."""
    def __init__(self, detector, prompt):
        self.detector, self.prompt = detector, prompt
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.pending = self.completed = self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, rgb, frame_number):
        with self.lock:
            self.pending = (rgb, frame_number, time.perf_counter())
            self.ready.set()

    def snapshot(self):
        with self.lock:
            return self.completed, self.error

    def _run(self):
        previous_completion = None
        try:
            while not self.stop.is_set():
                if not self.ready.wait(0.1):
                    continue
                with self.lock:
                    item, self.pending = self.pending, None
                    self.ready.clear()
                if item is None or self.stop.is_set():
                    continue
                rgb, frame_number, received = item
                result = self.detector.infer(rgb, self.prompt)
                now = time.perf_counter()
                fps = 1.0 / (now - previous_completion) if previous_completion is not None else None
                previous_completion = now
                with self.lock:
                    self.completed = (rgb, result, frame_number, received, fps)
        except Exception as exc:
            with self.lock:
                self.error = f'{type(exc).__name__}: {exc}'

    def close(self):
        self.stop.set()
        self.ready.set()
        # Keep quitting responsive even if the current CPU inference is slow.
        self.thread.join(timeout=1.0)


def preview_panel(rgb, title):
    canvas = np.full((rgb.shape[0] + 44, max(400, rgb.shape[1]), 3), 28, dtype=np.uint8)
    canvas[44:, :rgb.shape[1]] = rgb
    cv2.putText(canvas, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def run(args):
    import pyrealsense2 as rs

    print(f'Loading {args.pipeline}; first use may download model/text encoder weights.', flush=True)
    detector = build_detector(args)
    detector.prepare_prompt(args.prompt)
    print(f'Device: {detector.device} | Prompt: {args.prompt}', flush=True)
    camera = rs.pipeline()
    config = rs.config()

    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    started, worker, window_created = False, None, False
    title = f'D455 | {args.pipeline} | q: quit, s: save'

    try:
        profile = camera.start(config)
        started = True
        print('Camera:', profile.get_device().get_info(rs.camera_info.name), flush=True)

        worker = InferenceWorker(detector, args.prompt)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        window_created = True
        cv2.resizeWindow(title, min(1600, 2 * args.width), args.height + 110)
        rgb = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        last_frame_time = time.perf_counter()
        print('Left: live RGB. Right: last processed frame with matching bbox/mask.', flush=True)

        while True:
            frames = camera.poll_for_frames()
            if frames:
                color = frames.get_color_frame()
                if color:
                    rgb = np.asanyarray(color.get_data()).copy()
                    last_frame_time = time.perf_counter()
                    worker.submit(rgb, int(color.get_frame_number()))
            if time.perf_counter() - last_frame_time > args.timeout:
                raise RuntimeError(f'No RGB frame for {args.timeout:g}s. Check camera/USB/stream settings.')
            completed, error = worker.snapshot()
            if error:
                raise RuntimeError(f'{args.pipeline} inference failed: {error}')
            left = preview_panel(rgb, f'LIVE RGB | configured camera FPS: {args.fps}')
            if completed is None:
                right = preview_panel(np.zeros_like(rgb), 'Waiting for first inference (cold start)...')
            else:
                inferred_rgb, result, number, received, fps = completed
                rate = f'{fps:.1f}' if fps is not None else 'warming up'
                age = (time.perf_counter() - received) * 1000
                right = draw_result(inferred_rgb, result,
                                    f'{args.pipeline} | {args.prompt} | frame {number} | '
                                    f'infer FPS {rate} | age {age:.0f} ms')
            display = side_by_side([left, right])
            cv2.imshow(title, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(10) & 0xFF
            if key in (ord('q'), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('s') and completed is not None:
                inferred_rgb, result, number, _, _ = completed
                stem = f'{args.pipeline}_{datetime.now():%Y%m%d_%H%M%S_%f}_frame_{number}'
                save_single_result(inferred_rgb, {**result, 'prompt': args.prompt,
                                                  'pipeline': args.pipeline, 'frame_number': number},
                                   args.output_dir, stem, title=f'{args.pipeline} | {args.prompt}')
                print(f'Saved: {args.output_dir / stem}', flush=True)
    finally:
        if worker is not None:
            worker.close()
        try:
            if started:
                camera.stop()
        finally:
            if window_created:
                cv2.destroyAllWindows()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', choices=['yoloe', 'dino'], required=True,
                        help='dino means Grounding DINO + SAM2')
    parser.add_argument('--prompt', default="rubik's cube")
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:0')
    parser.add_argument('--serial', default='', help='Optional RealSense serial number')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30, help='Camera stream FPS, not inference FPS')
    parser.add_argument('--timeout', type=float, default=5.0, help='Camera frame timeout in seconds')
    parser.add_argument('--output_dir', type=Path, default=Path('live_captures'))
    parser.add_argument('--yoloe_checkpoint', default='yoloe-11s-seg.pt')
    parser.add_argument('--yoloe_conf', type=float, default=0.15)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--dino_model', default='IDEA-Research/grounding-dino-tiny')
    parser.add_argument('--box_threshold', type=float, default=0.3)
    parser.add_argument('--text_threshold', type=float, default=0.25)
    parser.add_argument('--sam2_model', default='facebook/sam2.1-hiera-tiny')
    parser.add_argument('--sam2_checkpoint', type=Path)
    parser.add_argument('--sam2_config', default='configs/sam2.1/sam2.1_hiera_t.yaml')
    args = parser.parse_args(argv)
    if not args.prompt.strip().strip('.'):
        parser.error('--prompt must describe an object')
    if min(args.width, args.height, args.fps, args.imgsz) <= 0:
        parser.error('width, height, fps and imgsz must be positive')
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be finite and positive')
    return args


def main():
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f'Live detection error: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
