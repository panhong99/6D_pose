"""Measure D455 capture FPS and FoundationPose register/track FPS independently.

Answers: is the real-time pipeline slow because of the camera, or the pose network?
"""
import argparse
import time
from pathlib import Path


def measure_camera_fps(camera, n_frames=100):
    """Read n_frames back-to-back (no processing) and return the achieved capture FPS."""
    camera.read()  # warm up: first frame after stream start can stall
    start = time.perf_counter()
    for _ in range(n_frames):
        camera.read()
    elapsed = time.perf_counter() - start
    return n_frames / elapsed


def measure_register_fps(tracker, frame, mask, n_iters=10):
    """Time repeated register() calls on one frozen frame/mask. Each call resets first,
    so this measures cold detection cost, not incremental tracking."""
    times = []
    for _ in range(n_iters):
        tracker.reset()
        start = time.perf_counter()
        tracker.estimate(frame, mask)
        times.append(time.perf_counter() - start)
    return times


def measure_track_fps(tracker, frame, n_iters=30):
    """Time repeated track_one() calls in a row after one register(), matching
    the per-frame cost paid continuously during real-time tracking."""
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        tracker.estimate(frame, None)
        times.append(time.perf_counter() - start)
    return times


def summarize(label, times):
    times = sorted(times)
    n = len(times)
    mean = sum(times) / n
    median = times[n // 2]
    worst = times[-1]
    print(f'{label}: mean={1/mean:.1f} FPS ({mean*1000:.1f} ms) | '
         f'median={1/median:.1f} FPS ({median*1000:.1f} ms) | '
         f'worst={1/worst:.1f} FPS ({worst*1000:.1f} ms) | n={n}', flush=True)


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh_file', type=Path, required=True)
    parser.add_argument('--mesh_scale', type=float, default=1.0, help='CAD units to meters: mm=0.001')
    parser.add_argument('--sam_checkpoint', type=Path,
                        default=root / 'weights/sam2.1_hiera_tiny_kimm.pt')
    parser.add_argument('--serial', default='')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--max_depth', type=float, default=3.0)
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug_dir', type=Path, default=root / 'debug_live_kimm')
    parser.add_argument('--camera_frames', type=int, default=100,
                        help='Frames to read back-to-back when measuring camera FPS')
    parser.add_argument('--register_iters', type=int, default=10,
                        help='Repeats when measuring register() FPS')
    parser.add_argument('--track_iters', type=int, default=30,
                        help='Repeats when measuring track_one() FPS')
    args = parser.parse_args()
    for name in ('mesh_file', 'sam_checkpoint'):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            parser.error(f'{name} does not exist: {path}')
        setattr(args, name, path)
    return args


def main():
    args = parse_args()
    import cv2
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('FoundationPose requires CUDA in the foundationpose environment')
    from FoundationPose.real_time_project.d455_source_kimm import D455Source
    from FoundationPose.real_time_project.sam_mask_kimm import SamMask
    from FoundationPose.real_time_project.pose_tracker_kimm import PoseTracker

    camera = D455Source(args.width, args.height, args.fps, args.serial, args.max_depth)
    try:
        print(f'Measuring camera capture FPS over {args.camera_frames} frames...', flush=True)
        cam_fps = measure_camera_fps(camera, args.camera_frames)
        print(f'D455 capture: {cam_fps:.1f} FPS (requested {args.fps} FPS)', flush=True)

        frame = camera.read()
        x, y, w, h = cv2.selectROI('Select cube - ENTER accept, C cancel',
                                  cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR),
                                  fromCenter=False, showCrosshair=True)
        cv2.destroyWindow('Select cube - ENTER accept, C cancel')
        if not (w and h):
            print('No selection made, aborting.', flush=True)
            return

        segmenter = SamMask(args.sam_checkpoint)
        mask = segmenter.predict(frame.rgb, [x, y, x + w, y + h])

        tracker = PoseTracker(args.mesh_file, args.mesh_scale, args.debug_dir,
                              args.est_refine_iter, args.track_refine_iter)

        print(f'Measuring register() FPS over {args.register_iters} calls '
             f'(iteration={args.est_refine_iter})...', flush=True)
        register_times = measure_register_fps(tracker, frame, mask, args.register_iters)
        summarize('FoundationPose register()', register_times)

        tracker.reset()
        tracker.estimate(frame, mask)  # one real registration to seed pose_last for tracking
        print(f'Measuring track_one() FPS over {args.track_iters} calls '
             f'(iteration={args.track_refine_iter})...', flush=True)
        track_times = measure_track_fps(tracker, frame, args.track_iters)
        summarize('FoundationPose track_one()', track_times)
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
