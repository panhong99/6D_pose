"""D455 -> Grounding DINO + SAM2 auto recovery -> FoundationPose -> optional UDP."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
import time

# Support both direct script execution and python -m from the workspace.
_ROOT = Path(__file__).resolve().parents[1]
for _directory in (_ROOT.parent, _ROOT):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))


def parse_args(argv=None):
    import math
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh_file', type=Path,
                        default=_ROOT / 'demo_data/077_rubiks_cube/google_16k/textured_57mm.obj')
    parser.add_argument('--mesh_scale', type=float, default=1.0, help='CAD units to meters: mm=0.001')
    checkpoint = _ROOT / 'weights/sam2.1_hiera_tiny_kimm.pt'
    parser.add_argument('--sam_checkpoint', '--sam2_checkpoint', dest='sam_checkpoint', type=Path,
                        default=checkpoint if checkpoint.is_file() else None,
                        help='Local SAM2 weights; defaults to existing weights/ checkpoint, otherwise HF')
    parser.add_argument('--sam2_model', default='facebook/sam2.1-hiera-tiny')
    parser.add_argument('--sam2_config', default='configs/sam2.1/sam2.1_hiera_t.yaml')
    parser.add_argument('--dino_model', default='IDEA-Research/grounding-dino-tiny')
    parser.add_argument('--prompt', default="rubik's cube")
    parser.add_argument('--box_threshold', type=float, default=0.3)
    parser.add_argument('--text_threshold', type=float, default=0.25)
    parser.add_argument('--validation_interval', type=float, default=0.5,
                        help='Seconds between DINO-only tracking checks; 0 disables geometry/watchdog guards')
    parser.add_argument('--retry_interval', type=float, default=0.1,
                        help='Seconds between unsuccessful recovery attempts (after each attempt)')
    parser.add_argument('--loss_patience', type=int, default=5,
                        help='Consecutive soft validation failures tolerated before forcing '
                             'a full re-detect; a double DINO miss always forces one immediately')
    parser.add_argument('--serial', default='')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--max_depth', type=float, default=3.0)
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--max_frame_gap', type=float, default=1.0,
                        help='Automatically recover after a gap between tracked frames; excludes registration')
    parser.add_argument('--drift_score_ratio', type=float, default=0.6,
                        help='Score drop threshold relative to rolling baseline; 0 disables score-based loss')
    parser.add_argument('--debug_dir', type=Path, default=root / 'debug_live_kimm')
    parser.add_argument('--verbose_pose', action='store_true', help='Print every 4x4 pose')
    parser.add_argument('--publish_pose', action='store_true', default=True,
                        help='UDP pose output for ROS2 udp_pose_bridge (always enabled)')
    parser.add_argument('--udp_host', default='127.0.0.1')
    parser.add_argument('--udp_port', type=int, default=5005)
    parser.add_argument('--frame_id', default='camera_color_optical_frame')
    args = parser.parse_args(argv)
    for name in ('mesh_file', 'sam_checkpoint'):
        path = getattr(args, name)
        if path is None:
            continue
        path = path.expanduser().resolve()
        if not path.is_file():
            parser.error(f'{name} does not exist: {path}')
        setattr(args, name, path)
    for name in ('mesh_scale', 'max_depth', 'max_frame_gap', 'width', 'height', 'fps',
                 'est_refine_iter', 'track_refine_iter'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'{name} must be positive and finite')
    for name in ('drift_score_ratio', 'box_threshold', 'text_threshold'):
        if not math.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) <= 1:
            parser.error(f'{name} must be finite and in [0, 1]')
    if not math.isfinite(args.retry_interval) or args.retry_interval < 0:
        parser.error('retry_interval must be nonnegative and finite')
    if not math.isfinite(args.validation_interval) or args.validation_interval < 0:
        parser.error('validation_interval must be nonnegative and finite')
    if args.loss_patience < 1:
        parser.error('loss_patience must be at least 1')
    if not args.prompt.strip().strip('.'):
        parser.error('prompt must describe an object')
    return args


def main():
    args = parse_args()
    import cv2
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('FoundationPose requires CUDA in the foundationpose environment')
    from FoundationPose.real_time_project.d455_source_kimm import D455Source
    from FoundationPose.real_time_project.pose_tracker_kimm import PoseTracker
    from FoundationPose.real_time_project.recovery_tracker_kimm import RecoveryTracker
    from FoundationPose.real_time_project.detector_comparison.pipelines.pipeline_grounding_dino_sam2 import GroundingDINOSAM2Pipeline
    from pose_sender_kimm import PoseSender
    from FoundationPose.real_time_project.capture_writer_kimm import CaptureWriter
    from FoundationPose.real_time_project.mask_viz_kimm import mask_overlay

    print('Loading Grounding DINO + SAM2 and FoundationPose ...', flush=True)
    detector = GroundingDINOSAM2Pipeline(
        sam2_checkpoint=args.sam_checkpoint, sam2_model=args.sam2_model,
        sam2_config=args.sam2_config, dino_model=args.dino_model, device='cuda',
        box_threshold=args.box_threshold, text_threshold=args.text_threshold)
    tracker = PoseTracker(args.mesh_file, args.mesh_scale, args.debug_dir,
                          args.est_refine_iter, args.track_refine_iter, args.drift_score_ratio)
    recovery = RecoveryTracker(detector, tracker, args.prompt, args.retry_interval,
                               args.max_frame_gap, args.validation_interval, args.loss_patience)
    captures = CaptureWriter(args.debug_dir, dict(
        mesh_file=str(args.mesh_file), mesh_scale=args.mesh_scale,
        sam_checkpoint=str(args.sam_checkpoint) if args.sam_checkpoint else None,
        sam2_model=args.sam2_model, dino_model=args.dino_model, prompt=args.prompt,
        box_threshold=args.box_threshold, text_threshold=args.text_threshold,
        frame_id=args.frame_id, est_refine_iter=args.est_refine_iter,
        track_refine_iter=args.track_refine_iter))

    with ExitStack() as resources:
        camera = D455Source(args.width, args.height, args.fps, args.serial, args.max_depth)
        resources.callback(camera.close)
        resources.callback(cv2.destroyAllWindows)
        sender = None
        if args.publish_pose:
            sender = PoseSender(args.udp_host, args.udp_port, args.frame_id)
            resources.callback(sender.close)
            print(f'Pose UDP -> {args.udp_host}:{args.udp_port}', flush=True)
        print('Automatic detection enabled. S: force recovery | Q/ESC: quit', flush=True)
        previous_status = None
        while True:
            frame = camera.read()
            start = time.perf_counter()
            outcome = recovery.process(frame)
            pose, detection = outcome['pose'], outcome['detection']
            image = frame.rgb.copy()
            status = outcome['status']
            if detection is not None:
                status += f" | detect+mask={detection['latency_ms']:.0f}ms"
                if detection['success']:
                    # Capture exactly the frame/mask used by register, including failed registrations.
                    capture = captures.save(frame, detection['mask'], detection['bbox'], True)
                    if outcome['registered']:
                        captures.result(capture, pose=pose)
                    elif outcome['error']:
                        captures.result(capture, error=outcome['error'])
                    image = mask_overlay(image, detection['mask'])
            if pose is not None:
                image = tracker.draw(image, frame.K, pose)
                # A suspect pose is still drawn (so tracking looks continuous) but
                # withheld from the robot until it clears loss_patience or is
                # confirmed again -- an unverified pose should not drive hardware.
                if sender and not outcome['suspect']:
                    sender.send(pose, frame.identifier, frame.timestamp_ns)
                xyz = pose[:3, 3]
                status += (f' | xyz={xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}m '
                           f'| score={tracker.last_score:.2f} | {time.perf_counter()-start:.3f}s')
                if args.verbose_pose:
                    print(f'frame={frame.identifier} pose=\n{pose}', flush=True)
            if outcome['status'] != previous_status:
                print(status, flush=True)
                previous_status = outcome['status']
            display = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(display, 'Auto recovery | S: force redetect | Q/ESC: quit', (10, 47),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.imshow('FoundationPose live kimm', display)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                recovery.reset()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
