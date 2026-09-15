"""D455 -> Grounding DINO + SAM2 auto recovery -> FoundationPose -> optional UDP."""
import argparse
from contextlib import ExitStack
import os
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
    parser.add_argument('--model_cache_dir', type=Path, default=_ROOT / 'weights/huggingface_cache',
                        help='Persistent Hugging Face cache; models download here only on first use')
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
    parser.add_argument('--detector_bbox_margin', type=float, default=0.5,
                        help='Extra fraction of DINO bbox allowed around the FoundationPose center')
    parser.add_argument('--auto_recovery', action='store_true',
                        help='Automatically re-detect after tracking is lost; default waits for S')
    parser.add_argument('--serial', default='')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--max_depth', type=float, default=3.0)
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--max_frame_gap', type=float, default=1.0,
                        help='Automatically recover after a gap between tracked frames; excludes registration')
    parser.add_argument('--drift_score_ratio', type=float, default=0.3,
                        help='Minimum score fraction of rolling baseline; 0 disables score-based loss')
    parser.add_argument('--debug_dir', type=Path, default=None,
                        help='Optional debug output directory; disabled by default')
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
    # Keep HF/Transformers downloads inside the project so later launches reuse
    # the exact same files instead of relying on a transient runtime cache.
    args.model_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(args.model_cache_dir)
    os.environ['HF_HUB_CACHE'] = str(args.model_cache_dir / 'hub')
    os.environ['TRANSFORMERS_CACHE'] = str(args.model_cache_dir / 'transformers')
    import cv2
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('FoundationPose requires CUDA in the foundationpose environment')
    from FoundationPose.real_time_project.d455_source_kimm import D455Source
    from FoundationPose.real_time_project.pose_tracker_kimm import PoseTracker
    from FoundationPose.real_time_project.recovery_tracker_kimm import RecoveryTracker
    from FoundationPose.real_time_project.detector_comparison.pipelines.pipeline_grounding_dino_sam2 import GroundingDINOSAM2Pipeline
    from pose_sender_kimm import PoseSender

    print('Loading Grounding DINO + SAM2 and FoundationPose ...', flush=True)
    detector = GroundingDINOSAM2Pipeline(
        sam2_checkpoint=args.sam_checkpoint, sam2_model=args.sam2_model,
        sam2_config=args.sam2_config, dino_model=args.dino_model, device='cuda',
        box_threshold=args.box_threshold, text_threshold=args.text_threshold)

    tracker = PoseTracker(args.mesh_file, args.mesh_scale, args.debug_dir,
                          args.est_refine_iter, args.track_refine_iter, args.drift_score_ratio,
                          args.detector_bbox_margin)

    recovery = RecoveryTracker(detector, tracker, args.prompt, args.retry_interval,
                               args.max_frame_gap, args.validation_interval, args.loss_patience)
    recovery.set_recovery_mode(args.auto_recovery)

    with ExitStack() as resources:
        camera = D455Source(args.width, args.height, args.fps, args.serial, args.max_depth)
        resources.callback(camera.close)
        resources.callback(cv2.destroyAllWindows)
        sender = None
        if args.publish_pose:
            sender = PoseSender(args.udp_host, args.udp_port, args.frame_id)
            resources.callback(sender.close)
        print(f'Pose UDP -> {args.udp_host}:{args.udp_port}', flush=True)
        cv2.namedWindow('FoundationPose live kimm', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('FoundationPose live kimm', 1600, 1200)
        mode_text = 'automatic recovery' if args.auto_recovery else 'manual recovery'
        print(f'Detection enabled ({mode_text}). S: search/re-detect | Q/ESC: quit', flush=True)
        previous_status = None
        while True:
            frame = camera.read()
            start = time.perf_counter()
            outcome = recovery.process(frame)
            pose, detection = outcome['pose'], outcome['detection']
            image = frame.rgb.copy()
            status = outcome['status']
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
            recovery_hint = ('Auto recovery' if args.auto_recovery else 'Manual recovery')
            cv2.putText(display, f'{recovery_hint} | S: search | Q/ESC: quit', (10, 47),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.imshow('FoundationPose live kimm', display)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                recovery.request_search()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
