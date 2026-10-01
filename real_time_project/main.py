"""D455 -> detector (YOLOE-seg | Grounding DINO + SAM2) -> FoundationPose + Cutie/Kalman -> UDP.

python real_time_project/main.py --detector yoloe
python real_time_project/main.py --detector dino
S: re-detect | Q/ESC: quit
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _directory in (_ROOT, _HERE):
    sys.path.insert(0, str(_directory))

from real_time_utils import D455Source, PoseTracker, RecoveryTracker
from pose_sender_kimm import PoseSender


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--detector', choices=['yoloe', 'dino'], default='yoloe',
                   help='yoloe: YOLOE-seg (box+mask in one pass); dino: Grounding DINO box -> SAM2 mask')
    p.add_argument('--mesh_file', type=Path,
                   default=_ROOT / 'demo_data/077_rubiks_cube/google_16k/textured_57mm.obj')
    p.add_argument('--mesh_scale', type=float, default=1.0, help='CAD units to meters: mm=0.001')
    p.add_argument('--prompt', default="rubik's cube")

    g = p.add_argument_group('YOLOE (--detector yoloe)')
    g.add_argument('--yoloe_checkpoint', default=str(_ROOT / 'yoloe-v8l-seg.pt'),
                   help='text-prompt yoloe-*-seg.pt (prompt-free -pf is rejected); a bare name like '
                        'yoloe-11s-seg.pt is auto-downloaded to the current directory')
    g.add_argument('--conf', type=float, default=0.15)
    g.add_argument('--imgsz', type=int, default=640)

    g = p.add_argument_group('Grounding DINO + SAM2 (--detector dino)')
    checkpoint = _ROOT / 'weights/sam2.1_hiera_tiny_kimm.pt'
    g.add_argument('--sam_checkpoint', '--sam2_checkpoint', dest='sam_checkpoint', type=Path,
                   default=checkpoint if checkpoint.is_file() else None,
                   help='Local SAM2 weights; defaults to weights/ checkpoint if present, else Hugging Face')
    g.add_argument('--sam2_model', default='facebook/sam2.1-hiera-tiny')
    g.add_argument('--sam2_config', default='configs/sam2.1/sam2.1_hiera_t.yaml')
    g.add_argument('--dino_model', default='IDEA-Research/grounding-dino-tiny')
    g.add_argument('--box_threshold', type=float, default=0.3)
    g.add_argument('--text_threshold', type=float, default=0.25)
    g.add_argument('--model_cache_dir', type=Path, default=_ROOT / 'weights/huggingface_cache',
                   help='Persistent Hugging Face cache; models download here only on first use')

    g = p.add_argument_group('FoundationPose tracking')
    g.add_argument('--register_iter', '--est_refine_iter', dest='register_iter', type=int, default=10,
                   help='Initial registration iterations (FoundationPose++ default)')
    g.add_argument('--track_iter', '--track_refine_iter', dest='track_iter', type=int, default=3,
                   help='Refiner iterations per tracked frame (FoundationPose++ default 5; '
                        'fewer = faster, less accurate)')
    g.add_argument('--score_interval', type=int, default=3,
                   help='Run the drift-check scorer every Nth tracked frame (1 = every frame)')
    g.add_argument('--drift_score_ratio', type=float, default=0.3,
                   help='Minimum score fraction of rolling baseline; 0 disables score-based loss')

    g = p.add_argument_group('Recovery')
    g.add_argument('--auto_recovery', action=argparse.BooleanOptionalAction, default=True,
                   help='Automatically re-detect after tracking is lost (default on); '
                        '--no-auto_recovery waits for S')
    g.add_argument('--validation_interval', type=float, default=0.5,
                   help='Seconds between detector bbox checks of the tracked pose; 0 disables '
                        'geometry/detector guards')
    g.add_argument('--loss_patience', type=int, default=5,
                   help='Consecutive suspect frames tolerated before forcing a re-detect')
    g.add_argument('--retry_interval', type=float, default=0.1,
                   help='Seconds between unsuccessful recovery attempts')
    g.add_argument('--max_frame_gap', type=float, default=1.0,
                   help='Recover after a gap between tracked frames (seconds)')
    g.add_argument('--detector_bbox_margin', type=float, default=0.5,
                   help='Extra fraction of detector bbox allowed around the pose center')
    g.add_argument('--relock_iterations', type=int, default=0,
                   help='Speed shortcut, off by default (0): track_one()-style refine iterations '
                        'for a prior-seeded relock tried before register()')
    g.add_argument('--roi_margin', type=float, default=1.5,
                   help='With --roi_patience > 0, recovery detection crops to the last known '
                        'bbox padded by this multiple of its size')
    g.add_argument('--roi_patience', type=int, default=0,
                   help='Speed shortcut, off by default (0 = always full-frame search): '
                        'consecutive ROI-crop misses before falling back to full frame')

    g = p.add_argument_group('Camera / output')
    g.add_argument('--serial', default='')
    g.add_argument('--width', type=int, default=640)
    g.add_argument('--height', type=int, default=480)
    g.add_argument('--fps', type=int, default=30)
    g.add_argument('--max_depth', type=float, default=3.0)
    g.add_argument('--udp_host', default='127.0.0.1')
    g.add_argument('--udp_port', type=int, default=5005)
    g.add_argument('--frame_id', default='camera_color_optical_frame')
    g.add_argument('--debug_dir', type=Path, default=None,
                   help='Optional debug output directory; disabled by default')
    g.add_argument('--verbose_pose', action='store_true', help='Print every 4x4 pose')
    args = p.parse_args(argv)

    for name in ('mesh_file', 'sam_checkpoint'):
        path = getattr(args, name)
        if path is not None:
            path = path.expanduser().resolve()
            if not path.is_file():
                p.error(f'{name} does not exist: {path}')
            setattr(args, name, path)
    for name in ('mesh_scale', 'max_depth', 'max_frame_gap', 'width', 'height', 'fps', 'imgsz',
                 'register_iter', 'track_iter', 'score_interval', 'roi_margin', 'loss_patience'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f'{name} must be positive and finite')
    for name in ('relock_iterations', 'roi_patience', 'retry_interval', 'validation_interval'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            p.error(f'{name} must be nonnegative and finite (0 disables it)')
    for name in ('drift_score_ratio', 'conf', 'box_threshold', 'text_threshold'):
        if not np.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) <= 1:
            p.error(f'{name} must be finite and in [0, 1]')
    if not args.prompt.strip().strip('.'):
        p.error('prompt must describe an object')
    return args


def build_detector(args):
    """Only the selected detector's dependencies are imported."""
    if args.detector == 'yoloe':
        from detectors import YOLOEPipeline
        print('Loading YOLOE and FoundationPose ...', flush=True)
        return YOLOEPipeline(args.yoloe_checkpoint, device='cuda', conf=args.conf, imgsz=args.imgsz)

    # Keep HF/Transformers downloads inside the project so later launches reuse them.
    args.model_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(args.model_cache_dir)
    os.environ['HF_HUB_CACHE'] = str(args.model_cache_dir / 'hub')
    os.environ['TRANSFORMERS_CACHE'] = str(args.model_cache_dir / 'transformers')
    from detectors import GroundingDINOSAM2Pipeline
    print('Loading Grounding DINO + SAM2 and FoundationPose ...', flush=True)
    return GroundingDINOSAM2Pipeline(
        sam2_checkpoint=args.sam_checkpoint, sam2_model=args.sam2_model,
        sam2_config=args.sam2_config, dino_model=args.dino_model, device='cuda',
        box_threshold=args.box_threshold, text_threshold=args.text_threshold)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('FoundationPose requires CUDA in the foundationpose environment')
    detector = build_detector(args)
    tracker = PoseTracker(args.mesh_file, args.mesh_scale, args.debug_dir, args.register_iter,
                          args.track_iter, args.drift_score_ratio, args.detector_bbox_margin,
                          args.relock_iterations, args.score_interval)
    recovery = RecoveryTracker(detector, tracker, args.prompt, args.retry_interval,
                               args.max_frame_gap, args.validation_interval, args.loss_patience,
                               args.roi_margin, args.roi_patience)
    recovery.set_recovery_mode(args.auto_recovery)

    camera = D455Source(args.width, args.height, args.fps, args.serial, args.max_depth)
    sender = PoseSender(args.udp_host, args.udp_port, args.frame_id)
    window = f'FoundationPose {args.detector.upper()} D455'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1600, 1200)
    mode = 'automatic' if args.auto_recovery else 'manual'
    print(f'{args.detector} + FoundationPose + Cutie/Kalman ready ({mode} recovery) | '
          f'Pose UDP -> {args.udp_host}:{args.udp_port} | S: re-detect | Q/ESC: quit', flush=True)
    stage_ms, frames, loop_start, previous_status = np.zeros(3), 0, time.perf_counter(), None
    try:
        while True:
            t0 = time.perf_counter()
            frame = camera.read()
            t1 = time.perf_counter()
            out = recovery.process(frame)
            t2 = time.perf_counter()
            image = frame.rgb.copy()
            if out['pose'] is not None:
                image = tracker.draw(image, frame.K, out['pose'])
                # A suspect pose is drawn but withheld from the robot until confirmed.
                if not out['suspect']:
                    sender.send(out['pose'], frame.identifier, frame.timestamp_ns)
                xyz = np.asarray(out['pose'])[:3, 3]
                cv2.putText(image, f'xyz {xyz[0]:.3f} {xyz[1]:.3f} {xyz[2]:.3f}m', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
                if args.verbose_pose:
                    print(f'frame={frame.identifier} pose=\n{out["pose"]}', flush=True)
            if out['status'] != previous_status:
                print(out['status'], flush=True)
                previous_status = out['status']
            cv2.putText(image, out['status'], (10, 48), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 0), 1)
            cv2.imshow(window, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xff
            t3 = time.perf_counter()
            # Per-stage timing so lag can be attributed: camera read (align+rectify),
            # pose pipeline, and draw/UDP/display. Printed every 30 frames.
            stage_ms += [(t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3]
            frames += 1
            if frames == 30:
                avg = stage_ms / frames
                print(f'[TIMING] fps={frames / (t3 - loop_start):.1f} read={avg[0]:.1f}ms '
                      f'process={avg[1]:.1f}ms draw+show={avg[2]:.1f}ms | frame age at display='
                      f'{(time.time_ns() - frame.timestamp_ns) / 1e6:.0f}ms', flush=True)
                stage_ms[:], frames, loop_start = 0, 0, time.perf_counter()
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                recovery.request_search()
    finally:
        sender.close()
        camera.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
