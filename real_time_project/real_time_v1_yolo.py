"""D455 -> YOLOE-seg -> FoundationPose + Cutie/Kalman -> ROS UDP."""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE / 'detector_comparison'))

from d455_source_kimm import D455Source
from pose_tracker_kimm import PoseTracker
from recovery_tracker_kimm import RecoveryTracker
from detector_comparison.pipelines.pipeline_yoloe import YOLOEPipeline
from pose_sender_kimm import PoseSender


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mesh_file', required=True)
    p.add_argument('--mesh_scale', type=float, default=1.0)
    p.add_argument('--prompt', default="rubik's cube")
    p.add_argument('--yoloe_checkpoint', default='real_time_project/detector_comparison/yoloe-11s-seg.pt')
    p.add_argument('--conf', type=float, default=0.15)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--serial', default='')
    p.add_argument('--width', type=int, default=640); p.add_argument('--height', type=int, default=480); p.add_argument('--fps', type=int, default=30)
    p.add_argument('--register_iter', type=int, default=5); p.add_argument('--track_iter', type=int, default=2)
    p.add_argument('--udp_host', default='127.0.0.1'); p.add_argument('--udp_port', type=int, default=5005)
    p.add_argument('--frame_id', default='camera_color_optical_frame')
    p.add_argument('--debug_dir', default='real_time_project/debug_yolo')
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for FoundationPose')
    detector = YOLOEPipeline(a.yoloe_checkpoint, device='cuda', conf=a.conf, imgsz=a.imgsz)
    tracker = PoseTracker(a.mesh_file, a.mesh_scale, a.debug_dir, a.register_iter, a.track_iter, 0.0, 0.5)
    recovery = RecoveryTracker(detector, tracker, a.prompt, retry_interval=0.1,
                               max_frame_gap=1.0, validation_interval=0.0, loss_patience=10)
    camera = D455Source(a.width, a.height, a.fps, a.serial)
    sender = PoseSender(a.udp_host, a.udp_port, a.frame_id)
    cv2.namedWindow('FoundationPose YOLOE D455', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('FoundationPose YOLOE D455', 1600, 1200)
    print('YOLOE + FoundationPose + Cutie/Kalman ready | S: re-detect | Q: quit', flush=True)
    try:
        while True:
            frame = camera.read()
            out = recovery.process(frame)
            image = frame.rgb.copy()
            if out['pose'] is not None:
                image = tracker.draw(image, frame.K, out['pose'])
                sender.send(out['pose'], frame.identifier, frame.timestamp_ns)
                xyz = np.asarray(out['pose'])[:3, 3]
                cv2.putText(image, f'xyz {xyz[0]:.3f} {xyz[1]:.3f} {xyz[2]:.3f}m', (10,25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 2)
            cv2.putText(image, out['status'], (10,48), cv2.FONT_HERSHEY_SIMPLEX, .45, (0,255,0), 1)
            cv2.imshow('FoundationPose YOLOE D455', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), 27): break
            if key == ord('s'): recovery.request_search()
    finally:
        sender.close(); camera.close(); cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
