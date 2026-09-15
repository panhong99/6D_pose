"""D455 live FoundationPose++ test using Qwen2-VL + SAM-HQ initialization.

Start the two APIs first, then run this file from the FoundationPose environment.
"""
import argparse, os, re, sys, tempfile
from pathlib import Path
import cv2
import numpy as np
import requests
import pyrealsense2 as rs

PLUS_ROOT = Path('/home/panhong/pan/FoundationPose-plus-plus')
if PLUS_ROOT.exists():
    sys.path.insert(0, str(PLUS_ROOT))
    sys.path.insert(0, str(PLUS_ROOT / 'FoundationPose'))

from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor, dr
import trimesh

def bbox_from_qwen(frame_path, object_name, url):
    out = requests.post(url, json={
        'image_paths': [str(frame_path)],
        'text_input': f'请在图片中找到{object_name}，只返回(x,y,x,y)格式的锚框，坐标范围为0到1000。'
    }, timeout=180).json()['output'][0]
    m = re.search(r'\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)', out)
    if not m:
        raise RuntimeError(f'Qwen bbox parse failed: {out}')
    x1,y1,x2,y2 = map(int, m.groups())
    h,w = cv2.imread(str(frame_path)).shape[:2]
    return [int(x1*w/1000), int(y1*h/1000), int((x2-x1)*w/1000), int((y2-y1)*h/1000)]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mesh', required=True)
    p.add_argument('--object', required=True)
    p.add_argument('--mesh-scale', type=float, default=1.0)
    p.add_argument('--qwen-url', default='http://127.0.0.1:9003/qwen2_vl')
    p.add_argument('--sam-url', default='http://127.0.0.1:9002/hq_sam')
    p.add_argument('--sam-checkpoint', default=str(PLUS_ROOT/'sam-hq/pretrained_checkpoints/sam_hq_vit_h.pth'))
    p.add_argument('--width', type=int, default=640); p.add_argument('--height', type=int, default=480); p.add_argument('--fps', type=int, default=30)
    p.add_argument('--register-iter', type=int, default=5); p.add_argument('--track-iter', type=int, default=2)
    a = p.parse_args()

    mesh = trimesh.load(a.mesh, force='mesh'); mesh.apply_scale(a.mesh_scale)
    K = None
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_stream(rs.stream.color, a.width, a.height, rs.format.rgb8, a.fps)
    config.enable_stream(rs.stream.depth, a.width, a.height, rs.format.z16, a.fps)
    profile = pipeline.start(config); align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]], dtype=np.float32)
    print('D455 ready. Place the object in view and press q to quit.', flush=True)
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    estimator = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                                scorer=scorer, refiner=refiner, glctx=dr.RasterizeCudaContext())
    registered = False; tmp = Path(tempfile.mkdtemp(prefix='fp_d455_')); frame_no = 0
    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_f, depth_f = frames.get_color_frame(), frames.get_depth_frame()
            if not color_f or not depth_f: continue
            rgb = np.asanyarray(color_f.get_data()).copy()
            depth = np.asanyarray(depth_f.get_data()).astype(np.float32) * depth_scale
            depth[(depth < .001) | (depth > 5)] = 0
            if not registered:
                path = tmp/'first.png'; cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                bbox = bbox_from_qwen(path, a.object, a.qwen_url)
                mask_path = tmp/'mask.png'
                r = requests.post(a.sam_url, json={'frame_path':str(path), 'bbox_xywh':bbox, 'output_mask_path':str(mask_path)}, timeout=180)
                r.raise_for_status(); mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
                pose = estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=mask.astype(np.uint8)*255, iteration=a.register_iter)
                registered = True; print('Registered. Tracking started.', flush=True)
            else:
                pose = estimator.track_one(rgb=rgb, depth=depth, K=K, iteration=a.track_iter)
            xyz = np.asarray(pose).reshape(4,4)[:3,3]
            cv2.putText(rgb, f'xyz {xyz[0]:.3f} {xyz[1]:.3f} {xyz[2]:.3f} m', (10,25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,255,0), 2)
            cv2.imshow('FoundationPose++ D455', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xff == ord('q'): break
            frame_no += 1
    finally:
        pipeline.stop(); cv2.destroyAllWindows()

if __name__ == '__main__': main()
