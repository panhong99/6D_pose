"""Live D455 preview for framing the real experiment setup. Read-only: camera only.

Keys:  s = save colour + depth + intrinsics to data/real_env/   q / Esc = quit
Run:   conda run -n foundationpose python calibration/real_d455_preview.py
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

OUT = Path(__file__).resolve().parent / 'data' / 'real_env'
WIDTH, HEIGHT, FPS = 1280, 720, 30
MAX_DEPTH_M = 3.0


def connected_d455_serials():
    return [device.get_info(rs.camera_info.serial_number)
            for device in rs.context().query_devices()
            if 'D455' in device.get_info(rs.camera_info.name)]


def preview_multiple(serials):
    """Show several cameras side by side without running the pose pipeline or saving frames."""
    pipelines = []
    try:
        align = rs.align(rs.stream.color)
        for serial in serials:
            pipeline, config = rs.pipeline(), rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
            profile = pipeline.start(config)
            scale = profile.get_device().first_depth_sensor().get_depth_scale()
            pipelines.append((pipeline, serial, scale))
            print(f'D455 {serial} preview started (640x480, 15 FPS)', flush=True)

        window = 'D455 cameras side by side  [q / Esc to close]'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1600, 500)
        while True:
            panels = []
            for pipeline, serial, scale in pipelines:
                frames = align.process(pipeline.wait_for_frames())
                color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    panel = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(panel, f'{serial}: waiting for frames', (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
                    panels.append(panel)
                    continue
                color = np.asanyarray(color_frame.get_data()).copy()
                depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * scale
                valid = 100 * float((depth_m > 0).mean())
                h, w = color.shape[:2]
                cv2.drawMarker(color, (w // 2, h // 2), (0, 255, 0), cv2.MARKER_CROSS, 30, 1)
                cv2.putText(color, f'D455 {serial} | depth valid {valid:.0f}%', (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                panels.append(color)
            cv2.imshow(window, np.hstack(panels))
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        for pipeline, _, _ in reversed(pipelines):
            pipeline.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true', help='preview every connected D455 side by side')
    group.add_argument('--serial', action='append', help='preview this D455; repeat to show several together')
    args = parser.parse_args()
    if args.all or args.serial:
        available = connected_d455_serials()
        serials = available if args.all else args.serial
        missing = [serial for serial in serials if serial not in available]
        if missing:
            raise SystemExit(f'D455 not connected: {", ".join(missing)}; found {", ".join(available) or "none"}')
        if not serials:
            raise SystemExit('No connected D455 cameras found')
        preview_multiple(serials)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    serial = profile.get_device().get_info(rs.camera_info.serial_number)
    align = rs.align(rs.stream.color)
    window = 'D455 preview  [s] save  [q] quit'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1600, 450)
    saved, flash_until = 0, 0.0
    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

            vis_depth = cv2.applyColorMap(
                cv2.convertScaleAbs(np.clip(depth_m, 0, MAX_DEPTH_M), alpha=255 / MAX_DEPTH_M), cv2.COLORMAP_JET)
            vis_depth[depth_m == 0] = 0
            view = color.copy()
            h, w = view.shape[:2]
            cx, cy = w // 2, h // 2
            cv2.line(view, (cx - 25, cy), (cx + 25, cy), (0, 255, 0), 1)
            cv2.line(view, (cx, cy - 25), (cx, cy + 25), (0, 255, 0), 1)
            patch = depth_m[cy - 5:cy + 6, cx - 5:cx + 6]
            patch = patch[patch > 0]
            centre = f'{np.median(patch):.3f} m' if patch.size else 'no depth'
            valid = 100.0 * float((depth_m > 0).mean())
            cv2.putText(view, f'centre depth {centre} | valid depth {valid:.0f}% | saved {saved}',
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if time.time() < flash_until:
                cv2.rectangle(view, (0, 0), (w - 1, h - 1), (0, 255, 0), 12)
            cv2.imshow(window, np.hstack([view, vis_depth]))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                stamp = time.strftime('%Y%m%d_%H%M%S')
                intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
                cv2.imwrite(str(OUT / f'real_color_{stamp}.png'), color)
                np.save(OUT / f'real_depth_{stamp}.npy', depth_m)
                (OUT / f'real_intrinsics_{stamp}.json').write_text(json.dumps(dict(
                    K=[[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dist=list(intr.coeffs),
                    width=intr.width, height=intr.height, depth_scale=depth_scale, serial=serial), indent=1))
                saved += 1
                flash_until = time.time() + 0.4
                print(f'saved {stamp}', flush=True)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
