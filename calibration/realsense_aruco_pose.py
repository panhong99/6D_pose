"""Detect an ArUco marker directly from a RealSense camera.

This does not require a RealSense ROS package or ROS topics. It uses the
intrinsics reported by the connected camera and prints the marker pose in the
color-camera frame.

Run from an environment containing pyrealsense2 and OpenCV-contrib:
  python calibration/realsense_aruco_pose.py --marker-id 0 --marker-size 0.05
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker-id", type=int, required=True)
    parser.add_argument("--marker-size", type=float, required=True, help="marker side in meters")
    parser.add_argument(
        "--dictionary",
        default="DICT_5X5_100",
        choices=["DICT_4X4_50", "DICT_5X5_50", "DICT_5X5_100", "DICT_6X6_250"],
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--flange-json", default="/tmp/doosan_flange_pose.json")
    args = parser.parse_args()

    aruco = cv2.aruco
    dictionary_id = getattr(aruco, args.dictionary)
    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    detector_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, detector_params)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_profile.get_intrinsics()
    camera_matrix = np.array(
        [[intrinsics.fx, 0.0, intrinsics.ppx],
         [0.0, intrinsics.fy, intrinsics.ppy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.asarray(intrinsics.coeffs, dtype=np.float64)
    half = args.marker_size / 2.0
    object_points = np.array(
        [[-half, half, 0.0], [half, half, 0.0],
         [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float32,
    )

    print("color intrinsics:")
    print(camera_matrix)
    print(
        f"marker_id={args.marker_id}, marker_size_m={args.marker_size}, "
        f"dictionary={args.dictionary}"
    )
    print("Press q or ESC to stop.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = np.asanyarray(frames.get_color_frame().get_data())
            corners, ids, _ = detector.detectMarkers(color)

            if ids is not None:
                print(f"detected_ids={ids.flatten().tolist()}")
                for marker_corners, marker_id in zip(corners, ids.flatten()):
                    if int(marker_id) != args.marker_id:
                        continue
                    image_points = marker_corners.reshape(4, 2).astype(np.float32)
                    ok, rvec, tvec = cv2.solvePnP(
                        object_points,
                        image_points,
                        camera_matrix,
                        distortion,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE,
                    )
                    if ok:
                        x, y, z = tvec.reshape(3)
                        flange_text = "flange_unavailable"
                        if os.path.exists(args.flange_json):
                            try:
                                with open(args.flange_json, encoding="utf-8") as f:
                                    flange = json.load(f)["pose_mm_deg"]
                                flange_text = "flange_base_mm_deg=[" + ", ".join(
                                    f"{float(v):.3f}" for v in flange
                                ) + "]"
                            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                                flange_text = "flange_unavailable"
                        print(
                            f"marker={marker_id} "
                            f"t_camera_marker_m=[{x:.4f}, {y:.4f}, {z:.4f}] "
                            f"rvec=[{rvec[0,0]:.4f}, {rvec[1,0]:.4f}, {rvec[2,0]:.4f}] "
                            f"{flange_text}"
                        )
                    aruco.drawDetectedMarkers(color, [marker_corners])

            cv2.imshow("RealSense ArUco", color)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            time.sleep(0.001)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
