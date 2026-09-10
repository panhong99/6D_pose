"""Build destination-to-source pixel maps for the installed RealSense SDK."""
import cv2
import numpy as np
import pyrealsense2 as rs


def rectification_maps(intr, K):
    if intr.model == rs.distortion.none or not np.any(np.asarray(intr.coeffs) != 0):
        return None, None
    if intr.model == rs.distortion.brown_conrady:
        return cv2.initUndistortRectifyMap(
            K, np.asarray(intr.coeffs), None, K,
            (intr.width, intr.height), cv2.CV_32FC1)
    if intr.model not in (rs.distortion.inverse_brown_conrady,
                          rs.distortion.modified_brown_conrady):
        raise ValueError(f'Unsupported color distortion: {intr.model}')
    # A rectified output pixel defines an ideal ray. Ask the SDK where that ray
    # lands in the original image. Do not reinterpret inverse coefficients as
    # OpenCV Brown coefficients. This lookup is computed only once at startup.
    pixels = np.empty((intr.height, intr.width, 2), dtype=np.float32)
    for v in range(intr.height):
        y = float((v - K[1, 2]) / K[1, 1])
        for u in range(intr.width):
            x = float((u - K[0, 2]) / K[0, 0])
            pixels[v, u] = rs.rs2_project_point_to_pixel(intr, [x, y, 1.0])
    if not np.isfinite(pixels).all():
        raise ValueError('Camera calibration produced non-finite rectification maps')
    return pixels[..., 0].copy(), pixels[..., 1].copy()
