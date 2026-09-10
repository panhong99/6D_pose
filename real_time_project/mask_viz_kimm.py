"""Mask visualization: contrast-safe regardless of object color."""
import cv2
import numpy as np


def mask_overlay(rgb, mask, contour_color=(255, 0, 255), dim=0.3, thickness=2):
    overlay = rgb.copy()
    overlay[~mask] = (dim * overlay[~mask]).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, contour_color, thickness)
    return overlay
