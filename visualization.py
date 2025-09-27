# visualization.py
# Functions for visualizing the lane detection output.

import cv2
import numpy as np
# Note: This file doesn't import 'config' directly anymore,
# it gets all necessary info from the 'data' dictionary.

def draw_lane_overlay(data):
    original_frame = data["original_frame"].copy()
    left_fit, right_fit = data["left_fit"], data["right_fit"]
    inv_matrix = data["inv_matrix"]

    if left_fit is None or right_fit is None:
        return original_frame

    h, w, _ = original_frame.shape
    persp_h, persp_w = data["warped_edges"].shape[:2]

    warped_overlay = np.zeros((persp_h, persp_w, 3), dtype=np.uint8)
    ploty = np.linspace(0, persp_h - 1, persp_h)

    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(warped_overlay, np.int_([pts]), (0, 255, 0))
    unwarped_overlay = cv2.warpPerspective(warped_overlay, inv_matrix, (w, h))
    return cv2.addWeighted(original_frame, 1, unwarped_overlay, 0.4, 0)