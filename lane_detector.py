# lane_detector.py
# Contains the class for the traditional OpenCV lane detection logic.

import cv2
import numpy as np
import collections
import config as default_config 

class LaneDetector:
    """
    Encapsulates all traditional lane detection logic using OpenCV.
    Accepts an optional config object to override default settings.
    """
    def __init__(self, cfg=None):
        # If no config is provided, use the default from the file
        self.config = cfg if cfg is not None else default_config

        # State variables for smoothing between frames
        self.prev_left_fits = collections.deque(maxlen=self.config.SMOOTHING_WINDOW_SIZE)
        self.prev_right_fits = collections.deque(maxlen=self.config.SMOOTHING_WINDOW_SIZE)

    def _detect_edges(self, image):
        """Applies Gaussian Blur and Canny Edge Detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.config.BLUR_KERNEL, 0)
        edges = cv2.Canny(blurred, self.config.CANNY_LOW_THRESHOLD, self.config.CANNY_HIGH_THRESHOLD)
        return edges

    def _apply_roi(self, image):
        """Applies a polygonal mask for the region of interest."""
        h, w = image.shape[:2]
        v = self.config.ROI_VERTICES_PCT
        
        tl = (int(v["tl_x"] * w), int(v["tl_y"] * h))
        bl = (int(v["bl_x"] * w), h - v["bl_y_offset"])
        tr = (int(v["tr_x"] * w), int(v["tr_y"] * h))
        br = (int(v["br_x"] * w), h - v["br_y_offset"])
        
        polygon = np.array([[bl, tl, tr, br]], dtype=np.int32)
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygon, 255)
        
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image, np.float32([bl, tl, tr, br])

    def _perspective_transform(self, image, src_points):
        """Warps the ROI into a bird's-eye view."""
        dst_points = np.float32([
            (0, self.config.PERSPECTIVE_HEIGHT),
            (0, 0),
            (self.config.PERSPECTIVE_WIDTH, 0),
            (self.config.PERSPECTIVE_WIDTH, self.config.PERSPECTIVE_HEIGHT)
        ])
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        inv_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
        
        warped = cv2.warpPerspective(image, matrix, (self.config.PERSPECTIVE_WIDTH, self.config.PERSPECTIVE_HEIGHT))
        return warped, inv_matrix

    def _find_lane_pixels_sliding_window(self, warped_image):
        """Finds lane pixels and creates a visualization of the sliding windows."""
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((warped_image, warped_image, warped_image))

        histogram = np.sum(warped_image[warped_image.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = warped_image.shape[0] // self.config.N_WINDOWS
        nonzero = warped_image.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

        left_current, right_current = left_base, right_base
        left_lane_inds, right_lane_inds = [], []

        for window in range(self.config.N_WINDOWS):
            win_y_low = warped_image.shape[0] - (window + 1) * window_height
            win_y_high = warped_image.shape[0] - window * window_height
            win_xleft_low = left_current - self.config.WINDOW_MARGIN
            win_xleft_high = left_current + self.config.WINDOW_MARGIN
            win_xright_low = right_current - self.config.WINDOW_MARGIN
            win_xright_high = right_current + self.config.WINDOW_MARGIN

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.config.MIN_PIXELS:
                left_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.config.MIN_PIXELS:
                right_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

        # Color the detected pixels
        out_img[lefty, leftx] = [255, 0, 0]  # Red for left lane
        out_img[righty, rightx] = [0, 0, 255] # Blue for right lane

        return leftx, lefty, rightx, righty, out_img

    def _fit_and_smooth_lanes(self, leftx, lefty, rightx, righty):
        """Fits a polynomial to the lane pixels and smooths over time."""
        left_fit, right_fit = None, None
        
        if len(leftx) > self.config.POLYNOMIAL_DEGREE:
            left_fit = np.polyfit(lefty, leftx, self.config.POLYNOMIAL_DEGREE)
        if len(rightx) > self.config.POLYNOMIAL_DEGREE:
            right_fit = np.polyfit(righty, rightx, self.config.POLYNOMIAL_DEGREE)

        if left_fit is not None:
            self.prev_left_fits.append(left_fit)
        if len(self.prev_left_fits) > 0:
            avg_left_fit = np.mean(self.prev_left_fits, axis=0)
        else:
            avg_left_fit = left_fit

        if right_fit is not None:
            self.prev_right_fits.append(right_fit)
        if len(self.prev_right_fits) > 0:
            avg_right_fit = np.mean(self.prev_right_fits, axis=0)
        else:
            avg_right_fit = right_fit
            
        return avg_left_fit, avg_right_fit

    def process_frame(self, frame):
        """Main processing pipeline for a single video frame."""
        frame_resized = cv2.resize(frame, (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT))
        edges = self._detect_edges(frame_resized)
        
        roi_edges, src_points = self._apply_roi(edges)
        warped_edges, inv_matrix = self._perspective_transform(roi_edges, src_points)
        
        leftx, lefty, rightx, righty, sliding_window_img = self._find_lane_pixels_sliding_window(warped_edges)
        left_fit, right_fit = self._fit_and_smooth_lanes(leftx, lefty, rightx, righty)
        
        return {
            "original_frame": frame_resized,
            "warped_edges": warped_edges,
            "left_fit": left_fit,
            "right_fit": right_fit,
            "inv_matrix": inv_matrix,
            "sliding_window_img": sliding_window_img
        }