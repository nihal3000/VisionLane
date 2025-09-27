# config.py
# Central configuration file for the Lane Detection project.
# These values serve as defaults and can be overridden by the Streamlit UI.

# --- Image Processing Parameters ---
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# --- Edge Detection ---
BLUR_KERNEL = (5, 5)
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# --- Region of Interest (ROI) ---
# Vertices are defined as percentages of image dimensions
ROI_VERTICES_PCT = {
    "tl_x": 0.38, "tl_y": 0.75,
    "bl_x": 0.20, "bl_y_offset": 10,
    "tr_x": 0.64, "tr_y": 0.75,
    "br_x": 0.84, "br_y_offset": 10
}

# --- Perspective Transform (Bird's-eye View) ---
PERSPECTIVE_WIDTH = 500
PERSPECTIVE_HEIGHT = 400

# --- Sliding Window Algorithm ---
N_WINDOWS = 12
WINDOW_MARGIN = 50      # Width of the windows +/- margin
MIN_PIXELS = 30         # Minimum number of pixels to recenter a window

# --- Lane Fitting and Smoothing ---
POLYNOMIAL_DEGREE = 2
SMOOTHING_WINDOW_SIZE = 5 # Number of frames to average over