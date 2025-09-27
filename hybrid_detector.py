# hybrid_detector.py

from lane_detector import LaneDetector
from yolo_lane_detector import YoloLaneDetector
import config as default_config

class HybridLaneDetector:
    """
    A detector that uses YOLOv8 as the primary method and falls back to the
    traditional OpenCV method if the deep learning model fails.
    """
    def __init__(self, cfg=None, yolo_model_path='best.pt'):
        self.config = cfg if cfg is not None else default_config
        self.yolo_detector = YoloLaneDetector(model_path=yolo_model_path)
        self.traditional_detector = LaneDetector(cfg=self.config)
        self.detection_failures = 0

    def process_frame(self, frame):
        """
        Processes a frame by first trying the YOLOv8 detector. If it fails
        (e.g., detects no lanes), it falls back to the traditional method.
        """
        # --- Try YOLOv8 First ---
        try:
            # We need to check the output to see if it's valid
            _, mask_image = self.yolo_detector.process_frame(frame)
            
            # A simple check for failure: is the mask all black?
            if mask_image.max() > 0:
                # Success, create the full dictionary for the app
                final_image, _ = self.yolo_detector.process_frame(frame) # Reprocess to get final overlay
                self.detection_failures = 0
                return {
                    "final_image": final_image,
                    "algorithm_viz": mask_image,
                    "viz_type": "mask",
                    "method_used": "YOLOv8",
                    "detection_failures": self.detection_failures
                }
        except Exception as e:
            # An error occurred with the YOLO model
            print(f"YOLOv8 detector failed with error: {e}")
            pass

        # --- Fallback to Traditional Method ---
        self.detection_failures += 1
        processed_data = self.traditional_detector.process_frame(frame)
        
        # We need to draw the overlay here since the app expects a final image
        from visualization import draw_lane_overlay
        final_image = draw_lane_overlay(processed_data)
        
        return {
            "final_image": final_image,
            "algorithm_viz": processed_data["sliding_window_img"],
            "viz_type": "sliding_window",
            "method_used": "Traditional (Fallback)",
            "detection_failures": self.detection_failures
        }