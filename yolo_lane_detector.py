# yolo_lane_detector.py
import cv2
import numpy as np
from ultralytics import YOLO

class YoloLaneDetector:
    """
    A detector class that uses a trained YOLOv8 segmentation model.
    """
    def __init__(self, model_path='best.pt'):
        """
        Initializes the detector with a trained YOLOv8 model.
        :param model_path: Path to the trained .pt model file.
        """
        self.model = YOLO(model_path)
        # Define the color and transparency for the lane overlay
        self.overlay_color = (0, 255, 0)  # Green
        self.overlay_alpha = 0.8

    def process_frame(self, frame):
        """
        Processes a single video frame to detect and draw lane markings.
        Returns the final image and the raw segmentation mask.
        """
        # Run inference on the frame
        # The model expects RGB images, so we convert from BGR
        results = self.model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Create an empty image for the colored mask
        color_mask_bgr = np.zeros_like(frame, dtype=np.uint8)

        # Check if the model detected any masks (lanes)
        if results[0].masks is not None:
            # The model might detect multiple lane segments. We combine them.
            for mask_data in results[0].masks.data:
                # Convert mask tensor to a numpy array
                mask_cpu = mask_data.cpu().numpy()
                
                # Resize the mask to the frame's original size
                mask_resized = cv2.resize(mask_cpu, (frame.shape[1], frame.shape[0]))
                
                # Apply the color to the areas where the mask is active
                color_mask_bgr[mask_resized > 0.5] = self.overlay_color

        # Blend the colored mask with the original frame for a transparent effect
        final_image = cv2.addWeighted(frame, 1, color_mask_bgr, self.overlay_alpha, 0)

        # Return the final processed image and the colored mask for visualization
        return final_image, color_mask_bgr