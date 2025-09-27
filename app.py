# app.py
# Advanced Streamlit app with multiple detection methods and performance metrics.

import streamlit as st
import cv2
import os
import types
import warnings
import time

# Import all our detector classes
from lane_detector import LaneDetector
from yolo_lane_detector import YoloLaneDetector
from hybrid_detector import HybridLaneDetector
import config as default_config

warnings.filterwarnings('ignore')

# --- Page Configuration & CSS ---
st.set_page_config(page_title="VisionLane | Advanced Lane Detection", page_icon="🛣️", layout="wide")

def load_css():
    """Loads custom CSS for a modern, dark-themed UI."""
    st.markdown("""
    <style>
        /* General App Styling */
        .main {
            background-color: #0E1117;
            color: #FAFAFA;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Sidebar Styling */
        .st-emotion-cache-16txtl3 {
            background-color: #1a1a2e;
            border-right: 1px solid #2a2a4e;
        }
        .st-emotion-cache-16txtl3 h1, .st-emotion-cache-16txtl3 h2, .st-emotion-cache-16txtl3 h3 {
            color: #e0e0ff;
        }

        /* Button Styling */
        .stButton>button {
            background-color: #1f77b4; /* A nice blue */
            color: #FFFFFF;
            border-radius: 8px;
            border: none;
            padding: 12px 24px;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stButton>button:hover {
            background-color: #2ca02c; /* A nice green on hover */
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        }
        .stButton>button:active {
            transform: translateY(0);
        }

        /* Information & Method Boxes */
        .info-box, .method-box {
            background-color: #161b22;
            border: 1px solid #30363d;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .method-box {
            margin: 10px 0;
            transition: all 0.3s ease;
        }
        .method-box:hover {
            border-color: #1f77b4;
            box-shadow: 0 0 15px rgba(31, 119, 180, 0.5);
        }
        .method-box h4 {
            margin-top: 0;
            color: #58a6ff;
        }
        .yolo-box { border-left: 5px solid #ff7f0e; } /* Orange for YOLO */
        .traditional-box { border-left: 5px solid #2ca02c; } /* Green for Traditional */

        /* Metric Display */
        .metric-container {
            display: flex;
            justify-content: space-around;
            padding: 20px;
            background-color: #161b22;
            border-radius: 10px;
            border: 1px solid #30363d;
        }
        .metric {
            text-align: center;
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #e0e0ff;
        }
        .metric-label {
            font-size: 1.1em;
            color: #8b949e;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# --- App Header ---
st.markdown("<h1 style='text-align: center; color: #e0e0ff; letter-spacing: 2px;'>🛣️ VisionLane: A Multi-Method Lane Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e;'>An interactive tool to compare Traditional Computer Vision with a custom-trained YOLOv8 model for lane detection.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    st.title("🎛️ Controls & Configuration")
    
    detection_method = st.selectbox(
        "Choose Detection Approach",
        ["🧠 YOLOv8 (Deep Learning)", "📐 Traditional (OpenCV)", "🔄 Hybrid (YOLOv8 w/ Fallback)"],
        help="Select the algorithm to process the video."
    )

    video_file = st.file_uploader("📁 Upload a video", type=["mp4", "avi", "mov"])

    if "Traditional" in detection_method or "Hybrid" in detection_method:
        with st.expander("📐 Traditional CV Parameters", expanded=False):
            st.info("These settings only affect the 'Traditional' and 'Hybrid' modes.")
            dynamic_config = types.SimpleNamespace()
            dynamic_config.CANNY_LOW_THRESHOLD = st.slider("Canny Low Threshold", 10, 200, default_config.CANNY_LOW_THRESHOLD, help="Lower values detect more edges.")
            dynamic_config.CANNY_HIGH_THRESHOLD = st.slider("Canny High Threshold", 20, 400, default_config.CANNY_HIGH_THRESHOLD, help="Higher values create cleaner edge maps.")
            dynamic_config.SMOOTHING_WINDOW_SIZE = st.slider("Smoothing Frames", 1, 20, default_config.SMOOTHING_WINDOW_SIZE, help="Averages the lane detection over multiple frames to reduce jitter.")
            # Ensure all other default configs are carried over
            for attr in dir(default_config):
                if not attr.startswith('__') and not hasattr(dynamic_config, attr):
                    setattr(dynamic_config, attr, getattr(default_config, attr))
    else:
        dynamic_config = default_config

# --- Method Information Boxes ---
st.subheader("Methodology Overview")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="method-box traditional-box">
        <h4>📐 Traditional (OpenCV)</h4>
        <ul>
            <li><strong>Method:</strong> A classic pipeline of Canny Edge Detection, Perspective Transformation, and a Sliding Window Search.</li>
            <li><strong>Pros:</strong> Highly interpretable, computationally fast on CPU, and requires no prior model training.</li>
            <li><strong>Cons:</strong> Brittle and highly sensitive to challenging conditions like shadows, poor lighting, and imperfect road markings.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="method-box yolo-box">
        <h4>🧠 YOLOv8 (Deep Learning)</h4>
        <ul>
            <li><strong>Method:</strong> Instance Segmentation using a custom-trained Convolutional Neural Network (CNN).</li>
            <li><strong>Pros:</strong> Extremely robust to a wide range of challenging conditions, including shadows, curves, and varied lighting.</li>
            <li><strong>Cons:</strong> Requires a pre-trained model, is computationally more intensive on a CPU, and acts as a "black box".</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- Main Application Logic ---
if video_file is not None:
    # Create a temporary directory for the video
    if not os.path.exists("temp_videos"):
        os.makedirs("temp_videos")
    video_path = os.path.join("temp_videos", video_file.name)
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())

    if st.sidebar.button("🚀 Process Video", use_container_width=True):
        try:
            # IMPORTANT: Please update this path to your `best.pt` model file
            yolo_model_path = "D:\\Lane_detection\\best.pt"
            if not os.path.exists(yolo_model_path):
                st.error(f"YOLOv8 model not found at path: {yolo_model_path}. Please update the path in the script.")
                st.stop()
            
            # Initialize the selected detector
            if "YOLOv8" in detection_method:
                detector = YoloLaneDetector(model_path=yolo_model_path)
            elif "Hybrid" in detection_method:
                detector = HybridLaneDetector(cfg=dynamic_config, yolo_model_path=yolo_model_path)
            else:  # Traditional
                detector = LaneDetector(cfg=dynamic_config)

            vidcap = cv2.VideoCapture(video_path)
            total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            st.markdown("---")
            st.subheader("🎬 Live Processing Output")
            
            # Create placeholders for the video frames and metrics
            main_image_placeholder = st.empty()
            viz_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            processing_times = []
            fallback_events = 0

            for frame_count in range(total_frames):
                success, frame = vidcap.read()
                if not success:
                    break
                
                start_time = time.time()
                
                # Process the frame based on the selected method
                if "Traditional" in detection_method:
                    from visualization import draw_lane_overlay
                    processed_data = detector.process_frame(frame)
                    final_image = draw_lane_overlay(processed_data)
                    algo_viz = processed_data["sliding_window_img"]
                    method_used = "Traditional (OpenCV)"
                elif "YOLOv8" in detection_method:
                    final_image, algo_viz = detector.process_frame(frame)
                    method_used = "YOLOv8"
                else:  # Hybrid
                    result = detector.process_frame(frame)
                    final_image = result["final_image"]
                    algo_viz = result["algorithm_viz"]
                    method_used = result["method_used"]
                    if "Fallback" in method_used:
                        fallback_events += 1

                processing_times.append(time.time() - start_time)
                
                # Display the processed frames
                # We use two columns to show the final result and the visualization side-by-side
                col_main, col_viz = st.columns(2)
                with col_main:
                    st.image(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB), caption="🎯 Final Result")
                with col_viz:
                    st.image(cv2.cvtColor(algo_viz, cv2.COLOR_BGR2RGB), caption="⚙️ Algorithm Visualization")

                # Update metrics
                avg_time = sum(processing_times) / len(processing_times)
                fps = 1.0 / avg_time
                
                # Create a more visually appealing metric display
                metrics_html = f"""
                <div class="metric-container">
                    <div class="metric">
                        <div class="metric-value">{fps:.2f}</div>
                        <div class="metric-label">Average FPS</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{processing_times[-1]*1000:.2f}</div>
                        <div class="metric-label">Last Frame Time (ms)</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{method_used}</div>
                        <div class="metric-label">Current Method</div>
                    </div>
                """
                if "Hybrid" in detection_method:
                    metrics_html += f"""
                    <div class="metric">
                        <div class="metric-value" style="color: #ff7f0e;">{fallback_events}</div>
                        <div class="metric-label">Fallback Events</div>
                    </div>
                    """
                metrics_html += "</div>"
                metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)
                
                # Update progress bar and status text
                progress_percentage = (frame_count + 1) / total_frames
                progress_bar.progress(progress_percentage)
                status_text.text(f"Processing frame {frame_count + 1}/{total_frames} ({progress_percentage:.0%})...")

            st.success(f"🎉 Processing complete! Final Average FPS: {1.0 / (sum(processing_times) / len(processing_times)):.2f}")
            vidcap.release()
        except Exception as e:
            st.error(f"❌ An error occurred during processing: {e}")
            st.exception(e) # Provides a full traceback for debugging
else:
    st.markdown("""
        <div class="info-box">
            <h4>🚀 Welcome to the Advanced VisionLane Detector!</h4>
            <p>This tool allows you to analyze and compare different lane detection algorithms on your own videos.</p>
            <ol>
                <li><strong>Choose your detection method</strong> in the sidebar on the left.</li>
                <li><strong>Upload a video file</strong> to begin the analysis.</li>
                <li>Click the <strong>'Process Video'</strong> button to see the magic!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
