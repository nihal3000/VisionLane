# pages/1_🚀_Live_Detection.py

import streamlit as st
import cv2
import os
import types
import warnings
import time

# Import all necessary detector classes
from lane_detector import LaneDetector
from yolo_lane_detector import YoloLaneDetector
from hybrid_detector import HybridLaneDetector
from visualization import draw_lane_overlay
import config as default_config

warnings.filterwarnings('ignore')

# --- Page Header ---
st.markdown("## Live Detection Dashboard")
st.markdown("Select a detection method, upload a video, and tune parameters in the sidebar to begin analysis.")
st.markdown("---")

# --- Sidebar Controls ---
st.sidebar.title("🎛️ Controls")
detection_method = st.sidebar.selectbox(
    "Choose Detection Approach",
    ["🧠 YOLOv8 (Deep Learning)", "📐 Traditional (OpenCV)", "🔄 Hybrid (YOLOv8 w/ Fallback)"],
    help="Select the algorithm for lane detection."
)

video_file = st.sidebar.file_uploader("📁 Upload a Video", type=["mp4", "avi", "mov"])

# --- Conditional UI based on method ---
if "Traditional" in detection_method or "Hybrid" in detection_method:
    with st.sidebar.expander("📐 Traditional CV Parameters", expanded=True):
        dynamic_config = types.SimpleNamespace()
        dynamic_config.CANNY_LOW_THRESHOLD = st.slider("Canny Low", 10, 200, default_config.CANNY_LOW_THRESHOLD)
        dynamic_config.CANNY_HIGH_THRESHOLD = st.slider("Canny High", 20, 400, default_config.CANNY_HIGH_THRESHOLD)
        dynamic_config.SMOOTHING_WINDOW_SIZE = st.slider("Smoothing Frames", 1, 20, default_config.SMOOTHING_WINDOW_SIZE)
        for attr in dir(default_config):
            if not attr.startswith('__') and not hasattr(dynamic_config, attr):
                setattr(dynamic_config, attr, getattr(default_config, attr))
else:
    dynamic_config = default_config
    st.sidebar.info("The YOLOv8 model is pre-trained and does not require manual tuning.")
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.sidebar.error(f"Model file '{model_path}' not found!")
        st.stop()
    else:
        st.sidebar.success(f"✅ YOLOv8 model loaded.")

# --- Main Application Logic ---
if video_file is not None:
    if not os.path.exists("temp_videos"): os.makedirs("temp_videos")
    video_path = os.path.join("temp_videos", video_file.name)
    with open(video_path, "wb") as f: f.write(video_file.getbuffer())

    if st.sidebar.button("🚀 Process Video", use_container_width=True):
        try:
            # Initialize the correct detector
            if "YOLOv8" in detection_method:
                detector = YoloLaneDetector(model_path="best.pt")
            elif "Hybrid" in detection_method:
                detector = HybridLaneDetector(cfg=dynamic_config, yolo_model_path="best.pt")
            else: # Traditional
                detector = LaneDetector(cfg=dynamic_config)

            vidcap = cv2.VideoCapture(video_path)
            
            # Setup UI placeholders
            st.markdown("---")
            st.subheader("🎬 Live Processing Output")
            tab1, tab2, tab3 = st.tabs(["🎯 Final Result", "⚙️ Algorithm Visualization", "📊 Performance Metrics"])
            with tab1: main_image_placeholder = st.empty()
            with tab2: viz_placeholder = st.empty()
            with tab3: metrics_placeholder = st.empty()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            processing_times = []

            for frame_count in range(total_frames):
                success, frame = vidcap.read()
                if not success: break
                
                start_time = time.time()
                
                # Process frame based on the selected method
                if "Traditional" in detection_method:
                    processed_data = detector.process_frame(frame)
                    final_image = draw_lane_overlay(processed_data)
                    algo_viz = processed_data["sliding_window_img"]
                    method_used = "Traditional"
                elif "YOLOv8" in detection_method:
                    final_image, algo_viz = detector.process_frame(frame)
                    method_used = "YOLOv8"
                else: # Hybrid
                    result = detector.process_frame(frame)
                    final_image = result["final_image"]
                    algo_viz = result["algorithm_viz"]
                    method_used = result["method_used"]

                processing_times.append(time.time() - start_time)
                
                # Update UI placeholders
                main_image_placeholder.image(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
                viz_placeholder.image(cv2.cvtColor(algo_viz, cv2.COLOR_BGR2RGB))
                
                avg_time = sum(processing_times) / len(processing_times)
                fps = 1.0 / avg_time
                
                metrics_placeholder.markdown(f"""
                <div class="info-box">
                    <p><strong>Average FPS:</strong> {fps:.2f}</p>
                    <p><strong>Processing Time (last frame):</strong> {processing_times[-1]*1000:.2f} ms</p>
                    <p><strong>Current Method:</strong> {method_used}</p>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar.progress((frame_count + 1) / total_frames)
                status_text.text(f"Processing frame {frame_count + 1}/{total_frames}...")

            st.success(f"🎉 Processing complete! Average FPS: {1.0 / avg_time:.2f}")
            vidcap.release()
        except Exception as e:
            st.error(f"❌ An error occurred during processing: {e}")

else:
    st.info("Please select a method and upload a video file in the sidebar to begin.")