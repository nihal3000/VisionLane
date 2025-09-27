# VisionLane: A Comparative Lane Detection System 🚀  

A **Streamlit web application** for real-time lane detection in driving videos, offering a **side-by-side comparison** of a Traditional Computer Vision approach and a Deep Learning (YOLOv8) model.  


## 📑 Table of Contents  

- [About The Project](#about-the-project)  
- [Key Features](#key-features-)  
- [Built With](#built-with-)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
  - [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Future Work](#future-work-)  
- [License](#license)  
- [Acknowledgments](#acknowledgments)  
- [Contact](#contact)  

---

## 🔍 About The Project  

This project was born from an interest in **autonomous vehicle systems**, particularly the challenges of perception.  

**VisionLane** serves as an interactive tool to:  
- Perform **lane detection** on uploaded videos.  
- Compare **classical CV techniques** with **modern deep learning methods**.  
- Visualize each step of the detection pipeline.  

It processes user-uploaded videos and visualizes detection for two distinct methodologies.  

---

## ✨ Key Features  

- **Dual Detection Methods:**  
  Switch between a Traditional CV method (Canny Edge Detection, Perspective Transform, Sliding Windows) and a Deep Learning method (YOLOv8 segmentation).  

- **Interactive UI:**  
  A clean, multi-page Streamlit interface for video uploads and parameter tuning.  

- **Step-by-Step Visualization:**  
  See intermediate outputs like masked images, warped frames, and segmentation masks.  

- **Performance Metrics:**  
  Compare **Frames Per Second (FPS)** between the two methods to evaluate efficiency.  

---

## 🛠️ Built With  

- [Streamlit](https://streamlit.io/) – Web App Framework  
- [OpenCV](https://opencv.org/) – Computer Vision Library  
- [PyTorch](https://pytorch.org/) – Deep Learning Framework  
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) – Object Detection & Segmentation Model  
- [NumPy](https://numpy.org/) – Numerical Computing  

---

## 🚀 Getting Started  

Follow these steps to set up the project locally.  

### ✅ Prerequisites  
Make sure you have the following installed:  
- Python **3.8+**  
- `pip` & `venv`  
- [Git](https://git-scm.com/)  

### ⚙️ Installation  

Clone the repository:  
```bash
git clone https://github.com/nihal3000/VisionLane.git
cd VisionLane
```
Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```
### ▶️ Usage

Run the Streamlit app:
```bash
streamlit run app.py
```
Your browser will open to the local Streamlit address.

Steps to test:

Navigate to the "Lane Detection" page.

Upload a driving video file.

Select Traditional CV or Deep Learning (YOLOv8).

Click Process Video to see results!

### 📂 Project Structure
```bash
VisionLane/
├── models/
│   └── best.pt           # YOLOv8 trained model weights
├── pages/
    └── live_detection.py
├── assets/
│   └── test_video.mp4    # Sample video for testing
├── app.py                # Main Streamlit app
├── config.py
├── hybrid_detector.py    # Classical CV lane detection logic
├── lane_detector.py   
├── yolo_lane_detector.py      # YOLOv8 lane detection logic
└── requirements.txt      # Dependencies
```
📝 Future Work

* Add real-time webcam support

* Integrate more DL models (e.g., U-Net)

* Deploy to Streamlit Community Cloud

* Improve handling of curved lanes with tracking

### 🙌 Acknowledgments

* Inspired by research on the Indian Driving Dataset (IDD) during my internship at IIIT Hyderabad.

* Thanks to the creators of Streamlit and Ultralytics YOLOv8 for their excellent tools.

## 📬 Contact  

**MD Nihal Hussain**  

📧 [Email](mailto:nihal220403@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/md-nihal-hussain-a95603254/)  
💻 [GitHub](https://github.com/nihal3000)  
🔗 [Project Link: VisionLane Repository](https://github.com/nihal3000/VisionLane)  


