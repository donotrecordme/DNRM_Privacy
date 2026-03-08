# DNRM (Do Not Record Me)

**Dual-engine pipeline for automated privacy blurring using apparel. An open-source, real-time face anonymization pipeline optimized for Apple Silicon.**

DNRM is an automated, dual-engine computer vision pipeline designed to protect privacy in public spaces. It detects specific DNRM markers (apparel, caps, bags, etc.) and dynamically applies a smooth, tracking Gaussian blur to the wearer's face in video footage.

Whether you are a privacy advocate, a developer, or a beginner running Python for the first time, this tool provides a robust, local solution to visual non-consent.

---

## 🚀 Key Features

- **Dual-AI Architecture**: Uses a Roboflow custom model for marker detection and Ultralytics YOLOv8 for high-speed facial detection.
- **Hardware Acceleration**: Heavily optimized for Apple Silicon (Mac M-Series). It leverages CoreML for marker detection and Metal Performance Shaders (MPS) for YOLO face detection.
- **Multi-Target Tracking**: Capable of tracking and blurring multiple subjects simultaneously without confusing targets.
- **Dual-Memory Occlusion Engine**: Continues tracking a subject's face for up to 3 seconds even if their marker is temporarily obscured by arms, coats, or passing objects.
- **Dynamic Geometry**: Scales the blur dynamically and handles both chest-level (shirts) and head-level (caps) markers using vertical proportion math.
- **Anti-Jitter Smoothing**: Utilizes Exponential Moving Average (EMA) math so the blur glides smoothly across the screen instead of snapping.

---

## 📁 Project Structure

- `dnrm_demo.py` - The core multi-target tracking script.
- `models/` - Directory containing the local YOLOv8 nano face weights (`yolov8n-face.pt`).
- `.env` - **(Required)** Hidden file to securely store your Roboflow API key.
- `requirements.txt` - List of Python dependencies (`opencv-python`, `inference`, `ultralytics`, `python-dotenv`).

---

## ⚙️ Installation & Setup

### 1. Prerequisites

You need Python installed on your computer. Open your **Terminal** (Command + Space, type "Terminal") and run:
```bash
python3 --version
```

If it doesn't show a version number, download it from https://www.python.org/.

### 2. Get the Code

Download this repository as a ZIP file and extract it, or clone it via terminal:
```bash
git clone https://github.com/YourUsername/DNRM.git
```
```bash
cd DNRM
```

### 3. Install Dependencies

Ensure you are in the project folder and run:

```bash
pip install -r requirements.txt
```


### 4. Secure Your API Key

Create a file named `.env` in the project root and add:

```bash
ROBOFLOW_API_KEY=your_actual_api_key_here
```

### 5. Ensure Local Models are Present

Make sure the file exists:

Make sure the `yolov8n-face.pt` file is successfully located inside the `models/` directory.


---

## 🎥 Usage

### Prepare Video

There is a test video included, place the target video(you can use your own name) in the main folder:

```bash
test_video.mp4
```

### Run the Pipeline

```bash
python dnrm_demo.py
```

Press **q** at any time to stop.  
```bash
The output will save as `your_video_name.mp4`.
```
---

## ⚠️ Known Issues / macOS Warnings

When running on Apple Silicon (M1/M2/M3), you may see red C++ errors or duplicate library warnings (e.g., `AVFFrameReceiver`).

You can safely ignore these. The pipeline will still process and export your video correctly.

---

## 📄 License

MIT License
