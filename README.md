# 🕳️ AI-Powered Real-Time Pothole Detection System

A computer vision application designed to detect road potholes in real time using a custom-trained **YOLO (You Only Look Once)** deep learning model. The system integrates camera video feeds with live GPS coordinates and reverse geocoding to log road hazards and export downloadable CSV reports for municipal and highway authorities.

---

## 🌟 Key Features

- 📹 **Real-Time Live WebCam Scanner**: Processes video stream frames instantly using HTML5 WebCam API and OpenCV.
- 🎯 **Deep Learning Pothole Detection**: Powered by a custom-trained YOLO object detection model fine-tuned for road surface hazard recognition.
- 📍 **GPS Location Integration**: Captures precise live latitude and longitude coordinates for every detected pothole.
- 🗺️ **Reverse Geocoding**: Automatically converts raw GPS coordinates into human-readable city and street address locations using OpenStreetMap Nominatim.
- 📊 **CSV Report Generation**: Export organized hazard reports (Timestamp, Latitude, Longitude, Detection Confidence, City/Address) ready to share with government/highway agencies.
- ⚡ **FastAPI High-Performance Backend**: Modern Python API backend using `FastAPI` and `Uvicorn` with sub-second CPU inference speeds.

---

## 🏗️ Project Architecture & Tech Stack

- **Computer Vision & AI**: Python, Ultralytics YOLO, PyTorch, OpenCV, NumPy
- **Backend Framework**: FastAPI, Uvicorn, Python 3.10+
- **Frontend Interface**: Modern Responsive Web UI (HTML5, Vanilla CSS, JavaScript, Canvas API)
- **Geolocation & Mapping**: HTML5 Geolocation API, OpenStreetMap Nominatim

```text
pothole_detection/
├── app.py                         # FastAPI web server and live detection API
├── index.html                     # Live WebCam scanner UI dashboard
├── requirements.txt               # Dependencies list
├── yolo_model_training.ipynb      # Model training & fine-tuning notebook
├── model/
│   └── best.pt                    # Trained YOLO model weights
└── gps_data/
    ├── load_gps.py                # GPS data handler
    └── reverse_geocode.py         # Reverse geocoding utility
```

---

## 🚀 How to Run Locally

### 1. Prerequisites & Virtual Environment

Ensure Python 3.10+ is installed on your system.

```bash
# Clone the repository
git clone https://github.com/Abhishekrouth/pothole_detection_system.git
cd pothole_detection_system

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Local Server

Run `app.py` directly using Python:

```bash
python app.py
```

Or using `uvicorn`:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### 4. Access the Application

Open your browser and navigate to:
```text
http://localhost:7860
```

Click **Start Scanning** and allow camera and location permissions. Point the camera at a road surface or pothole image to see live bounding box annotations and hazard tracking in real-time.

---

## 📡 API Endpoints

### 1. Web UI Scanner
- **`GET /`** or **`GET /scanner`**: Serves the interactive live camera dashboard.

### 2. Live Frame Pothole Detection
- **`POST /api/detect_live`**
  - **Payload**: `{"image": "data:image/jpeg;base64,...", "latitude": "23.019313", "longitude": "72.534227"}`
  - **Response**: Returns annotated base64 frame, `detected` boolean, detection `count`, confidence scores list, and geocoded `city`/address.

---

## 🎓 Presentation & Demo Tips

- To demonstrate pothole detection indoors, point your camera at an image or video of a road pothole displayed on a smartphone/monitor screen.
- Click **Export CSV Report** to download the structured detection log with exact timestamps, GPS coordinates, and address information.
