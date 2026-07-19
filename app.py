import os
import cv2
import numpy as np
import base64
import csv
import time
from ultralytics import YOLO

import gradio_client.utils as _gc_utils
_orig_json_schema_to_python_type = _gc_utils._json_schema_to_python_type
def _safe_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "bool" if schema else "Any"
    return _orig_json_schema_to_python_type(schema, defs)
_gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type

import gradio as gr
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from gps_data.load_gps import load_gps_data, get_gps_for_timestamp
from gps_data.reverse_geocode import geocode

model = None

# GPU decorator wrapper for Hugging Face ZeroGPU
try:
    import spaces
    @spaces.GPU
    def predict_potholes(img, conf=0.15):
        global model
        if model is None:
            model = YOLO("model/best.pt")
        return model(img, conf=conf)
except ImportError:
    def predict_potholes(img, conf=0.15):
        global model
        if model is None:
            model = YOLO("model/best.pt")
        return model(img, conf=conf)

location_cache = {}

# Load GPS data on startup if available
gps_data_list = []
if os.path.exists("gps_data/gps_data.csv"):
    try:
        gps_data_list = load_gps_data("gps_data/gps_data.csv")
    except Exception:
        gps_data_list = []

# 2. Define Gradio Interface function (so Hugging Face is happy)
def gradio_predict(img):
    if img is None:
        return None
    results = predict_potholes(img)
    return results[0].plot()

demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="AI Pothole Detector",
    description="Live inference endpoint"
)

# 3. Get the underlying FastAPI app and enable CORS
app = demo.app

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Custom API route for image detection
@app.post("/detect_potholes_images")
async def detect(
    image: UploadFile = File(...),
    latitude: str = Form(None),
    longitude: str = Form(None),
    conf: float = Form(0.15)
):
    image_bytes = await image.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = predict_potholes(img, conf=conf)
    annotated = results[0].plot()

    # Encode annotated image to base64
    _, img_encoded = cv2.imencode('.jpg', annotated)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    boxes = results[0].boxes
    pothole_detected = len(boxes) > 0
    confidences = [float(box.conf[0]) for box in boxes] if pothole_detected else []

    # Get optional coordinates for geocoding
    city = ""
    if latitude and longitude and pothole_detected:
        try:
            lat_f = round(float(latitude), 4)
            lon_f = round(float(longitude), 4)
            key = (lat_f, lon_f)
            if key not in location_cache:
                location_cache[key] = geocode(lat_f, lon_f)
            city = location_cache[key]
        except Exception:
            pass

    return {
        "image": img_base64,
        "detected": pothole_detected,
        "confidences": confidences,
        "city": city
    }

# 5. Video detection endpoint
@app.post("/detect_potholes_videos")
async def detect_video(
    video: UploadFile = File(...),
    conf: float = Form(0.15)
):
    os.makedirs("source2", exist_ok=True)
    input_path = "source2/input_potholes_video.mp4"
    output_path = "source2/output_potholes_video.mp4"
    csv_path = "source2/potholes_detected.csv"

    video_bytes = await video.read()
    with open(input_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return {"error": "Could not open uploaded video file."}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    records = []
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = predict_potholes(frame, conf=conf)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        boxes = results[0].boxes
        if len(boxes) > 0:
            timestamp = round(frame_index / fps, 2)
            lat, lon = (0.0, 0.0)
            city = ""

            if gps_data_list:
                try:
                    lat, lon = get_gps_for_timestamp(timestamp, gps_data_list)
                    lat_r, lon_r = round(lat, 4), round(lon, 4)
                    key = (lat_r, lon_r)
                    if key not in location_cache:
                        location_cache[key] = geocode(lat_r, lon_r)
                    city = location_cache[key]
                except Exception:
                    city = ""

            for box in boxes:
                conf = float(box.conf[0])
                records.append([timestamp, lat, lon, conf, city])

        frame_index += 1

    cap.release()
    out.release()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "lat", "lon", "confidence", "city"])
        writer.writerows(records)

    return FileResponse(output_path, media_type="video/mp4", filename="pothole_detected.mp4")

# 6. Download detected pothole data CSV
@app.get("/download_potholes_data")
async def download_potholes_data():
    csv_path = "source2/potholes_detected.csv"
    if not os.path.exists(csv_path):
        return {"error": "No pothole data records found."}
    return FileResponse(csv_path, media_type="text/csv", filename="potholes_detected.csv")

# 7. Local development launch (if run directly)
if __name__ == "__main__":
    demo.launch()