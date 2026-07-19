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
from fastapi.middleware.cors import CORSMiddleware
from gps_data.load_gps import load_gps_data, get_gps_for_timestamp
from gps_data.reverse_geocode import geocode

model = None

# GPU decorator wrapper for Hugging Face ZeroGPU
try:
    import spaces
    @spaces.GPU
    def predict_potholes(img, conf=0.25):
        global model
        if model is None:
            model = YOLO("model/best.pt")
        return model(img, conf=conf)
except ImportError:
    def predict_potholes(img, conf=0.25):
        global model
        if model is None:
            model = YOLO("model/best.pt")
        return model(img, conf=conf)

location_cache = {}

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

# 4. Custom API route for Vercel webcam frame detection
@app.post("/detect_potholes_images")
async def detect(image: UploadFile = File(...), latitude: str = Form(None), longitude: str = Form(None)):
    image_bytes = await image.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = predict_potholes(img)
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

# 5. Local development launch (if run directly)
if __name__ == "__main__":
    demo.launch()