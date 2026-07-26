import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp"

import cv2
import numpy as np
import base64
import csv
import json
import time
from pathlib import Path
from ultralytics import YOLO

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from gps_data.load_gps import load_gps_data, get_gps_for_timestamp
from gps_data.reverse_geocode import geocode

app = FastAPI(title="AI-Powered Pothole Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

def _run_yolo(img, conf=0.025):
    global model
    if model is None:
        model = YOLO("model/best.pt")
    
    # Resize frame to max 640px for accurate YOLO feature extraction
    h, w = img.shape[:2]
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized_img = img

    return model(resized_img, conf=conf, imgsz=640)

def predict_potholes(img, conf=0.025):
    return _run_yolo(img, conf=conf)

location_cache = {}

gps_data_list = []
if os.path.exists("gps_data/gps_data.csv"):
    try:
        gps_data_list = load_gps_data("gps_data/gps_data.csv")
    except Exception:
        gps_data_list = []

# ── API Endpoints ──────────────────────────────────────────────────────────

_BASE_DIR = Path(__file__).parent

@app.get("/", response_class=HTMLResponse)
@app.get("/scanner", response_class=HTMLResponse)
async def serve_scanner():
    index_path = _BASE_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Scanner UI not found</h1>", status_code=404)

@app.get("/highway.jpg")
async def serve_highway_image():
    img_path = _BASE_DIR / "highway.jpg"
    if img_path.exists():
        return FileResponse(str(img_path), media_type="image/jpeg")
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.options("/api/detect_live")
async def options_detect_live():
    return JSONResponse(content={"status": "ok"}, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "*"
    })

@app.post("/api/detect_live")
async def api_detect_live(request: Request):
    print("Received POST /api/detect_live request")
    try:
        data = await request.json()
        img_data_url = data.get("image", "")
        latitude = data.get("latitude", "")
        longitude = data.get("longitude", "")

        if not img_data_url or not isinstance(img_data_url, str):
            return JSONResponse(content={"image": "", "detected": False, "count": 0, "confidences": [], "city": ""})

        raw_b64 = img_data_url
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        raw_b64 = raw_b64.replace(' ', '+')
        missing_padding = len(raw_b64) % 4
        if missing_padding:
            raw_b64 += '=' * (4 - missing_padding)

        img_bytes = base64.b64decode(raw_b64)
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(content={"image": "", "detected": False, "count": 0, "confidences": [], "city": ""})

        results = predict_potholes(img, conf=0.025)
        if not results or len(results) == 0:
            return JSONResponse(content={"image": "", "detected": False, "count": 0, "confidences": [], "city": ""})

        boxes = results[0].boxes
        pothole_detected = len(boxes) > 0
        annotated = results[0].plot()
        _, img_encoded = cv2.imencode('.jpg', annotated)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        confidences = [round(float(b.conf[0]), 3) for b in boxes] if pothole_detected else []

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

        img_src = f"data:image/jpeg;base64,{img_base64}" if pothole_detected else ""
        return JSONResponse(content={
            "image": img_src,
            "detected": pothole_detected,
            "count": len(boxes) if pothole_detected else 0,
            "confidences": confidences,
            "city": city
        })
    except Exception as e:
        print("Live detection error:", e)
        return JSONResponse(content={"image": "", "detected": False, "count": 0, "confidences": [], "city": ""}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)