from flask import Flask, request, send_file, render_template
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import time
from gps_data.load_gps import load_gps_data, get_gps_for_timestamp
from gps_data.reverse_geocode import geocode
import csv
import base64

app = Flask(__name__)
CORS(app)

model = YOLO("model/best.pt")

location_cache = {}
gps_data = load_gps_data("gps_data/gps_data.csv")

def save_potholes_to_csv(records, file_path):

    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file,fieldnames=["timestamp", "lat", "lon", "confidence", "city"])
        writer.writeheader()

        for record in records:
            lat_rounded = round(record["lat"], 4)
            lon_rounded = round(record["lon"], 4)
            key = (lat_rounded, lon_rounded)
            if key not in location_cache:
                location_cache[key] = geocode(lat_rounded, lon_rounded)
                time.sleep(1)   

            writer.writerow({**record,"city": location_cache[key]})

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/templates/highway.jpg", methods=["GET"])
def serve_highway():
    return send_file("templates/highway.jpg")

@app.route("/detect_potholes_images", methods=["POST"])
def detect():

    if "image" not in request.files:
        return {"error": "No image provided"}, 400

    image_file = request.files["image"].read()
    np_img = np.frombuffer(image_file, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = model(img)
    annotated = results[0].plot()

    output_path = "source/photo_detected_2.jpg"
    cv2.imwrite(output_path, annotated)

    # Encode annotated image to base64
    _, img_encoded = cv2.imencode('.jpg', annotated)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    boxes = results[0].boxes
    pothole_detected = len(boxes) > 0
    confidences = [float(box.conf[0]) for box in boxes] if pothole_detected else []

    # Get optional coordinates for geocoding
    city = ""
    lat = request.form.get("latitude")
    lon = request.form.get("longitude")
    if lat and lon and pothole_detected:
        try:
            lat_f = round(float(lat), 4)
            lon_f = round(float(lon), 4)
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

@app.route("/detect_potholes_videos", methods=["POST"])
def detect_videos():

    if "video" not in request.files:
        return {"error": "No video provided"}, 400

    video_file = request.files["video"]
    pothole_records = []

    if "gps_log" in request.files:
        gps_file = request.files["gps_log"]
        gps_input_path = "source2/uploaded_gps_log.csv"
        gps_file.save(gps_input_path)
        current_gps_data = load_gps_data(gps_input_path)
    else:
        current_gps_data = gps_data # Fallback to global GPS log

    input_path = "source2/input_potholes_video.mp4"
    output_path = "source2/output_potholes_video.mp4"

    video_file.save(input_path)

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_index / fps

        results = model(frame, conf=0.5)

        for box in results[0].boxes:
            confidence = float(box.conf[0])
            lat, lon = get_gps_for_timestamp(timestamp, current_gps_data)

            pothole_records.append({
                "lat": lat,
                "lon": lon,
                "confidence": confidence,
                "timestamp": timestamp
            })

        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        frame_index += 1

    cap.release()
    out.release()

    csv_output_path = "source2/potholes_detected.csv"
    save_potholes_to_csv(pothole_records, csv_output_path)

    return send_file(
        output_path,
        as_attachment=True,
        download_name="pothole_detected.mp4",
        mimetype="video/mp4"
    )


@app.route("/download_potholes_data", methods=["GET"])
def download_potholes_csv():
    
    return send_file(
        "source2/potholes_detected.csv",
        as_attachment=True,
        download_name="potholes.csv",
        mimetype="text/csv"
    )


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)