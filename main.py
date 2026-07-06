from flask import Flask, request, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import csv, requests, time

app = Flask(__name__)

model = YOLO("model/best.pt")
pothole_records = []

def load_gps_data(csv_path):
    gps_points = []

    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            gps_points.append({
                "timestamp": float(row["timestamp"]),
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"])
            })

    return gps_points

location_cache = {}

def reverse_geocode(lat, lon):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json"
    }
    headers = {
        "User": "pothole detection app"
    }

    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data.get("display_name", "Unknown location")

    return "Unknown location"


gps_data = load_gps_data("data/gps_log.csv")

def get_gps_for_timestamp(timestamp, gps_data):
    closest_point = min(
        gps_data,
        key=lambda point: abs(point["timestamp"] - timestamp)
    )
    return closest_point["lat"], closest_point["lon"]

def save_potholes_to_csv(records, file_path):

    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file,fieldnames=["timestamp", "lat", "lon", "confidence", "area"])
        writer.writeheader()

        for record in records:
            key = (record["lat"], record["lon"])

            if key not in location_cache:
                location_cache[key] = reverse_geocode(record["lat"], record["lon"])
                time.sleep(1)   

            writer.writerow({
                **record,
                "area": location_cache[key]
            })


@app.route("/gps_lookup_test", methods=["GET"])
def gps_lookup_test():
    lat, lon = get_gps_for_timestamp(1.2, gps_data)
    return {
        "query_time": 1.2,
        "lat": lat,
        "lon": lon
    }

@app.route("/", methods=["GET"])
def home():
    return "App is running"

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

    return send_file(
        output_path,
        as_attachment=True,
        download_name="detected_image.jpg",
        mimetype="image/jpeg"
    )

@app.route("/detect_potholes_videos", methods=["POST"])
def detect_videos():

    if "video" not in request.files:
        return {"error": "No video provided"}, 400

    video_file = request.files["video"]

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
            lat, lon = get_gps_for_timestamp(timestamp, gps_data)

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

    csv_output_path = "source/potholes_detected.csv"
    save_potholes_to_csv(pothole_records, csv_output_path)

    return send_file(
        output_path,
        as_attachment=True,
        download_name="pothole_detected.mp4",
        mimetype="video/mp4"
    )



@app.route("/download_potholes_csv", methods=["GET"])
def download_potholes_csv():
    return send_file(
        "source2/potholes_detected.csv",
        as_attachment=True,
        download_name="potholes.csv",
        mimetype="text/csv"
    )



if __name__ == "__main__":
    app.run(debug=True)