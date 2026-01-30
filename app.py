from flask import Flask, request, send_file
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("model/best.pt")

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
    input_path = "source/input_potholes_video.mp4"
    output_path = "source/output_potholes_video.mp4"

    video_file.save(input_path)

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        out.write(annotated_frame)

    cap.release()
    out.release()

    return send_file(
        output_path,
        as_attachment=True,
        download_name="pothole_detected.mp4",
        mimetype="video/mp4"
    )

if __name__ == "__main__":
    app.run(debug=True)