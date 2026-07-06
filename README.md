# Pothole Detection System

This project is a Flask-based pothole detection API. It uses a custom YOLO model with OpenCV to detect potholes in uploaded images and videos. For videos, it can also match detected potholes with GPS coordinates and export the results to a CSV file.

## Project Overview

The application exposes HTTP endpoints for:

- Checking whether the Flask app is running.
- Uploading an image and receiving an annotated image with pothole detections.
- Uploading a video and receiving an annotated video with pothole detections.
- Downloading a CSV file containing pothole detection records from the last processed video.

The main application file is `app.py`. The older `main.py` file contains a similar implementation with GPS and reverse-geocoding logic written directly in the same file. The current app is cleaner because GPS loading and reverse geocoding are split into helper modules inside `gps_data/`.

## Tech Stack

- Python
- Flask
- OpenCV
- NumPy
- Ultralytics YOLO
- OpenStreetMap Nominatim reverse geocoding

## Project Structure

```text
pothole detection/
├── app.py                         # Main Flask API used by the project
├── main.py                        # Earlier combined version of the app
├── requirements.txt               # Python dependencies
├── yolo_model_training.ipynb      # Notebook used for YOLO model training
├── model/
│   └── best.pt                    # Trained YOLO model weights
├── gps_data/
│   ├── gps_data.csv               # GPS timestamp data used by app.py
│   ├── gps_log.csv                # Extra GPS sample/log file
│   ├── gps_log_1.csv              # Extra GPS sample/log file
│   ├── load_gps.py                # GPS CSV loader and timestamp matcher
│   └── reverse_geocode.py         # Converts latitude/longitude to address text
├── source/
│   ├── photo_detected.jpg         # Sample detected image
│   └── photo_detected_2.jpg       # Latest image endpoint output
└── source2/
    ├── input_potholes_video.mp4   # Uploaded video saved by the API
    ├── output_potholes_video.mp4  # Annotated video output
    └── potholes_detected.csv      # CSV records generated from video detections
```

## Model Training

The model training work is documented in `yolo_model_training.ipynb`.

The notebook trains a YOLO model on pothole images, saves the best trained weights as `best.pt`, and the Flask app loads those weights from:

```text
model/best.pt
```

In `app.py`, the model is loaded once when the server starts:

```python
model = YOLO("model/best.pt")
```

## Installation

Create and activate a Python virtual environment, then install the required packages:

```bash
pip install -r requirements.txt
```

If reverse geocoding fails because `requests` is missing, install it:

```bash
pip install requests
```

## Running the App

Start the Flask server:

```bash
python app.py
```

By default, Flask runs at:

```text
http://127.0.0.1:5000
```

Open the root route to check the server:

```text
GET /
```

Expected response:

```text
App is running
```

## API Endpoints

### 1. Health Check

```text
GET /
```

Returns a simple text response confirming that the app is running.

### 2. Detect Potholes in an Image

```text
POST /detect_potholes_images
```

Form-data field:

```text
image
```

Flow:

1. The uploaded image is read from the request.
2. The image bytes are converted into a NumPy array.
3. OpenCV decodes the array into an image.
4. The YOLO model detects potholes.
5. The detected image is annotated with bounding boxes.
6. The annotated image is saved to `source/photo_detected_2.jpg`.
7. The annotated image is returned as `detected_image.jpg`.

Example using curl:

```bash
curl -X POST http://127.0.0.1:5000/detect_potholes_images \
  -F "image=@path/to/image.jpg" \
  --output detected_image.jpg
```

### 3. Detect Potholes in a Video

```text
POST /detect_potholes_videos
```

Form-data field:

```text
video
```

Flow:

1. The uploaded video is saved to `source2/input_potholes_video.mp4`.
2. OpenCV opens the video and reads its width, height, and FPS.
3. Each frame is passed to the YOLO model.
4. Detected potholes are drawn on each frame.
5. For every detection, the app calculates the frame timestamp.
6. The timestamp is matched with the closest GPS point from `gps_data/gps_data.csv`.
7. The latitude, longitude, detection confidence, timestamp, and city/address are saved to CSV.
8. The annotated video is saved to `source2/output_potholes_video.mp4`.
9. The annotated video is returned as `pothole_detected.mp4`.

Example using curl:

```bash
curl -X POST http://127.0.0.1:5000/detect_potholes_videos \
  -F "video=@path/to/video.mp4" \
  --output pothole_detected.mp4
```

### 4. Download Detection CSV

```text
GET /download_potholes_data
```

Returns:

```text
source2/potholes_detected.csv
```

Example:

```bash
curl http://127.0.0.1:5000/download_potholes_data --output potholes.csv
```

## GPS Data Format

`app.py` loads GPS data from:

```text
gps_data/gps_data.csv
```

The GPS CSV should contain these columns:

```text
timestamp,latitude,longitude
```

Example:

```csv
timestamp,latitude,longitude
0.0,12.9716,77.5946
1.0,12.9717,77.5947
2.0,12.9718,77.5948
```

During video processing, the app calculates each frame timestamp using:

```python
timestamp = frame_index / fps
```

Then it picks the GPS row whose timestamp is closest to that frame timestamp.

## CSV Output

Video detections are exported to:

```text
source2/potholes_detected.csv
```

The CSV columns are:

```text
timestamp,lat,lon,confidence,city
```

Where:

- `timestamp` is the video time in seconds.
- `lat` and `lon` are the closest GPS coordinates for that video time.
- `confidence` is the YOLO detection confidence.
- `city` is the reverse-geocoded location returned by Nominatim.

## Important Notes

- The model file `model/best.pt` must exist before running the app.
- The folders `source/` and `source2/` must exist because the app writes output files there.
- Reverse geocoding requires internet access because it calls the Nominatim API.
- The app sleeps for one second between new reverse-geocoding requests to avoid sending too many requests too quickly.
- `pothole_records` is an in-memory list. If multiple videos are processed during one server run, records can accumulate unless the app is restarted or the list is cleared in code.
- `main.py` and `app.py` have similar logic, but `app.py` is the better organized version to run.

## Sample Outputs

Detected video:

[Watch sample output video](source/output_potholes_video.mp4)

Detected images:

![Detected pothole image](source/photo_detected.jpg)

![Detected pothole image 2](source/photo_detected_2.jpg)
