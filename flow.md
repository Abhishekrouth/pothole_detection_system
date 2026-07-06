# Application Flow

This file explains how the pothole detection project works from startup to final output.

## 1. Server Startup Flow

```text
Start app.py
    |
    v
Create Flask app
    |
    v
Load YOLO model from model/best.pt
    |
    v
Load GPS rows from gps_data/gps_data.csv
    |
    v
Start Flask development server
```

When `python app.py` runs, the app loads the trained YOLO model once:

```python
model = YOLO("model/best.pt")
```

It also loads GPS data once:

```python
gps_data = load_gps_data("gps_data/gps_data.csv")
```

The GPS data is kept in memory so video frames can quickly be matched to nearby GPS timestamps.

## 2. Image Detection Flow

Endpoint:

```text
POST /detect_potholes_images
```

Expected request field:

```text
image
```

Flow:

```text
Client uploads image
    |
    v
Flask reads image bytes from request.files["image"]
    |
    v
NumPy converts bytes into an array
    |
    v
OpenCV decodes array into an image frame
    |
    v
YOLO model runs detection on the image
    |
    v
YOLO draws bounding boxes and labels
    |
    v
Annotated image is saved to source/photo_detected_2.jpg
    |
    v
Flask returns detected_image.jpg to the client
```

Important code steps:

```python
image_file = request.files["image"].read()
np_img = np.frombuffer(image_file, np.uint8)
img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
results = model(img)
annotated = results[0].plot()
cv2.imwrite("source/photo_detected_2.jpg", annotated)
```

The endpoint returns the processed image as a downloadable JPEG.

## 3. Video Detection Flow

Endpoint:

```text
POST /detect_potholes_videos
```

Expected request field:

```text
video
```

Flow:

```text
Client uploads video
    |
    v
Flask saves upload to source2/input_potholes_video.mp4
    |
    v
OpenCV opens the saved video
    |
    v
Read video width, height, and FPS
    |
    v
Create output video writer
    |
    v
Loop through frames
    |
    v
Run YOLO detection on each frame
    |
    v
For each detected pothole:
        calculate video timestamp
        find closest GPS row
        save timestamp, lat, lon, and confidence in memory
    |
    v
Draw detection boxes on the frame
    |
    v
Write annotated frame to output video
    |
    v
After all frames, save pothole records to CSV
    |
    v
Return output_potholes_video.mp4 to the client
```

The video is saved here:

```text
source2/input_potholes_video.mp4
```

The output video is saved here:

```text
source2/output_potholes_video.mp4
```

The CSV is saved here:

```text
source2/potholes_detected.csv
```

## 4. Frame Processing Logic

For each video frame, the app calculates the timestamp:

```python
timestamp = frame_index / fps
```

For example, if the video is 25 FPS:

```text
frame 0  -> 0.00 seconds
frame 25 -> 1.00 seconds
frame 50 -> 2.00 seconds
```

The app runs YOLO on the frame:

```python
results = model(frame, conf=0.5)
```

The `conf=0.5` setting means detections below 50% confidence are ignored.

For every detected box, the app stores:

```text
timestamp
latitude
longitude
confidence
```

## 5. GPS Matching Flow

GPS data comes from:

```text
gps_data/gps_data.csv
```

The helper function `load_gps_data()` reads the CSV into a list of points:

```text
[
  {"timestamp": 0.0, "lat": 12.9716, "long": 77.5946},
  {"timestamp": 1.0, "lat": 12.9717, "long": 77.5947}
]
```

When a pothole is detected at a video timestamp, `get_gps_for_timestamp()` finds the GPS point with the closest timestamp.

Example:

```text
Detection timestamp: 1.2 seconds

GPS points:
1.0 seconds
2.0 seconds

Closest GPS point: 1.0 seconds
```

That closest GPS latitude and longitude are attached to the pothole detection record.

## 6. Reverse Geocoding Flow

After video processing, the app writes the pothole records to CSV.

Before writing each row, it converts latitude and longitude into a readable location using:

```python
geocode(record["lat"], record["lon"])
```

That helper calls the OpenStreetMap Nominatim reverse geocoding API:

```text
https://nominatim.openstreetmap.org/reverse
```

The result is stored in the CSV as the `city` column.

The app also uses `location_cache`:

```text
Same latitude/longitude seen again
    |
    v
Use cached address instead of calling API again
```

This reduces repeated network calls for the same coordinates.

## 7. CSV Export Flow

CSV export happens after the full video has been processed.

Flow:

```text
Collected pothole records
    |
    v
Open source2/potholes_detected.csv
    |
    v
Write CSV header
    |
    v
For each pothole:
        reverse geocode location if needed
        write timestamp, lat, lon, confidence, city
    |
    v
CSV is ready for download
```

CSV columns:

```text
timestamp,lat,lon,confidence,city
```

Download endpoint:

```text
GET /download_potholes_data
```

## 8. Complete End-to-End Flow

```text
Train YOLO model in yolo_model_training.ipynb
    |
    v
Save best model as model/best.pt
    |
    v
Run python app.py
    |
    v
Upload image or video through API
    |
    v
YOLO detects potholes
    |
    v
OpenCV saves annotated output
    |
    v
For video: GPS data is attached to each detection
    |
    v
For video: coordinates are reverse geocoded
    |
    v
User receives annotated image/video and can download CSV
```

## 9. Notes About app.py and main.py

Both files implement a pothole detection Flask app, but they are organized differently.

`main.py` keeps GPS loading and reverse geocoding directly inside the main file. It also references `data/gps_log.csv`, which does not match the current `gps_data/` folder layout.

`app.py` imports helper functions:

```python
from gps_data.load_gps import load_gps_data, get_gps_for_timestamp
from gps_data.reverse_geocode import geocode
```

For this project structure, `app.py` is the recommended file to run.
