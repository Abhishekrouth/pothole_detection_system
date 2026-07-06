import csv

def load_gps_data(csv_path):
    gps_points = []

    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            gps_points.append({
                "timestamp": float(row["timestamp"]),
                "lat": float(row["latitude"]),
                "long": float(row["longitude"])
            })

    return gps_points

def get_gps_for_timestamp(timestamp, gps_data):
    
    closest_point = min(gps_data,key=lambda point: abs(point["timestamp"] - timestamp))
    return closest_point["lat"], closest_point["long"]