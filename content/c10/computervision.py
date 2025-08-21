"""
Chapter 10: Computer Vision for Utilities
Power line and vegetation encroachment detection using YOLOv8 with geospatial mapping.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import geopandas as gpd
from shapely.geometry import Point

def setup_yolo_model():
    """
    Initialize YOLOv8 nano model.
    """
    return YOLO("yolov8n.pt")

def train_model(yaml_path, epochs=30):
    """
    Train YOLOv8 on a dataset with power lines and vegetation.
    """
    model = setup_yolo_model()
    model.train(data=yaml_path, epochs=epochs, imgsz=640, batch=16)
    return model

def evaluate_model(model):
    """
    Validate YOLOv8 model.
    """
    results = model.val()
    print(results)

def run_inference(model, image_path, output_path="detected_output.jpg"):
    """
    Run inference and visualize detections.
    """
    results = model(image_path)
    results[0].save(filename=output_path)

    img = plt.imread(output_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Detected Power Lines and Vegetation Encroachment")
    plt.tight_layout()
    plt.savefig("chapter10_vegetation_detection.png")
    plt.show()
    return results[0]

def detect_vegetation_risk(model_result, risk_distance=50, gps_metadata=None):
    """
    Analyze vegetation encroachment proximity and map risks geospatially.
    
    Args:
        model_result: YOLO results object.
        risk_distance (int): Pixel threshold for vegetation proximity.
        gps_metadata (dict): Contains {'lat': float, 'lon': float, 'alt': float}.
    
    Returns:
        GeoDataFrame of vegetation risk points.
    """
    boxes = model_result.boxes.xyxy.cpu().numpy()
    classes = model_result.boxes.cls.cpu().numpy()

    power_lines = boxes[classes == 0]
    vegetation = boxes[classes == 1]

    risk_points = []
    for v_box in vegetation:
        v_center = [(v_box[0] + v_box[2]) / 2, (v_box[1] + v_box[3]) / 2]
        for p_box in power_lines:
            p_center = [(p_box[0] + p_box[2]) / 2, (p_box[1] + p_box[3]) / 2]
            distance = np.linalg.norm([v_center[0]-p_center[0], v_center[1]-p_center[1]])
            if distance <= risk_distance:
                if gps_metadata:
                    # Convert pixel offsets to approximate geospatial offsets (simple scaling)
                    lat_offset = (v_center[1] - 320) * 1e-5
                    lon_offset = (v_center[0] - 320) * 1e-5
                    lat = gps_metadata["lat"] + lat_offset
                    lon = gps_metadata["lon"] + lon_offset
                    risk_points.append(Point(lon, lat))
                else:
                    risk_points.append(Point(v_center[0], v_center[1]))

    gdf = gpd.GeoDataFrame(geometry=risk_points, crs="EPSG:4326")
    print(f"Vegetation risk points mapped: {len(gdf)}")
    return gdf

def plot_geospatial_risks(gdf, base_map=None):
    """
    Plot geospatial vegetation risk points over utility area.
    
    Args:
        gdf (GeoDataFrame): Risk points.
        base_map (GeoDataFrame): Optional utility asset map (e.g., feeders).
    """
    plt.figure(figsize=(8, 8))
    if base_map is not None:
        base_map.plot(ax=plt.gca(), color="lightgray", linewidth=0.5)
    gdf.plot(ax=plt.gca(), color="red", markersize=50, label="Vegetation Risk")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Vegetation Encroachment Risks on Utility Map")
    plt.legend()
    plt.tight_layout()
    plt.savefig("chapter10_geospatial_risks.png")
    plt.show()

if __name__ == "__main__":
    # Dataset YAML file (with classes: powerline=0, vegetation=1)
    dataset_yaml = "data/PLD/pld_vegetation.yaml"
    model = train_model(dataset_yaml, epochs=30)
    evaluate_model(model)

    # Inference on sample image with GPS metadata
    sample_image = "data/PLD/images/test/sample_vegetation.jpg"
    gps_metadata = {"lat": 30.2672, "lon": -97.7431, "alt": 120}  # Example GPS from drone EXIF
    result = run_inference(model, sample_image)

    # Detect vegetation risk and generate geospatial mapping
    gdf_risks = detect_vegetation_risk(result, risk_distance=50, gps_metadata=gps_metadata)

    # Plot geospatial risk points (optionally overlay with feeder shapefile if available)
    plot_geospatial_risks(gdf_risks)
