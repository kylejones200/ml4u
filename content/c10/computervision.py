"""Chapter 10: Computer Vision for Utilities."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from ultralytics import YOLO
import geopandas as gpd
from shapely.geometry import Point

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)


def setup_yolo_model():
    """Initialize YOLOv8 nano model."""
    return YOLO(config["yolo"]["model_path"])


def train_model(yaml_path, epochs=None):
    """Train YOLOv8 on a dataset with power lines and vegetation."""
    model = setup_yolo_model()
    epochs = epochs or config["yolo"]["epochs"]
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=config["yolo"]["imgsz"],
        batch=config["yolo"]["batch"]
    )
    return model


def evaluate_model(model):
    """Validate YOLOv8 model."""
    results = model.val()
    print(results)


def run_inference(model, image_path, output_path="detected_output.jpg"):
    """Run inference and visualize detections."""
    results = model(image_path)
    results[0].save(filename=output_path)

    img = plt.imread(output_path)
    fig, ax = plt.subplots(figsize=config["plotting"]["figsize_inference"])
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Detected Power Lines and Vegetation Encroachment")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(config["plotting"]["output_files"]["vegetation"])
    plt.close()
    return results[0]


def detect_vegetation_risk(model_result, gps_metadata=None):
    """Analyze vegetation encroachment proximity and map risks geospatially."""
    boxes = model_result.boxes.xyxy.cpu().numpy()
    classes = model_result.boxes.cls.cpu().numpy()

    power_lines = boxes[classes == 0]
    vegetation = boxes[classes == 1]

    risk_points = []
    risk_dist = config["detection"]["risk_distance"]
    for v_box in vegetation:
        v_center = [(v_box[0] + v_box[2]) / 2, (v_box[1] + v_box[3]) / 2]
        for p_box in power_lines:
            p_center = [(p_box[0] + p_box[2]) / 2, (p_box[1] + p_box[3]) / 2]
            distance = np.linalg.norm([v_center[0]-p_center[0], v_center[1]-p_center[1]])
            if distance <= risk_dist:
                if gps_metadata:
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
    """Plot geospatial vegetation risk points over utility area."""
    fig, ax = plt.subplots(figsize=config["plotting"]["figsize_geospatial"])
    if base_map is not None:
        base_map.plot(ax=ax, color="lightgray", linewidth=0.5)
    gdf.plot(ax=ax, color="red", markersize=50, label="Vegetation Risk")
    ax.set_title("Vegetation Encroachment Risk Points - Latitude vs Longitude")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.savefig(config["plotting"]["output_files"]["geospatial"])
    plt.close()


if __name__ == "__main__":
    dataset_yaml = config["paths"]["dataset_yaml"]
    if not os.path.exists(dataset_yaml):
        print(f"Dataset YAML not found at '{dataset_yaml}'.\n"
              "Skipping YOLO training and inference.\n"
              "To download sample data, run: content/scripts/fetch_computer_vision_data.py")
    else:
        model = train_model(dataset_yaml, epochs=1)
        evaluate_model(model)

        sample_image = config["paths"]["sample_image"]
        if not os.path.exists(sample_image):
            print("Sample image not found; skipping inference and mapping.")
        else:
            gps_metadata = {
                "lat": config["detection"]["gps_lat"],
                "lon": config["detection"]["gps_lon"],
                "alt": config["detection"]["gps_alt"]
            }
            result = run_inference(model, sample_image)
            gdf_risks = detect_vegetation_risk(result, gps_metadata=gps_metadata)
            plot_geospatial_risks(gdf_risks)
