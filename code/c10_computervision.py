"""Chapter 10: Computer Vision for Utilities."""

import logging
import os
import numpy as np
import pandas as pd
import signalplot as sp
import yaml
from pathlib import Path
from ultralytics import YOLO
import geopandas as gpd
from shapely.geometry import Point

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

sp.apply()

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
    logger.debug(results)


def run_inference(model, image_path, output_path="detected_output.jpg"):
    """Run inference and visualize detections."""
    results = model(image_path)
    results[0].save(filename=output_path)

    img = sp.mpl.pyplot.imread(output_path)
    fig, ax = sp.figure()
    ax.imshow(img)
    ax.axis("off")
    sp.savefig(config["plotting"]["output_files"]["vegetation"])
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
            distance = np.linalg.norm([
                v_center[0] - p_center[0],
                v_center[1] - p_center[1]
            ])
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
    logger.info(f"Vegetation risk points: {len(gdf)}")
    return gdf


def plot_geospatial_risks(gdf, base_map=None):
    """Plot geospatial vegetation risk points over utility area."""
    fig, ax = sp.figure()
    if base_map is not None:
        base_map.plot(ax=ax, color="lightgray", linewidth=0.5)
    gdf.plot(ax=ax, color="red", markersize=50)
    sp.style_scatter_plot(ax)
    sp.savefig(config["plotting"]["output_files"]["geospatial"])


if __name__ == "__main__":
    dataset_yaml = config["paths"]["dataset_yaml"]
    if not os.path.exists(dataset_yaml):
        logger.warning(f"Dataset YAML not found at '{dataset_yaml}'")
    else:
        model = train_model(dataset_yaml, epochs=1)
        evaluate_model(model)

        sample_image = config["paths"]["sample_image"]
        if not os.path.exists(sample_image):
            logger.warning("Sample image not found; skipping inference")
        else:
            gps_metadata = {
                "lat": config["detection"]["gps_lat"],
                "lon": config["detection"]["gps_lon"],
                "alt": config["detection"]["gps_alt"]
            }
            result = run_inference(model, sample_image)
            gdf_risks = detect_vegetation_risk(result, gps_metadata=gps_metadata)
            plot_geospatial_risks(gdf_risks)
