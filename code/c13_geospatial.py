"""Chapter 13: Integration with Enterprise Systems (GIS, SCADA, EAM)."""

import logging
import os
import pandas as pd
import geopandas as gpd
import yaml
from pathlib import Path
from kafka import KafkaConsumer
import json
import threading

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)


def load_feeder_map():
    """Load GIS feeder data as a GeoDataFrame."""
    shapefile = config["paths"]["shapefile"]
    if not os.path.exists(shapefile):
        logger.warning(f"Shapefile not found at '{shapefile}'")
        return None
    gdf = gpd.read_file(shapefile)
    logger.info(f"Loaded GIS map: {len(gdf)} features")
    return gdf


def plot_assets_on_map(feeder_map, assets_df):
    """Plot asset locations (transformers) on feeder map."""
    assets_gdf = gpd.GeoDataFrame(
        assets_df,
        geometry=gpd.points_from_xy(assets_df["Longitude"], assets_df["Latitude"]),
        crs="EPSG:4326"
    )
    if feeder_map is not None:
        base = feeder_map.plot(color="lightgray", figsize=(8, 8))
        assets_gdf.plot(ax=base, color="red", markersize=50)
    else:
        assets_gdf.plot(color="red", figsize=(8, 8), markersize=50)


def load_eam_data():
    """Load synthetic asset data from EAM."""
    return pd.DataFrame({
        "AssetID": ["TX-101", "TX-102", "TX-103"],
        "Latitude": [30.25, 30.28, 30.26],
        "Longitude": [-97.72, -97.75, -97.70],
        "Age_Years": [15, 20, 8],
        "Condition": ["Fair", "Poor", "Good"]
    })


def start_scada_stream():
    """Simulate SCADA stream consumption."""
    consumer = KafkaConsumer(
        config["kafka"]["topic"],
        bootstrap_servers=config["kafka"]["bootstrap_servers"],
        value_deserializer=lambda x: json.loads(x.decode("utf-8"))
    )
    logger.info("Listening to SCADA stream")
    for msg in consumer:
        data = msg.value
        logger.debug(f"SCADA: Transformer={data['TransformerID']} "
              f"Temp={data['Temperature_C']}C Vibration={data['Vibration_g']}g")


if __name__ == "__main__":
    feeders = load_feeder_map()
    eam_assets = load_eam_data()
    plot_assets_on_map(feeders, eam_assets)

    if os.environ.get("ML4U_CI") == "1":
        logger.info("ML4U_CI=1: skipping Kafka consumption")
    else:
        threading.Thread(target=start_scada_stream, daemon=True).start()
