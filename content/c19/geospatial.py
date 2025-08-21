"""
Chapter 19: Integration with Enterprise Systems (GIS, SCADA, EAM)
Combine GIS (asset maps), SCADA telemetry, and EAM records for integrated analytics.
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from kafka import KafkaConsumer
import json
import threading

# --- GIS Integration ---
def load_feeder_map(shapefile="data/gis/feeders.shp"):
    """
    Load GIS feeder data as a GeoDataFrame.
    """
    gdf = gpd.read_file(shapefile)
    print(f"Loaded feeder GIS map with {len(gdf)} features.")
    return gdf

def plot_assets_on_map(feeder_map, assets_df):
    """
    Plot asset locations (transformers) on feeder map.
    """
    assets_gdf = gpd.GeoDataFrame(
        assets_df,
        geometry=gpd.points_from_xy(assets_df["Longitude"], assets_df["Latitude"]),
        crs="EPSG:4326"
    )
    base = feeder_map.plot(color="lightgray", figsize=(8, 8))
    assets_gdf.plot(ax=base, color="red", markersize=50)
    print("Plotted assets on feeder map.")

# --- EAM Integration ---
def load_eam_data():
    """
    Load synthetic asset data from EAM.
    """
    return pd.DataFrame({
        "AssetID": ["TX-101", "TX-102", "TX-103"],
        "Latitude": [30.25, 30.28, 30.26],
        "Longitude": [-97.72, -97.75, -97.70],
        "Age_Years": [15, 20, 8],
        "Condition": ["Fair", "Poor", "Good"]
    })

# --- SCADA Streaming Integration ---
def start_scada_stream(topic="scada_stream", bootstrap="localhost:9092"):
    """
    Simulate SCADA stream consumption.
    """
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        value_deserializer=lambda x: json.loads(x.decode("utf-8"))
    )
    print("Listening to SCADA telemetry stream...")
    for msg in consumer:
        data = msg.value
        print(f"[SCADA Update] Transformer={data['TransformerID']} Temp={data['Temperature_C']}C Vibration={data['Vibration_g']}g")

if __name__ == "__main__":
    feeders = load_feeder_map()
    eam_assets = load_eam_data()
    plot_assets_on_map(feeders, eam_assets)

    # Start SCADA streaming in background
    threading.Thread(target=start_scada_stream, daemon=True).start()
