"""Chapter 2: Data in Power and Utilities."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)


def load_smart_meter_data(file_path):
    """Load smart meter data."""
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df = df.rename(columns={"consumption_kwh": "Consumption_kWh"})
    print(f"Smart meter data loaded: {df.shape[0]} rows")
    return df


def clean_and_resample(df):
    """Clean missing values and resample hourly."""
    df = df.set_index("timestamp").sort_index()
    df = df.resample(config["data"]["resample_freq"]).mean()
    df["Consumption_kWh"] = df["Consumption_kWh"].ffill()
    return df


def plot_consumption(df):
    """Plot hourly consumption."""
    fig, ax = plt.subplots(figsize=config["plotting"]["figsize"])
    ax.plot(df.index, df["Consumption_kWh"], color=config["plotting"]["colors"]["consumption"])
    ax.set_title("Hourly Smart Meter Consumption (kWh)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(config["plotting"]["output_files"]["smart_meter"])
    plt.close()


def generate_synthetic_scada_data():
    """Generate synthetic SCADA-like grid frequency data."""
    time = pd.date_range(
        "2022-01-01",
        periods=config["data"]["scada_samples"],
        freq=config["data"]["scada_freq"]
    )
    freq = (config["data"]["nominal_frequency"] + 
            np.random.normal(0, config["data"]["frequency_noise"], len(time)))
    return pd.DataFrame({"timestamp": time, "frequency_hz": freq})


def plot_scada(df):
    """Plot SCADA-like frequency data."""
    fig, ax = plt.subplots(figsize=config["plotting"]["figsize"])
    ax.plot(df["timestamp"], df["frequency_hz"], color=config["plotting"]["colors"]["frequency"])
    ax.set_title("Grid Frequency (Hz) - SCADA Telemetry")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(config["plotting"]["output_files"]["scada"])
    plt.close()


if __name__ == "__main__":
    # Example with synthetic smart meter data
    smart_meter_data = pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=96, freq="15min"),
        "consumption_kwh": np.random.uniform(0.2, 1.5, size=96)
    })
    os.makedirs("data", exist_ok=True)
    smart_meter_data.to_csv("data/smart_meter_sample.csv", index=False)

    df_meter = load_smart_meter_data("data/smart_meter_sample.csv")
    df_meter = clean_and_resample(df_meter)
    plot_consumption(df_meter)

    # SCADA synthetic example
    df_scada = generate_synthetic_scada_data()
    plot_scada(df_scada)
