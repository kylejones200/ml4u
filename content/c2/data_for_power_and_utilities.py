"""
Chapter 2: Data in Power and Utilities
Loading, cleaning, and visualizing AMI (smart meter) and SCADA-like time series.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_smart_meter_data(file_path):
    """
    Load smart meter data.

    Args:
        file_path (str): Path to CSV file.

    Returns:
        pd.DataFrame: Cleaned smart meter data.
    """
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df = df.rename(columns={"consumption_kwh": "Consumption_kWh"})
    print(f"Smart meter data loaded: {df.shape[0]} rows")
    return df

def clean_and_resample(df):
    """
    Clean missing values and resample hourly.
    """
    df = df.set_index("timestamp").sort_index()
    df = df.resample("H").mean()
    df["Consumption_kWh"] = df["Consumption_kWh"].fillna(method="ffill")
    return df

def plot_consumption(df):
    """
    Plot hourly consumption.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df["Consumption_kWh"], color="black")
    plt.xlabel("Time")
    plt.ylabel("Consumption (kWh)")
    plt.title("Hourly Smart Meter Consumption")
    plt.tight_layout()
    plt.savefig("chapter2_smart_meter_plot.png")
    plt.show()

def generate_synthetic_scada_data():
    """
    Generate synthetic SCADA-like grid frequency data.
    """
    time = pd.date_range("2022-01-01", periods=1440, freq="T")  # 1 day of minute data
    freq = 60 + np.random.normal(0, 0.02, size=1440)  # Nominal 60 Hz with noise
    return pd.DataFrame({"timestamp": time, "frequency_hz": freq})

def plot_scada(df):
    """
    Plot SCADA-like frequency data.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(df["timestamp"], df["frequency_hz"], color="black")
    plt.xlabel("Time")
    plt.ylabel("Frequency (Hz)")
    plt.title("Synthetic SCADA Grid Frequency")
    plt.tight_layout()
    plt.savefig("chapter2_scada_frequency.png")
    plt.show()

if __name__ == "__main__":
    # Example with synthetic smart meter data
    smart_meter_data = pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=96, freq="15T"),
        "consumption_kwh": np.random.uniform(0.2, 1.5, size=96)
    })
    smart_meter_data.to_csv("data/smart_meter_sample.csv", index=False)

    df_meter = load_smart_meter_data("data/smart_meter_sample.csv")
    df_meter = clean_and_resample(df_meter)
    plot_consumption(df_meter)

    # SCADA synthetic example
    df_scada = generate_synthetic_scada_data()
    plot_scada(df_scada)
