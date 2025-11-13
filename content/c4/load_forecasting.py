"""Chapter 4: Load Forecasting and Demand Analytics."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA

# Optional: Darts dependencies
HAS_DARTS = True
try:
    from darts import TimeSeries
    from darts.models import RNNModel
    from darts.metrics import rmse
except Exception:
    HAS_DARTS = False

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

np.random.seed(config["model"]["random_state"])


def generate_synthetic_load():
    """Generate synthetic hourly load data for one year."""
    date_rng = pd.date_range(
        start=config["data"]["start_date"],
        periods=config["data"]["periods"],
        freq="h"
    )
    base = config["data"]["base_load"]
    seasonal = config["data"]["seasonal_amplitude"]
    daily = config["data"]["daily_cycle_amplitude"]
    noise_std = config["data"]["noise_std"]
    
    base_load = base + seasonal * np.sin(2 * np.pi * date_rng.dayofyear / 365)
    daily_cycle = daily * np.sin(2 * np.pi * date_rng.hour / 24)
    noise = np.random.normal(0, noise_std, len(date_rng))
    load = base_load + daily_cycle + noise
    
    return pd.DataFrame({"timestamp": date_rng, "Load_MW": load})


def plot_load(df):
    """Plot hourly load."""
    fig, ax = plt.subplots(figsize=config["plotting"]["figsize"])
    ax.plot(df["timestamp"], df["Load_MW"], color=config["plotting"]["colors"]["load"])
    ax.set_title("Hourly Load (MW) Over One Year")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(config["plotting"]["output_files"]["load"])
    plt.close()


def arima_forecast(df):
    """Forecast using ARIMA."""
    ts = df.set_index("timestamp")["Load_MW"]
    order = tuple(config["arima"]["order"])
    model = ARIMA(ts, order=order)
    fit = model.fit()
    forecast = fit.forecast(steps=config["arima"]["forecast_steps"])

    fig, ax = plt.subplots(figsize=config["plotting"]["figsize"])
    ax.plot(ts[-config["arima"]["forecast_steps"]:], 
             label="Observed", color=config["plotting"]["colors"]["observed"])
    ax.plot(forecast, label="ARIMA Forecast", 
             color=config["plotting"]["colors"]["forecast"])
    ax.set_title("Load (MW) - ARIMA Forecast vs Observed")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.savefig(config["plotting"]["output_files"]["arima"])
    plt.close()


def lstm_forecast(df):
    """Forecast using LSTM (Darts)."""
    if not HAS_DARTS:
        print("Darts not available; skipping LSTM forecast.")
        return
    
    df = df.copy()
    df["Load_MW"] = df["Load_MW"].astype(np.float32)
    series = TimeSeries.from_dataframe(df, "timestamp", "Load_MW")
    train, val = series[:-config["arima"]["forecast_steps"]], series[-config["arima"]["forecast_steps"]:]
    
    model = RNNModel(
        model="LSTM",
        input_chunk_length=config["lstm"]["input_chunk_length"],
        output_chunk_length=config["lstm"]["output_chunk_length"],
        training_length=config["lstm"]["training_length"],
        n_epochs=config["lstm"]["n_epochs"],
        random_state=config["model"]["random_state"],
        pl_trainer_kwargs={"accelerator": "cpu", "devices": 1, "enable_progress_bar": False},
    )
    model.fit(train)
    pred = model.predict(len(val))

    fig, ax = plt.subplots(figsize=config["plotting"]["figsize"])
    train[-config["arima"]["forecast_steps"]:].plot(
        ax=ax, label="Observed", lw=1, color=config["plotting"]["colors"]["observed"])
    pred.plot(ax=ax, label="LSTM Forecast", lw=2, color=config["plotting"]["colors"]["forecast"])
    ax.set_title(f"Load (MW) - LSTM Forecast vs Observed (RMSE: {rmse(val, pred):.2f})")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.savefig(config["plotting"]["output_files"]["lstm"])
    plt.close()


if __name__ == "__main__":
    df_load = generate_synthetic_load()
    plot_load(df_load)
    arima_forecast(df_load)
    lstm_forecast(df_load)
