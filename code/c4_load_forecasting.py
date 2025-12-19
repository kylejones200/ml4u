"""Chapter 4: Load Forecasting and Demand Analytics."""

import logging
import os
import pandas as pd
import numpy as np
import signalplot as sp
import yaml
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

sp.apply()

# Optional: Darts dependencies
HAS_DARTS = True
try:
    from darts import TimeSeries
    from darts.models import RNNModel
    from darts.dataprocessing.transformers import Scaler
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
    fig, ax = sp.figure()
    ax.plot(df["timestamp"], df["Load_MW"])
    sp.style_line_plot(ax)
    sp.savefig(config["plotting"]["output_files"]["load"])


def arima_forecast(df):
    """Forecast using ARIMA."""
    ts = df.set_index("timestamp")["Load_MW"]
    order = tuple(config["arima"]["order"])
    model = ARIMA(ts, order=order)
    fit = model.fit()
    forecast = fit.forecast(steps=config["arima"]["forecast_steps"])

    fig, ax = sp.figure()
    ax.plot(ts[-config["arima"]["forecast_steps"]:])
    ax.plot(forecast, linestyle="--")
    sp.style_line_plot(ax)
    sp.savefig(config["plotting"]["output_files"]["arima"])


def lstm_forecast(df):
    """Forecast using LSTM (Darts)."""
    if not HAS_DARTS:
        logger.warning("Darts not available; skipping LSTM forecast.")
        return
    
    df = df.copy()
    df["Load_MW"] = df["Load_MW"].astype(np.float32)
    series = TimeSeries.from_dataframe(df, "timestamp", "Load_MW")
    forecast_steps = config["arima"]["forecast_steps"]
    train = series[:-forecast_steps]
    val = series[-forecast_steps:]
    
    # Scale the data for better LSTM training
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train)
    
    # Use output_chunk_length=1 for autoregressive forecasting
    model = RNNModel(
        model="LSTM",
        input_chunk_length=config["lstm"]["input_chunk_length"],
        output_chunk_length=1,
        training_length=config["lstm"]["training_length"],
        n_epochs=50,
        random_state=config["model"]["random_state"],
        pl_trainer_kwargs={
            "accelerator": "cpu",
            "devices": 1,
            "enable_progress_bar": False
        },
    )
    model.fit(train_scaled)
    pred_scaled = model.predict(forecast_steps)
    
    # Inverse transform to get predictions in original scale
    pred = scaler.inverse_transform(pred_scaled)

    # Extract values for plotting - use pandas for proper datetime handling
    train_series = train[-forecast_steps:]
    train_df = train_series.pd_dataframe()
    pred_df = pred.pd_dataframe()

    fig, ax = sp.figure()
    ax.plot(train_df.index, train_df.iloc[:, 0], lw=1, label="Historical")
    ax.plot(pred_df.index, pred_df.iloc[:, 0], lw=2, linestyle="--", label="Forecast")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("Load_MW")
    sp.style_line_plot(ax)
    sp.savefig(config["plotting"]["output_files"]["lstm"])


if __name__ == "__main__":
    df_load = generate_synthetic_load()
    plot_load(df_load)
    arima_forecast(df_load)
    lstm_forecast(df_load)
