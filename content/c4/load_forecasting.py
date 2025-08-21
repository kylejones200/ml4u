"""
Chapter 4: Load Forecasting and Demand Analytics
Time series forecasting using ARIMA, Prophet, and LSTM (Darts).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from darts import TimeSeries
from darts.models import RNNModel
from darts.metrics import rmse

def generate_synthetic_load():
    """
    Generate synthetic hourly load data for one year with seasonality and noise.
    """
    date_rng = pd.date_range(start="2022-01-01", periods=24*365, freq="H")
    base_load = 800 + 150 * np.sin(2 * np.pi * date_rng.dayofyear / 365)
    daily_cycle = 50 * np.sin(2 * np.pi * date_rng.hour / 24)
    noise = np.random.normal(0, 20, len(date_rng))
    load = base_load + daily_cycle + noise
    return pd.DataFrame({"timestamp": date_rng, "Load_MW": load})

def plot_load(df):
    """
    Plot hourly load.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(df["timestamp"], df["Load_MW"], color="black")
    plt.xlabel("Time")
    plt.ylabel("Load (MW)")
    plt.title("Hourly Load Profile")
    plt.tight_layout()
    plt.savefig("chapter4_load.png")
    plt.show()

def arima_forecast(df):
    """
    Forecast using ARIMA.
    """
    ts = df.set_index("timestamp")["Load_MW"]
    model = ARIMA(ts, order=(3, 1, 2))
    fit = model.fit()
    forecast = fit.forecast(steps=24*7)  # 1-week forecast

    plt.figure(figsize=(12, 4))
    plt.plot(ts[-24*7:], label="Observed", color="gray")
    plt.plot(forecast, label="ARIMA Forecast", color="black")
    plt.xlabel("Time")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("chapter4_arima.png")
    plt.show()

def prophet_forecast(df):
    """
    Forecast using Prophet.
    """
    df_prophet = df.rename(columns={"timestamp": "ds", "Load_MW": "y"})
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=24*7, freq="H")
    forecast = model.predict(future)

    fig = model.plot(forecast)
    plt.title("Prophet Forecast")
    plt.tight_layout()
    plt.savefig("chapter4_prophet.png")
    plt.show()

def lstm_forecast(df):
    """
    Forecast using LSTM (Darts).
    """
    series = TimeSeries.from_dataframe(df, "timestamp", "Load_MW")
    train, val = series[:-24*7], series[-24*7:]
    model = RNNModel(model="LSTM", input_chunk_length=48, output_chunk_length=24, n_epochs=50, random_state=42)
    model.fit(train)
    pred = model.predict(len(val))

    plt.figure(figsize=(12, 4))
    train[-24*7:].plot(label="Observed", lw=1, color="gray")
    pred.plot(label="LSTM Forecast", lw=2, color="black")
    plt.legend()
    plt.title(f"LSTM Forecast (RMSE: {rmse(val, pred):.2f})")
    plt.tight_layout()
    plt.savefig("chapter4_lstm.png")
    plt.show()

if __name__ == "__main__":
    df_load = generate_synthetic_load()
    plot_load(df_load)
    arima_forecast(df_load)
    prophet_forecast(df_load)
    lstm_forecast(df_load)
