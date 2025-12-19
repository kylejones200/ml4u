"""Chapter 1: Introduction to Machine Learning in Power and Utilities.

This script demonstrates temperature-to-load modeling using linear regression.
It generates synthetic temperature and load data, then trains a model to predict
load from temperature--a fundamental relationship in utility operations.
"""

import logging
import numpy as np
import pandas as pd
import signalplot as sp
import yaml
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

sp.apply()

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

np.random.seed(config["model"]["random_state"])


def generate_synthetic_data():
    """Generate synthetic temperature and load data with realistic relationships.
    
    Creates data where load increases with temperature (cooling demand) and
    also increases at very low temperatures (heating demand), forming a
    U-shaped relationship that utilities commonly observe.
    """
    dates = pd.date_range(
        config["data"]["start_date"],
        periods=config["data"]["days"],
        freq="D"
    )
    
    # Generate temperature with seasonal variation
    base_temp = config["data"]["base_temp"]
    temp_amplitude = config["data"]["temp_amplitude"]
    temperature = (base_temp + 
                   temp_amplitude * np.sin(2 * np.pi * dates.dayofyear / 365) +
                   np.random.normal(0, config["data"]["temp_noise_std"], len(dates)))
    
    # Generate load that depends on temperature
    # Higher temps -> more cooling -> higher load
    # Very low temps -> more heating -> higher load
    base_load = config["data"]["base_load"]
    temp_coef = config["data"]["temp_coef"]
    load = (base_load + 
            temp_coef * temperature +
            config["data"]["temp_coef_squared"] * (temperature - base_temp) ** 2 +
            np.random.normal(0, config["data"]["noise_std"], len(dates)))
    
    return pd.DataFrame({
        "Date": dates,
        "Temperature_C": temperature,
        "Load_MW": load
    })


def plot_temperature_vs_load(df):
    """Plot temperature vs. load relationship."""
    fig, ax = sp.figure()
    ax.scatter(df["Temperature_C"], df["Load_MW"])
    sp.style_scatter_plot(ax)
    sp.savefig(config["plotting"]["output_files"]["load_plot"])


def train_temperature_to_load_model(df):
    """Train a linear regression model to predict load from temperature.
    
    This function:
    1. Prepares temperature as the feature (X) and load as the target (y)
    2. Trains a linear regression model
    3. Makes predictions and evaluates model performance
    4. Visualizes predictions vs. actual values
    """
    X = df[["Temperature_C"]].values
    y = df["Load_MW"].values
    
    # Train the model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    logger.info(f"R²={r2:.3f}, MSE={mse:.2f}, Coef={model.coef_[0]:.2f}")
    
    # Plot predictions vs. actual
    fig, ax = sp.figure()
    ax.scatter(df["Temperature_C"], y)
    ax.plot(df["Temperature_C"], y_pred)
    sp.style_line_plot(ax)
    sp.savefig(config["plotting"]["output_files"]["regression_trend"])
    
    return model


if __name__ == "__main__":
    df = generate_synthetic_data()
    plot_temperature_vs_load(df)
    model = train_temperature_to_load_model(df)
