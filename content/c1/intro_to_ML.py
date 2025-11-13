"""Chapter 1: Introduction to Machine Learning in Power and Utilities.

This script demonstrates temperature-to-load modeling using linear regression.
It generates synthetic temperature and load data, then trains a model to predict
load from temperature—a fundamental relationship in utility operations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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
    """Plot temperature vs. load to visualize the relationship.
    
    This scatter plot shows how load varies with temperature, with the
    regression line overlaid to show the model's learned relationship.
    """
    fig, ax = plt.subplots(figsize=config["plotting"]["figsize"])
    ax.scatter(df["Temperature_C"], df["Load_MW"], 
                alpha=0.5, color=config["plotting"]["colors"]["observed"],
                label="Observed Data")
    ax.set_title("Load (MW) vs Temperature (°C)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.savefig(config["plotting"]["output_files"]["load_plot"])
    plt.close()
    print(f"Saved temperature vs. load plot to {config['plotting']['output_files']['load_plot']}")


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
    
    print(f"\nModel Performance:")
    print(f"  Coefficient (MW per °C): {model.coef_[0]:.2f}")
    print(f"  Intercept (MW): {model.intercept_:.2f}")
    print(f"  R² Score: {r2:.3f}")
    print(f"  Mean Squared Error: {mse:.2f}")
    
    # Plot predictions vs. actual
    fig, ax = plt.subplots(figsize=config["plotting"]["figsize"])
    ax.scatter(df["Temperature_C"], y, 
                alpha=0.5, color=config["plotting"]["colors"]["observed"],
                label="Observed Load")
    ax.plot(df["Temperature_C"], y_pred, 
             color=config["plotting"]["colors"]["trend"], 
             linewidth=2, label="Predicted Load (Linear Regression)")
    ax.set_title(f"Load (MW) vs Temperature (°C) - Linear Regression Model (R² = {r2:.3f})")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.savefig(config["plotting"]["output_files"]["regression_trend"])
    plt.close()
    print(f"Saved regression plot to {config['plotting']['output_files']['regression_trend']}")
    
    return model


if __name__ == "__main__":
    print("Generating synthetic temperature and load data...")
    df = generate_synthetic_data()
    print(f"Generated {len(df)} days of data")
    
    print("\nVisualizing temperature vs. load relationship...")
    plot_temperature_vs_load(df)
    
    print("\nTraining temperature-to-load regression model...")
    model = train_temperature_to_load_model(df)
    
    print("\nModel training complete!")
