"""
Chapter 1: Introduction to Machine Learning in Power and Utilities
Basic demonstration using synthetic load data and linear regression.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def generate_synthetic_load(days=365):
    """
    Generate synthetic daily load data with seasonal and random components.

    Args:
        days (int): Number of days to simulate.

    Returns:
        pd.DataFrame: Date and load values.
    """
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    base_load = 1000 + 200 * np.sin(2 * np.pi * dates.dayofyear / 365)
    noise = np.random.normal(0, 50, size=days)
    load = base_load + noise
    return pd.DataFrame({"Date": dates, "Load_MW": load})

def plot_load(df):
    """
    Plot synthetic load data.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(df["Date"], df["Load_MW"], color="black")
    plt.xlabel("Date")
    plt.ylabel("Load (MW)")
    plt.title("Synthetic Daily Load")
    plt.tight_layout()
    plt.savefig("chapter1_load_plot.png")
    plt.show()

def train_simple_regression(df):
    """
    Train a linear regression model to predict load trend.

    Args:
        df (pd.DataFrame): Load data.

    Returns:
        LinearRegression: Trained model.
    """
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Load_MW"].values
    model = LinearRegression()
    model.fit(X, y)
    trend = model.predict(X)

    plt.figure(figsize=(12, 4))
    plt.plot(df["Date"], y, label="Observed", color="gray")
    plt.plot(df["Date"], trend, label="Trend (Linear Regression)", color="black")
    plt.xlabel("Date")
    plt.ylabel("Load (MW)")
    plt.title("Load Trend Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig("chapter1_regression_trend.png")
    plt.show()

    print(f"Regression slope: {model.coef_[0]:.4f} MW/day")
    return model

if __name__ == "__main__":
    df_load = generate_synthetic_load()
    plot_load(df_load)
    train_simple_regression(df_load)
