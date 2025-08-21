"""
Chapter 14: Case Studies and Implementation Roadmaps
End-to-end multi-use case pipelines integrating predictive maintenance, load forecasting, and outage prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Predictive Maintenance ---
def generate_asset_data(samples=500):
    np.random.seed(42)
    temp = np.random.normal(60, 5, samples)
    vibration = np.random.normal(0.2, 0.05, samples)
    oil_quality = np.random.normal(70, 10, samples)
    age = np.random.randint(1, 30, samples)
    failure_prob = 1 / (1 + np.exp(-(0.05*(temp-65) + 8*(vibration-0.25))))
    failure = np.random.binomial(1, failure_prob)
    return pd.DataFrame({"Temperature": temp, "Vibration": vibration, "OilQuality": oil_quality, "Age": age, "Failure": failure})

def predictive_maintenance_pipeline():
    df = generate_asset_data()
    X = df[["Temperature", "Vibration", "OilQuality", "Age"]]
    y = df["Failure"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Predictive Maintenance Report:")
    print(classification_report(y_test, preds))

# --- Load Forecasting ---
def generate_load_data():
    date_rng = pd.date_range(start="2023-01-01", periods=24*30, freq="H")
    load = 900 + 100 * np.sin(2 * np.pi * date_rng.hour / 24) + np.random.normal(0, 30, len(date_rng))
    return pd.DataFrame({"timestamp": date_rng, "Load_MW": load})

def load_forecasting_pipeline():
    df = generate_load_data()
    ts = df.set_index("timestamp")["Load_MW"]
    model = ARIMA(ts, order=(2, 1, 2))
    fit = model.fit()
    forecast = fit.forecast(steps=24)
    plt.figure(figsize=(10, 4))
    plt.plot(ts[-72:], label="Observed", color="gray")
    plt.plot(forecast.index, forecast, label="Forecast", color="black")
    plt.legend()
    plt.title("Load Forecast (ARIMA)")
    plt.tight_layout()
    plt.savefig("chapter14_load_forecast.png")
    plt.show()

# --- Outage Prediction ---
def generate_outage_data(samples=1000):
    wind = np.random.normal(20, 7, samples)
    trees = np.random.uniform(0, 1, samples)
    rainfall = np.random.normal(50, 15, samples)
    outage_prob = 1 / (1 + np.exp(-(0.15*(wind-25) + 2*(trees-0.5))))
    outage = np.random.binomial(1, outage_prob)
    return pd.DataFrame({"WindSpeed": wind, "TreeDensity": trees, "Rainfall": rainfall, "Outage": outage})

def outage_prediction_pipeline():
    df = generate_outage_data()
    X = df[["WindSpeed", "TreeDensity", "Rainfall"]]
    y = df["Outage"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Outage Prediction Report:")
    print(classification_report(y_test, preds))

# --- Integrated Pipeline Execution ---
if __name__ == "__main__":
    print("\n--- Predictive Maintenance ---")
    predictive_maintenance_pipeline()
    print("\n--- Load Forecasting ---")
    load_forecasting_pipeline()
    print("\n--- Outage Prediction ---")
    outage_prediction_pipeline()
