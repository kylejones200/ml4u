"""
Chapter 16: Integrated Orchestration of ML Pipelines
Pipeline scheduling and orchestration using Prefect for utilities.
"""

import pandas as pd
import numpy as np
from prefect import flow, task
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from statsmodels.tsa.arima.model import ARIMA

# --- TASKS ---

@task
def predictive_maintenance_task(samples=500):
    np.random.seed(42)
    temp = np.random.normal(60, 5, samples)
    vibration = np.random.normal(0.2, 0.05, samples)
    oil_quality = np.random.normal(70, 10, samples)
    age = np.random.randint(1, 30, samples)
    failure_prob = 1 / (1 + np.exp(-(0.05*(temp-65) + 8*(vibration-0.25))))
    failure = np.random.binomial(1, failure_prob)

    df = pd.DataFrame({"Temperature": temp, "Vibration": vibration, "OilQuality": oil_quality, "Age": age, "Failure": failure})
    X, y = df[["Temperature", "Vibration", "OilQuality", "Age"]], df["Failure"]
    model = RandomForestClassifier(n_estimators=100).fit(X, y)
    preds = model.predict(X)
    print("Predictive Maintenance Report:")
    print(classification_report(y, preds))
    return df

@task
def load_forecasting_task():
    date_rng = pd.date_range(start="2023-01-01", periods=24*14, freq="H")
    load = 900 + 100*np.sin(2*np.pi*date_rng.hour/24) + np.random.normal(0, 30, len(date_rng))
    ts = pd.Series(load, index=date_rng)
    model = ARIMA(ts, order=(2, 1, 2)).fit()
    forecast = model.forecast(steps=24)
    print("\nLoad Forecast (Next 24 hours):")
    print(forecast.tail())
    return forecast

@task
def outage_prediction_task(samples=1000):
    wind = np.random.normal(20, 7, samples)
    trees = np.random.uniform(0, 1, samples)
    rainfall = np.random.normal(50, 15, samples)
    outage_prob = 1 / (1 + np.exp(-(0.15*(wind-25) + 2*(trees-0.5))))
    outage = np.random.binomial(1, outage_prob)
    df = pd.DataFrame({"WindSpeed": wind, "TreeDensity": trees, "Rainfall": rainfall, "Outage": outage})
    X, y = df[["WindSpeed", "TreeDensity", "Rainfall"]], df["Outage"]
    model = GradientBoostingClassifier().fit(X, y)
    preds = model.predict(X)
    print("\nOutage Prediction Report:")
    print(classification_report(y, preds))
    return df

@task
def cybersecurity_task():
    print("\nCybersecurity task simulated (CICIDS2017 streaming detection placeholder).")
    # In production, link to Chapter 13’s intrusion detection pipeline here.

# --- ORCHESTRATED FLOW ---
@flow
def utility_ml_pipeline():
    print("\n--- Running Utility ML Pipeline ---")
    predictive_maintenance_task()
    load_forecasting_task()
    outage_prediction_task()
    cybersecurity_task()
    print("\n--- Pipeline Run Complete ---")

if __name__ == "__main__":
    utility_ml_pipeline()
