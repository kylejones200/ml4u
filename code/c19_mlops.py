"""Chapter 19: Integrated Orchestration of ML Pipelines."""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from prefect import flow, task
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from statsmodels.tsa.arima.model import ARIMA

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

np.random.seed(config["model"]["random_state"])


@task
def predictive_maintenance_task():
    """Predictive maintenance task."""
    samples = config["data"]["maintenance_samples"]
    temp = np.random.normal(60, 5, samples)
    vibration = np.random.normal(0.2, 0.05, samples)
    oil_quality = np.random.normal(70, 10, samples)
    age = np.random.randint(1, 30, samples)
    failure_prob = 1 / (1 + np.exp(-(0.05*(temp-65) + 8*(vibration-0.25))))
    failure = np.random.binomial(1, failure_prob)

    df = pd.DataFrame({
        "Temperature": temp, "Vibration": vibration,
        "OilQuality": oil_quality, "Age": age, "Failure": failure
    })
    X, y = df[["Temperature", "Vibration", "OilQuality", "Age"]], df["Failure"]
    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"]
    ).fit(X, y)
    preds = model.predict(X)
    logger.info(classification_report(y, preds))
    return df


@task
def load_forecasting_task():
    """Load forecasting task."""
    date_rng = pd.date_range(start="2023-01-01", periods=24*14, freq="H")
    load = (
        900 + 100 * np.sin(2 * np.pi * date_rng.hour / 24) +
        np.random.normal(0, 30, len(date_rng))
    )
    ts = pd.Series(load, index=date_rng)
    order = tuple(config["arima"]["order"])
    model = ARIMA(ts, order=order).fit()
    forecast = model.forecast(steps=config["arima"]["forecast_steps"])
    logger.debug(f"Forecast:\n{forecast.tail()}")
    return forecast


@task
def outage_prediction_task():
    """Outage prediction task."""
    samples = config["data"]["outage_samples"]
    wind = np.random.normal(20, 7, samples)
    trees = np.random.uniform(0, 1, samples)
    rainfall = np.random.normal(50, 15, samples)
    outage_prob = 1 / (1 + np.exp(-(0.15*(wind-25) + 2*(trees-0.5))))
    outage = np.random.binomial(1, outage_prob)
    df = pd.DataFrame({
        "WindSpeed": wind, "TreeDensity": trees,
        "Rainfall": rainfall, "Outage": outage
    })
    X, y = df[["WindSpeed", "TreeDensity", "Rainfall"]], df["Outage"]
    model = GradientBoostingClassifier().fit(X, y)
    preds = model.predict(X)
    logger.info(classification_report(y, preds))
    return df


@task
def cybersecurity_task():
    """Cybersecurity task."""
    logger.info("Cybersecurity task simulated")


@flow
def utility_ml_pipeline():
    """Orchestrated utility ML pipeline."""
    logger.info("Running utility ML pipeline")
    predictive_maintenance_task()
    load_forecasting_task()
    outage_prediction_task()
    cybersecurity_task()
    logger.info("Pipeline complete")


if __name__ == "__main__":
    utility_ml_pipeline()
