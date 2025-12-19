"""Chapter 5: Predictive Maintenance for Grid Assets."""

import logging
import numpy as np
import pandas as pd
import signalplot as sp
import yaml
from pathlib import Path
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

sp.apply()

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

np.random.seed(config["model"]["random_state"])


def generate_synthetic_scada_data():
    """Generate synthetic SCADA data for transformer monitoring."""
    samples = config["data"]["samples"]
    temp = np.random.normal(config["sensor"]["temp_mean"], 
                           config["sensor"]["temp_std"], samples)
    vibration = np.random.normal(config["sensor"]["vibration_mean"], 
                                 config["sensor"]["vibration_std"], samples)
    oil_pressure = np.random.normal(config["sensor"]["oil_pressure_mean"], 
                                    config["sensor"]["oil_pressure_std"], samples)
    load = np.random.normal(config["sensor"]["load_mean"], 
                           config["sensor"]["load_std"], samples)

    failure_prob = 1 / (1 + np.exp(-(0.05*(temp-65) + 8*(vibration-0.25))))
    failures = np.random.binomial(1, failure_prob)

    return pd.DataFrame({
        "Temperature_C": temp,
        "Vibration_g": vibration,
        "OilPressure_psi": oil_pressure,
        "Load_kVA": load,
        "Failure": failures
    })


def plot_sensor_trends(df):
    """Plot sample SCADA signals over time."""
    window = config["plotting"]["window_size"]
    fig, ax = sp.figure()
    ax.plot(df.index[:window], df["Temperature_C"][:window])
    ax.plot(df.index[:window], df["Vibration_g"][:window])
    sp.style_line_plot(ax)
    sp.savefig(config["plotting"]["output_files"]["trends"])


def anomaly_detection(df):
    """Apply Isolation Forest for anomaly detection."""
    features = df[["Temperature_C", "Vibration_g", "OilPressure_psi", "Load_kVA"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    model = IsolationForest(
        contamination=config["model"]["contamination"],
        random_state=config["model"]["random_state"]
    )
    preds = model.fit_predict(X_scaled)
    anomalies = np.where(preds == -1)[0]

    fig, ax = sp.figure()
    ax.scatter(df.index, df["Temperature_C"], alpha=0.5)
    ax.scatter(df.index[anomalies], df["Temperature_C"].iloc[anomalies])
    sp.style_scatter_plot(ax)
    sp.savefig(config["plotting"]["output_files"]["anomaly"])


def failure_prediction(df):
    """Train Random Forest to classify failure risk."""
    X = df[["Temperature_C", "Vibration_g", "OilPressure_psi", "Load_kVA"]]
    y = df["Failure"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["model"]["test_size"], 
        random_state=config["model"]["random_state"], stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        random_state=config["model"]["random_state"]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    logger.info(classification_report(y_test, y_pred, target_names=["Healthy", "Failure"]))
    logger.info(f"ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")


if __name__ == "__main__":
    df_scada = generate_synthetic_scada_data()
    plot_sensor_trends(df_scada)
    anomaly_detection(df_scada)
    failure_prediction(df_scada)
