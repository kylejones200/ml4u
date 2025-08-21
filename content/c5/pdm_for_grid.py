"""
Chapter 5: Predictive Maintenance for Grid and Plant Assets
Uses synthetic SCADA sensor data to detect anomalies and predict equipment failures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

def generate_synthetic_scada_data(samples=2000):
    """
    Generate synthetic SCADA data for transformer monitoring.
    Features: temperature, vibration, oil pressure, load.
    """
    np.random.seed(42)
    temp = np.random.normal(60, 5, samples)
    vibration = np.random.normal(0.2, 0.05, samples)
    oil_pressure = np.random.normal(25, 3, samples)
    load = np.random.normal(800, 100, samples)

    # Simulate failures: elevated temp/vibration correlated with failures
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
    """
    Plot sample SCADA signals over time.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(df.index[:200], df["Temperature_C"][:200], label="Temperature (C)", color="black")
    plt.plot(df.index[:200], df["Vibration_g"][:200], label="Vibration (g)", color="gray")
    plt.xlabel("Time (Sample Index)")
    plt.ylabel("Sensor Readings")
    plt.title("Transformer Sensor Trends (Sample Window)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("chapter5_sensor_trends.png")
    plt.show()

def anomaly_detection(df):
    """
    Apply Isolation Forest for anomaly detection.
    """
    features = df[["Temperature_C", "Vibration_g", "OilPressure_psi", "Load_kVA"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    model = IsolationForest(contamination=0.05, random_state=42)
    preds = model.fit_predict(X_scaled)
    anomalies = np.where(preds == -1)[0]

    plt.figure(figsize=(10, 4))
    plt.scatter(df.index, df["Temperature_C"], c="black", label="Normal", alpha=0.5)
    plt.scatter(df.index[anomalies], df["Temperature_C"].iloc[anomalies], c="red", label="Anomaly")
    plt.xlabel("Sample")
    plt.ylabel("Temperature (C)")
    plt.title("Anomaly Detection in Transformer Sensor Data")
    plt.legend()
    plt.tight_layout()
    plt.savefig("chapter5_anomaly.png")
    plt.show()

def failure_prediction(df):
    """
    Train Random Forest to classify failure risk.
    """
    X = df[["Temperature_C", "Vibration_g", "OilPressure_psi", "Load_kVA"]]
    y = df["Failure"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Failure Prediction Report:")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Failure"]))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.3f}")

if __name__ == "__main__":
    df_scada = generate_synthetic_scada_data()
    plot_sensor_trends(df_scada)
    anomaly_detection(df_scada)
    failure_prediction(df_scada)
