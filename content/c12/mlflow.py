"""
Chapter 12: MLOps for Utilities
Automatic MLflow model serving with FastAPI and Kafka for real-time transformer failure prediction.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from kafka import KafkaConsumer
import json
import threading

# ---------- MODEL TRAINING & REGISTRY ----------

def generate_asset_data(samples=500):
    """
    Synthetic transformer asset health dataset.
    """
    np.random.seed(42)
    temp = np.random.normal(60, 5, samples)
    vibration = np.random.normal(0.2, 0.05, samples)
    oil_quality = np.random.normal(70, 10, samples)
    age = np.random.randint(1, 30, samples)

    failure_prob = 1 / (1 + np.exp(-(0.05*(temp-65) + 8*(vibration-0.25))))
    failure = np.random.binomial(1, failure_prob)

    return pd.DataFrame({
        "Temperature_C": temp,
        "Vibration_g": vibration,
        "OilQuality_Index": oil_quality,
        "Age_Years": age,
        "Failure": failure
    })

def train_and_register_model(df, experiment_name="Utilities_MLOps", model_name="TransformerFailureModel"):
    """
    Train a model, log to MLflow, and register it to the MLflow model registry.
    """
    X = df[["Temperature_C", "Vibration_g", "OilQuality_Index", "Age_Years"]]
    y = df["Failure"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        print("Transformer Failure Prediction Report:")
        print(classification_report(y_test, preds))

        mlflow.sklearn.log_model(model, "rf_model", registered_model_name=model_name)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", (preds == y_test).mean())

    print(f"Model '{model_name}' registered in MLflow.")
    return model

def load_production_model(model_name="TransformerFailureModel"):
    """
    Load the latest production model from MLflow.
    """
    print(f"Loading latest production model '{model_name}' from MLflow registry...")
    return mlflow.sklearn.load_model(f"models:/{model_name}/Production")

# ---------- FASTAPI ENDPOINT ----------

app = FastAPI()
class TransformerData(BaseModel):
    Temperature_C: float
    Vibration_g: float
    OilQuality_Index: float
    Age_Years: int

model = None  # Will be set dynamically

@app.post("/predict")
def predict(data: TransformerData):
    """
    Predict transformer failure risk.
    """
    X = np.array([[data.Temperature_C, data.Vibration_g, data.OilQuality_Index, data.Age_Years]])
    pred = model.predict(X)[0]
    return {"failure_risk": int(pred)}

# ---------- KAFKA STREAMING ----------

def start_kafka_consumer(model, topic="scada_stream", bootstrap_servers="localhost:9092"):
    """
    Start Kafka consumer to ingest SCADA data and run live predictions.
    """
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    print(f"Kafka consumer connected to topic: {topic}")

    for msg in consumer:
        data = msg.value
        X = np.array([[data["Temperature_C"], data["Vibration_g"], data["OilQuality_Index"], data["Age_Years"]]])
        pred = model.predict(X)[0]
        print(f"[Kafka Prediction] Transformer={data.get('TransformerID', 'Unknown')} | Failure Risk={pred}")

# ---------- MAIN ENTRY POINT ----------

if __name__ == "__main__":
    # Train and register a model (only needed once)
    df = generate_asset_data()
    train_and_register_model(df)

    # Load the latest Production model from MLflow
    model = load_production_model()

    # Start Kafka consumer in a background thread
    threading.Thread(target=start_kafka_consumer, args=(model,), daemon=True).start()

    # Start FastAPI server for REST predictions
    print("Starting FastAPI at http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
