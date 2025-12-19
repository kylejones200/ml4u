"""Chapter 14: MLOps for Utilities."""

import logging
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from kafka import KafkaConsumer
import json
import threading

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

np.random.seed(config["model"]["random_state"])


def generate_asset_data():
    """Synthetic transformer asset health dataset."""
    samples = config["data"]["samples"]
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


def train_and_register_model(df):
    """Train a model, log to MLflow, and register it."""
    X = df[["Temperature_C", "Vibration_g", "OilQuality_Index", "Age_Years"]]
    y = df["Failure"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["model"]["test_size"], 
        random_state=config["model"]["random_state"]
    )

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=config["model"]["n_estimators"],
            random_state=config["model"]["random_state"]
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        logger.info(classification_report(y_test, preds))

        mlflow.sklearn.log_model(model, "rf_model", 
                                 registered_model_name=config["mlflow"]["model_name"])
        mlflow.log_param("n_estimators", config["model"]["n_estimators"])
        mlflow.log_metric("accuracy", (preds == y_test).mean())

    logger.info(f"Model registered: {config['mlflow']['model_name']}")
    return model


def load_production_model():
    """Load the latest production model from MLflow."""
    model_name = config["mlflow"]["model_name"]
    logger.info(f"Loading model: {model_name}")
    return mlflow.sklearn.load_model(f"models:/{model_name}/Production")


# FastAPI endpoint
app = FastAPI()

class TransformerData(BaseModel):
    Temperature_C: float
    Vibration_g: float
    OilQuality_Index: float
    Age_Years: int

model = None


@app.post("/predict")
def predict(data: TransformerData):
    """Predict transformer failure risk."""
    X = np.array([[data.Temperature_C, data.Vibration_g, 
                  data.OilQuality_Index, data.Age_Years]])
    pred = model.predict(X)[0]
    return {"failure_risk": int(pred)}


def start_kafka_consumer(model):
    """Start Kafka consumer to ingest SCADA data and run live predictions."""
    consumer = KafkaConsumer(
        config["kafka"]["topic"],
        bootstrap_servers=config["kafka"]["bootstrap_servers"],
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    logger.info(f"Kafka connected: {config['kafka']['topic']}")

    for msg in consumer:
        data = msg.value
        X = np.array([[data["Temperature_C"], data["Vibration_g"], 
                      data["OilQuality_Index"], data["Age_Years"]]])
        pred = model.predict(X)[0]
        logger.debug(f"Kafka: Transformer={data.get('TransformerID', 'Unknown')}, Risk={pred}")


if __name__ == "__main__":
    df = generate_asset_data()
    trained_model = train_and_register_model(df)

    if os.environ.get("ML4U_CI") == "1":
        model = trained_model
        logger.info("ML4U_CI=1: using freshly trained model")
    else:
        model = load_production_model()

    if os.environ.get("ML4U_CI") == "1":
        logger.info("ML4U_CI=1: skipping Kafka and FastAPI")
    else:
        threading.Thread(
            target=start_kafka_consumer, args=(model,), daemon=True
        ).start()
        logger.info(f"FastAPI: http://127.0.0.1:{config['api']['port']}")
        uvicorn.run(app, host=config["api"]["host"], port=config["api"]["port"])
