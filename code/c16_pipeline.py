"""Chapter 16: Full Utility AI Platform Deployment."""

import logging
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
import os

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

np.random.seed(config["model"]["random_state"])


def train_model():
    """Train and save model."""
    samples = config["data"]["samples"]
    temp = np.random.normal(60, 5, samples)
    vibration = np.random.normal(0.2, 0.05, samples)
    oil_quality = np.random.normal(70, 10, samples)
    age = np.random.randint(1, 30, samples)
    failure_prob = 1 / (1 + np.exp(-(0.05*(temp-65) + 8*(vibration-0.25))))
    failure = np.random.binomial(1, failure_prob)

    X = pd.DataFrame({
        "Temperature": temp, "Vibration": vibration,
        "OilQuality": oil_quality, "Age": age
    })
    y = failure

    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        random_state=config["model"]["random_state"]
    ).fit(X, y)
    joblib.dump(model, config["paths"]["model_file"])
    logger.info("Model trained and saved")
    return model


# API Setup
app = FastAPI()

class TransformerInput(BaseModel):
    Temperature: float
    Vibration: float
    OilQuality: float
    Age: int

if os.path.exists(config["paths"]["model_file"]):
    model = joblib.load(config["paths"]["model_file"])
else:
    model = train_model()


@app.post("/predict")
def predict(data: TransformerInput):
    """Predict transformer failure risk."""
    X = np.array([[data.Temperature, data.Vibration, data.OilQuality, data.Age]])
    pred = model.predict(X)[0]
    return {"failure_risk": int(pred)}


@app.get("/health")
def health_check():
    """Health endpoint for Kubernetes and monitoring probes."""
    return {"status": "ok"}


if __name__ == "__main__":
    if os.environ.get("ML4U_CI") == "1":
        logger.info("ML4U_CI=1: not starting Uvicorn. "
                    "API is available when you run this script locally without ML4U_CI.")
    else:
        uvicorn.run(app, host=config["api"]["host"], port=config["api"]["port"])
