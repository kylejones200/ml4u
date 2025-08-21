"""
Chapter 20: Full Utility AI Platform Deployment
Deploy ML models as APIs using FastAPI, Docker, and Kubernetes with monitoring.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
import os

# --- Model Training ---
def train_model():
    np.random.seed(42)
    temp = np.random.normal(60, 5, 500)
    vibration = np.random.normal(0.2, 0.05, 500)
    oil_quality = np.random.normal(70, 10, 500)
    age = np.random.randint(1, 30, 500)
    failure_prob = 1 / (1 + np.exp(-(0.05*(temp-65) + 8*(vibration-0.25))))
    failure = np.random.binomial(1, failure_prob)

    X = pd.DataFrame({"Temperature": temp, "Vibration": vibration, "OilQuality": oil_quality, "Age": age})
    y = failure

    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    joblib.dump(model, "transformer_model.pkl")
    print("Model trained and saved.")
    return model

# --- API Setup ---
app = FastAPI()
class TransformerInput(BaseModel):
    Temperature: float
    Vibration: float
    OilQuality: float
    Age: int

if os.path.exists("transformer_model.pkl"):
    model = joblib.load("transformer_model.pkl")
else:
    model = train_model()

@app.post("/predict")
def predict(data: TransformerInput):
    X = np.array([[data.Temperature, data.Vibration, data.OilQuality, data.Age]])
    pred = model.predict(X)[0]
    return {"failure_risk": int(pred)}

# --- Monitoring Integration ---
@app.get("/health")
def health_check():
    """
    Health endpoint for Kubernetes and monitoring probes.
    """
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
