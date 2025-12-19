"""Chapter 18: AI Ethics, Regulation, and the Future of Utilities."""

import logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from fairlearn.metrics import (
    MetricFrame, selection_rate, false_negative_rate, false_positive_rate
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

np.random.seed(config["model"]["random_state"])


def generate_asset_data():
    """Synthetic transformer dataset with sensitive attribute (region)."""
    samples = config["data"]["samples"]
    temp = np.random.normal(60, 5, samples)
    vibration = np.random.normal(0.2, 0.05, samples)
    oil_quality = np.random.normal(70, 10, samples)
    age = np.random.randint(1, 30, samples)
    region = np.random.choice(
        ["Urban", "Rural"], 
        size=samples, 
        p=[config["regions"]["urban_prob"], config["regions"]["rural_prob"]]
    )

    failure_prob = 1 / (1 + np.exp(-(0.05*(temp-65) + 8*(vibration-0.25))))
    failure = np.random.binomial(1, failure_prob)

    return pd.DataFrame({
        "Temperature": temp, "Vibration": vibration,
        "OilQuality": oil_quality, "Age": age,
        "Region": region, "Failure": failure
    })


def train_model(df):
    """Train Random Forest for failure prediction."""
    X = df[["Temperature", "Vibration", "OilQuality", "Age"]]
    y = df["Failure"]
    X_train, X_test, y_train, y_test, region_train, region_test = train_test_split(
        X, y, df["Region"], test_size=config["model"]["test_size"], 
        stratify=y, random_state=config["model"]["random_state"]
    )
    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        random_state=config["model"]["random_state"]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    logger.info(classification_report(y_test, preds))
    return model, X_test, y_test, region_test


def audit_fairness(model, X_test, y_test, sensitive_feature):
    """Compute fairness metrics by region."""
    preds = model.predict(X_test)
    metric_frame = MetricFrame(
        metrics={
            "Selection Rate": selection_rate,
            "False Negative Rate": false_negative_rate,
            "False Positive Rate": false_positive_rate
        },
        y_true=y_test,
        y_pred=preds,
        sensitive_features=sensitive_feature
    )
    logger.info(f"Fairness audit:\n{metric_frame.by_group}")


if __name__ == "__main__":
    df = generate_asset_data()
    model, X_test, y_test, region_test = train_model(df)
    audit_fairness(model, X_test, y_test, region_test)
