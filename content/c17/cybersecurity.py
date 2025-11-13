"""Chapter 17: Cybersecurity Analytics for Critical Infrastructure."""

import os
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

np.random.seed(config["model"]["random_state"])


def load_cicids_sample():
    """Load a cleaned subset of the CICIDS2017 dataset."""
    file_path = config["data"]["file_path"]
    if not os.path.exists(file_path):
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        if os.path.isdir(data_dir):
            for name in os.listdir(data_dir):
                if name.lower().endswith(".csv"):
                    file_path = os.path.join("data", name)
                    break
    df = pd.read_csv(file_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns from {file_path}")
    return df


def preprocess(df):
    """Prepare data: select numeric features, scale them, and encode labels."""
    label_col = None
    for c in df.columns:
        if str(c).strip().lower() == "label":
            label_col = c
            break
    if label_col is None:
        raise ValueError("Could not find a 'Label' column in the dataset.")

    y_raw = df[label_col].astype(str).str.strip().str.upper()
    y = y_raw.apply(lambda x: 1 if x == "BENIGN" else 0)

    X = df.select_dtypes(include=[np.number]).copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.dropna(axis=1, how="all")
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if y.nunique() < 2:
        n = max(1, int(0.01 * len(y)))
        y.iloc[:n] = 1 - y.iloc[:n]
        print(f"Note: dataset contained a single class; flipped {n} samples for demo.")

    return X_scaled, y


def run_anomaly_detection(X, y):
    """Detect anomalies using Isolation Forest."""
    iso = IsolationForest(
        contamination=config["model"]["contamination"],
        random_state=config["model"]["random_state"]
    )
    preds = iso.fit_predict(X)
    preds_binary = np.where(preds == 1, 1, 0)
    accuracy = (preds_binary == y).mean()
    print(f"Isolation Forest Accuracy: {accuracy:.2f}")


def run_supervised_detection(X, y):
    """Train Random Forest classifier for intrusion detection."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["model"]["test_size"], 
        stratify=y, random_state=config["model"]["random_state"]
    )
    clf = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        random_state=config["model"]["random_state"]
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    print("Random Forest Classification Report:")
    print(classification_report(y_test, preds, target_names=["Attack", "Benign"]))
    print(f"ROC AUC Score: {roc_auc_score(y_test, probs):.3f}")


if __name__ == "__main__":
    df = load_cicids_sample()
    X, y = preprocess(df)
    run_anomaly_detection(X, y)
    run_supervised_detection(X, y)
