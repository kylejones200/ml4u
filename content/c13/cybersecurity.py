"""
Chapter 13: Cybersecurity Analytics for Critical Infrastructure
Intrusion detection using anomaly detection (Isolation Forest) and supervised ML (Random Forest).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def load_cicids_sample(file_path="data/CICIDS2017_sample.csv"):
    """
    Load a cleaned subset of the CICIDS2017 dataset.
    """
    df = pd.read_csv(file_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess(df):
    """
    Prepare data: scale features and encode labels (BENIGN=1, Attack=0).
    """
    X = df.drop(columns=["Label"])
    y = df["Label"].apply(lambda x: 1 if x == "BENIGN" else 0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def run_anomaly_detection(X, y):
    """
    Detect anomalies using Isolation Forest.
    """
    iso = IsolationForest(contamination=0.1, random_state=42)
    preds = iso.fit_predict(X)
    preds_binary = np.where(preds == 1, 1, 0)
    accuracy = (preds_binary == y).mean()
    print(f"Isolation Forest Accuracy: {accuracy:.2f}")

def run_supervised_detection(X, y):
    """
    Train Random Forest classifier for intrusion detection.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
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
