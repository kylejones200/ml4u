"""
Chapter 6: Outage Prediction and Reliability Analytics
Uses weather and asset exposure data to predict storm-driven outages.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.inspection import permutation_importance

def generate_storm_outage_data(samples=1500):
    """
    Simulate weather events and outages for overhead distribution lines.
    Features: wind speed, rainfall, tree density, asset age.
    """
    np.random.seed(42)
    wind_speed = np.random.normal(20, 8, samples)   # m/s
    rainfall = np.random.normal(50, 20, samples)    # mm
    tree_density = np.random.uniform(0, 1, samples) # fraction canopy
    asset_age = np.random.uniform(1, 40, samples)   # years

    outage_prob = 1 / (1 + np.exp(-(0.15*(wind_speed-25) + 0.03*(rainfall-60) + 2*(tree_density-0.5))))
    outages = np.random.binomial(1, outage_prob)

    return pd.DataFrame({
        "WindSpeed_mps": wind_speed,
        "Rainfall_mm": rainfall,
        "TreeDensity": tree_density,
        "AssetAge_years": asset_age,
        "Outage": outages
    })

def train_outage_model(df):
    """
    Train Gradient Boosting classifier for outage prediction.
    """
    X = df[["WindSpeed_mps", "Rainfall_mm", "TreeDensity", "AssetAge_years"]]
    y = df["Outage"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Outage Prediction Report:")
    print(classification_report(y_test, y_pred, target_names=["No Outage", "Outage"]))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.3f}")

    # Feature importance
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": result.importances_mean
    }).sort_values("Importance", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(importance_df["Feature"], importance_df["Importance"], color="black")
    plt.xlabel("Permutation Importance")
    plt.title("Weather & Asset Features Driving Outages")
    plt.tight_layout()
    plt.savefig("chapter6_feature_importance.png")
    plt.show()

if __name__ == "__main__":
    df_outage = generate_storm_outage_data()
    train_outage_model(df_outage)
