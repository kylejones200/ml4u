"""Chapter 6: Outage Prediction and Reliability Analytics."""

import logging
import numpy as np
import pandas as pd
import signalplot as sp
import yaml
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.inspection import permutation_importance

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

sp.apply()

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

np.random.seed(config["model"]["random_state"])


def generate_storm_outage_data():
    """Simulate weather events and outages for overhead distribution lines."""
    samples = config["data"]["samples"]
    wind_speed = np.random.normal(config["weather"]["wind_mean"], 
                                  config["weather"]["wind_std"], samples)
    rainfall = np.random.normal(config["weather"]["rainfall_mean"], 
                               config["weather"]["rainfall_std"], samples)
    tree_density = np.random.uniform(config["weather"]["tree_density_min"], 
                                      config["weather"]["tree_density_max"], samples)
    asset_age = np.random.uniform(config["weather"]["asset_age_min"], 
                                  config["weather"]["asset_age_max"], samples)

    logit = (
        0.15 * (wind_speed - 25) +
        0.03 * (rainfall - 60) +
        2 * (tree_density - 0.5)
    )
    outage_prob = 1 / (1 + np.exp(-logit))
    outages = np.random.binomial(1, outage_prob)

    return pd.DataFrame({
        "WindSpeed_mps": wind_speed,
        "Rainfall_mm": rainfall,
        "TreeDensity": tree_density,
        "AssetAge_years": asset_age,
        "Outage": outages
    })


def train_outage_model(df):
    """Train Gradient Boosting classifier for outage prediction."""
    X = df[["WindSpeed_mps", "Rainfall_mm", "TreeDensity", "AssetAge_years"]]
    y = df["Outage"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["model"]["test_size"], 
        random_state=config["model"]["random_state"], stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=config["model"]["n_estimators"],
        learning_rate=config["model"]["learning_rate"],
        max_depth=config["model"]["max_depth"],
        random_state=config["model"]["random_state"]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    logger.info(classification_report(y_test, y_pred, target_names=["No Outage", "Outage"]))
    logger.info(f"ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")

    # Feature importance
    result = permutation_importance(model, X_test, y_test, n_repeats=10, 
                                    random_state=config["model"]["random_state"])
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": result.importances_mean
    }).sort_values("Importance", ascending=True)

    fig, ax = sp.figure()
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    sp.style_bar_plot(ax, horizontal=True)
    sp.savefig(config["plotting"]["output_file"])


if __name__ == "__main__":
    df_outage = generate_storm_outage_data()
    train_outage_model(df_outage)
