"""Chapter 3: Machine Learning Fundamentals for Power and Utilities."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, classification_report

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

np.random.seed(config["model"]["random_state"])


def generate_regression_data():
    """Generate synthetic regression data: temperature vs. daily load (MW)."""
    samples = config["data"]["regression_samples"]
    temp = np.random.normal(config["regression"]["temp_mean"], 
                           config["regression"]["temp_std"], samples)
    load = (config["regression"]["base_load"] + 
            config["regression"]["temp_coef"] * temp + 
            np.random.normal(0, config["regression"]["noise_std"], samples))
    return pd.DataFrame({"Temperature_C": temp, "Load_MW": load})


def regression_example(df):
    """Fit and visualize linear regression (Temperature -> Load)."""
    X = df[["Temperature_C"]]
    y = df["Load_MW"]
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)

    fig, ax = plt.subplots(figsize=config["plotting"]["figsize_regression"])
    ax.scatter(X, y, color=config["plotting"]["colors"]["observed"], 
                alpha=0.7, label="Observed")
    ax.plot(X, y_pred, color=config["plotting"]["colors"]["regression"], 
             label="Regression Line")
    ax.set_title(f"Load (MW) vs Temperature (°C) - Linear Regression (MSE = {mse:.2f})")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.savefig(config["plotting"]["output_files"]["regression"])
    plt.close()


def generate_classification_data():
    """Generate synthetic classification data: equipment age, load -> failure probability."""
    samples = config["data"]["classification_samples"]
    age = np.random.uniform(config["classification"]["age_min"], 
                           config["classification"]["age_max"], samples)
    load = np.random.uniform(config["classification"]["load_min"], 
                            config["classification"]["load_max"], samples)
    failure_prob = 1 / (1 + np.exp(-(0.1 * age + 0.002 * load - 7)))
    failure = np.random.binomial(1, failure_prob)
    return pd.DataFrame({"Age_Years": age, "Load_kVA": load, "Failure": failure})


def classification_example(df):
    """Train and evaluate logistic regression for equipment failure prediction."""
    X = df[["Age_Years", "Load_kVA"]]
    y = df["Failure"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["model"]["test_size"], 
        random_state=config["model"]["random_state"], stratify=y
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Failure"]))


def clustering_example():
    """Apply clustering to synthetic smart meter load profiles (daily kWh)."""
    np.random.seed(config["model"]["random_state"])
    means = config["clustering"]["cluster_means"]
    stds = config["clustering"]["cluster_stds"]
    n_samples = config["clustering"]["samples_per_cluster"]
    
    clusters = [np.random.normal(m, s, (n_samples, 24)) for m, s in zip(means, stds)]
    data = np.vstack(clusters)
    
    kmeans = KMeans(n_clusters=config["model"]["n_clusters"], 
                    random_state=config["model"]["random_state"])
    labels = kmeans.fit_predict(data)

    fig, ax = plt.subplots(figsize=config["plotting"]["figsize_clustering"])
    for i in range(config["model"]["n_clusters"]):
        cluster_profiles = data[labels == i]
        ax.plot(cluster_profiles.T, color="gray", alpha=0.2)
        ax.plot(cluster_profiles.mean(axis=0), label=f"Cluster {i+1}", linewidth=2)
    ax.set_title("Daily Load Profiles (kWh) by Hour - Customer Segmentation")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.savefig(config["plotting"]["output_files"]["clustering"])
    plt.close()


if __name__ == "__main__":
    # Regression
    df_reg = generate_regression_data()
    regression_example(df_reg)

    # Classification
    df_class = generate_classification_data()
    retries = 0
    while df_class["Failure"].nunique() < 2 and retries < 5:
        df_class = generate_classification_data()
        retries += 1
    classification_example(df_class)

    # Clustering
    clustering_example()
