"""Chapter 9: Customer Analytics and Demand Response."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

np.random.seed(config["model"]["random_state"])


def generate_smart_meter_data():
    """Generate synthetic smart meter data (hourly kWh for multiple customers)."""
    rng = pd.date_range("2022-01-01", periods=24 * config["data"]["days"], freq="h")
    profiles = []
    for _ in range(config["data"]["customers"]):
        base = np.random.uniform(0.3, 1.0)
        morning_peak = base + 0.5 * np.exp(-0.5 * (np.arange(24) - 7) ** 2)
        evening_peak = base + 0.8 * np.exp(-0.5 * (np.arange(24) - 19) ** 2)
        daily_profile = (morning_peak + evening_peak) / 2
        load = np.tile(daily_profile, config["data"]["days"]) + np.random.normal(0, 0.05, 24 * config["data"]["days"])
        profiles.append(load)
    return pd.DataFrame(profiles, columns=rng)


def cluster_load_profiles(df):
    """Cluster customer load profiles using KMeans on daily averages."""
    scaler = StandardScaler()
    daily_avg = df.values.reshape(df.shape[0], -1, 24).mean(axis=1)
    daily_scaled = scaler.fit_transform(daily_avg)

    kmeans = KMeans(n_clusters=config["model"]["n_clusters"], 
                   random_state=config["model"]["random_state"])
    labels = kmeans.fit_predict(daily_scaled)

    fig, ax = plt.subplots(figsize=config["plotting"]["figsize"])
    for c in range(config["model"]["n_clusters"]):
        cluster_profiles = daily_avg[labels == c]
        ax.plot(cluster_profiles.T, color=config["plotting"]["colors"]["cluster_lines"], alpha=0.2)
        ax.plot(cluster_profiles.mean(axis=0), label=f"Cluster {c+1}", linewidth=2)
    ax.set_title("Average Load (kWh) by Hour - Customer Segmentation for Demand Response")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.savefig(config["plotting"]["output_file"])
    plt.close()

    return labels


def identify_dr_targets(df, labels, target_cluster=2):
    """Identify customers in the highest-load cluster for DR targeting."""
    high_load_customers = np.where(labels == target_cluster)[0]
    print(f"Identified {len(high_load_customers)} customers for DR targeting.")
    return high_load_customers


if __name__ == "__main__":
    df_smart = generate_smart_meter_data()
    labels = cluster_load_profiles(df_smart)
    dr_targets = identify_dr_targets(df_smart, labels)
