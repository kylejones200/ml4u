"""
Chapter 9: Customer Analytics and Demand Response
Load segmentation using smart meter data and clustering for DR targeting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans

def generate_smart_meter_data(customers=200, days=14):
    """
    Generate synthetic smart meter data (hourly kWh for multiple customers).
    """
    rng = pd.date_range("2022-01-01", periods=24 * days, freq="H")
    profiles = []
    for _ in range(customers):
        base = np.random.uniform(0.3, 1.0)  # Base consumption
        morning_peak = base + 0.5 * np.exp(-0.5 * (np.arange(24) - 7) ** 2)
        evening_peak = base + 0.8 * np.exp(-0.5 * (np.arange(24) - 19) ** 2)
        daily_profile = (morning_peak + evening_peak) / 2
        load = np.tile(daily_profile, days) + np.random.normal(0, 0.05, 24 * days)
        profiles.append(load)
    df = pd.DataFrame(profiles, columns=rng)
    return df

def cluster_load_profiles(df):
    """
    Cluster customer load profiles using KMeans on daily averages.
    """
    scaler = StandardScaler()
    daily_avg = df.values.reshape(df.shape[0], -1, 24).mean(axis=1)  # Average daily profile
    daily_scaled = scaler.fit_transform(daily_avg)

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(daily_scaled)

    plt.figure(figsize=(10, 6))
    for c in range(3):
        cluster_profiles = daily_avg[labels == c]
        plt.plot(cluster_profiles.T, color="gray", alpha=0.2)
        plt.plot(cluster_profiles.mean(axis=0), label=f"Cluster {c+1}", linewidth=2)
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Load (kWh)")
    plt.title("Customer Segmentation for Demand Response")
    plt.legend()
    plt.tight_layout()
    plt.savefig("chapter9_dr_clusters.png")
    plt.show()

    return labels

def identify_dr_targets(df, labels, target_cluster=2):
    """
    Identify customers in the highest-load cluster for DR targeting.
    """
    high_load_customers = np.where(labels == target_cluster)[0]
    print(f"Identified {len(high_load_customers)} customers for DR targeting.")
    return high_load_customers

if __name__ == "__main__":
    df_smart = generate_smart_meter_data()
    labels = cluster_load_profiles(df_smart)
    dr_targets = identify_dr_targets(df_smart, labels)
