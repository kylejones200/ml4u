"""Chapter 9: Customer Analytics and Demand Response."""

import logging
import numpy as np
import pandas as pd
import signalplot as sp
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

sp.apply()

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
        load = (
            np.tile(daily_profile, config["data"]["days"]) +
            np.random.normal(0, 0.05, 24 * config["data"]["days"])
        )
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

    fig, ax = sp.figure()
    for c in range(config["model"]["n_clusters"]):
        cluster_profiles = daily_avg[labels == c]
        ax.plot(cluster_profiles.T, alpha=0.2)
        ax.plot(cluster_profiles.mean(axis=0))
    sp.style_line_plot(ax)
    sp.savefig(config["plotting"]["output_file"])

    return labels


def calculate_average_day_profile(
    df, customer_col='customer_id', value_col='Consumption_kWh'
):
    """
    Calculate average day profile for each customer.
    
    Computes the mean consumption for each hour of the day across all days
    in the dataset. This creates a representative 24-hour profile for each customer
    that captures typical daily patterns while averaging out day-to-day variation.
    
    Args:
        df: DataFrame with datetime index and customer consumption data
        customer_col: Column name for customer identifier (if multiple customers)
        value_col: Column name for consumption values
    
    Returns:
        DataFrame with 24 columns (hours) and one row per customer
    """
    # Extract hour of day from index
    df['hour'] = df.index.hour
    
    # If multiple customers, group by customer and hour
    if customer_col in df.columns:
        daily_profiles = (
            df.groupby([customer_col, 'hour'])[value_col]
            .mean()
            .unstack(level=1)
        )
    else:
        # Single customer case
        daily_profiles = df.groupby('hour')[value_col].mean().to_frame().T
    
    return daily_profiles


def segment_customers_from_profiles(profiles, n_clusters=3):
    """
    Segment customers using average day profiles with outlier elimination.
    
    First removes outlier customers (e.g., those with unusually high/low consumption)
    before clustering, which improves segmentation quality by focusing on typical
    consumption patterns rather than anomalous cases.
    
    Args:
        profiles: DataFrame with average day profiles (one row per customer, 24 cols)
        n_clusters: Number of clusters for KMeans
    
    Returns:
        labels: Cluster assignments for each customer
        kmeans: Fitted KMeans model
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.neighbors import LocalOutlierFactor
    
    # Remove outliers using Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=min(20, len(profiles) - 1), contamination=0.1)
    outlier_labels = lof.fit_predict(profiles.values)
    inlier_mask = outlier_labels == 1
    
    # Scale profiles (focus on shape, not magnitude)
    scaler = StandardScaler()
    profiles_scaled = scaler.fit_transform(profiles[inlier_mask].values)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=config["model"]["random_state"])
    labels_inlier = kmeans.fit_predict(profiles_scaled)
    
    # Map labels back to original customers (outliers get -1)
    labels_full = np.full(len(profiles), -1, dtype=int)
    labels_full[inlier_mask] = labels_inlier
    
    logger.info(f"Segmented {sum(inlier_mask)} customers, {sum(~inlier_mask)} outliers")
    
    return labels_full, kmeans, inlier_mask


def identify_dr_targets(df, labels, target_cluster=2):
    """Identify customers in the highest-load cluster for DR targeting."""
    # Filter out outliers (label == -1)
    valid_labels = labels[labels != -1]
    if valid_labels.size:
        high_load_customers = np.where(labels == target_cluster)[0]
        logger.info(f"DR targets: {len(high_load_customers)} customers")
        return high_load_customers
    else:
        logger.warning("No valid customer segments found")
        return np.array([])


if __name__ == "__main__":
    df_smart = generate_smart_meter_data()
    labels = cluster_load_profiles(df_smart)
    dr_targets = identify_dr_targets(df_smart, labels)
