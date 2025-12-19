"""Chapter 24: Feature Engineering for Utility Data."""

import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path

config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

np.random.seed(config["model"]["random_state"])


def create_weather_features(temp_df):
    """Create weather features from temperature data."""
    df = temp_df.copy()
    base_temp = config["weather"]["base_temp_f"]
    
    df['cdd'] = np.maximum(0, df['temperature'] - base_temp)
    df['hdd'] = np.maximum(0, base_temp - df['temperature'])
    
    df['wind_chill'] = (
        35.74 + 0.6215*df['temperature'] - 35.75*(df['wind_speed']**0.16) 
        + 0.4275*df['temperature']*(df['wind_speed']**0.16)
        if 'wind_speed' in df.columns
        else df['temperature']
    )
    
    for lag in config["weather"]["lag_days"]:
        df[f'temp_lag_{lag}'] = df['temperature'].shift(lag)
    
    for window in config["weather"]["rolling_windows"]:
        df[f'temp_{window}day_avg'] = (
            df['temperature'].rolling(window=window, min_periods=1).mean()
        )
    
    return df


def create_temporal_features(df, date_col='timestamp'):
    """Create temporal features from timestamps."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    df['hour'] = df[date_col].dt.hour
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['month'] = df[date_col].dt.month
    df['is_weekend'] = df['day_of_week'].isin(config["temporal"]["weekend_days"])
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    df['is_holiday'] = False
    for holiday in config["temporal"]["holidays"]:
        mask = (
            (df[date_col].dt.month == holiday['month']) &
            (df[date_col].dt.day == holiday['day'])
        )
        df.loc[mask, 'is_holiday'] = True
    
    return df


def create_geospatial_features(asset_df):
    """Create geospatial features from asset coordinates."""
    df = asset_df.copy()
    geo = config["geospatial"]
    km_per_degree = geo["km_per_degree"]
    
    df['distance_to_coast_km'] = np.minimum(
        np.abs(df['longitude'] - geo["east_coast_lon"]) * km_per_degree,
        np.abs(df['longitude'] - geo["west_coast_lon"]) * km_per_degree
    )
    
    df['elevation_m'] = (
        np.maximum(0,
            (df['latitude'] - geo["elevation_base_lat"]) *
            geo["elevation_factor"] +
            np.random.normal(0, 50, len(df))
        )
        if 'elevation' not in df.columns
        else df['elevation']
    )
    
    climate_zones = {
        (float('-inf'), 30): 'Tropical',
        (30, 35): 'Subtropical',
        (35, 45): 'Temperate',
        (45, float('inf')): 'Cold'
    }
    
    def assign_climate_zone(lat):
        return next(
            zone for (min_lat, max_lat), zone in climate_zones.items()
            if min_lat <= lat < max_lat
        )
    
    df['climate_zone'] = df['latitude'].apply(assign_climate_zone)
    df['is_urban'] = (
        (df['latitude'].between(35, 45)) &
        (df['longitude'].between(-80, -70))
    )
    
    return df


def create_topology_features(asset_df, scada_df=None):
    """Create grid topology features."""
    df = asset_df.copy()
    
    df['asset_age_years'] = (
        (datetime.now() - pd.to_datetime(df['installation_date'])).dt.days / 365.25
        if 'installation_date' in df.columns
        else np.random.uniform(5, 40, len(df))
    )
    
    if scada_df is not None:
        current_loading = (
            scada_df.groupby('asset_id')['current_amps']
            .mean()
            .reset_index()
        )
        current_loading.columns = ['asset_id', 'current_amps']
        df = df.merge(current_loading, on='asset_id', how='left')
        rated_current = (df['rated_current_amps']
                        if 'rated_current_amps' in df.columns
                        else 200)
        df['loading_pct'] = (df['current_amps'] / rated_current) * 100
    else:
        df['loading_pct'] = np.random.uniform(30, 85, len(df))
    
    voltage_cats = config["topology"]["voltage_categories"]
    def voltage_category(kv):
        return (
            'Distribution' if kv < voltage_cats["distribution_max"]
            else 'Sub-Transmission' if kv < voltage_cats["subtransmission_max"]
            else 'Transmission'
        )
    
    df['voltage_category'] = (
        df['voltage_kv'].apply(voltage_category)
        if 'voltage_kv' in df.columns
        else 'Distribution'
    )
    
    df['circuit_type'] = np.random.choice(
        config["topology"]["circuit_types"],
        size=len(df),
        p=config["topology"]["circuit_probabilities"]
    )
    
    return df


if __name__ == "__main__":
    n_days = config["data"]["n_days"]
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')
    
    weather_df = pd.DataFrame({
        'timestamp': dates,
        'temperature': (
            50 + 25 * np.sin(2 * np.pi * np.arange(n_days) / 365) +
            np.random.normal(0, 5, n_days)
        ),
        'wind_speed': np.random.uniform(5, 15, n_days)
    })
    
    n_assets = config["data"]["n_assets"]
    asset_df = pd.DataFrame({
        'asset_id': [f'ASSET_{i:03d}' for i in range(1, n_assets+1)],
        'latitude': np.random.uniform(30, 45, n_assets),
        'longitude': np.random.uniform(-120, -70, n_assets),
        'installation_date': pd.to_datetime('2010-01-01') + pd.to_timedelta(
            np.random.randint(0, 5000, n_assets), unit='D'
        ),
        'voltage_kv': np.random.choice([12.5, 34.5, 69, 138], n_assets),
        'rated_current_amps': np.random.choice([200, 400, 800], n_assets)
    })
    
    weather_features = create_weather_features(weather_df)
    temporal_features = create_temporal_features(weather_features)
    geospatial_features = create_geospatial_features(asset_df)
    topology_features = create_topology_features(geospatial_features)
