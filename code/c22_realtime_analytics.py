"""Chapter 22: Real-Time Analytics and Control Room Integration."""

import logging
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

np.random.seed(config["model"]["random_state"])


def simulate_scada_stream(
    n_points: int = None, interval_seconds: int = None
) -> pd.DataFrame:
    """Simulate streaming SCADA telemetry data."""
    n_points = n_points or config["data"]["n_points"]
    interval_seconds = interval_seconds or config["data"]["interval_seconds"]
    
    asset_ids = [f"SUB_{i:03d}" for i in range(1, config["data"]["n_assets"] + 1)]
    start_time = datetime.now() - timedelta(hours=1)
    timestamps = [
        start_time + timedelta(seconds=i * interval_seconds)
        for i in range(n_points)
    ]
    
    anomaly_rate = config["anomaly"]["anomaly_rate"]
    high_load_rate = config["anomaly"]["high_load_rate"]
    sensor = config["sensor"]
    
    data = []
    for i, ts in enumerate(timestamps):
        asset_id = asset_ids[i % len(asset_ids)]
        r = np.random.random()
        
        if r < anomaly_rate:
            voltage = sensor["voltage_sag_kv"] + np.random.normal(0, 0.3)
            current = sensor["current_elevated_amps"] + np.random.normal(0, 10)
        elif r < anomaly_rate + high_load_rate:
            voltage = sensor["voltage_normal_kv"] + np.random.normal(0, 0.1)
            current = sensor["current_high_amps"] + np.random.normal(0, 15)
        else:
            voltage = sensor["voltage_normal_kv"] + np.random.normal(0, 0.1)
            current = sensor["current_normal_amps"] + np.random.normal(0, 10)
        
        power_mw = (voltage * current) / 1000
        
        data.append({
            'timestamp': ts,
            'asset_id': asset_id,
            'voltage_kv': round(voltage, 2),
            'current_amps': round(current, 1),
            'power_mw': round(power_mw, 2)
        })
    
    return pd.DataFrame(data)


def normalize_protocol_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize SCADA data from different vendor protocols."""
    df = df.copy()
    anomaly = config["anomaly"]
    sensor = config["sensor"]
    
    warning_threshold = anomaly["voltage_warning_threshold"]
    critical_threshold = anomaly["voltage_critical_threshold"]
    
    df['quality_flag'] = 0
    df.loc[df['voltage_kv'] < warning_threshold, 'quality_flag'] = 1
    df.loc[df['voltage_kv'] < critical_threshold, 'quality_flag'] = 2
    
    df['loading_pct'] = (df['current_amps'] / sensor["rated_current_amps"]) * 100
    df['power_factor'] = 0.95 + np.random.normal(0, 0.02, len(df))
    df['asset_type'] = 'Transformer'
    df['rated_voltage_kv'] = sensor["rated_voltage_kv"]
    df['rated_current_amps'] = sensor["rated_current_amps"]
    
    return df


def real_time_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Real-time anomaly detection using rolling statistics."""
    df = df.copy()
    anomaly = config["anomaly"]
    window = anomaly["rolling_window"]
    
    df['voltage_mean'] = df.groupby('asset_id')['voltage_kv'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    df['voltage_std'] = df.groupby('asset_id')['voltage_kv'].transform(
        lambda x: x.rolling(window=window, min_periods=1).std()
    )
    
    df['voltage_zscore'] = (
        (df['voltage_kv'] - df['voltage_mean']) / (df['voltage_std'] + 1e-6)
    )
    
    df['is_anomaly'] = (
        (df['voltage_zscore'].abs() > anomaly["zscore_threshold"]) |
        (df['voltage_kv'] < anomaly["voltage_warning_threshold"]) |
        (df['loading_pct'] > anomaly["loading_warning_pct"])
    )
    
    df['severity'] = df['is_anomaly'].astype(int)
    df.loc[
        (df['voltage_kv'] < anomaly["voltage_critical_threshold"]) |
        (df['loading_pct'] > anomaly["loading_critical_pct"]),
        'severity'
    ] = 2
    
    df['confidence'] = 1.0 - np.minimum(df['voltage_zscore'].abs() / 5.0, 1.0)
    
    return df


def _get_alert_details(row: pd.Series) -> tuple:
    """Determine alert type and recommendation from row data."""
    anomaly = config["anomaly"]
    voltage = row['voltage_kv']
    loading = row['loading_pct']
    
    alert_rules = [
        (
            voltage < anomaly["voltage_critical_threshold"],
            "VOLTAGE_CRITICAL",
            "Check transformer tap position, verify capacitor bank status"
        ),
        (
            voltage < anomaly["voltage_warning_threshold"],
            "VOLTAGE_WARNING",
            "Monitor voltage trend, consider load shedding if continues"
        ),
        (
            loading > anomaly["loading_critical_pct"],
            "OVERLOAD_CRITICAL",
            "Reduce load or transfer to alternate feeder"
        ),
        (
            loading > anomaly["loading_warning_pct"],
            "OVERLOAD_WARNING",
            "Monitor loading, prepare for load transfer if needed"
        ),
    ]
    
    for condition, alert_type, recommendation in alert_rules:
        if condition:
            return alert_type, recommendation
    
    return "ANOMALY_DETECTED", "Review telemetry, check for equipment issues"


def generate_control_room_alerts(df_anomalies: pd.DataFrame) -> List[Dict]:
    """Generate alerts for control room operators."""
    alerts_df = df_anomalies[df_anomalies['is_anomaly']].copy()
    alerts_df = alerts_df.sort_values(
        ['severity', 'timestamp'], ascending=[False, False]
    )
    
    alerts = []
    for _, row in alerts_df.iterrows():
        alert_type, recommendation = _get_alert_details(row)
        
        alerts.append({
            'timestamp': row['timestamp'].isoformat(),
            'asset_id': row['asset_id'],
            'alert_type': alert_type,
            'severity': int(row['severity']),
            'voltage_kv': round(row['voltage_kv'], 2),
            'current_amps': round(row['current_amps'], 1),
            'loading_pct': round(row['loading_pct'], 1),
            'confidence': round(row['confidence'], 2),
            'recommendation': recommendation
        })
    
    return alerts


def simulate_streaming_pipeline():
    """Complete streaming analytics pipeline simulation."""
    scada_df = simulate_scada_stream()
    normalized_df = normalize_protocol_data(scada_df)
    anomalies_df = real_time_anomaly_detection(normalized_df)
    alerts = generate_control_room_alerts(anomalies_df)
    
    return {
        'scada_data': scada_df,
        'normalized_data': normalized_df,
        'anomalies': anomalies_df,
        'alerts': alerts
    }


if __name__ == "__main__":
    results = simulate_streaming_pipeline()
