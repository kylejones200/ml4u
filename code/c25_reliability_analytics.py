"""Chapter 25: Reliability Analytics and Performance Metrics."""

import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

np.random.seed(config["model"]["random_state"])


def calculate_reliability_metrics(outage_df, total_customers):
    """Calculate comprehensive reliability metrics."""
    outage_df['customer_minutes'] = (
        outage_df['duration_minutes'] * outage_df['customers_affected']
    )
    
    total_customer_minutes = outage_df['customer_minutes'].sum()
    total_customer_interruptions = outage_df['customers_affected'].sum()
    
    saidi = total_customer_minutes / total_customers
    saifi = total_customer_interruptions / total_customers
    caidi = (total_customer_minutes / total_customer_interruptions
             if total_customer_interruptions > 0 else 0)
    
    momentary_threshold = config["reliability"]["momentary_threshold_minutes"]
    momentary_outages = outage_df[outage_df['duration_minutes'] < momentary_threshold]
    momentary_interruptions = momentary_outages['customers_affected'].sum()
    maifi = momentary_interruptions / total_customers
    
    affected_customers = outage_df['customers_affected'].sum()
    caifi = (total_customer_interruptions / affected_customers
             if affected_customers > 0 else 0)
    
    return {
        'SAIDI': round(saidi, 2),
        'SAIFI': round(saifi, 2),
        'CAIDI': round(caidi, 2),
        'MAIFI': round(maifi, 2),
        'CAIFI': round(caifi, 2),
        'total_customer_minutes': int(total_customer_minutes),
        'total_customer_interruptions': int(total_customer_interruptions),
        'total_customers': total_customers,
        'number_of_outages': len(outage_df)
    }


def analyze_outage_causes(outage_df):
    """Analyze outage causes to identify improvement opportunities."""
    cause_analysis = outage_df.groupby('cause').agg({
        'outage_id': 'count',
        'customers_affected': 'sum',
        'customer_minutes': 'sum',
        'duration_minutes': 'mean'
    }).reset_index()
    
    cause_analysis.columns = [
        'cause', 'outage_count', 'total_customers_affected',
        'total_customer_minutes', 'avg_duration_minutes'
    ]
    
    total_minutes = outage_df['customer_minutes'].sum()
    cause_analysis['pct_of_outages'] = (
        cause_analysis['outage_count'] / len(outage_df) * 100
    )
    cause_analysis['pct_of_customer_minutes'] = (
        cause_analysis['total_customer_minutes'] / total_minutes * 100
    )
    
    return cause_analysis.sort_values('total_customer_minutes', ascending=False)


def build_reliability_prediction_model(outage_df, asset_df):
    """Build predictive model for reliability performance."""
    feeder_outages = outage_df.groupby('asset_id').agg({
        'customer_minutes': 'sum',
        'outage_id': 'count'
    }).reset_index()
    feeder_outages.columns = ['feeder_id', 'customer_minutes', 'outage_count']
    
    model_data = asset_df.merge(
        feeder_outages,
        left_on='asset_id',
        right_on='feeder_id',
        how='left'
    )
    model_data[['customer_minutes', 'outage_count']] = (
        model_data[['customer_minutes', 'outage_count']].fillna(0)
    )
    
    model_data['asset_age_years'] = (
        (datetime.now() - pd.to_datetime(model_data['installation_date']))
        .dt.days / 365.25
    )
    
    X = model_data[['asset_age_years', 'voltage_kv']].fillna(0)
    y = model_data['customer_minutes']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"]
    )
    
    model = RandomForestRegressor(
        n_estimators=config["model"]["n_estimators"],
        random_state=config["model"]["random_state"]
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': model,
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'feature_importance': feature_importance
    }


def track_customer_interruptions(outage_df):
    """Track customer-level interruption patterns."""
    customer_impact = outage_df.groupby('cause').agg({
        'customers_affected': ['sum', 'mean', 'max'],
        'duration_minutes': 'mean'
    }).reset_index()
    
    customer_impact.columns = [
        'cause', 'total_customers_affected',
        'avg_customers_per_outage', 'max_customers_per_outage', 'avg_duration'
    ]
    
    return customer_impact.sort_values('total_customers_affected', ascending=False)


if __name__ == "__main__":
    np.random.seed(config["model"]["random_state"])
    n_outages = config["data"]["n_outages"]
    total_customers = config["data"]["n_customers"]
    
    causes = [
        'Equipment Failure', 'Weather', 'Vegetation', 'Animal Contact',
        'Vehicle Accident', 'Planned Maintenance', 'Unknown'
    ]
    cause_probs = [0.25, 0.20, 0.15, 0.10, 0.10, 0.10, 0.10]
    
    outages = [
        {
            'outage_id': f'OUT_{i+1:04d}',
            'asset_id': f'FEEDER_{np.random.randint(1, config["data"]["n_feeders"]+1):03d}',
            'start_time': datetime.now() - timedelta(
                days=np.random.randint(0, config["data"]["days_back"])
            ),
            'duration_minutes': np.random.exponential(60),
            'customers_affected': np.random.randint(10, 2000),
            'cause': np.random.choice(causes, p=cause_probs)
        }
        for i in range(n_outages)
    ]
    
    outage_df = pd.DataFrame(outages)
    outage_df['customer_minutes'] = (
        outage_df['duration_minutes'] * outage_df['customers_affected']
    )
    
    asset_df = pd.DataFrame({
        'asset_id': [
            f'FEEDER_{i:03d}'
            for i in range(1, config["data"]["n_feeders"]+1)
        ],
        'installation_date': pd.to_datetime('2010-01-01') + pd.to_timedelta(
            np.random.randint(0, 5000, config["data"]["n_feeders"]), unit='D'
        ),
        'voltage_kv': np.random.choice([12.5, 34.5], config["data"]["n_feeders"])
    })
    
    metrics = calculate_reliability_metrics(outage_df, total_customers)
    cause_analysis = analyze_outage_causes(outage_df)
    model_results = build_reliability_prediction_model(outage_df, asset_df)
    customer_analysis = track_customer_interruptions(outage_df)
