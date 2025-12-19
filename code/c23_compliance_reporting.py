"""Chapter 23: Regulatory Compliance and Reliability Reporting."""

import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from pathlib import Path

config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

np.random.seed(config["model"]["random_state"])


def generate_outage_data(n_outages=None, n_customers=None):
    """Generate synthetic outage data for reliability metric calculation."""
    n_outages = n_outages or config["data"]["n_outages"]
    n_customers = n_customers or config["data"]["n_customers"]
    
    start_date = datetime.now() - timedelta(days=config["data"]["days_back"])
    outage_config = config["outage"]
    
    outages = []
    for i in range(n_outages):
        outage_start = start_date + timedelta(
            days=np.random.randint(0, config["data"]["days_back"]),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )
        
        duration_minutes = np.random.exponential(
            (outage_config["short_outage_mean_minutes"]
             if np.random.random() < outage_config["short_outage_rate"]
             else outage_config["long_outage_mean_minutes"])
        )
        duration_minutes = min(duration_minutes, outage_config["max_duration_minutes"])
        
        customers_affected = np.random.randint(
            *(outage_config["small_outage_min"], outage_config["small_outage_max"])
            if np.random.random() < outage_config["small_outage_rate"]
            else (outage_config["large_outage_min"], outage_config["large_outage_max"])
        )
        
        cause = np.random.choice(
            config["causes"],
            p=config["cause_probabilities"]
        )
        
        outages.append({
            'outage_id': f'OUT_{i+1:04d}',
            'start_time': outage_start,
            'end_time': outage_start + timedelta(minutes=duration_minutes),
            'duration_minutes': duration_minutes,
            'customers_affected': customers_affected,
            'cause': cause
        })
    
    df = pd.DataFrame(outages)
    return df.sort_values('start_time'), n_customers


def calculate_saidi_saifi(outage_df, total_customers):
    """Calculate SAIDI and SAIFI reliability metrics."""
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


def generate_audit_trail(model_operations):
    """Generate audit trail for ML model operations."""
    audit_df = pd.DataFrame(model_operations)
    audit_df['timestamp'] = pd.to_datetime(audit_df['timestamp'])
    return audit_df.sort_values('timestamp')


def demonstrate_compliance_tracking():
    """Demonstrate compliance tracking for ML systems."""
    base_time = datetime.now() - timedelta(days=30)
    
    operations = [
        {
            'timestamp': base_time + timedelta(days=1),
            'user_id': 'data_scientist_001',
            'operation_type': 'model_training',
            'model_id': 'transformer_failure_v2',
            'data_sources': ['SCADA', 'EAM', 'weather'],
            'training_samples': 50000,
            'performance_metrics': {'accuracy': 0.92, 'precision': 0.88, 'recall': 0.85}
        },
        {
            'timestamp': base_time + timedelta(days=2),
            'user_id': 'ml_engineer_001',
            'operation_type': 'model_deployment',
            'model_id': 'transformer_failure_v2',
            'environment': 'production',
            'approver': 'operations_manager_001'
        },
        {
            'timestamp': base_time + timedelta(days=3),
            'user_id': 'analyst_001',
            'operation_type': 'data_access',
            'data_source': 'SCADA',
            'records_accessed': 1000000,
            'purpose': 'load_forecasting'
        },
        {
            'timestamp': base_time + timedelta(days=4),
            'user_id': 'system_auto',
            'operation_type': 'prediction',
            'model_id': 'transformer_failure_v2',
            'predictions_generated': 500,
            'high_risk_flags': 12
        },
        {
            'timestamp': base_time + timedelta(days=5),
            'user_id': 'ml_engineer_001',
            'operation_type': 'configuration_change',
            'change_type': 'threshold_adjustment',
            'old_value': 0.8,
            'new_value': 0.75,
            'approver': 'operations_manager_001'
        }
    ]
    
    return generate_audit_trail(operations)


if __name__ == "__main__":
    outage_df, total_customers = generate_outage_data()
    metrics = calculate_saidi_saifi(outage_df, total_customers)
    cause_analysis = analyze_outage_causes(outage_df)
    audit_trail = demonstrate_compliance_tracking()
