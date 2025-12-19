"""Chapter 4: Advanced Load Forecasting with Feature Engineering and LightGBM.
Production-grade forecasting with multi-tier models, scenario planning, and EIA integration."""

import logging
import pandas as pd
import numpy as np
import signalplot as sp
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

sp.apply()

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

np.random.seed(config["model"]["random_state"])

# Optional dependencies
try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not available")

try:
    from pmdarima import auto_arima
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False
    logger.warning("pmdarima not available")

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    logger.warning("MLflow not available")


def prepare_features(load_data: pd.DataFrame, lookback_days: int = 90) -> pd.DataFrame:
    """Prepare feature dataset for modeling with advanced feature engineering.
    
    Args:
        load_data: DataFrame with columns [timestamp, Load_MW] or similar
        lookback_days: Number of days of historical data to use.
        
    Returns:
        DataFrame with engineered features.
    """
    df = load_data.copy()
    
    # Ensure timestamp column exists
    if 'timestamp' not in df.columns and 'ts_utc' not in df.columns:
        if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if 'index' in df.columns:
                df = df.rename(columns={'index': 'timestamp'})
        else:
            raise ValueError("Need timestamp column or DatetimeIndex")
    
    ts_col = 'timestamp' if 'timestamp' in df.columns else 'ts_utc'
    load_col = 'Load_MW' if 'Load_MW' in df.columns else 'mw'
    
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True)
    
    # Calendar features
    df['hour'] = df[ts_col].dt.hour
    df['dow'] = df[ts_col].dt.dayofweek + 1  # 1=Monday
    df['month'] = df[ts_col].dt.month
    df['day_of_year'] = df[ts_col].dt.dayofyear
    df['is_weekend'] = (df['dow'] >= 6).astype(int)
    
    # Cyclical encoding for hour (captures 23->0 continuity)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Cyclical encoding for day of year (captures Dec 31 -> Jan 1)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # Lag features (key for time series forecasting)
    df['mw_lag1'] = df[load_col].shift(1)      # 1 hour ago
    df['mw_lag24'] = df[load_col].shift(24)    # Same hour yesterday
    df['mw_lag168'] = df[load_col].shift(168)  # Same hour last week
    
    # Rolling statistics (capture recent trends)
    df['mw_ma24'] = df[load_col].rolling(window=24, min_periods=1).mean()
    df['mw_ma168'] = df[load_col].rolling(window=168, min_periods=1).mean()
    df['mw_std24'] = df[load_col].rolling(window=24, min_periods=1).std()
    
    # Temperature features (synthetic if real weather unavailable)
    df['temperature'] = (70 + 20 * np.sin(2 * np.pi * df['day_of_year'] / 365.25) + 
                        10 * np.sin(2 * np.pi * df['hour'] / 24))
    df['temp_squared'] = df['temperature'] ** 2
    df['cooling_degree_days'] = np.maximum(df['temperature'] - 65, 0)
    df['heating_degree_days'] = np.maximum(55 - df['temperature'], 0)
    
    # Holiday indicator (simplified - enhance with actual holiday calendar)
    df['is_holiday'] = 0
    
    return df


def train_arima_model(df: pd.DataFrame, ba: str = "SYNTHETIC") -> Optional[Dict]:
    """Train auto_arima baseline model.
    
    Args:
        df: Feature DataFrame with timestamp and load columns.
        ba: Balancing authority code.
        
    Returns:
        Dictionary with model and metrics if successful, None otherwise.
    """
    if not HAS_PMDARIMA:
        logger.warning("pmdarima not available; skipping ARIMA")
        return None
    
    logger.info(f"Training ARIMA for {ba}")
    
    # Prepare time series data
    ts_col = 'timestamp' if 'timestamp' in df.columns else 'ts_utc'
    load_col = 'Load_MW' if 'Load_MW' in df.columns else 'mw'
    
    ts_data = df[[ts_col, load_col]].copy()
    ts_data = ts_data.dropna().sort_values(ts_col)
    
    if len(ts_data) < 168:  # Need at least 1 week of data
        logger.warning(f"Insufficient data: {len(ts_data)} records")
        return None
    
    # Set datetime index
    ts_data.set_index(ts_col, inplace=True)
    ts_series = ts_data[load_col]
    
    # Use auto_arima to find optimal parameters
    model = auto_arima(
        ts_series,
        start_p=1, start_q=1,
        max_p=3, max_q=3,
        seasonal=True,
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        m=24,  # 24-hour seasonality
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore'
    )
    
    # Make in-sample predictions
    fitted_values = model.fittedvalues()
    actual_values = ts_series[fitted_values.index]
    
    # Calculate metrics
    from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
    mape = mean_absolute_percentage_error(actual_values, fitted_values)
    mae = mean_absolute_error(actual_values, fitted_values)
    
    logger.info(f"ARIMA: order={model.order}, seasonal={model.seasonal_order}, MAPE={mape:.4f}, MAE={mae:.2f}MW")
    
    # Log to MLflow if available
    if HAS_MLFLOW:
        try:
            with mlflow.start_run(run_name=f"arima_{ba}", nested=True):
                mlflow.log_metric("mape", mape)
                mlflow.log_metric("mae", mae)
                mlflow.log_param("model_type", "auto_arima")
                mlflow.log_param("order", str(model.order))
                mlflow.log_param("seasonal_order", str(model.seasonal_order))
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
    
    return {
        'model': model,
        'mape': mape,
        'mae': mae,
        'order': model.order,
        'seasonal_order': model.seasonal_order
    }


def train_lightgbm_model(df: pd.DataFrame, ba: str = "SYNTHETIC") -> Optional[Dict]:
    """Train LightGBM advanced model.
    
    Args:
        df: Feature DataFrame.
        ba: Balancing authority code.
        
    Returns:
        Dictionary with model and metrics if successful, None otherwise.
    """
    if not HAS_LIGHTGBM:
        logger.warning("LightGBM not available; skipping")
        return None
    
    logger.info(f"Training LightGBM for {ba}")
    
    # Select features
    feature_cols = [
        'mw_lag1', 'mw_lag24', 'mw_lag168',
        'mw_ma24', 'mw_ma168', 'mw_std24',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'dow', 'month', 'is_weekend', 'is_holiday',
        'temperature', 'temp_squared',
        'cooling_degree_days', 'heating_degree_days'
    ]
    
    load_col = 'Load_MW' if 'Load_MW' in df.columns else 'mw'
    
    # Filter to complete cases
    model_df = df[feature_cols + [load_col]].dropna()
    
    if len(model_df) < 168:
        logger.warning(f"Insufficient data: {len(model_df)} records")
        return None
    
    X = model_df[feature_cols]
    y = model_df[load_col]
    
    # Time series cross-validation (respects temporal order)
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=8,
        num_leaves=31,
        random_state=config["model"]["random_state"],
        verbose=-1
    )
    
    # Cross-validation predictions
    cv_predictions = np.full(len(y), np.nan)
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        cv_predictions[val_idx] = model.predict(X_val)
    
    # Calculate metrics
    valid_mask = ~np.isnan(cv_predictions)
    mape = mean_absolute_percentage_error(y[valid_mask], cv_predictions[valid_mask])
    mae = mean_absolute_error(y[valid_mask], cv_predictions[valid_mask])
    rmse = np.sqrt(mean_squared_error(y[valid_mask], cv_predictions[valid_mask]))
    
    # Train final model on all data
    model.fit(X, y)
    
    logger.info(f"LightGBM: MAPE={mape:.4f}, MAE={mae:.2f}MW, RMSE={rmse:.2f}MW")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = ", ".join([f"{row['feature']}={row['importance']:.2f}" for _, row in importance_df.head(5).iterrows()])
    logger.debug(f"Top features: {top_features}")
    
    # Log to MLflow if available
    if HAS_MLFLOW:
        try:
            with mlflow.start_run(run_name=f"lightgbm_{ba}", nested=True):
                mlflow.log_metric("mape", mape)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_param("model_type", "lightgbm")
                mlflow.log_param("n_estimators", 500)
                mlflow.log_param("learning_rate", 0.1)
                mlflow.log_text(importance_df.to_string(), "feature_importance.txt")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
    
    return {
        'model': model,
        'mape': mape,
        'mae': mae,
        'rmse': rmse,
        'feature_importance': importance_df,
        'feature_cols': feature_cols
    }


def generate_forecast(model, df: pd.DataFrame, horizon_hours: int = 24) -> List[float]:
    """Generate multi-hour forecast using trained model.
    
    Args:
        model: Trained LightGBM model.
        df: Historical data with features.
        horizon_hours: Number of hours to forecast.
        
    Returns:
        List of forecast values.
    """
    if not HAS_LIGHTGBM:
        return []
    
    feature_cols = [
        'mw_lag1', 'mw_lag24', 'mw_lag168',
        'mw_ma24', 'mw_ma168', 'mw_std24',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'dow', 'month', 'is_weekend', 'is_holiday',
        'temperature', 'temp_squared',
        'cooling_degree_days', 'heating_degree_days'
    ]
    
    ts_col = 'timestamp' if 'timestamp' in df.columns else 'ts_utc'
    load_col = 'Load_MW' if 'Load_MW' in df.columns else 'mw'
    
    # Get last complete row
    last_row = df.dropna(subset=feature_cols).iloc[-1].copy()
    forecasts = []
    
    for i in range(horizon_hours):
        # Update time-based features
        future_time = pd.to_datetime(last_row[ts_col]) + timedelta(hours=i+1)
        hour = future_time.hour
        day_of_year = future_time.dayofyear
        
        last_row['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        last_row['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        last_row['day_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        last_row['day_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
        last_row['dow'] = future_time.dayofweek + 1
        last_row['month'] = future_time.month
        last_row['is_weekend'] = int(last_row['dow'] >= 6)
        
        # Update temperature projection
        last_row['temperature'] = (70 + 20 * np.sin(2 * np.pi * day_of_year / 365.25) + 
                                   10 * np.sin(2 * np.pi * hour / 24))
        last_row['temp_squared'] = last_row['temperature'] ** 2
        last_row['cooling_degree_days'] = np.maximum(last_row['temperature'] - 65, 0)
        last_row['heating_degree_days'] = np.maximum(55 - last_row['temperature'], 0)
        
        # Prepare features
        X = last_row[feature_cols].values.reshape(1, -1)
        
        # Make prediction
        forecast = model.predict(X)[0]
        forecasts.append(forecast)
        
        # Update lag features for next iteration
        last_row['mw_lag1'] = forecast
        last_row['mw_ma24'] = 0.95 * last_row['mw_ma24'] + 0.05 * forecast
    
    return forecasts


def apply_scenario(df: pd.DataFrame, scenario_id: str) -> pd.DataFrame:
    """Apply scenario adjustments to feature DataFrame.
    
    Args:
        df: Base features DataFrame.
        scenario_id: Scenario identifier (baseline, hot_weather, high_growth, demand_response).
        
    Returns:
        Modified DataFrame with scenario adjustments.
    """
    scenario_df = df.copy()
    
    if scenario_id == "hot_weather":
        # Model heat wave: +15 degrees F temperature increase
        scenario_df['temperature'] += 15
        scenario_df['temp_squared'] = scenario_df['temperature'] ** 2
        scenario_df['cooling_degree_days'] = np.maximum(scenario_df['temperature'] - 65, 0)
        scenario_df['heating_degree_days'] = np.maximum(55 - scenario_df['temperature'], 0)
    
    elif scenario_id == "high_growth":
        # Model 5% load growth across all hours
        lag_cols = ['mw_lag1', 'mw_lag24', 'mw_lag168', 'mw_ma24', 'mw_ma168']
        existing_cols = [col for col in lag_cols if col in scenario_df.columns]
        scenario_df[existing_cols] *= 1.05
    
    elif scenario_id == "demand_response":
        # Model demand response program: reduce peak load 10%
        peak_hours = scenario_df['hour'].between(16, 21)
        if 'mw_lag1' in scenario_df.columns:
            scenario_df.loc[peak_hours, 'mw_lag1'] *= 0.9
        if 'mw_lag24' in scenario_df.columns:
            scenario_df.loc[peak_hours, 'mw_lag24'] *= 0.9
    
    # baseline: no adjustments
    return scenario_df


def visualize_forecasts(df: pd.DataFrame, arima_results: Optional[Dict], 
                        lgbm_results: Optional[Dict], forecasts: List[float]):
    """Visualize forecasting results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    ts_col = 'timestamp' if 'timestamp' in df.columns else 'ts_utc'
    load_col = 'Load_MW' if 'Load_MW' in df.columns else 'mw'
    
    # Plot 1: Historical load with forecasts
    ax1 = axes[0, 0]
    recent_data = df.tail(168)  # Last week
    ax1.plot(recent_data[ts_col], recent_data[load_col], 
            'o-', linewidth=2, markersize=3, label='Historical', color='#3498db')
    
    if forecasts:
        forecast_times = pd.date_range(
            start=recent_data[ts_col].iloc[-1] + timedelta(hours=1),
            periods=len(forecasts),
            freq='h'
        )
        ax1.plot(forecast_times, forecasts, 
                's-', linewidth=2, markersize=4, label='Forecast', color='#e74c3c')
    
    ax1.set_xlabel('Time', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Load (MW)', fontweight='bold', fontsize=11)
    ax1.set_title('Historical Load and Forecast', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Model comparison
    ax2 = axes[0, 1]
    if arima_results and lgbm_results:
        models = ['ARIMA', 'LightGBM']
        mapes = [arima_results['mape'], lgbm_results['mape']]
        colors = ['#f39c12', '#2ecc71']
        
        bars = ax2.bar(models, mapes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('MAPE', fontweight='bold', fontsize=11)
        ax2.set_title('Model Performance Comparison', fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        for bar, mape in zip(bars, mapes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mape:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 3: Feature importance (if available)
    ax3 = axes[1, 0]
    if lgbm_results and 'feature_importance' in lgbm_results:
        importance = lgbm_results['feature_importance'].head(10)
        ax3.barh(importance['feature'], importance['importance'], 
                color='#9b59b6', alpha=0.8, edgecolor='black')
        ax3.set_xlabel('Importance', fontweight='bold', fontsize=11)
        ax3.set_title('Top 10 Feature Importance', fontweight='bold', fontsize=12)
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
    else:
        ax3.text(0.5, 0.5, 'Feature Importance\nNot Available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.axis('off')
    
    # Plot 4: Scenario planning
    ax4 = axes[1, 1]
    if forecasts and lgbm_results:
        scenarios = ['baseline', 'hot_weather', 'high_growth', 'demand_response']
        scenario_forecasts = {}
        
        for scenario in scenarios:
            scenario_df = apply_scenario(df.tail(168), scenario)
            scenario_forecasts[scenario] = generate_forecast(
                lgbm_results['model'], scenario_df, horizon_hours=24
            )
        
        hours = range(24)
        for scenario, sc_forecasts in scenario_forecasts.items():
            if sc_forecasts:
                ax4.plot(hours, sc_forecasts, 'o-', linewidth=2, markersize=4, label=scenario)
        
        ax4.set_xlabel('Hours Ahead', fontweight='bold', fontsize=11)
        ax4.set_ylabel('Load (MW)', fontweight='bold', fontsize=11)
        ax4.set_title('Scenario Planning', fontweight='bold', fontsize=12)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
    else:
        ax4.text(0.5, 0.5, 'Scenario Planning\nNot Available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.axis('off')
    
    plt.suptitle('Advanced Load Forecasting: Multi-Tier Models and Scenario Planning',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = config["plotting"]["output_files"]["advanced_forecasting"]
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.debug(f"Saved: {output_path}")
    plt.close()


def main():
    """Main execution."""
    logger.info("Advanced load forecasting")
    
    # Generate or load synthetic data
    from load_forecasting import generate_synthetic_load
    df_load = generate_synthetic_load()
    df_load = df_load.rename(columns={'timestamp': 'timestamp', 'Load_MW': 'Load_MW'})
    
    logger.info(f"Loaded {len(df_load):,} hours")
    
    # Prepare features
    df_features = prepare_features(df_load, lookback_days=90)
    logger.info(f"Features: {len(df_load.columns)} -> {len(df_features.columns)}")
    
    # Train models
    arima_results = train_arima_model(df_features, "SYNTHETIC")
    lgbm_results = train_lightgbm_model(df_features, "SYNTHETIC")
    
    # Generate forecasts
    forecasts = []
    if lgbm_results:
        forecasts = generate_forecast(lgbm_results['model'], df_features, horizon_hours=24)
        logger.info(f"Forecast: {min(forecasts):.0f}-{max(forecasts):.0f} MW")
    
    # Visualize
    visualize_forecasts(df_features, arima_results, lgbm_results, forecasts)
    
    
    return {
        'arima': arima_results,
        'lightgbm': lgbm_results,
        'forecasts': forecasts
    }


if __name__ == '__main__':
    results = main()

