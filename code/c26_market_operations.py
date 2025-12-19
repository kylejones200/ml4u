"""Chapter 26: Market Operations and Energy Trading."""

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


def generate_price_data(n_days=None):
    """Generate synthetic electricity price data."""
    n_days = n_days or config["data"]["n_days"]
    start_date = datetime.now() - timedelta(days=n_days)
    timestamps = pd.date_range(start=start_date, periods=n_days * 24, freq='H')
    
    pricing = config["pricing"]
    load_cfg = config["load"]
    peak_hours = pricing["peak_hours"]
    base_price = pricing["base_price_dollars_per_mwh"]
    
    prices = []
    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.dayofweek
        day_of_year = ts.dayofyear
        
        time_multiplier = (pricing["peak_multiplier"]
                          if peak_hours[0] <= hour <= peak_hours[1]
                          else pricing["offpeak_multiplier"])
        day_multiplier = (pricing["weekday_multiplier"]
                         if day_of_week < 5
                         else pricing["weekend_multiplier"])
        seasonal_effect = (
            1.0 + pricing["seasonal_amplitude"] *
            np.sin(2 * np.pi * day_of_year / 365)
        )
        volatility = np.random.normal(1.0, pricing["volatility_std"])
        
        price = base_price * time_multiplier * day_multiplier * seasonal_effect * volatility
        load = (
            load_cfg["base_load_mw"] +
            load_cfg["load_price_factor"] * (price / base_price) +
            np.random.normal(0, load_cfg["load_noise_std"])
        )
        temp = (
            load_cfg["temp_base"] +
            load_cfg["temp_amplitude"] * np.sin(2 * np.pi * day_of_year / 365) +
            np.random.normal(0, load_cfg["temp_noise_std"])
        )
        
        prices.append({
            'timestamp': ts,
            'price_dollars_per_mwh': round(price, 2),
            'load_mw': round(load, 1),
            'temperature_f': round(temp, 1),
            'hour': hour,
            'day_of_week': day_of_week,
            'is_peak': int(peak_hours[0] <= hour <= peak_hours[1])
        })
    
    return pd.DataFrame(prices)


def forecast_prices(price_df, forecast_hours=None):
    """Forecast electricity prices for day-ahead market."""
    forecast_hours = forecast_hours or config["data"]["forecast_hours"]
    price_df = price_df.copy()
    
    price_df['hour_sin'] = np.sin(2 * np.pi * price_df['hour'] / 24)
    price_df['hour_cos'] = np.cos(2 * np.pi * price_df['hour'] / 24)
    price_df['day_of_week_sin'] = np.sin(2 * np.pi * price_df['day_of_week'] / 7)
    price_df['day_of_week_cos'] = np.cos(2 * np.pi * price_df['day_of_week'] / 7)
    
    feature_cols = [
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
        'load_mw', 'temperature_f', 'is_peak'
    ]
    X = price_df[feature_cols].fillna(0)
    y = price_df['price_dollars_per_mwh']
    
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
    
    last_24h = price_df.tail(24)
    peak_hours = config["pricing"]["peak_hours"]
    
    forecasts = []
    for i in range(forecast_hours):
        next_hour = (last_24h.iloc[-1]['hour'] + 1) % 24
        next_day_of_week = (last_24h.iloc[-1]['day_of_week'] + (next_hour == 0)) % 7
        
        forecast_features = pd.DataFrame({
            'hour_sin': [np.sin(2 * np.pi * next_hour / 24)],
            'hour_cos': [np.cos(2 * np.pi * next_hour / 24)],
            'day_of_week_sin': [np.sin(2 * np.pi * next_day_of_week / 7)],
            'day_of_week_cos': [np.cos(2 * np.pi * next_day_of_week / 7)],
            'load_mw': [last_24h['load_mw'].mean()],
            'temperature_f': [last_24h['temperature_f'].mean()],
            'is_peak': [int(peak_hours[0] <= next_hour <= peak_hours[1])]
        })
        
        forecast_price = model.predict(forecast_features)[0]
        forecasts.append({
            'forecast_hour': i + 1,
            'hour': next_hour,
            'forecast_price': round(forecast_price, 2)
        })
    
    return pd.DataFrame(forecasts), model


def optimize_bidding(forecast_df, generation_costs=None):
    """Optimize bidding strategy for day-ahead market."""
    generation_costs = generation_costs or {
        'Gas_Combined_Cycle': config["generation"]["gas_combined_cycle_cost"],
        'Gas_Peaker': config["generation"]["gas_peaker_cost"],
        'Coal': config["generation"]["coal_cost"],
        'Nuclear': config["generation"]["nuclear_cost"]
    }
    
    gen = config["generation"]
    bids = []
    
    for _, row in forecast_df.iterrows():
        forecast_price = row['forecast_price']
        hour = row['hour']
        
        for unit_name, unit_cost in generation_costs.items():
            is_profitable = forecast_price > unit_cost
            bid_price = (forecast_price * gen["bid_margin"] if is_profitable else None)
            bid_quantity = (gen["default_bid_quantity_mw"] if is_profitable else 0)
            expected_profit = ((forecast_price - unit_cost) * bid_quantity
                              if is_profitable else 0)
            
            bids.append({
                'hour': hour,
                'unit': unit_name,
                'forecast_price': forecast_price,
                'unit_cost': unit_cost,
                'bid_price': round(bid_price, 2) if bid_price else None,
                'bid_quantity_mw': bid_quantity,
                'expected_profit': round(expected_profit, 2)
            })
    
    return pd.DataFrame(bids)


def analyze_risk(price_df):
    """Analyze price risk for market participation."""
    price = price_df['price_dollars_per_mwh']
    price_std = price.std()
    price_mean = price.mean()
    
    spike_threshold = (
        price_mean + config["risk"]["spike_std_multiplier"] * price_std
    )
    n_spikes = (price > spike_threshold).sum()
    
    return {
        'mean_price': round(price_mean, 2),
        'std_price': round(price_std, 2),
        'coefficient_of_variation': round(price_std / price_mean, 3),
        'spike_threshold': round(spike_threshold, 2),
        'n_spikes': n_spikes,
        'pct_spikes': round(n_spikes / len(price_df) * 100, 2),
        'var_95': round(price.quantile(config["risk"]["var_percentile"]), 2)
    }


if __name__ == "__main__":
    price_df = generate_price_data()
    forecast_df, model = forecast_prices(price_df)
    bid_df = optimize_bidding(forecast_df)
    risk_metrics = analyze_risk(price_df)
