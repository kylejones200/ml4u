"""Chapter 8: Renewable Integration and DER Forecasting."""

import logging
import pandas as pd
import numpy as np
import signalplot as sp
import yaml
from pathlib import Path
from pvlib import location, modelchain, pvsystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from statsmodels.tsa.statespace.sarimax import SARIMAX

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

sp.apply()

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)


def simulate_solar_pv():
    """Simulate solar PV output using PVLib for a 1MW system."""
    site = location.Location(
        latitude=config["pv"]["latitude"],
        longitude=config["pv"]["longitude"],
        tz=config["pv"]["timezone"],
        altitude=config["pv"]["altitude"],
        name=config["pv"]["location_name"]
    )
    times = pd.date_range(config["pv"]["start_date"], config["pv"]["end_date"], 
                         freq="h", tz=site.tz)

    clearsky = site.get_clearsky(times)
    temp_air = np.random.normal(config["weather"]["temp_mean"], 
                                config["weather"]["temp_std"], len(times))
    wind_speed = np.random.uniform(config["weather"]["wind_min"], 
                                   config["weather"]["wind_max"], len(times))

    temp_params = TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]
    module_db = pvsystem.retrieve_sam("CECMod")
    module_key = module_db.columns[0]
    module = module_db[module_key]
    inv_db = pvsystem.retrieve_sam("cecinverter")
    inv_key = inv_db.columns[0]
    inverter = inv_db[inv_key]

    system = pvsystem.PVSystem(
        surface_tilt=config["pv"]["surface_tilt"],
        surface_azimuth=config["pv"]["surface_azimuth"],
        module_parameters=module,
        inverter_parameters=inverter,
        temperature_model_parameters=temp_params
    )

    mc = modelchain.ModelChain(
        system, site, aoi_model="no_loss", spectral_model="no_loss"
    )
    weather = pd.DataFrame({
        "ghi": clearsky["ghi"],
        "dni": clearsky["dni"],
        "dhi": clearsky["dhi"],
        "temp_air": temp_air,
        "wind_speed": wind_speed
    }, index=times)

    mc.run_model(weather)
    ac_power = mc.results.ac
    return pd.DataFrame({"timestamp": times, "AC_Power_kW": ac_power})


def plot_pv(df):
    """Plot PV output."""
    fig, ax = sp.figure()
    ax.plot(df["timestamp"], df["AC_Power_kW"])
    sp.style_line_plot(ax)
    sp.savefig(config["plotting"]["output_files"]["pv"])


def sarima_forecast(df):
    """Forecast PV output using SARIMA."""
    df = df.set_index("timestamp")
    ts = df["AC_Power_kW"].asfreq("H")

    order = tuple(config["sarima"]["order"])
    seasonal_order = tuple(config["sarima"]["seasonal_order"])
    model = SARIMAX(ts, order=order, seasonal_order=seasonal_order,
                   enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)

    forecast = fit.forecast(steps=config["sarima"]["forecast_steps"])

    fig, ax = sp.figure()
    ax.plot(ts[-48:])
    ax.plot(forecast.index, forecast, linestyle="--")
    sp.style_line_plot(ax)
    sp.savefig(config["plotting"]["output_files"]["forecast"])


if __name__ == "__main__":
    df_pv = simulate_solar_pv()
    plot_pv(df_pv)
    sarima_forecast(df_pv)
