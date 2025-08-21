"""
Chapter 8: Renewable Integration and DER Forecasting
Solar PV generation modeling using PVLib and forecasting with SARIMA.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pvlib import location, modelchain, pvsystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from statsmodels.tsa.statespace.sarimax import SARIMAX

def simulate_solar_pv():
    """
    Simulate solar PV output using PVLib for a 1MW system in Texas.
    """
    site = location.Location(latitude=30.27, longitude=-97.74, tz="US/Central", altitude=149, name="Austin, TX")
    times = pd.date_range("2022-06-01", "2022-06-14", freq="H", tz=site.tz)

    # Clear-sky irradiance
    clearsky = site.get_clearsky(times)
    temp_air = np.random.normal(30, 3, len(times))
    wind_speed = np.random.uniform(1, 5, len(times))

    # PV system configuration
    temp_params = TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]
    module = pvsystem.retrieve_sam("CECMod")["Canadian_Solar_CS5P_220M___2009_"]
    inverter = pvsystem.retrieve_sam("cecinverter")["ABB__MICRO_0_25_I_OUTD_US_208__208V_"]

    system = pvsystem.PVSystem(surface_tilt=25, surface_azimuth=180, module_parameters=module,
                               inverter_parameters=inverter, temperature_model_parameters=temp_params)

    mc = modelchain.ModelChain(system, site)
    weather = pd.DataFrame({
        "ghi": clearsky["ghi"],
        "dni": clearsky["dni"],
        "dhi": clearsky["dhi"],
        "temp_air": temp_air,
        "wind_speed": wind_speed
    }, index=times)

    mc.run_model(weather)
    ac_power = mc.results.ac
    df = pd.DataFrame({"timestamp": times, "AC_Power_kW": ac_power})
    return df

def plot_pv(df):
    """
    Plot PV output.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(df["timestamp"], df["AC_Power_kW"], color="black")
    plt.xlabel("Time")
    plt.ylabel("AC Power (kW)")
    plt.title("Simulated Solar PV Output (Austin, TX)")
    plt.tight_layout()
    plt.savefig("chapter8_pv_output.png")
    plt.show()

def sarima_forecast(df):
    """
    Forecast PV output using SARIMA (Statsmodels).
    """
    df = df.set_index("timestamp")
    ts = df["AC_Power_kW"].asfreq("H")

    # Fit SARIMA model (daily seasonality: 24 hours)
    model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 0, 24), enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)

    forecast_steps = 24  # 1 day ahead
    forecast = fit.forecast(steps=forecast_steps)

    plt.figure(figsize=(12, 4))
    plt.plot(ts[-48:], label="Observed (Last 2 Days)", color="gray")
    plt.plot(forecast.index, forecast, label="SARIMA Forecast (Next 24h)", color="black")
    plt.xlabel("Time")
    plt.ylabel("AC Power (kW)")
    plt.title("SARIMA Forecast of Solar PV Output")
    plt.legend()
    plt.tight_layout()
    plt.savefig("chapter8_sarima_forecast.png")
    plt.show()

if __name__ == "__main__":
    df_pv = simulate_solar_pv()
    plot_pv(df_pv)
    sarima_forecast(df_pv)
