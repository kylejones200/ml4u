---
title: "Renewable Integration and DER Forecasting"
description: "Forecasting PV and wind, modeling net load, and DER operational impacts."
weight: 8
draft: false
pyfile: "DER_forecasting.py"
---

## What You'll Learn

By the end of this chapter, you will understand why renewable forecasting is critical for grid operations and market participation. You'll learn to use PVLib for physics-based solar PV modeling. You'll apply SARIMA time series models to forecast renewable generation. You'll see how DER forecasting enables net load prediction, which is load minus behind-the-meter generation, and you'll recognize the operational challenges of variable renewable resources.

---

## The Business Problem: Balancing Variability from Renewable Energy

Renewable generation is rapidly reshaping power systems. Solar and wind resources reduce carbon emissions and fuel costs but also introduce variability and uncertainty. Unlike conventional plants, their output depends on weather conditions that can change hourly or even minute to minute.

This variability complicates grid operations. Large-scale solar can cause steep midday declines in net load, followed by sharp evening ramps as the sun sets. Wind farms can see output fluctuate significantly within hours. Operators must compensate by dispatching flexible generation, adjusting imports and exports, and sometimes curtailing renewables to preserve system stability.

A utility I worked with had to curtail solar generation because they couldn't forecast it accurately enough, costing them revenue and wasting clean energy. That's the cost of poor forecasting.

At the distribution level, rooftop solar presents new challenges. High penetration can drive voltage beyond acceptable limits, especially on sunny days with low local demand. Managing these impacts requires accurate forecasts of DER output to anticipate when and where problems may arise.

Without precise renewable and DER forecasting, utilities risk inefficient operations, costly reserves, and reliability concerns. The transition to cleaner grids demands analytics that can predict renewable behavior and integrate it seamlessly into planning and operations.

---

## The Analytics Solution: Forecasting Renewable and DER Output

Renewable integration relies on accurate forecasting of generation from both utility-scale and distributed sources. These forecasts combine meteorological data—irradiance, cloud cover, wind speed—with system-specific parameters such as panel orientation, inverter efficiency, and location.

For solar forecasting, tools like PVLib model photovoltaic output based on physical characteristics and weather inputs. These models can be combined with time series forecasting methods such as SARIMA to capture daily and seasonal patterns. For wind, similar physics-informed models translate wind speed forecasts into power curves.

DER forecasting extends these techniques to behind-the-meter systems. Utilities can estimate net load by subtracting predicted DER generation from gross demand forecasts, allowing better scheduling and voltage management. Advanced approaches integrate satellite imagery and sky cameras to predict short-term cloud cover impacts on solar output.

---

## Understanding PVLib

PVLib, which stands for Photovoltaic Library, is a Python package that models solar PV system performance using physical principles. It combines location data including latitude, longitude, timezone, and altitude, weather data such as solar irradiance in the form of GHI, DNI, and DHI, along with air temperature and wind speed, and system parameters including panel orientation with tilt and azimuth, module characteristics, and inverter specifications.

PVLib calculates AC power output by first converting irradiance to DC power using module characteristics, then accounting for temperature effects where panels lose efficiency when hot, and finally applying inverter efficiency to convert DC to AC.

The code uses PVLib to simulate realistic PV output, which then feeds into forecasting models. In production, utilities use actual weather forecasts and system parameters to predict generation.

---

## Operational Benefits

Improved renewable forecasting reduces the need for expensive spinning reserves and helps prevent over- or under-commitment of generation. It allows more efficient dispatch, minimizes renewable curtailment, and enhances market bidding strategies. For distribution utilities, accurate DER output estimates help avoid voltage violations and feeder overloads by informing inverter settings or reconfiguration actions.

Forecasting also enables new business models. Aggregators can bid aggregated DER resources into wholesale markets with confidence, and operators can use forecasts to design dynamic pricing programs that align demand with renewable availability.

Sympower, Europe's leading energy flexibility service provider, manages over 2GW of flexible distributed resources across about 200 customers. The challenge they faced is common: data fragmentation. Before they modernized their platform, data was either inaccessible or sitting in different places, and internal stakeholders were trying to get data themselves, creating inconsistencies in forecasting and energy market bidding.

With increasing electrification, both production and consumption become much more volatile, so they need sophisticated forecasting and market bidding capabilities. Having a unified data platform lets them harmonize all their data in one place. They use machine learning environments with Spark, MLflow, notebooks, and orchestration workflows to bring forecasts to team members daily.

The impact is real. Their trading team used to spend hours per week on forecasts, and now that's down to minutes. With Unity Catalog, colleagues in operational data-intense roles come to the platform themselves to collaborate with data teams and develop their own insights. That's the kind of democratization that actually works—when people can access data without going through a bottleneck.

NextEra Energy Resources demonstrates how data platforms enable renewable energy innovation at scale. They operate about 67 gigawatts of generation, with NextEra Energy Resources managing close to 40 gigawatts of renewable generation. They've built a central data platform on AWS that combines their portfolio of clean energy solutions with analytics capabilities to develop decarbonization roadmaps for commercial and industrial customers. The platform uses data to optimize renewable project economics—they're using analytics to squeeze the best results from solar, wind, and battery storage projects. The key is having a unified data foundation that can handle the multidimensional challenges of renewable integration: weather variability, market dynamics, grid constraints, and customer needs. They've learned that you have to look inward first—decarbonize your own operations—before you can help others. That's how you build credibility and expertise that scales.

---

## Building Renewable Forecasting Models

Let's walk through a complete renewable forecasting workflow: using PVLib to model solar PV output from weather data, then applying SARIMA to forecast future generation. This two-step approach (physics-based modeling + time series forecasting) is common in production systems.

First, we simulate solar PV output using PVLib:

{{< pyfile file="DER_forecasting.py" from="18" to="65" >}}

PVLib calculates realistic AC power from a 1MW solar system in Austin, Texas. It handles all the physics: converting irradiance to DC power, accounting for temperature effects, and applying inverter efficiency. The simulation uses clear-sky irradiance with realistic temperature and wind variations. In practice, you'd use actual weather forecasts, but the principles are the same.

Next, let's visualize what we're working with:

{{< pyfile file="DER_forecasting.py" from="66" to="77" >}}

The plot reveals the characteristic solar curve: zero at night, gradual rise in morning, peak around midday, decline in afternoon, zero at night. Variations from day to day reflect weather conditions (clouds reduce output). Always plot your data first—forecasts fail when someone doesn't notice the data has missing days.

Finally, we apply SARIMA forecasting:

{{< pyfile file="DER_forecasting.py" from="78" to="104" >}}

SARIMA extends ARIMA to handle seasonality—critical for solar which has strong 24-hour cycles. The forecast should follow the daily cycle (low at night, peak midday) and match the magnitude of recent generation. For solar, forecast accuracy typically ranges from 5-15% MAPE depending on weather conditions and forecast horizon. Utilities can get down to 8% MAPE for day-ahead forecasts, which is good enough for most operations.

The complete, runnable script is at `content/c8/DER_forecasting.py`. Note: This requires the `pvlib` package which can be installed via `pip install pvlib`.

---

## What I Want You to Remember

Renewable forecasting is essential for integration. Accurate forecasts enable utilities to schedule generation, manage reserves, and avoid curtailment while maintaining reliability. Physics-based models combined with time series methods create a powerful combination. PVLib models the physical system, while SARIMA captures temporal patterns. Together, they provide accurate, interpretable forecasts.

DER forecasting enables net load prediction. By forecasting behind-the-meter solar, utilities can predict net load, which is gross demand minus DER generation, and represents what the grid must actually supply. Forecast accuracy varies with weather. Clear days are easier to forecast than cloudy days. Short-term forecasts measured in hours are more accurate than day-ahead forecasts.

Uncertainty management is critical. Renewable forecasts have higher uncertainty than load forecasts. Operators must plan reserves and flexibility accordingly. Utilities get burned by overconfident renewable forecasts—always communicate uncertainty, not just point forecasts.

---

## What's Next

In Chapter 9, we'll shift to customer analytics—using smart meter data and clustering to segment customers for targeted demand response programs that help manage peak demand. The principles are the same, but the use case is different.
