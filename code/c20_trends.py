"""Chapter 20: Future Trends and Strategic Roadmap."""

import logging
import numpy as np
import pandas as pd
import signalplot as sp
import yaml
from pathlib import Path

sp.apply()

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)


def simulate_ai_adoption():
    """Simulate utility KPI improvements over time from AI adoption."""
    years = config["simulation"]["years"]
    start_year = config["simulation"]["start_year"]
    timeline = np.arange(start_year, start_year + years)
    
    return pd.DataFrame({
        "Year": timeline,
        "Cost_Savings_%": np.linspace(0, 50, years) + np.random.normal(0, 3, years),
        "Outage_Reduction_%": np.linspace(0, 40, years) + np.random.normal(0, 2, years),
        "Forecast_Accuracy_%": np.linspace(70, 95, years)
    })


def plot_kpi_trends(df):
    """Plot projected KPI improvements from AI adoption."""
    fig, ax = sp.figure()
    ax.plot(df["Year"], df["Cost_Savings_%"])
    ax.plot(df["Year"], df["Outage_Reduction_%"])
    ax.plot(df["Year"], df["Forecast_Accuracy_%"])
    sp.style_line_plot(ax)
    sp.savefig(config["plotting"]["trends_output_file"])


def strategic_recommendations(df):
    """Print roadmap recommendations based on KPI improvements."""
    final_year = df['Year'].iloc[-1]
    final_savings = df["Cost_Savings_%"].iloc[-1]
    final_outage_reduction = df["Outage_Reduction_%"].iloc[-1]
    
    logger.info(f"\nProjected O&M Savings in {final_year}: {final_savings:.1f}%")
    logger.info(f"Projected Outage Reduction in {final_year}: {final_outage_reduction:.1f}%")
    
    recommendations = [
        "Prioritize predictive maintenance to accelerate O&M cost savings.",
        "Deploy DER forecasting for renewable-heavy feeders to enhance grid stability.",
        "Integrate cybersecurity analytics early to mitigate increasing attack surfaces.",
        "Use cloud-native MLOps platforms for scalable model management and compliance."
    ]
    
    logger.info("\nStrategic Recommendations:")
    for rec in recommendations:
        logger.info(f"- {rec}")


if __name__ == "__main__":
    df = simulate_ai_adoption()
    plot_kpi_trends(df)
    strategic_recommendations(df)
