"""Chapter 20: Future Trends and Strategic Roadmap."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)


def simulate_ai_adoption():
    """Simulate utility KPI improvements over time from AI adoption."""
    years = config["simulation"]["years"]
    timeline = np.arange(config["simulation"]["start_year"], 
                        config["simulation"]["start_year"] + years)
    cost_savings = np.linspace(0, 50, years) + np.random.normal(0, 3, years)
    outage_reduction = np.linspace(0, 40, years) + np.random.normal(0, 2, years)
    renewable_forecast_accuracy = np.linspace(70, 95, years)

    return pd.DataFrame({
        "Year": timeline,
        "Cost_Savings_%": cost_savings,
        "Outage_Reduction_%": outage_reduction,
        "Forecast_Accuracy_%": renewable_forecast_accuracy
    })


def plot_kpi_trends(df):
    """Plot projected KPI improvements from AI adoption."""
    fig, ax = plt.subplots(figsize=config["plotting"]["figsize"])
    ax.plot(df["Year"], df["Cost_Savings_%"], 
             label="O&M Cost Savings (%)", color=config["plotting"]["colors"]["savings"])
    ax.plot(df["Year"], df["Outage_Reduction_%"], 
             label="Outage Reduction (%)", color=config["plotting"]["colors"]["outage"])
    ax.plot(df["Year"], df["Forecast_Accuracy_%"], 
             label="Renewable Forecast Accuracy (%)", 
             color=config["plotting"]["colors"]["forecast"])
    ax.set_title("Performance Metrics (%) by Year - AI Adoption Impact on Utility KPIs")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.savefig(config["plotting"]["output_file"])
    plt.close()


def strategic_recommendations(df):
    """Print roadmap recommendations based on KPI improvements."""
    final_savings = df["Cost_Savings_%"].iloc[-1]
    final_outage_reduction = df["Outage_Reduction_%"].iloc[-1]
    print(f"\nProjected O&M Savings in {df['Year'].iloc[-1]}: {final_savings:.1f}%")
    print(f"Projected Outage Reduction in {df['Year'].iloc[-1]}: {final_outage_reduction:.1f}%")
    print("\nStrategic Recommendations:")
    print("- Prioritize predictive maintenance to accelerate O&M cost savings.")
    print("- Deploy DER forecasting for renewable-heavy feeders to enhance grid stability.")
    print("- Integrate cybersecurity analytics early to mitigate increasing attack surfaces.")
    print("- Use cloud-native MLOps platforms for scalable model management and compliance.")


if __name__ == "__main__":
    df = simulate_ai_adoption()
    plot_kpi_trends(df)
    strategic_recommendations(df)
