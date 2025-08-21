"""
Chapter 18: Future Trends and Strategic Roadmap
Simulate AI adoption scenarios and project KPI improvements for utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_ai_adoption(years=10):
    """
    Simulate utility KPI improvements over time from AI adoption.
    """
    timeline = np.arange(2024, 2024+years)
    cost_savings = np.linspace(0, 50, years) + np.random.normal(0, 3, years)   # % O&M savings
    outage_reduction = np.linspace(0, 40, years) + np.random.normal(0, 2, years)  # % outage reduction
    renewable_forecast_accuracy = np.linspace(70, 95, years)                     # % accuracy increase

    df = pd.DataFrame({
        "Year": timeline,
        "Cost_Savings_%": cost_savings,
        "Outage_Reduction_%": outage_reduction,
        "Forecast_Accuracy_%": renewable_forecast_accuracy
    })
    return df

def plot_kpi_trends(df):
    """
    Plot projected KPI improvements from AI adoption.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df["Year"], df["Cost_Savings_%"], label="O&M Cost Savings (%)", color="black")
    plt.plot(df["Year"], df["Outage_Reduction_%"], label="Outage Reduction (%)", color="gray")
    plt.plot(df["Year"], df["Forecast_Accuracy_%"], label="Renewable Forecast Accuracy (%)", color="darkblue")
    plt.xlabel("Year")
    plt.ylabel("Performance Metric (%)")
    plt.title("AI Adoption Impact on Utility KPIs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("chapter18_ai_kpi_trends.png")
    plt.show()

def strategic_recommendations(df):
    """
    Print roadmap recommendations based on KPI improvements.
    """
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
