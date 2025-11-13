"""Chapter 19: Cost Optimization and ROI for Utility ML Projects."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from datetime import datetime, timedelta

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

np.random.seed(config["model"]["random_state"])


def calculate_project_costs():
    """Calculate total project costs across lifecycle."""
    costs = {
        "development": {
            "data_preparation": config["costs"]["data_prep"],
            "model_development": config["costs"]["model_dev"],
            "infrastructure": config["costs"]["infrastructure"],
            "personnel": config["costs"]["personnel"],
        },
        "deployment": {
            "integration": config["costs"]["integration"],
            "training": config["costs"]["training"],
            "compliance": config["costs"]["compliance"],
        },
        "maintenance": {
            "monitoring": config["costs"]["monitoring_annual"],
            "retraining": config["costs"]["retraining_annual"],
            "support": config["costs"]["support_annual"],
        }
    }
    
    total_development = sum(costs["development"].values())
    total_deployment = sum(costs["deployment"].values())
    total_maintenance_annual = sum(costs["maintenance"].values())
    
    return {
        "costs": costs,
        "total_development": total_development,
        "total_deployment": total_deployment,
        "total_maintenance_annual": total_maintenance_annual,
    }


def calculate_benefits():
    """Calculate annual benefits from ML project."""
    # Base metrics
    annual_outages = config["benefits"]["annual_outages"]
    outage_reduction_pct = config["benefits"]["outage_reduction_pct"]
    cost_per_outage_hour = config["benefits"]["cost_per_outage_hour"]
    avg_outage_hours = config["benefits"]["avg_outage_hours"]
    
    # Calculate avoided outages
    avoided_outages = annual_outages * (outage_reduction_pct / 100)
    avoided_hours = avoided_outages * avg_outage_hours
    
    # Calculate savings
    outage_savings = avoided_hours * cost_per_outage_hour
    
    # Other benefits
    o_m_savings = config["benefits"]["o_m_savings_annual"]
    capital_deferral = config["benefits"]["capital_deferral_annual"]
    
    total_annual_benefits = outage_savings + o_m_savings + capital_deferral
    
    return {
        "avoided_outages": avoided_outages,
        "avoided_hours": avoided_hours,
        "outage_savings": outage_savings,
        "o_m_savings": o_m_savings,
        "capital_deferral": capital_deferral,
        "total_annual_benefits": total_annual_benefits,
    }


def calculate_npv(cash_flows, discount_rate):
    """Calculate Net Present Value."""
    npv = 0
    for i, cf in enumerate(cash_flows):
        npv += cf / ((1 + discount_rate) ** i)
    return npv


def calculate_irr(cash_flows, tolerance=1e-6, max_iterations=100):
    """Calculate Internal Rate of Return using binary search."""
    def npv_at_rate(rate):
        return sum(cf / ((1 + rate) ** i) for i, cf in enumerate(cash_flows))
    
    # Initial bounds
    low = -0.99  # Can't have rate < -100%
    high = 10.0  # Upper bound (1000%)
    
    # Check if NPV is positive at high rate (no solution)
    if npv_at_rate(high) > 0:
        return None
    
    # Binary search for IRR
    for _ in range(max_iterations):
        mid = (low + high) / 2
        npv = npv_at_rate(mid)
        
        if abs(npv) < tolerance:
            return mid
        
        if npv > 0:
            low = mid
        else:
            high = mid
        
        if high - low < tolerance:
            break
    
    return (low + high) / 2


def calculate_payback_period(cash_flows):
    """Calculate payback period in months."""
    cumulative = 0
    for i, cf in enumerate(cash_flows):
        cumulative += cf
        if cumulative > 0:
            # Interpolate to get exact month
            if i == 0:
                return 0
            prev_cumulative = sum(cash_flows[:i])
            months = i - 1 + abs(prev_cumulative) / abs(cf)
            return months
    return None


def build_roi_model():
    """Build complete ROI model with costs and benefits over time."""
    cost_data = calculate_project_costs()
    benefit_data = calculate_benefits()
    
    years = config["roi"]["analysis_years"]
    discount_rate = config["roi"]["discount_rate"]
    
    # Build cash flow timeline
    cash_flows = []
    cost_timeline = []
    benefit_timeline = []
    cumulative_net = []
    
    # Year 0: Development and deployment costs
    year_0_costs = (cost_data["total_development"] + 
                   cost_data["total_deployment"])
    cash_flows.append(-year_0_costs)
    cost_timeline.append(year_0_costs)
    benefit_timeline.append(0)
    cumulative_net.append(-year_0_costs)
    
    # Years 1-N: Maintenance costs and benefits
    for year in range(1, years + 1):
        year_costs = cost_data["total_maintenance_annual"]
        year_benefits = benefit_data["total_annual_benefits"]
        net_cash_flow = year_benefits - year_costs
        
        cash_flows.append(net_cash_flow)
        cost_timeline.append(year_costs)
        benefit_timeline.append(year_benefits)
        cumulative_net.append(cumulative_net[-1] + net_cash_flow)
    
    # Calculate financial metrics
    npv = calculate_npv(cash_flows, discount_rate)
    irr = calculate_irr(cash_flows)
    payback_months = calculate_payback_period(cash_flows)
    
    return {
        "cash_flows": cash_flows,
        "cost_timeline": cost_timeline,
        "benefit_timeline": benefit_timeline,
        "cumulative_net": cumulative_net,
        "npv": npv,
        "irr": irr,
        "payback_months": payback_months,
        "cost_data": cost_data,
        "benefit_data": benefit_data,
    }


def plot_roi_analysis(roi_model):
    """Visualize ROI analysis."""
    years = config["roi"]["analysis_years"]
    time_points = list(range(years + 1))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config["plotting"]["figsize"])
    
    # Plot 1: Cumulative costs and benefits
    ax1.plot(time_points, 
             [sum(roi_model["cost_timeline"][:i+1]) for i in range(len(time_points))],
             label="Cumulative Costs", 
             color=config["plotting"]["colors"]["costs"],
             linewidth=2)
    ax1.plot(time_points,
             [sum(roi_model["benefit_timeline"][:i+1]) for i in range(len(time_points))],
             label="Cumulative Benefits",
             color=config["plotting"]["colors"]["benefits"],
             linewidth=2)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Cumulative Value ($)")
    ax1.set_title("ROI Timeline: Costs vs. Benefits")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Net cash flow
    ax2.bar(time_points, roi_model["cumulative_net"],
            color=[config["plotting"]["colors"]["positive"] if x > 0 
                   else config["plotting"]["colors"]["negative"] 
                   for x in roi_model["cumulative_net"]],
            alpha=0.7)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Cumulative Net Value ($)")
    ax2.set_title("Cumulative Net ROI")
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(config["plotting"]["output_file"])
    plt.close()


def print_roi_summary(roi_model):
    """Print ROI summary report."""
    print("\n" + "="*60)
    print("ROI ANALYSIS SUMMARY")
    print("="*60)
    
    print("\n--- COSTS ---")
    print(f"Development & Deployment: ${roi_model['cost_data']['total_development'] + roi_model['cost_data']['total_deployment']:,.0f}")
    print(f"Annual Maintenance: ${roi_model['cost_data']['total_maintenance_annual']:,.0f}")
    
    print("\n--- BENEFITS (Annual) ---")
    print(f"Outage Savings: ${roi_model['benefit_data']['outage_savings']:,.0f}")
    print(f"O&M Savings: ${roi_model['benefit_data']['o_m_savings']:,.0f}")
    print(f"Capital Deferral: ${roi_model['benefit_data']['capital_deferral']:,.0f}")
    print(f"Total Annual Benefits: ${roi_model['benefit_data']['total_annual_benefits']:,.0f}")
    
    print("\n--- FINANCIAL METRICS ---")
    print(f"NPV (10% discount): ${roi_model['npv']:,.0f}")
    if roi_model['irr']:
        print(f"IRR: {roi_model['irr']*100:.1f}%")
    if roi_model['payback_months']:
        print(f"Payback Period: {roi_model['payback_months']:.1f} months")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print("\n--- Building ROI Model ---")
    roi_model = build_roi_model()
    
    print_roi_summary(roi_model)
    
    print("\n--- Generating ROI Visualization ---")
    plot_roi_analysis(roi_model)
    print(f"Plot saved to: {config['plotting']['output_file']}")

