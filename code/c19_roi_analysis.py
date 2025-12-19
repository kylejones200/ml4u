"""Chapter 19: Cost Optimization and ROI for Utility ML Projects."""

import logging
import numpy as np
import pandas as pd
import signalplot as sp
import yaml
from pathlib import Path
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

sp.apply()

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
    cash_flows = np.array(cash_flows)
    discount_factors = (1 + discount_rate) ** np.arange(len(cash_flows))
    return np.sum(cash_flows / discount_factors)


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
    cumulative = np.cumsum(cash_flows)
    positive_idx = np.where(cumulative > 0)[0]
    if not positive_idx.size:
        return None
    i = positive_idx[0]
    if i == 0:
        return 0
    prev_cumulative = cumulative[i - 1]
    months = i - 1 + abs(prev_cumulative) / abs(cash_flows[i])
    return months


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
    year_costs = cost_data["total_maintenance_annual"]
    year_benefits = benefit_data["total_annual_benefits"]
    net_cash_flow = year_benefits - year_costs
    
    # Vectorized operations for years 1-N
    annual_cash_flows = np.full(years, net_cash_flow)
    annual_costs = np.full(years, year_costs)
    annual_benefits = np.full(years, year_benefits)
    
    cash_flows.extend(annual_cash_flows.tolist())
    cost_timeline.extend(annual_costs.tolist())
    benefit_timeline.extend(annual_benefits.tolist())
    
    # Calculate cumulative net using cumsum
    all_cash_flows = np.array(cash_flows)
    cumulative_net = np.cumsum(all_cash_flows).tolist()
    
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
    
    fig, (ax1, ax2) = sp.figure(ncols=2)
    
    # Plot 1: Cumulative costs and benefits
    cost_cumulative = np.cumsum(roi_model["cost_timeline"])
    benefit_cumulative = np.cumsum(roi_model["benefit_timeline"])
    ax1.plot(time_points, cost_cumulative)
    ax1.plot(time_points, benefit_cumulative)
    sp.style_line_plot(ax1)
    
    # Plot 2: Net cash flow
    ax2.bar(time_points, roi_model["cumulative_net"])
    sp.style_bar_plot(ax2)
    
    sp.savefig(config["plotting"]["roi_output_file"])


def print_roi_summary(roi_model):
    """Print ROI summary report."""
    dev_deploy = (
        roi_model['cost_data']['total_development'] +
        roi_model['cost_data']['total_deployment']
    )
    logger.info(f"Costs: Dev/Deploy=${dev_deploy:,.0f}, Annual=${roi_model['cost_data']['total_maintenance_annual']:,.0f}")
    logger.info(f"Benefits: Outage=${roi_model['benefit_data']['outage_savings']:,.0f}, O&M=${roi_model['benefit_data']['o_m_savings']:,.0f}, Total=${roi_model['benefit_data']['total_annual_benefits']:,.0f}")
    irr_str = f", IRR={roi_model['irr']*100:.1f}%" if roi_model['irr'] else ""
    payback_str = f", Payback={roi_model['payback_months']:.1f} months" if roi_model['payback_months'] else ""
    logger.info(f"NPV: ${roi_model['npv']:,.0f}{irr_str}{payback_str}")


if __name__ == "__main__":
    roi_model = build_roi_model()
    print_roi_summary(roi_model)
    plot_roi_analysis(roi_model)
    logger.info(f"Plot: {config['plotting']['roi_output_file']}")

