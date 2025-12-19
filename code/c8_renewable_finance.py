"""Chapter 8: Financial Modeling for Renewable Energy Projects.
Based on NREL SAM (System Advisor Model) financial equations."""

import logging
import numpy as np
import pandas as pd
import signalplot as sp
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

sp.apply()

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

np.random.seed(config["model"]["random_state"])


def npv(rate, cashflows):
    """Calculate Net Present Value."""
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))


def irr(cashflows, guess=0.1):
    """Calculate Internal Rate of Return using Newton's method."""
    try:
        from scipy.optimize import fsolve
        def npv_func(r):
            return sum(cf / ((1 + r) ** t) for t, cf in enumerate(cashflows))
        result = fsolve(npv_func, guess)
        return float(result[0]) if result[0] > -1 else None
    except:
        # Simple approximation if scipy not available
        return None


def lcoe(costs, energy, discount_rate):
    """Calculate Levelized Cost of Energy."""
    costs = np.array(costs)
    energy = np.array(energy)
    discount_factors = (1 + discount_rate) ** np.arange(len(costs))
    pv_costs = np.sum(costs / discount_factors)
    pv_energy = np.sum(energy / discount_factors)
    return pv_costs / pv_energy if pv_energy > 0 else float('inf')


def macrs_depreciation(class_years, analysis_years, depreciable_basis):
    """Calculate MACRS depreciation schedule."""
    # MACRS percentages (200% DB, half-year convention)
    macrs_tables = {
        5: [0.20, 0.32, 0.192, 0.1152, 0.1152, 0.0576],
        7: [0.1429, 0.2449, 0.1749, 0.1249, 0.0893, 0.0892, 0.0893, 0.0446],
        15: [0.05, 0.095, 0.0855, 0.077, 0.0693, 0.0623, 0.059, 0.059, 0.059, 0.059,
             0.059, 0.059, 0.059, 0.059, 0.059, 0.0295],
    }
    
    table = macrs_tables.get(class_years, [])
    depreciation = np.zeros(analysis_years + 1)
    
    for i, pct in enumerate(table, start=1):
        if i > analysis_years:
            break
        depreciation[i] = depreciable_basis * pct
    
    return depreciation.tolist()


def compute_ppa_single_owner(capex, energy_y1, ppa_price_y1, ppa_escalation,
                             fixed_om, var_om_per_kwh, degradation, discount_rate,
                             tax_rate, itc_fraction, macrs_years, analysis_years):
    """
    Compute cash flows for a PPA Single Owner project.
    
    This is a simplified version based on NREL SAM financial models.
    """
    # Initialize arrays (year 0 at index 0)
    energy = [0.0] * (analysis_years + 1)
    ppa_price = [0.0] * (analysis_years + 1)
    revenue = [0.0] * (analysis_years + 1)
    fixed_om_costs = [0.0] * (analysis_years + 1)
    var_om_costs = [0.0] * (analysis_years + 1)
    depreciation = [0.0] * (analysis_years + 1)
    taxes = [0.0] * (analysis_years + 1)
    project_cf = [0.0] * (analysis_years + 1)
    
    # ITC reduces depreciable basis
    itc_value = capex * itc_fraction
    depreciable_basis = capex - (0.5 * itc_value)  # Standard ITC basis reduction
    
    # MACRS depreciation
    depreciation = macrs_depreciation(macrs_years, analysis_years, depreciable_basis)
    
    # Year 0: Initial investment
    project_cf[0] = -capex
    
    # Years 1-N: Operations
    for y in range(1, analysis_years + 1):
        # Energy with degradation
        if y == 1:
            energy[y] = energy_y1
            ppa_price[y] = ppa_price_y1
        else:
            energy[y] = energy[y - 1] * (1 - degradation)
            ppa_price[y] = ppa_price[y - 1] * (1 + ppa_escalation)
        
        # Revenue
        revenue[y] = energy[y] * ppa_price[y]
        
        # O&M costs
        fixed_om_costs[y] = fixed_om * ((1 + 0.02) ** (y - 1))  # 2% escalation
        var_om_costs[y] = var_om_per_kwh * energy[y]
        
        # Taxable income
        ebitda = revenue[y] - fixed_om_costs[y] - var_om_costs[y]
        ebt = ebitda - depreciation[y]
        taxes[y] = max(0.0, ebt * tax_rate)  # No negative tax benefit
        
        # Project cash flow
        project_cf[y] = ebitda - taxes[y]
    
    # Year 0 ITC (if applicable)
    if itc_value > 0:
        project_cf[0] += itc_value
    
    return {
        'energy': energy,
        'revenue': revenue,
        'project_cf': project_cf,
        'depreciation': depreciation,
        'taxes': taxes
    }


def analyze_project_economics():
    """Analyze a sample solar project using financial modeling."""
    
    # Project parameters (typical 10MW solar project)
    capex = 10_000_000  # $10M for 10MW = $1/W
    energy_y1 = 15_000_000  # 15 GWh first year (1.5 capacity factor)
    ppa_price_y1 = 0.05  # $0.05/kWh
    ppa_escalation = 0.02  # 2% annual escalation
    fixed_om = 50_000  # $50k/year
    var_om_per_kwh = 0.002  # $0.002/kWh
    degradation = 0.005  # 0.5% annual degradation
    discount_rate = 0.07  # 7% discount rate
    tax_rate = 0.21  # 21% federal tax rate
    itc_fraction = 0.30  # 30% ITC
    macrs_years = 5  # 5-year MACRS
    analysis_years = 25
    
    logger.info(f"Project: 10MW, CAPEX=${capex:,.0f}, Energy={energy_y1/1e6:.1f}GWh, PPA=${ppa_price_y1:.3f}/kWh, ITC={itc_fraction*100:.0f}%")
    
    # Compute cash flows
    results = compute_ppa_single_owner(
        capex, energy_y1, ppa_price_y1, ppa_escalation,
        fixed_om, var_om_per_kwh, degradation, discount_rate,
        tax_rate, itc_fraction, macrs_years, analysis_years
    )
    
    # Calculate metrics
    npv_value = npv(discount_rate, results['project_cf'])
    irr_value = irr(results['project_cf'])
    
    # LCOE calculation
    costs = [-capex + (capex * itc_fraction)] + [
        results['revenue'][y] - results['project_cf'][y]  # O&M + taxes
        for y in range(1, analysis_years + 1)
    ]
    energy_list = results['energy']
    lcoe_value = lcoe(costs, energy_list, discount_rate)
    
    logger.info(f"Metrics: NPV=${npv_value:,.0f}, IRR={irr_value*100:.2f}%" if irr_value else f"Metrics: NPV=${npv_value:,.0f}, LCOE=${lcoe_value:.3f}/kWh")
    
    # Cash flow summary
    total_revenue = sum(results['revenue'])
    total_cf = sum(results['project_cf'])
    cumulative_cf = np.cumsum(results['project_cf'])
    positive_idx = np.where(cumulative_cf >= 0)[0]
    payback = positive_idx[0] if positive_idx.size > 0 else 'N/A'
    logger.info(f"Cash flow: Revenue=${total_revenue:,.0f}, CF=${total_cf:,.0f}, Payback={payback}yr")
    
    return results


def visualize_cash_flows(results, analysis_years):
    """Visualize project cash flows."""
    
    fig, axes = sp.figure(nrows=2, ncols=2, figsize=(16, 10))
    
    years = range(analysis_years + 1)
    
    # Plot 1: Revenue and O&M
    ax1 = axes[0, 0]
    revenue = np.array(results['revenue']) / 1e6
    om = (np.array(results['revenue']) - np.array(results['project_cf'])) / 1e6
    
    ax1.plot(years, revenue, 'o-', linewidth=2, markersize=4, label='Revenue', color='#2ecc71')
    ax1.plot(years, om, 's-', linewidth=2, markersize=4, label='O&M + Taxes', color='#e74c3c')
    ax1.set_xlabel('Year', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Annual Cash Flow ($M)', fontweight='bold', fontsize=11)
    ax1.set_title('Revenue and Operating Costs', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Project Cash Flow
    ax2 = axes[0, 1]
    project_cf = np.array(results['project_cf']) / 1e6
    cumulative_cf = np.cumsum(project_cf)
    
    ax2_twin = ax2.twinx()
    ax2.bar(years, project_cf, alpha=0.7, color='#3498db', label='Annual CF')
    ax2_twin.plot(years, cumulative_cf, 'o-', linewidth=2, markersize=4, 
                  color='#e74c3c', label='Cumulative CF')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Year', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Annual Cash Flow ($M)', fontweight='bold', fontsize=11, color='#3498db')
    ax2_twin.set_ylabel('Cumulative Cash Flow ($M)', fontweight='bold', fontsize=11, color='#e74c3c')
    ax2.set_title('Project Cash Flows', fontweight='bold', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2_twin.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2_twin.spines['top'].set_visible(False)
    ax2_twin.spines['right'].set_visible(False)
    
    # Plot 3: Energy and Price
    ax3 = axes[1, 0]
    energy = np.array(results['energy']) / 1e6
    revenue_arr = np.array(results['revenue'])
    energy_arr = np.array(results['energy'])
    price = np.concatenate([[0], np.where(energy_arr[1:] > 0, revenue_arr[1:] / energy_arr[1:], 0) * 1000])
    
    ax3_twin = ax3.twinx()
    ax3.plot(years, energy, 'o-', linewidth=2, markersize=4, label='Energy', color='#f39c12')
    ax3_twin.plot(years, price, 's-', linewidth=2, markersize=4, label='PPA Price', color='#9b59b6')
    ax3.set_xlabel('Year', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Energy (GWh)', fontweight='bold', fontsize=11, color='#f39c12')
    ax3_twin.set_ylabel('PPA Price ($/MWh)', fontweight='bold', fontsize=11, color='#9b59b6')
    ax3.set_title('Energy Production and PPA Price', fontweight='bold', fontsize=12)
    ax3.legend(loc='upper left', fontsize=10)
    ax3_twin.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3_twin.spines['top'].set_visible(False)
    ax3_twin.spines['right'].set_visible(False)
    
    # Plot 4: Depreciation Schedule
    ax4 = axes[1, 1]
    depreciation = np.array(results['depreciation']) / 1e6
    
    ax4.bar(years, depreciation, alpha=0.7, color='#16a085', edgecolor='black')
    ax4.set_xlabel('Year', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Depreciation ($M)', fontweight='bold', fontsize=11)
    ax4.set_title('MACRS Depreciation Schedule', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    for ax in axes.flat:
        sp.tidy_axes(ax)
    
    output_path = config["plotting"]["output_files"]["renewable_finance"]
    sp.savefig(output_path)
    logger.debug(f"Saved: {output_path}")


def main():
    """Main execution."""
    results = analyze_project_economics()
    visualize_cash_flows(results, 25)
    
    
    return results


if __name__ == '__main__':
    results = main()

