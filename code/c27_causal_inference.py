"""Chapter 27: Causal Inference for Policy and Program Evaluation."""

import logging
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize
import statsmodels.formula.api as smf
import signalplot as sp

sp.apply()

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

np.random.seed(config["model"]["random_state"])

EGRID_PATH = Path(__file__).parent.parent.parent / "egrid_all_plants_1996-2023.parquet"
USE_REAL_DATA = EGRID_PATH.exists()
TREATMENT_YEAR = config["data"]["treatment_year"]
TREATED_STATES = config["data"]["treated_states"]
RANDOM_STATE = config["model"]["random_state"]


def generate_synthetic_state_data():
    """Generate synthetic state-level emissions data for causal inference."""
    logger.info("Generating synthetic state-level data...")
    
    years = range(2010, 2024)
    states = ['CA', 'NY', 'MA', 'WA', 'OR', 'TX', 'FL', 'IL', 'PA', 'OH', 
              'GA', 'NC', 'MI', 'NJ', 'VA', 'AZ', 'TN', 'IN', 'MO', 'MD']
    
    data = []
    for year in years:
        for state in states:
            base_intensity = 0.5 + np.random.uniform(0, 0.3)
            time_trend = -0.01 * (year - 2010)
            treatment_effect = (-0.05 
                               if state in TREATED_STATES and year >= TREATMENT_YEAR 
                               else 0)
            noise = np.random.normal(0, 0.02)
            
            carbon_intensity = base_intensity + time_trend + treatment_effect + noise
            generation = np.random.uniform(50_000, 200_000)
            co2 = carbon_intensity * generation
            
            data.append({
                'year': year,
                'state': state,
                'generation': generation,
                'co2': co2,
                'carbon_intensity': carbon_intensity
            })
    
    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df):,} state-year observations")
    return df


def load_state_level_data():
    """Load state-level data from eGrid or generate synthetic."""
    if not USE_REAL_DATA:
        return generate_synthetic_state_data()
    
    logger.info("Loading eGrid data and aggregating to state level...")
    plants = pd.read_parquet(EGRID_PATH)
    
    state_col = [c for c in plants.columns 
                 if 'state' in c.lower() and 'abbr' in c.lower()][0]
    
    state_data = plants.groupby(['data_year', state_col]).agg({
        'Plant annual net generation (MWh)': lambda x: pd.to_numeric(x, errors='coerce').sum(),
        'Plant annual CO2 emissions (tons)': lambda x: pd.to_numeric(x, errors='coerce').sum(),
    }).reset_index()
    
    state_data.columns = ['year', 'state', 'generation', 'co2']
    state_data['carbon_intensity'] = state_data['co2'] / state_data['generation']
    state_data = state_data.replace([np.inf, -np.inf], np.nan).dropna()
    
    logger.info(f"Loaded {len(state_data):,} state-year observations")
    return state_data


def run_difference_in_differences(data, treated_states, treatment_year):
    """Estimate treatment effect using Difference-in-Differences."""
    logger.info("\n" + "=" * 80)
    logger.info("[1/3] DIFFERENCE-IN-DIFFERENCES")
    logger.info("=" * 80)
    
    df = data.copy()
    df['treated'] = df['state'].isin(treated_states).astype(int)
    df['post'] = (df['year'] >= treatment_year).astype(int)
    df['treat_post'] = df['treated'] * df['post']
    
    pre_data = df[df['year'] < treatment_year]
    logger.info("\nPre-treatment trends:")
    for group in [0, 1]:
        group_data = pre_data[pre_data['treated'] == group]
        if not len(group_data):
            continue
        trend = group_data.groupby('year')['carbon_intensity'].mean()
        if not len(trend) > 1:
            continue
        slope = np.polyfit(trend.index, trend.values, 1)[0]
        group_name = 'Treated' if group else 'Control'
        logger.info(f"  {group_name} states slope: {slope:.6f}")
    
    formula = 'carbon_intensity ~ treated + post + treat_post'
    model = smf.ols(formula, data=df).fit(cov_type='cluster', 
                                          cov_kwds={'groups': df['state']})
    
    logger.info("\nDiD Regression Results:")
    logger.info("-" * 80)
    logger.info(f"Treatment Effect: {model.params['treat_post']:.6f} tons/MWh")
    logger.info(f"Standard Error: {model.bse['treat_post']:.6f}")
    logger.info(f"T-statistic: {model.tvalues['treat_post']:.3f}")
    logger.info(f"P-value: {model.pvalues['treat_post']:.4f}")
    logger.info(f"95% CI: [{model.conf_int().loc['treat_post', 0]:.6f}, "
          f"{model.conf_int().loc['treat_post', 1]:.6f}]")
    
    pvalue = model.pvalues['treat_post']
    if pvalue < 0.05:
        direction = 'reduced' if model.params['treat_post'] < 0 else 'increased'
        pct = abs(model.params['treat_post']) / df['carbon_intensity'].mean() * 100
        logger.info(f"\n[OK] Policy significantly {direction} carbon intensity by {pct:.1f}%")
    else:
        logger.info("\n[FAIL] No significant policy effect detected")
    
    logger.info("\nEvent Study (dynamic effects):")
    df['years_to_treatment'] = df['year'] - treatment_year
    
    for year in range(-5, 6):
        if year != -1:
            df[f'treat_year_{year}'] = (
                df['treated'] * (df['years_to_treatment'] == year)
            ).astype(int)
    
    event_formula = 'carbon_intensity ~ treated + ' + ' + '.join([
        f'treat_year_{y}' for y in range(-5, 6) if y != -1
    ])
    
    event_model = smf.ols(event_formula, data=df).fit(cov_type='HC1')
    
    event_time = np.arange(-5, 6)
    coefficients = np.array([0 if year == -1 
                            else event_model.params.get(f'treat_year_{year}', 0)
                            for year in event_time])
    
    pre_coefs = coefficients[event_time < 0]
    if all(abs(c) < 0.05 for c in pre_coefs):
        logger.info("  [OK] Pre-trends look parallel (coefficients near zero)")
    else:
        logger.info("  [WARNING] Warning: Pre-trends may not be parallel")
    
    return {
        'model': model,
        'treatment_effect': model.params['treat_post'],
        'pvalue': model.pvalues['treat_post'],
        'event_study': {
            'time': event_time,
            'coefficients': coefficients
        }
    }


def synthetic_control_weights(treated_pre, control_pre):
    """Find optimal weights for synthetic control."""
    def objective(weights):
        synthetic = control_pre @ weights
        return np.sum((treated_pre - synthetic)**2)
    
    n_controls = control_pre.shape[1]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(n_controls)]
    initial = np.ones(n_controls) / n_controls
    
    result = minimize(objective, initial, method='SLSQP', 
                     bounds=bounds, constraints=constraints,
                     options={'maxiter': 1000})
    
    return result.x if result.success else initial


def run_synthetic_control(data, treated_state, treatment_year):
    """Estimate treatment effect using Synthetic Control Method."""
    logger.info("\n" + "=" * 80)
    logger.info(f"[2/3] SYNTHETIC CONTROL (Treated: {treated_state})")
    logger.info("=" * 80)
    
    treated_data = data[data['state'] == treated_state].sort_values('year')
    control_data = data[data['state'] != treated_state].sort_values(['state', 'year'])
    
    if not len(treated_data) > 0:
        logger.warning(f"  [FAIL] No data for treated state {treated_state}")
        return None
    
    treated_pre = treated_data[treated_data['year'] < treatment_year]['carbon_intensity'].values
    treated_post = treated_data[treated_data['year'] >= treatment_year]['carbon_intensity'].values
    
    if not (len(treated_pre) and len(treated_post)):
        logger.warning(f"  [FAIL] Insufficient data for {treated_state}")
        return None
    
    control_states = control_data['state'].unique()
    control_pre_matrix = []
    control_post_matrix = []
    valid_states = []
    
    for state in control_states:
        state_data = control_data[control_data['state'] == state]
        pre = state_data[state_data['year'] < treatment_year]['carbon_intensity'].values
        post = state_data[state_data['year'] >= treatment_year]['carbon_intensity'].values
        
        if len(pre) == len(treated_pre) and len(post) == len(treated_post):
            control_pre_matrix.append(pre)
            control_post_matrix.append(post)
            valid_states.append(state)
    
    if not valid_states:
        logger.warning("  [FAIL] No valid control states found")
        return None
    
    control_pre_matrix = np.array(control_pre_matrix).T
    control_post_matrix = np.array(control_post_matrix).T
    
    weights = synthetic_control_weights(treated_pre, control_pre_matrix)
    
    logger.info(f"\nSynthetic {treated_state} composed of:")
    top_contributors = sorted(zip(valid_states, weights), key=lambda x: x[1], reverse=True)
    for state, weight in top_contributors[:10]:
        if weight > 0.01:
            logger.info(f"  {state}: {weight*100:.1f}%")
    
    synthetic_pre = control_pre_matrix @ weights
    synthetic_post = control_post_matrix @ weights
    
    gap = treated_post - synthetic_post
    avg_effect = gap.mean()
    pre_rmse = np.sqrt(np.mean((treated_pre - synthetic_pre)**2))
    
    logger.info(f"\nPre-treatment fit RMSE: {pre_rmse:.6f}")
    logger.info(f"Average treatment effect: {avg_effect:.6f} tons/MWh")
    
    return {
        'weights': weights,
        'states': valid_states,
        'treated_pre': treated_pre,
        'treated_post': treated_post,
        'synthetic_pre': synthetic_pre,
        'synthetic_post': synthetic_post,
        'treatment_effect': avg_effect,
        'pre_rmse': pre_rmse
    }


def run_propensity_score_matching(data, treatment_year):
    """Estimate treatment effect using Propensity Score Matching."""
    logger.info("\n" + "=" * 80)
    logger.info("[3/3] PROPENSITY SCORE MATCHING")
    logger.info("=" * 80)
    
    pre_data = data[data['year'] < treatment_year].copy()
    
    state_features = pre_data.groupby('state').agg({
        'carbon_intensity': ['mean', 'std'],
        'generation': 'mean',
        'co2': 'mean'
    }).reset_index()
    
    state_features.columns = ['state', 'avg_carbon', 'std_carbon', 'avg_gen', 'avg_co2']
    state_features['treated'] = state_features['state'].isin(TREATED_STATES).astype(int)
    state_features = state_features.dropna()
    
    if len(state_features) < 10:
        logger.warning("  [FAIL] Insufficient states for PSM")
        return None
    
    feature_cols = ['avg_carbon', 'std_carbon', 'avg_gen', 'avg_co2']
    X = state_features[feature_cols]
    y = state_features['treated']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    ps_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    ps_model.fit(X_scaled, y)
    
    state_features['propensity_score'] = ps_model.predict_proba(X_scaled)[:, 1]
    
    logger.info("\nPropensity Score Distribution:")
    logger.info(state_features.groupby('treated')['propensity_score'].describe()[['mean', 'min', 'max']])
    
    treated = state_features[state_features['treated'] == 1]
    control = state_features[state_features['treated'] == 0]
    
    if not (len(treated) and len(control)):
        logger.warning("  [FAIL] Need both treated and control units")
        return None
    
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(control[['propensity_score']])
    
    distances, indices = nn.kneighbors(treated[['propensity_score']])
    
    matched_control = control.iloc[indices.flatten()]
    
    post_data = data[data['year'] >= treatment_year]
    
    treated_outcomes = []
    control_outcomes = []
    
    for _, treated_state in treated.iterrows():
        state = treated_state['state']
        outcome = post_data[post_data['state'] == state]['carbon_intensity'].mean()
        if not np.isnan(outcome):
            treated_outcomes.append(outcome)
    
    for _, control_state in matched_control.iterrows():
        state = control_state['state']
        outcome = post_data[post_data['state'] == state]['carbon_intensity'].mean()
        if not np.isnan(outcome):
            control_outcomes.append(outcome)
    
    if not (len(treated_outcomes) and len(control_outcomes)):
        logger.warning("  [FAIL] Could not compute outcomes")
        return None
    
    att = np.mean(treated_outcomes) - np.mean(control_outcomes)
    se = (np.std(np.array(treated_outcomes) - np.array(control_outcomes)) 
          / np.sqrt(len(treated_outcomes)))
    
    logger.info(f"\nAverage Treatment Effect on Treated (ATT): {att:.6f}")
    logger.info(f"Standard Error: {se:.6f}")
    logger.info(f"95% CI: [{att - 1.96*se:.6f}, {att + 1.96*se:.6f}]")
    
    if abs(att) / se > 1.96:
        logger.info("  [OK] Statistically significant at 5% level")
    else:
        logger.info("  [FAIL] Not statistically significant")
    
    return {
        'att': att,
        'se': se,
        'treated_outcomes': treated_outcomes,
        'control_outcomes': control_outcomes,
        'propensity_scores': state_features
    }


def visualize_results(data, did_results, sc_results, treatment_year, treated_states):
    """Create visualization of causal inference results."""
    logger.info("Generating visualizations...")
    
    fig, axes = sp.figure(nrows=2, ncols=2, figsize=(16, 10))
    
    ax1 = axes[0, 0]
    
    treated_trend = data[data['state'].isin(treated_states)].groupby('year')['carbon_intensity'].mean()
    control_trend = data[~data['state'].isin(treated_states)].groupby('year')['carbon_intensity'].mean()
    
    pre_mask = treated_trend.index < treatment_year
    post_mask = treated_trend.index >= treatment_year
    
    treated_pre = treated_trend[pre_mask]
    treated_post = treated_trend[post_mask]
    control_pre = control_trend[pre_mask]
    control_post = control_trend[post_mask]
    
    if len(treated_pre):
        ax1.plot(treated_pre.index, treated_pre.values, 
                'o-', linewidth=2, markersize=6, color='#e74c3c', label='Treated (pre)')
    if len(treated_post):
        ax1.plot(treated_post.index, treated_post.values, 
                'o-', linewidth=2, markersize=6, color='#e74c3c', label='Treated (post)')
    
    if len(control_pre):
        ax1.plot(control_pre.index, control_pre.values, 
                's-', linewidth=2, markersize=6, color='#3498db', label='Control (pre)')
    if len(control_post):
        ax1.plot(control_post.index, control_post.values, 
                's-', linewidth=2, markersize=6, color='#3498db', label='Control (post)')
    
    ax1.axvline(treatment_year - 0.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax1.set_xlabel('Year', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Carbon Intensity', fontweight='bold', fontsize=11)
    ax1.set_title('Difference-in-Differences', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax2 = axes[0, 1]
    
    event_time = did_results['event_study']['time']
    event_coefs = did_results['event_study']['coefficients']
    
    ax2.plot(event_time, event_coefs, 'o-', linewidth=2, markersize=7, color='#e74c3c')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.axvline(-0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax2.fill_between(event_time, -0.02, 0.02, alpha=0.2, color='green')
    
    ax2.set_xlabel('Years Relative to Treatment', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Treatment Effect', fontweight='bold', fontsize=11)
    ax2.set_title('Event Study', fontweight='bold', fontsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    ax3 = axes[1, 0]
    
    if sc_results:
        years_pre = range(treatment_year - len(sc_results['treated_pre']), treatment_year)
        years_post = range(treatment_year, treatment_year + len(sc_results['treated_post']))
        
        ax3.plot(list(years_pre), sc_results['treated_pre'], 
                'o-', linewidth=2, markersize=6, color='#e74c3c', label='Actual')
        ax3.plot(list(years_post), sc_results['treated_post'], 
                'o-', linewidth=2, markersize=6, color='#e74c3c')
        
        ax3.plot(list(years_pre), sc_results['synthetic_pre'], 
                's--', linewidth=2, markersize=5, color='gray', label='Synthetic', alpha=0.7)
        ax3.plot(list(years_post), sc_results['synthetic_post'], 
                's--', linewidth=2, markersize=5, color='gray', alpha=0.7)
        
        ax3.axvline(treatment_year - 0.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax3.set_xlabel('Year', fontweight='bold', fontsize=11)
        ax3.set_ylabel('Carbon Intensity', fontweight='bold', fontsize=11)
        ax3.set_title('Synthetic Control', fontweight='bold', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
    else:
        ax3.text(0.5, 0.5, 'Synthetic Control\nNot Available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.axis('off')
    
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    sc_effect = (f"{sc_results['treatment_effect']:.6f}" if sc_results else "Not computed")
    sc_rmse = (f"{sc_results['pre_rmse']:.6f}" if sc_results else "")
    did_effect = did_results['treatment_effect']
    did_pvalue = did_results['pvalue']
    interpretation = ("Policy reduced emissions" if did_effect < 0 else "Policy increased emissions")
    significance = ("significantly" if did_pvalue < 0.05 else "(not significant)")
    
    summary_text = f"""
CAUSAL INFERENCE SUMMARY

Treatment Year: {treatment_year}
Treated States: {', '.join(treated_states)}

DiD Estimate:
  Effect: {did_effect:.6f}
  P-value: {did_pvalue:.4f}
  Significant: {'Yes' if did_pvalue < 0.05 else 'No'}

Synthetic Control:
  Effect: {sc_effect}
  Pre-fit RMSE: {sc_rmse}

Interpretation:
  {interpretation} {significance}
    """
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    for ax in axes.flat:
        sp.tidy_axes(ax)
    
    output_path = config["plotting"]["output_files"]["causal_inference"]
    sp.savefig(output_path)
    logger.info(f"  Saved: {output_path}")


def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("CAUSAL INFERENCE - POLICY EVALUATION")
    logger.info("=" * 80)
    
    data = load_state_level_data()
    
    did_results = run_difference_in_differences(data, TREATED_STATES, TREATMENT_YEAR)
    sc_results = run_synthetic_control(data, TREATED_STATES[0], TREATMENT_YEAR)
    psm_results = run_propensity_score_matching(data, TREATMENT_YEAR)
    
    visualize_results(data, did_results, sc_results, TREATMENT_YEAR, TREATED_STATES)
    
    logger.info("\n" + "=" * 80)
    logger.info("[OK] Complete!")
    logger.info("=" * 80)
    
    return {
        'did': did_results,
        'synthetic_control': sc_results,
        'psm': psm_results
    }


if __name__ == '__main__':
    results = main()
