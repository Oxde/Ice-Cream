import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# Load data and recreate model (simplified version)
train_data = pd.read_csv('data/mmm_ready/consistent_channels_train_set.csv')
test_data = pd.read_csv('data/mmm_ready/consistent_channels_test_set.csv')

# Media columns
media_cols = [col for col in train_data.columns if 'cost' in col or 'spend' in col]

# Calculate baseline weekly spend
baseline_weekly_spend = {}
for col in media_cols:
    baseline_weekly_spend[col] = train_data[col].median()

# Transformation functions (simplified)
def apply_adstock(x, decay_rate):
    """Apply adstock transformation"""
    adstocked = np.zeros_like(x)
    adstocked[0] = x[0]
    for i in range(1, len(x)):
        adstocked[i] = x[i] + decay_rate * adstocked[i-1]
    return adstocked

def apply_saturation(x, transformation_type, exp=0.5):
    """Apply saturation transformation"""
    if transformation_type == 'linear':
        return x / 1000
    elif transformation_type == 'sqrt':
        return np.sqrt(x / 100)
    elif transformation_type == 'power':
        return np.power(x / 1000, exp)
    else:
        return x / 1000

# Parameters from model
ADSTOCK_PARAMS = {
    'radio_local_radio_local_cost': 0.847,
    'tv_branding_tv_branding_cost': 0.692,
    'search_cost': 0.123,
    'social_costs': 0.089,
    'tv_promo_tv_promo_cost': 0.389,
    'radio_national_radio_national_cost': 0.000,
    'ooh_ooh_spend': 0.000
}

SATURATION_PARAMS = {
    'tv_branding_tv_branding_cost': {'type': 'power', 'exp': 0.3},
    'tv_promo_tv_promo_cost': {'type': 'power', 'exp': 0.3},
    'search_cost': {'type': 'power', 'exp': 0.5},
    'social_costs': {'type': 'linear'},
    'radio_local_radio_local_cost': {'type': 'sqrt'},
    'radio_national_radio_national_cost': {'type': 'power', 'exp': 0.7},
    'ooh_ooh_spend': {'type': 'power', 'exp': 0.3}
}

def transform_media_channels(df, media_columns):
    """Apply transformations"""
    df_transformed = df.copy()
    
    for col in media_columns:
        if col in df.columns:
            # Apply adstock
            decay_rate = ADSTOCK_PARAMS.get(col, 0.0)
            if decay_rate > 0.05:
                adstocked = apply_adstock(df[col].values, decay_rate)
            else:
                adstocked = df[col].values
            
            # Apply saturation
            sat_params = SATURATION_PARAMS.get(col, {'type': 'linear'})
            transformed = apply_saturation(
                adstocked, 
                sat_params['type'], 
                sat_params.get('exp', 0.5)
            )
            
            df_transformed[f'{col}_transformed'] = transformed
            df_transformed = df_transformed.drop(columns=[col])
    
    return df_transformed

# Recreate model
train_final = transform_media_channels(train_data, media_cols)
feature_cols = [col for col in train_final.columns if col not in ['date', 'sales']]
X_train = train_final[feature_cols].fillna(0)
y_train = train_final['sales']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_scaled, y_train)

# Simulation function (simplified)
def run_simulation(scenario_name, budget_changes, color):
    """Run simulation and return results"""
    baseline_weeks = train_data.tail(52).copy()
    
    # Apply budget changes
    scenario_weeks = baseline_weeks.copy()
    for col in media_cols:
        if col in budget_changes:
            scenario_weeks[col] = budget_changes[col]
        else:
            scenario_weeks[col] = baseline_weeks[col]
    
    # Transform datasets
    baseline_transformed = transform_media_channels(baseline_weeks, media_cols)
    scenario_transformed = transform_media_channels(scenario_weeks, media_cols)
    
    # Ensure features exist
    for col in feature_cols:
        if col not in baseline_transformed.columns:
            baseline_transformed[col] = 0
        if col not in scenario_transformed.columns:
            scenario_transformed[col] = 0
    
    # Predict
    X_baseline = baseline_transformed[feature_cols].fillna(0)
    X_scenario = scenario_transformed[feature_cols].fillna(0)
    
    X_baseline_scaled = scaler.transform(X_baseline)
    X_scenario_scaled = scaler.transform(X_scenario)
    
    baseline_pred = ridge.predict(X_baseline_scaled)
    scenario_pred = ridge.predict(X_scenario_scaled)
    
    # Calculate results
    baseline_total = sum(baseline_pred)
    scenario_total = sum(scenario_pred)
    sales_lift = ((scenario_total - baseline_total) / baseline_total) * 100
    incremental_sales = scenario_total - baseline_total
    
    return {
        'name': scenario_name,
        'color': color,
        'baseline_pred': baseline_pred,
        'scenario_pred': scenario_pred,
        'sales_lift': sales_lift,
        'incremental_sales': incremental_sales,
        'budget_changes': budget_changes
    }

# Run scenarios
scenario1 = run_simulation(
    "Conservative",
    {
        'search_cost': baseline_weekly_spend['search_cost'] * 1.6,
        'social_costs': baseline_weekly_spend['social_costs'] * 1.4,
        'tv_branding_tv_branding_cost': baseline_weekly_spend['tv_branding_tv_branding_cost'] * 0.9
    },
    '#3498db'
)

scenario2 = run_simulation(
    "Aggressive",
    {
        'search_cost': baseline_weekly_spend['search_cost'] * 1.5,
        'social_costs': baseline_weekly_spend['social_costs'] * 1.4,
        'radio_local_radio_local_cost': baseline_weekly_spend['radio_local_radio_local_cost'] * 1.2,
        'tv_branding_tv_branding_cost': baseline_weekly_spend['tv_branding_tv_branding_cost'] * 0.7,
        'tv_promo_tv_promo_cost': baseline_weekly_spend['tv_promo_tv_promo_cost'] * 0.8,
        'radio_national_radio_national_cost': baseline_weekly_spend['radio_national_radio_national_cost'] * 0.5,
    },
    '#e74c3c'
)

scenario3 = run_simulation(
    "Balanced",
    {
        'search_cost': baseline_weekly_spend['search_cost'] * 1.4,
        'social_costs': baseline_weekly_spend['social_costs'] * 1.3,
        'radio_local_radio_local_cost': baseline_weekly_spend['radio_local_radio_local_cost'] * 1.1,
        'tv_branding_tv_branding_cost': baseline_weekly_spend['tv_branding_tv_branding_cost'] * 0.85
    },
    '#27ae60'
)

simulation_results = [scenario1, scenario2, scenario3]

# Create clean, professional plot
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Budget Optimization Scenario Analysis', fontsize=18, fontweight='bold', y=0.95)

# 1. Sales Performance Comparison (Left) - Keep as user likes it
weeks = range(1, 53)
baseline_sales = simulation_results[0]['baseline_pred']

# Plot baseline with improved styling
ax1.plot(weeks, baseline_sales, label='Current Performance', linewidth=3, linestyle='--', 
         color='#2c3e50', alpha=0.8)

# Plot scenarios with cleaner styling
for result in simulation_results:
    ax1.plot(weeks, result['scenario_pred'], label=f"{result['name']} Scenario", 
             linewidth=2.5, color=result['color'], alpha=0.9)

ax1.set_xlabel('Week', fontsize=14, fontweight='bold')
ax1.set_ylabel('Weekly Sales (€)', fontsize=14, fontweight='bold')
ax1.set_title('Sales Performance Comparison', fontsize=16, fontweight='bold', pad=20)
ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x/1000:.0f}K'))

# 2. Scenario Results Summary (Right) - Clean bar chart
scenarios = [r['name'] for r in simulation_results]
sales_lifts = [r['sales_lift'] for r in simulation_results]
incremental = [r['incremental_sales']/1000 for r in simulation_results]
colors = [r['color'] for r in simulation_results]

bars = ax2.bar(range(len(scenarios)), sales_lifts, color=colors, alpha=0.8, 
               edgecolor='white', linewidth=2)
ax2.set_title('Expected Sales Lift by Scenario', fontsize=16, fontweight='bold', pad=20)
ax2.set_ylabel('Sales Lift (%)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Optimization Scenario', fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(scenarios)))
ax2.set_xticklabels(scenarios, fontsize=12)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, max(sales_lifts) * 1.2)

# Add clean value labels
for bar, lift, inc in zip(bars, sales_lifts, incremental):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{lift:.1f}%\n€{inc:.0f}K', ha='center', va='bottom', 
             fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/section_08_budget_simulation_clean.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

print("✅ Clean budget simulation plot created successfully!")
print("📊 Generated: plots/section_08_budget_simulation_clean.png") 