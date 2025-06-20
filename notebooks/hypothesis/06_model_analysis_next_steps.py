# %% [markdown]
# # Model Analysis & Next Steps Strategy
# 
# **Current Status**: Respectful MMM with 46.7% Test R² (vs 45.1% original)
# **Goal**: Analyze performance and identify highest-impact improvement opportunities
# 
# **Analysis Framework:**
# 1. **Model Diagnostics** - What's working/not working
# 2. **Data Deep Dive** - Hidden patterns and opportunities  
# 3. **Business Logic Check** - Validate findings against business sense
# 4. **Next Steps Roadmap** - Prioritized improvement strategies

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("🔍 MMM MODEL ANALYSIS & NEXT STEPS")
print("=" * 50)
print("📊 Analyzing current model performance and improvement opportunities")

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (20, 15)

# %%
# Load our enhanced model results
print(f"\n📁 LOADING DATA & RECREATING MODEL")
print("=" * 40)

train_data = pd.read_csv('../data/mmm_ready/consistent_channels_train_set.csv')
test_data = pd.read_csv('../data/mmm_ready/consistent_channels_test_set.csv')

train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

# Recreate the enhanced model quickly
media_channels = [
    'search_cost', 'tv_branding_tv_branding_cost', 'social_costs',
    'ooh_ooh_spend', 'radio_national_radio_national_cost',
    'radio_local_radio_local_cost', 'tv_promo_tv_promo_cost'
]

control_variables = [
    'month_sin', 'month_cos', 'week_sin', 'week_cos', 'holiday_period',
    'weather_temperature_mean', 'weather_sunshine_duration', 'promo_promotion_type'
]

# Apply transformations (simplified version)
def apply_adstock_simple(x, decay_rate=0.4):
    adstocked = np.zeros_like(x)
    adstocked[0] = x[0] if not np.isnan(x[0]) else 0
    for i in range(1, len(x)):
        current_spend = x[i] if not np.isnan(x[i]) else 0
        adstocked[i] = current_spend + decay_rate * adstocked[i-1]
    return adstocked

# Quick model recreation
train_clean = train_data.fillna(0)
test_clean = test_data.fillna(0)

# Add adstock
for ch in media_channels:
    train_clean[f"{ch}_adstock"] = apply_adstock_simple(train_clean[ch].values)
    test_clean[f"{ch}_adstock"] = apply_adstock_simple(test_clean[ch].values)

# Prepare features
features = [f"{ch}_adstock" for ch in media_channels] + control_variables
X_train = train_clean[features]
X_test = test_clean[features]
y_train = train_clean['sales']
y_test = test_clean['sales']

# Fit model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Ridge(alpha=50.0)
model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

current_test_r2 = r2_score(y_test, y_test_pred)
current_gap = r2_score(y_train, y_train_pred) - current_test_r2

print(f"✅ Model recreated: Test R² = {current_test_r2:.3f} ({current_test_r2*100:.1f}%)")

# %%
# ANALYSIS 1: Model Diagnostics - Where are we losing performance?
print(f"\n🩺 ANALYSIS 1: MODEL DIAGNOSTICS")
print("=" * 40)

# 1. Residual Analysis
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

print(f"📊 Residual Analysis:")
print(f"   Training RMSE: ${np.sqrt(np.mean(residuals_train**2)):,.0f}")
print(f"   Test RMSE: ${np.sqrt(np.mean(residuals_test**2)):,.0f}")
print(f"   Test MAE: ${np.mean(np.abs(residuals_test)):,.0f}")

# 2. Prediction ranges
print(f"\n📊 Prediction Quality:")
print(f"   Actual sales range: ${y_test.min():,.0f} - ${y_test.max():,.0f}")
print(f"   Predicted range: ${y_test_pred.min():,.0f} - ${y_test_pred.max():,.0f}")

# 3. Temporal performance - how does accuracy change over time?
test_dates = test_clean['date'].values
test_errors = np.abs(residuals_test)

# Group by month to see temporal patterns
test_df_analysis = pd.DataFrame({
    'date': test_dates,
    'actual': y_test.values,
    'predicted': y_test_pred,
    'error': test_errors,
    'month': pd.to_datetime(test_dates).month
})

monthly_performance = test_df_analysis.groupby('month').agg({
    'error': ['mean', 'std'],
    'actual': 'mean'
}).round(0)

print(f"\n📊 Monthly Prediction Accuracy (Test Period):")
print(f"   Month | Avg Error | Std Error | Avg Sales")
for month in sorted(test_df_analysis['month'].unique()):
    month_data = test_df_analysis[test_df_analysis['month'] == month]
    avg_error = month_data['error'].mean()
    std_error = month_data['error'].std()
    avg_sales = month_data['actual'].mean()
    month_name = pd.to_datetime(f'2024-{month:02d}-01').strftime('%B')
    print(f"   {month_name[:3]:<3} | ${avg_error:>8,.0f} | ${std_error:>8,.0f} | ${avg_sales:>8,.0f}")

# %%
# ANALYSIS 2: Data Deep Dive - What patterns are we missing?
print(f"\n🔍 ANALYSIS 2: DATA DEEP DIVE")
print("=" * 35)

# 1. Correlation analysis - are we missing relationships?
print(f"📊 Media Channel Correlations with Sales:")
correlations = {}
for ch in media_channels:
    corr_original, p_val = pearsonr(train_clean[ch], train_clean['sales'])
    corr_adstock, _ = pearsonr(train_clean[f"{ch}_adstock"], train_clean['sales'])
    
    correlations[ch] = {
        'original_corr': corr_original,
        'adstock_corr': corr_adstock,
        'improvement': corr_adstock - corr_original
    }
    
    print(f"   {ch}:")
    print(f"      Original: {corr_original:.3f} → Adstock: {corr_adstock:.3f} (Δ{corr_adstock-corr_original:+.3f})")

# 2. Spending pattern analysis
print(f"\n💰 Spending Pattern Analysis:")
print(f"   Channel | Consistency | Volatility | Trend")
for ch in media_channels:
    spend_data = train_clean[ch]
    consistency = (spend_data > 0).mean()  # % of weeks with spend
    volatility = spend_data.std() / spend_data.mean() if spend_data.mean() > 0 else 0
    
    # Simple trend: correlation with time
    time_trend, _ = pearsonr(range(len(spend_data)), spend_data)
    
    trend_desc = "📈 Up" if time_trend > 0.1 else "📉 Down" if time_trend < -0.1 else "➡️ Flat"
    
    print(f"   {ch[:20]:<20} | {consistency:.1%} | {volatility:.2f} | {trend_desc}")

# 3. Identify potential interaction opportunities
print(f"\n🤝 Potential Channel Interactions (Strong Correlations):")
interaction_threshold = 0.3
potential_interactions = []

for i, ch1 in enumerate(media_channels):
    for ch2 in media_channels[i+1:]:
        corr, _ = pearsonr(train_clean[ch1], train_clean[ch2])
        if abs(corr) > interaction_threshold:
            potential_interactions.append((ch1, ch2, corr))
            print(f"   {ch1} ↔ {ch2}: {corr:.3f}")

if not potential_interactions:
    print(f"   No strong correlations (>{interaction_threshold}) found between channels")

# %%
# ANALYSIS 3: Feature Importance Deep Dive
print(f"\n🏆 ANALYSIS 3: FEATURE IMPORTANCE DEEP DIVE")
print("=" * 50)

# Get feature importance
coefficients = model.coef_
feature_names = features

# Media vs Control importance
media_importance = []
control_importance = []

for i, feature in enumerate(feature_names):
    abs_coef = abs(coefficients[i])
    if any(ch in feature for ch in media_channels):
        media_importance.append(abs_coef)
    else:
        control_importance.append(abs_coef)

print(f"📊 Feature Category Analysis:")
print(f"   Media channels avg importance: {np.mean(media_importance):.2e}")
print(f"   Control variables avg importance: {np.mean(control_importance):.2e}")
print(f"   Media/Control ratio: {np.mean(media_importance)/np.mean(control_importance):.2f}")

# Individual channel analysis
print(f"\n💰 Individual Channel Performance:")
for ch in media_channels:
    ch_feature = f"{ch}_adstock"
    if ch_feature in feature_names:
        idx = feature_names.index(ch_feature)
        coef = coefficients[idx]
        abs_coef = abs(coef)
        
        avg_spend = train_clean[ch].mean()
        
        # Estimate incremental impact
        # If we increase spend by $1000, what's the sales impact?
        impact_per_1k = coef * 1000
        
        direction = "📈 Positive" if coef > 0 else "📉 Negative"
        
        print(f"   {ch}:")
        print(f"      Coefficient: {coef:.2e} ({direction})")
        print(f"      Impact of +$1K spend: ${impact_per_1k:,.0f} sales")
        print(f"      Current avg spend: ${avg_spend:,.0f}/week")

# %%
# ANALYSIS 4: Identify Specific Improvement Opportunities
print(f"\n🚀 ANALYSIS 4: IMPROVEMENT OPPORTUNITIES")
print("=" * 45)

improvement_opportunities = []

# 1. Temporal Effects
train_clean['time_trend'] = range(len(train_clean))
time_sales_corr, _ = pearsonr(train_clean['time_trend'], train_clean['sales'])

if abs(time_sales_corr) > 0.1:
    improvement_opportunities.append({
        'type': 'Temporal Trend',
        'description': f'Strong time trend detected ({time_sales_corr:.3f})',
        'priority': 'High',
        'effort': 'Low'
    })

# 2. Lagged Effects
print(f"🔄 Testing Lagged Effects:")
best_lags = {}
for ch in media_channels:
    best_lag = 0
    best_corr = abs(correlations[ch]['adstock_corr'])
    
    # Test 1-4 week lags
    for lag in range(1, 5):
        if lag < len(train_clean):
            lagged_spend = train_clean[ch].shift(lag)
            if not lagged_spend.isnull().all():
                lagged_corr, _ = pearsonr(lagged_spend.dropna(), 
                                        train_clean['sales'][lag:])
                if abs(lagged_corr) > best_corr:
                    best_corr = abs(lagged_corr)
                    best_lag = lag
    
    best_lags[ch] = best_lag
    if best_lag > 0:
        print(f"   {ch}: Best lag = {best_lag} weeks (corr: {best_corr:.3f})")
        improvement_opportunities.append({
            'type': 'Lagged Effects',
            'description': f'{ch} shows {best_lag}-week delay effect',
            'priority': 'Medium',
            'effort': 'Low'
        })

# 3. Non-linear relationships
print(f"\n📈 Testing Non-linear Relationships:")
for ch in media_channels:
    spend_data = train_clean[ch]
    
    # Test if sqrt or log transformation improves correlation
    if spend_data.min() >= 0:
        sqrt_data = np.sqrt(spend_data)
        sqrt_corr, _ = pearsonr(sqrt_data, train_clean['sales'])
        
        if abs(sqrt_corr) > abs(correlations[ch]['original_corr']) + 0.05:
            print(f"   {ch}: √ transformation improves correlation")
            improvement_opportunities.append({
                'type': 'Non-linear Transform',
                'description': f'{ch} benefits from sqrt transformation',
                'priority': 'Medium',
                'effort': 'Low'
            })

# 4. External factors
external_factors = ['weather_temperature_mean', 'weather_sunshine_duration']
for factor in external_factors:
    if factor in train_clean.columns:
        factor_corr, _ = pearsonr(train_clean[factor], train_clean['sales'])
        if abs(factor_corr) > 0.2:
            improvement_opportunities.append({
                'type': 'External Amplifier',
                'description': f'{factor} has strong correlation ({factor_corr:.3f})',
                'priority': 'Low',
                'effort': 'Medium'
            })

# %%
# Next Steps Strategy
print(f"\n🎯 NEXT STEPS STRATEGY - PRIORITIZED ROADMAP")
print("=" * 55)

print(f"📊 CURRENT MODEL STATUS:")
print(f"   • Test R²: {current_test_r2:.1%} (Target: >55%)")
print(f"   • Overfitting Gap: {current_gap:.1%} (Target: <10%)")
print(f"   • Business Logic: Mostly sound (TV Branding anomaly)")

print(f"\n🚀 IMPROVEMENT ROADMAP (Effort vs Impact):")

# High Impact, Low Effort (Quick Wins)
print(f"\n🟢 PHASE 1: QUICK WINS (Next 2-3 Days)")
quick_wins = [op for op in improvement_opportunities if op['priority'] in ['High'] or (op['priority'] == 'Medium' and op['effort'] == 'Low')]
if quick_wins:
    for i, opp in enumerate(quick_wins, 1):
        print(f"   {i}. {opp['type']}: {opp['description']}")
else:
    print(f"   1. Advanced Lag Testing: Test 1-6 week lags for all channels")
    print(f"   2. Non-linear Transforms: Test sqrt, log transforms")
    print(f"   3. Rolling Averages: 4-week, 8-week media averages")

print(f"\n🟡 PHASE 2: MEDIUM EFFORT IMPROVEMENTS (Next 1-2 Weeks)")
print(f"   1. Hierarchical Modeling: Different coefficients per season")
print(f"   2. Channel Interaction Matrix: Test all pairwise interactions")
print(f"   3. Competitive Pressure: External market factors")
print(f"   4. Advanced Adstock: Gamma function, flexible decay shapes")

print(f"\n🔴 PHASE 3: MAJOR ENHANCEMENTS (Next Month)")
print(f"   1. Bayesian MMM: PyMC3/Stan implementation")
print(f"   2. Neural Network MMM: Deep learning approach")
print(f"   3. Time-Varying Coefficients: Dynamic parameter modeling")
print(f"   4. Geo-level Modeling: If location data available")

# %%
# Specific Recommendations
print(f"\n💡 SPECIFIC RECOMMENDATIONS FOR NEXT MODEL")
print("=" * 50)

print(f"🎯 IMMEDIATE ACTIONS:")
print(f"   1. 🔄 Test 2-4 week lags for TV Branding (investigate negative coefficient)")
print(f"   2. 📊 Add rolling 4-week averages for all media channels")
print(f"   3. 🧮 Test polynomial features (spend², spend³) for high-spend channels")
print(f"   4. 📈 Add time trend and seasonal adjustments")

print(f"\n🔧 CODE IMPLEMENTATION PRIORITY:")
print(f"   1. Enhanced lag testing function")
print(f"   2. Polynomial feature engineering")
print(f"   3. Time-based feature engineering")
print(f"   4. Advanced validation framework")

print(f"\n📈 EXPECTED IMPROVEMENTS:")
print(f"   • Target Test R²: 50-55% (vs current 46.7%)")
print(f"   • Target Gap: <10% (vs current 11.9%)")
print(f"   • Better business interpretability")
print(f"   • More robust future predictions")

print(f"\n🎪 THE BIG QUESTION TO INVESTIGATE:")
print(f"   Why is TV Branding negative despite highest spend?")
print(f"   → Test delayed effects (2-8 weeks)")
print(f"   → Test interaction with TV Promo")
print(f"   → Test competitive pressure hypothesis")
print(f"   → Test brand equity vs sales distinction")

print(f"\n✅ NEXT SESSION GOAL:")
print(f"   Build an enhanced model targeting 50%+ Test R² with better business logic")
print(f"   Focus on the TV Branding mystery and lag effects") 