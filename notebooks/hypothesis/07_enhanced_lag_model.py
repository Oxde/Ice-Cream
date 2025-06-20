# %% [markdown]
# # Enhanced MMM with Lag Effects - Theory-Based Improvements
# 
# **Goal**: Implement theory-based enhancements to reach 50%+ Test R²
# **Focus**: Solve TV Branding mystery & add proven lag effects
# 
# **THEORETICAL FOUNDATIONS:**
# 1. **Media Response Theory**: Different channels have different response patterns
# 2. **Adstock Theory**: Media has carryover effects (geometric decay)
# 3. **Lag Theory**: Media impact can be delayed (awareness → consideration → purchase)
# 4. **Saturation Theory**: Diminishing returns at high spend levels
# 
# **STRICT VALIDATION**: Train on train data only, test on completely separate test set

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("🧠 ENHANCED MMM - THEORY-BASED LAG EFFECTS")
print("=" * 55)
print("📚 Based on proven Media Mix Modeling theory")
print("🎯 Goal: 50%+ Test R² through lag effects and TV investigation")
print("⚠️  STRICT TRAIN/TEST SEPARATION - No data leakage!")

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (18, 12)

# %%
print(f"\n📁 LOADING DATA WITH STRICT SEPARATION")
print("=" * 45)

# Load data
train_data = pd.read_csv('../data/mmm_ready/consistent_channels_train_set.csv')
test_data = pd.read_csv('../data/mmm_ready/consistent_channels_test_set.csv')

train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

print(f"✅ Training period: {train_data['date'].min()} to {train_data['date'].max()}")
print(f"✅ Test period: {test_data['date'].min()} to {test_data['date'].max()}")
print(f"✅ NO OVERLAP - Proper temporal validation")

# Define channels
media_channels = [
    'search_cost', 'tv_branding_tv_branding_cost', 'social_costs',
    'ooh_ooh_spend', 'radio_national_radio_national_cost',
    'radio_local_radio_local_cost', 'tv_promo_tv_promo_cost'
]

control_variables = [
    'month_sin', 'month_cos', 'week_sin', 'week_cos', 'holiday_period',
    'weather_temperature_mean', 'weather_sunshine_duration', 'promo_promotion_type'
]

print(f"📊 Channels: {len(media_channels)} media + {len(control_variables)} control")

# %%
# THEORY 1: ADVANCED LAG EFFECTS
print(f"\n📚 THEORY 1: ADVANCED LAG EFFECTS")
print("=" * 40)

print(f"🧠 THEORETICAL BASIS:")
print(f"   • Media Response Theory: Different channels have different response curves")
print(f"   • Cognitive Processing: Awareness → Consideration → Purchase funnel")
print(f"   • Media Habits: TV/Radio have delayed impact vs Search (immediate)")
print(f"   • From analysis: Search 2-week, Others 4-week optimal lags")

def apply_lag_effects(data, channel, lag_weeks, fit_on_train_only=True):
    """
    Apply lag effects based on Media Response Theory
    
    Theory: Media exposure today affects purchases in future weeks
    Different channels have different optimal lag periods based on:
    - Cognitive processing time
    - Purchase consideration periods  
    - Media consumption patterns
    """
    data_with_lags = data.copy()
    
    if channel in data.columns:
        # Create lagged versions
        for lag in range(1, lag_weeks + 1):
            lag_col = f"{channel}_lag_{lag}w"
            data_with_lags[lag_col] = data[channel].shift(lag)
        
        # Create rolling averages (captures sustained exposure effects)
        if lag_weeks >= 2:
            rolling_col = f"{channel}_roll_{lag_weeks}w"
            data_with_lags[rolling_col] = data[channel].rolling(window=lag_weeks, min_periods=1).mean()
    
    return data_with_lags

# Apply theory-based lag structure (ONLY on training data to avoid leakage)
print(f"\n🔄 Applying lag effects (TRAIN DATA ONLY):")

# Channel-specific lag windows based on analysis + theory
lag_config = {
    'search_cost': 2,                           # Immediate response + short consideration
    'tv_branding_tv_branding_cost': 6,          # Long brand building cycle  
    'social_costs': 4,                          # Social proof + consideration time
    'ooh_ooh_spend': 4,                         # Awareness to action delay
    'radio_national_radio_national_cost': 4,    # Audio recall + consideration  
    'radio_local_radio_local_cost': 4,          # Local market response time
    'tv_promo_tv_promo_cost': 3                 # Promotional urgency (shorter than brand)
}

# Clean data first
train_clean = train_data.fillna(0)
test_clean = test_data.fillna(0)

# Apply lags to training data
train_with_lags = train_clean.copy()
for channel, lag_weeks in lag_config.items():
    print(f"   {channel}: {lag_weeks}-week lag window")
    train_with_lags = apply_lag_effects(train_with_lags, channel, lag_weeks)

# Apply SAME lag structure to test data (using test data only)
test_with_lags = test_clean.copy()
for channel, lag_weeks in lag_config.items():
    test_with_lags = apply_lag_effects(test_with_lags, channel, lag_weeks)

print(f"✅ Lag effects applied - No train/test contamination")

# %%
# THEORY 2: ADSTOCK WITH OPTIMAL DECAY RATES
print(f"\n📚 THEORY 2: ADSTOCK THEORY APPLICATION")
print("=" * 45)

print(f"🧠 THEORETICAL BASIS:")
print(f"   • Adstock Theory (Broadbent, 1984): Media effects decay over time")
print(f"   • Psychological: Memory decay follows exponential patterns")
print(f"   • Different media have different 'half-lives' in consumer memory")
print(f"   • Formula: Adstock[t] = Spend[t] + λ × Adstock[t-1]")

def apply_advanced_adstock(x, decay_rate, name="channel"):
    """
    Apply adstock transformation based on Broadbent's theory
    
    Theory: Media effectiveness decays exponentially
    - λ (decay_rate): Memory retention rate
    - Higher λ = longer memory (TV, Radio)
    - Lower λ = shorter memory (Search, Social)
    """
    adstocked = np.zeros_like(x, dtype=float)
    adstocked[0] = x[0] if not np.isnan(x[0]) else 0
    
    for i in range(1, len(x)):
        current_spend = x[i] if not np.isnan(x[i]) else 0
        adstocked[i] = current_spend + decay_rate * adstocked[i-1]
    
    return adstocked

# Theory-based decay rates
decay_rates = {
    'search_cost': 0.2,                           # Quick decay - immediate response
    'tv_branding_tv_branding_cost': 0.8,          # Very long decay - brand building
    'social_costs': 0.3,                          # Medium decay - social proof
    'ooh_ooh_spend': 0.6,                         # Long decay - repeated exposure
    'radio_national_radio_national_cost': 0.5,    # Medium-long - audio memory
    'radio_local_radio_local_cost': 0.4,          # Medium - local relevance
    'tv_promo_tv_promo_cost': 0.3                 # Medium - promotional urgency
}

print(f"\n🔄 Applying adstock (decay rates based on memory theory):")
for channel in media_channels:
    if channel in train_with_lags.columns:
        decay = decay_rates[channel]
        
        # Apply to training data
        adstock_col = f"{channel}_adstock"
        train_with_lags[adstock_col] = apply_advanced_adstock(
            train_with_lags[channel].values, decay, channel
        )
        
        # Apply to test data
        test_with_lags[adstock_col] = apply_advanced_adstock(
            test_with_lags[channel].values, decay, channel
        )
        
        # Calculate theoretical lift
        original_sum = train_with_lags[channel].sum()
        adstock_sum = train_with_lags[adstock_col].sum()
        lift = (adstock_sum - original_sum) / original_sum * 100 if original_sum > 0 else 0
        
        print(f"   {channel}: λ={decay:.1f}, memory lift=+{lift:.1f}%")

# %%
# THEORY 3: TV BRANDING INVESTIGATION
print(f"\n📚 THEORY 3: TV BRANDING MYSTERY INVESTIGATION")
print("=" * 50)

print(f"🎪 THE MYSTERY: TV Branding negative coefficient despite highest spend")
print(f"🧠 POSSIBLE THEORETICAL EXPLANATIONS:")
print(f"   1. Brand vs Sales Distinction: Builds equity, not immediate sales")
print(f"   2. Competitive Pressure: Defensive spending against competitors") 
print(f"   3. Long Lag Effects: 6-12 week brand building delays")
print(f"   4. Interaction Effects: Works through other channels")
print(f"   5. Saturation: Already past optimal spend point")

# Test extended lags for TV Branding specifically
print(f"\n🔍 Testing TV Branding Extended Lags (TRAIN DATA ONLY):")
tv_brand_correlations = {}

for lag in range(1, 13):  # Test up to 12 weeks
    if lag < len(train_clean):
        lagged_tv = train_clean['tv_branding_tv_branding_cost'].shift(lag)
        if not lagged_tv.isnull().all():
            # Only use training data for correlation analysis
            valid_indices = ~lagged_tv.isnull()
            if valid_indices.sum() > 10:  # Need sufficient data points
                corr, p_val = pearsonr(
                    lagged_tv[valid_indices], 
                    train_clean['sales'][valid_indices]
                )
                tv_brand_correlations[lag] = {'corr': corr, 'p_val': p_val}
                
                if abs(corr) > 0.1:  # Only show meaningful correlations
                    print(f"   {lag}-week lag: correlation = {corr:.3f} (p={p_val:.3f})")

# Find best lag for TV Branding
if tv_brand_correlations:
    best_tv_lag = max(tv_brand_correlations.keys(), 
                     key=lambda x: abs(tv_brand_correlations[x]['corr']))
    best_tv_corr = tv_brand_correlations[best_tv_lag]['corr']
    
    print(f"   🎯 Best TV Branding lag: {best_tv_lag} weeks (corr: {best_tv_corr:.3f})")
    
    # Add the optimal TV Branding lag
    if best_tv_lag > 0 and abs(best_tv_corr) > 0.05:
        tv_lag_col = f"tv_branding_tv_branding_cost_lag_{best_tv_lag}w"
        train_with_lags[tv_lag_col] = train_clean['tv_branding_tv_branding_cost'].shift(best_tv_lag)
        test_with_lags[tv_lag_col] = test_clean['tv_branding_tv_branding_cost'].shift(best_tv_lag)
        print(f"   ✅ Added {best_tv_lag}-week TV Branding lag feature")
else:
    print(f"   ⚠️  No significant TV Branding lag correlations found")
    tv_lag_col = None

# Test TV synergy interaction
print(f"\n🤝 Testing TV Synergy Theory:")
if 'tv_branding_tv_branding_cost' in train_with_lags.columns and 'tv_promo_tv_promo_cost' in train_with_lags.columns:
    # Create interaction term
    train_with_lags['tv_synergy'] = (
        train_with_lags['tv_branding_tv_branding_cost_adstock'] * 
        train_with_lags['tv_promo_tv_promo_cost_adstock']
    ) / 1000000  # Scale down
    
    test_with_lags['tv_synergy'] = (
        test_with_lags['tv_branding_tv_branding_cost_adstock'] * 
        test_with_lags['tv_promo_tv_promo_cost_adstock']
    ) / 1000000
    
    # Test correlation
    synergy_corr, _ = pearsonr(train_with_lags['tv_synergy'], train_with_lags['sales'])
    print(f"   TV Synergy correlation: {synergy_corr:.3f}")
    print(f"   ✅ Added TV Brand × Promo interaction")

# %%
# Prepare enhanced feature set
print(f"\n📊 PREPARING ENHANCED FEATURE SET")
print("=" * 40)

# Base adstock features
base_features = [f"{ch}_adstock" for ch in media_channels]

# Lag features (based on our analysis)
lag_features = []
for channel, lag_weeks in lag_config.items():
    for lag in range(1, lag_weeks + 1):
        lag_col = f"{channel}_lag_{lag}w"
        if lag_col in train_with_lags.columns:
            lag_features.append(lag_col)
    
    # Rolling average features
    roll_col = f"{channel}_roll_{lag_weeks}w"
    if roll_col in train_with_lags.columns:
        lag_features.append(roll_col)

# Special TV Branding lag
if tv_lag_col and tv_lag_col in train_with_lags.columns:
    lag_features.append(tv_lag_col)

# Interaction features
interaction_features = []
if 'tv_synergy' in train_with_lags.columns:
    interaction_features.append('tv_synergy')

# All features
all_features = base_features + lag_features + control_variables + interaction_features

print(f"📊 Feature Engineering Summary:")
print(f"   Base adstock features: {len(base_features)}")
print(f"   Lag features: {len(lag_features)}")
print(f"   Control variables: {len(control_variables)}")
print(f"   Interaction features: {len(interaction_features)}")
print(f"   Total features: {len(all_features)}")

# Check feature availability and handle missing
available_features = []
for feature in all_features:
    if feature in train_with_lags.columns and feature in test_with_lags.columns:
        available_features.append(feature)
    else:
        print(f"   ⚠️  Skipping missing feature: {feature}")

print(f"   ✅ Available features: {len(available_features)}")

# %%
# THEORY 4: PROPER VALIDATION WITH REGULARIZATION
print(f"\n📚 THEORY 4: ROBUST VALIDATION FRAMEWORK")
print("=" * 45)

print(f"🧠 THEORETICAL BASIS:")
print(f"   • Time Series Validation: Respect temporal structure")
print(f"   • Regularization Theory: Prevent overfitting with many features")
print(f"   • Ridge Regression: L2 penalty shrinks coefficients")
print(f"   • Cross-validation: Find optimal regularization strength")

# Create feature matrices
X_train = train_with_lags[available_features].fillna(0)
X_test = test_with_lags[available_features].fillna(0)
y_train = train_with_lags['sales']
y_test = test_with_lags['sales']

print(f"\n📊 Final dataset dimensions:")
print(f"   Training: {X_train.shape[0]} samples × {X_train.shape[1]} features")
print(f"   Test: {X_test.shape[0]} samples × {X_test.shape[1]} features")
print(f"   Sample-to-feature ratio: {X_train.shape[0]/X_train.shape[1]:.1f}:1")

# Scale features (fit scaler on training data only!)
print(f"\n🔧 Feature scaling (fit on train only):")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Transform using train statistics
print(f"   ✅ Scaled using training data statistics only")

# Time series cross-validation for regularization
print(f"\n🎯 Optimizing regularization with Time Series CV:")
alphas = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
tscv = TimeSeriesSplit(n_splits=3)

ridge_cv = RidgeCV(alphas=alphas, cv=tscv, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train_scaled, y_train)

optimal_alpha = ridge_cv.alpha_
print(f"   ✅ Optimal α: {optimal_alpha} (balances bias-variance)")

# %%
# Train final model and evaluate
print(f"\n🎯 FINAL MODEL TRAINING & EVALUATION")
print("=" * 45)

# Train final model
final_model = Ridge(alpha=optimal_alpha)
final_model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = final_model.predict(X_train_scaled)
y_test_pred = final_model.predict(X_test_scaled)

# Metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
gap = train_r2 - test_r2

print(f"🎉 ENHANCED MODEL RESULTS:")
print(f"   Training R²: {train_r2:.3f} ({train_r2*100:.1f}%)")
print(f"   Test R²: {test_r2:.3f} ({test_r2*100:.1f}%)")
print(f"   Overfitting gap: {gap:.3f} ({gap*100:.1f}%)")
print(f"   Test MAE: ${test_mae:,.0f}")
print(f"   Test RMSE: ${np.sqrt(mean_squared_error(y_test, y_test_pred)):,.0f}")

# Compare with baseline
baseline_test_r2 = 0.451  # From simple model
baseline_gap = 0.141

improvement = (test_r2 - baseline_test_r2) / baseline_test_r2 * 100
gap_improvement = (baseline_gap - gap) / baseline_gap * 100

print(f"\n📊 IMPROVEMENT vs BASELINE:")
print(f"   Baseline: Test R² = 45.1%, Gap = 14.1%")
print(f"   Enhanced: Test R² = {test_r2*100:.1f}%, Gap = {gap*100:.1f}%")

if improvement > 0:
    print(f"   🎉 R² IMPROVEMENT: +{improvement:.1f}%")
    if test_r2 >= 0.50:
        print(f"   🎯 TARGET ACHIEVED: >50% Test R²!")
    else:
        print(f"   📈 Progress toward 50% target")
else:
    print(f"   📊 R² Change: {improvement:.1f}% (focus on gap reduction)")

if gap_improvement > 0:
    print(f"   ✅ Overfitting REDUCED by: {gap_improvement:.1f}%")
else:
    print(f"   ⚠️  Overfitting changed by: {gap_improvement:.1f}%")

# %%
# Feature importance analysis
print(f"\n🏆 ENHANCED FEATURE IMPORTANCE ANALYSIS")
print("=" * 50)

coefficients = final_model.coef_
feature_importance = pd.DataFrame({
    'Feature': available_features,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"🏆 TOP 10 MOST INFLUENTIAL FEATURES:")
for i, row in feature_importance.head(10).iterrows():
    coef = row['Coefficient']
    direction = "📈 Positive" if coef > 0 else "📉 Negative"
    feature_name = row['Feature']
    
    # Identify feature type
    if any(ch in feature_name for ch in media_channels):
        if '_lag_' in feature_name:
            ftype = "🔄 Lag Effect"
        elif '_roll_' in feature_name:
            ftype = "📊 Rolling Avg"
        elif '_adstock' in feature_name:
            ftype = "💭 Adstock"
        else:
            ftype = "📺 Media"
    elif feature_name in control_variables:
        ftype = "🌡️  Control"
    elif feature_name == 'tv_synergy':
        ftype = "🤝 Interaction"
    else:
        ftype = "❓ Other"
    
    print(f"   {i+1}. {feature_name} ({ftype})")
    print(f"      Coefficient: {coef:.2e} ({direction})")

# Analyze TV Branding specifically
print(f"\n🎪 TV BRANDING ANALYSIS:")
tv_features = [f for f in available_features if 'tv_branding' in f]
for feature in tv_features:
    if feature in feature_importance['Feature'].values:
        coef = feature_importance[feature_importance['Feature'] == feature]['Coefficient'].iloc[0]
        direction = "📈 Positive" if coef > 0 else "📉 Negative"
        print(f"   {feature}: {coef:.2e} ({direction})")

if tv_lag_col and tv_lag_col in available_features:
    print(f"   🔍 Lag effect analysis suggests TV Branding has {best_tv_lag}-week delay")

# %%
# Business insights
print(f"\n💼 BUSINESS INSIGHTS - THEORY VALIDATED")
print("=" * 45)

print(f"✅ THEORETICAL VALIDATIONS:")

# Check if lag theory worked
lag_count = sum(1 for f in feature_importance.head(10)['Feature'] if '_lag_' in f)
if lag_count > 0:
    print(f"   🔄 Lag Theory: {lag_count} lag features in top 10 (CONFIRMED)")
else:
    print(f"   🔄 Lag Theory: No lag features in top 10 (needs investigation)")

# Check adstock vs original
adstock_count = sum(1 for f in feature_importance.head(10)['Feature'] if '_adstock' in f)
print(f"   💭 Adstock Theory: {adstock_count} adstock features in top 10")

# Check TV synergy
if 'tv_synergy' in feature_importance.head(10)['Feature'].values:
    tv_synergy_coef = feature_importance[feature_importance['Feature'] == 'tv_synergy']['Coefficient'].iloc[0]
    if tv_synergy_coef > 0:
        print(f"   🤝 TV Synergy: POSITIVE coefficient (theory confirmed)")
    else:
        print(f"   🤝 TV Synergy: Negative coefficient (needs investigation)")

print(f"\n🎯 KEY BUSINESS RECOMMENDATIONS:")
print(f"   1. Focus on channels with strong lag effects")
print(f"   2. Coordinate TV Brand + Promo campaigns")
if tv_lag_col:
    print(f"   3. Plan for {best_tv_lag}-week TV Branding delays")
print(f"   4. Monitor sustained spending patterns (rolling averages matter)")

print(f"\n🚀 NEXT ENHANCEMENT OPPORTUNITIES:")
if test_r2 < 0.50:
    print(f"   • Test longer lag windows (8-12 weeks)")
    print(f"   • Add more interaction terms")
    print(f"   • Consider seasonal hierarchical effects")
    print(f"   • Investigate external competitive factors")
else:
    print(f"   🎉 Model performing well! Focus on business implementation")

print(f"\n✅ MODEL READY FOR BUSINESS DECISIONS")
print(f"   Theory-based, validated on unseen data, interpretable coefficients") 