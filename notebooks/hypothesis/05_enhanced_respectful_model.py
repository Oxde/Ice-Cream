# %% [markdown]
# # Enhanced MMM - Respectful Approach (Keep All Channels)
# 
# **Philosophy**: If the company spends money on it, there's a business reason
# **Goal**: Improve model performance WITHOUT dropping any media channels
# 
# **ENHANCEMENT STRATEGIES:**
# 1. **Better Data Preprocessing** - Handle missing values more intelligently
# 2. **Advanced Adstock Modeling** - More sophisticated carryover effects  
# 3. **Regularization Optimization** - Find the perfect balance
# 4. **Interaction Effects** - Capture channel synergies
# 5. **Better Validation** - More robust performance measurement
# 
# **KEEP ALL 7 MEDIA CHANNELS** - Respect business decisions

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("🤝 ENHANCED MMM - RESPECTFUL APPROACH")
print("=" * 50)
print("📊 Goal: Improve model while keeping ALL media channels")
print("💼 Philosophy: If they spend on it, there's a business reason")

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)

# %%
# Load data
print(f"\n📁 LOADING DATA")
print("=" * 30)

train_data = pd.read_csv('../data/mmm_ready/consistent_channels_train_set.csv')
test_data = pd.read_csv('../data/mmm_ready/consistent_channels_test_set.csv')

train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

print(f"✅ Training: {train_data.shape[0]} weeks, Test: {test_data.shape[0]} weeks")

# ALL 7 media channels - KEEP THEM ALL
media_channels = [
    'search_cost', 'tv_branding_tv_branding_cost', 'social_costs',
    'ooh_ooh_spend', 'radio_national_radio_national_cost',
    'radio_local_radio_local_cost', 'tv_promo_tv_promo_cost'
]

control_variables = [
    'month_sin', 'month_cos', 'week_sin', 'week_cos', 'holiday_period',
    'weather_temperature_mean', 'weather_sunshine_duration', 'promo_promotion_type'
]

print(f"\n📊 KEEPING ALL MEDIA CHANNELS:")
for i, channel in enumerate(media_channels, 1):
    avg_spend = train_data[channel].fillna(0).mean()
    print(f"   {i}. {channel}: ${avg_spend:,.0f}/week avg")

# %%
# ENHANCEMENT 1: Intelligent Missing Value Handling
print(f"\n🧠 ENHANCEMENT 1: INTELLIGENT MISSING VALUE HANDLING")
print("=" * 55)

def smart_missing_value_handling(data, media_cols, control_cols):
    """Handle missing values more intelligently"""
    data_clean = data.copy()
    
    print(f"🔍 Analyzing missing values:")
    
    # Media channels: Use forward fill + interpolation (spending often continues)
    for channel in media_cols:
        if channel in data.columns:
            missing_count = data[channel].isnull().sum()
            if missing_count > 0:
                print(f"   {channel}: {missing_count} missing values")
                # Forward fill first (spending continues), then interpolate, then fill with 0
                data_clean[channel] = data[channel].fillna(method='ffill').interpolate().fillna(0)
    
    # Control variables: Use more sophisticated imputation
    for col in control_cols:
        if col in data.columns:
            missing_count = data[col].isnull().sum()
            if missing_count > 0:
                print(f"   {col}: {missing_count} missing values")
                if col == 'promo_promotion_type':
                    # Promotional types: use mode (most common type)
                    mode_val = data[col].mode().iloc[0] if not data[col].mode().empty else 0
                    data_clean[col] = data[col].fillna(mode_val)
                elif 'weather' in col:
                    # Weather: use seasonal interpolation
                    data_clean[col] = data[col].interpolate(method='time').fillna(data[col].median())
                else:
                    # Other controls: median imputation
                    data_clean[col] = data[col].fillna(data[col].median())
    
    return data_clean

# Apply intelligent missing value handling
train_clean = smart_missing_value_handling(train_data, media_channels, control_variables)
test_clean = smart_missing_value_handling(test_data, media_channels, control_variables)

# %%
# ENHANCEMENT 2: Advanced Adstock with Multiple Decay Patterns
print(f"\n📈 ENHANCEMENT 2: ADVANCED ADSTOCK MODELING")
print("=" * 50)

def apply_advanced_adstock(x, decay_rate, conc_param=1.0):
    """Apply adstock with concentration parameter for more flexible decay"""
    adstocked = np.zeros_like(x)
    adstocked[0] = x[0] if not np.isnan(x[0]) else 0
    
    for i in range(1, len(x)):
        current_spend = x[i] if not np.isnan(x[i]) else 0
        # Apply concentration parameter for different decay shapes
        decay_effect = decay_rate ** conc_param
        adstocked[i] = current_spend + decay_effect * adstocked[i-1]
    
    return adstocked

# Enhanced channel-specific decay rates with business logic
enhanced_decay_rates = {
    'search_cost': {'decay': 0.2, 'conc': 1.0},                          # Quick decay, immediate
    'tv_branding_tv_branding_cost': {'decay': 0.7, 'conc': 0.8},         # Long decay, gradual
    'social_costs': {'decay': 0.3, 'conc': 1.2},                         # Medium decay, concentrated
    'ooh_ooh_spend': {'decay': 0.6, 'conc': 0.9},                        # Long decay, outdoor visibility
    'radio_national_radio_national_cost': {'decay': 0.5, 'conc': 1.0},   # Medium decay, broad reach
    'radio_local_radio_local_cost': {'decay': 0.4, 'conc': 1.1},         # Shorter decay, local
    'tv_promo_tv_promo_cost': {'decay': 0.4, 'conc': 1.3}                # Medium decay, promotional
}

def transform_media_advanced_adstock(data, media_cols):
    """Apply advanced adstock to all media channels"""
    data_transformed = data.copy()
    
    print(f"🔄 Applying advanced adstock to ALL channels:")
    for channel in media_cols:
        if channel in data.columns:
            params = enhanced_decay_rates.get(channel, {'decay': 0.4, 'conc': 1.0})
            clean_spend = data[channel].fillna(0)
            
            # Apply advanced adstock
            adstocked = apply_advanced_adstock(clean_spend.values, 
                                             params['decay'], 
                                             params['conc'])
            
            new_col = f"{channel}_adstock"
            data_transformed[new_col] = adstocked
            
            # Calculate impact
            original_sum = clean_spend.sum()
            adstock_sum = adstocked.sum()
            lift = (adstock_sum - original_sum) / original_sum * 100 if original_sum > 0 else 0
            
            print(f"   ✅ {channel}:")
            print(f"      Decay: {params['decay']:.1f}, Concentration: {params['conc']:.1f}")
            print(f"      Adstock lift: +{lift:.1f}%")
    
    return data_transformed

# Apply advanced adstock
train_adstock = transform_media_advanced_adstock(train_clean, media_channels)
test_adstock = transform_media_advanced_adstock(test_clean, media_channels)

# %%
# ENHANCEMENT 3: Create interaction terms for major channels
print(f"\n🤝 ENHANCEMENT 3: CHANNEL INTERACTION EFFECTS")
print("=" * 50)

def add_channel_interactions(data):
    """Add interaction terms for channels that might work together"""
    data_interactions = data.copy()
    
    # TV channels might work together (brand + promo)
    if 'tv_branding_tv_branding_cost_adstock' in data.columns and 'tv_promo_tv_promo_cost_adstock' in data.columns:
        data_interactions['tv_synergy'] = (data['tv_branding_tv_branding_cost_adstock'] * 
                                         data['tv_promo_tv_promo_cost_adstock']) / 1000000  # Scale down
        print(f"   ✅ Created TV synergy interaction")
    
    # Radio channels might work together (national + local)
    if 'radio_national_radio_national_cost_adstock' in data.columns and 'radio_local_radio_local_cost_adstock' in data.columns:
        data_interactions['radio_synergy'] = (data['radio_national_radio_national_cost_adstock'] * 
                                            data['radio_local_radio_local_cost_adstock']) / 1000000  # Scale down
        print(f"   ✅ Created Radio synergy interaction")
    
    # Digital channels might work together (search + social)
    if 'search_cost_adstock' in data.columns and 'social_costs_adstock' in data.columns:
        data_interactions['digital_synergy'] = (data['search_cost_adstock'] * 
                                               data['social_costs_adstock']) / 1000000  # Scale down
        print(f"   ✅ Created Digital synergy interaction")
    
    return data_interactions

train_with_interactions = add_channel_interactions(train_adstock)
test_with_interactions = add_channel_interactions(test_adstock)

# %%
# Prepare feature sets - ALL MEDIA CHANNELS INCLUDED
print(f"\n📊 PREPARING FEATURES - ALL CHANNELS INCLUDED")
print("=" * 50)

# Use all media channels with adstock
media_features = [f"{ch}_adstock" for ch in media_channels]

# Add interaction terms
interaction_features = []
if 'tv_synergy' in train_with_interactions.columns:
    interaction_features.append('tv_synergy')
if 'radio_synergy' in train_with_interactions.columns:
    interaction_features.append('radio_synergy')
if 'digital_synergy' in train_with_interactions.columns:
    interaction_features.append('digital_synergy')

# All features together
all_features = media_features + control_variables + interaction_features

print(f"📊 Feature Summary:")
print(f"   Media channels: {len(media_features)} (ALL KEPT)")
print(f"   Control variables: {len(control_variables)}")
print(f"   Interaction terms: {len(interaction_features)}")
print(f"   Total features: {len(all_features)}")

# Create feature matrices
X_train = train_with_interactions[all_features]
X_test = test_with_interactions[all_features]
y_train = train_clean['sales']
y_test = test_clean['sales']

# Check for any remaining missing values
print(f"\n🔍 Final data quality check:")
train_missing = X_train.isnull().sum().sum()
test_missing = X_test.isnull().sum().sum()
print(f"   Training missing values: {train_missing}")
print(f"   Test missing values: {test_missing}")

if train_missing > 0 or test_missing > 0:
    print(f"   🔧 Fixing remaining missing values...")
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

# %%
# ENHANCEMENT 4: Advanced Regularization with Cross-Validation
print(f"\n⚙️ ENHANCEMENT 4: ADVANCED REGULARIZATION OPTIMIZATION")
print("=" * 60)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"📊 Model setup:")
print(f"   Training samples: {X_train_scaled.shape[0]}")
print(f"   Features: {X_train_scaled.shape[1]}")
print(f"   Sample-to-feature ratio: {X_train_scaled.shape[0]/X_train_scaled.shape[1]:.1f}:1")

# Use RidgeCV for optimal alpha selection with time series CV
print(f"\n🔄 Finding optimal regularization with time series CV...")

# Test a wide range of alphas
alphas = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]

# Use time series cross-validation
tscv = TimeSeriesSplit(n_splits=3)
ridge_cv = RidgeCV(alphas=alphas, cv=tscv, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train_scaled, y_train)

optimal_alpha = ridge_cv.alpha_
print(f"   ✅ Optimal α: {optimal_alpha}")

# Fit final model
final_model = Ridge(alpha=optimal_alpha)
final_model.fit(X_train_scaled, y_train)

# %%
# Evaluate the enhanced model
print(f"\n📊 ENHANCED MODEL EVALUATION")
print("=" * 40)

# Predictions
y_train_pred = final_model.predict(X_train_scaled)
y_test_pred = final_model.predict(X_test_scaled)

# Metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
gap = train_r2 - test_r2

print(f"🎯 PERFORMANCE METRICS:")
print(f"   Training R²: {train_r2:.3f} ({train_r2*100:.1f}%)")
print(f"   Test R²: {test_r2:.3f} ({test_r2*100:.1f}%)")
print(f"   Overfitting gap: {gap:.3f} ({gap*100:.1f}%)")
print(f"   Test MAE: ${test_mae:,.0f}")

print(f"\n📊 COMPARISON WITH ORIGINAL SIMPLE MODEL:")
original_test_r2 = 0.451
original_gap = 0.141

improvement = (test_r2 - original_test_r2) / original_test_r2 * 100
gap_improvement = (original_gap - gap) / original_gap * 100

print(f"   Original: Test R² = 45.1%, Gap = 14.1%")
print(f"   Enhanced: Test R² = {test_r2*100:.1f}%, Gap = {gap*100:.1f}%")

if improvement > 0:
    print(f"   🎉 R² IMPROVEMENT: +{improvement:.1f}%")
else:
    print(f"   📉 R² Change: {improvement:.1f}%")

if gap_improvement > 0:
    print(f"   ✅ Overfitting REDUCED by: {gap_improvement:.1f}%")
else:
    print(f"   ⚠️  Overfitting changed by: {gap_improvement:.1f}%")

# %%
# Business insights - ALL CHANNELS INCLUDED
print(f"\n💼 BUSINESS INSIGHTS - ALL CHANNELS ANALYSIS")
print("=" * 55)

# Feature importance
coefficients = final_model.coef_
feature_importance = pd.DataFrame({
    'Feature': all_features,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"🏆 TOP 10 MOST INFLUENTIAL FEATURES:")
for i, row in feature_importance.head(10).iterrows():
    coef = row['Coefficient']
    direction = "📈 Positive" if coef > 0 else "📉 Negative"
    feature_name = row['Feature']
    print(f"   {i+1}. {feature_name}: {coef:.2e} ({direction})")

# Media channel specific insights
print(f"\n💰 ALL MEDIA CHANNELS - BUSINESS PERFORMANCE:")
media_insights = {}
for i, feature in enumerate(all_features):
    if any(ch in feature for ch in media_channels) and '_adstock' in feature:
        # Extract original channel name
        original_channel = None
        for ch in media_channels:
            if ch in feature:
                original_channel = ch
                break
        
        if original_channel:
            coef_val = coefficients[i]
            avg_spend = train_clean[original_channel].mean()
            
            # Determine impact level
            abs_coef = abs(coef_val)
            coef_std = np.std(np.abs(coefficients))
            
            if abs_coef > coef_std:
                impact_level = "🔥 High Impact"
            elif abs_coef > coef_std * 0.5:
                impact_level = "⭐ Medium Impact"
            else:
                impact_level = "💡 Low Impact"
            
            direction = "Positive ROI" if coef_val > 0 else "Negative ROI"
            
            media_insights[original_channel] = {
                'coefficient': coef_val,
                'avg_spend': avg_spend,
                'impact_level': impact_level,
                'direction': direction
            }

# Sort by absolute coefficient value
sorted_channels = sorted(media_insights.items(), 
                        key=lambda x: abs(x[1]['coefficient']), 
                        reverse=True)

for rank, (channel, info) in enumerate(sorted_channels, 1):
    coef = info['coefficient']
    spend = info['avg_spend']
    impact = info['impact_level']
    direction = info['direction']
    
    print(f"\n   {rank}. {channel}:")
    print(f"      {impact} - {direction}")
    print(f"      Average weekly spend: ${spend:,.0f}")
    print(f"      Coefficient: {coef:.2e}")

# %%
# Final recommendations
print(f"\n🎯 FINAL RECOMMENDATIONS - RESPECTFUL APPROACH")
print("=" * 55)

print(f"✅ WHAT WE ACHIEVED:")
print(f"   • Kept ALL 7 media channels (respecting business decisions)")
print(f"   • Test R²: {test_r2:.1%} (vs 45.1% original)")
print(f"   • Overfitting gap: {gap:.1%} (vs 14.1% original)")
print(f"   • Added channel synergy effects")
print(f"   • More sophisticated adstock modeling")

print(f"\n💼 BUSINESS INSIGHTS:")
print(f"   🏆 TOP PERFORMING CHANNELS:")
for rank, (channel, info) in enumerate(sorted_channels[:3], 1):
    print(f"      {rank}. {channel}: {info['impact_level']} - {info['direction']}")

if len(sorted_channels) > 3:
    print(f"\n   💡 OTHER CHANNELS (Still valuable!):")
    for rank, (channel, info) in enumerate(sorted_channels[3:], 4):
        print(f"      {rank}. {channel}: {info['impact_level']} - {info['direction']}")

print(f"\n🚀 NEXT STEPS:")
print(f"   1. Focus budget increases on top performing channels")
print(f"   2. Optimize spend levels for all channels")
print(f"   3. Test channel synergies (TV, Radio, Digital combinations)")
print(f"   4. Monitor performance and adjust based on results")
print(f"")
print(f"💡 KEY INSIGHT: Every channel has a role - optimize rather than eliminate!") 