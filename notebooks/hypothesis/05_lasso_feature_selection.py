# %% [markdown]
# # LASSO Feature Selection MMM - Smart Simplicity
# 
# **Philosophy**: Let the algorithm choose the best features automatically
# **Goal**: Beat 45.1% Test R² through intelligent feature selection, not complexity
# 
# **Why LASSO?**
# - Automatic feature selection (L1 penalty drives coefficients to exactly zero)
# - Built-in regularization (prevents overfitting)
# - Interpretable results (clear which features matter)
# - Works well with correlated features (picks the best representative)
# 
# **Strategy**:
# 1. Start with simple model foundation (proven 45.1% baseline)
# 2. Add minimal feature engineering (basic adstock + few lags)
# 3. Let LASSO automatically select the best combination
# 4. Target: >45% Test R² with <15% overfitting gap

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

print("🎯 LASSO FEATURE SELECTION MMM - SMART SIMPLICITY")
print("=" * 55)
print("🧠 Philosophy: Let the algorithm choose the best features")
print("📊 Goal: Beat 45.1% baseline through intelligent selection")
print("⚡ Strategy: Minimal engineering + automatic selection")

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)

# %%
print(f"\n📁 LOADING DATA - PROVEN TEMPORAL SPLIT")
print("=" * 45)

# Load the same data split as our baseline
train_data = pd.read_csv('../data/mmm_ready/consistent_channels_train_set.csv')
test_data = pd.read_csv('../data/mmm_ready/consistent_channels_test_set.csv')

train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

print(f"✅ Training: {train_data['date'].min()} to {train_data['date'].max()}")
print(f"✅ Test: {test_data['date'].min()} to {test_data['date'].max()}")

# Clean data
train_clean = train_data.fillna(0)
test_clean = test_data.fillna(0)

print(f"📊 Training samples: {len(train_clean)}")
print(f"📊 Test samples: {len(test_clean)}")

# %%
print(f"\n🔧 MINIMAL FEATURE ENGINEERING")
print("=" * 35)

print(f"🎯 Strategy: Create candidate features, let LASSO choose the best")

def create_candidate_features(data):
    """
    Create a rich but focused set of candidate features
    LASSO will automatically select the best ones
    """
    features_df = data.copy()
    
    # 1. BASE MEDIA CHANNELS (current spend) - from simple model
    base_channels = [
        'search_cost', 'tv_branding_tv_branding_cost', 'social_costs',
        'ooh_ooh_spend', 'radio_national_radio_national_cost',
        'radio_local_radio_local_cost', 'tv_promo_tv_promo_cost'
    ]
    
    # 2. SIMPLE ADSTOCK (proven from research)
    print(f"   Adding simple adstock transformations...")
    adstock_params = {
        'search_cost': 0.2,
        'tv_branding_tv_branding_cost': 0.5,  # Less aggressive than before
        'social_costs': 0.3,
        'ooh_ooh_spend': 0.4,
        'radio_national_radio_national_cost': 0.4,
        'radio_local_radio_local_cost': 0.4,
        'tv_promo_tv_promo_cost': 0.3
    }
    
    for channel, decay in adstock_params.items():
        if channel in features_df.columns:
            adstock_col = f"{channel}_adstock"
            adstock_values = np.zeros_like(features_df[channel], dtype=float)
            adstock_values[0] = features_df[channel].iloc[0]
            
            for i in range(1, len(features_df)):
                current_spend = features_df[channel].iloc[i]
                adstock_values[i] = current_spend + decay * adstock_values[i-1]
            
            features_df[adstock_col] = adstock_values
    
    # 3. STRATEGIC LAG FEATURES (only most promising from research)
    print(f"   Adding strategic lag features...")
    
    # Radio lags (proven strong in research)
    if 'radio_local_radio_local_cost' in features_df.columns:
        features_df['radio_local_lag_2w'] = features_df['radio_local_radio_local_cost'].shift(2)
        features_df['radio_local_lag_4w'] = features_df['radio_local_radio_local_cost'].shift(4)
    
    if 'radio_national_radio_national_cost' in features_df.columns:
        features_df['radio_national_lag_2w'] = features_df['radio_national_radio_national_cost'].shift(2)
        features_df['radio_national_lag_4w'] = features_df['radio_national_radio_national_cost'].shift(4)
    
    # TV branding longer lags (testing the breakthrough theory)
    if 'tv_branding_tv_branding_cost' in features_df.columns:
        features_df['tv_branding_lag_4w'] = features_df['tv_branding_tv_branding_cost'].shift(4)
        features_df['tv_branding_lag_6w'] = features_df['tv_branding_tv_branding_cost'].shift(6)
        features_df['tv_branding_lag_8w'] = features_df['tv_branding_tv_branding_cost'].shift(8)
    
    # Fast response channels (1-2 week lags)
    if 'search_cost' in features_df.columns:
        features_df['search_lag_1w'] = features_df['search_cost'].shift(1)
        features_df['search_lag_2w'] = features_df['search_cost'].shift(2)
    
    if 'social_costs' in features_df.columns:
        features_df['social_lag_1w'] = features_df['social_costs'].shift(1)
    
    # 4. INTERACTION TERMS (minimal, only most logical)
    print(f"   Adding key interaction terms...")
    
    # TV synergy (brand × promo)
    if 'tv_branding_tv_branding_cost' in features_df.columns and 'tv_promo_tv_promo_cost' in features_df.columns:
        features_df['tv_synergy'] = (
            features_df['tv_branding_tv_branding_cost'] * 
            features_df['tv_promo_tv_promo_cost']
        ) / 1000000  # Scale down
    
    # Weather × promotion interaction (logical for ice cream)
    if 'weather_temperature_mean' in features_df.columns and 'promo_promotion_type' in features_df.columns:
        features_df['weather_promo_boost'] = (
            features_df['weather_temperature_mean'] * 
            features_df['promo_promotion_type']
        )
    
    return features_df

# Apply feature engineering
print(f"\n🔄 Creating candidate feature pool:")
train_enhanced = create_candidate_features(train_clean)
test_enhanced = create_candidate_features(test_clean)

# Define all potential features
base_media = [
    'search_cost', 'tv_branding_tv_branding_cost', 'social_costs',
    'ooh_ooh_spend', 'radio_national_radio_national_cost',
    'radio_local_radio_local_cost', 'tv_promo_tv_promo_cost'
]

adstock_features = [f"{ch}_adstock" for ch in base_media]

lag_features = [
    'radio_local_lag_2w', 'radio_local_lag_4w',
    'radio_national_lag_2w', 'radio_national_lag_4w',
    'tv_branding_lag_4w', 'tv_branding_lag_6w', 'tv_branding_lag_8w',
    'search_lag_1w', 'search_lag_2w', 'social_lag_1w'
]

interaction_features = ['tv_synergy', 'weather_promo_boost']

control_variables = [
    'month_sin', 'month_cos', 'week_sin', 'week_cos',
    'holiday_period', 'weather_temperature_mean', 
    'weather_sunshine_duration', 'promo_promotion_type'
]

# Combine all candidate features
all_candidates = base_media + adstock_features + lag_features + interaction_features + control_variables

# Check availability
available_candidates = []
for feature in all_candidates:
    if feature in train_enhanced.columns and feature in test_enhanced.columns:
        available_candidates.append(feature)
    else:
        print(f"   ⚠️  Skipping missing: {feature}")

print(f"\n📊 Candidate feature pool:")
print(f"   Base media: {len(base_media)}")
print(f"   Adstock: {len(adstock_features)}")
print(f"   Lag features: {len(lag_features)}")
print(f"   Interactions: {len(interaction_features)}")
print(f"   Controls: {len(control_variables)}")
print(f"   Total candidates: {len(available_candidates)}")

sample_ratio = len(train_clean) / len(available_candidates)
print(f"   Sample-to-feature ratio: {sample_ratio:.1f}:1")
print(f"   Status: {'✅ MANAGEABLE' if sample_ratio >= 2 else '⚠️ RISKY'}")

# %%
print(f"\n🎯 LASSO FEATURE SELECTION")
print("=" * 35)

# Prepare feature matrices
X_train = train_enhanced[available_candidates].fillna(0)
X_test = test_enhanced[available_candidates].fillna(0)
y_train = train_enhanced['sales']
y_test = test_enhanced['sales']

print(f"📊 Feature matrices:")
print(f"   Training: {X_train.shape}")
print(f"   Test: {X_test.shape}")

# Scale features (essential for LASSO)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Features scaled (LASSO requires standardization)")

# LASSO Cross-Validation to find optimal alpha
print(f"\n🔍 Finding optimal LASSO regularization:")
alphas = np.logspace(-4, 1, 50)  # Wide range for LASSO
tscv = TimeSeriesSplit(n_splits=3)

lasso_cv = LassoCV(alphas=alphas, cv=tscv, max_iter=2000, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)

optimal_alpha = lasso_cv.alpha_
print(f"   ✅ Optimal α: {optimal_alpha:.6f}")

# Train final LASSO model
final_lasso = Lasso(alpha=optimal_alpha, max_iter=2000, random_state=42)
final_lasso.fit(X_train_scaled, y_train)

# Count selected features
selected_features = []
selected_coefficients = []
for i, (feature, coef) in enumerate(zip(available_candidates, final_lasso.coef_)):
    if abs(coef) > 1e-10:  # LASSO threshold
        selected_features.append(feature)
        selected_coefficients.append(coef)

print(f"   ✅ Features selected by LASSO: {len(selected_features)} out of {len(available_candidates)}")
print(f"   📉 Feature reduction: {(1 - len(selected_features)/len(available_candidates))*100:.1f}%")

# %%
print(f"\n🎉 LASSO MODEL PERFORMANCE")
print("=" * 35)

# Make predictions
y_train_pred = final_lasso.predict(X_train_scaled)
y_test_pred = final_lasso.predict(X_test_scaled)

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
gap = train_r2 - test_r2

print(f"🎯 LASSO MODEL RESULTS:")
print(f"   Training R²: {train_r2:.3f} ({train_r2*100:.1f}%)")
print(f"   Test R²: {test_r2:.3f} ({test_r2*100:.1f}%)")
print(f"   Overfitting gap: {gap:.3f} ({gap*100:.1f}%)")
print(f"   Test MAE: ${test_mae:,.0f}")
print(f"   Test RMSE: ${test_rmse:,.0f}")

# Compare with baseline
baseline_test_r2 = 0.451
baseline_gap = 0.141

r2_improvement = (test_r2 - baseline_test_r2) / baseline_test_r2 * 100
gap_improvement = (baseline_gap - gap) / baseline_gap * 100

print(f"\n📊 vs SIMPLE MODEL BASELINE:")
print(f"   Baseline: Test R² = 45.1%, Gap = 14.1%")
print(f"   LASSO: Test R² = {test_r2*100:.1f}%, Gap = {gap*100:.1f}%")

if test_r2 > baseline_test_r2:
    print(f"   🎉 R² IMPROVEMENT: +{r2_improvement:.1f}%")
else:
    print(f"   📊 R² Change: {r2_improvement:.1f}%")

if gap < baseline_gap:
    print(f"   ✅ Gap IMPROVEMENT: {gap_improvement:.1f}% better")
else:
    print(f"   ⚠️ Gap Change: {gap_improvement:.1f}%")

# Target achievement
print(f"\n🎯 TARGET ACHIEVEMENT:")
print(f"   Beat 45.1% baseline: {'✅ SUCCESS' if test_r2 > baseline_test_r2 else '❌ missed'}")
print(f"   Gap < 15%: {'✅ SUCCESS' if gap < 0.15 else '❌ missed'}")

if test_r2 >= 0.50:
    print(f"   🎉 BONUS: 50%+ Test R² achieved!")

# %%
print(f"\n🏆 LASSO SELECTED FEATURES ANALYSIS")
print("=" * 45)

if len(selected_features) > 0:
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Coefficient': selected_coefficients,
        'Abs_Coefficient': np.abs(selected_coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print(f"🔍 FEATURES CHOSEN BY LASSO (ranked by importance):")
    for i, (_, row) in enumerate(feature_importance.iterrows()):
        coef = row['Coefficient']
        feature = row['Feature']
        
        # Categorize feature type
        if any(ch in feature for ch in base_media):
            if '_adstock' in feature:
                ftype = "💭 Adstock"
            elif '_lag_' in feature:
                ftype = "🔄 Lag Effect"
            else:
                ftype = "📺 Base Media"
        elif feature in control_variables:
            if 'weather' in feature:
                ftype = "🌡️ Weather"
            elif feature in ['month_sin', 'month_cos', 'week_sin', 'week_cos']:
                ftype = "📅 Seasonality"
            else:
                ftype = "🎯 Control"
        elif feature in interaction_features:
            ftype = "🤝 Interaction"
        else:
            ftype = "❓ Other"
        
        direction = "📈 Positive" if coef > 0 else "📉 Negative"
        
        print(f"   {i+1}. {feature} ({ftype})")
        print(f"      Coefficient: {coef:.0f} ({direction})")
    
    print(f"\n🧠 LASSO INSIGHTS:")
    
    # Analyze what LASSO chose
    base_selected = sum(1 for f in selected_features if f in base_media)
    adstock_selected = sum(1 for f in selected_features if '_adstock' in f)
    lag_selected = sum(1 for f in selected_features if '_lag_' in f)
    control_selected = sum(1 for f in selected_features if f in control_variables)
    interaction_selected = sum(1 for f in selected_features if f in interaction_features)
    
    print(f"   📊 Feature type selection:")
    print(f"     Base media: {base_selected}/{len(base_media)} ({base_selected/len(base_media)*100:.0f}%)")
    print(f"     Adstock: {adstock_selected}/{len(adstock_features)} ({adstock_selected/len(adstock_features)*100:.0f}%)")
    print(f"     Lag effects: {lag_selected}/{len(lag_features)} ({lag_selected/len(lag_features)*100:.0f}%)")
    print(f"     Controls: {control_selected}/{len(control_variables)} ({control_selected/len(control_variables)*100:.0f}%)")
    print(f"     Interactions: {interaction_selected}/{len(interaction_features)} ({interaction_selected/len(interaction_features)*100:.0f}%)")
    
    # TV branding analysis
    tv_features = [f for f in selected_features if 'tv_branding' in f]
    if tv_features:
        print(f"\n🎪 TV BRANDING LASSO SELECTION:")
        for feature in tv_features:
            coef = feature_importance[feature_importance['Feature'] == feature]['Coefficient'].iloc[0]
            print(f"     {feature}: {coef:.0f}")
    else:
        print(f"\n🎪 TV BRANDING: No features selected by LASSO")
    
    # Radio analysis
    radio_features = [f for f in selected_features if 'radio' in f]
    if radio_features:
        print(f"\n📻 RADIO LASSO SELECTION:")
        for feature in radio_features:
            coef = feature_importance[feature_importance['Feature'] == feature]['Coefficient'].iloc[0]
            print(f"     {feature}: {coef:.0f}")

else:
    print(f"⚠️ No features selected by LASSO (alpha too high)")

# %%
print(f"\n💼 BUSINESS IMPLICATIONS")
print("=" * 30)

if len(selected_features) > 0:
    print(f"🎯 LASSO BUSINESS INSIGHTS:")
    
    # Top positive drivers
    positive_features = feature_importance[feature_importance['Coefficient'] > 0]
    negative_features = feature_importance[feature_importance['Coefficient'] < 0]
    
    if len(positive_features) > 0:
        print(f"\n✅ SCALE UP (LASSO-selected growth drivers):")
        for _, row in positive_features.head(5).iterrows():
            feature = row['Feature']
            coef = row['Coefficient']
            
            if 'radio' in feature and 'lag' in feature:
                timing = feature.split('_lag_')[1] if '_lag_' in feature else 'immediate'
                print(f"   • Radio investment with {timing} planning horizon (+{coef:.0f})")
            elif 'weather' in feature:
                print(f"   • Weather-responsive activation strategy (+{coef:.0f})")
            elif 'holiday' in feature:
                print(f"   • Holiday period focus (+{coef:.0f})")
            elif 'tv_synergy' in feature:
                print(f"   • Coordinate TV brand + promo campaigns (+{coef:.0f})")
            else:
                print(f"   • Optimize {feature} (+{coef:.0f})")
    
    if len(negative_features) > 0:
        print(f"\n⚠️ OPTIMIZE (LASSO-identified challenges):")
        for _, row in negative_features.head(3).iterrows():
            feature = row['Feature']
            coef = row['Coefficient']
            print(f"   • Review {feature} strategy ({coef:.0f})")

print(f"\n🚀 LASSO MODEL SUMMARY:")
print(f"   • Automatic feature selection: {len(selected_features)} best features")
print(f"   • Performance: {test_r2:.1%} Test R² ({gap:.1%} gap)")
print(f"   • Business clarity: Clear priority channels identified")
print(f"   • Overfitting control: Built-in L1 regularization")

if test_r2 > baseline_test_r2:
    print(f"   🎉 SUCCESS: Beat simple model baseline!")
else:
    print(f"   📊 Result: {test_r2:.1%} vs {baseline_test_r2:.1%} baseline")

print(f"\n✅ LASSO MODEL READY FOR EVALUATION") 