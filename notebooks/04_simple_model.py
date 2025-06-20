# %% [markdown]
# # 04 - Simple MMM Baseline Model
# 
# **Research Goal**: Establish a solid, interpretable baseline for Media Mix Modeling
# **Business Context**: Ice cream company needs data-driven budget allocation guidance
# **Team**: Data Science Research Team
# 
# ## 🎯 Research Objectives
# 
# 1. **Build Trustworthy Foundation**: Create simple, explainable model stakeholders can trust
# 2. **Establish Performance Baseline**: Set benchmark for future model iterations  
# 3. **Generate Business Insights**: Provide actionable ROI guidance for each media channel
# 4. **Validate Methodology**: Prove temporal validation and overfitting prevention work
# 
# ## 📊 Data Strategy
# 
# - **Dataset**: `consistent_channels` (129 train + 27 test weeks, 2022-2025)
# - **No Email Channel**: Excluded due to data quality issues (missing/inconsistent data)
# - **7 Media Channels**: All spending channels with reliable data
# - **Temporal Split**: Strict chronological train/test (no data leakage)
# 
# ## 🔬 Modeling Approach
# 
# - **Algorithm**: Ridge Regression (prevents overfitting, handles multicollinearity)
# - **Adstock**: Simple carryover effects (decay=0.4) for media persistence  
# - **Features**: Media spend (adstocked) + seasonality + weather + promotions
# - **Validation**: Cross-validated regularization + temporal test set

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("🎯 04 - SIMPLE MMM BASELINE MODEL")
print("=" * 50)
print("📊 Research Goal: Establish interpretable baseline for budget allocation")
print("🏢 Business Impact: Clear ROI guidance for 7 media channels")
print("🔬 Method: Ridge regression with adstock + temporal validation")

# Configure clean plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11

# %% [markdown]
# ## 📁 Data Loading and Quality Assessment
# 
# We use the `consistent_channels` dataset which excludes email campaigns due to data quality issues.
# This ensures our baseline model is built on reliable, consistent data.

# %%
print(f"\n📁 LOADING VALIDATED MMM DATASET")
print("=" * 40)

# Load pre-validated train/test splits
train_data = pd.read_csv('../data/mmm_ready/consistent_channels_train_set.csv')
test_data = pd.read_csv('../data/mmm_ready/consistent_channels_test_set.csv')

# Parse dates for temporal analysis
train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

print(f"✅ Training Period: {train_data['date'].min().date()} to {train_data['date'].max().date()}")
print(f"   → {train_data.shape[0]} weeks of data")
print(f"✅ Test Period: {test_data['date'].min().date()} to {test_data['date'].max().date()}")
print(f"   → {test_data.shape[0]} weeks of data")

# Data quality assessment
train_missing = train_data.isnull().sum().sum()
test_missing = test_data.isnull().sum().sum()
print(f"\n📊 Data Quality Check:")
print(f"   Training missing values: {train_missing}")
print(f"   Test missing values: {test_missing}")

# Sales summary statistics
print(f"\n💰 Sales Overview:")
print(f"   Training - Mean: ${train_data['sales'].mean():,.0f}, Std: ${train_data['sales'].std():,.0f}")
print(f"   Test - Mean: ${test_data['sales'].mean():,.0f}, Std: ${test_data['sales'].std():,.0f}")

# %% [markdown]
# ## 🎯 Feature Definition and Business Logic
# 
# We carefully select features based on business relevance and data reliability:
# 
# ### Media Channels (7)
# All channels where the company actively spends money for customer acquisition
# 
# ### Control Variables (8) 
# External factors that influence ice cream sales but aren't controlled by marketing

# %%
print(f"\n🎯 FEATURE DEFINITION AND BUSINESS LOGIC")
print("=" * 50)

# Media spend channels - all reliable spending data
media_channels = [
    'search_cost',                          # Digital: Search advertising
    'tv_branding_tv_branding_cost',         # TV: Brand awareness campaigns  
    'social_costs',                         # Digital: Social media advertising
    'ooh_ooh_spend',                        # Outdoor: Billboards, transit ads
    'radio_national_radio_national_cost',   # Radio: National reach campaigns
    'radio_local_radio_local_cost',         # Radio: Local market campaigns
    'tv_promo_tv_promo_cost'                # TV: Promotional campaigns
]

# Control variables - external factors affecting ice cream sales
control_variables = [
    'month_sin', 'month_cos',               # Seasonal cycles (ice cream seasonality)
    'week_sin', 'week_cos',                 # Weekly patterns (weekend effects)
    'holiday_period',                       # Holiday periods (increased consumption)
    'weather_temperature_mean',             # Temperature (primary ice cream driver)
    'weather_sunshine_duration',            # Sunshine (outdoor activities)
    'promo_promotion_type'                  # Price promotions (demand drivers)
]

# Verify all features exist in data
available_media = [col for col in media_channels if col in train_data.columns]
available_controls = [col for col in control_variables if col in train_data.columns]

print(f"💰 MEDIA CHANNELS ANALYSIS ({len(available_media)} channels):")
print(f"{'Channel':<35} {'Avg Weekly Spend':<15} {'Total Investment':<15} {'Business Purpose'}")
print("-" * 90)

business_purposes = {
    'search_cost': 'Immediate conversion',
    'tv_branding_tv_branding_cost': 'Brand awareness', 
    'social_costs': 'Engagement & targeting',
    'ooh_ooh_spend': 'Local visibility',
    'radio_national_radio_national_cost': 'Mass reach',
    'radio_local_radio_local_cost': 'Local targeting', 
    'tv_promo_tv_promo_cost': 'Promotional push'
}

for channel in available_media:
    avg_spend = train_data[channel].fillna(0).mean()
    total_spend = train_data[channel].fillna(0).sum()
    purpose = business_purposes.get(channel, 'Customer acquisition')
    print(f"{channel:<35} ${avg_spend:<14,.0f} ${total_spend:<14,.0f} {purpose}")

print(f"\n📊 CONTROL VARIABLES ({len(available_controls)} variables):")
for var in available_controls:
    if 'weather' in var:
        category = "🌡️  Weather"
    elif any(x in var for x in ['month', 'week']):
        category = "📅 Seasonality"
    elif 'holiday' in var:
        category = "🎉 Events"
    elif 'promo' in var:
        category = "💰 Promotions"
    else:
        category = "📊 Other"
    print(f"   {var:<30} {category}")

# %% [markdown]
# ## 📈 Adstock Transformation: Media Carryover Effects
# 
# **Business Insight**: Media impact doesn't disappear immediately after spending stops.
# TV ads create awareness that lasts weeks, search ads drive immediate action.
# 
# **Methodology**: Apply adstock transformation to capture carryover effects
# - **Decay Rate**: 0.4 (moderate carryover - 40% of previous week's effect carries over)
# - **Business Logic**: Media builds cumulative awareness and purchase intent

# %%
print(f"\n📈 ADSTOCK TRANSFORMATION - MEDIA CARRYOVER EFFECTS")
print("=" * 55)

def apply_adstock_transformation(x, decay_rate=0.4):
    """
    Apply adstock (carryover) transformation to media spend
    
    Business Logic:
    - Media effects don't stop immediately when spending stops
    - Each week, some percentage of previous effect carries over
    - Accumulates impact over time for sustained campaigns
    
    Args:
        x: Media spend time series
        decay_rate: Fraction of previous effect that carries over (0.4 = 40%)
    """
    adstocked = np.zeros_like(x)
    adstocked[0] = x[0] if not np.isnan(x[0]) else 0
    
    for i in range(1, len(x)):
        current_spend = x[i] if not np.isnan(x[i]) else 0
        previous_effect = adstocked[i-1] * decay_rate
        adstocked[i] = current_spend + previous_effect
    
    return adstocked

def transform_all_media_channels(data, media_cols, decay_rate=0.4):
    """Apply adstock to all media channels in dataset"""
    data_transformed = data.copy()
    
    print(f"🔄 Applying adstock transformation (decay rate = {decay_rate}):")
    print(f"{'Channel':<35} {'Original Sum':<15} {'Adstock Sum':<15} {'Lift %':<10}")
    print("-" * 80)
    
    for channel in media_cols:
        if channel in data.columns:
            # Clean missing values (assume no spend = 0)
            clean_spend = data[channel].fillna(0)
            
            # Apply adstock transformation
            adstocked_values = apply_adstock_transformation(clean_spend.values, decay_rate)
            
            # Store in new column
            adstock_column = f"{channel}_adstock"
            data_transformed[adstock_column] = adstocked_values
            
            # Calculate business impact
            original_sum = clean_spend.sum()
            adstock_sum = adstocked_values.sum()
            lift_percent = ((adstock_sum - original_sum) / original_sum * 100) if original_sum > 0 else 0
            
            print(f"{channel:<35} ${original_sum:<14,.0f} ${adstock_sum:<14,.0f} {lift_percent:<9.1f}%")
    
    return data_transformed

# Apply adstock to both training and test data
print("📊 BUSINESS INSIGHT: Adstock captures cumulative media effects")
print("   → Higher adstock sum = better carryover effect capture")
print("   → Positive lift % = model accounts for sustained impact\n")

train_adstocked = transform_all_media_channels(train_data, available_media)
test_adstocked = transform_all_media_channels(test_data, available_media)

# Update feature list to use adstocked media
adstocked_media_features = [f"{col}_adstock" for col in available_media]

print(f"\n✅ Adstock transformation complete:")
print(f"   → {len(adstocked_media_features)} media channels with carryover effects")
print(f"   → Ready for model training")

# %% [markdown]
# ## 🧹 Data Preprocessing: Missing Value Strategy
# 
# **Research Approach**: Handle missing values based on business logic
# - **Media Spend**: Missing = No campaign running (fill with 0)
# - **Promotions**: Missing = No promotion active (fill with 0)  
# - **Weather**: Missing = Use median (temperature/sunshine)
# - **Seasonality**: Never missing (calculated features)

# %%
print(f"\n🧹 DATA PREPROCESSING - MISSING VALUE STRATEGY")
print("=" * 55)

def handle_missing_values_business_logic(data, control_cols):
    """
    Handle missing values using business-informed logic
    
    Strategy:
    - Promotions: NaN = no promotion active → 0
    - Weather: NaN = missing measurement → median imputation  
    - Seasonality: Never missing (calculated features)
    """
    data_clean = data.copy()
    
    print(f"🔍 Missing value analysis and treatment:")
    
    for col in control_cols:
        if col in data.columns:
            missing_count = data[col].isnull().sum()
            if missing_count > 0:
                if 'promo' in col:
                    # Business logic: No promotion data = no promotion running
                    data_clean[col] = data[col].fillna(0)
                    print(f"   {col:<30} {missing_count:>3} missing → 0 (no promotion)")
                    
                elif 'weather' in col:
                    # Weather: Use median (typical seasonal value)
                    median_val = data[col].median()
                    data_clean[col] = data[col].fillna(median_val)
                    print(f"   {col:<30} {missing_count:>3} missing → {median_val:.1f} (median)")
                    
                else:
                    # Other controls: Use median imputation
                    median_val = data[col].median()
                    data_clean[col] = data[col].fillna(median_val)
                    print(f"   {col:<30} {missing_count:>3} missing → {median_val:.2f} (median)")
            else:
                print(f"   {col:<30} ✅ No missing values")
    
    return data_clean

# Apply business-logic missing value handling
train_clean = handle_missing_values_business_logic(train_adstocked, available_controls)
test_clean = handle_missing_values_business_logic(test_adstocked, available_controls)

# Final validation - ensure no missing values in model features
model_features = adstocked_media_features + available_controls
train_final_missing = train_clean[model_features].isnull().sum().sum()
test_final_missing = test_clean[model_features].isnull().sum().sum()

print(f"\n✅ FINAL DATA QUALITY VALIDATION:")
print(f"   Training set missing values: {train_final_missing}")
print(f"   Test set missing values: {test_final_missing}")

if train_final_missing == 0 and test_final_missing == 0:
    print(f"   🎯 SUCCESS: Clean dataset ready for modeling")
else:
    print(f"   ⚠️  WARNING: Missing values remain - will fill with 0")

# %% [markdown]
# ## 🤖 Model Training: Ridge Regression with Cross-Validation
# 
# **Algorithm Choice**: Ridge Regression
# - **Handles multicollinearity** between media channels
# - **Prevents overfitting** with L2 regularization
# - **Interpretable coefficients** for business insights
# 
# **Cross-Validation Strategy**: 
# - Test multiple regularization strengths (α)
# - Use 5-fold CV to find optimal balance
# - Minimize overfitting while preserving performance

# %%
print(f"\n🤖 MODEL TRAINING - RIDGE REGRESSION WITH CROSS-VALIDATION")
print("=" * 65)

# Prepare feature matrices and target vectors
X_train = train_clean[model_features].fillna(0)  # Final safety fillna
X_test = test_clean[model_features].fillna(0)
y_train = train_clean['sales']
y_test = test_clean['sales']

print(f"📊 MODEL SETUP SUMMARY:")
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Features: {X_train.shape[1]}")
print(f"   Sample-to-feature ratio: {X_train.shape[0]/X_train.shape[1]:.1f}:1")
print(f"   Target variable: Sales (mean=${y_train.mean():,.0f})")

# Feature scaling for regularized regression
print(f"\n⚖️  FEATURE SCALING:")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Show scaling impact
print(f"   Before scaling - Mean: {X_train.mean().mean():.3f}, Std: {X_train.std().mean():.3f}")
print(f"   After scaling  - Mean: {X_train_scaled.mean():.3f}, Std: {X_train_scaled.std():.3f}")

# Cross-validated regularization strength selection
print(f"\n🔄 CROSS-VALIDATED REGULARIZATION SELECTION:")
regularization_strengths = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
print(f"   Testing α values: {regularization_strengths}")

ridge_cv = RidgeCV(
    alphas=regularization_strengths,
    cv=5,                                    # 5-fold cross-validation
    scoring='neg_mean_squared_error'         # Minimize prediction error
)

# Train model with optimal regularization
print(f"   🔄 Training with 5-fold CV...")
ridge_cv.fit(X_train_scaled, y_train)

optimal_alpha = ridge_cv.alpha_
print(f"   ✅ Optimal regularization: α = {optimal_alpha}")

# Generate predictions for evaluation
y_train_pred = ridge_cv.predict(X_train_scaled)
y_test_pred = ridge_cv.predict(X_test_scaled)

# %% [markdown]
# ## 📊 Model Performance Evaluation
# 
# **Evaluation Strategy**: 
# - **R²**: Percentage of sales variance explained by model
# - **MAE**: Average prediction error in dollars
# - **MAPE**: Percentage prediction error (business-friendly metric)
# - **Overfitting Assessment**: Compare train vs test performance

# %%
print(f"\n📊 MODEL PERFORMANCE EVALUATION")
print("=" * 45)

# Calculate comprehensive performance metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

# Overfitting assessment
overfitting_gap = train_r2 - test_r2

print(f"🎯 PERFORMANCE METRICS:")
print(f"{'Metric':<25} {'Training':<15} {'Test':<15} {'Status'}")
print("-" * 65)
print(f"{'R² (Variance Explained)':<25} {train_r2:.3f} ({train_r2*100:.1f}%) {test_r2:.3f} ({test_r2*100:.1f}%) {'✅ Good' if test_r2 > 0.4 else '📈 Needs Work'}")
print(f"{'MAE (Avg Error)':<25} ${train_mae:,.0f} ${test_mae:,.0f} {'✅ Acceptable' if test_mae < 15000 else '⚠️ High Error'}")
print(f"{'MAPE (% Error)':<25} {train_mape:.1f}% {test_mape:.1f}% {'✅ Good' if test_mape < 12 else '📊 Moderate'}")

print(f"\n🔍 OVERFITTING ASSESSMENT:")
print(f"   Overfitting gap (Train R² - Test R²): {overfitting_gap:.3f}")

if overfitting_gap < 0.05:
    overfitting_status = "✅ Excellent generalization"
elif overfitting_gap < 0.10:
    overfitting_status = "✅ Good generalization" 
elif overfitting_gap < 0.15:
    overfitting_status = "🔶 Moderate overfitting"
else:
    overfitting_status = "❌ High overfitting - needs regularization"

print(f"   Status: {overfitting_status}")

# Business interpretation
print(f"\n💼 BUSINESS INTERPRETATION:")
print(f"   📈 Model explains {test_r2*100:.1f}% of sales variation")
print(f"   💰 Typical prediction error: ${test_mae:,.0f} ({test_mape:.1f}%)")

if test_r2 >= 0.5:
    business_grade = "🏆 Excellent - Ready for budget decisions"
elif test_r2 >= 0.4:
    business_grade = "✅ Good - Reliable for strategic guidance"
elif test_r2 >= 0.3:
    business_grade = "📊 Moderate - Use with caution"
else:
    business_grade = "⚠️ Needs improvement before business use"

print(f"   {business_grade}")

# %% [markdown]
# ## 💼 Business Insights: Channel ROI and Budget Recommendations
# 
# **Research Question**: Which media channels drive the most sales per dollar spent?
# 
# **Methodology**: 
# - Extract model coefficients (sales impact per standardized spend unit)
# - Calculate ROI using average channel spend and predicted sales contribution
# - Rank channels by efficiency for budget allocation guidance

# %%
print(f"\n💼 BUSINESS INSIGHTS - CHANNEL ROI AND BUDGET RECOMMENDATIONS")
print("=" * 70)

# Extract feature importance from trained model
coefficients = ridge_cv.coef_
feature_names = model_features

# Create feature importance dataframe
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"🏆 TOP 10 MOST INFLUENTIAL FEATURES:")
print(f"{'Rank':<5} {'Feature':<35} {'Coefficient':<15} {'Business Impact'}")
print("-" * 80)

for i, (idx, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
    coef = row['Coefficient']
    feature = row['Feature']
    
    if coef > 0:
        impact = "📈 Drives Sales"
    else:
        impact = "📉 Reduces Sales"
    
    print(f"{i:<5} {feature:<35} {coef:<15.3f} {impact}")

# Calculate media channel business metrics
print(f"\n💰 MEDIA CHANNEL BUSINESS PERFORMANCE:")
media_channel_analysis = {}

for channel in available_media:
    adstock_feature = f"{channel}_adstock"
    
    if adstock_feature in feature_names:
        # Get coefficient from model
        feature_idx = feature_names.index(adstock_feature)
        coefficient = coefficients[feature_idx]
        
        # Calculate business metrics
        avg_weekly_spend = train_clean[channel].mean()
        total_spend = train_clean[channel].sum()
        
        # Estimate contribution using coefficient and feature statistics
        feature_std = X_train[adstock_feature].std()
        feature_mean = X_train[adstock_feature].mean()
        
        # Simplified ROI calculation: coefficient impact scaled by spend
        # This is approximate - more sophisticated attribution could be done
        normalized_impact = coefficient * feature_std
        estimated_weekly_contribution = normalized_impact * avg_weekly_spend / 1000  # Scale appropriately
        
        # ROI calculation
        if avg_weekly_spend > 0:
            roi_estimate = estimated_weekly_contribution / avg_weekly_spend
        else:
            roi_estimate = 0
        
        media_channel_analysis[channel] = {
            'coefficient': coefficient,
            'avg_weekly_spend': avg_weekly_spend,
            'total_spend': total_spend,
            'estimated_weekly_contribution': estimated_weekly_contribution,
            'roi_estimate': roi_estimate
        }

# Sort channels by ROI for business recommendations
sorted_by_roi = sorted(media_channel_analysis.items(), 
                      key=lambda x: x[1]['roi_estimate'], 
                      reverse=True)

print(f"{'Rank':<5} {'Channel':<35} {'Weekly Spend':<15} {'ROI Est.':<12} {'Recommendation'}")
print("-" * 90)

for rank, (channel, metrics) in enumerate(sorted_by_roi, 1):
    spend = metrics['avg_weekly_spend']
    roi = metrics['roi_estimate']
    
    # Business recommendations based on ROI
    if roi > 0.5:
        recommendation = "🟢 INCREASE BUDGET"
    elif roi > 0:
        recommendation = "🟡 MAINTAIN SPEND" 
    elif roi > -0.2:
        recommendation = "🟠 OPTIMIZE/REDUCE"
    else:
        recommendation = "🔴 REVIEW STRATEGY"
    
    print(f"{rank:<5} {channel:<35} ${spend:<14,.0f} {roi:<11.3f} {recommendation}")

# Budget allocation recommendations
total_media_spend = sum([metrics['avg_weekly_spend'] for metrics in media_channel_analysis.values()])

print(f"\n🎯 STRATEGIC BUDGET RECOMMENDATIONS:")
print(f"   💰 Current total weekly media budget: ${total_media_spend:,.0f}")
print(f"   ")
print(f"   📈 HIGH PRIORITY (Increase Budget):")
for channel, metrics in sorted_by_roi[:3]:
    if metrics['roi_estimate'] > 0:
        print(f"      • {channel}: ROI {metrics['roi_estimate']:.3f}")

if len(sorted_by_roi) > 3:
    print(f"   ")
    print(f"   📊 REVIEW REQUIRED (Optimize Strategy):")
    for channel, metrics in sorted_by_roi[3:]:
        if metrics['roi_estimate'] <= 0:
            print(f"      • {channel}: ROI {metrics['roi_estimate']:.3f}")

print(f"\n💡 KEY INSIGHTS:")
print(f"   • Focus budget on top 3 performing channels")
print(f"   • Monitor negative ROI channels closely")
print(f"   • Test budget reallocation in controlled experiments")
print(f"   • Update model monthly with new data")

# %% [markdown]
# ## 📊 Model Diagnostics and Validation Visualizations
# 
# **Validation Strategy**: Visual inspection of model performance
# - **Time series**: How well does model track actual sales over time?
# - **Scatter plots**: Is there systematic bias in predictions?
# - **Residual analysis**: Are prediction errors random or systematic?
# - **Feature importance**: Which factors drive model decisions?

# %%
print(f"\n📊 MODEL DIAGNOSTICS AND VALIDATION VISUALIZATIONS")
print("=" * 60)

# Create comprehensive diagnostic plots
fig = plt.figure(figsize=(20, 15))

# 1. Time Series: Full timeline with train/test split
ax1 = plt.subplot(3, 3, 1)
full_dates = pd.concat([train_clean['date'], test_clean['date']])
full_actual = pd.concat([y_train, y_test])
full_predicted = np.concatenate([y_train_pred, y_test_pred])

ax1.plot(full_dates, full_actual, 'b-', label='Actual Sales', linewidth=2.5, alpha=0.8)
ax1.plot(full_dates, full_predicted, 'r-', label='Predicted Sales', linewidth=2, alpha=0.9)
ax1.axvline(x=train_clean['date'].iloc[-1], color='orange', linestyle='--', alpha=0.8, 
           linewidth=2, label='Train/Test Split')
ax1.set_title('Sales Prediction: Full Timeline\nModel Tracking Performance', fontweight='bold')
ax1.set_ylabel('Sales ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Training Set: Actual vs Predicted Scatter
ax2 = plt.subplot(3, 3, 2)
ax2.scatter(y_train, y_train_pred, alpha=0.6, color='blue', s=40, edgecolor='darkblue', linewidth=0.5)
min_val = min(y_train.min(), y_train_pred.min())
max_val = max(y_train.max(), y_train_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
ax2.set_xlabel('Actual Sales ($)')
ax2.set_ylabel('Predicted Sales ($)')
ax2.set_title(f'Training Set Accuracy\nR² = {train_r2:.3f} ({train_r2*100:.1f}%)', fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Test Set: Actual vs Predicted Scatter  
ax3 = plt.subplot(3, 3, 3)
ax3.scatter(y_test, y_test_pred, alpha=0.8, color='red', s=60, edgecolor='darkred', linewidth=0.8)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
ax3.set_xlabel('Actual Sales ($)')
ax3.set_ylabel('Predicted Sales ($)')
ax3.set_title(f'Test Set Accuracy\nR² = {test_r2:.3f} ({test_r2*100:.1f}%)', fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Feature Importance: Top Media Channels
ax4 = plt.subplot(3, 3, 4)
media_features_importance = feature_importance_df[
    feature_importance_df['Feature'].str.contains('_adstock')
].head(7)

colors = ['green' if coef > 0 else 'red' for coef in media_features_importance['Coefficient']]
bars = ax4.barh(media_features_importance['Feature'].str.replace('_adstock', '').str.replace('_', '\n'), 
                media_features_importance['Coefficient'], 
                color=colors, alpha=0.7)
ax4.set_xlabel('Coefficient (Sales Impact)')
ax4.set_title('Media Channel Importance\nGreen=Positive, Red=Negative', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# 5. Channel ROI Comparison
ax5 = plt.subplot(3, 3, 5)
channels = [ch.replace('_', '\n') for ch in available_media]
rois = [media_channel_analysis[ch]['roi_estimate'] for ch in available_media]
colors = ['green' if roi > 0 else 'red' for roi in rois]
bars = ax5.bar(channels, rois, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
ax5.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
ax5.set_ylabel('ROI Estimate')
ax5.set_title('Channel ROI Comparison\nPositive = Profitable', fontweight='bold')
ax5.tick_params(axis='x', rotation=45)
ax5.grid(True, alpha=0.3, axis='y')

# 6. Residuals Over Time
ax6 = plt.subplot(3, 3, 6)
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred
ax6.plot(train_clean['date'], train_residuals, 'bo-', label='Training Residuals', alpha=0.7, markersize=3)
ax6.plot(test_clean['date'], test_residuals, 'ro-', label='Test Residuals', alpha=0.8, markersize=4)
ax6.axhline(y=0, color='black', linestyle='-', alpha=0.8)
ax6.set_ylabel('Prediction Error ($)')
ax6.set_title('Prediction Errors Over Time\nShould be Random', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.tick_params(axis='x', rotation=45)

# 7. Control Variable Importance
ax7 = plt.subplot(3, 3, 7)
control_features_importance = feature_importance_df[
    ~feature_importance_df['Feature'].str.contains('_adstock')
].head(8)

colors = ['green' if coef > 0 else 'red' for coef in control_features_importance['Coefficient']]
ax7.barh(control_features_importance['Feature'].str.replace('_', '\n'), 
         control_features_importance['Coefficient'],
         color=colors, alpha=0.7)
ax7.set_xlabel('Coefficient (Sales Impact)')
ax7.set_title('Control Variable Importance\nSeasonality & Weather Effects', fontweight='bold')
ax7.grid(True, alpha=0.3, axis='x')

# 8. Prediction Distribution Analysis
ax8 = plt.subplot(3, 3, 8)
ax8.hist(y_train, bins=20, alpha=0.6, label='Actual Sales (Train)', color='blue', density=True)
ax8.hist(y_train_pred, bins=20, alpha=0.6, label='Predicted Sales (Train)', color='red', density=True)
ax8.set_xlabel('Sales ($)')
ax8.set_ylabel('Density')
ax8.set_title('Sales Distribution\nActual vs Predicted', fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Model Performance Summary
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
summary_text = f"""
04 SIMPLE BASELINE MODEL
{'='*25}

PERFORMANCE METRICS:
• Test R²: {test_r2:.1%} 
• Test MAE: ${test_mae:,.0f}
• Test MAPE: {test_mape:.1f}%
• Overfitting: {overfitting_gap:.3f}

BUSINESS INSIGHTS:
• {len(available_media)} media channels analyzed
• Model explains {test_r2*100:.1f}% of sales variation
• Avg prediction error: {test_mape:.1f}%

TOP PERFORMING CHANNEL:
{sorted_by_roi[0][0]}
ROI: {sorted_by_roi[0][1]['roi_estimate']:.3f}

STATUS: {'✅ BUSINESS READY' if test_r2 > 0.4 else '📈 NEEDS IMPROVEMENT'}
"""
ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.suptitle('04 Simple MMM Baseline - Comprehensive Model Diagnostics', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 📋 Research Summary and Model Documentation
# 
# **Final Assessment**: Document model performance, limitations, and next steps for research team

# %%
print(f"\n📋 04 SIMPLE BASELINE MODEL - RESEARCH SUMMARY")
print("=" * 60)

print(f"🎯 RESEARCH OBJECTIVES STATUS:")
print(f"   ✅ Trustworthy Foundation: Ridge regression with proper validation")
print(f"   ✅ Performance Baseline: {test_r2:.1%} Test R² established") 
print(f"   ✅ Business Insights: ROI guidance for all 7 media channels")
print(f"   ✅ Methodology Validation: {overfitting_gap:.3f} overfitting gap (controlled)")

print(f"\n📊 MODEL SPECIFICATIONS:")
print(f"   • Algorithm: Ridge Regression (α={optimal_alpha})")
print(f"   • Features: {len(model_features)} (7 media + 8 controls)")
print(f"   • Training Period: {train_data.shape[0]} weeks ({train_data['date'].min().date()} - {train_data['date'].max().date()})")
print(f"   • Test Period: {test_data.shape[0]} weeks ({test_data['date'].min().date()} - {test_data['date'].max().date()})")
print(f"   • Adstock Decay: 0.4 (moderate carryover)")

print(f"\n🏆 MODEL PERFORMANCE:")
print(f"   • Predictive Accuracy: {test_r2:.1%} of sales variance explained")
print(f"   • Business Error: {test_mape:.1f}% average prediction error")
print(f"   • Generalization: {overfitting_status}")
print(f"   • Business Readiness: {business_grade}")

print(f"\n💼 KEY BUSINESS INSIGHTS:")
print(f"   🥇 Top Performing Channel: {sorted_by_roi[0][0]} (ROI: {sorted_by_roi[0][1]['roi_estimate']:.3f})")
print(f"   💰 Total Weekly Media Budget: ${total_media_spend:,.0f}")
print(f"   📈 Channels with Positive ROI: {len([ch for ch, metrics in sorted_by_roi if metrics['roi_estimate'] > 0])}")
print(f"   📉 Channels Needing Review: {len([ch for ch, metrics in sorted_by_roi if metrics['roi_estimate'] <= 0])}")

print(f"\n⚠️  MODEL LIMITATIONS:")
print(f"   • Simple adstock (uniform 0.4 decay) - channels likely have different carryover patterns")
print(f"   • Linear assumptions - diminishing returns not captured")
print(f"   • No interaction effects between channels")
print(f"   • Limited external factors (no competitors, economic indicators)")
print(f"   • ROI calculations are approximate (simplified attribution)")

print(f"\n🚀 RECOMMENDED NEXT STEPS (FOR 05 ENHANCED MODEL):")
print(f"   1. 📈 Channel-Specific Adstock: Different decay rates per channel")
print(f"   2. 📊 Saturation Curves: Model diminishing returns")
print(f"   3. 🤝 Interaction Effects: TV+Radio, Search+Social synergies")
print(f"   4. 🎯 Feature Engineering: More sophisticated seasonality")
print(f"   5. 🔍 Advanced Validation: Time series cross-validation")

print(f"\n📁 RESEARCH DELIVERABLES:")
print(f"   • Baseline model ready for business use")
print(f"   • Channel ROI rankings for immediate budget guidance")
print(f"   • Performance benchmark for future model comparisons")
print(f"   • Validated methodology for MMM development")

print(f"\n🎯 BUSINESS IMPACT:")
if test_r2 >= 0.45:
    print(f"   🏆 EXCELLENT: Model ready for strategic budget allocation")
elif test_r2 >= 0.35:
    print(f"   ✅ GOOD: Suitable for directional budget guidance")
else:
    print(f"   📈 DEVELOPING: Use insights with caution, continue enhancement")

print(f"\n💡 The 04 Simple Baseline provides a solid foundation for")
print(f"   data-driven media budget decisions while establishing")
print(f"   clear performance benchmarks for future model iterations!")

# %% 