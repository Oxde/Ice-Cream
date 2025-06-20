# %%
# 🎯 ADVANCED SEASONALITY TEST - ICE CREAM SPECIFIC
# ================================================================
# Goal: Test ice cream specific seasonality patterns to improve 50% baseline
# Focus: Holidays, temperature interactions, week-of-month effects

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

print("🎯 ADVANCED SEASONALITY TEST - ICE CREAM BUSINESS")
print("=" * 55)
print("📊 Goal: Improve 50% baseline with ice cream specific patterns")

# %%
# Load proper train/test datasets
train_data = pd.read_csv('../data/mmm_ready/consistent_channels_train_set.csv')
test_data = pd.read_csv('../data/mmm_ready/consistent_channels_test_set.csv')

train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

print(f"✅ Training: {len(train_data)} weeks, Test: {len(test_data)} weeks")

# Base features
media_channels = [
    'search_cost', 'tv_branding_tv_branding_cost', 'social_costs', 
    'ooh_ooh_spend', 'radio_national_radio_national_cost', 
    'radio_local_radio_local_cost', 'tv_promo_tv_promo_cost'
]

base_controls = [
    'month_sin', 'month_cos', 'week_sin', 'week_cos', 'holiday_period',
    'weather_temperature_mean', 'weather_sunshine_duration', 
    'promo_promotion_type'
]

# %%
# ADVANCED SEASONALITY FEATURE ENGINEERING
print(f"\n🌟 ADVANCED SEASONALITY ENGINEERING")
print("=" * 40)

def create_advanced_seasonality(data):
    """Create ice cream specific seasonality features"""
    enhanced_data = data.copy()
    
    # Extract date components
    enhanced_data['month'] = enhanced_data['date'].dt.month
    enhanced_data['day_of_year'] = enhanced_data['date'].dt.dayofyear
    enhanced_data['week_of_month'] = (enhanced_data['date'].dt.day - 1) // 7 + 1
    enhanced_data['is_weekend'] = enhanced_data['date'].dt.dayofweek >= 5
    
    # 1. ICE CREAM SPECIFIC HOLIDAYS
    print("   🎆 Ice cream holidays...")
    
    # Memorial Day (last Monday in May) - Ice cream season start
    enhanced_data['memorial_day_week'] = (
        (enhanced_data['month'] == 5) & 
        (enhanced_data['week_of_month'] >= 4)
    ).astype(int)
    
    # July 4th period - Peak ice cream
    enhanced_data['july_4th_period'] = (
        (enhanced_data['month'] == 7) & 
        (enhanced_data['day_of_year'].between(180, 195))
    ).astype(int)
    
    # Labor Day (first Monday in September) - End of summer
    enhanced_data['labor_day_week'] = (
        (enhanced_data['month'] == 9) & 
        (enhanced_data['week_of_month'] == 1)
    ).astype(int)
    
    # School vacation periods (higher ice cream consumption)
    enhanced_data['summer_break'] = (
        (enhanced_data['month'].isin([6, 7, 8]))
    ).astype(int)
    
    enhanced_data['spring_break'] = (
        (enhanced_data['month'] == 3) & 
        (enhanced_data['week_of_month'].isin([2, 3, 4]))
    ).astype(int)
    
    # 2. WEEK-OF-MONTH EFFECTS (payday patterns)
    print("   💰 Payday effects...")
    enhanced_data['week1_of_month'] = (enhanced_data['week_of_month'] == 1).astype(int)
    enhanced_data['week2_of_month'] = (enhanced_data['week_of_month'] == 2).astype(int)
    enhanced_data['week3_of_month'] = (enhanced_data['week_of_month'] == 3).astype(int)
    enhanced_data['week4_of_month'] = (enhanced_data['week_of_month'] == 4).astype(int)
    
    # 3. TEMPERATURE INTERACTION EFFECTS
    print("   🌡️ Temperature interactions...")
    
    # Heat wave effect (temperature > 25°C)
    enhanced_data['heat_wave'] = (enhanced_data['weather_temperature_mean'] > 25).astype(int)
    
    # Unexpected warm days in spring/fall
    enhanced_data['warm_spring'] = (
        (enhanced_data['month'].isin([3, 4, 5])) & 
        (enhanced_data['weather_temperature_mean'] > 20)
    ).astype(int)
    
    enhanced_data['warm_fall'] = (
        (enhanced_data['month'].isin([9, 10, 11])) & 
        (enhanced_data['weather_temperature_mean'] > 18)
    ).astype(int)
    
    # Temperature × Season interactions
    enhanced_data['temp_summer_boost'] = (
        enhanced_data['weather_temperature_mean'] * enhanced_data['summer_break']
    )
    
    # 4. ADVANCED SEASONAL CURVES
    print("   📈 Advanced seasonal curves...")
    
    # Ice cream peak season curve (stronger than simple sin/cos)
    enhanced_data['ice_cream_season'] = np.sin(2 * np.pi * (enhanced_data['day_of_year'] - 80) / 365) ** 2
    
    # Weekend boost (higher ice cream consumption)
    enhanced_data['weekend_boost'] = enhanced_data['is_weekend'].astype(int)
    
    return enhanced_data

# Apply to both datasets
print("   🔄 Applying to train/test data...")
train_enhanced = create_advanced_seasonality(train_data)
test_enhanced = create_advanced_seasonality(test_data)

# New seasonality features
advanced_seasonality = [
    'memorial_day_week', 'july_4th_period', 'labor_day_week',
    'summer_break', 'spring_break',
    'week1_of_month', 'week2_of_month', 'week3_of_month', 'week4_of_month',
    'heat_wave', 'warm_spring', 'warm_fall', 'temp_summer_boost',
    'ice_cream_season', 'weekend_boost'
]

print(f"✅ Created {len(advanced_seasonality)} advanced seasonality features")

# %%
# MODEL COMPARISON TEST
print(f"\n🥊 MODEL COMPARISON")
print("=" * 25)

def test_model(train_data, test_data, features, model_name):
    """Test model with given features"""
    
    X_train = train_data[features].fillna(0)
    X_test = test_data[features].fillna(0)
    y_train = train_data['sales']
    y_test = test_data['sales']
    
    # Feature selection if too many features
    if len(features) > 15:
        selector = SelectKBest(score_func=f_regression, k=15)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        selected_features = selector.get_support()
        feature_names = [features[i] for i, selected in enumerate(selected_features) if selected]
    else:
        X_train_selected = X_train
        X_test_selected = X_test
        feature_names = features
    
    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    model = RidgeCV(alphas=[1, 5, 10, 20, 50, 100], cv=TimeSeriesSplit(n_splits=3))
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    gap = train_r2 - test_r2
    
    print(f"{model_name}:")
    print(f"   Features used: {len(feature_names)}")
    print(f"   Train R²: {train_r2:.3f}")
    print(f"   Test R²: {test_r2:.3f}")
    print(f"   Gap: {gap:.3f}")
    
    if gap < 0.10:
        print(f"   ✅ Low overfitting")
    elif gap < 0.15:
        print(f"   🔶 Moderate overfitting")
    else:
        print(f"   ⚠️  High overfitting")
    
    return {
        'name': model_name,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'gap': gap,
        'features': feature_names,
        'y_test': y_test,
        'y_test_pred': y_test_pred
    }

# Test different feature combinations
print("1️⃣ BASELINE MODEL (Current Best)")
baseline_features = media_channels + base_controls
baseline_results = test_model(train_data, test_data, baseline_features, "Baseline")

print(f"\n2️⃣ + BASIC SEASONALITY")
basic_seasonal_features = baseline_features + ['ice_cream_season', 'weekend_boost']
basic_seasonal_results = test_model(train_enhanced, test_enhanced, basic_seasonal_features, "Basic Seasonal")

print(f"\n3️⃣ + ICE CREAM HOLIDAYS")
holiday_features = basic_seasonal_features + ['memorial_day_week', 'july_4th_period', 'labor_day_week', 'summer_break', 'spring_break']
holiday_results = test_model(train_enhanced, test_enhanced, holiday_features, "Holiday Enhanced")

print(f"\n4️⃣ + TEMPERATURE INTERACTIONS")
temp_features = holiday_features + ['heat_wave', 'warm_spring', 'warm_fall', 'temp_summer_boost']
temp_results = test_model(train_enhanced, test_enhanced, temp_features, "Temperature Enhanced")

print(f"\n5️⃣ + WEEK-OF-MONTH EFFECTS")
full_features = temp_features + ['week1_of_month', 'week2_of_month', 'week3_of_month', 'week4_of_month']
full_results = test_model(train_enhanced, test_enhanced, full_features, "Full Advanced")

# %%
# RESULTS SUMMARY
print(f"\n🏆 RESULTS SUMMARY")
print("=" * 30)

results = [baseline_results, basic_seasonal_results, holiday_results, temp_results, full_results]

print(f"{'Model':<20} {'Test R²':<10} {'vs Baseline':<12} {'Gap':<8} {'Status'}")
print("-" * 65)

baseline_r2 = baseline_results['test_r2']
best_model = None
best_r2 = 0

for result in results:
    test_r2 = result['test_r2']
    improvement = test_r2 - baseline_r2
    gap_status = "✅" if result['gap'] < 0.10 else ("🔶" if result['gap'] < 0.15 else "⚠️")
    
    improvement_str = f"+{improvement:.3f}" if improvement > 0 else f"{improvement:.3f}"
    
    print(f"{result['name']:<20} {test_r2:.3f}      {improvement_str:<12} {result['gap']:.3f}    {gap_status}")
    
    if test_r2 > best_r2 and result['gap'] < 0.15:  # Must have acceptable overfitting
        best_r2 = test_r2
        best_model = result

if best_model:
    improvement_pct = ((best_r2 - baseline_r2) / baseline_r2) * 100
    print(f"\n🎯 BEST MODEL: {best_model['name']}")
    print(f"   Test R²: {best_r2:.3f}")
    print(f"   Improvement: +{improvement_pct:.1f}% over baseline")
    
    if improvement_pct > 5:
        print(f"   🎉 SIGNIFICANT IMPROVEMENT!")
    elif improvement_pct > 2:
        print(f"   ✅ Good improvement")
    else:
        print(f"   💡 Marginal improvement")
else:
    print(f"\n📊 No significant improvement found")

# %%
# FEATURE IMPORTANCE ANALYSIS
if best_model and best_model['name'] != 'Baseline':
    print(f"\n🔍 WHAT HELPED MOST")
    print("=" * 25)
    
    # Test incremental feature additions
    print("🧪 Feature ablation study:")
    
    # Test removing each feature group
    feature_groups = {
        'Ice Cream Season': ['ice_cream_season'],
        'Weekend Effect': ['weekend_boost'],
        'Ice Cream Holidays': ['memorial_day_week', 'july_4th_period', 'labor_day_week', 'summer_break', 'spring_break'],
        'Temperature Effects': ['heat_wave', 'warm_spring', 'warm_fall', 'temp_summer_boost'],
        'Week-of-Month': ['week1_of_month', 'week2_of_month', 'week3_of_month', 'week4_of_month']
    }
    
    for group_name, group_features in feature_groups.items():
        # Test without this group
        test_features = [f for f in best_model['features'] if f not in group_features]
        if len(test_features) >= len(baseline_features):  # Ensure we have enough features
            temp_result = test_model(train_enhanced, test_enhanced, test_features, f"Without {group_name}")
            impact = best_model['test_r2'] - temp_result['test_r2']
            if impact > 0.005:  # Meaningful impact
                print(f"   {group_name}: +{impact:.3f} R² contribution")

# %%
# VISUALIZATION
print(f"\n📊 PERFORMANCE VISUALIZATION")
print("=" * 35)

if best_model:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Baseline vs Best Model
    ax1 = axes[0]
    ax1.plot(test_data['date'], baseline_results['y_test'], 'b-', label='Actual', linewidth=2, alpha=0.7)
    ax1.plot(test_data['date'], baseline_results['y_test_pred'], 'r--', label='Baseline Pred', linewidth=2)
    ax1.plot(test_data['date'], best_model['y_test_pred'], 'g:', label='Enhanced Pred', linewidth=2)
    ax1.set_title(f'Baseline vs Enhanced Model\nBaseline: {baseline_results["test_r2"]:.3f}, Enhanced: {best_model["test_r2"]:.3f}')
    ax1.set_ylabel('Sales')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Improvement over time
    ax2 = axes[1]
    improvement_ts = best_model['y_test_pred'] - baseline_results['y_test_pred']
    ax2.plot(test_data['date'], improvement_ts, 'purple', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('Prediction Improvement Over Time')
    ax2.set_ylabel('Sales Difference (Enhanced - Baseline)')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# %%
# FINAL RECOMMENDATION
print(f"\n🎯 FINAL RECOMMENDATION")
print("=" * 30)

if best_model and best_model != baseline_results:
    improvement_pct = ((best_model['test_r2'] - baseline_r2) / baseline_r2) * 100
    
    if improvement_pct > 3 and best_model['gap'] < 0.12:
        print(f"✅ RECOMMEND: {best_model['name']} model")
        print(f"   🎯 Test R²: {best_model['test_r2']:.3f} (was {baseline_r2:.3f})")
        print(f"   📈 Improvement: +{improvement_pct:.1f}%")
        print(f"   🔍 Overfitting: {best_model['gap']:.3f} (acceptable)")
        print(f"\n💼 BUSINESS IMPACT:")
        print(f"   • Better captures ice cream seasonality")
        print(f"   • More accurate sales predictions")
        print(f"   • Improved media planning for seasonal patterns")
    else:
        print(f"📊 CONTINUE with baseline model")
        print(f"   💡 Advanced seasonality shows marginal improvement")
        print(f"   🎯 Focus on other enhancement areas")
else:
    print(f"📊 CONTINUE with baseline model")
    print(f"   💡 No significant seasonality improvements found")
    print(f"   🎯 Consider other enhancement approaches") 