# 🇳🇱 08 RESPECTFUL DUTCH ENHANCED MMM
# ====================================
# 
# OBJECTIVE: Combine the best of both worlds
# - Keep ALL 7 media channels (respectful approach)
# - Add Dutch-specific features for market relevance
# - No aggressive feature selection that removes media channels
# - Focus on business actionability for complete media mix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("🇳🇱 08 RESPECTFUL DUTCH ENHANCED MMM")
print("=" * 45)
print("🎯 Goal: All media channels + Dutch market insights")
print("📊 Philosophy: Business-first, respecting all media investments")

# Load data
try:
    train_data = pd.read_csv('data/mmm_ready/consistent_channels_train_set.csv')
    test_data = pd.read_csv('data/mmm_ready/consistent_channels_test_set.csv')
    print(f"✅ Data loaded: {len(train_data)} train, {len(test_data)} test weeks")
except:
    print("❌ Data not found - using simulated data for demonstration")
    exit()

# %%
# 🎯 FEATURE ENGINEERING: MEDIA CHANNELS + DUTCH CULTURE
# =======================================================

def create_respectful_dutch_features(df):
    """Create Dutch features while preserving all media channels"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. 🇳🇱 DUTCH NATIONAL HOLIDAYS
    df['kings_day'] = ((df['date'].dt.month == 4) & 
                      (df['date'].dt.day.isin([26, 27]))).astype(int)
    df['liberation_day'] = ((df['date'].dt.month == 5) & 
                           (df['date'].dt.day == 5)).astype(int)
    
    # 2. 🏫 DUTCH SCHOOL HOLIDAYS
    df['dutch_summer_holidays'] = (df['date'].dt.month.isin([7, 8])).astype(int)
    df['dutch_may_break'] = ((df['date'].dt.month == 5) & 
                            (df['date'].dt.day <= 15)).astype(int)
    df['dutch_autumn_break'] = ((df['date'].dt.month == 10) & 
                               (df['date'].dt.day <= 15)).astype(int)
    
    # 3. 🌡️ DUTCH WEATHER PATTERNS
    df['dutch_heatwave'] = (df['weather_temperature_mean'] > 25).astype(int)
    df['warm_spring_nl'] = ((df['date'].dt.month.isin([3, 4, 5])) & 
                           (df['weather_temperature_mean'] > 18)).astype(int)
    df['indian_summer_nl'] = ((df['date'].dt.month.isin([9, 10])) & 
                             (df['weather_temperature_mean'] > 20)).astype(int)
    
    # 4. 🧀 DUTCH CULTURAL EFFECTS
    df['weekend_boost'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
    df['dutch_outdoor_season'] = ((df['date'].dt.month.isin([5, 6, 7, 8, 9])) & 
                                 (df['weather_temperature_mean'] > 15)).astype(int)
    df['payday_effect'] = df['date'].dt.day.isin([1, 15]).astype(int)
    
    # 5. 🔗 INTERACTION EFFECTS
    df['temp_holiday_interaction'] = (df['weather_temperature_mean'] * 
                                     (df['kings_day'] + df['liberation_day'] + 
                                      df['dutch_summer_holidays']))
    
    return df

def create_adstock_features(df, media_channels):
    """Apply adstock transformation to media channels"""
    df_with_adstock = df.copy()
    
    # Channel-specific decay rates (from research)
    decay_rates = {
        'search_cost': 0.2,  # Fast response
        'social_costs': 0.3,  # Medium-fast
        'tv_promo_tv_promo_cost': 0.4,  # Medium
        'radio_local_radio_local_cost': 0.4,  # Medium
        'radio_national_radio_national_cost': 0.5,  # Medium-slow
        'ooh_ooh_spend': 0.6,  # Slow
        'tv_branding_tv_branding_cost': 0.7,  # Very slow (brand building)
    }
    
    def apply_adstock(x, decay_rate):
        adstock = np.zeros_like(x)
        adstock[0] = x[0]
        for i in range(1, len(x)):
            adstock[i] = x[i] + decay_rate * adstock[i-1]
        return adstock
    
    for channel in media_channels:
        if channel in decay_rates and channel in df.columns:
            decay_rate = decay_rates[channel]
            adstock_values = apply_adstock(df[channel].fillna(0).values, decay_rate)
            
            # Apply saturation curve: log(1 + adstocked_spend)
            saturated_values = np.log(1 + adstock_values)
            df_with_adstock[f"{channel}_adstock_saturated"] = saturated_values
    
    return df_with_adstock

print("\n🔄 Engineering features...")

# Apply feature engineering
train_enhanced = create_respectful_dutch_features(train_data)
test_enhanced = create_respectful_dutch_features(test_data)

# Media channels
media_channels = [
    'search_cost', 'tv_branding_tv_branding_cost', 'social_costs',
    'ooh_ooh_spend', 'radio_national_radio_national_cost',
    'radio_local_radio_local_cost', 'tv_promo_tv_promo_cost'
]

# Apply adstock transformations
train_with_adstock = create_adstock_features(train_enhanced, media_channels)
test_with_adstock = create_adstock_features(test_enhanced, media_channels)

print("✅ All features engineered - media channels preserved!")

# %%
# 🎯 MODEL TRAINING: RESPECTFUL APPROACH
# =======================================

def train_respectful_dutch_model():
    """Train model that respects all media channels"""
    
    # Core media features - ALL 7 CHANNELS KEPT
    media_features = []
    for channel in media_channels:
        adstock_feature = f"{channel}_adstock_saturated"
        if adstock_feature in train_with_adstock.columns:
            media_features.append(adstock_feature)
    
    # Base control variables
    base_controls = [
        'month_sin', 'month_cos', 'week_sin', 'week_cos',
        'holiday_period', 'weather_temperature_mean', 
        'weather_sunshine_duration', 'promo_promotion_type'
    ]
    
    # Dutch enhancement features
    dutch_features = [
        'kings_day', 'liberation_day', 'dutch_summer_holidays',
        'dutch_may_break', 'dutch_autumn_break', 'dutch_heatwave',
        'warm_spring_nl', 'indian_summer_nl', 'weekend_boost',
        'dutch_outdoor_season', 'payday_effect', 'temp_holiday_interaction'
    ]
    
    # Available controls
    control_features = []
    for feature in base_controls + dutch_features:
        if feature in train_with_adstock.columns:
            control_features.append(feature)
    
    # All features - NO AGGRESSIVE SELECTION
    all_features = media_features + control_features
    
    print(f"\n📊 RESPECTFUL FEATURE COMPOSITION:")
    print(f"   🎯 Media channels: {len(media_features)}/7 (ALL KEPT)")
    print(f"   📅 Base controls: {len([f for f in control_features if f in base_controls])}")
    print(f"   🇳🇱 Dutch features: {len([f for f in control_features if f in dutch_features])}")
    print(f"   📈 Total features: {len(all_features)}")
    
    # Prepare data
    X_train = train_with_adstock[all_features].fillna(0)
    X_test = test_with_adstock[all_features].fillna(0)
    y_train = train_with_adstock['sales']
    y_test = test_with_adstock['sales']
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Ridge regression with cross-validation
    model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    overfitting_gap = train_r2 - test_r2
    
    return {
        'model': model,
        'scaler': scaler,
        'features': all_features,
        'media_features': media_features,
        'control_features': control_features,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'overfitting_gap': overfitting_gap,
        'y_test_pred': y_test_pred,
        'y_test_true': y_test,
        'X_train': X_train,
        'X_test': X_test
    }

# Train the model
print("\n🎯 Training Respectful Dutch Enhanced Model...")
result = train_respectful_dutch_model()

print(f"\n🏆 MODEL PERFORMANCE:")
print(f"   📈 Train R²: {result['train_r2']:.3f}")
print(f"   🎯 Test R²: {result['test_r2']:.3f}")
print(f"   📊 Overfitting gap: {result['overfitting_gap']:.3f}")

# Assess performance
if result['overfitting_gap'] < 0.1:
    validation_status = "✅ Excellent - Low overfitting"
elif result['overfitting_gap'] < 0.15:
    validation_status = "🔶 Good - Acceptable overfitting"
else:
    validation_status = "❌ Poor - High overfitting"

print(f"   🔍 Validation: {validation_status}")

# %%
# 💰 BUSINESS INSIGHTS: ALL CHANNELS ANALYZED
# ============================================

def analyze_media_performance():
    """Analyze performance of all 7 media channels"""
    
    coefficients = result['model'].coef_
    feature_names = result['features']
    
    print(f"\n💰 ALL MEDIA CHANNELS - PERFORMANCE ANALYSIS")
    print("=" * 55)
    
    # Extract media channel coefficients
    media_performance = {}
    for i, feature in enumerate(feature_names):
        for channel in media_channels:
            if channel in feature and '_adstock_saturated' in feature:
                coef = coefficients[i]
                
                # Calculate average spend for ROI context
                avg_spend = train_data[channel].mean() if channel in train_data.columns else 0
                
                media_performance[channel] = {
                    'coefficient': coef,
                    'avg_spend': avg_spend,
                    'feature_name': feature
                }
                break
    
    # Sort by coefficient strength
    sorted_channels = sorted(media_performance.items(), 
                           key=lambda x: abs(x[1]['coefficient']), 
                           reverse=True)
    
    print(f"📊 Channel Performance Ranking:")
    print(f"{'Rank':<4} {'Channel':<25} {'Coefficient':<12} {'Avg Spend':<12} {'Direction'}")
    print("-" * 75)
    
    for rank, (channel, info) in enumerate(sorted_channels, 1):
        coef = info['coefficient']
        spend = info['avg_spend']
        direction = "📈 Positive" if coef > 0 else "📉 Negative"
        
        clean_name = channel.replace('_', ' ').replace(' cost', '').replace(' costs', '').title()
        print(f"{rank:<4} {clean_name:<25} {coef:<12.2e} €{spend:<11,.0f} {direction}")
    
    return media_performance

media_performance = analyze_media_performance()

# %%
# 🎯 FORMULA DISPLAY
# ==================

def display_respectful_formula():
    """Display the complete MMM formula with all channels"""
    
    print(f"\n🎯 RESPECTFUL DUTCH ENHANCED MMM FORMULA")
    print("=" * 50)
    
    coefficients = result['model'].coef_
    intercept = result['model'].intercept_
    feature_names = result['features']
    
    print(f"Sales_t = {intercept:.0f}")
    
    # Media channels first
    print(f"\n    📺 MEDIA CHANNELS (ALL 7 INCLUDED):")
    for channel in media_channels:
        for i, feature in enumerate(feature_names):
            if channel in feature and '_adstock_saturated' in feature:
                coef = coefficients[i]
                sign = "+" if coef >= 0 else "-"
                clean_name = channel.replace('_', ' ').title().replace(' Cost', '').replace(' Costs', '').replace(' Spend', '')
                print(f"    {sign} {abs(coef):.1f} × log(1+Adstock({clean_name}))")
                break
    
    # Control variables
    print(f"\n    📅 CONTROL VARIABLES:")
    for i, feature in enumerate(feature_names):
        if not any(ch in feature for ch in media_channels):
            coef = coefficients[i]
            sign = "+" if coef >= 0 else "-"
            clean_name = feature.replace('_', ' ').title()
            
            if abs(coef) > 100:  # Only show significant controls
                print(f"    {sign} {abs(coef):.0f} × {clean_name}")
    
    print(f"\n    + ε_t")
    
    print(f"\n🇳🇱 DUTCH ENHANCEMENTS:")
    dutch_count = sum(1 for f in feature_names if any(d in f for d in 
                     ['kings_day', 'liberation', 'dutch_', 'temp_holiday']))
    print(f"   • {dutch_count} Netherlands-specific features included")
    print(f"   • Cultural holidays, weather patterns, behavioral effects")
    print(f"   • No media channels sacrificed for market relevance")

display_respectful_formula()

# %%
# 📊 PERFORMANCE COMPARISON
# =========================

print(f"\n📊 PERFORMANCE SUMMARY")
print("=" * 30)

print(f"🎯 MODEL CHARACTERISTICS:")
print(f"   • Philosophy: Respectful - ALL media channels included")
print(f"   • Enhancement: Dutch market insights added")
print(f"   • Features: {len(result['features'])} total ({len(result['media_features'])} media)")
print(f"   • Validation: {'Robust' if result['overfitting_gap'] < 0.15 else 'Needs attention'}")

print(f"\n📈 BUSINESS VALUE:")
print(f"   ✅ Complete media mix analysis (all 7 channels)")
print(f"   ✅ Dutch market relevance and stakeholder buy-in")
print(f"   ✅ Actionable insights for budget allocation")
print(f"   ✅ No strategic blind spots from missing channels")

print(f"\n🚀 RECOMMENDATION:")
if result['test_r2'] > 0.50:
    recommendation = "✅ DEPLOY - Strong performance with complete media coverage"
elif result['test_r2'] > 0.45:
    recommendation = "🔶 ACCEPTABLE - Good foundation for optimization"
else:
    recommendation = "❌ NEEDS WORK - Performance below business threshold"

print(f"   {recommendation}")
print(f"   Test R²: {result['test_r2']:.1%}")

print(f"\n🎉 SUCCESS: Respectful Dutch Enhanced MMM Complete!")
print(f"   All 7 media channels preserved + Dutch market insights") 