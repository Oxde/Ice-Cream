# %%
# 🎯 FAIR MODEL COMPARISON
# ================================================================
# Goal: Compare all our best techniques on the SAME dataset
# Dataset: Complete channels (2022-2023) for fair comparison

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("🎯 FAIR MODEL COMPARISON - SAME DATASET")
print("=" * 50)
print("📊 Using: Complete channels 2022-2023 (best performing period)")

# %%
# Load the best performing dataset (complete channels)
data = pd.read_csv('data/mmm_ready/mmm_complete_channels_2022_2023.csv')
data['date'] = pd.to_datetime(data['date'])

print(f"✅ Loaded data: {data.shape[0]} weeks from {data['date'].min().date()} to {data['date'].max().date()}")

# Use same train/test split as 06 model for fair comparison
split_idx = int(len(data) * 0.8)  # 80/20 split
train_data = data.iloc[:split_idx].copy()
test_data = data.iloc[split_idx:].copy()

print(f"📊 Split: {len(train_data)} train, {len(test_data)} test weeks")

# %%
# Define our model approaches
media_channels = [
    'search_cost', 'tv_branding_tv_branding_cost', 'social_costs', 
    'ooh_ooh_spend', 'radio_national_radio_national_cost', 
    'radio_local_radio_local_cost', 'tv_promo_tv_promo_cost'
]

control_variables = [
    'month_sin', 'month_cos', 'week_sin', 'week_cos', 'holiday_period',
    'weather_temperature_mean', 'weather_sunshine_duration', 
    'email_email_campaigns', 'promo_promotion_type'
]

# %%
# APPROACH 1: Simple Baseline (no transformations)
print(f"\n1️⃣ SIMPLE BASELINE MODEL")
print("=" * 30)

def build_simple_model(train_data, test_data):
    """Simple model with raw features"""
    all_features = media_channels + control_variables
    
    X_train = train_data[all_features].fillna(0)
    X_test = test_data[all_features].fillna(0)
    y_train = train_data['sales']
    y_test = test_data['sales']
    
    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RidgeCV(alphas=[1, 5, 10, 20, 50, 100], cv=3)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    return {
        'name': 'Simple Baseline',
        'train_r2': train_r2,
        'test_r2': test_r2,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'dates': test_data['date']
    }

simple_results = build_simple_model(train_data, test_data)
print(f"   Train R²: {simple_results['train_r2']:.3f}")
print(f"   Test R²: {simple_results['test_r2']:.3f}")

# %%
# APPROACH 2: With Adstock (from 06 baseline)
print(f"\n2️⃣ ADSTOCK MODEL (06 Baseline)")
print("=" * 35)

def apply_adstock(x, decay_rate):
    """Apply adstock transformation"""
    adstocked = np.zeros_like(x)
    adstocked[0] = x[0]
    for i in range(1, len(x)):
        adstocked[i] = x[i] + decay_rate * adstocked[i-1]
    return adstocked

# Channel-specific decay rates (from industry knowledge)
decay_rates = {
    'search_cost': 0.2,  # Search - short memory
    'social_costs': 0.3,  # Social - medium-short memory  
    'radio_local_radio_local_cost': 0.4,  # Local radio
    'tv_promo_tv_promo_cost': 0.4,  # TV promo
    'radio_national_radio_national_cost': 0.5,  # National radio
    'ooh_ooh_spend': 0.6,  # OOH - longer memory
    'tv_branding_tv_branding_cost': 0.6,  # TV branding - long memory
}

def build_adstock_model(train_data, test_data):
    """Model with channel-specific adstock"""
    train_with_adstock = train_data.copy()
    test_with_adstock = test_data.copy()
    
    # Apply adstock to each channel
    adstock_features = []
    for channel in media_channels:
        if channel in decay_rates:
            decay_rate = decay_rates[channel]
            
            # Apply to train data
            train_adstock = apply_adstock(train_data[channel].fillna(0).values, decay_rate)
            train_with_adstock[f"{channel}_adstock"] = train_adstock
            
            # Apply to test data (continuing from train)
            combined_data = np.concatenate([train_data[channel].fillna(0).values, 
                                          test_data[channel].fillna(0).values])
            combined_adstock = apply_adstock(combined_data, decay_rate)
            test_adstock = combined_adstock[len(train_data):]
            test_with_adstock[f"{channel}_adstock"] = test_adstock
            
            adstock_features.append(f"{channel}_adstock")
        else:
            # Keep original if no decay rate defined
            adstock_features.append(channel)
    
    # Build model with adstock features
    all_features = adstock_features + control_variables
    X_train = train_with_adstock[all_features].fillna(0)
    X_test = test_with_adstock[all_features].fillna(0)
    y_train = train_data['sales']
    y_test = test_data['sales']
    
    # Feature selection (top 10)
    selector = SelectKBest(score_func=f_regression, k=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    model = RidgeCV(alphas=[1, 5, 10, 20, 50, 100], cv=3)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    return {
        'name': 'Adstock Model',
        'train_r2': train_r2,
        'test_r2': test_r2,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'dates': test_data['date'],
        'adstock_features': adstock_features
    }

adstock_results = build_adstock_model(train_data, test_data)
print(f"   Train R²: {adstock_results['train_r2']:.3f}")
print(f"   Test R²: {adstock_results['test_r2']:.3f}")

# %%
# APPROACH 3: Adstock + Saturation (06 Enhanced)
print(f"\n3️⃣ ADSTOCK + SATURATION MODEL (06 Enhanced)")
print("=" * 45)

def simple_saturation(x, half_saturation_point):
    """Simple saturation: x / (1 + x/half_sat)"""
    return x / (1 + x / half_saturation_point)

def build_saturation_model(train_data, test_data):
    """Model with adstock + saturation"""
    # First get adstock features
    train_with_adstock = train_data.copy()
    test_with_adstock = test_data.copy()
    
    adstock_features = []
    for channel in media_channels:
        if channel in decay_rates:
            decay_rate = decay_rates[channel]
            
            # Apply adstock
            train_adstock = apply_adstock(train_data[channel].fillna(0).values, decay_rate)
            train_with_adstock[f"{channel}_adstock"] = train_adstock
            
            combined_data = np.concatenate([train_data[channel].fillna(0).values, 
                                          test_data[channel].fillna(0).values])
            combined_adstock = apply_adstock(combined_data, decay_rate)
            test_adstock = combined_adstock[len(train_data):]
            test_with_adstock[f"{channel}_adstock"] = test_adstock
            
            adstock_features.append(f"{channel}_adstock")
    
    # Now apply saturation to adstock features
    saturated_features = []
    for adstock_feature in adstock_features:
        if adstock_feature in train_with_adstock.columns:
            # Calculate half-saturation from training data
            train_values = train_with_adstock[adstock_feature].fillna(0)
            half_sat = np.median(train_values[train_values > 0]) if np.any(train_values > 0) else 100
            
            # Apply saturation
            train_saturated = simple_saturation(train_values.values, half_sat)
            test_saturated = simple_saturation(test_with_adstock[adstock_feature].fillna(0).values, half_sat)
            
            saturated_feature = f"{adstock_feature}_saturated"
            train_with_adstock[saturated_feature] = train_saturated
            test_with_adstock[saturated_feature] = test_saturated
            
            saturated_features.append(saturated_feature)
    
    # Build model with saturated features
    all_features = saturated_features + control_variables
    X_train = train_with_adstock[all_features].fillna(0)
    X_test = test_with_adstock[all_features].fillna(0)
    y_train = train_data['sales']
    y_test = test_data['sales']
    
    # Feature selection (top 10)
    selector = SelectKBest(score_func=f_regression, k=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    model = RidgeCV(alphas=[1, 5, 10, 20, 50, 100], cv=3)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    return {
        'name': 'Adstock + Saturation',
        'train_r2': train_r2,
        'test_r2': test_r2,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'dates': test_data['date']
    }

saturation_results = build_saturation_model(train_data, test_data)
print(f"   Train R²: {saturation_results['train_r2']:.3f}")
print(f"   Test R²: {saturation_results['test_r2']:.3f}")

# %%
# APPROACH 4: 05 Enhanced Approach (Advanced Adstock + Interactions)
print(f"\n4️⃣ ADVANCED MODEL (05 Enhanced Style)")
print("=" * 40)

def apply_advanced_adstock(x, decay_rate, conc_param=1.0):
    """Advanced adstock with concentration parameter"""
    adstocked = np.zeros_like(x)
    if len(x) > 0:
        adstocked[0] = x[0]
        for i in range(1, len(x)):
            adstocked[i] = x[i] + decay_rate * (adstocked[i-1] ** conc_param)
    return adstocked

def build_advanced_model(train_data, test_data):
    """Advanced model like 05 enhanced"""
    train_advanced = train_data.copy()
    test_advanced = test_data.copy()
    
    # Advanced adstock parameters
    advanced_params = {
        'search_cost': {'decay': 0.2, 'conc': 1.0},
        'tv_branding_tv_branding_cost': {'decay': 0.7, 'conc': 0.8},
        'social_costs': {'decay': 0.3, 'conc': 1.2},
        'ooh_ooh_spend': {'decay': 0.6, 'conc': 0.9},
        'radio_national_radio_national_cost': {'decay': 0.5, 'conc': 1.0},
        'radio_local_radio_local_cost': {'decay': 0.4, 'conc': 1.1},
        'tv_promo_tv_promo_cost': {'decay': 0.4, 'conc': 1.3}
    }
    
    advanced_features = []
    for channel in media_channels:
        if channel in advanced_params:
            params = advanced_params[channel]
            
            # Apply advanced adstock
            train_adstock = apply_advanced_adstock(
                train_data[channel].fillna(0).values, 
                params['decay'], params['conc']
            )
            train_advanced[f"{channel}_adstock"] = train_adstock
            
            # For test, continue from training
            combined_data = np.concatenate([train_data[channel].fillna(0).values, 
                                          test_data[channel].fillna(0).values])
            combined_adstock = apply_advanced_adstock(combined_data, params['decay'], params['conc'])
            test_adstock = combined_adstock[len(train_data):]
            test_advanced[f"{channel}_adstock"] = test_adstock
            
            advanced_features.append(f"{channel}_adstock")
    
    # Add channel interactions
    if 'tv_branding_tv_branding_cost_adstock' in train_advanced.columns and 'tv_promo_tv_promo_cost_adstock' in train_advanced.columns:
        train_advanced['tv_synergy'] = (
            train_advanced['tv_branding_tv_branding_cost_adstock'] * 
            train_advanced['tv_promo_tv_promo_cost_adstock']
        )
        test_advanced['tv_synergy'] = (
            test_advanced['tv_branding_tv_branding_cost_adstock'] * 
            test_advanced['tv_promo_tv_promo_cost_adstock']
        )
        advanced_features.append('tv_synergy')
    
    if 'radio_national_radio_national_cost_adstock' in train_advanced.columns and 'radio_local_radio_local_cost_adstock' in train_advanced.columns:
        train_advanced['radio_synergy'] = (
            train_advanced['radio_national_radio_national_cost_adstock'] * 
            train_advanced['radio_local_radio_local_cost_adstock']
        )
        test_advanced['radio_synergy'] = (
            test_advanced['radio_national_radio_national_cost_adstock'] * 
            test_advanced['radio_local_radio_local_cost_adstock']
        )
        advanced_features.append('radio_synergy')
    
    # Build model
    all_features = advanced_features + control_variables
    X_train = train_advanced[all_features].fillna(0)
    X_test = test_advanced[all_features].fillna(0)
    y_train = train_data['sales']
    y_test = test_data['sales']
    
    # Feature selection (top 10)
    selector = SelectKBest(score_func=f_regression, k=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    model = RidgeCV(alphas=[1, 5, 10, 20, 50, 100], cv=3)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    return {
        'name': 'Advanced (05 Style)',
        'train_r2': train_r2,
        'test_r2': test_r2,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'dates': test_data['date']
    }

advanced_results = build_advanced_model(train_data, test_data)
print(f"   Train R²: {advanced_results['train_r2']:.3f}")
print(f"   Test R²: {advanced_results['test_r2']:.3f}")

# %%
# RESULTS COMPARISON
print(f"\n🏆 FINAL RESULTS COMPARISON")
print("=" * 50)

results = [simple_results, adstock_results, saturation_results, advanced_results]

print(f"{'Model':<25} {'Train R²':<10} {'Test R²':<10} {'Overfitting':<12}")
print("-" * 60)

best_test_r2 = 0
best_model = None

for result in results:
    train_r2 = result['train_r2']
    test_r2 = result['test_r2']
    gap = train_r2 - test_r2
    
    print(f"{result['name']:<25} {train_r2:.3f}     {test_r2:.3f}     {gap:.3f}")
    
    if test_r2 > best_test_r2:
        best_test_r2 = test_r2
        best_model = result

print(f"\n🏆 WINNER: {best_model['name']} with {best_test_r2:.3f} Test R²")

# %%
# VISUALIZATION
print(f"\n📊 VISUALIZATION: ACTUAL vs PREDICTED")
print("=" * 45)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, result in enumerate(results):
    ax = axes[i]
    
    # Time series plot
    ax.plot(result['dates'], result['y_test'], 'b-', label='Actual', linewidth=2, alpha=0.7)
    ax.plot(result['dates'], result['y_test_pred'], 'r--', label='Predicted', linewidth=2)
    ax.set_title(f"{result['name']}\nTest R² = {result['test_r2']:.3f}")
    ax.set_ylabel('Sales')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Model Performance Comparison - Same Dataset', fontsize=16, y=1.02)
plt.show()

print(f"\n✅ CONCLUSION:")
print(f"   • All models tested on same dataset (complete channels 2022-2023)")
print(f"   • Same train/test split for fair comparison")
print(f"   • Best approach: {best_model['name']} ({best_test_r2:.3f} Test R²)")

if best_model['name'] != 'Simple Baseline':
    improvement = ((best_test_r2 - simple_results['test_r2']) / simple_results['test_r2']) * 100
    print(f"   • Improvement over baseline: +{improvement:.1f}%") 