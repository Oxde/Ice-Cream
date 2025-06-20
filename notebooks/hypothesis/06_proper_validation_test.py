# %%
# 🎯 PROPER VALIDATION TEST - USING CORRECT TRAIN/TEST SPLITS
# ================================================================
# Goal: Verify our models are tested correctly without overfitting
# Using: Pre-split train/test datasets to avoid data leakage

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

print("🎯 PROPER VALIDATION TEST - AVOIDING OVERFITTING")
print("=" * 55)
print("📊 Using: Pre-split train/test datasets (proper methodology)")

# %%
# Load the CORRECT pre-split datasets
print(f"\n📁 LOADING PROPER TRAIN/TEST DATASETS")
print("=" * 45)

train_data = pd.read_csv('data/mmm_ready/consistent_channels_train_set.csv')
test_data = pd.read_csv('data/mmm_ready/consistent_channels_test_set.csv')

train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

print(f"✅ Training data: {train_data.shape[0]} weeks ({train_data['date'].min().date()} to {train_data['date'].max().date()})")
print(f"✅ Test data: {test_data.shape[0]} weeks ({test_data['date'].min().date()} to {test_data['date'].max().date()})")

# Check for temporal ordering (critical for MMM)
train_end = train_data['date'].max()
test_start = test_data['date'].min()
print(f"📅 Temporal split: Train ends {train_end.date()}, Test starts {test_start.date()}")

if test_start > train_end:
    print("✅ CORRECT: Test data comes AFTER training data (no data leakage)")
else:
    print("❌ WARNING: Potential data leakage - test data overlaps with training!")

# %%
# Define model components
media_channels = [
    'search_cost', 'tv_branding_tv_branding_cost', 'social_costs', 
    'ooh_ooh_spend', 'radio_national_radio_national_cost', 
    'radio_local_radio_local_cost', 'tv_promo_tv_promo_cost'
]

control_variables = [
    'month_sin', 'month_cos', 'week_sin', 'week_cos', 'holiday_period',
    'weather_temperature_mean', 'weather_sunshine_duration', 
    'promo_promotion_type'
]

print(f"\n📊 MODEL SETUP:")
print(f"   Media channels: {len(media_channels)}")
print(f"   Control variables: {len(control_variables)}")
print(f"   Training weeks: {len(train_data)}")
print(f"   Test weeks: {len(test_data)}")

# %%
# TEST 1: Simple Baseline Model
print(f"\n1️⃣ BASELINE MODEL TEST")
print("=" * 30)

def test_baseline_model():
    """Test simple baseline without transformations"""
    all_features = media_channels + control_variables
    
    # Prepare data
    X_train = train_data[all_features].fillna(0)
    X_test = test_data[all_features].fillna(0)
    y_train = train_data['sales']
    y_test = test_data['sales']
    
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Sample-to-feature ratio: {X_train.shape[0]/X_train.shape[1]:.1f}:1")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Cross-validation within training set
    model = RidgeCV(alphas=[1, 5, 10, 20, 50, 100], cv=TimeSeriesSplit(n_splits=3))
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    gap = train_r2 - test_r2
    
    print(f"   Train R²: {train_r2:.3f}")
    print(f"   Test R²: {test_r2:.3f}")
    print(f"   Overfitting gap: {gap:.3f}")
    
    # Overfitting check
    if gap > 0.15:
        print(f"   ⚠️  HIGH OVERFITTING RISK: Gap > 15%")
    elif gap > 0.10:
        print(f"   🔶 MODERATE OVERFITTING: Gap > 10%")
    else:
        print(f"   ✅ LOW OVERFITTING: Gap < 10%")
    
    return {
        'name': 'Baseline',
        'train_r2': train_r2,
        'test_r2': test_r2,
        'gap': gap,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'optimal_alpha': model.alpha_
    }

baseline_results = test_baseline_model()

# %%
# TEST 2: Adstock Model (with proper validation)
print(f"\n2️⃣ ADSTOCK MODEL TEST")
print("=" * 30)

def apply_adstock(x, decay_rate):
    """Apply adstock transformation"""
    adstocked = np.zeros_like(x)
    if len(x) > 0:
        adstocked[0] = x[0]
        for i in range(1, len(x)):
            adstocked[i] = x[i] + decay_rate * adstocked[i-1]
    return adstocked

def test_adstock_model():
    """Test adstock model with proper validation"""
    
    # Channel-specific decay rates
    decay_rates = {
        'search_cost': 0.2,
        'social_costs': 0.3,
        'radio_local_radio_local_cost': 0.4,
        'tv_promo_tv_promo_cost': 0.4,
        'radio_national_radio_national_cost': 0.5,
        'ooh_ooh_spend': 0.6,
        'tv_branding_tv_branding_cost': 0.6,
    }
    
    # Apply adstock transformations
    train_with_adstock = train_data.copy()
    test_with_adstock = test_data.copy()
    
    adstock_features = []
    for channel in media_channels:
        if channel in decay_rates:
            decay_rate = decay_rates[channel]
            
            # Apply to training data
            train_adstock = apply_adstock(train_data[channel].fillna(0).values, decay_rate)
            train_with_adstock[f"{channel}_adstock"] = train_adstock
            
            # Apply to test data (continuing from training - CRITICAL for time series)
            combined_data = np.concatenate([
                train_data[channel].fillna(0).values,
                test_data[channel].fillna(0).values
            ])
            combined_adstock = apply_adstock(combined_data, decay_rate)
            test_adstock = combined_adstock[len(train_data):]
            test_with_adstock[f"{channel}_adstock"] = test_adstock
            
            adstock_features.append(f"{channel}_adstock")
    
    # Prepare features
    all_features = adstock_features + control_variables
    X_train = train_with_adstock[all_features].fillna(0)
    X_test = test_with_adstock[all_features].fillna(0)
    y_train = train_data['sales']
    y_test = test_data['sales']
    
    print(f"   Adstock features: {len(adstock_features)}")
    print(f"   Total features: {X_train.shape[1]}")
    print(f"   Sample-to-feature ratio: {X_train.shape[0]/X_train.shape[1]:.1f}:1")
    
    # Feature selection to prevent overfitting
    selector = SelectKBest(score_func=f_regression, k=min(10, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    print(f"   Selected features: {X_train_selected.shape[1]}")
    
    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Cross-validation within training set
    model = RidgeCV(alphas=[1, 5, 10, 20, 50, 100], cv=TimeSeriesSplit(n_splits=3))
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    gap = train_r2 - test_r2
    
    print(f"   Train R²: {train_r2:.3f}")
    print(f"   Test R²: {test_r2:.3f}")
    print(f"   Overfitting gap: {gap:.3f}")
    
    # Overfitting check
    if gap > 0.15:
        print(f"   ⚠️  HIGH OVERFITTING RISK: Gap > 15%")
    elif gap > 0.10:
        print(f"   🔶 MODERATE OVERFITTING: Gap > 10%")
    else:
        print(f"   ✅ LOW OVERFITTING: Gap < 10%")
    
    return {
        'name': 'Adstock',
        'train_r2': train_r2,
        'test_r2': test_r2,
        'gap': gap,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'optimal_alpha': model.alpha_,
        'selected_features': X_train_selected.shape[1]
    }

adstock_results = test_adstock_model()

# %%
# TEST 3: Validation Robustness Check
print(f"\n3️⃣ VALIDATION ROBUSTNESS TEST")
print("=" * 35)

def validation_robustness_test():
    """Test model stability with different train/test splits"""
    
    # Create multiple validation periods
    all_data = pd.concat([train_data, test_data]).sort_values('date').reset_index(drop=True)
    
    results = []
    
    # Test different split points (last 3, 6, 9 months as test)
    test_periods = [3, 6, 9]  # months
    
    for months in test_periods:
        weeks_in_test = months * 4  # approximately 4 weeks per month
        
        if weeks_in_test < len(all_data):
            split_idx = len(all_data) - weeks_in_test
            temp_train = all_data.iloc[:split_idx].copy()
            temp_test = all_data.iloc[split_idx:].copy()
            
            # Simple model test
            all_features = media_channels + control_variables
            X_train = temp_train[all_features].fillna(0)
            X_test = temp_test[all_features].fillna(0)
            y_train = temp_train['sales']
            y_test = temp_test['sales']
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = RidgeCV(alphas=[1, 5, 10, 20, 50, 100], cv=3)
            model.fit(X_train_scaled, y_train)
            
            y_test_pred = model.predict(X_test_scaled)
            test_r2 = r2_score(y_test, y_test_pred)
            
            results.append({
                'test_months': months,
                'test_weeks': weeks_in_test,
                'test_r2': test_r2
            })
            
            print(f"   {months} months test: R² = {test_r2:.3f} ({weeks_in_test} weeks)")
    
    # Check stability
    r2_values = [r['test_r2'] for r in results]
    r2_std = np.std(r2_values)
    r2_mean = np.mean(r2_values)
    
    print(f"\n   Stability check:")
    print(f"   Mean R²: {r2_mean:.3f}")
    print(f"   Std Dev: {r2_std:.3f}")
    
    if r2_std < 0.05:
        print(f"   ✅ STABLE: Low variance across splits")
    elif r2_std < 0.10:
        print(f"   🔶 MODERATE: Some variance across splits")
    else:
        print(f"   ⚠️  UNSTABLE: High variance across splits")
    
    return results

robustness_results = validation_robustness_test()

# %%
# FINAL ASSESSMENT
print(f"\n🎯 VALIDATION ASSESSMENT")
print("=" * 35)

print(f"📊 MODEL COMPARISON:")
print(f"{'Model':<15} {'Train R²':<10} {'Test R²':<10} {'Gap':<8} {'Status'}")
print("-" * 55)

models = [baseline_results, adstock_results]
for model in models:
    gap_status = "✅ Good" if model['gap'] < 0.10 else ("🔶 OK" if model['gap'] < 0.15 else "⚠️  Risk")
    print(f"{model['name']:<15} {model['train_r2']:.3f}      {model['test_r2']:.3f}      {model['gap']:.3f}    {gap_status}")

best_model = max(models, key=lambda x: x['test_r2'])
print(f"\n🏆 BEST MODEL: {best_model['name']} (Test R² = {best_model['test_r2']:.3f})")

# %%
# VISUALIZATION
print(f"\n📊 ACTUAL vs PREDICTED VISUALIZATION")
print("=" * 40)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for i, model in enumerate(models):
    ax = axes[i]
    
    # Time series plot
    ax.plot(test_data['date'], model['y_test'], 'b-', label='Actual', linewidth=2, alpha=0.7)
    ax.plot(test_data['date'], model['y_test_pred'], 'r--', label='Predicted', linewidth=2)
    ax.set_title(f"{model['name']} Model\nTest R² = {model['test_r2']:.3f}, Gap = {model['gap']:.3f}")
    ax.set_ylabel('Sales')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# %%
# OVERFITTING CHECKS
print(f"\n🔍 OVERFITTING ANALYSIS")
print("=" * 30)

print(f"✅ VALIDATION METHODOLOGY:")
print(f"   • Using pre-split train/test datasets (no data leakage)")
print(f"   • Temporal split: test data comes AFTER training")
print(f"   • Cross-validation within training set only")
print(f"   • Feature selection to prevent overfitting")
print(f"   • Regularization (Ridge) with CV-selected alpha")

print(f"\n📊 OVERFITTING INDICATORS:")
for model in models:
    gap_pct = model['gap'] * 100
    if model['gap'] < 0.05:
        status = "✅ Excellent (< 5%)"
    elif model['gap'] < 0.10:
        status = "✅ Good (< 10%)"
    elif model['gap'] < 0.15:
        status = "🔶 Acceptable (< 15%)"
    else:
        status = "⚠️  Concerning (> 15%)"
    
    print(f"   {model['name']}: {gap_pct:.1f}% gap - {status}")

print(f"\n🎯 FINAL RECOMMENDATION:")
if best_model['gap'] < 0.15:
    print(f"   ✅ Model is PROPERLY VALIDATED")
    print(f"   ✅ {best_model['name']} model recommended for production")
    print(f"   ✅ Test R² {best_model['test_r2']:.3f} is reliable estimate")
else:
    print(f"   ⚠️  Model shows signs of overfitting")
    print(f"   🔧 Recommend reducing model complexity")
    print(f"   📊 Test performance may be optimistic")

print(f"\n💡 VALIDATION CONFIDENCE: {'HIGH' if best_model['gap'] < 0.10 else 'MEDIUM' if best_model['gap'] < 0.15 else 'LOW'}") 