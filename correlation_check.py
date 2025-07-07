#!/usr/bin/env python3
"""
🔍 CORRELATION AND DISCREPANCY INVESTIGATION
===========================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_data = pd.read_csv('data/mmm_ready/consistent_channels_train_set.csv')
test_data = pd.read_csv('data/mmm_ready/consistent_channels_test_set.csv')

print("🔍 INVESTIGATING MODEL DISCREPANCIES")
print("=" * 50)

# ================================
# 1. CHECK CHANNEL CORRELATIONS
# ================================

print("\n📊 DETAILED CORRELATION ANALYSIS")
print("=" * 40)

# Media columns
media_cols = [col for col in train_data.columns if 'cost' in col or 'spend' in col]
print(f"Media columns found: {len(media_cols)}")
for col in media_cols:
    print(f"   • {col}")

# Check specific correlations mentioned in the original model
tv_branding = train_data['tv_branding_tv_branding_cost']
tv_promo = train_data['tv_promo_tv_promo_cost']
radio_national = train_data['radio_national_radio_national_cost']
radio_local = train_data['radio_local_radio_local_cost']

tv_corr = tv_branding.corr(tv_promo)
radio_corr = radio_national.corr(radio_local)

print(f"\n🔍 CORRELATION VERIFICATION:")
print(f"   • TV Branding ↔ TV Promo: {tv_corr:.3f}")
print(f"   • Radio National ↔ Radio Local: {radio_corr:.3f}")

# Check non-zero spend weeks only
tv_branding_nz = tv_branding[tv_branding > 0]
tv_promo_nz = tv_promo[tv_promo > 0]
if len(tv_branding_nz) > 10 and len(tv_promo_nz) > 10:
    # Find overlapping weeks
    overlap_weeks = train_data[(tv_branding > 0) & (tv_promo > 0)]
    if len(overlap_weeks) > 10:
        tv_corr_overlap = overlap_weeks['tv_branding_tv_branding_cost'].corr(
            overlap_weeks['tv_promo_tv_promo_cost'])
        print(f"   • TV correlation (overlap weeks only): {tv_corr_overlap:.3f}")

# Full correlation matrix for media channels
corr_matrix = train_data[media_cols].corr()
print(f"\n📈 ALL MEDIA CHANNEL CORRELATIONS:")
print("Correlations > 0.7:")
high_corr_found = False
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr = corr_matrix.iloc[i, j]
        if abs(corr) > 0.7:
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            print(f"   • {col1} ↔ {col2}: {corr:.3f}")
            high_corr_found = True

if not high_corr_found:
    print("   • No correlations > 0.7 found!")
    print("\nCorrelations > 0.5:")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.5:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                print(f"   • {col1} ↔ {col2}: {corr:.3f}")

# ================================
# 2. CHECK DATA PROPERTIES
# ================================

print(f"\n📊 DATA PROPERTIES CHECK")
print("=" * 30)

# Check spend levels
tv_total = tv_branding + tv_promo
radio_total = radio_national + radio_local

print(f"Spend levels:")
print(f"   • TV total: ${tv_total.sum():,.0f}")
print(f"   • Radio total: ${radio_total.sum():,.0f}")
print(f"   • Search: ${train_data['search_cost'].sum():,.0f}")
print(f"   • Social: ${train_data['social_costs'].sum():,.0f}")
print(f"   • OOH: ${train_data['ooh_ooh_spend'].sum():,.0f}")

# Check spend distribution
print(f"\nSpend distribution (% of weeks with spend > 0):")
for col in media_cols:
    pct_active = (train_data[col] > 0).mean() * 100
    print(f"   • {col.replace('_', ' ')}: {pct_active:.1f}%")

# ================================
# 3. RECREATE MODEL EXACTLY AS ORIGINAL
# ================================

print(f"\n🔧 RECREATING ORIGINAL MODEL EXACTLY")
print("=" * 40)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

# Exact recreation of the original functions
def aggregate_channels(train_df, test_df):
    """Aggregate correlated channels - EXACT COPY"""
    train_agg = train_df.copy()
    test_agg = test_df.copy()
    
    # TV aggregation
    train_agg['tv_total_spend'] = (train_agg['tv_branding_tv_branding_cost'] + 
                                   train_agg['tv_promo_tv_promo_cost'])
    test_agg['tv_total_spend'] = (test_agg['tv_branding_tv_branding_cost'] + 
                                  test_agg['tv_promo_tv_promo_cost'])
    
    # Radio aggregation
    train_agg['radio_total_spend'] = (train_agg['radio_national_radio_national_cost'] + 
                                      train_agg['radio_local_radio_local_cost'])
    test_agg['radio_total_spend'] = (test_agg['radio_national_radio_national_cost'] + 
                                     test_agg['radio_local_radio_local_cost'])
    
    # Drop original channels
    channels_to_drop = ['tv_branding_tv_branding_cost', 'tv_promo_tv_promo_cost',
                       'radio_national_radio_national_cost', 'radio_local_radio_local_cost']
    
    train_agg = train_agg.drop(columns=channels_to_drop)
    test_agg = test_agg.drop(columns=channels_to_drop)
    
    return train_agg, test_agg

def apply_transformations(df, media_cols):
    """Apply saturation transformations - EXACT COPY"""
    df_transformed = df.copy()
    transformation_log = {}
    
    for col in media_cols:
        if col in df_transformed.columns:
            spend_level = df[col].sum()
            
            if 'tv' in col and spend_level > 500000:
                df_transformed[f'{col}_transformed'] = np.log1p(df[col] / 1000)
                transformation_log[col] = 'log (strong saturation)'
            elif spend_level > 200000:
                df_transformed[f'{col}_transformed'] = np.sqrt(df[col] / 100)
                transformation_log[col] = 'sqrt (moderate saturation)'
            else:
                df_transformed[f'{col}_transformed'] = df[col] / 1000
                transformation_log[col] = 'linear (minimal saturation)'
            
            df_transformed = df_transformed.drop(columns=[col])
    
    return df_transformed, transformation_log

# Apply exact same process
train_aggregated, test_aggregated = aggregate_channels(train_data, test_data)
new_media_cols = ['tv_total_spend', 'radio_total_spend', 'search_cost', 'social_costs', 'ooh_ooh_spend']

train_transformed, trans_log = apply_transformations(train_aggregated, new_media_cols)
test_transformed, _ = apply_transformations(test_aggregated, new_media_cols)

print("Transformations applied:")
for channel, transformation in trans_log.items():
    print(f"   • {channel}: {transformation}")

# Build model exactly as original
feature_cols = [col for col in train_transformed.columns if col not in ['date', 'sales']]
X_train = train_transformed[feature_cols].fillna(0)
y_train = train_transformed['sales']
X_test = test_transformed[feature_cols].fillna(0)
y_test = test_transformed['sales']

print(f"\nFeatures used ({len(feature_cols)}):")
for i, col in enumerate(feature_cols[:10]):  # Show first 10
    print(f"   {i+1}. {col}")
if len(feature_cols) > 10:
    print(f"   ... and {len(feature_cols) - 10} more")

# Standardize and train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = ridge.predict(X_train_scaled)
y_test_pred = ridge.predict(X_test_scaled)

# Performance metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

print(f"\n📊 EXACT RECREATION RESULTS:")
print(f"   • Training R²: {train_r2:.3f} ({train_r2*100:.1f}%)")
print(f"   • Test R²: {test_r2:.3f} ({test_r2*100:.1f}%)")
print(f"   • Test MAPE: {test_mape:.1f}%")
print(f"   • Overfitting gap: {train_r2-test_r2:.3f}")

# ================================
# 4. COMPARE WITH REPORTED RESULTS
# ================================

print(f"\n📊 COMPARISON WITH REPORTED RESULTS")
print("=" * 40)

reported_test_r2 = 0.653  # From the report
reported_test_mape = 7.4  # From the report

print(f"Reported vs Actual:")
print(f"   • Test R²: {reported_test_r2:.3f} vs {test_r2:.3f} (diff: {abs(reported_test_r2-test_r2):.3f})")
print(f"   • Test MAPE: {reported_test_mape:.1f}% vs {test_mape:.1f}% (diff: {abs(reported_test_mape-test_mape):.1f}%)")

if abs(reported_test_r2 - test_r2) > 0.05:
    print("⚠️  SIGNIFICANT DISCREPANCY IN R² DETECTED!")
if abs(reported_test_mape - test_mape) > 1.0:
    print("⚠️  SIGNIFICANT DISCREPANCY IN MAPE DETECTED!")

print(f"\n🎯 CONCLUSION:")
if abs(reported_test_r2 - test_r2) < 0.02 and abs(reported_test_mape - test_mape) < 0.5:
    print("✅ Model recreation is accurate - results match reported performance")
else:
    print("❌ Model recreation shows different results - need investigation")
    print("   Possible causes:")
    print("   • Different data splits")
    print("   • Different random seeds")
    print("   • Different preprocessing")
    print("   • Different model parameters") 