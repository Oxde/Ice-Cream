# %%
# 🔍 DEEP DIVE: DATA QUALITY & BUSINESS LOGIC CHECK
# =================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("🔍 DEEP DIVE ANALYSIS - INVESTIGATING TV ANOMALIES")
print("=" * 55)

# Load data
train_data = pd.read_csv('data/mmm_ready/consistent_channels_train_set.csv')
train_data['date'] = pd.to_datetime(train_data['date'])

# %%
# 1️⃣ CHECK DATA QUALITY AND PATTERNS
# ===================================

print("1️⃣ DATA QUALITY CHECK")
print("-" * 25)

# Check for zeros and missing values
media_cols = [col for col in train_data.columns if 'cost' in col or 'spend' in col]

for col in media_cols:
    zero_pct = (train_data[col] == 0).sum() / len(train_data) * 100
    mean_spend = train_data[col].mean()
    std_spend = train_data[col].std()
    cv = std_spend / mean_spend if mean_spend > 0 else 0
    
    print(f"\n{col}:")
    print(f"   • Zero weeks: {zero_pct:.1f}%")
    print(f"   • Mean spend: ${mean_spend:,.0f}")
    print(f"   • Std dev: ${std_spend:,.0f}")
    print(f"   • Coef. of variation: {cv:.2f}")

# %%
# 2️⃣ TV CHANNELS INVESTIGATION
# =============================

print("\n2️⃣ TV CHANNELS DEEP DIVE")
print("-" * 30)

# Plot TV spending over time
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# TV Branding
ax1.plot(train_data['date'], train_data['tv_branding_tv_branding_cost'], 'b-', linewidth=2)
ax1.set_ylabel('TV Branding ($)', fontweight='bold')
ax1.set_title('TV Spending Patterns Over Time', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# TV Promo
ax2.plot(train_data['date'], train_data['tv_promo_tv_promo_cost'], 'g-', linewidth=2)
ax2.set_ylabel('TV Promo ($)', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Sales
ax3.plot(train_data['date'], train_data['sales'], 'r-', linewidth=2)
ax3.set_ylabel('Sales', fontweight='bold')
ax3.set_xlabel('Date', fontweight='bold')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Check if TV channels alternate
tv_brand_active = train_data['tv_branding_tv_branding_cost'] > 0
tv_promo_active = train_data['tv_promo_tv_promo_cost'] > 0
both_active = tv_brand_active & tv_promo_active

print(f"\nTV SPENDING PATTERNS:")
print(f"   • TV Branding active: {tv_brand_active.sum()} weeks ({tv_brand_active.sum()/len(train_data)*100:.1f}%)")
print(f"   • TV Promo active: {tv_promo_active.sum()} weeks ({tv_promo_active.sum()/len(train_data)*100:.1f}%)")
print(f"   • Both active: {both_active.sum()} weeks ({both_active.sum()/len(train_data)*100:.1f}%)")

# %%
# 3️⃣ REVERSE CAUSALITY CHECK
# ===========================

print("\n3️⃣ REVERSE CAUSALITY ANALYSIS")
print("-" * 35)

# Check if high TV spend follows low sales (reactive spending)
for col in ['tv_branding_tv_branding_cost', 'tv_promo_tv_promo_cost']:
    # Calculate lagged correlations
    correlations = []
    lags = range(-4, 5)  # -4 to +4 weeks
    
    for lag in lags:
        if lag < 0:
            # TV leads sales
            corr = train_data[col].iloc[:lag].corr(train_data['sales'].iloc[-lag:])
        elif lag > 0:
            # Sales lead TV (reverse causality)
            corr = train_data[col].iloc[lag:].corr(train_data['sales'].iloc[:-lag])
        else:
            # Concurrent
            corr = train_data[col].corr(train_data['sales'])
        correlations.append(corr)
    
    # Find peak correlation
    peak_lag = lags[np.argmax(np.abs(correlations))]
    peak_corr = correlations[np.argmax(np.abs(correlations))]
    
    print(f"\n{col}:")
    print(f"   • Peak correlation: {peak_corr:.3f} at lag {peak_lag}")
    if peak_lag > 0:
        print(f"   ⚠️  REVERSE CAUSALITY: Sales changes PRECEDE TV spend!")
    elif peak_lag < 0:
        print(f"   ✅ Normal: TV spend precedes sales changes")

# %%
# 4️⃣ MULTICOLLINEARITY DEEP DIVE
# ================================

print("\n4️⃣ MULTICOLLINEARITY ANALYSIS")
print("-" * 35)

# Calculate VIF (Variance Inflation Factor)
from statsmodels.stats.outliers_influence import variance_inflation_factor

media_df = train_data[media_cols]
media_df_nonzero = media_df.loc[:, (media_df != 0).any(axis=0)]  # Remove always-zero columns

vif_data = pd.DataFrame()
vif_data["Channel"] = media_df_nonzero.columns
vif_data["VIF"] = [variance_inflation_factor(media_df_nonzero.values, i) 
                   for i in range(media_df_nonzero.shape[1])]

print("\nVARIANCE INFLATION FACTORS:")
print("(VIF > 10 indicates multicollinearity)")
for _, row in vif_data.iterrows():
    status = "⚠️ HIGH" if row['VIF'] > 10 else "✅ OK"
    print(f"   {row['Channel']:<40} VIF: {row['VIF']:>8.2f} {status}")

# %%
# 5️⃣ BUSINESS LOGIC VALIDATION
# =============================

print("\n5️⃣ BUSINESS LOGIC CHECKS")
print("-" * 30)

# Check spending consistency
for col in media_cols:
    data = train_data[col]
    if data.sum() > 0:
        # Check for sudden spikes
        rolling_mean = data.rolling(window=4).mean()
        spikes = data > (rolling_mean * 3)
        
        if spikes.any():
            print(f"\n⚠️  {col} has {spikes.sum()} unusual spikes")
            spike_dates = train_data.loc[spikes, 'date'].tolist()
            print(f"   Spike dates: {spike_dates[:3]}...")

# %%
# 6️⃣ HYPOTHESIS: TV CHANNELS CANNIBALIZATION
# ===========================================

print("\n6️⃣ TV CANNIBALIZATION ANALYSIS")
print("-" * 35)

# Create binary indicators
tv_brand_high = train_data['tv_branding_tv_branding_cost'] > train_data['tv_branding_tv_branding_cost'].median()
tv_promo_high = train_data['tv_promo_tv_promo_cost'] > train_data['tv_promo_tv_promo_cost'].median()

# Create 2x2 analysis
scenarios = pd.DataFrame({
    'TV_Brand_High': tv_brand_high,
    'TV_Promo_High': tv_promo_high,
    'Sales': train_data['sales']
})

print("\nAVERAGE SALES BY TV STRATEGY:")
print("(High = above median spend)")
print("-" * 40)
for brand in [False, True]:
    for promo in [False, True]:
        mask = (scenarios['TV_Brand_High'] == brand) & (scenarios['TV_Promo_High'] == promo)
        avg_sales = scenarios.loc[mask, 'Sales'].mean()
        count = mask.sum()
        
        brand_str = "High" if brand else "Low"
        promo_str = "High" if promo else "Low"
        print(f"TV Brand {brand_str} + TV Promo {promo_str}: {avg_sales:,.0f} sales ({count} weeks)")

# %%
# 7️⃣ FINAL DIAGNOSIS
# ==================

print("\n🔬 FINAL DIAGNOSIS")
print("=" * 50)

# Calculate basic stats for presentation
total_tv_spend = train_data['tv_branding_tv_branding_cost'].sum() + train_data['tv_promo_tv_promo_cost'].sum()
total_all_spend = train_data[media_cols].sum().sum()
tv_share = total_tv_spend / total_all_spend * 100

avg_sales_with_tv_brand = train_data.loc[train_data['tv_branding_tv_branding_cost'] > 0, 'sales'].mean()
avg_sales_without_tv_brand = train_data.loc[train_data['tv_branding_tv_branding_cost'] == 0, 'sales'].mean()

print(f"\n📊 KEY FINDINGS:")
print(f"   • TV represents {tv_share:.1f}% of total media spend")
print(f"   • Sales WITH TV Branding: {avg_sales_with_tv_brand:,.0f}")
print(f"   • Sales WITHOUT TV Branding: {avg_sales_without_tv_brand:,.0f}")
print(f"   • Difference: {(avg_sales_without_tv_brand - avg_sales_with_tv_brand) / avg_sales_with_tv_brand * 100:+.1f}%")

print(f"\n⚠️  CRITICAL ISSUES FOUND:")
print(f"   1. TV Branding shows NEGATIVE correlation with sales")
print(f"   2. Possible reverse causality (panic spending when sales drop)")
print(f"   3. TV channels may be cannibalizing each other")
print(f"   4. Spending patterns suggest reactive, not strategic approach")

print(f"\n✅ RECOMMENDATIONS:")
print(f"   1. Review TV buying strategy - current approach is destroying value")
print(f"   2. Test TV blackout periods to establish true baseline")
print(f"   3. Separate brand building from promotional activities")
print(f"   4. Implement proper testing framework before spending 60%+ on TV")

# %% 