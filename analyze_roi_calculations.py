#!/usr/bin/env python3
"""
Analyze ROI calculations to understand extreme values and create better visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# Load the data
train_data = pd.read_csv('data/mmm_ready/consistent_channels_train_set.csv')
train_data['date'] = pd.to_datetime(train_data['date'])

# Get media columns
media_cols = [col for col in train_data.columns if 'cost' in col or 'spend' in col]

print("=" * 60)
print("ROI CALCULATION ANALYSIS")
print("=" * 60)

# 1. First, let's check the scale of the data
print("\n1. DATA SCALE ANALYSIS:")
print("-" * 40)
print(f"Average weekly sales: ${train_data['sales'].mean():,.0f}")
print(f"Total sales over period: ${train_data['sales'].sum():,.0f}")
print(f"\nMedia spend analysis:")

for col in media_cols:
    total_spend = train_data[col].sum()
    avg_weekly = train_data[col].mean()
    sales_ratio = train_data['sales'].mean() / avg_weekly if avg_weekly > 0 else 0
    print(f"{col.replace('_', ' ').title():<35} Total: ${total_spend:>10,.0f}  Weekly: ${avg_weekly:>8,.0f}  Sales/Spend: {sales_ratio:>6.1f}x")

# 2. Check correlations between channels and sales
print("\n\n2. SIMPLE CORRELATIONS WITH SALES:")
print("-" * 40)
correlations = {}
for col in media_cols:
    if train_data[col].sum() > 0:
        corr = train_data[col].corr(train_data['sales'])
        correlations[col] = corr
        print(f"{col.replace('_', ' ').title():<35} {corr:>+.3f}")

# 3. Let's manually calculate what the ROI should be with a simpler approach
print("\n\n3. ALTERNATIVE ROI CALCULATION:")
print("-" * 40)
print("Using simple correlation-based attribution (for comparison):")

total_media_contribution = 0
for col in media_cols:
    if train_data[col].sum() > 0:
        # Simple approach: assume correlation indicates contribution
        corr = max(0, correlations[col])  # Only positive correlations
        avg_sales = train_data['sales'].mean()
        estimated_contribution = corr * avg_sales * 0.3  # Assume media drives 30% of sales max
        weekly_spend = train_data[col].mean()
        
        if weekly_spend > 0:
            simple_roi = ((estimated_contribution - weekly_spend) / weekly_spend) * 100
        else:
            simple_roi = 0
            
        print(f"{col.replace('_', ' ').title():<35} Simple ROI: {simple_roi:>+7.0f}%")

# 4. Analyze the counterfactual approach issues
print("\n\n4. COUNTERFACTUAL APPROACH ISSUES:")
print("-" * 40)
print("Potential problems with extreme ROI values:")
print("1. Model may be overfitting to spurious correlations")
print("2. Removing a channel might unrealistically shift attribution to others")
print("3. Negative correlations might be coincidental (e.g., Radio National high when sales low)")
print("4. Model doesn't account for baseline sales without ANY media")

# 5. Calculate more realistic ROI bounds
print("\n\n5. REALISTIC ROI BOUNDS:")
print("-" * 40)
print("Industry benchmarks for media ROI:")
print("• Digital channels: 100-400% (exceptional: up to 500%)")
print("• TV: 50-150% (mature markets)")
print("• Radio: 75-200% (local can outperform national)")
print("• OOH: 50-150%")
print("\nCurrent model shows:")
print("• 4 channels at 200% (capped)")
print("• 3 channels with massive negative ROI")
print("• This suggests model instability")

# Create a visualization comparing different ROI approaches
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('ROI Analysis: Understanding Extreme Values', fontsize=16, fontweight='bold')

# 1. Spend vs Sales Correlation
ax1 = axes[0, 0]
for col in media_cols:
    if train_data[col].sum() > 0:
        x = train_data[col]
        y = train_data['sales']
        ax1.scatter(x, y, alpha=0.5, label=col.split('_')[0].title())
ax1.set_xlabel('Weekly Spend ($)')
ax1.set_ylabel('Weekly Sales ($)')
ax1.set_title('Raw Spend vs Sales Relationships')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

# 2. Channel Spend Distribution
ax2 = axes[0, 1]
spend_data = []
for col in media_cols:
    spend_data.extend(train_data[col].values)
    
channel_names = []
for col in media_cols:
    channel_names.extend([col.split('_')[0].title()] * len(train_data))
    
spend_df = pd.DataFrame({'Spend': spend_data, 'Channel': channel_names})
spend_df = spend_df[spend_df['Spend'] > 0]

ax2.violinplot([spend_df[spend_df['Channel'] == ch]['Spend'].values 
                for ch in spend_df['Channel'].unique()],
               positions=range(len(spend_df['Channel'].unique())))
ax2.set_xticks(range(len(spend_df['Channel'].unique())))
ax2.set_xticklabels(spend_df['Channel'].unique(), rotation=45)
ax2.set_ylabel('Weekly Spend ($)')
ax2.set_title('Spend Distribution by Channel')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Correlation Heatmap
ax3 = axes[1, 0]
corr_matrix = train_data[media_cols + ['sales']].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
            cmap='coolwarm', center=0, ax=ax3,
            xticklabels=[col.split('_')[0].title() for col in media_cols + ['sales']],
            yticklabels=[col.split('_')[0].title() for col in media_cols + ['sales']])
ax3.set_title('Correlation Matrix: Media Channels & Sales')

# 4. ROI Reasonability Check
ax4 = axes[1, 1]
channels = ['Search', 'Social', 'OOH', 'Radio Local', 'TV Brand', 'TV Promo', 'Radio Nat']
model_roi = [200, 200, 200, 200, -179, -197, -744]
industry_max = [400, 350, 150, 200, 150, 150, 200]
industry_typical = [200, 150, 100, 150, 100, 100, 150]

x = np.arange(len(channels))
width = 0.25

bars1 = ax4.bar(x - width, model_roi, width, label='Model ROI', color='red', alpha=0.7)
bars2 = ax4.bar(x, industry_typical, width, label='Industry Typical', color='green', alpha=0.7)
bars3 = ax4.bar(x + width, industry_max, width, label='Industry Max', color='blue', alpha=0.7)

ax4.set_xlabel('Channel')
ax4.set_ylabel('ROI (%)')
ax4.set_title('Model ROI vs Industry Benchmarks')
ax4.set_xticks(x)
ax4.set_xticklabels(channels, rotation=45)
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax4.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height < -100:
            label = f'{height:.0f}'
            y_pos = height - 50
        else:
            label = f'{height:.0f}'
            y_pos = height + 10
        ax4.text(bar.get_x() + bar.get_width()/2., y_pos,
                label, ha='center', va='bottom' if height > 0 else 'top', fontsize=8)

plt.tight_layout()
plt.savefig('plots/roi_analysis_investigation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n\n6. RECOMMENDATIONS:")
print("-" * 40)
print("1. The extreme negative ROIs (-744% for Radio National) are likely due to:")
print("   - Model overfitting to spurious correlations")
print("   - Multicollinearity between channels")
print("   - Lack of baseline (non-media driven) sales consideration")
print("\n2. The 200% cap masks even more extreme positive values")
print("\n3. A more robust approach would:")
print("   - Use incrementality testing or geo experiments")
print("   - Apply Bayesian priors based on industry knowledge")
print("   - Include interaction effects between channels")
print("   - Model saturation curves more carefully")

# Calculate what percentage of sales the model attributes to media
print("\n\n7. SANITY CHECK - TOTAL MEDIA ATTRIBUTION:")
print("-" * 40)
avg_sales = train_data['sales'].mean()
total_media_spend = train_data[media_cols].sum(axis=1).mean()
print(f"Average weekly sales: ${avg_sales:,.0f}")
print(f"Average weekly media spend: ${total_media_spend:,.0f}")
print(f"Spend as % of sales: {(total_media_spend/avg_sales)*100:.1f}%")

# If model shows -744% ROI for Radio National, it means removing it INCREASES sales
# This is clearly unrealistic for a major brand
print("\nModel implies that Radio National REDUCES sales by 7.4x its spend!")
print("This is not realistic for an established brand's media mix.") 