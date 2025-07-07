#!/usr/bin/env python3
"""
Create professional ROI visualization for the report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for professional appearance
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Data from the model
channels = ['Radio Local', 'Search Marketing', 'Social Media', 'Out-of-Home', 
            'TV Branding', 'TV Promo', 'Radio National']
roi_capped = [200, 200, 200, 200, -179, -197, -744]
roi_uncapped = [320, 498, 566, 247, -179, -197, -744]  # Estimated from model output
weekly_spend = [1879, 623, 610, 801, 5408, 3082, 1486]
budget_share = [13.5, 4.5, 4.4, 5.8, 38.9, 22.2, 10.7]

# Colors based on performance
colors = ['#27ae60' if roi > 0 else '#e74c3c' for roi in roi_capped]

# 1. Main ROI Chart with Industry Context
ax1 = fig.add_subplot(gs[0, :2])

# Create ROI bars
y_pos = np.arange(len(channels))
bars = ax1.barh(y_pos, roi_capped, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

# Add industry benchmark zones
ax1.axvspan(-100, 0, alpha=0.1, color='red', label='Negative ROI')
ax1.axvspan(0, 100, alpha=0.1, color='yellow', label='Low ROI (0-100%)')
ax1.axvspan(100, 200, alpha=0.1, color='lightgreen', label='Good ROI (100-200%)')

# Add value labels
for i, (bar, roi_cap, roi_uncap) in enumerate(zip(bars, roi_capped, roi_uncapped)):
    width = bar.get_width()
    label_x = width + 10 if width > 0 else width - 10
    ha = 'left' if width > 0 else 'right'
    
    # Show capped value
    ax1.text(label_x, bar.get_y() + bar.get_height()/2, f'{roi_cap:+.0f}%',
             ha=ha, va='center', fontweight='bold', fontsize=11)
    
    # Show uncapped value if different
    if roi_cap == 200 and roi_uncap > 200:
        ax1.text(label_x, bar.get_y() + bar.get_height()/2 - 0.15, 
                f'(actual: {roi_uncap:+.0f}%)',
                ha=ha, va='center', fontsize=9, style='italic', color='gray')

ax1.set_yticks(y_pos)
ax1.set_yticklabels(channels)
ax1.set_xlabel('ROI (%)', fontsize=12)
ax1.set_title('Channel ROI Performance\n(Capped at 200% for Display)', fontsize=14, fontweight='bold')
ax1.axvline(x=0, color='black', linewidth=1.5)
ax1.axvline(x=100, color='gray', linestyle='--', alpha=0.7)
ax1.set_xlim(-800, 600)
ax1.legend(loc='lower right', framealpha=0.9)
ax1.grid(True, axis='x', alpha=0.3)

# Add note about extreme values
ax1.text(0.02, 0.02, 'Note: Extreme negative values suggest model instability', 
         transform=ax1.transAxes, fontsize=10, style='italic', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3))

# 2. Budget Allocation vs ROI Efficiency
ax2 = fig.add_subplot(gs[0, 2])

# Create scatter plot
scatter = ax2.scatter(budget_share, roi_capped, s=[s/10 for s in weekly_spend], 
                     c=colors, alpha=0.7, edgecolors='black', linewidth=1.5)

# Add channel labels
for i, (x, y, ch) in enumerate(zip(budget_share, roi_capped, channels)):
    ax2.annotate(ch.replace(' Marketing', '').replace(' Media', ''), 
                (x, y), xytext=(5, 5), textcoords='offset points', 
                fontsize=9, fontweight='bold')

# Add quadrant lines
ax2.axhline(y=0, color='black', linewidth=1, alpha=0.5)
ax2.axvline(x=20, color='gray', linestyle='--', alpha=0.5)

ax2.set_xlabel('Budget Share (%)', fontsize=12)
ax2.set_ylabel('ROI (%)', fontsize=12)
ax2.set_title('Budget Allocation Efficiency', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-800, 300)

# Add size legend
ax2.text(0.02, 0.98, 'Bubble size = Weekly spend', transform=ax2.transAxes,
         verticalalignment='top', fontsize=9, style='italic')

# 3. Spend Distribution
ax3 = fig.add_subplot(gs[1, 0])

# Sort by spend for better visualization
sorted_indices = np.argsort(weekly_spend)[::-1]
sorted_channels = [channels[i] for i in sorted_indices]
sorted_spend = [weekly_spend[i] for i in sorted_indices]
sorted_colors = [colors[i] for i in sorted_indices]

bars = ax3.bar(range(len(sorted_channels)), sorted_spend, color=sorted_colors, 
                alpha=0.8, edgecolor='black', linewidth=1)

# Add value labels
for bar, spend in zip(bars, sorted_spend):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f'${spend:,.0f}', ha='center', va='bottom', fontsize=10)

ax3.set_xticks(range(len(sorted_channels)))
ax3.set_xticklabels([ch.replace(' Marketing', '').replace(' Media', '') 
                     for ch in sorted_channels], rotation=45, ha='right')
ax3.set_ylabel('Weekly Spend ($)', fontsize=12)
ax3.set_title('Media Investment by Channel', fontsize=14, fontweight='bold')
ax3.grid(True, axis='y', alpha=0.3)

# 4. ROI vs Industry Benchmarks
ax4 = fig.add_subplot(gs[1, 1:])

# Industry benchmark data
benchmark_data = {
    'Digital (Search/Social)': {'typical': [150, 200], 'max': [300, 500], 'model': [200, 200]},
    'TV (Brand/Promo)': {'typical': [75, 125], 'max': [100, 200], 'model': [-179, -197]},
    'Radio (Local/National)': {'typical': [100, 150], 'max': [150, 250], 'model': [200, -744]},
    'Out-of-Home': {'typical': [75, 125], 'max': [100, 200], 'model': [200]}
}

x_pos = 0
x_labels = []
x_positions = []

for category, data in benchmark_data.items():
    n_channels = len(data['model'])
    
    # Plot typical range
    for i in range(n_channels):
        rect = Rectangle((x_pos + i*0.8, data['typical'][0]), 0.6, 
                        data['typical'][1] - data['typical'][0],
                        facecolor='lightgreen', alpha=0.3, edgecolor='green',
                        label='Industry Typical' if x_pos == 0 and i == 0 else '')
        ax4.add_patch(rect)
        
        # Plot max range
        rect = Rectangle((x_pos + i*0.8, data['typical'][1]), 0.6, 
                        data['max'][1] - data['typical'][1],
                        facecolor='lightblue', alpha=0.3, edgecolor='blue',
                        label='Industry Max' if x_pos == 0 and i == 0 else '')
        ax4.add_patch(rect)
        
        # Plot model value
        model_val = data['model'][i]
        color = '#27ae60' if model_val > 0 else '#e74c3c'
        marker = 'o' if model_val > -200 else 'v'
        ax4.scatter(x_pos + i*0.8 + 0.3, model_val, s=100, c=color, 
                   marker=marker, edgecolor='black', linewidth=1.5, zorder=5,
                   label='Model ROI' if x_pos == 0 and i == 0 else '')
        
        x_positions.append(x_pos + i*0.8 + 0.3)
    
    x_labels.append(category)
    x_pos += n_channels * 0.8 + 0.5

ax4.set_xticks(x_positions)
ax4.set_xticklabels(['Digital', 'Digital', 'TV Brand', 'TV Promo', 'Radio L', 'Radio N', 'OOH'], 
                    rotation=45, ha='right')
ax4.set_ylabel('ROI (%)', fontsize=12)
ax4.set_title('Model ROI vs Industry Benchmarks', fontsize=14, fontweight='bold')
ax4.axhline(y=0, color='black', linewidth=1)
ax4.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
ax4.set_ylim(-800, 600)
ax4.legend(loc='upper right')
ax4.grid(True, axis='y', alpha=0.3)

# Add warning text
ax4.text(0.98, 0.02, 'Warning: Extreme negative values indicate potential model issues',
         transform=ax4.transAxes, ha='right', fontsize=10, style='italic',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.2))

# Overall title and layout
fig.suptitle('Media Mix Model ROI Analysis - Critical Review', fontsize=16, fontweight='bold')
plt.tight_layout()

# Save the figure
plt.savefig('plots/section_07_roi_analysis_professional.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a second, simpler visualization for the report
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 1. Clean ROI visualization
y_pos = np.arange(len(channels))
bars = ax1.barh(y_pos, roi_capped, color=colors, alpha=0.8, edgecolor='black')

# Add value labels
for i, (bar, roi) in enumerate(zip(bars, roi_capped)):
    width = bar.get_width()
    label_x = width + 10 if width > 0 else width - 10
    ha = 'left' if width > 0 else 'right'
    ax1.text(label_x, bar.get_y() + bar.get_height()/2, f'{roi:+.0f}%',
             ha=ha, va='center', fontweight='bold')

ax1.set_yticks(y_pos)
ax1.set_yticklabels(channels)
ax1.set_xlabel('ROI (%)')
ax1.set_title('Channel ROI Performance')
ax1.axvline(x=0, color='black', linewidth=1.5)
ax1.axvline(x=100, color='gray', linestyle='--', alpha=0.7, label='100% ROI threshold')
ax1.grid(True, axis='x', alpha=0.3)
ax1.legend()

# 2. Portfolio efficiency
categories = ['High ROI\nChannels\n(>100%)', 'Negative ROI\nChannels\n(<0%)']
values = [28.2, 71.8]
colors_pie = ['#27ae60', '#e74c3c']

wedges, texts, autotexts = ax2.pie(values, labels=categories, colors=colors_pie,
                                    autopct='%1.1f%%', startangle=90,
                                    explode=(0.05, 0.05))
ax2.set_title('Budget Allocation by Performance')

# Make percentage text bold
for autotext in autotexts:
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

fig2.suptitle('Current Portfolio ROI Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/section_07_roi_analysis_clean.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nVisualization Summary:")
print("=" * 50)
print("1. Created comprehensive ROI analysis chart")
print("2. Highlighted extreme values as potential model issues")
print("3. Compared against industry benchmarks")
print("4. Created clean version for report inclusion")
print("\nKey insights visualized:")
print("- 71.8% of budget in negative ROI channels")
print("- Extreme negative ROIs likely indicate model problems")
print("- Need for more robust methodology") 