#!/usr/bin/env python3
"""
Clean ROI Visualization for Business Report
==========================================
Creates a professional, concise visualization with just the key insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set professional styling
plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11

def create_clean_roi_visualization():
    """Create a clean, professional ROI visualization"""
    
    # Data
    channels = ['Social Media', 'Search Marketing', 'Radio Local', 'TV Branding', 
                'TV Promo', 'Out-of-Home', 'Radio National']
    
    marginal_roi = [19, 17, 15, -3, -5, -8, -21]
    efficiency_scores = [100, 93, 67, 37, 33, 19, 0]
    budget_share = [4.3, 4.4, 13.2, 39.1, 22.2, 5.6, 10.5]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Chart 1: Marginal ROI by Channel
    colors = ['#27ae60' if roi > 10 else '#f39c12' if roi > 0 else '#e74c3c' for roi in marginal_roi]
    
    bars1 = ax1.barh(range(len(channels)), marginal_roi, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    ax1.set_yticks(range(len(channels)))
    ax1.set_yticklabels(channels)
    ax1.set_xlabel('Marginal ROI (%)', fontweight='bold')
    ax1.set_title('Marginal ROI by Channel', fontweight='bold', fontsize=14, pad=20)
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_xlim(-25, 25)
    
    # Add value labels
    for i, (bar, roi) in enumerate(zip(bars1, marginal_roi)):
        label_x = roi + (1.5 if roi > 0 else -1.5)
        ha = 'left' if roi > 0 else 'right'
        ax1.text(label_x, i, f'{roi:+.0f}%', va='center', ha=ha, fontweight='bold', fontsize=10)
    
    # Chart 2: Efficiency vs Budget Allocation Bubble Chart
    # Bubble size represents budget share
    bubble_sizes = [size * 15 for size in budget_share]  # Scale for visibility
    colors2 = ['#27ae60' if eff > 70 else '#f39c12' if eff > 40 else '#e74c3c' for eff in efficiency_scores]
    
    scatter = ax2.scatter(budget_share, efficiency_scores, s=bubble_sizes, c=colors2, 
                         alpha=0.7, edgecolors='white', linewidth=2)
    
    # Add channel labels
    for i, (budget, eff, channel) in enumerate(zip(budget_share, efficiency_scores, channels)):
        # Adjust label position to avoid overlap
        offset_x = 1.5 if budget < 20 else -1.5
        offset_y = 3 if i % 2 == 0 else -3
        ha = 'left' if budget < 20 else 'right'
        
        # Create better abbreviated labels that distinguish between similar channels
        label_mapping = {
            'Social Media': 'Social Media',
            'Search Marketing': 'Search Marketing',
            'Radio Local': 'Radio Local',
            'TV Branding': 'TV Brand',
            'TV Promo': 'TV Promo',
            'Out-of-Home': 'Out-of-Home',
            'Radio National': 'Radio National'
        }
        
        ax2.annotate(label_mapping[channel], (budget, eff), 
                    xytext=(offset_x, offset_y), textcoords='offset points',
                    fontsize=9, ha=ha, va='center', fontweight='bold')
    
    ax2.set_xlabel('Budget Share (%)', fontweight='bold')
    ax2.set_ylabel('Efficiency Score (0-100)', fontweight='bold')
    ax2.set_title('Efficiency vs Budget Allocation\n(Bubble size = Budget Share)', fontweight='bold', fontsize=14, pad=20)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 45)
    ax2.set_ylim(-5, 105)
    
    # Add performance zones
    ax2.axhspan(70, 100, alpha=0.1, color='green', label='High Performance')
    ax2.axhspan(40, 70, alpha=0.1, color='orange', label='Medium Performance')
    ax2.axhspan(0, 40, alpha=0.1, color='red', label='Low Performance')
    
    # Add ideal allocation line (diagonal)
    ax2.plot([0, 100], [0, 100], '--', color='gray', alpha=0.5, linewidth=1, label='Ideal Allocation')
    
    plt.tight_layout()
    plt.savefig('roi_analysis_clean.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig

def generate_summary_stats():
    """Generate key summary statistics"""
    
    print("ROI ANALYSIS - KEY INSIGHTS")
    print("=" * 40)
    
    # Key metrics
    total_budget = 13946
    digital_budget = 614 + 600  # Search + Social
    digital_share = (digital_budget / total_budget) * 100
    tv_budget = 5453 + 3097  # Branding + Promo
    tv_share = (tv_budget / total_budget) * 100
    
    print(f"📊 BUDGET ALLOCATION:")
    print(f"   • Digital channels: {digital_share:.1f}% of budget")
    print(f"   • TV channels: {tv_share:.1f}% of budget")
    print(f"   • Traditional media: {100 - digital_share - tv_share:.1f}% of budget")
    
    print(f"\n🎯 PERFORMANCE LEADERS:")
    print(f"   • Top performing: Social Media (19% marginal ROI)")
    print(f"   • Most efficient: Social Media (100/100 efficiency)")
    print(f"   • Underperformer: Radio National (-21% marginal ROI)")
    
    print(f"\n💡 KEY OPPORTUNITY:")
    print(f"   • Digital underinvestment: High performance, low allocation")
    print(f"   • TV oversaturation: Low performance, high allocation")

if __name__ == "__main__":
    create_clean_roi_visualization()
    generate_summary_stats() 