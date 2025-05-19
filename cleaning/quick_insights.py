import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ===============================================================
# INITIAL SETUP
# ===============================================================
# Create a plots directory if it doesn't exist
# This is a good practice to organize outputs separately from code
plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# ===============================================================
# DATA LOADING & PREPARATION
# ===============================================================
# Load the MMM dataset - this contains all the marketing channels and sales data
# as well as derived features like lags and adstocks for modeling
df = pd.read_csv('processed/mmm_dataset.csv')
print(f"Loaded data from 2022-01-03 to 2024-12-23 with {len(df)} rows and {len(df.columns)} columns")

# Convert date to datetime - essential for time series analysis and plotting
# This allows pandas to understand the chronological order and enables time-based operations
df['date'] = pd.to_datetime(df['date'])

# ===============================================================
# VISUALIZATION SETUP
# ===============================================================
# Set plotting style - ggplot is clean and professional
# Increasing font scale improves readability in presentations
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# ===============================================================
# ANALYSIS 1: MARKETING SPEND BY CHANNEL
# ===============================================================
# This analysis helps understand where the budget is allocated
# and identifies the highest-investment channels
print("\n=== Marketing Spend by Channel ===")

# Identify cost/spend columns - using list comprehension with filter conditions
# We want only the original cost columns, not derived features like lags or adstock
# This technique extracts all marketing spend variables across channels
cost_cols = [col for col in df.columns if any(term in col for term in ['cost', 'spend']) and not any(term in col for term in ['lag', 'adstock'])]
total_spend = df[cost_cols].sum().sum()
print(f"Total marketing spend: ${total_spend:,.2f}")

# Calculate percentage by channel 
# This converts absolute spend to relative percentages for easier interpretation
# Sorting provides immediate visibility of the highest-spend channels
spend_by_channel = df[cost_cols].sum().sort_values(ascending=False)
pct_by_channel = (spend_by_channel / total_spend * 100).round(1)

print("\nMarketing spend by channel:")
for channel, spend in spend_by_channel.items():
    print(f"- {channel}: ${spend:,.2f} ({pct_by_channel[channel]}%)")

# Visualize marketing spend by channel using bar chart
# Bar charts are ideal for comparing discrete categories
plt.figure(figsize=(10, 6))  # Set figure size for better readability
spend_by_channel.plot(kind='bar', color='skyblue')
plt.title('Total Marketing Spend by Channel (2022-2024)')
plt.ylabel('Total Spend ($)')
plt.xticks(rotation=45, ha='right')  # Rotate labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.savefig(f"{plots_dir}/spend_by_channel.png", dpi=300)  # Save high-resolution image

# ===============================================================
# ANALYSIS 2: MARKETING SPEND OVER TIME
# ===============================================================
# Line plots work well for time series data to show trends and patterns
# By plotting all channels together, we can see relative spending patterns
fig, ax = plt.subplots(figsize=(12, 6))
for col in cost_cols:
    ax.plot(df['date'], df[col], label=col)
ax.set_title('Marketing Spend Over Time by Channel')
ax.set_xlabel('Date')
ax.set_ylabel('Spend ($)')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f"{plots_dir}/spend_over_time.png", dpi=300)

# ===============================================================
# ANALYSIS 3: SALES TREND
# ===============================================================
# Simple line chart to visualize the target variable (sales) over time
# This helps identify trends, seasonality, and potential anomalies
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['sales'], color='forestgreen', linewidth=2)
plt.title('Sales Trend (2022-2024)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{plots_dir}/sales_trend.png", dpi=300)

# ===============================================================
# ANALYSIS 4: CORRELATION ANALYSIS
# ===============================================================
# Correlation analysis measures the linear relationship between variables
# Pearson correlation coefficients range from -1 (perfect negative) to +1 (perfect positive)
# This helps identify which marketing channels have the strongest relationship with sales
correlation_cols = ['sales'] + cost_cols
correlation = df[correlation_cols].corr()  # Calculate correlation matrix
sales_correlation = correlation['sales'].sort_values(ascending=False)  # Sort by correlation with sales

print("\n=== Correlation with Sales ===")
for channel, corr in sales_correlation.items():
    if channel != 'sales':
        print(f"- {channel}: {corr:.3f}")

# Plot correlation heatmap
# Heatmaps provide an intuitive visual representation of correlation strength
# The color intensity indicates the correlation strength and direction
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Between Sales and Marketing Channels')
plt.tight_layout()
plt.savefig(f"{plots_dir}/correlation_heatmap.png", dpi=300)

# ===============================================================
# ANALYSIS 5: SEASONALITY IN SALES
# ===============================================================
# Analyzing sales by month reveals seasonal patterns
# This is crucial for understanding cyclical behavior in the business
plt.figure(figsize=(10, 6))
df.groupby('month')['sales'].mean().plot(kind='bar', color='coral')
plt.title('Average Sales by Month')
plt.xlabel('Month')
plt.ylabel('Average Sales')
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.tight_layout()
plt.savefig(f"{plots_dir}/sales_by_month.png", dpi=300)

# ===============================================================
# ANALYSIS 6: MONTHLY MARKETING SPEND
# ===============================================================
# Analyzing marketing spend by month to detect seasonal patterns in investment
# Using stacked bars to show both total spend and channel-specific allocation
monthly_spend = df.groupby('month')[cost_cols].mean()
plt.figure(figsize=(12, 8))
monthly_spend.plot(kind='bar', stacked=True)  # Stacked bars show both total and composition
plt.title('Average Monthly Marketing Spend by Channel')
plt.xlabel('Month')
plt.ylabel('Average Spend ($)')
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f"{plots_dir}/monthly_spend.png", dpi=300)

# ===============================================================
# ANALYSIS 7: OUTLIER DETECTION
# ===============================================================
# Z-score is a statistical measure that quantifies how many standard deviations
# a data point is from the mean
# Formula: Z = (X - μ) / σ where:
#   X = individual data point
#   μ = mean of the dataset
#   σ = standard deviation of the dataset
print("\n=== Outlier Analysis ===")
# Calculate Z-scores for sales
df['sales_zscore'] = (df['sales'] - df['sales'].mean()) / df['sales'].std()

# Identify outliers (typically using |Z| > 2 or |Z| > 3 threshold)
# Z > 2 captures roughly the top and bottom 2.5% of a normal distribution
outliers = df[abs(df['sales_zscore']) > 2][['date', 'sales', 'sales_zscore']]
print(f"Potential sales outliers (Z-score > 2):")
if len(outliers) > 0:
    print(outliers.sort_values('sales_zscore', ascending=False).to_string())
else:
    print("No major outliers found in sales data.")

# Visualize sales outliers
# This scatter plot highlights potential anomalies in the sales pattern
plt.figure(figsize=(12, 6))
plt.scatter(df['date'], df['sales'], color='blue', alpha=0.6)
if len(outliers) > 0:
    plt.scatter(outliers['date'], outliers['sales'], color='red', s=100, label='Outliers')
    plt.legend()
plt.title('Sales with Outliers Highlighted')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{plots_dir}/sales_outliers.png", dpi=300)

# ===============================================================
# ANALYSIS 8: SALES DISTRIBUTION
# ===============================================================
# Histogram with KDE (Kernel Density Estimation) shows the distribution of sales
# This helps understand the central tendency and spread of the sales data
plt.figure(figsize=(10, 6))
sns.histplot(df['sales'], kde=True, color='skyblue')  # KDE adds a smooth curve estimation of the distribution
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(f"{plots_dir}/sales_distribution.png", dpi=300)

# ===============================================================
# ANALYSIS 9: MARKETING EFFICIENCY
# ===============================================================
# Calculate ROI-like metric: sales per marketing dollar spent
# This measures efficiency of marketing investment over time
# Calculate total marketing spend per week by summing across all channels
df['total_marketing_spend'] = df[cost_cols].sum(axis=1)
df['sales_per_dollar'] = df['sales'] / df['total_marketing_spend']

# Visualize marketing efficiency over time
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['sales_per_dollar'], color='purple', linewidth=2)
plt.title('Marketing Efficiency (Sales per $ Spent)')
plt.xlabel('Date')
plt.ylabel('Sales per Marketing Dollar')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{plots_dir}/marketing_efficiency.png", dpi=300)

# Print key efficiency statistics
print("\n=== Marketing Efficiency ===")
print(f"Average sales per marketing dollar: ${df['sales_per_dollar'].mean():.2f}")
print(f"Minimum efficiency: ${df['sales_per_dollar'].min():.2f} on {df.loc[df['sales_per_dollar'].idxmin(), 'date']}")
print(f"Maximum efficiency: ${df['sales_per_dollar'].max():.2f} on {df.loc[df['sales_per_dollar'].idxmax(), 'date']}")

# ===============================================================
# ANALYSIS 10: KEY STATISTICS SUMMARY
# ===============================================================
# Simple summary statistics for the main KPIs
print("\n=== Key Statistics ===")
print(f"Average weekly sales: ${df['sales'].mean():,.2f}")
print(f"Average weekly marketing spend: ${df['total_marketing_spend'].mean():,.2f}")

# Calculate sales trend using correlation with time
# This method uses correlation with a sequence (0,1,2,...) to measure if sales
# have been trending up (positive correlation) or down (negative correlation) over time
print(f"Sales trend (correlation with time): {df['sales'].corr(pd.Series(range(len(df)))):.3f}")

print("\nAll plots saved to the 'plots' directory.") 