# %% [markdown]
# # Data Preprocessing for Modeling
# 
# **Goal**: Prepare cleaned datasets for machine learning modeling
# 
# **Process:**
# 1. Load cleaned datasets
# 2. Quality verification  
# 3. Minimal outlier treatment
# 4. Time feature engineering
# 5. Save preprocessed datasets

# %%
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("🔧 Data Preprocessing for Modeling")
print("=" * 40)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)

# %%
# Load all cleaned datasets
interim_dir = '../data/interim'
cleaned_files = glob.glob(os.path.join(interim_dir, '*_basic_clean.csv'))

datasets = {}
for file_path in cleaned_files:
    filename = os.path.basename(file_path).replace('_basic_clean.csv', '')
    df = pd.read_csv(file_path)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    datasets[filename] = df
    print(f"  📁 {filename}: {df.shape}")

# %%
# Quick quality check
def quick_quality_check(df, name):
    print(f"\n✅ {name.upper()}:")
    print(f"  Shape: {df.shape}")
    
    if 'date' in df.columns:
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    missing = df.isnull().sum().sum()
    print(f"  Missing values: {missing}")
    
    # Check numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'date' in numeric_cols:
        numeric_cols.remove('date')
    
    for col in numeric_cols:
        if len(df[col].dropna()) > 0:
            data = df[col].dropna()
            Q1, Q3 = data.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
            outlier_pct = len(outliers) / len(data) * 100
            print(f"  {col}: outliers={outlier_pct:.1f}%")

for name, df in datasets.items():
    quick_quality_check(df, name)

# %%
# Minimal outlier treatment
def handle_outliers_minimal(df, name):
    print(f"\n🎯 Outlier treatment: {name}")
    
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if 'date' in numeric_cols:
        numeric_cols.remove('date')
    
    outlier_info = {}
    
    for col in numeric_cols:
        data = df_clean[col].dropna()
        if len(data) == 0:
            continue
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
        outliers_count = outliers_mask.sum()
        
        if outliers_count > 0:
            outlier_pct = outliers_count / len(data) * 100
            
            # Conservative treatment: only cap if <2% outliers
            if outlier_pct < 2:
                df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
                print(f"  {col}: {outliers_count} outliers capped ({outlier_pct:.1f}%)")
            else:
                print(f"  {col}: {outliers_count} outliers kept ({outlier_pct:.1f}%)")
    
    return df_clean, outlier_info

# %%
# Time feature engineering
def engineer_time_features(df, name):
    print(f"\n⚙️ Time features: {name}")
    
    if 'date' not in df.columns:
        print("  No date column - skipping")
        return df
    
    df_features = df.copy()
    
    # Basic time features
    df_features['year'] = df_features['date'].dt.year
    df_features['month'] = df_features['date'].dt.month
    df_features['week'] = df_features['date'].dt.isocalendar().week
    df_features['quarter'] = df_features['date'].dt.quarter
    
    # Cyclical features for seasonality
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    df_features['week_sin'] = np.sin(2 * np.pi * df_features['week'] / 52)
    df_features['week_cos'] = np.cos(2 * np.pi * df_features['week'] / 52)
    
    # Business features
    def get_season(month):
        if month in [12, 1, 2]: return 'winter'
        elif month in [3, 4, 5]: return 'spring'
        elif month in [6, 7, 8]: return 'summer'
        else: return 'autumn'
    
    df_features['season'] = df_features['month'].apply(get_season)
    
    # Holiday periods
    def is_holiday_period(date):
        month, day = date.month, date.day
        if (month == 1 and day <= 7) or (month == 12 and day >= 25): return 1
        elif month in [7, 8]: return 1
        elif month in [3, 4] and 15 <= day <= 25: return 1
        else: return 0
    
    df_features['holiday_period'] = df_features['date'].apply(is_holiday_period)
    df_features['is_month_end'] = (df_features['date'].dt.day >= 25).astype(int)
    
    new_features = df_features.shape[1] - df.shape[1]
    print(f"  Added {new_features} time features")
    
    return df_features

# %%
# Process all datasets
print(f"\n🔧 Processing all datasets...")

preprocessed_datasets = {}

for name, df in datasets.items():
    print(f"\n{'='*10} {name.upper()} {'='*10}")
    
    # Outlier treatment
    df_step1, _ = handle_outliers_minimal(df, name)
    
    # Time features
    df_final = engineer_time_features(df_step1, name)
    
    preprocessed_datasets[name] = df_final
    print(f"  Final shape: {df.shape} → {df_final.shape}")

# %%
# Distribution comparison for key datasets
def plot_key_distributions(original_datasets, preprocessed_datasets):
    # Only plot distributions for datasets with meaningful numeric variables
    key_datasets = ['email', 'facebook', 'google', 'ooh']  # Most important media channels
    
    for name in key_datasets:
        if name not in original_datasets:
            continue
            
        df_orig = original_datasets[name]
        df_proc = preprocessed_datasets[name]
        
        # Get main numeric column (usually the spend/impression/click column)
        numeric_cols = df_orig.select_dtypes(include=[np.number]).columns.tolist()
        if 'date' in numeric_cols:
            numeric_cols.remove('date')
        
        if not numeric_cols:
            continue
            
        # Plot the main variable
        main_col = numeric_cols[0]  # Usually the most important metric
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Original
        data_orig = df_orig[main_col].dropna()
        axes[0].hist(data_orig, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title(f'{name.title()} - {main_col} (Original)')
        skew_orig = stats.skew(data_orig)
        axes[0].text(0.02, 0.98, f'Skew: {skew_orig:.2f}', 
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Processed
        data_proc = df_proc[main_col].dropna()
        axes[1].hist(data_proc, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1].set_title(f'{name.title()} - {main_col} (Processed)')
        skew_proc = stats.skew(data_proc)
        axes[1].text(0.02, 0.98, f'Skew: {skew_proc:.2f}', 
                    transform=axes[1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

plot_key_distributions(datasets, preprocessed_datasets)

# %%
# Save preprocessed datasets
processed_dir = '../data/processed'
os.makedirs(processed_dir, exist_ok=True)

print(f"\n💾 Saving preprocessed datasets...")

for name, df in preprocessed_datasets.items():
    output_path = os.path.join(processed_dir, f"{name}_preprocessed.csv")
    df.to_csv(output_path, index=False)
    print(f"  ✅ {name}_preprocessed.csv ({df.shape})")

# %%
# Final summary
print(f"\n📋 PREPROCESSING COMPLETE")
print("=" * 40)

print(f"\nDatasets processed: {len(preprocessed_datasets)}")
for name, df in preprocessed_datasets.items():
    original_shape = datasets[name].shape
    final_shape = df.shape
    features_added = final_shape[1] - original_shape[1]
    print(f"  {name}: {original_shape} → {final_shape} (+{features_added} features)")

print(f"\n✅ Ready for next steps:")
print(f"  1. Data unification")
print(f"  2. EDA on unified dataset")
print(f"  3. Media Mix Modeling")

# %% [markdown]
# ## Preprocessing Summary
# 
# **What was done:**
# - Quality verification (no missing values, minimal outliers)
# - Conservative outlier treatment (only extreme cases)
# - Time feature engineering (cyclical features for seasonality)
# - Business features (seasons, holidays, month-end effects)
# 
# **Output:**
# - Individual preprocessed datasets saved in `data/processed/`
# - Ready for unification and modeling
# 
# **Next:** Run data unification to combine all datasets 