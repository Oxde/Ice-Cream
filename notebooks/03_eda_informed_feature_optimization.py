# %% [markdown]
# # 03 - EDA-Informed Feature Optimization for MMM
# 
# **Goal**: Optimize features based on EDA insights for Media Mix Modeling
# 
# **EDA Key Findings:**
# - Weather variables are CRITICAL (sunshine: 0.664, temp: 0.626 correlation)
# - Seasonality is crucial (week_cos: 0.724 correlation - strongest predictor!)
# - All media channels show effectiveness and budget justification
# - Perfect 3-year data quality with 100% media channel coverage
# 
# **Process:**
# 1. Load both corrected datasets (with EDA insights)
# 2. Business-informed feature selection (keep weather + seasonality!)
# 3. Remove only true redundancies and multicollinearity
# 4. Preserve business-critical relationships
# 5. Create final MMM-ready datasets

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("🎯 EDA-INFORMED FEATURE OPTIMIZATION FOR MMM")
print("=" * 60)
print("🔍 Based on EDA insights: Weather & seasonality are CRITICAL for ice cream!")
print("📊 Key correlations: week_cos(0.724), sunshine(0.664), temperature(0.626)")

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)

# %%
# Load both corrected unified datasets
print("📊 Loading corrected unified datasets...")

# Dataset 1: Complete channels including email (2022-2023)
complete_path = 'data/processed/mmm_dataset_complete_channels_2022_2023.csv'
# Dataset 2: Consistent channels excluding email (2022-2024)
consistent_path = 'data/processed/mmm_dataset_consistent_channels_2022_2024.csv'

try:
    df_complete = pd.read_csv(complete_path)
    df_complete['date'] = pd.to_datetime(df_complete['date'])
    print(f"✅ Complete channels (2022-2023): {df_complete.shape}")
    print(f"  📅 Date range: {df_complete['date'].min().date()} to {df_complete['date'].max().date()}")
    print(f"  📧 Includes email: {'email_email_campaigns' in df_complete.columns}")
    
    df_consistent = pd.read_csv(consistent_path)
    df_consistent['date'] = pd.to_datetime(df_consistent['date'])
    print(f"✅ Consistent channels (2022-2024): {df_consistent.shape}")
    print(f"  📅 Date range: {df_consistent['date'].min().date()} to {df_consistent['date'].max().date()}")
    print(f"  📧 Excludes email: {'email_email_campaigns' not in df_consistent.columns}")
    
except Exception as e:
    print(f"❌ Error loading datasets: {e}")
    exit()

# %%
# STEP 1: Analyze cross-channel correlations (the main issue!)
def analyze_cross_channel_correlations(df, threshold=0.75):
    """
    Analyze correlations between GRPs and Spend across all channels
    This is the main multicollinearity issue in MMM!
    """
    print(f"\n🔍 CROSS-CHANNEL CORRELATION ANALYSIS")
    print("=" * 45)
    
    # Get all numeric columns except date and basic features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['date', 'sales', 'year', 'month', 'dayofyear', 'week', 'quarter', 
                   'month_sin', 'month_cos', 'week_sin', 'week_cos', 'holiday_period']
    
    media_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"📺 Media/Weather columns to analyze: {len(media_cols)}")
    
    # Calculate correlation matrix for media variables
    if len(media_cols) < 2:
        print("Not enough media columns for correlation analysis")
        return [], []
    
    corr_matrix = df[media_cols].corr().abs()
    
    # Find high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if corr_value > threshold:
                pair = (corr_matrix.columns[i], corr_matrix.columns[j], corr_value)
                high_corr_pairs.append(pair)
    
    # Sort by correlation strength
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n⚠️  HIGH CORRELATIONS FOUND (>{threshold}):")
    if high_corr_pairs:
        for col1, col2, corr in high_corr_pairs:
            print(f"  {col1} ↔ {col2}: {corr:.3f}")
    else:
        print(f"  ✅ No high correlations found")
    
    # Identify GRP vs Spend pairs specifically
    grp_spend_pairs = []
    for col1, col2, corr in high_corr_pairs:
        if ('grp' in col1.lower() and ('spend' in col2.lower() or 'cost' in col2.lower())) or \
           ('grp' in col2.lower() and ('spend' in col1.lower() or 'cost' in col1.lower())):
            grp_spend_pairs.append((col1, col2, corr))
    
    if grp_spend_pairs:
        print(f"\n🚨 GRP vs SPEND PERFECT CORRELATIONS (MMM Problem!):")
        for col1, col2, corr in grp_spend_pairs:
            print(f"  {col1} ↔ {col2}: {corr:.3f}")
    
    return high_corr_pairs, grp_spend_pairs

# Analyze correlations for both datasets
print(f"\n🔍 ANALYZING BOTH DATASETS:")
print("=" * 50)

print(f"\n📊 COMPLETE CHANNELS DATASET (2022-2023):")
high_corrs_complete, grp_spend_corrs_complete = analyze_cross_channel_correlations(df_complete)

print(f"\n📊 CONSISTENT CHANNELS DATASET (2022-2024):")
high_corrs_consistent, grp_spend_corrs_consistent = analyze_cross_channel_correlations(df_consistent)

# %%
# STEP 2: EDA-Informed Feature optimization for MMM
def optimize_features_for_mmm(df, grp_spend_corrs, high_corrs, dataset_name=""):
    """
    Optimize features for MMM based on EDA insights and business logic
    
    EDA KEY INSIGHTS:
    - Weather variables CRITICAL for ice cream (sunshine: 0.664, temp: 0.626)
    - Seasonality crucial (week_cos: 0.724 - strongest predictor!)
    - All media channels effective and budget-justified
    """
    print(f"\n⚙️ EDA-INFORMED FEATURE OPTIMIZATION - {dataset_name}")
    print("=" * 60)
    print(f"🔍 Applying business insights from EDA analysis...")
    
    df_optimized = df.copy()
    removed_features = []
    
    print(f"🔧 Starting with {df.shape[1]} features")
    
    # 1. Remove redundant time features (EDA-informed: keep crucial seasonality!)
    print(f"\n1️⃣ EDA-Informed Time Feature Optimization...")
    print(f"    🔍 EDA insight: week_cos has 0.724 correlation - KEEP!")
    print(f"    🔍 EDA insight: month_sin/cos capture ice cream seasonality - KEEP!")
    
    # Remove only truly redundant linear time features
    redundant_time = ['year', 'month', 'week', 'quarter', 'dayofyear']
    for feature in redundant_time:
        if feature in df_optimized.columns:
            df_optimized = df_optimized.drop(columns=[feature])
            removed_features.append(f"{feature} (linear time - redundant with cyclical)")
            print(f"  ❌ Removed: {feature} (keeping cyclical versions)")
    
    # KEEP the important cyclical features based on EDA
    cyclical_features = ['month_sin', 'month_cos', 'week_sin', 'week_cos']
    for feature in cyclical_features:
        if feature in df_optimized.columns:
            if feature == 'week_cos':
                print(f"  ✅ KEPT: {feature} (0.724 correlation - strongest predictor!)")
            else:
                print(f"  ✅ KEPT: {feature} (crucial for ice cream seasonality)")
    
    # 2. Handle GRP vs Spend multicollinearity (keep spend, remove GRPs)
    print(f"\n2️⃣ Handling GRP vs Spend multicollinearity...")
    for col1, col2, corr in grp_spend_corrs:
        # Determine which one to keep (prefer spend/cost)
        if 'grp' in col1.lower() and ('spend' in col2.lower() or 'cost' in col2.lower()):
            # Keep col2 (spend), remove col1 (GRP)
            if col1 in df_optimized.columns:
                df_optimized = df_optimized.drop(columns=[col1])
                removed_features.append(f"{col1} (corr={corr:.3f} with {col2})")
                print(f"  ❌ Removed: {col1} (keeping {col2})")
        elif 'grp' in col2.lower() and ('spend' in col1.lower() or 'cost' in col1.lower()):
            # Keep col1 (spend), remove col2 (GRP)
            if col2 in df_optimized.columns:
                df_optimized = df_optimized.drop(columns=[col2])
                removed_features.append(f"{col2} (corr={corr:.3f} with {col1})")
                print(f"  ❌ Removed: {col2} (keeping {col1})")
    
    # 3. Handle other high correlations (impressions vs cost, etc.)
    print(f"\n3️⃣ Handling other high correlations...")
    for col1, col2, corr in high_corrs:
        # Skip if already handled in GRP vs spend
        if (col1, col2, corr) in grp_spend_corrs:
            continue
        
        # For impressions vs cost, keep cost
        if col1 in df_optimized.columns and col2 in df_optimized.columns:
            if 'impression' in col1.lower() and 'cost' in col2.lower():
                df_optimized = df_optimized.drop(columns=[col1])
                removed_features.append(f"{col1} (corr={corr:.3f} with {col2})")
                print(f"  ❌ Removed: {col1} (keeping {col2})")
            elif 'impression' in col2.lower() and 'cost' in col1.lower():
                df_optimized = df_optimized.drop(columns=[col2])
                removed_features.append(f"{col2} (corr={corr:.3f} with {col1})")
                print(f"  ❌ Removed: {col2} (keeping {col1})")
    
    # 3b. EDA-Informed Weather Variable Optimization (CRITICAL for ice cream!)
    print(f"\n3️⃣b EDA-Informed Weather Optimization...")
    print(f"    🔍 EDA insight: sunshine_duration has 0.664 correlation - CRITICAL!")
    print(f"    🔍 EDA insight: temperature_mean has 0.626 correlation - CRITICAL!")
    print(f"    🌞 Weather is one of the strongest predictors for ice cream sales!")
    
    # Remove only redundant temperature variables (keep mean, remove min/max)
    weather_to_remove = ['weather_temperature_min', 'weather_temperature_max']
    for weather_var in weather_to_remove:
        if weather_var in df_optimized.columns:
            df_optimized = df_optimized.drop(columns=[weather_var])
            removed_features.append(f"{weather_var} (redundant with temperature_mean)")
            print(f"  ❌ Removed: {weather_var} (keeping temperature_mean)")
    
    # KEEP the critical weather variables based on EDA
    critical_weather = ['weather_sunshine_duration', 'weather_temperature_mean']
    for weather_var in critical_weather:
        if weather_var in df_optimized.columns:
            if 'sunshine' in weather_var:
                print(f"  ✅ KEPT: {weather_var} (0.664 correlation - ice cream weather!)")
            else:
                print(f"  ✅ KEPT: {weather_var} (0.626 correlation - temperature drives sales!)")
    
    # 3c. EDA-Informed granular feature handling
    print(f"\n3️⃣c EDA-Informed Granular Feature Review...")
    print(f"    🔍 EDA shows is_month_end has 0.117 correlation - but too granular for MMM")
    
    # Remove overly granular features
    granular_features = ['is_month_end']
    for feature in granular_features:
        if feature in df_optimized.columns:
            df_optimized = df_optimized.drop(columns=[feature])
            removed_features.append(f"{feature} (too granular for MMM, captured by seasonality)")
            print(f"  ❌ Removed: {feature} (too granular, seasonality captured by cyclical features)")
    
    # 4. EDA-Informed categorical variable handling
    print(f"\n4️⃣ EDA-Informed Categorical Variable Handling...")
    print(f"    🔍 EDA shows promotions have 30% coverage - sporadic nature is normal")
    print(f"    🔍 3 promotion types found - manageable for MMM")
    
    categorical_cols = df_optimized.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'date']
    
    for col in categorical_cols:
        # For promotion type - EDA shows this is important to keep
        if 'promo' in col.lower():
            unique_values = df_optimized[col].nunique()
            non_missing_count = df_optimized[col].notna().sum()
            total_count = len(df_optimized)
            coverage_pct = (non_missing_count / total_count) * 100
            
            print(f"    📊 {col}: {unique_values} categories, {coverage_pct:.1f}% coverage")
            
            if unique_values <= 5:  # Manageable categories
                # Convert to dummy variables
                dummies = pd.get_dummies(df_optimized[col], prefix=col, dummy_na=True)
                df_optimized = df_optimized.drop(columns=[col])
                df_optimized = pd.concat([df_optimized, dummies], axis=1)
                print(f"  ✅ KEPT: {col} converted to {len(dummies.columns)} dummy variables")
                print(f"    🔍 Promotions are important for MMM despite sparse coverage")
            else:
                df_optimized = df_optimized.drop(columns=[col])
                removed_features.append(f"{col} (too many categories: {unique_values})")
                print(f"  ❌ Removed: {col} (too many categories)")
        else:
            # Remove other categorical variables
            df_optimized = df_optimized.drop(columns=[col])
            removed_features.append(f"{col} (categorical - not business critical)")
            print(f"  ❌ Removed: {col} (categorical)")
    
    print(f"\n✅ EDA-Informed Optimization Complete:")
    print(f"  📊 Before: {df.shape[1]} features")
    print(f"  📊 After: {df_optimized.shape[1]} features")
    print(f"  🗑️ Removed: {len(removed_features)} features")
    print(f"  🌞 PRESERVED: Critical weather variables (sunshine + temperature)")
    print(f"  📅 PRESERVED: Critical seasonality (week_cos + monthly cycles)")
    print(f"  💰 PRESERVED: All effective media channels")
    
    return df_optimized, removed_features

# Apply feature optimization to both datasets
print(f"\n🔧 OPTIMIZING BOTH DATASETS:")
print("=" * 40)

df_complete_optimized, removed_features_complete = optimize_features_for_mmm(
    df_complete, grp_spend_corrs_complete, high_corrs_complete, "COMPLETE CHANNELS"
)

df_consistent_optimized, removed_features_consistent = optimize_features_for_mmm(
    df_consistent, grp_spend_corrs_consistent, high_corrs_consistent, "CONSISTENT CHANNELS"
)

# %%
# STEP 3: Check feature-to-sample ratio
def check_feature_sample_ratio(df):
    """
    Check if we have appropriate feature-to-sample ratio for MMM
    """
    print(f"\n📊 FEATURE-TO-SAMPLE RATIO CHECK")
    print("=" * 40)
    
    n_rows = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude date and target variable
    feature_cols = [col for col in numeric_cols if col not in ['date', 'sales']]
    n_features = len(feature_cols)
    
    recommended_max_features = n_rows // 10  # Rule of thumb: 1 feature per 10 samples
    feature_ratio = n_features / n_rows
    
    print(f"📏 Dataset size: {n_rows} samples")
    print(f"🔧 Current features: {n_features}")
    print(f"📋 Recommended max features: {recommended_max_features}")
    print(f"📊 Feature-to-sample ratio: {feature_ratio:.3f}")
    print(f"📈 Status: {'✅ Good' if n_features <= recommended_max_features else '⚠️ Too many features'}")
    
    if n_features > recommended_max_features:
        excess_features = n_features - recommended_max_features
        print(f"⚠️  Consider removing {excess_features} more features")
    
    return n_features <= recommended_max_features

print(f"\n📊 CHECKING FEATURE-SAMPLE RATIOS:")
print("=" * 45)

print(f"\n📊 Complete Channels Dataset:")
ratio_check_complete = check_feature_sample_ratio(df_complete_optimized)

print(f"\n📊 Consistent Channels Dataset:")
ratio_check_consistent = check_feature_sample_ratio(df_consistent_optimized)

# %%
# STEP 4: Temporal validation setup
def setup_temporal_validation(df, test_months=6):
    """
    Set up proper temporal validation for MMM (no data leakage!)
    """
    print(f"\n📅 TEMPORAL VALIDATION SETUP")
    print("=" * 35)
    
    if 'date' not in df.columns:
        print("❌ Date column required for temporal validation")
        return None
    
    df_sorted = df.sort_values('date').reset_index(drop=True)
    
    # Split data temporally
    max_date = df_sorted['date'].max()
    split_date = max_date - pd.DateOffset(months=test_months)
    
    train_df = df_sorted[df_sorted['date'] < split_date].copy()
    test_df = df_sorted[df_sorted['date'] >= split_date].copy()
    
    print(f"📊 Train set: {len(train_df)} samples")
    print(f"  📅 Period: {train_df['date'].min().date()} to {train_df['date'].max().date()}")
    print(f"📊 Test set: {len(test_df)} samples")
    print(f"  📅 Period: {test_df['date'].min().date()} to {test_df['date'].max().date()}")
    
    # Check for seasonality coverage
    train_months = train_df['date'].dt.month.unique()
    test_months = test_df['date'].dt.month.unique()
    
    print(f"\n🔄 Seasonality coverage:")
    print(f"  Train months: {sorted(train_months)}")
    print(f"  Test months: {sorted(test_months)}")
    
    # Rolling forecast validation setup
    print(f"\n🔄 Rolling forecast validation periods:")
    validation_periods = []
    
    # Create multiple validation periods (holdout forecasting)
    for months_back in [3, 6, 9, 12]:
        if months_back <= len(df_sorted) // 4:  # Only if we have enough data
            val_split = max_date - pd.DateOffset(months=months_back)
            val_train = df_sorted[df_sorted['date'] < val_split]
            val_test = df_sorted[df_sorted['date'] >= val_split]
            
            if len(val_train) > 20 and len(val_test) > 4:  # Minimum sizes
                validation_periods.append({
                    'months_back': months_back,
                    'split_date': val_split,
                    'train_size': len(val_train),
                    'test_size': len(val_test)
                })
                print(f"  {months_back}M back: {len(val_train)} train, {len(val_test)} test")
    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'validation_periods': validation_periods,
        'split_date': split_date
    }

# Set up temporal validation for both datasets
print(f"\n📅 SETTING UP TEMPORAL VALIDATION:")
print("=" * 45)

print(f"\n📅 Complete Channels Dataset (2022-2023):")
validation_setup_complete = setup_temporal_validation(df_complete_optimized, test_months=3)

print(f"\n📅 Consistent Channels Dataset (2022-2024):")
validation_setup_consistent = setup_temporal_validation(df_consistent_optimized, test_months=6)

# %%
# STEP 5: Final feature summary and correlation heatmap
def create_final_feature_summary(df):
    """
    Create final summary of optimized features
    """
    print(f"\n📋 FINAL FEATURE SUMMARY")
    print("=" * 30)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in ['date', 'sales']]
    
    # Categorize features
    feature_categories = {
        'Time/Seasonality': [],
        'Media Spend': [],
        'Media Other': [],
        'Weather': [],
        'Promotion': [],
        'Other': []
    }
    
    for col in feature_cols:
        col_lower = col.lower()
        if any(x in col_lower for x in ['month_sin', 'month_cos', 'week_sin', 'week_cos', 'season', 'holiday']):
            feature_categories['Time/Seasonality'].append(col)
        elif any(x in col_lower for x in ['spend', 'cost']):
            feature_categories['Media Spend'].append(col)
        elif any(x in col_lower for x in ['grp', 'impression', 'campaign', 'click']):
            feature_categories['Media Other'].append(col)
        elif 'weather' in col_lower:
            feature_categories['Weather'].append(col)
        elif 'promo' in col_lower:
            feature_categories['Promotion'].append(col)
        else:
            feature_categories['Other'].append(col)
    
    print(f"🎯 Target variable: sales")
    print(f"📊 Total features: {len(feature_cols)}")
    print()
    
    for category, cols in feature_categories.items():
        if cols:
            print(f"📈 {category}: {len(cols)} features")
            for col in cols:
                print(f"  - {col}")
    
    # Create correlation heatmap of final features
    if len(feature_cols) > 1:
        plt.figure(figsize=(12, 10))
        corr_matrix = df[feature_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Final Feature Correlation Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    return feature_categories

print(f"\n📋 CREATING FEATURE SUMMARIES:")
print("=" * 40)

print(f"\n📋 Complete Channels Dataset:")
feature_summary_complete = create_final_feature_summary(df_complete_optimized)

print(f"\n📋 Consistent Channels Dataset:")
feature_summary_consistent = create_final_feature_summary(df_consistent_optimized)

# %%
# STEP 6: Save MMM-ready datasets
def save_mmm_ready_datasets(df_complete, df_consistent, validation_complete, validation_consistent, 
                           removed_complete, removed_consistent, summary_complete, summary_consistent):
    """
    Save both final MMM-ready datasets and configurations
    """
    print(f"\n💾 SAVING MMM-READY DATASETS")
    print("=" * 40)
    
    # Create output directory
    output_dir = 'data/mmm_ready'
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save both optimized datasets
    complete_path = os.path.join(output_dir, 'mmm_complete_channels_2022_2023.csv')
    consistent_path = os.path.join(output_dir, 'mmm_consistent_channels_2022_2024.csv')
    
    df_complete.to_csv(complete_path, index=False)
    df_consistent.to_csv(consistent_path, index=False)
    
    print(f"✅ Saved: mmm_complete_channels_2022_2023.csv ({df_complete.shape})")
    print(f"  📧 Includes email campaigns")
    print(f"  📅 Period: 2022-2023 (2 years)")
    
    print(f"✅ Saved: mmm_consistent_channels_2022_2024.csv ({df_consistent.shape})")
    print(f"  🚫 Excludes email campaigns")
    print(f"  📅 Period: 2022-2024 (3 years)")
    
    # Save train/test splits for both datasets
    if validation_complete:
        complete_train_path = os.path.join(output_dir, 'complete_channels_train_set.csv')
        complete_test_path = os.path.join(output_dir, 'complete_channels_test_set.csv')
        
        validation_complete['train_df'].to_csv(complete_train_path, index=False)
        validation_complete['test_df'].to_csv(complete_test_path, index=False)
        
        print(f"✅ Saved: complete_channels_train_set.csv ({validation_complete['train_df'].shape})")
        print(f"✅ Saved: complete_channels_test_set.csv ({validation_complete['test_df'].shape})")
    
    if validation_consistent:
        consistent_train_path = os.path.join(output_dir, 'consistent_channels_train_set.csv')
        consistent_test_path = os.path.join(output_dir, 'consistent_channels_test_set.csv')
        
        validation_consistent['train_df'].to_csv(consistent_train_path, index=False)
        validation_consistent['test_df'].to_csv(consistent_test_path, index=False)
        
        print(f"✅ Saved: consistent_channels_train_set.csv ({validation_consistent['train_df'].shape})")
        print(f"✅ Saved: consistent_channels_test_set.csv ({validation_consistent['test_df'].shape})")
    
    # Save optimization reports for both datasets
    import json
    
    # Complete channels report
    complete_report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'dataset_type': 'complete_channels_with_email',
        'period': '2022-2023',
        'original_shape': df_complete.shape,
        'optimized_shape': df_complete.shape,
        'removed_features': removed_complete,
        'feature_categories': summary_complete,
        'validation_setup': {
            'method': 'temporal_split',
            'test_months': 3,
            'split_date': validation_complete['split_date'].isoformat() if validation_complete else None,
            'train_size': len(validation_complete['train_df']) if validation_complete else None,
            'test_size': len(validation_complete['test_df']) if validation_complete else None,
            'validation_periods': validation_complete['validation_periods'] if validation_complete else []
        }
    }
    
    # Consistent channels report
    consistent_report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'dataset_type': 'consistent_channels_no_email',
        'period': '2022-2024',
        'original_shape': df_consistent.shape,
        'optimized_shape': df_consistent.shape,
        'removed_features': removed_consistent,
        'feature_categories': summary_consistent,
        'validation_setup': {
            'method': 'temporal_split',
            'test_months': 6,
            'split_date': validation_consistent['split_date'].isoformat() if validation_consistent else None,
            'train_size': len(validation_consistent['train_df']) if validation_consistent else None,
            'test_size': len(validation_consistent['test_df']) if validation_consistent else None,
            'validation_periods': validation_consistent['validation_periods'] if validation_consistent else []
        }
    }
    
    # Combined report
    combined_report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'datasets': {
            'complete_channels': complete_report,
            'consistent_channels': consistent_report
        },
        'mmm_guidelines': {
            'feature_sample_ratio': 'Aim for 1 feature per 10 samples',
            'validation_method': 'Temporal cross-validation (no random splits)',
            'avoided_issues': ['GRP-spend multicollinearity', 'data leakage', 'excess features', 'email inconsistency'],
            'best_practices': ['Holdout forecasting', 'Seasonal coverage', 'Business logic validation']
        },
        'dataset_selection_guide': {
            'complete_channels': 'Use for email effectiveness analysis and short-term insights (2022-2023)',
            'consistent_channels': 'Use for long-term MMM and channel optimization (2022-2024)'
        }
    }
    
    # Convert non-serializable objects
    def convert_for_json(obj):
        if isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        else:
            return obj
    
    # Save individual reports
    complete_report_path = os.path.join(output_dir, 'mmm_complete_channels_report.json')
    consistent_report_path = os.path.join(output_dir, 'mmm_consistent_channels_report.json')
    combined_report_path = os.path.join(output_dir, 'mmm_optimization_report.json')
    
    with open(complete_report_path, 'w') as f:
        json.dump(convert_for_json(complete_report), f, indent=2)
    
    with open(consistent_report_path, 'w') as f:
        json.dump(convert_for_json(consistent_report), f, indent=2)
        
    with open(combined_report_path, 'w') as f:
        json.dump(convert_for_json(combined_report), f, indent=2)
    
    print(f"✅ Saved: mmm_complete_channels_report.json")
    print(f"✅ Saved: mmm_consistent_channels_report.json")
    print(f"✅ Saved: mmm_optimization_report.json")
    print(f"\n📂 All files saved to: {output_dir}")

# Save MMM-ready datasets
save_mmm_ready_datasets(
    df_complete_optimized, df_consistent_optimized,
    validation_setup_complete, validation_setup_consistent,
    removed_features_complete, removed_features_consistent,
    feature_summary_complete, feature_summary_consistent
)

# %%
# STEP 7: Final recommendations
print(f"\n🚀 DUAL MMM OPTIMIZATION COMPLETE!")
print("=" * 50)

print(f"\n✅ FEATURE OPTIMIZATION RESULTS:")
print(f"\n📊 Complete Channels Dataset (2022-2023):")
print(f"  📊 Optimized features: {df_complete_optimized.shape[1]}")
print(f"  📏 Feature-sample ratio: {'✅ Good' if ratio_check_complete else '⚠️ Needs review'}")
print(f"  📧 Includes email campaigns")

print(f"\n📊 Consistent Channels Dataset (2022-2024):")
print(f"  📊 Optimized features: {df_consistent_optimized.shape[1]}")
print(f"  📏 Feature-sample ratio: {'✅ Good' if ratio_check_consistent else '⚠️ Needs review'}")
print(f"  🚫 Excludes email campaigns")

print(f"\n🔧 EDA-INFORMED OPTIMIZATIONS APPLIED:")
print(f"  ✅ Fixed email campaign inconsistency issue")
print(f"  ✅ PRESERVED critical weather variables (0.664 & 0.626 correlations!)")
print(f"  ✅ PRESERVED crucial seasonality (week_cos: 0.724 correlation!)")
print(f"  ✅ Removed GRP-spend perfect correlations (kept effective spend vars)")
print(f"  ✅ KEPT all media channels (EDA shows all are budget-justified)")
print(f"  ✅ Business-informed categorical handling")
print(f"  ✅ Set up temporal validation (no data leakage)")

print(f"\n📅 VALIDATION SETUP:")
print(f"  🎯 Method: Temporal cross-validation")
print(f"  📊 Complete dataset: {validation_setup_complete['train_df'].shape[0] if validation_setup_complete else 'N/A'} train, {validation_setup_complete['test_df'].shape[0] if validation_setup_complete else 'N/A'} test")
print(f"  📊 Consistent dataset: {validation_setup_consistent['train_df'].shape[0] if validation_setup_consistent else 'N/A'} train, {validation_setup_consistent['test_df'].shape[0] if validation_setup_consistent else 'N/A'} test")

print(f"\n📋 DATASET SELECTION GUIDE:")
print(f"  📧 Complete Channels: Email effectiveness analysis + short-term insights")
print(f"  🎯 Consistent Channels: Long-term MMM + channel optimization")

print(f"\n🎯 NEXT STEPS:")
print(f"  1. 📊 Use EDA-optimized datasets for MMM development")
print(f"  2. 🌞 Leverage weather variables as key predictors")
print(f"  3. 📅 Capture seasonal cycles in MMM (crucial for ice cream!)")
print(f"  4. 🤖 Build sophisticated MMM with business-informed features")
print(f"  5. ✅ Validate against EDA insights and business logic")

print(f"\n✅ READY FOR ADVANCED MMM WITH EDA-INFORMED FEATURES!")

# %% [markdown]
# ## EDA-Informed MMM Feature Optimization Complete! 🎯
# 
# ### ✅ **EDA-Informed Optimizations Applied:**
# - **Email Inconsistency Fixed**: Two datasets for different analysis needs
# - **Weather Variables PRESERVED**: Critical predictors (sunshine: 0.664, temp: 0.626)
# - **Seasonality PRESERVED**: Key predictor (week_cos: 0.724 correlation!)
# - **All Media Channels KEPT**: EDA shows budget justification and effectiveness
# - **Multicollinearity Fixed**: Removed GRP variables, kept effective spend/cost
# - **Business-Informed Decisions**: Based on ice cream sales patterns
# - **Temporal Validation**: Proper train/test splits (no data leakage)
# 
# ### 📊 **Files Created:**
# - `mmm_complete_channels_2022_2023.csv` - Dataset WITH email campaigns
# - `mmm_consistent_channels_2022_2024.csv` - Dataset WITHOUT email campaigns
# - Train/test splits for both datasets
# - Individual and combined optimization reports
# 
# ### 📋 **Dataset Selection Guide:**
# - **Complete Channels (2022-2023)**: Use for email effectiveness analysis
# - **Consistent Channels (2022-2024)**: Use for long-term MMM and optimization
# 
# ### 🚀 **Next Phase:**
# Ready for MMM modeling with two optimized datasets and proper validation framework! 