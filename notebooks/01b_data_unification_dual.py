# %% [markdown]
# # Dual Data Unification - Two Strategic Datasets
# 
# **STRATEGIC APPROACH**: Create two unified datasets for comparison
# 
# **Dataset A - Full Range**: 2022-2024 (9 channels, 156 weeks)
# - Excludes email (missing 2024)
# - Maximum data volume and recency
# - Better for trend analysis
# 
# **Dataset B - Complete Coverage**: 2022-2023 (10 channels, 104 weeks)
# - Includes all channels including email
# - Complete attribution modeling
# - Better for channel interaction analysis
# 
# **Goal**: Enable data-driven decision on which approach works better for MMM

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print("🔗 DUAL DATA UNIFICATION - Strategic Comparison")
print("=" * 60)
print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("🎯 CREATING TWO STRATEGIC DATASETS:")
print("   📊 Dataset A: Full Range (2022-2024, 9 channels)")
print("   📊 Dataset B: Complete Coverage (2022-2023, 10 channels)")

# %%
# Step 1: Load All Preprocessed Datasets
def load_preprocessed_datasets():
    """
    Load all preprocessed datasets from the processed directory
    """
    print(f"\n📂 LOADING PREPROCESSED DATASETS")
    print("=" * 40)
    
    processed_dir = '../data/processed'
    datasets = {}
    
    # Find all preprocessed CSV files
    csv_files = [f for f in os.listdir(processed_dir) if f.endswith('_preprocessed.csv')]
    
    if not csv_files:
        print("❌ No preprocessed files found!")
        return None
    
    print(f"Found {len(csv_files)} preprocessed datasets:")
    
    for file in csv_files:
        # Extract dataset name (remove _preprocessed.csv)
        dataset_name = file.replace('_preprocessed.csv', '')
        file_path = os.path.join(processed_dir, file)
        
        try:
            df = pd.read_csv(file_path)
            
            # Convert date column if it exists
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            datasets[dataset_name] = df
            
            # Display info
            date_range = f"{df['date'].min().date()} to {df['date'].max().date()}" if 'date' in df.columns else "No date"
            print(f"  ✅ {dataset_name}: {df.shape} | {date_range}")
            
        except Exception as e:
            print(f"  ❌ Error loading {file}: {e}")
    
    return datasets

# Load all datasets
preprocessed_datasets = load_preprocessed_datasets()

# %%
# Step 2: Define Date Ranges and Dataset Compositions
def analyze_dataset_strategies(datasets_dict):
    """
    Analyze the two strategic approaches
    """
    print(f"\n🎯 STRATEGIC DATASET ANALYSIS")
    print("=" * 50)
    
    if not datasets_dict:
        print("❌ No datasets to analyze!")
        return None, None
    
    # Define date ranges
    FULL_RANGE_START = '2022-01-03'
    FULL_RANGE_END = '2024-12-23'
    COMPLETE_RANGE_START = '2022-01-03'
    COMPLETE_RANGE_END = '2023-12-25'
    
    print(f"\n📊 DATASET A - FULL RANGE:")
    print(f"   📅 Period: {FULL_RANGE_START} to {FULL_RANGE_END}")
    print(f"   📈 Duration: ~156 weeks (3 years)")
    print(f"   📊 Channels: 9 (excludes email - missing 2024)")
    
    print(f"\n📊 DATASET B - COMPLETE COVERAGE:")
    print(f"   📅 Period: {COMPLETE_RANGE_START} to {COMPLETE_RANGE_END}")
    print(f"   📈 Duration: ~104 weeks (2 years)")
    print(f"   📊 Channels: 10 (includes all channels)")
    
    # Analyze which datasets fit each strategy
    strategy_a_datasets = {}
    strategy_b_datasets = {}
    
    print(f"\n🔍 DATASET COMPATIBILITY ANALYSIS:")
    
    for name, df in datasets_dict.items():
        if 'date' not in df.columns:
            continue
            
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        # Check Strategy A compatibility (2022-2024) - ANY overlap with period
        strategy_a_compatible = (max_date >= pd.Timestamp(FULL_RANGE_START) and 
                                min_date <= pd.Timestamp(FULL_RANGE_END))
        
        # Check Strategy B compatibility (2022-2023) - ANY overlap with period  
        strategy_b_compatible = (max_date >= pd.Timestamp(COMPLETE_RANGE_START) and 
                                min_date <= pd.Timestamp(COMPLETE_RANGE_END))
        
        print(f"  {name}:")
        print(f"    Range: {min_date.date()} to {max_date.date()}")
        print(f"    Strategy A (2022-2024): {'✅' if strategy_a_compatible else '❌'}")
        print(f"    Strategy B (2022-2023): {'✅' if strategy_b_compatible else '❌'}")
        
        if strategy_a_compatible:
            strategy_a_datasets[name] = df
        if strategy_b_compatible:
            strategy_b_datasets[name] = df
    
    print(f"\n📋 FINAL DATASET COMPOSITIONS:")
    print(f"  Strategy A: {len(strategy_a_datasets)} datasets - {list(strategy_a_datasets.keys())}")
    print(f"  Strategy B: {len(strategy_b_datasets)} datasets - {list(strategy_b_datasets.keys())}")
    
    return {
        'strategy_a': {
            'datasets': strategy_a_datasets,
            'date_range': (FULL_RANGE_START, FULL_RANGE_END),
            'name': 'Full Range (2022-2024)'
        },
        'strategy_b': {
            'datasets': strategy_b_datasets,
            'date_range': (COMPLETE_RANGE_START, COMPLETE_RANGE_END),
            'name': 'Complete Coverage (2022-2023)'
        }
    }

# Analyze strategies
strategies = analyze_dataset_strategies(preprocessed_datasets)

# %%
# Step 3: Create Unified Dataset Function
def create_unified_dataset_strategy(datasets_dict, date_range, strategy_name):
    """
    Create a unified dataset for a specific strategy
    """
    print(f"\n🔗 CREATING UNIFIED DATASET: {strategy_name}")
    print("=" * 50)
    
    start_date, end_date = date_range
    print(f"📅 Date range: {start_date} to {end_date}")
    print(f"📊 Datasets: {len(datasets_dict)}")
    
    # Choose base dataset (sales - most complete)
    if 'sales' not in datasets_dict:
        print("❌ Sales dataset not found!")
        return None
    
    # Filter base dataset to date range
    base_df = datasets_dict['sales'].copy()
    base_df = base_df[(base_df['date'] >= start_date) & (base_df['date'] <= end_date)]
    
    print(f"📅 Base dataset (sales): {base_df.shape}")
    print(f"   Date range: {base_df['date'].min().date()} to {base_df['date'].max().date()}")
    
    unified_df = base_df.copy()
    
    # Define time features to avoid duplication
    time_features = ['date', 'year', 'month', 'dayofyear', 'week', 'quarter',
                    'month_sin', 'month_cos', 'week_sin', 'week_cos', 
                    'season', 'holiday_period', 'is_month_end']
    
    # Merge other datasets
    merge_summary = {}
    
    for dataset_name, df in datasets_dict.items():
        if dataset_name == 'sales':
            continue
        
        print(f"\n  🔄 Merging {dataset_name}...")
        
        # Filter to date range
        merge_df = df.copy()
        merge_df = merge_df[(merge_df['date'] >= start_date) & (merge_df['date'] <= end_date)]
        
        # Identify columns to rename (business columns only)
        business_columns = [col for col in merge_df.columns if col not in time_features]
        
        # Add dataset prefix to business columns to avoid conflicts
        rename_dict = {col: f"{dataset_name}_{col}" for col in business_columns}
        merge_df = merge_df.rename(columns=rename_dict)
        
        print(f"    Filtered shape: {merge_df.shape}")
        print(f"    Renamed {len(business_columns)} business columns")
        
        # Merge on date
        before_shape = unified_df.shape
        unified_df = unified_df.merge(merge_df, on='date', how='left')
        after_shape = unified_df.shape
        
        # Remove duplicate time features (keep only from base dataset)
        duplicate_time_cols = [col for col in unified_df.columns 
                              if col.endswith('_x') or col.endswith('_y')]
        
        if duplicate_time_cols:
            # Keep the original columns (without suffix) and drop duplicates
            cols_to_drop = [col for col in duplicate_time_cols if col.endswith('_y')]
            cols_to_rename = {col: col.replace('_x', '') for col in duplicate_time_cols if col.endswith('_x')}
            
            unified_df = unified_df.drop(columns=cols_to_drop)
            unified_df = unified_df.rename(columns=cols_to_rename)
            
            print(f"    Removed {len(cols_to_drop)} duplicate time features")
        
        # Calculate merge statistics
        new_columns = after_shape[1] - before_shape[1] - len(duplicate_time_cols)
        records_with_data = unified_df[f"{dataset_name}_{business_columns[0]}"].notna().sum() if business_columns else 0
        
        merge_summary[dataset_name] = {
            'columns_added': new_columns,
            'records_with_data': records_with_data,
            'coverage_pct': (records_with_data / len(unified_df)) * 100
        }
        
        print(f"    Shape: {before_shape} → {unified_df.shape}")
        print(f"    Data coverage: {records_with_data}/{len(unified_df)} ({merge_summary[dataset_name]['coverage_pct']:.1f}%)")
    
    print(f"\n✅ UNIFIED DATASET CREATED: {strategy_name}")
    print(f"   Final shape: {unified_df.shape}")
    print(f"   Date range: {unified_df['date'].min().date()} to {unified_df['date'].max().date()}")
    
    return unified_df, merge_summary

# %%
# Step 4: Create Both Strategic Datasets
print(f"\n🚀 CREATING BOTH STRATEGIC DATASETS")
print("=" * 60)

unified_datasets = {}
merge_summaries = {}

for strategy_key, strategy_info in strategies.items():
    strategy_name = strategy_info['name']
    datasets = strategy_info['datasets']
    date_range = strategy_info['date_range']
    
    unified_df, merge_summary = create_unified_dataset_strategy(
        datasets, date_range, strategy_name
    )
    
    unified_datasets[strategy_key] = unified_df
    merge_summaries[strategy_key] = merge_summary

# %%
# Step 5: Compare the Two Strategies
def compare_strategies(unified_datasets, merge_summaries, strategies):
    """
    Compare the two strategic approaches
    """
    print(f"\n📊 STRATEGIC COMPARISON")
    print("=" * 50)
    
    comparison_data = []
    
    for strategy_key, unified_df in unified_datasets.items():
        strategy_info = strategies[strategy_key]
        merge_summary = merge_summaries[strategy_key]
        
        # Calculate metrics
        total_channels = len(merge_summary) + 1  # +1 for sales (base)
        total_weeks = len(unified_df)
        total_features = unified_df.shape[1]
        
        # Calculate data completeness
        non_date_cols = [col for col in unified_df.columns if col != 'date']
        completeness = (unified_df[non_date_cols].notna().sum().sum() / 
                       (len(unified_df) * len(non_date_cols))) * 100
        
        comparison_data.append({
            'Strategy': strategy_info['name'],
            'Date Range': f"{strategy_info['date_range'][0]} to {strategy_info['date_range'][1]}",
            'Weeks': total_weeks,
            'Channels': total_channels,
            'Features': total_features,
            'Completeness': f"{completeness:.1f}%",
            'Shape': unified_df.shape
        })
        
        print(f"\n🎯 {strategy_info['name'].upper()}:")
        print(f"   📅 Period: {strategy_info['date_range'][0]} to {strategy_info['date_range'][1]}")
        print(f"   📊 Shape: {unified_df.shape}")
        print(f"   📈 Weeks: {total_weeks}")
        print(f"   📺 Channels: {total_channels}")
        print(f"   🔧 Features: {total_features}")
        print(f"   ✅ Completeness: {completeness:.1f}%")
        
        # Channel breakdown
        print(f"   📋 Channels included:")
        channels = ['sales'] + list(merge_summary.keys())
        for i, channel in enumerate(channels, 1):
            print(f"     {i}. {channel}")
    
    # Create comparison table
    comparison_df = pd.DataFrame(comparison_data)
    print(f"\n📋 COMPARISON SUMMARY:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df

# Compare strategies
comparison_summary = compare_strategies(unified_datasets, merge_summaries, strategies)

# %%
# Step 6: Save Both Datasets
def save_unified_datasets(unified_datasets, strategies, merge_summaries):
    """
    Save both unified datasets with descriptive names
    """
    print(f"\n💾 SAVING BOTH UNIFIED DATASETS")
    print("=" * 40)
    
    processed_dir = '../data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    saved_files = {}
    
    for strategy_key, unified_df in unified_datasets.items():
        strategy_info = strategies[strategy_key]
        
        # Create descriptive filename
        if strategy_key == 'strategy_a':
            filename = "unified_dataset_full_range_2022_2024.csv"
            description = "Full Range (2022-2024, 9 channels, 156 weeks)"
        else:
            filename = "unified_dataset_complete_coverage_2022_2023.csv"
            description = "Complete Coverage (2022-2023, 10 channels, 104 weeks)"
        
        # Save dataset
        file_path = os.path.join(processed_dir, filename)
        unified_df.to_csv(file_path, index=False)
        
        saved_files[strategy_key] = {
            'filename': filename,
            'path': file_path,
            'description': description,
            'shape': unified_df.shape
        }
        
        print(f"  ✅ Saved: {filename}")
        print(f"     {description}")
        print(f"     Shape: {unified_df.shape}")
    
    # Save comparison report
    import json
    
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(item) for item in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        return obj
    
    dual_report = {
        'timestamp': datetime.now().isoformat(),
        'strategy_comparison': {
            'strategy_a_full_range': {
                'description': 'Full Range (2022-2024)',
                'date_range': strategies['strategy_a']['date_range'],
                'shape': unified_datasets['strategy_a'].shape,
                'channels': len(merge_summaries['strategy_a']) + 1,
                'filename': saved_files['strategy_a']['filename']
            },
            'strategy_b_complete': {
                'description': 'Complete Coverage (2022-2023)',
                'date_range': strategies['strategy_b']['date_range'],
                'shape': unified_datasets['strategy_b'].shape,
                'channels': len(merge_summaries['strategy_b']) + 1,
                'filename': saved_files['strategy_b']['filename']
            }
        },
        'merge_summaries': convert_numpy(merge_summaries),
        'recommendation': {
            'approach': 'Test both strategies in parallel',
            'strategy_a_pros': ['More recent data', 'Longer time series', 'Better trend analysis'],
            'strategy_a_cons': ['Missing email channel', 'Incomplete attribution'],
            'strategy_b_pros': ['Complete channel coverage', 'Full attribution', 'Channel interactions'],
            'strategy_b_cons': ['Less recent data', 'Shorter time series']
        }
    }
    
    report_path = os.path.join(processed_dir, "dual_unification_report.json")
    with open(report_path, 'w') as f:
        json.dump(dual_report, f, indent=2)
    
    print(f"  ✅ Saved: dual_unification_report.json")
    
    return saved_files, dual_report

# Save both datasets
saved_files, dual_report = save_unified_datasets(unified_datasets, strategies, merge_summaries)

# %%
# Step 7: Final Recommendations
print(f"\n🎯 STRATEGIC RECOMMENDATIONS")
print("=" * 50)

print(f"\n✅ BOTH DATASETS SUCCESSFULLY CREATED!")

print(f"\n📊 DATASET A - FULL RANGE:")
print(f"   📁 File: {saved_files['strategy_a']['filename']}")
print(f"   📈 {saved_files['strategy_a']['description']}")
print(f"   🎯 Best for: Trend analysis, recent performance, forecasting")
print(f"   ⚠️  Limitation: Missing email channel attribution")

print(f"\n📊 DATASET B - COMPLETE COVERAGE:")
print(f"   📁 File: {saved_files['strategy_b']['filename']}")
print(f"   📈 {saved_files['strategy_b']['description']}")
print(f"   🎯 Best for: Full attribution, channel interactions, ROI optimization")
print(f"   ⚠️  Limitation: Less recent data (missing 2024)")

print(f"\n🚀 NEXT STEPS - PARALLEL MMM DEVELOPMENT:")
print(f"   1. 📊 Run EDA on both datasets")
print(f"   2. 🤖 Develop identical MMM pipelines for both")
print(f"   3. 📈 Compare model performance and insights")
print(f"   4. 💰 Evaluate attribution accuracy and ROI insights")
print(f"   5. 🎯 Make data-driven decision on final approach")

print(f"\n💡 RECOMMENDATION:")
print(f"   Start with Dataset B (Complete Coverage) for initial MMM")
print(f"   Use Dataset A (Full Range) for trend validation")
print(f"   This gives you both complete attribution AND recent trends!")

print(f"\n✅ DUAL UNIFICATION COMPLETE!")

# %% [markdown]
# ## Dual Unification Strategy Complete! 🎉
# 
# ### 🎯 **Strategic Approach Implemented:**
# 
# #### **Dataset A - Full Range (2022-2024):**
# - **Period**: 156 weeks of recent data
# - **Channels**: 9 (excludes email)
# - **Strength**: Maximum data volume, recent trends
# - **Use Case**: Trend analysis, forecasting, recent performance
# 
# #### **Dataset B - Complete Coverage (2022-2023):**
# - **Period**: 104 weeks of complete data
# - **Channels**: 10 (includes all channels)
# - **Strength**: Full attribution modeling
# - **Use Case**: Channel interactions, ROI optimization
# 
# ### 📊 **Files Created:**
# - `unified_dataset_full_range_2022_2024.csv`
# - `unified_dataset_complete_coverage_2022_2023.csv`
# - `dual_unification_report.json`
# 
# ### 🚀 **Next Phase:**
# **Parallel MMM Development** - Run identical modeling pipelines on both datasets to determine which approach provides better business insights!
# 
# **This strategic approach ensures we make data-driven decisions about our modeling strategy.** 📈 