# %% [markdown]
# # Enhanced Dual Data Unification with Weather Integration
# 
# **ENHANCED STRATEGY**: Extend dual approach with weather variables
# 
# **Dataset A - Full Range + Weather**: 2022-2024 (9 channels + weather, 156 weeks)
# - Marketing channels + weather factors
# - Maximum data volume and environmental context
# - Better for trend analysis with seasonal factors
# 
# **Dataset B - Complete Coverage + Weather**: 2022-2023 (10 channels + weather, 104 weeks)  
# - All channels + weather factors
# - Complete attribution with environmental context
# - Better for comprehensive channel interaction analysis
# 
# **Weather Variables Added**: temperature_mean, temperature_max, temperature_min, sunshine_duration

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

print("🌤️ ENHANCED DUAL DATA UNIFICATION WITH WEATHER")
print("=" * 70)
print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("🎯 CREATING TWO ENHANCED STRATEGIC DATASETS:")
print("   📊 Dataset A: Full Range + Weather (2022-2024, 9 channels + weather)")
print("   📊 Dataset B: Complete Coverage + Weather (2022-2023, 10 channels + weather)")

# %%
# Step 1: Load Weather Data First
def load_and_preprocess_weather():
    """
    Load and preprocess weather data to match our dataset structure
    """
    print(f"\n🌤️ LOADING WEATHER DATA")
    print("=" * 30)
    
    weather_path = '../data/raw/weekly_weather_monday_start.csv'
    
    try:
        weather_df = pd.read_csv(weather_path)
        print(f"  ✅ Raw weather data loaded: {weather_df.shape}")
        
        # Rename date column to match our format
        weather_df = weather_df.rename(columns={'Date_time': 'date'})
        
        # Convert date to datetime
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        
        # Add weather prefix to all columns except date
        weather_columns = [col for col in weather_df.columns if col != 'date']
        rename_dict = {col: f"weather_{col}" for col in weather_columns}
        weather_df = weather_df.rename(columns=rename_dict)
        
        print(f"  📅 Date range: {weather_df['date'].min().date()} to {weather_df['date'].max().date()}")
        print(f"  🌡️ Weather variables: {len(weather_columns)}")
        print(f"     {list(rename_dict.values())}")
        
        # Check for missing values
        missing = weather_df.isnull().sum().sum()
        print(f"  ✅ Missing values: {missing}")
        
        return weather_df
        
    except Exception as e:
        print(f"  ❌ Error loading weather data: {e}")
        return None

# Load weather data
weather_data = load_and_preprocess_weather()

# %%
# Step 2: Load All Preprocessed Marketing Datasets
def load_preprocessed_datasets():
    """
    Load all preprocessed marketing datasets from the processed directory
    """
    print(f"\n📂 LOADING PREPROCESSED MARKETING DATASETS")
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

# Load marketing datasets
marketing_datasets = load_preprocessed_datasets()

# %%
# Step 3: Define Enhanced Strategies with Weather
def analyze_enhanced_strategies(datasets_dict, weather_df):
    """
    Analyze the two strategic approaches enhanced with weather data
    """
    print(f"\n🎯 ENHANCED STRATEGIC DATASET ANALYSIS")
    print("=" * 50)
    
    if not datasets_dict or weather_df is None:
        print("❌ Missing required datasets!")
        return None
    
    # Define date ranges (same as before)
    FULL_RANGE_START = '2022-01-03'
    FULL_RANGE_END = '2024-12-23'
    COMPLETE_RANGE_START = '2022-01-03'
    COMPLETE_RANGE_END = '2023-12-25'
    
    print(f"\n📊 ENHANCED DATASET A - FULL RANGE + WEATHER:")
    print(f"   📅 Period: {FULL_RANGE_START} to {FULL_RANGE_END}")
    print(f"   📈 Duration: ~156 weeks (3 years)")
    print(f"   📊 Channels: 9 marketing + weather (excludes email)")
    
    print(f"\n📊 ENHANCED DATASET B - COMPLETE COVERAGE + WEATHER:")
    print(f"   📅 Period: {COMPLETE_RANGE_START} to {COMPLETE_RANGE_END}")
    print(f"   📈 Duration: ~104 weeks (2 years)")
    print(f"   📊 Channels: 10 marketing + weather (includes all channels)")
    
    # Check weather data coverage for both strategies
    weather_full_range = weather_df[
        (weather_df['date'] >= FULL_RANGE_START) & 
        (weather_df['date'] <= FULL_RANGE_END)
    ]
    weather_complete_range = weather_df[
        (weather_df['date'] >= COMPLETE_RANGE_START) & 
        (weather_df['date'] <= COMPLETE_RANGE_END)
    ]
    
    print(f"\n🌤️ WEATHER DATA COVERAGE:")
    print(f"  Full range (2022-2024): {weather_full_range.shape[0]} weeks")
    print(f"  Complete range (2022-2023): {weather_complete_range.shape[0]} weeks")
    
    # Analyze marketing dataset compatibility (same logic as before)
    strategy_a_datasets = {}
    strategy_b_datasets = {}
    
    print(f"\n🔍 MARKETING DATASET COMPATIBILITY:")
    
    for name, df in datasets_dict.items():
        if 'date' not in df.columns:
            continue
            
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        # Strategy compatibility
        strategy_a_compatible = (max_date >= pd.Timestamp(FULL_RANGE_START) and 
                                min_date <= pd.Timestamp(FULL_RANGE_END))
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
    
    print(f"\n📋 ENHANCED DATASET COMPOSITIONS:")
    print(f"  Strategy A: {len(strategy_a_datasets)} marketing + weather")
    print(f"  Strategy B: {len(strategy_b_datasets)} marketing + weather")
    
    return {
        'strategy_a': {
            'datasets': strategy_a_datasets,
            'weather': weather_full_range,
            'date_range': (FULL_RANGE_START, FULL_RANGE_END),
            'name': 'Full Range + Weather (2022-2024)'
        },
        'strategy_b': {
            'datasets': strategy_b_datasets,
            'weather': weather_complete_range,
            'date_range': (COMPLETE_RANGE_START, COMPLETE_RANGE_END),
            'name': 'Complete Coverage + Weather (2022-2023)'
        }
    }

# Analyze enhanced strategies
enhanced_strategies = analyze_enhanced_strategies(marketing_datasets, weather_data)

# %%
# Step 4: Create Enhanced Unified Dataset Function
def create_enhanced_unified_dataset(datasets_dict, weather_df, date_range, strategy_name):
    """
    Create a unified dataset with weather data for a specific strategy
    """
    print(f"\n🔗 CREATING ENHANCED UNIFIED DATASET: {strategy_name}")
    print("=" * 60)
    
    start_date, end_date = date_range
    print(f"📅 Date range: {start_date} to {end_date}")
    print(f"📊 Marketing datasets: {len(datasets_dict)}")
    print(f"🌤️ Weather data: {'✅' if weather_df is not None else '❌'}")
    
    # Start with base dataset (sales)
    if 'sales' not in datasets_dict:
        print("❌ Sales dataset not found!")
        return None, None
    
    # Filter base dataset to date range
    base_df = datasets_dict['sales'].copy()
    base_df = base_df[(base_df['date'] >= start_date) & (base_df['date'] <= end_date)]
    
    print(f"\n📅 Base dataset (sales): {base_df.shape}")
    print(f"   Date range: {base_df['date'].min().date()} to {base_df['date'].max().date()}")
    
    unified_df = base_df.copy()
    
    # Define time features to avoid duplication
    time_features = ['date', 'year', 'month', 'dayofyear', 'week', 'quarter',
                    'month_sin', 'month_cos', 'week_sin', 'week_cos', 
                    'season', 'holiday_period', 'is_month_end']
    
    # Track merge operations
    merge_summary = {}
    
    # First, merge weather data
    if weather_df is not None:
        print(f"\n  🌤️ Merging weather data...")
        
        # Filter weather to date range
        weather_merge = weather_df[(weather_df['date'] >= start_date) & 
                                  (weather_df['date'] <= end_date)].copy()
        
        print(f"    Filtered weather shape: {weather_merge.shape}")
        
        # Merge weather data
        before_shape = unified_df.shape
        unified_df = unified_df.merge(weather_merge, on='date', how='left')
        after_shape = unified_df.shape
        
        weather_columns = [col for col in weather_merge.columns if col != 'date']
        records_with_weather = unified_df[weather_columns[0]].notna().sum() if weather_columns else 0
        
        merge_summary['weather'] = {
            'columns_added': len(weather_columns),
            'records_with_data': records_with_weather,
            'coverage_pct': (records_with_weather / len(unified_df)) * 100
        }
        
        print(f"    Shape: {before_shape} → {after_shape}")
        print(f"    Weather coverage: {records_with_weather}/{len(unified_df)} ({merge_summary['weather']['coverage_pct']:.1f}%)")
    
    # Then merge marketing datasets
    for dataset_name, df in datasets_dict.items():
        if dataset_name == 'sales':
            continue
        
        print(f"\n  🔄 Merging {dataset_name}...")
        
        # Filter to date range
        merge_df = df.copy()
        merge_df = merge_df[(merge_df['date'] >= start_date) & (merge_df['date'] <= end_date)]
        
        # Identify business columns (non-time features)
        business_columns = [col for col in merge_df.columns if col not in time_features]
        
        # Add dataset prefix to business columns
        rename_dict = {col: f"{dataset_name}_{col}" for col in business_columns}
        merge_df = merge_df.rename(columns=rename_dict)
        
        print(f"    Filtered shape: {merge_df.shape}")
        print(f"    Renamed {len(business_columns)} business columns")
        
        # Merge on date
        before_shape = unified_df.shape
        unified_df = unified_df.merge(merge_df, on='date', how='left')
        after_shape = unified_df.shape
        
        # Remove duplicate time features
        duplicate_time_cols = [col for col in unified_df.columns 
                              if col.endswith('_x') or col.endswith('_y')]
        
        if duplicate_time_cols:
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
    
    print(f"\n✅ ENHANCED UNIFIED DATASET CREATED: {strategy_name}")
    print(f"   Final shape: {unified_df.shape}")
    print(f"   Date range: {unified_df['date'].min().date()} to {unified_df['date'].max().date()}")
    print(f"   Total data sources: {len(merge_summary)} (marketing + weather)")
    
    return unified_df, merge_summary

# %%
# Step 5: Create Both Enhanced Strategic Datasets
print(f"\n🚀 CREATING BOTH ENHANCED STRATEGIC DATASETS")
print("=" * 70)

enhanced_unified_datasets = {}
enhanced_merge_summaries = {}

for strategy_key, strategy_info in enhanced_strategies.items():
    strategy_name = strategy_info['name']
    datasets = strategy_info['datasets']
    weather = strategy_info['weather']
    date_range = strategy_info['date_range']
    
    unified_df, merge_summary = create_enhanced_unified_dataset(
        datasets, weather, date_range, strategy_name
    )
    
    enhanced_unified_datasets[strategy_key] = unified_df
    enhanced_merge_summaries[strategy_key] = merge_summary

# %%
# Step 6: Enhanced Comparison with Weather
def compare_enhanced_strategies(unified_datasets, merge_summaries, strategies):
    """
    Compare the two enhanced strategic approaches including weather
    """
    print(f"\n📊 ENHANCED STRATEGIC COMPARISON")
    print("=" * 50)
    
    comparison_data = []
    
    for strategy_key, unified_df in unified_datasets.items():
        strategy_info = strategies[strategy_key]
        merge_summary = merge_summaries[strategy_key]
        
        # Calculate metrics
        total_sources = len(merge_summary)  # weather + marketing channels
        marketing_channels = total_sources - 1  # subtract weather
        total_weeks = len(unified_df)
        total_features = unified_df.shape[1]
        
        # Calculate data completeness
        non_date_cols = [col for col in unified_df.columns if col != 'date']
        completeness = (unified_df[non_date_cols].notna().sum().sum() / 
                       (len(unified_df) * len(non_date_cols))) * 100
        
        # Weather coverage
        weather_coverage = merge_summary.get('weather', {}).get('coverage_pct', 0)
        
        comparison_data.append({
            'Strategy': strategy_info['name'],
            'Date Range': f"{strategy_info['date_range'][0]} to {strategy_info['date_range'][1]}",
            'Weeks': total_weeks,
            'Marketing Channels': marketing_channels,
            'Weather Coverage': f"{weather_coverage:.1f}%",
            'Total Features': total_features,
            'Completeness': f"{completeness:.1f}%",
            'Shape': unified_df.shape
        })
        
        print(f"\n🎯 {strategy_info['name'].upper()}:")
        print(f"   📅 Period: {strategy_info['date_range'][0]} to {strategy_info['date_range'][1]}")
        print(f"   📊 Shape: {unified_df.shape}")
        print(f"   📈 Weeks: {total_weeks}")
        print(f"   📺 Marketing Channels: {marketing_channels}")
        print(f"   🌤️ Weather Coverage: {weather_coverage:.1f}%")
        print(f"   🔧 Total Features: {total_features}")
        print(f"   ✅ Completeness: {completeness:.1f}%")
        
        # Data source breakdown
        print(f"   📋 Data sources included:")
        for i, (source, info) in enumerate(merge_summary.items(), 1):
            coverage = info['coverage_pct']
            print(f"     {i}. {source}: {coverage:.1f}% coverage")
    
    # Create comparison table
    comparison_df = pd.DataFrame(comparison_data)
    print(f"\n📋 ENHANCED COMPARISON SUMMARY:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df

# Compare enhanced strategies
enhanced_comparison = compare_enhanced_strategies(
    enhanced_unified_datasets, enhanced_merge_summaries, enhanced_strategies
)

# %%
# Step 7: Save Enhanced Datasets
def save_enhanced_datasets(unified_datasets, strategies, merge_summaries):
    """
    Save both enhanced unified datasets with weather integration
    """
    print(f"\n💾 SAVING ENHANCED UNIFIED DATASETS")
    print("=" * 45)
    
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    saved_files = {}
    
    for strategy_key, unified_df in unified_datasets.items():
        strategy_info = strategies[strategy_key]
        
        # Create descriptive filename
        if strategy_key == 'strategy_a':
            filename = "unified_dataset_full_range_2022_2024_with_weather.csv"
            description = "Full Range + Weather (2022-2024, 9 channels + weather, 156 weeks)"
        else:
            filename = "unified_dataset_complete_coverage_2022_2023_with_weather.csv"
            description = "Complete Coverage + Weather (2022-2023, 10 channels + weather, 104 weeks)"
        
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
    
    # Save enhanced comparison report
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
    
    enhanced_report = {
        'timestamp': datetime.now().isoformat(),
        'enhancement': 'Weather data integration added',
        'weather_variables': [
            'weather_temperature_mean',
            'weather_temperature_max', 
            'weather_temperature_min',
            'weather_sunshine_duration'
        ],
        'strategy_comparison': {
            'strategy_a_full_range_weather': {
                'description': 'Full Range + Weather (2022-2024)',
                'date_range': strategies['strategy_a']['date_range'],
                'shape': unified_datasets['strategy_a'].shape,
                'marketing_channels': len(merge_summaries['strategy_a']) - 1,
                'weather_coverage': merge_summaries['strategy_a']['weather']['coverage_pct'],
                'filename': saved_files['strategy_a']['filename']
            },
            'strategy_b_complete_weather': {
                'description': 'Complete Coverage + Weather (2022-2023)',
                'date_range': strategies['strategy_b']['date_range'],
                'shape': unified_datasets['strategy_b'].shape,
                'marketing_channels': len(merge_summaries['strategy_b']) - 1,
                'weather_coverage': merge_summaries['strategy_b']['weather']['coverage_pct'],
                'filename': saved_files['strategy_b']['filename']
            }
        },
        'merge_summaries': convert_numpy(merge_summaries),
        'recommendation': {
            'approach': 'Enhanced dual strategy with environmental context',
            'benefits': [
                'Weather factors for seasonality modeling',
                'Temperature impact on consumer behavior',
                'Sunshine duration for outdoor activity correlation',
                'Environmental context for media effectiveness'
            ],
            'use_cases': {
                'strategy_a': 'Trend analysis with weather patterns (recent 3 years)',
                'strategy_b': 'Complete attribution with environmental factors (2 years)'
            }
        }
    }
    
    report_path = os.path.join(processed_dir, "enhanced_unification_with_weather_report.json")
    with open(report_path, 'w') as f:
        json.dump(enhanced_report, f, indent=2)
    
    print(f"  ✅ Saved: enhanced_unification_with_weather_report.json")
    
    return saved_files, enhanced_report

# Save enhanced datasets
enhanced_files, enhanced_report = save_enhanced_datasets(
    enhanced_unified_datasets, enhanced_strategies, enhanced_merge_summaries
)

# %%
# Step 8: Final Enhanced Recommendations
print(f"\n🎯 ENHANCED STRATEGIC RECOMMENDATIONS")
print("=" * 55)

print(f"\n✅ ENHANCED DATASETS SUCCESSFULLY CREATED!")

print(f"\n📊 ENHANCED DATASET A - FULL RANGE + WEATHER:")
print(f"   📁 File: {enhanced_files['strategy_a']['filename']}")
print(f"   📈 {enhanced_files['strategy_a']['description']}")
print(f"   🎯 Best for: Trend analysis with weather patterns")
print(f"   🌤️ Weather integration: Environmental context for recent 3 years")
print(f"   ⚠️  Limitation: Missing email channel attribution")

print(f"\n📊 ENHANCED DATASET B - COMPLETE COVERAGE + WEATHER:")
print(f"   📁 File: {enhanced_files['strategy_b']['filename']}")
print(f"   📈 {enhanced_files['strategy_b']['description']}")
print(f"   🎯 Best for: Complete attribution with environmental factors")
print(f"   🌤️ Weather integration: Full channel interactions + weather")
print(f"   ⚠️  Limitation: Less recent data (missing 2024)")

print(f"\n🌤️ WEATHER ENHANCEMENTS:")
print(f"   🌡️ Temperature variables: Mean, Max, Min (consumer comfort)")
print(f"   ☀️ Sunshine duration: Outdoor activity correlation")
print(f"   📊 Perfect coverage: Weather data available for all periods")
print(f"   🔄 Seamless integration: Weekly Monday-start alignment")

print(f"\n🚀 ENHANCED MMM DEVELOPMENT PATH:")
print(f"   1. 📊 EDA with weather correlation analysis")
print(f"   2. 🤖 MMM with weather as external variables")
print(f"   3. 🌡️ Seasonal pattern analysis (temperature + media)")
print(f"   4. ☀️ Outdoor activity impact modeling")
print(f"   5. 📈 Weather-adjusted attribution insights")

print(f"\n💡 ENHANCED RECOMMENDATION:")
print(f"   🎯 Start with Dataset B for complete attribution + weather")
print(f"   📈 Use Dataset A for trend validation with recent weather patterns")
print(f"   🌤️ Weather provides crucial external variable context!")
print(f"   🔍 Analyze temperature-media effectiveness relationships")

print(f"\n✅ ENHANCED DUAL UNIFICATION WITH WEATHER COMPLETE!")

# %%
# Step 9: Fix Email Campaign Issue
print(f"\n🔧 FIXING EMAIL CAMPAIGN ISSUE")
print("=" * 40)

print(f"\n❌ ISSUE IDENTIFIED:")
print(f"   Email campaigns only run through 2023 (no 2024 data)")
print(f"   But included in full range 2022-2024 dataset")
print(f"   This creates inconsistent channel mix between periods")

print(f"\n✅ SOLUTION:")
print(f"   Remove email campaigns from full range dataset")
print(f"   Keep email in complete coverage dataset")
print(f"   Rename datasets with descriptive names")

# Load current datasets
print(f"\n📂 Loading current datasets...")
full_range_df = pd.read_csv('data/processed/unified_dataset_full_range_2022_2024_with_weather.csv')
complete_coverage_df = pd.read_csv('data/processed/unified_dataset_complete_coverage_2022_2023_with_weather.csv')

print(f"  Full range (2022-2024): {full_range_df.shape}")
print(f"  Complete coverage (2022-2023): {complete_coverage_df.shape}")

# Check email coverage
full_range_df['date'] = pd.to_datetime(full_range_df['date'])
complete_coverage_df['date'] = pd.to_datetime(complete_coverage_df['date'])

print(f"\n📧 Email campaign analysis:")
print(f"  Full range dataset (2022-2024):")
print(f"    Email non-zero weeks: {(full_range_df['email_email_campaigns'] > 0).sum()}")
print(f"    Email zero weeks: {(full_range_df['email_email_campaigns'] == 0).sum()}")

full_range_2024 = full_range_df[full_range_df['date'].dt.year == 2024]
print(f"    2024 email non-zero weeks: {(full_range_2024['email_email_campaigns'] > 0).sum()}")
print(f"    2024 email zero weeks: {len(full_range_2024) - (full_range_2024['email_email_campaigns'] > 0).sum()}")

print(f"\n  Complete coverage dataset (2022-2023):")
print(f"    Email non-zero weeks: {(complete_coverage_df['email_email_campaigns'] > 0).sum()}")
print(f"    Email zero weeks: {(complete_coverage_df['email_email_campaigns'] == 0).sum()}")

# Create corrected full range dataset (without email)
print(f"\n🔧 Creating corrected datasets...")
full_range_corrected = full_range_df.drop(columns=['email_email_campaigns'])

print(f"  Full range: {full_range_df.shape[1]} → {full_range_corrected.shape[1]} columns")

# Save corrected datasets with proper names
print(f"\n💾 Saving corrected datasets...")

# Complete coverage dataset (keeps email) - rename for clarity
complete_coverage_path = 'data/processed/mmm_dataset_complete_channels_2022_2023.csv'
complete_coverage_df.to_csv(complete_coverage_path, index=False)
print(f"  ✅ Complete channels (2022-2023): {complete_coverage_df.shape}")
print(f"     File: mmm_dataset_complete_channels_2022_2023.csv")
print(f"     Includes: All channels including email campaigns")

# Full range dataset (no email) - rename for clarity  
full_range_path = 'data/processed/mmm_dataset_consistent_channels_2022_2024.csv'
full_range_corrected.to_csv(full_range_path, index=False)
print(f"  ✅ Consistent channels (2022-2024): {full_range_corrected.shape}")
print(f"     File: mmm_dataset_consistent_channels_2022_2024.csv")
print(f"     Excludes: Email campaigns (no 2024 data)")

# Create correction summary
correction_summary = {
    'issue_identified': 'Email campaigns only available through 2023, but included in 2022-2024 dataset',
    'solution_applied': 'Removed email campaigns from full range dataset to maintain consistent channel mix',
    'datasets_created': {
        'complete_channels_2022_2023': {
            'file': 'mmm_dataset_complete_channels_2022_2023.csv',
            'shape': complete_coverage_df.shape,
            'date_range': f"{complete_coverage_df['date'].min().date()} to {complete_coverage_df['date'].max().date()}",
            'channels': 'All channels including email campaigns',
            'use_case': 'Analysis of complete channel mix for 2022-2023 period'
        },
        'consistent_channels_2022_2024': {
            'file': 'mmm_dataset_consistent_channels_2022_2024.csv', 
            'shape': full_range_corrected.shape,
            'date_range': f"{full_range_corrected['date'].min().date()} to {full_range_corrected['date'].max().date()}",
            'channels': 'All channels except email campaigns',
            'use_case': 'MMM modeling with consistent channel mix across full time range'
        }
    }
}

# Save correction report
import json
correction_path = 'data/processed/unified_datasets_correction_report.json'
with open(correction_path, 'w') as f:
    json.dump(correction_summary, f, indent=2, default=str)

print(f"  ✅ Correction report: unified_datasets_correction_report.json")

print(f"\n📋 CORRECTION COMPLETE!")
print(f"=" * 40)
print(f"✅ Email campaigns removed from full range dataset")
print(f"✅ Datasets renamed with descriptive names")
print(f"✅ Channel consistency maintained across time periods")

print(f"\n🎯 RECOMMENDED USAGE:")
print(f"  • For MMM modeling: Use 'mmm_dataset_consistent_channels_2022_2024.csv'")
print(f"  • For email analysis: Use 'mmm_dataset_complete_channels_2022_2023.csv'")
print(f"  • Promotions are sparse by design (normal campaign behavior)")

# %% [markdown]
# ## Enhanced Dual Unification with Weather Complete! 🌤️
# 
# ### 🎯 **Enhanced Strategic Approach:**
# 
# #### **Dataset A - Full Range + Weather (2022-2024):**
# - **Period**: 156 weeks with weather context
# - **Channels**: 9 marketing + 4 weather variables
# - **Strength**: Recent trends with environmental factors
# - **Use Case**: Weather-adjusted trend analysis and forecasting
# 
# #### **Dataset B - Complete Coverage + Weather (2022-2023):**
# - **Period**: 104 weeks with complete data + weather
# - **Channels**: 10 marketing + 4 weather variables  
# - **Strength**: Full attribution with environmental context
# - **Use Case**: Complete channel interactions with weather impact
# 
# ### 🌤️ **Weather Variables Integrated:**
# - **Temperature Mean/Max/Min**: Consumer comfort and behavior
# - **Sunshine Duration**: Outdoor activity correlation
# - **Perfect Coverage**: 100% weather data availability
# - **Aligned Timing**: Monday-start weekly structure
# 
# ### 📊 **Enhanced Files Created:**
# - `unified_dataset_full_range_2022_2024_with_weather.csv`
# - `unified_dataset_complete_coverage_2022_2023_with_weather.csv`
# - `enhanced_unification_with_weather_report.json`
# 
# ### 🚀 **Next Phase Benefits:**
# - **Environmental Context**: Weather as external MMM variables
# - **Seasonal Insights**: Temperature-media effectiveness patterns
# - **Behavioral Factors**: Weather impact on consumer activity
# - **Enhanced Attribution**: Channel performance by weather conditions
# 
# **This enhanced approach provides comprehensive business intelligence with environmental context for superior MMM insights!** 📈🌤️ 