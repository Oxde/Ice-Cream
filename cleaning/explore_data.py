import pandas as pd
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Path to the data folder
data_dir = '../data_csv'

def explore_dataset(file_path):
    """
    Load and explore a dataset, printing out key information:
    - Shape (rows, columns)
    - Column names and types
    - Date range if applicable
    - Summary statistics
    - Missing values
    - Sample rows
    """
    # Extract file name without extension
    file_name = os.path.basename(file_path).split('.')[0]
    print(f"\n{'='*80}")
    print(f"Exploring dataset: {file_name}")
    print(f"{'='*80}")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Basic information
    print(f"\n1. Basic Information:")
    print(f"   - Rows: {df.shape[0]}")
    print(f"   - Columns: {df.shape[1]}")
    print(f"   - Column names: {list(df.columns)}")
    
    # Data types
    print(f"\n2. Data Types:")
    for col in df.columns:
        print(f"   - {col}: {df[col].dtype}")
    
    # Date range if applicable
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'dag' in col.lower() or 'datum' in col.lower()]
    if date_cols:
        print(f"\n3. Date Range:")
        for date_col in date_cols:
            # Convert to datetime
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Print date range
            if not df[date_col].isna().all():
                min_date = df[date_col].min()
                max_date = df[date_col].max()
                print(f"   - {date_col}: {min_date} to {max_date}")
                print(f"   - Total weeks: {(max_date - min_date).days // 7 if min_date and max_date else 'N/A'}")
    
    # Check for missing values
    print(f"\n4. Missing Values:")
    missing = df.isna().sum()
    for col, count in missing.items():
        if count > 0:
            print(f"   - {col}: {count} missing values ({count/len(df)*100:.2f}%)")
    
    if missing.sum() == 0:
        print("   - No missing values found")
    
    # Summary statistics for numeric columns
    print(f"\n5. Summary Statistics (Numeric Columns):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe().to_string())
    else:
        print("   - No numeric columns found")
    
    # Sample rows
    print(f"\n6. Sample Rows:")
    print(df.head(5).to_string())
    
    # Special handling for promo_wide (different structure)
    if file_name == 'promo_wide':
        print(f"\n7. Special Analysis for promo_wide:")
        print(f"   This dataset has a special structure with years in the second row and promotion types in the fourth row.")
        if df.shape[0] >= 4:
            print(f"   Years: {df.iloc[0].dropna().to_dict()}")
            print(f"   Promotion Types (sample): {list(set(df.iloc[3].dropna().tolist()))}")
    
    return df

# List all CSV files in the data_csv directory
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
print(f"Found {len(csv_files)} CSV files in {data_dir}:")
for file in csv_files:
    print(f" - {file}")

# Create output directory for plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Explore each dataset
datasets = {}
for file in csv_files:
    df = explore_dataset(os.path.join(data_dir, file))
    datasets[file.split('.')[0]] = df

# Additional analysis: check overlapping date ranges
print("\n\nDate Range Analysis for All Datasets:")
print("="*80)

all_date_ranges = {}
for name, df in datasets.items():
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'dag' in col.lower() or 'datum' in col.lower()]
    
    for date_col in date_cols:
        if date_col in df.columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Get min and max dates
            if not df[date_col].isna().all():
                min_date = df[date_col].min()
                max_date = df[date_col].max()
                all_date_ranges[f"{name} ({date_col})"] = (min_date, max_date)

# Print all date ranges
print("\nDate ranges for all datasets:")
for dataset, (min_date, max_date) in all_date_ranges.items():
    print(f" - {dataset}: {min_date} to {max_date}")

# Find common date range
if all_date_ranges:
    max_min_date = max([range_[0] for range_ in all_date_ranges.values()])
    min_max_date = min([range_[1] for range_ in all_date_ranges.values()])
    
    print(f"\nOverlapping date range across all datasets:")
    if max_min_date <= min_max_date:
        print(f" - From {max_min_date} to {min_max_date}")
        print(f" - Total weeks: {(min_max_date - max_min_date).days // 7}")
    else:
        print(" - No overlapping date range found across all datasets")

# Find 2022 onwards date range
print(f"\nDate range from 2022-01-01 onwards:")
year_2022 = pd.Timestamp('2022-01-01')
datasets_with_2022_data = {dataset: (min_date, max_date) for dataset, (min_date, max_date) in all_date_ranges.items() 
                          if max_date >= year_2022}

if datasets_with_2022_data:
    for dataset, (min_date, max_date) in datasets_with_2022_data.items():
        effective_start = max(min_date, year_2022)
        print(f" - {dataset}: {effective_start} to {max_date}")
    
    # Find common date range from 2022
    max_min_date_2022 = max([max(range_[0], year_2022) for range_ in datasets_with_2022_data.values()])
    min_max_date_2022 = min([range_[1] for range_ in datasets_with_2022_data.values()])
    
    print(f"\nCommon date range from 2022:")
    if max_min_date_2022 <= min_max_date_2022:
        print(f" - From {max_min_date_2022} to {min_max_date_2022}")
        print(f" - Total weeks: {(min_max_date_2022 - max_min_date_2022).days // 7}")
    else:
        print(" - No common date range found from 2022 onwards")
else:
    print(" - No datasets with data from 2022 onwards")

print("\nExploration complete. Check the output above for data insights.") 