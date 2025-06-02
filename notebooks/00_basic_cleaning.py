# %% [markdown]
# # Basic Data Cleaning
# 
# **Goal**: Clean individual datasets and save them separately for inspection.
# 
# **What we do:**
# 1. Load raw Excel files
# 2. Basic cleaning (dates, remove empty columns, handle missing values)
# 3. **Filter to 2022+ only** (as discussed with client)
# 4. **FIXED: Proper date parsing** for search data (DD-MM-YYYY string format)
# 5. Special processing for promo data (pivot and classification)
# 6. Save each dataset as clean CSV
# 7. **NO UNIFICATION YET** - we inspect each dataset first
# 
# **What we DON'T do:**
# - Merge datasets
# - Complex transformations beyond promo processing
# - Weekly aggregation (yet)

# %%
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

print("🧹 Basic Data Cleaning - Individual Datasets")
print("=" * 50)
print("🚨 FIXED: Search data date parsing (DD-MM-YYYY string format)")

# %%
# Step 1: Load all raw Excel files
raw_dir = '../data/raw'
excel_files = glob.glob(os.path.join(raw_dir, '*.xlsx'))

print(f"Found {len(excel_files)} Excel files:")
for file in excel_files:
    print(f"  📁 {os.path.basename(file)}")

# %%
# Step 2: FIXED Basic cleaning function with proper date handling
def basic_clean_dataset(file_path):
    """
    Basic cleaning for individual dataset:
    - Load Excel file
    - Standardize date column names
    - FIXED: Convert dates to datetime (special handling for search data)
    - Filter to 2022+ (as discussed with client)
    - Remove completely empty columns
    - Basic info about the dataset
    """
    filename = os.path.basename(file_path).replace('.xlsx', '')
    print(f"\n🔧 Cleaning: {filename}")
    print("-" * 30)
    
    # Load data
    df = pd.read_excel(file_path)
    print(f"Original shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Standardize date column names
    date_mapping = {
        'datum': 'date',
        'dag': 'date'
    }
    
    for old_col, new_col in date_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
            print(f"  ✅ Renamed '{old_col}' → '{new_col}'")
    
    # FIXED: Convert date column to datetime with proper format handling
    if 'date' in df.columns:
        original_date_range = f"{df['date'].min()} to {df['date'].max()}"
        print(f"  📅 Original date range: {original_date_range}")
        
        # CRITICAL FIX: Only search data needs DD-MM-YYYY parsing
        if filename == 'search' and isinstance(df['date'].iloc[0], str):
            print(f"  🚨 SEARCH DATA: Applying DD-MM-YYYY parsing")
            # Parse as DD-MM-YYYY format (search data specific)
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        else:
            print(f"  ✅ Standard datetime parsing")
            # Standard datetime parsing for all other datasets
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Filter to 2022+ (as discussed with client)
        df = df[df['date'].dt.year >= 2022]
        filtered_date_range = f"{df['date'].min()} to {df['date'].max()}"
        print(f"  📅 Filtered to 2022+: {filtered_date_range}")
        print(f"  📊 Records after 2022 filter: {len(df)}")
        
        # Check for invalid dates
        invalid_dates = df['date'].isna().sum()
        if invalid_dates > 0:
            print(f"  ⚠️  {invalid_dates} invalid dates found")
    
    # Remove completely empty columns
    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        df = df.drop(columns=empty_cols)
        print(f"  🗑️  Removed empty columns: {empty_cols}")
    
    # Remove unnamed columns (Excel artifacts)
    unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        print(f"  🗑️  Removed unnamed columns: {unnamed_cols}")
    
    # Basic statistics
    print(f"  📊 Final shape: {df.shape}")
    print(f"  📊 Missing values: {df.isna().sum().sum()}")
    print(f"  📊 Duplicate rows: {df.duplicated().sum()}")
    
    return df, filename

# %%
# Step 3: Special promo processing function
def process_promo_data(df):
    """
    Special processing for promo data:
    - Pivot promo types into columns
    - Create promotion classification (Buy One Get One=1, Limited Time Offer=2, Price Discount=3, No Promotion=0)
    - Return processed promo dataset
    """
    print(f"\n🎯 Special Promo Processing")
    print("-" * 30)
    
    # Check if this is promo data
    if 'promotion_type' not in df.columns:
        print(f"  ℹ️  Not promo data - skipping special processing")
        return df
    
    print(f"  📊 Original promo shape: {df.shape}")
    print(f"  📊 Promotion types: {df['promotion_type'].unique()}")
    
    # Create separate flags by pivoting
    promo_counts = df.pivot_table(
        index="date", 
        columns="promotion_type", 
        aggfunc="size", 
        fill_value=0
    ).reset_index()
    
    # Clean column names
    promo_counts.columns.name = None
    
    # Rename columns to be more code-friendly
    column_mapping = {
        "Buy One Get One": "buy_one_get_one",
        "Limited Time Offer": "limited_time_offer", 
        "Price Discount": "price_discount"
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in promo_counts.columns:
            promo_counts = promo_counts.rename(columns={old_name: new_name})
    
    print(f"  📊 Pivoted columns: {list(promo_counts.columns)}")
    
    # Assign a single promotion_type value based on priority
    def classify_promo(row):
        """
        Classification priority:
        1 = Buy One Get One (highest priority)
        2 = Limited Time Offer
        3 = Price Discount
        0 = No Promotion
        """
        # Only check numeric columns (exclude 'date')
        numeric_cols = [col for col in row.index if col != 'date']
        
        if row.get("buy_one_get_one", 0) > 0:
            return 1
        elif row.get("limited_time_offer", 0) > 0:
            return 2
        elif row.get("price_discount", 0) > 0:
            return 3
        else:
            return 0
    
    promo_counts["promotion_type"] = promo_counts.apply(classify_promo, axis=1)
    
    # Keep only date and final promotion_type
    promo_final = promo_counts[["date", "promotion_type"]].copy()
    
    print(f"  📊 Final promo shape: {promo_final.shape}")
    print(f"  📊 Promotion distribution:")
    promo_dist = promo_final['promotion_type'].value_counts().sort_index()
    for promo_code, count in promo_dist.items():
        promo_names = {0: "No Promotion", 1: "Buy One Get One", 2: "Limited Time Offer", 3: "Price Discount"}
        print(f"    {promo_code} ({promo_names.get(promo_code, 'Unknown')}): {count}")
    
    return promo_final

# %%
# Step 4: Clean each dataset and save
cleaned_dir = '../data/interim'
os.makedirs(cleaned_dir, exist_ok=True)

cleaned_datasets = {}

for file_path in excel_files:
    try:
        df_clean, name = basic_clean_dataset(file_path)
        
        # Special processing for promo data
        if name == 'promo':
            df_clean = process_promo_data(df_clean)
        
        # Save cleaned dataset
        output_path = os.path.join(cleaned_dir, f"{name}_basic_clean.csv")
        df_clean.to_csv(output_path, index=False)
        
        cleaned_datasets[name] = df_clean
        print(f"  💾 Saved: {output_path}")
        
    except Exception as e:
        print(f"  ❌ Error cleaning {file_path}: {e}")

# %%
# Step 5: Summary of all cleaned datasets
print(f"\n📋 SUMMARY - Cleaned Datasets (2022+ Only)")
print("=" * 50)

for name, df in cleaned_datasets.items():
    print(f"\n📊 {name.upper()}:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    if 'date' in df.columns:
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Total records: {len(df.dropna(subset=['date']))}")
    
    missing = df.isna().sum().sum()
    if missing > 0:
        print(f"   ⚠️  Missing values: {missing}")
    else:
        print(f"   ✅ No missing values")

print(f"\n✅ Basic cleaning complete!")
print(f"📁 Cleaned files saved in: {cleaned_dir}")
print(f"📅 All datasets filtered to 2022+ (as discussed with client)")
print(f"🎯 Promo data specially processed with promotion classification")
print(f"📓 Next: Run 01_data_inspection.py to analyze each dataset")

# %% [markdown]
# ## What's Next?
# 
# **Files created:**
# - Each dataset saved as `{name}_basic_clean.csv` in `data/interim/`
# - **All filtered to 2022+** (as discussed with client)
# - **Promo data specially processed** with promotion type classification
# - Ready for individual inspection
# 
# **Promo Processing Details:**
# - Pivoted promotion types into separate columns
# - Applied priority-based classification:
#   - 1 = Buy One Get One (highest priority)
#   - 2 = Limited Time Offer  
#   - 3 = Price Discount
#   - 0 = No Promotion
# 
# **Next steps:**
# 1. **Inspect each dataset individually** in next script
# 2. **Identify data quality issues** (gaps, outliers, inconsistencies)
# 3. **Make decisions** about how to handle each issue
# 4. **Then** proceed with unification
# 
# **No unification yet** - we want to understand each dataset first! 