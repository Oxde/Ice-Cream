import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

def convert_excel_to_csv(raw_dir='data/raw', interim_dir='data/interim'):
    """
    Converts all .xlsx files from raw directory to .csv format in interim directory.
    Skips promo files as requested.
    
    Parameters:
    raw_dir: Path to raw Excel files
    interim_dir: Path to save CSV files
    
    Returns:
    List of converted files
    """
    # Create interim directory if it doesn't exist
    os.makedirs(interim_dir, exist_ok=True)
    
    # Find Excel files, excluding promo files
    excel_files = glob.glob(os.path.join(raw_dir, '*.xlsx'))
    excel_files = [f for f in excel_files if 'promo' not in os.path.basename(f).lower()]
    
    if not excel_files:
        print(f"No Excel files found in {raw_dir} (excluding promo files)")
        return []
    
    print(f"Found {len(excel_files)} Excel files to convert (excluding promo files)")
    converted_files = []
    
    for file_path in excel_files:
        try:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            df = pd.read_excel(file_path)
            
            csv_path = os.path.join(interim_dir, f"{base_name}.csv")
            df.to_csv(csv_path, index=False)
            
            print(f"Converted: {base_name}.xlsx → {base_name}.csv")
            converted_files.append(csv_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return converted_files

def explore_dataset(file_path):
    """
    Load and explore a dataset, returning key information.
    
    Parameters:
    file_path: Path to the CSV file
    
    Returns:
    Dictionary with dataset information
    """
    file_name = os.path.basename(file_path).split('.')[0]
    df = pd.read_csv(file_path)
    
    # Find date columns
    date_cols = [col for col in df.columns if any(term in col.lower() for term in ['date', 'dag', 'datum'])]
    
    # Convert date columns and get range
    date_info = {}
    for date_col in date_cols:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if not df[date_col].isna().all():
            date_info[date_col] = {
                'min_date': df[date_col].min(),
                'max_date': df[date_col].max(),
                'total_weeks': (df[date_col].max() - df[date_col].min()).days // 7
            }
    
    # Missing values
    missing_values = df.isna().sum()
    missing_info = {col: count for col, count in missing_values.items() if count > 0}
    
    # Basic stats for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    return {
        'name': file_name,
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'date_info': date_info,
        'missing_values': missing_info,
        'numeric_columns': list(numeric_cols),
        'sample_data': df.head(3).to_dict('records')
    }

def clean_dataset(file_path, start_date='2022-01-01'):
    """
    Clean a single dataset with consistent processing.
    
    Parameters:
    file_path: Path to the CSV file
    start_date: Filter data from this date onwards
    
    Returns:
    Cleaned DataFrame
    """
    file_name = os.path.basename(file_path).split('.')[0]
    df = pd.read_csv(file_path)
    
    # Handle different date column names
    date_col_mapping = {
        'tv_promo': 'datum',
        'tv_branding': 'datum', 
        'radio_local': 'dag',
        'radio_national': 'dag'
    }
    
    date_col = date_col_mapping.get(file_name, 'date')
    
    if date_col not in df.columns:
        print(f"Warning: Expected date column '{date_col}' not found in {file_name}")
        return None
    
    # Convert date column
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Filter from start date
    df = df[df[date_col] >= start_date]
    
    # Sort by date
    df = df.sort_values(by=date_col)
    
    # Handle TV branding - remove unnamed columns
    if file_name == 'tv_branding':
        df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col])
    
    # Remove rows with NaN dates
    if df[date_col].isna().any():
        print(f"Removing {df[date_col].isna().sum()} rows with NaN dates from {file_name}")
        df = df.dropna(subset=[date_col])
    
    # Fill missing values in other columns
    for col in df.columns:
        if col != date_col:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(method='ffill')
    
    return df

def create_unified_dataset(interim_dir='data/interim', processed_dir='data/processed'):
    """
    Create unified dataset from cleaned individual files.
    
    Parameters:
    interim_dir: Directory with CSV files
    processed_dir: Directory to save processed files
    
    Returns:
    Unified DataFrame
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    # Get CSV files (excluding promo files)
    csv_files = glob.glob(os.path.join(interim_dir, '*.csv'))
    csv_files = [f for f in csv_files if 'promo' not in os.path.basename(f).lower()]
    
    # Clean each dataset and save individual cleaned files
    datasets = {}
    for file_path in csv_files:
        file_name = os.path.basename(file_path).split('.')[0]
        cleaned_df = clean_dataset(file_path)
        
        if cleaned_df is not None:
            # Save individual cleaned file
            cleaned_path = os.path.join(processed_dir, f"{file_name}_cleaned.csv")
            cleaned_df.to_csv(cleaned_path, index=False)
            datasets[file_name] = cleaned_df
            print(f"Cleaned and saved: {file_name}")
    
    # Create unified dataset starting with sales
    if 'sales' not in datasets:
        print("Error: Sales dataset not found")
        return None
    
    unified_df = datasets['sales'].copy()
    date_col = 'date'
    
    # Merge other datasets
    for name, df in datasets.items():
        if name == 'sales':
            continue
            
        # Determine date column for merging
        if name in ['tv_promo', 'tv_branding']:
            merge_date_col = 'datum'
        elif name in ['radio_local', 'radio_national']:
            merge_date_col = 'dag'
        else:
            merge_date_col = 'date'
        
        # Rename columns to avoid conflicts
        rename_dict = {}
        for col in df.columns:
            if col != merge_date_col:
                rename_dict[col] = f"{name}_{col}"
        
        df_renamed = df.rename(columns=rename_dict)
        
        # Merge
        unified_df = pd.merge(unified_df, df_renamed, 
                             left_on=date_col, right_on=merge_date_col, 
                             how='left')
        
        # Remove duplicate date column
        if merge_date_col != date_col and merge_date_col in unified_df.columns:
            unified_df = unified_df.drop(columns=[merge_date_col])
    
    # Fill remaining NaN values
    unified_df = unified_df.fillna(0)
    
    # Save unified dataset
    unified_path = os.path.join(processed_dir, 'unified_dataset.csv')
    unified_df.to_csv(unified_path, index=False)
    
    print(f"Unified dataset created: {unified_df.shape[0]} rows, {unified_df.shape[1]} columns")
    print(f"Saved to: {unified_path}")
    
    return unified_df
