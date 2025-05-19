import pandas as pd
import numpy as np
import os
from datetime import datetime

# Path to the data folder
data_dir = '../data_csv'
output_dir = 'processed'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define start date for filtering (2022-01-01)
START_DATE = '2022-01-01'

def clean_dataset(file_path, date_col='date'):
    """
    Load and clean a dataset:
    - Ensure consistent date format
    - Filter data from 2022 onwards
    - Handle missing values
    - Save cleaned data
    
    Parameters:
    file_path: Path to the CSV file
    date_col: Name of the date column
    
    Returns:
    Cleaned DataFrame
    """
    # Extract file name without extension
    file_name = os.path.basename(file_path).split('.')[0]
    print(f"Processing {file_name}...")
    
    # Skip promo_wide for now
    if file_name == 'promo_wide':
        print(f"Skipping {file_name} dataset for now")
        return None
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Special handling for search dataset (has many missing dates)
    if file_name == 'search':
        return process_search(df, date_col)
    
    # Check if the dataset has a date column
    if date_col not in df.columns:
        print(f"Error: Date column '{date_col}' not found in {file_name}")
        return None
    
    # Convert date column to datetime format
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Filter data from 2022 onwards
    df = df[df[date_col] >= START_DATE]
    
    # Sort by date
    df = df.sort_values(by=date_col)
    
    # Handle missing values based on dataset
    if file_name == 'tv_branding':
        # Drop unnecessary columns (Unnamed columns)
        for col in df.columns:
            if 'Unnamed' in col:
                df = df.drop(columns=[col])
    
    # Check for and handle NaN values in date column
    if df[date_col].isna().any():
        print(f"Warning: NaN dates found in {file_name}. Removing rows with NaN dates.")
        df = df.dropna(subset=[date_col])
    
    # For datasets with missing values in non-date columns, fill with appropriate values
    # (0 for numeric columns, method-specific for others)
    for col in df.columns:
        if col != date_col:
            if df[col].dtype in [np.float64, np.int64]:
                # Fill numeric missing values with 0
                df[col] = df[col].fillna(0)
            else:
                # For non-numeric columns, forward fill
                df[col] = df[col].fillna(method='ffill')
    
    # Save cleaned data
    df.to_csv(f"{output_dir}/{file_name}_cleaned.csv", index=False)
    
    return df

def process_search(df, date_col='date'):
    """
    Special processing for search dataset which has many missing dates
    """
    print("Processing search dataset with special handling...")
    
    # Convert date to datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Create a complete date range
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    
    if pd.isna(min_date) or pd.isna(max_date):
        # Find the first and last valid dates
        valid_dates = df[date_col].dropna()
        if len(valid_dates) == 0:
            print("Error: No valid dates found in search dataset")
            return None
        min_date = valid_dates.min()
        max_date = valid_dates.max()
    
    # Filter for dates from 2022 onwards
    if min_date < pd.Timestamp(START_DATE):
        min_date = pd.Timestamp(START_DATE)
    
    # Create a complete date range at weekly intervals
    complete_dates = pd.date_range(start=min_date, end=max_date, freq='W-MON')
    date_df = pd.DataFrame({date_col: complete_dates})
    
    # Merge with the original data
    merged_df = pd.merge(date_df, df, on=date_col, how='left')
    
    # Fill missing values
    for col in merged_df.columns:
        if col != date_col:
            if merged_df[col].dtype in [np.float64, np.int64]:
                # Interpolate numeric missing values
                merged_df[col] = merged_df[col].interpolate(method='linear').fillna(merged_df[col].mean())
            else:
                # For non-numeric columns, forward fill
                merged_df[col] = merged_df[col].fillna(method='ffill')
    
    # Save cleaned data
    merged_df.to_csv(f"{output_dir}/search_cleaned.csv", index=False)
    
    return merged_df

def create_unified_dataset(datasets):
    """
    Create a unified dataset by merging all cleaned datasets.
    
    Parameters:
    datasets: Dictionary of {dataset_name: dataframe}
    
    Returns:
    Unified DataFrame
    """
    print("\nCreating unified dataset...")
    
    # Get the sales dataset as the base for dates
    if 'sales' in datasets:
        unified_df = datasets['sales'].rename(columns={'sales': 'sales'})
        date_col = 'date'
        
        # Merge all other datasets
        for name, df in datasets.items():
            if name != 'sales' and df is not None:  # Skip None datasets
                # Identify the date column
                if name in ['tv_promo', 'tv_branding']:
                    df_date_col = 'datum'
                elif name in ['radio_local', 'radio_national']:
                    df_date_col = 'dag'
                else:
                    df_date_col = 'date'
                
                # Rename columns to avoid duplicates
                rename_dict = {}
                for col in df.columns:
                    if col != df_date_col and col != 'date':
                        rename_dict[col] = f"{name}_{col}"
                
                df = df.rename(columns=rename_dict)
                
                # Merge with the unified dataset
                if df_date_col in df.columns:
                    merge_col = df_date_col
                else:
                    merge_col = 'date'
                
                unified_df = pd.merge(unified_df, 
                                     df, 
                                     left_on=date_col, 
                                     right_on=merge_col, 
                                     how='left')
                
                # Remove the duplicate date column if needed
                if merge_col != date_col and merge_col in unified_df.columns:
                    unified_df = unified_df.drop(columns=[merge_col])
        
        # Fill any remaining NaN values with zeros
        unified_df = unified_df.fillna(0)
        
        # Sort by date
        unified_df = unified_df.sort_values(by=date_col)
        
        return unified_df
    else:
        print("Error: Sales dataset not found, cannot create unified dataset")
        return None

def add_time_features(df, date_col='date'):
    """
    Add time-based features for seasonality:
    - Month (as number and one-hot encoded)
    - Week of year
    - Quarter
    - Year
    - Days to holidays (optional)
    
    Parameters:
    df: DataFrame with date column
    date_col: Name of the date column
    
    Returns:
    DataFrame with additional time features
    """
    print("Adding time-based features...")
    
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Basic time features
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['year'] = df[date_col].dt.year
    
    # One-hot encode months for seasonality
    for month in range(1, 13):
        df[f'month_{month}'] = (df['month'] == month).astype(int)
    
    # One-hot encode quarters
    for quarter in range(1, 5):
        df[f'quarter_{quarter}'] = (df['quarter'] == quarter).astype(int)
    
    return df

def add_marketing_features(df):
    """
    Add marketing-specific features:
    - Lagged variables for marketing channels
    - Adstock transformations
    - Interaction terms
    
    Parameters:
    df: DataFrame with marketing variables
    
    Returns:
    DataFrame with additional marketing features
    """
    print("Adding marketing-specific features...")
    
    # Identify marketing spend columns
    spend_cols = [col for col in df.columns if any(term in col for term in ['cost', 'spend', 'grps'])]
    
    # Create lagged variables (1-4 weeks)
    for col in spend_cols:
        for lag in range(1, 5):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    # Simple adstock transformation (decay rate of 0.7)
    decay_rate = 0.7
    for col in spend_cols:
        # Initialize the adstock column
        adstock_col = f"{col}_adstock"
        df[adstock_col] = 0
        
        # Calculate adstock (exponentially weighted sum of current and past values)
        for i in range(1, len(df)):
            df.loc[df.index[i], adstock_col] = df.loc[df.index[i], col] + decay_rate * df.loc[df.index[i-1], adstock_col]
    
    # Fill NaN values created by lagging
    df = df.fillna(0)
    
    return df

def main():
    """
    Main function to clean and prepare the data for MMM
    """
    print("Starting data cleaning and preparation process...")
    
    # Handle different date column names
    date_columns = {
        'email.csv': 'date',
        'sales.csv': 'date',
        'tv_promo.csv': 'datum',
        'tv_branding.csv': 'datum',
        'radio_local.csv': 'dag',
        'social.csv': 'date',
        'radio_national.csv': 'dag',
        'promo_wide.csv': 'Week',  # Will be skipped
        'search.csv': 'date',
        'ooh.csv': 'date'
    }
    
    # Process all CSV files in the data_csv directory
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found. Please run the conversion script first or check the path.")
        return
        
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {data_dir}. Ensure that '01_convert_excel_to_csv.py' has been run successfully.")
        return
    print(f"Found {len(csv_files)} CSV files in {data_dir}")
    
    # Clean each dataset
    datasets = {}
    for file in csv_files:
        date_col = date_columns.get(file, 'date')  # Default to 'date' if not specified
        df = clean_dataset(os.path.join(data_dir, file), date_col=date_col)
        if df is not None:
            datasets[file.split('.')[0]] = df
    
    # Create a unified dataset
    unified_df_base = create_unified_dataset(datasets)
    
    if unified_df_base is not None:
        # Save the base unified dataset
        unified_df_base.to_csv(f"{output_dir}/unified_dataset_base.csv", index=False)
        print(f"Base unified dataset created with {unified_df_base.shape[0]} rows and {unified_df_base.shape[1]} columns.")
        print(f"Saved to {os.path.abspath(output_dir)}/unified_dataset_base.csv")
    else:
        print("Failed to create the unified base dataset.")
            
    print("\nData preparation complete. Base unified file saved in the 'processed' directory.")

if __name__ == "__main__":
    main() 