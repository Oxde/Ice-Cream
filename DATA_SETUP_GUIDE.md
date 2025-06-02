# 📊 Data Setup & Unification Guide

This guide explains how to set up, clean, and unify data for the Ice Cream MMM project.

## 🎯 Overview

The data pipeline consists of two key stages:
1. **Basic Cleaning** (`00_basic_cleaning.ipynb`)
2. **Data Unification** (`01b_data_unification_dual.ipynb`) - **CRITICAL**

## 📁 Data Directory Structure

```
data/
├── raw/                    # Original data files (NEVER modify)
│   ├── sales_data.csv
│   ├── media_spend.xlsx
│   ├── tv_metrics.csv
│   └── ...
├── interim/                # Intermediate processing files
│   ├── cleaned_sales.csv
│   ├── cleaned_media.csv
│   └── ...
├── processed/              # Final unified datasets
│   ├── unified_dataset.csv
│   ├── mmm_ready_data.csv
│   └── ...
└── explanations/           # Data documentation
    ├── column_definitions.md
    ├── data_sources.md
    └── quality_notes.md
```

## 🚀 Quick Start - Data Setup

### Step 1: Place Raw Data
```bash
# Navigate to project directory
cd /Users/nikmarf/IceCream

# Create data directories if they don't exist
mkdir -p data/{raw,interim,processed,explanations}

# Place your raw data files in data/raw/
# Example files:
# - sales_data.csv
# - media_spend.xlsx  
# - tv_grp_data.csv
# - print_impressions.csv
# - email_metrics.csv
```

### Step 2: Run Data Cleaning Pipeline
```bash
# Activate your virtual environment
source .venv/bin/activate  # On macOS

# Start Jupyter
jupyter notebook

# Run notebooks in order:
# 1. notebooks/00_basic_cleaning.ipynb
# 2. notebooks/01b_data_unification_dual.ipynb  (CRITICAL)
```

## 📋 Notebook Details

### 🧹 `00_basic_cleaning.ipynb` - Basic Cleaning

**Purpose**: Initial data cleaning and standardization

**What it does**:
- Loads raw data files
- Standardizes date formats
- Handles missing values
- Removes duplicates
- Basic data type conversions
- Saves cleaned files to `data/interim/`

**Key Functions**:
```python
# Date standardization
df['date'] = pd.to_datetime(df['date'])

# Missing value handling
df = df.dropna(subset=['critical_columns'])
df['optional_column'] = df['optional_column'].fillna(0)

# Data type conversion
df['spend'] = pd.to_numeric(df['spend'], errors='coerce')
```

**Outputs**:
- `data/interim/cleaned_*.csv` files
- Data quality report
- Summary of cleaning operations

### 🔗 `01b_data_unification_dual.ipynb` - Data Unification (CRITICAL)

**Purpose**: **THE MOST IMPORTANT STEP** - Unifies all data sources into a single dataset

**What it does**:
- Merges multiple data sources on date
- Creates unified column structure
- Handles different date granularities (daily/weekly)
- Standardizes media channel names
- Creates final MMM-ready dataset
- Generates comprehensive data validation

**Key Operations**:
```python
# Date harmonization
def standardize_dates(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col])
    df['week'] = df[date_col].dt.to_period('W')
    return df

# Multi-source merge
unified_df = sales_df.merge(media_df, on='date', how='left')
unified_df = unified_df.merge(tv_df, on='date', how='left')

# Column standardization
channel_mapping = {
    'tv_spend': 'TV',
    'print_spend': 'Print', 
    'email_spend': 'Email'
}
```

**Critical Validations**:
- Date range consistency across all sources
- No missing critical periods
- Media spend vs metrics correlation
- Data completeness report

**Outputs**:
- `data/processed/unified_dataset.csv` - **MAIN DATASET**
- `data/processed/mmm_ready_data.csv` - **FOR MODELING**
- Validation report
- Data quality dashboard

## ⚠️ Critical Data Requirements

### 📅 **Date Requirements**
- **Format**: YYYY-MM-DD or compatible
- **Granularity**: Weekly (preferred) or Daily
- **Range**: Minimum 2 years for reliable MMM
- **Completeness**: No gaps in time series

### 📊 **Sales Data Requirements**
```csv
date,sales,units,revenue
2022-01-03,125000,5000,250000
2022-01-10,130000,5200,260000
```

### 💰 **Media Spend Requirements**
```csv
date,tv_spend,print_spend,digital_spend,email_spend
2022-01-03,10000,5000,8000,1000
2022-01-10,12000,4000,9000,1200
```

### 📺 **Media Metrics (Optional)**
```csv
date,tv_grps,tv_reach,print_impressions,email_opens
2022-01-03,150,0.45,2000000,15000
2022-01-10,180,0.52,1800000,18000
```

## 🔧 Troubleshooting Common Issues

### ❌ **"Date Mismatch Error"**
**Problem**: Different date formats across files
**Solution**:
```python
# In cleaning notebook, standardize all dates
df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
```

### ❌ **"Missing Data Periods"**
**Problem**: Gaps in time series
**Solution**:
```python
# Create complete date range and fill gaps
date_range = pd.date_range(start=df['date'].min(), 
                          end=df['date'].max(), 
                          freq='W')
df = df.set_index('date').reindex(date_range).fillna(0)
```

### ❌ **"Column Name Conflicts"**
**Problem**: Same metrics with different names
**Solution**:
```python
# Standardize column names
column_mapping = {
    'TV Spend': 'tv_spend',
    'Television_Budget': 'tv_spend',
    'tv_investment': 'tv_spend'
}
df = df.rename(columns=column_mapping)
```

### ❌ **"Media Metrics vs Spend Mismatch"**
**Problem**: Zero spend but non-zero metrics
**Solution**:
```python
# Validate spend vs metrics consistency
def validate_spend_metrics(df):
    for channel in ['tv', 'print', 'digital']:
        spend_col = f'{channel}_spend'
        metrics_col = f'{channel}_grps'  # or impressions
        
        # Flag inconsistencies
        zero_spend_nonzero_metrics = (df[spend_col] == 0) & (df[metrics_col] > 0)
        if zero_spend_nonzero_metrics.any():
            print(f"Warning: {channel} has metrics without spend")
```

## 📈 Data Quality Checks

### ✅ **Automated Validations**
The unification notebook includes these automatic checks:

1. **Date Continuity**: No gaps in time series
2. **Value Ranges**: Spend and metrics within reasonable bounds
3. **Correlation Check**: Spend vs metrics correlation > 0.7
4. **Completeness**: < 5% missing values in critical columns
5. **Outlier Detection**: Values beyond 3 standard deviations

### 📊 **Quality Report Output**
```
=== DATA QUALITY REPORT ===
✅ Date Range: 2022-01-03 to 2023-12-25 (104 weeks)
✅ Missing Values: Sales (0%), TV Spend (0%), Print Spend (2%)
✅ Outliers Detected: 3 weeks with exceptional sales (flagged)
✅ Correlations: TV Spend vs GRPs (0.95), Print Spend vs Impressions (0.89)
⚠️  Warning: Email data missing for weeks 45-52 of 2022
```

## 🔄 Updating Data

### **Adding New Data**
1. Place new files in `data/raw/`
2. Update date ranges in cleaning notebook
3. Re-run both notebooks: `00_basic_cleaning.ipynb` → `01b_data_unification_dual.ipynb`
4. Validate new unified dataset
5. Re-run MMM models with updated data

### **Monthly/Weekly Updates**
```bash
# Quick update script (save as update_data.sh)
#!/bin/bash
echo "🔄 Updating MMM Data Pipeline"
echo "1. Running basic cleaning..."
jupyter nbconvert --execute notebooks/00_basic_cleaning.ipynb

echo "2. Running data unification..."
jupyter nbconvert --execute notebooks/01b_data_unification_dual.ipynb

echo "3. Data update complete! Check data/processed/ for new files"
```

## 📞 Support

### **If You Get Stuck**:
1. **Check notebook outputs**: Look for error messages in cell outputs
2. **Validate raw data**: Ensure files are in correct format
3. **Review data quality**: Check for missing dates or extreme values
4. **Consult documentation**: See `data/explanations/` folder

### **Common Files to Check**:
- `data/explanations/column_definitions.md` - What each column means
- `data/explanations/data_sources.md` - Where data comes from
- `notebooks/01b_data_unification_dual.ipynb` - Main unification logic

---

**Remember**: The `01b_data_unification_dual.ipynb` notebook is the **CRITICAL** step that makes all subsequent analysis possible. Always run this after any data changes! 