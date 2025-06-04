# 📊 Data Setup & Unification Guide

This guide explains how to set up, clean, and unify data for the Ice Cream MMM project using the automated shell script pipeline.

## 🎯 Overview

The data pipeline consists of three automated stages:
1. **Basic Cleaning** (`00_basic_cleaning.py`)
2. **Data Preprocessing** (`01_data_preprocessing.py`) 
3. **Dual Data Unification** (`01b_data_unification_dual.py`) - **CRITICAL**

All steps are automated via the `setup_data.sh` script! 🚀

## 📁 Data Directory Structure

```
data/
├── raw/                    # Original data files (NEVER modify)
│   ├── sales.xlsx
│   ├── search.xlsx  
│   ├── tv_promo.xlsx
│   ├── tv_branding.xlsx
│   ├── radio_national.xlsx
│   ├── radio_local.xlsx
│   ├── social.xlsx
│   ├── ooh.xlsx
│   ├── email.xlsx
│   └── promo.xlsx
├── interim/                # Cleaned individual datasets
│   ├── sales_basic_clean.csv
│   ├── search_basic_clean.csv
│   └── ...
├── processed/              # Final unified datasets & preprocessed files
│   ├── unified_dataset_full_range_2022_2024.csv      # Strategy A
│   ├── unified_dataset_complete_coverage_2022_2023.csv # Strategy B
│   ├── dual_unification_report.json
│   ├── *_preprocessed.csv  # Individual preprocessed datasets
│   └── preprocessing_report.json
└── explanations/           # Data documentation
    ├── column_definitions.md
    ├── data_sources.md
    └── quality_notes.md
```

## 🚀 Quick Start - Automated Data Setup

### Step 1: Virtual Environment Setup

```bash
# Navigate to project directory
cd /Users/nikmarf/IceCream

# Create virtual environment (if not exists)
python -m venv .venv

# Activate virtual environment (macOS/Linux)
source .venv/bin/activate

# Install required packages
pip install pandas numpy matplotlib seaborn scipy openpyxl jupyter

# Verify installation
python -c "import pandas as pd; print('✅ Dependencies installed successfully!')"
```

### Step 2: Run Automated Data Pipeline

```bash
# Ensure you're in the project root with virtual environment activated
source .venv/bin/activate

# Make script executable (if needed)
chmod +x setup_data.sh

# Run the complete data pipeline
./setup_data.sh
```

**That's it!** 🎉 The script will automatically:
1. ✅ Clean all raw datasets (10 files)
2. ✅ Apply preprocessing and feature engineering
3. ✅ Create TWO strategic unified datasets
4. ✅ Generate validation reports
5. ✅ Show data summary

## 📋 Pipeline Details

### 🧹 Stage 1: Basic Cleaning (`00_basic_cleaning.py`)

**Purpose**: Initial data cleaning and standardization

**What it does**:
- Loads all raw Excel files from `data/raw/`
- Standardizes date formats (handles DD-MM-YYYY for search data)
- Filters to 2022+ data only
- Handles missing values and duplicates
- Special promo data processing (pivot and classification)
- Saves cleaned files to `data/interim/`

**Key Features**:
```python
# Date standardization with special handling
if filename == 'search':
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
else:
    df['date'] = pd.to_datetime(df['date'])

# Promo classification (priority-based)
# 1 = Buy One Get One, 2 = Limited Time Offer, 3 = Price Discount, 0 = No Promotion
```

### 🔧 Stage 2: Data Preprocessing (`01_data_preprocessing.py`)

**Purpose**: Feature engineering and data preparation for modeling

**What it does**:
- Loads cleaned datasets from `data/interim/`
- Minimal outlier treatment (conservative approach)
- **Time feature engineering optimized for weekly data**:
  - Basic: year, month, dayofyear, week, quarter
  - **Cyclical features**: month_sin/cos, week_sin/cos  
  - **Business features**: season, holiday_period, is_month_end
- Data validation and quality checks
- Saves preprocessed files to `data/processed/`

**Key Features**:
```python
# Cyclical encoding for seasonality (CRITICAL for MMM)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Weekly patterns for business cycles  
df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
```

### 🔗 Stage 3: Dual Data Unification (`01b_data_unification_dual.py`) - CRITICAL

**Purpose**: **THE MOST IMPORTANT STEP** - Creates two strategic unified datasets

**Dual Strategy Approach**:

#### 📊 **Strategy A - Full Range (2022-2024)**
- **File**: `unified_dataset_full_range_2022_2024.csv`
- **Period**: 156 weeks (3 years)
- **Channels**: 9 (excludes email - missing 2024)  
- **Best for**: Trend analysis, recent performance, forecasting

#### 📊 **Strategy B - Complete Coverage (2022-2023)**
- **File**: `unified_dataset_complete_coverage_2022_2023.csv`
- **Period**: 104 weeks (2 years)
- **Channels**: 10 (includes ALL channels including email)
- **Best for**: Full attribution, channel interactions, ROI optimization

**What it does**:
- Merges all preprocessed datasets on date
- Handles different date coverage strategically
- Creates unified column structure with dataset prefixes
- Removes duplicate time features
- Generates comprehensive validation reports
- **Dual unification report** for strategy comparison

**Key Operations**:
```python
# Strategic date filtering
FULL_RANGE = '2022-01-03' to '2024-12-23'      # Strategy A
COMPLETE_RANGE = '2022-01-03' to '2023-12-25'  # Strategy B

# Smart column renaming to avoid conflicts
rename_dict = {col: f"{dataset_name}_{col}" for col in business_columns}

# Merge with coverage tracking
merge_summary[dataset_name] = {
    'columns_added': new_columns,
    'records_with_data': records_with_data,
    'coverage_pct': (records_with_data / len(unified_df)) * 100
}
```

## ⚠️ Critical Data Requirements

### 📅 **Date Requirements**
- **Format**: YYYY-MM-DD or DD-MM-YYYY (search data)
- **Granularity**: Weekly (Mondays)
- **Range**: 2022+ (automatically filtered)
- **Completeness**: Handled by dual strategy approach

### 📊 **Expected Raw Data Files** (in `data/raw/`)
```
sales.xlsx          # Weekly sales data
search.xlsx         # Search impressions & cost  
tv_promo.xlsx       # TV promotional GRPs & cost
tv_branding.xlsx    # TV branding GRPs & cost
radio_national.xlsx # National radio GRPs & cost
radio_local.xlsx    # Local radio GRPs & cost
social.xlsx         # Social impressions & cost
ooh.xlsx           # Out-of-home spend
email.xlsx         # Email campaigns (2022-2023 only)
promo.xlsx         # Promotion types & dates
```

## 🔧 Troubleshooting

### ❌ **"Permission Denied" Error**
```bash
chmod +x setup_data.sh
./setup_data.sh
```

### ❌ **"Python not found" or Import Errors**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Install missing packages
pip install pandas numpy matplotlib seaborn scipy openpyxl
```

### ❌ **"print_status: command not found"**
This is a minor shell script issue but doesn't affect functionality. The data processing will complete successfully.

### ❌ **Missing Raw Data Files**
Ensure all 10 Excel files are in `data/raw/`:
```bash
ls -la data/raw/*.xlsx
# Should show 10 files
```

### ❌ **Data Quality Issues**
Check the automated quality reports:
- `data/processed/preprocessing_report.json`
- `data/processed/dual_unification_report.json`

## 📈 Data Quality Validation

### ✅ **Automated Checks**
The pipeline includes comprehensive quality validation:

1. **Missing Values**: All datasets checked (excellent quality - 0% missing)
2. **Outliers**: Conservative detection and treatment (<2% capped)
3. **Skewness**: Acceptable levels (-0.20 to 0.41)
4. **Date Continuity**: Weekly Monday-based time series
5. **Dual Strategy Comparison**: Coverage and completeness metrics

### 📊 **Quality Report Example**
```
✅ EXCELLENT DATA QUALITY CONFIRMED!
   ✅ No missing values across all datasets
   ✅ Acceptable skewness levels (-0.20 to 0.41)  
   ✅ Minimal outliers (only email: 5.8%, ooh: 3.8%)
   ✅ Well-structured time series data

📊 DUAL STRATEGY DATASETS CREATED:
   Strategy A: 156 weeks, 9 channels, 96.3% completeness
   Strategy B: 104 weeks, 10 channels, 97.5% completeness
```

## 🔄 Updating Data

### **Adding New Data**
1. Place new Excel files in `data/raw/`
2. Re-run the pipeline:
```bash
source .venv/bin/activate
./setup_data.sh
```

### **Monthly Updates**
The pipeline is designed for easy updates:
```bash
#!/bin/bash
# Save as update_pipeline.sh

echo "🔄 Updating Ice Cream MMM Data Pipeline"
source .venv/bin/activate
./setup_data.sh
echo "✅ Pipeline update complete!"
```

## 🚀 Next Steps After Data Setup

Once the pipeline completes successfully:

1. **📊 Explore Datasets**:
   ```bash
   # Quick look at the unified datasets
   head data/processed/unified_dataset_*.csv
   ```

2. **🔍 Run EDA**:
   ```bash
   cd notebooks
   python 02_unified_data_eda.py
   ```

3. **🤖 Develop MMM Models**:
   - Use both strategic datasets for comparison
   - Start with Strategy B (complete coverage) for full attribution
   - Use Strategy A (full range) for trend validation

4. **💰 ROI Optimization**:
   - Compare model performance across strategies
   - Make data-driven decision on final approach

## 📞 Support

### **If the Pipeline Fails**:
1. **Check virtual environment**: `source .venv/bin/activate`
2. **Verify raw data**: Ensure all 10 Excel files exist in `data/raw/`
3. **Check Python output**: Look for specific error messages
4. **Validate file permissions**: `chmod +x setup_data.sh`

### **Success Indicators**:
✅ Two unified datasets created in `data/processed/`  
✅ Dual unification report generated  
✅ 10 preprocessed individual datasets  
✅ No critical errors in pipeline output  

---

**Remember**: The automated `setup_data.sh` pipeline handles all the complexity! Just ensure your virtual environment is set up and run the script. The dual strategy approach gives you both complete attribution AND recent trends for optimal MMM development! 🍦 
