# 🍦 Ice Cream Company - Media Mix Modeling (MMM) Project

A comprehensive Media Mix Modeling analysis for an ice cream company, focusing on understanding the impact of various marketing channels on sales performance with enhanced adstock effects and seasonality controls.

## 📊 Project Status & Achievements

### 🎯 **Current Model Performance**
- **Enhanced MMM Model**: ~55% R² (significant improvement from 11.9% baseline)
- **Key Breakthrough**: Proper adstock effects and seasonality controls
- **ROI Analysis**: Complete spend vs media metrics comparison

### 🏆 **Major Achievements**
1. **Adstock Implementation**: Proper media carryover effects with decay rates
2. **Seasonality Controls**: Critical for ice cream business (high seasonal variation)
3. **Unified Data Pipeline**: Clean, standardized data processing
4. **Media Channel Analysis**: TV, Print, Promo Email, and other channels
5. **Industry Standard Validation**: Tested against industry-standard decay rates

## 📁 Project Structure

```
IceCream/
├── data/                          # Data storage (add to .gitignore)
│   ├── raw/                       # Original, unmodified data
│   ├── interim/                   # Intermediate processing files
│   ├── processed/                 # Final, clean datasets
│   └── explanations/              # Data documentation
├── notebooks/                     # Analysis notebooks
│   ├── hypothesis/                # Hypothesis testing notebooks
│   ├── 00_basic_cleaning.ipynb    # Initial data cleaning
│   ├── 01_data_preprocessing.ipynb # Data preprocessing
│   ├── 01b_data_unification_dual.ipynb # Data unification (KEY)
│   ├── 02_unified_data_eda.ipynb  # Exploratory data analysis
│   ├── 03_mmm_foundation_corrected.ipynb # Basic MMM model
│   ├── 04_mmm_enhanced.ipynb      # Enhanced MMM model (MAIN)
│   └── 05_promo_email_analysis.ipynb # Specific channel analysis
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── cleaning.py                # Data cleaning functions
│   ├── adstock.py                 # Adstock transformation functions
│   ├── features.py                # Feature engineering
│   └── modeling.py                # MMM modeling functions
├── reports/                       # Generated reports and visualizations
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment (if used)
└── README.md                      # This file
```

## 📚 Notebook Guide & Workflow

### 🔧 **Data Preparation** (Run First)
1. **`00_basic_cleaning.ipynb`** - Initial data cleaning and standardization
2. **`01b_data_unification_dual.ipynb`** - **KEY FILE** - Unifies all data sources
   - Merges multiple data sources
   - Creates unified date structure
   - Standardizes column names
   - **Run this after getting new data**

### 📈 **Exploratory Analysis**
3. **`01_data_preprocessing.ipynb`** - Advanced preprocessing
4. **`02_unified_data_eda.ipynb`** - Comprehensive exploratory data analysis
   - Data quality checks
   - Correlation analysis
   - Seasonal patterns
   - Missing data analysis

### 🎯 **Media Mix Modeling**
5. **`03_mmm_foundation_corrected.ipynb`** - Basic MMM model (R² ~11.9%)
6. **`04_mmm_enhanced.ipynb`** - **MAIN MMM MODEL** (R² ~55%)
   - Adstock effects implementation
   - Seasonality controls
   - Time trends
   - Channel attribution
   - See `04_mmm_enhanced_explanation.md` for detailed explanation

### 🔍 **Specialized Analysis**
7. **`05_promo_email_analysis.ipynb`** - Deep dive into promotional email effects

## 🧪 Hypothesis Testing & Experimentation

### 📍 **Where to Store Hypotheses**
All hypothesis testing notebooks are stored in:
```
notebooks/hypothesis/
```

### 🔬 **Current Hypothesis Tests**
1. **`adstock_demonstration.ipynb`** - Demonstrates adstock effects
2. **`06_spend_vs_media_metrics_analysis.ipynb`** - Spend vs Media Metrics comparison
   - **Key Finding**: Perfect correlation between spend and media metrics
   - **Recommendation**: Use spend-based approach for simplicity
3. **`05_mmm_industry_standard_adstock.py`** - Industry standard decay rate testing
   - **Result**: Industry standards didn't improve model performance
   - **Conclusion**: Current decay rates are optimal for this business

### 📝 **Hypothesis Documentation Format**
When creating new hypothesis tests:
1. Create notebook in `notebooks/hypothesis/`
2. Use naming convention: `##_descriptive_name.ipynb`
3. Include:
   - **Hypothesis statement**
   - **Methodology**
   - **Results**
   - **Business implications**
   - **Next steps**

## 🚀 Getting Started

### ⚡ Quick Start (Automated)
**For fast setup, use our automated script:**
```bash
# 1. Navigate to project directory
cd /Users/nikmarf/IceCream

# 2. Place your raw data files in data/raw/
# (sales_data.csv, media_spend.xlsx, etc.)

# 3. Run the automated setup script
./setup_data.sh
```

The script will:
- ✅ Check and activate virtual environment
- ✅ Install missing dependencies
- ✅ Run data cleaning (`00_basic_cleaning.ipynb`)
- ✅ Run data unification (`01b_data_unification_dual.ipynb`) 
- ✅ Validate data quality
- ✅ Optionally run EDA analysis

### 🔧 Manual Setup (Step by Step)

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate icecream-mmm
```

### 2. Data Preparation
```bash
# Step 1: Place raw data in data/raw/
# Step 2: Run data cleaning and unification
jupyter notebook notebooks/00_basic_cleaning.ipynb
jupyter notebook notebooks/01b_data_unification_dual.ipynb
```

### 3. Main Analysis
```bash
# Run the enhanced MMM model
jupyter notebook notebooks/04_mmm_enhanced.ipynb
```

## 🔧 Troubleshooting

### 🤖 **Automated Script Issues**
**Script won't run**:
```bash
# Make sure script is executable
chmod +x setup_data.sh

# Run with explicit bash
bash setup_data.sh
```

**"Permission denied"**:
```bash
# Fix permissions
chmod 755 setup_data.sh
```

**"Virtual environment not found"**:
```bash
# Create virtual environment first
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**"Notebook execution failed"**:
- Check if data files are in `data/raw/`
- Ensure data files are not corrupted
- Run notebooks manually to see specific errors
- Check Python package versions

### 📊 **Data Issues**
**"No raw data found"**:
- Place your data files in `data/raw/` directory
- Supported formats: `.csv`, `.xlsx`, `.xls`
- Ensure files have proper column headers

**"Date parsing errors"**:
- Check date formats in your data
- Common formats: `YYYY-MM-DD`, `MM/DD/YYYY`, `DD/MM/YYYY`
- Ensure no missing dates in time series

## 🔑 Key Insights & Learnings

### 📊 **Adstock Effects**
- **Decay Rate**: 0.5 (50% carryover each week)
- **Critical for ROI accuracy**: Without adstock, media ROI is significantly underestimated
- **Channel Differences**: Different channels may have different decay patterns

### 🌡️ **Seasonality Impact**
- **Ice cream business is highly seasonal**: Critical to control for natural demand patterns
- **Summer peaks**: Natural sales increase not attributable to media
- **Winter lows**: Media effects may be amplified due to lower baseline

### 💰 **Spend vs Media Metrics**
- **Perfect correlation found**: Spend and media metrics (GRPs, impressions) are highly correlated
- **Recommendation**: Use spend-based modeling for simplicity and direct ROI calculation
- **Business value**: Easier interpretation and budget allocation

### 📈 **Model Performance**
- **R² improvement**: From 11.9% to ~55% through proper controls
- **Remaining opportunity**: 45% of variance still unexplained
- **Potential factors**: External events, competitive activity, product changes

## 🎯 Next Steps & Opportunities

### 🔍 **Model Improvements**
1. **External factors**: Weather, events, competitive activity
2. **Saturation curves**: Diminishing returns modeling
3. **Interaction effects**: Cross-channel synergies
4. **Granular analysis**: Category/SKU level modeling

### 📊 **Business Applications**
1. **Budget optimization**: Optimal spend allocation across channels
2. **Scenario planning**: What-if analysis for different spend levels
3. **Channel planning**: Timing and sequencing optimization
4. **ROI reporting**: Regular performance monitoring

### 🧪 **Future Hypotheses**
1. **Weather impact**: Correlation between temperature and sales
2. **Competitive effects**: Impact of competitor advertising
3. **Cross-channel synergies**: Do channels work better together?
4. **Category cannibalization**: Do promotions steal from base sales?

## 👥 Team & Collaboration

### 📋 **Code Standards**
- **Modular code**: Functions in `src/` modules
- **Documentation**: Clear comments and docstrings
- **Reproducibility**: Fixed random seeds, version control
- **Testing**: Validate results across different time periods

### 🔄 **Version Control**
- **Data**: Not included in git (see .gitignore)
- **Code**: All notebooks and scripts tracked
- **Results**: Key findings documented in markdown files

## 📞 Support & Questions

For questions about:
- **Data issues**: Check `data/explanations/` folder
- **Model interpretation**: See `04_mmm_enhanced_explanation.md`
- **Hypothesis testing**: Review existing tests in `notebooks/hypothesis/`
- **Code functionality**: Check relevant `src/` modules

---

**Last Updated**: January 2024  
**Model Version**: Enhanced MMM v1.0  
**Data Version**: Unified dataset post-cleaning
