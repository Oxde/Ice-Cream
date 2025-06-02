#!/bin/bash

# =============================================================================
# Ice Cream MMM - Data Setup and Unification Script (macOS)
# =============================================================================

echo "🍦 Ice Cream MMM - Data Setup & Unification"
echo "==========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "notebooks" ]; then
    echo "❌ Error: Please run this script from the IceCream project root directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected files: README.md, notebooks/"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_warning "Virtual environment not detected. Attempting to activate..."
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        print_status "Activated .venv"
    elif [ -d "venv" ]; then
        source venv/bin/activate  
        print_status "Activated venv"
    else
        print_error "No virtual environment found. Please create one first:"
        echo "   python -m venv .venv"
        echo "   source .venv/bin/activate"
        echo "   pip install -r requirements.txt"
        exit 1
    fi
fi

# Check if required packages are installed
print_info "Checking required packages..."
python -c "import pandas, numpy, matplotlib, seaborn, sklearn, statsmodels" 2>/dev/null
if [ $? -ne 0 ]; then
    print_warning "Some required packages missing. Installing..."
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        print_status "Packages installed successfully"
    else
        print_error "Failed to install packages. Please check requirements.txt"
        exit 1
    fi
else
    print_status "All required packages are installed"
fi

# Create data directories if they don't exist
print_info "Setting up data directory structure..."
mkdir -p data/{raw,interim,processed,explanations}
print_status "Data directories created"
print_info "Directory structure:"
echo "   data/raw/        - Place your original data files here"
echo "   data/interim/    - Cleaned data (generated automatically)"
echo "   data/processed/  - Final unified datasets (generated automatically)"
echo "   data/explanations/ - Data documentation"

# Check if raw data exists
if [ ! "$(ls -A data/raw/ 2>/dev/null)" ]; then
    print_warning "No raw data found in data/raw/"
    print_info "Please add your raw data files to data/raw/ before running this script"
    print_info "Expected files:"
    echo "   - sales_data.csv (or similar)"
    echo "   - media_spend.xlsx (or similar)" 
    echo "   - tv_metrics.csv (or similar)"
    echo ""
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Please add raw data files and run this script again"
        exit 0
    fi
else
    print_status "Raw data files found in data/raw/"
    ls data/raw/ | head -5
fi

# Function to run notebook and check for errors
run_notebook() {
    local notebook_path=$1
    local description=$2
    
    print_info "Running $description..."
    
    # Run the notebook
    jupyter nbconvert --execute --to notebook --inplace "$notebook_path" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        print_status "$description completed successfully"
        return 0
    else
        print_error "$description failed"
        print_info "Please check the notebook manually: $notebook_path"
        return 1
    fi
}

# Main data processing pipeline
echo ""
echo "🚀 Starting Data Processing Pipeline"
echo "====================================="

# Step 1: Basic Cleaning
if [ -f "notebooks/00_basic_cleaning.ipynb" ]; then
    run_notebook "notebooks/00_basic_cleaning.ipynb" "Basic Data Cleaning"
    
    if [ $? -eq 0 ]; then
        # Check if interim files were created
        if [ "$(ls -A data/interim/ 2>/dev/null)" ]; then
            print_status "Cleaned data files created in data/interim/"
            echo "   Files: $(ls data/interim/ | wc -l | tr -d ' ') files created"
        else
            print_warning "No interim files created. Check the cleaning notebook."
        fi
    fi
else
    print_error "Basic cleaning notebook not found: notebooks/00_basic_cleaning.ipynb"
    exit 1
fi

echo ""

# Step 2: Data Unification (CRITICAL)
if [ -f "notebooks/01b_data_unification_dual.ipynb" ]; then
    print_info "🔑 Running CRITICAL data unification step..."
    run_notebook "notebooks/01b_data_unification_dual.ipynb" "Data Unification (CRITICAL)"
    
    if [ $? -eq 0 ]; then
        # Check if processed files were created
        if [ "$(ls -A data/processed/ 2>/dev/null)" ]; then
            print_status "✨ Unified dataset created in data/processed/"
            echo "   Files created:"
            ls -la data/processed/ | grep -v "^total" | grep -v "^d"
            
            # Check for the main datasets
            if [ -f "data/processed/unified_dataset.csv" ]; then
                print_status "Main dataset: unified_dataset.csv ✅"
            fi
            if [ -f "data/processed/mmm_ready_data.csv" ]; then
                print_status "MMM-ready dataset: mmm_ready_data.csv ✅"
            fi
        else
            print_error "No processed files created. Check the unification notebook."
            exit 1
        fi
    else
        print_error "Data unification failed. This is critical for the MMM analysis."
        exit 1
    fi
else
    print_error "Data unification notebook not found: notebooks/01b_data_unification_dual.ipynb"
    exit 1
fi

echo ""
echo "🎉 Data Setup Complete!"
echo "======================"
print_status "Data cleaning and unification completed successfully"
echo ""
print_info "Next Steps:"
echo "1. 📊 Review the unified dataset: data/processed/unified_dataset.csv"
echo "2. 🔍 Run EDA notebook: notebooks/02_unified_data_eda.ipynb" 
echo "3. 🎯 Run enhanced MMM model: notebooks/04_mmm_enhanced.ipynb"
echo ""
print_info "File structure:"
echo "   data/raw/        - Original data ($(ls data/raw/ 2>/dev/null | wc -l | tr -d ' ') files)"
echo "   data/interim/    - Cleaned data ($(ls data/interim/ 2>/dev/null | wc -l | tr -d ' ') files)"
echo "   data/processed/  - Final datasets ($(ls data/processed/ 2>/dev/null | wc -l | tr -d ' ') files)"
echo ""

# Optional: Ask if user wants to run EDA
read -p "🤔 Would you like to run the EDA notebook now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "notebooks/02_unified_data_eda.ipynb" ]; then
        run_notebook "notebooks/02_unified_data_eda.ipynb" "Exploratory Data Analysis"
        print_status "EDA completed! Check the notebook for visualizations and insights."
    else
        print_error "EDA notebook not found: notebooks/02_unified_data_eda.ipynb"
    fi
fi

echo ""
print_status "🍦 Ice Cream MMM data setup is ready for analysis!"

# Final check - show data summary
if [ -f "data/processed/unified_dataset.csv" ]; then
    echo ""
    print_info "📈 Quick data summary:"
    python -c "
import pandas as pd
try:
    df = pd.read_csv('data/processed/unified_dataset.csv')
    print(f'   📅 Date range: {df.iloc[:, 0].min()} to {df.iloc[:, 0].max()}')
    print(f'   📊 Rows: {len(df):,}')
    print(f'   📋 Columns: {len(df.columns)}')
    print(f'   🔍 Missing values: {df.isnull().sum().sum()}')
except Exception as e:
    print(f'   ⚠️  Could not read dataset: {e}')
"
fi

echo "✨ Done!" 