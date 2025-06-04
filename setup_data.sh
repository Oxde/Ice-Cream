# Function to run Python script and check for errors
run_python_script() {
    local script_path=$1
    local description=$2
    
    print_info "Running $description..."
    
    # Check if script exists
    if [ ! -f "$script_path" ]; then
        print_error "Script not found: $script_path"
        return 1
    fi
    
    # Run the Python script
    cd notebooks
    python "$(basename "$script_path")" 2>&1
    local exit_code=$?
    cd ..
    
    if [ $exit_code -eq 0 ]; then
        print_status "$description completed successfully"
        return 0
    else
        print_error "$description failed (exit code: $exit_code)"
        print_info "Please check the script manually: $script_path"
        return 1
    fi
}

# Main data processing pipeline
echo ""
echo "🚀 Starting Data Processing Pipeline"
echo "====================================="

# Step 1: Basic Cleaning
if [ -f "notebooks/00_basic_cleaning.py" ]; then
    run_python_script "notebooks/00_basic_cleaning.py" "Basic Data Cleaning"
    
    if [ $? -eq 0 ]; then
        # Check if interim files were created
        if [ "$(ls -A data/interim/ 2>/dev/null)" ]; then
            print_status "Cleaned data files created in data/interim/"
            echo "   Files: $(ls data/interim/ | wc -l | tr -d ' ') files created"
        else
            print_warning "No interim files created. Check the cleaning script."
        fi
    else
        print_error "Basic cleaning failed. Cannot proceed."
        exit 1
    fi
else
    print_error "Basic cleaning script not found: notebooks/00_basic_cleaning.py"
    exit 1
fi

echo ""

# Step 2: Data Preprocessing 
if [ -f "notebooks/01_data_preprocessing.py" ]; then
    print_info "🔧 Running data preprocessing step..."
    run_python_script "notebooks/01_data_preprocessing.py" "Data Preprocessing"
    
    if [ $? -eq 0 ]; then
        # Check if processed files were created
        if [ "$(ls -A data/processed/ 2>/dev/null)" ]; then
            print_status "Preprocessed data files created in data/processed/"
            echo "   Files: $(ls data/processed/*_preprocessed.csv 2>/dev/null | wc -l | tr -d ' ') preprocessed files created"
        else
            print_warning "No processed files created. Check the preprocessing script."
        fi
    else
        print_error "Data preprocessing failed. Cannot proceed."
        exit 1
    fi
else
    print_error "Data preprocessing script not found: notebooks/01_data_preprocessing.py"
    exit 1
fi

echo ""

# Step 3: Data Unification (CRITICAL)
if [ -f "notebooks/01b_data_unification_dual.py" ]; then
    print_info "🔑 Running CRITICAL dual data unification step..."
    run_python_script "notebooks/01b_data_unification_dual.py" "Dual Data Unification (CRITICAL)"
    
    if [ $? -eq 0 ]; then
        # Check if unified datasets were created
        if [ "$(ls -A data/processed/ 2>/dev/null)" ]; then
            print_status "✨ Unified datasets created in data/processed/"
            echo "   Files created:"
            ls -la data/processed/ | grep -v "^total" | grep -v "^d"
            
            # Check for the dual strategy datasets
            if [ -f "data/processed/unified_dataset_full_range_2022_2024.csv" ]; then
                print_status "Strategy A dataset: unified_dataset_full_range_2022_2024.csv ✅"
            fi
            if [ -f "data/processed/unified_dataset_complete_coverage_2022_2023.csv" ]; then
                print_status "Strategy B dataset: unified_dataset_complete_coverage_2022_2023.csv ✅"
            fi
            if [ -f "data/processed/dual_unification_report.json" ]; then
                print_status "Unification report: dual_unification_report.json ✅"
            fi
        else
            print_error "No processed files created. Check the unification script."
            exit 1
        fi
    else
        print_error "Data unification failed. This is critical for the MMM analysis."
        exit 1
    fi
else
    print_error "Data unification script not found: notebooks/01b_data_unification_dual.py"
    exit 1
fi

echo ""
echo "🎉 Data Setup Complete!"
echo "======================"
print_status "Data cleaning, preprocessing, and dual unification completed successfully"
echo ""
print_info "🎯 DUAL STRATEGY DATASETS CREATED:"
echo "1. 📊 Strategy A (Full Range): data/processed/unified_dataset_full_range_2022_2024.csv"
echo "   📅 2022-2024 (156 weeks) | 9 channels | Best for trend analysis"
echo "2. 📊 Strategy B (Complete): data/processed/unified_dataset_complete_coverage_2022_2023.csv"
echo "   📅 2022-2023 (104 weeks) | 10 channels | Best for full attribution"
echo "3. 📋 Comparison Report: data/processed/dual_unification_report.json"
echo ""
print_info "Next Steps:"
echo "1. 📊 Review both unified datasets and choose modeling strategy"
echo "2. 🔍 Run EDA on selected dataset(s)" 
echo "3. 🎯 Develop parallel MMM models for comparison"
echo "4. 💰 Optimize ROI and attribution insights"
echo ""
print_info "File structure:"
echo "   data/raw/        - Original data ($(ls data/raw/ 2>/dev/null | wc -l | tr -d ' ') files)"
echo "   data/interim/    - Cleaned data ($(ls data/interim/ 2>/dev/null | wc -l | tr -d ' ') files)"
echo "   data/processed/  - Final datasets ($(ls data/processed/ 2>/dev/null | wc -l | tr -d ' ') files)"
echo ""

# Optional: Ask if user wants to run EDA
read -p "🤔 Would you like to see a quick data summary? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    print_info "📈 Quick Data Summary:"
    
    # Check Strategy A dataset
    if [ -f "data/processed/unified_dataset_full_range_2022_2024.csv" ]; then
        echo ""
        echo "📊 Strategy A (Full Range 2022-2024):"
        python -c "
import pandas as pd
try:
    df = pd.read_csv('data/processed/unified_dataset_full_range_2022_2024.csv')
    print(f'   📅 Date range: {df[\"date\"].min()} to {df[\"date\"].max()}')
    print(f'   📊 Rows: {len(df):,} weeks')
    print(f'   📋 Columns: {len(df.columns)} features')
    print(f'   🔍 Missing values: {df.isnull().sum().sum()}')
    print(f'   📺 Media channels: ~9 (excludes email)')
except Exception as e:
    print(f'   ⚠️  Could not read dataset: {e}')
"
    fi
    
    # Check Strategy B dataset
    if [ -f "data/processed/unified_dataset_complete_coverage_2022_2023.csv" ]; then
        echo ""
        echo "📊 Strategy B (Complete Coverage 2022-2023):"
        python -c "
import pandas as pd
try:
    df = pd.read_csv('data/processed/unified_dataset_complete_coverage_2022_2023.csv')
    print(f'   📅 Date range: {df[\"date\"].min()} to {df[\"date\"].max()}')
    print(f'   📊 Rows: {len(df):,} weeks')
    print(f'   📋 Columns: {len(df.columns)} features')
    print(f'   🔍 Missing values: {df.isnull().sum().sum()}')
    print(f'   📺 Media channels: ~10 (includes all channels)')
except Exception as e:
    print(f'   ⚠️  Could not read dataset: {e}')
"
    fi
fi

echo ""
print_status "🍦 Ice Cream MMM dual data strategy is ready for analysis!"

echo "✨ Done!" 