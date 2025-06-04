#!/bin/bash

# 📚 Jupyter Notebook to PDF Converter for macOS
# This script converts all .ipynb files to PDF and organizes them in pdfbooks folder

set -e  # Exit on any error

echo "🚀 Starting Jupyter Notebook to PDF Conversion Process"
echo "======================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -d "notebooks" ]; then
    print_error "notebooks directory not found. Please run this script from the IceCream project root."
    exit 1
fi

# Create pdfbooks directory if it doesn't exist
print_status "Creating pdfbooks directory..."
mkdir -p pdfbooks

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Homebrew is installed
if ! command_exists brew; then
    print_error "Homebrew is not installed. Please install Homebrew first:"
    echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

# Check if LaTeX is installed
if ! command_exists pdflatex && ! command_exists xelatex; then
    print_warning "LaTeX not found. Installing BasicTeX (lighter LaTeX distribution)..."
    print_status "This may take a few minutes..."
    
    # Install BasicTeX (lighter than full MacTeX)
    brew install --cask basictex
    
    # Update PATH to include LaTeX binaries
    export PATH="/usr/local/texlive/2024basic/bin/universal-darwin:$PATH"
    export PATH="/Library/TeX/texbin:$PATH"
    
    # Update tlmgr and install required packages
    print_status "Updating LaTeX package manager..."
    sudo tlmgr update --self
    
    print_status "Installing required LaTeX packages..."
    sudo tlmgr install adjustbox babel-german background bidi collectbox csquotes everypage filehook footmisc footnotebackref framed fvextra letltxmacro ly1 mdframed mweights needspace pagecolor sourcecodepro sourcesanspro titling ucharcat ulem unicode-math upquote xecjk xstring
    
    print_success "LaTeX installation completed!"
else
    print_success "LaTeX is already installed!"
fi

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    print_status "Activating virtual environment..."
    source .venv/bin/activate
    print_success "Virtual environment activated!"
else
    print_warning "No virtual environment found. Using system Python."
fi

# Check if jupyter-nbconvert is installed
print_status "Checking for nbconvert..."
if ! python -c "import nbconvert" 2>/dev/null; then
    print_status "Installing nbconvert..."
    pip install nbconvert
fi

# Find all .ipynb files in notebooks directory
print_status "Scanning for Jupyter notebook files..."
notebook_files=($(find notebooks -name "*.ipynb" -type f))

if [ ${#notebook_files[@]} -eq 0 ]; then
    print_warning "No .ipynb files found in notebooks directory."
    exit 0
fi

print_success "Found ${#notebook_files[@]} notebook files to convert:"
for file in "${notebook_files[@]}"; do
    echo "  - $(basename "$file")"
done

echo ""
print_status "Starting conversion process..."

# Counter for successful conversions
successful_conversions=0
failed_conversions=0

# Convert each notebook
for notebook in "${notebook_files[@]}"; do
    filename=$(basename "$notebook" .ipynb)
    print_status "Converting: $filename.ipynb"
    
    # Try conversion with different engines
    conversion_successful=false
    
    # Method 1: Try with xelatex (best for complex documents)
    if command_exists xelatex && [ "$conversion_successful" = false ]; then
        print_status "  Attempting conversion with xelatex..."
        if jupyter nbconvert --to pdf --pdf-engine=xelatex "$notebook" --output-dir=pdfbooks 2>/dev/null; then
            conversion_successful=true
            print_success "  ✅ Converted with xelatex: $filename.pdf"
        fi
    fi
    
    # Method 2: Try with pdflatex (fallback)
    if command_exists pdflatex && [ "$conversion_successful" = false ]; then
        print_status "  Attempting conversion with pdflatex..."
        if jupyter nbconvert --to pdf --pdf-engine=pdflatex "$notebook" --output-dir=pdfbooks 2>/dev/null; then
            conversion_successful=true
            print_success "  ✅ Converted with pdflatex: $filename.pdf"
        fi
    fi
    
    # Method 3: Try HTML to PDF conversion (universal fallback)
    if [ "$conversion_successful" = false ]; then
        print_status "  LaTeX failed, trying HTML conversion..."
        if jupyter nbconvert --to html "$notebook" --output-dir=pdfbooks 2>/dev/null; then
            print_success "  ✅ Converted to HTML: $filename.html"
            print_warning "  Note: HTML version created instead of PDF"
            conversion_successful=true
        fi
    fi
    
    # Update counters
    if [ "$conversion_successful" = true ]; then
        ((successful_conversions++))
    else
        print_error "  ❌ Failed to convert: $filename.ipynb"
        ((failed_conversions++))
    fi
    
    echo ""
done

# Summary
echo "======================================================"
print_status "Conversion Summary:"
print_success "Successfully converted: $successful_conversions files"
if [ $failed_conversions -gt 0 ]; then
    print_error "Failed conversions: $failed_conversions files"
fi

print_status "Output location: pdfbooks/"
echo ""

# List generated files
if [ $successful_conversions -gt 0 ]; then
    print_status "Generated files:"
    ls -la pdfbooks/*.pdf pdfbooks/*.html 2>/dev/null | while read line; do
        echo "  $line"
    done
fi

echo ""
print_success "🎉 Notebook conversion process completed!"

# Optional: Open the pdfbooks folder
read -p "Would you like to open the pdfbooks folder? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    open pdfbooks/
fi 