# 📚 Jupyter Notebook to PDF Conversion Scripts

This directory contains two scripts to convert all your Jupyter notebooks to PDF format and organize them in the `pdfbooks` folder.

## 🎯 Quick Start

### Option 1: Bash Script (Recommended for macOS)
```bash
./convert_notebooks_to_pdf.sh
```

### Option 2: Python Script (More Advanced)
```bash
python convert_notebooks_python.py
```

## 📋 What These Scripts Do

1. **Automatically detect** all `.ipynb` files in the `notebooks/` directory
2. **Install missing dependencies** (LaTeX, Python packages)
3. **Convert notebooks to PDF** using multiple fallback methods
4. **Organize output** in the `pdfbooks/` folder
5. **Provide detailed progress** and error reporting

## 🛠 Requirements

### For Both Scripts:
- **macOS** (scripts optimized for macOS)
- **Homebrew** installed
- **Python virtual environment** (recommended)

### Dependencies Installed Automatically:
- **LaTeX** (BasicTeX via Homebrew)
- **nbconvert** (Python package)
- **Additional LaTeX packages** (as needed)

## 🔧 Conversion Methods

The scripts try multiple conversion methods in order:

1. **nbconvert with xelatex** (Best quality)
2. **nbconvert with pdflatex** (Good fallback)
3. **HTML intermediate** (Always works)
4. **Pandoc** (If available)

## 📊 Features

### Bash Script Features:
- ✅ **Automatic LaTeX installation** via Homebrew
- ✅ **Colored terminal output** for better readability
- ✅ **Multiple PDF engines** (xelatex, pdflatex)
- ✅ **HTML fallback** when PDF fails
- ✅ **Virtual environment detection**
- ✅ **Interactive folder opening**

### Python Script Features:
- ✅ **Advanced error handling** and logging
- ✅ **Dependency checking** and auto-installation
- ✅ **Multiple conversion engines**
- ✅ **HTML-to-PDF conversion** with weasyprint
- ✅ **Detailed conversion statistics**
- ✅ **File size reporting**

## 🚀 Example Usage

### Running the Bash Script:
```bash
cd IceCream
./convert_notebooks_to_pdf.sh
```

**Output:**
```
🚀 Starting Jupyter Notebook to PDF Conversion Process
======================================================
[INFO] Creating pdfbooks directory...
[SUCCESS] LaTeX is already installed!
[INFO] Activating virtual environment...
[SUCCESS] Found 6 notebook files to convert:
  - 00_basic_cleaning.ipynb
  - 01_data_preprocessing.ipynb
  - 02_unified_data_eda.ipynb
  - 03_mmm_foundation_corrected.ipynb
  - 04_mmm_enhanced.ipynb
  - 05_promo_email_analysis.ipynb
```

### Running the Python Script:
```bash
cd IceCream
python convert_notebooks_python.py
```

## 📁 Output Structure

After running either script:
```
pdfbooks/
├── 00_basic_cleaning.pdf
├── 01_data_preprocessing.pdf
├── 02_unified_data_eda.pdf
├── 03_mmm_foundation_corrected.pdf
├── 04_mmm_enhanced.pdf
└── 05_promo_email_analysis.pdf
```

## 🛠 Troubleshooting

### If LaTeX Installation Fails:
```bash
# Install manually
brew install --cask basictex

# Update PATH
export PATH="/Library/TeX/texbin:$PATH"

# Install required packages
sudo tlmgr update --self
sudo tlmgr install collection-fontsrecommended
```

### If Python Dependencies Are Missing:
```bash
# Activate virtual environment
source .venv/bin/activate

# Install required packages
pip install nbconvert nbformat weasyprint
```

### If All PDF Methods Fail:
The scripts will automatically fall back to HTML format, which can be:
- **Printed to PDF** from your browser
- **Converted later** using online tools
- **Viewed directly** in any web browser

## 💡 Pro Tips

1. **Use the Bash script first** - it's simpler and handles most cases
2. **Run in virtual environment** - prevents dependency conflicts
3. **Check output folder** - scripts will show you what was created
4. **Large notebooks** may take longer to convert
5. **Complex plots** might require manual adjustment in some cases

## 🔍 What's Next?

After conversion, your PDFs will be ready for:
- 📧 **Email sharing** with stakeholders
- 📊 **Presentation preparation**
- 📁 **Archive storage**
- 🖨 **Printing** for offline review

## 📞 Need Help?

If conversion fails:
1. Check the terminal output for specific error messages
2. Ensure you're in the correct directory (`IceCream/`)
3. Verify that notebook files are valid (can be opened in Jupyter)
4. Try the alternative script if one doesn't work

---
*🎯 Goal: Transform your data science notebooks into professional PDF reports ready for senior stakeholder discussions!* 