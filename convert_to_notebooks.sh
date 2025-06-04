#!/bin/bash

# Convert Python MMM scripts to Jupyter Notebooks
# Usage: ./convert_to_notebooks.sh

echo "🔄 Converting Python files to Jupyter Notebooks..."

# Convert Foundation Model
echo "📊 Converting Foundation Model..."
jupytext --to notebook notebooks/03_mmm_foundation_corrected.py

# Convert Enhanced Model
echo "📈 Converting Enhanced Model..."
jupytext --to notebook notebooks/04_mmm_enhanced.py

# Convert Clean Performance
echo "📋 Converting Clean Performance..."
jupytext --to notebook notebooks/clean_model_performance.py

echo "✅ Conversion Complete!"
echo "📁 Generated .ipynb files:"
ls -la notebooks/*.ipynb | grep -E "(03_mmm|04_mmm|clean_model)"

echo ""
echo "🚀 Ready to open in Jupyter!"
echo "💡 Run: jupyter lab notebooks/" 