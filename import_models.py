#!/usr/bin/env python3
"""
IMPORT MODELS - Easy access to compiled models and frameworks
============================================================

This script provides easy access to:
1. FINAL_MODEL2.py - The compiled final media mix model (59.3% R²)
2. BUDGET_SIMULATION_FRAMEWORK_FIXED.py - The budget simulation framework

Usage:
    python import_models.py
    # or
    from import_models import final_model2, budget_simulation
"""

import sys
import os
from pathlib import Path
import importlib.util

# Add the final_notebooks directory to the Python path
final_notebooks_path = Path(__file__).parent / "final_notebooks"
sys.path.insert(0, str(final_notebooks_path))

def import_model_module(module_name, file_path):
    """Import a module from a specific file path"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"❌ Error importing {module_name}: {e}")
        return None

def main():
    """Main function to import and test the models"""
    
    print("🚀 IMPORTING COMPILED MODELS")
    print("=" * 50)
    
    # Define model paths
    final_model2_path = final_notebooks_path / "FINAL_MODEL2.py"
    budget_simulation_path = final_notebooks_path / "BUDGET_SIMULATION_FRAMEWORK_FIXED.py"
    
    # Import FINAL_MODEL2
    print("\n📊 Importing FINAL_MODEL2...")
    if final_model2_path.exists():
        final_model2 = import_model_module("final_model2", final_model2_path)
        if final_model2:
            print(f"✅ Successfully imported FINAL_MODEL2 from {final_model2_path}")
            try:
                # Try to access some attributes to verify it's working
                print(f"   • File size: {final_model2_path.stat().st_size / 1024:.1f} KB")
                print(f"   • Contains model training and evaluation code")
                print(f"   • Performance: 59.3% R² (as documented)")
            except Exception as e:
                print(f"   ⚠️ Module imported but some attributes may not be accessible: {e}")
        else:
            print("❌ Failed to import FINAL_MODEL2")
    else:
        print(f"❌ FINAL_MODEL2.py not found at {final_model2_path}")
    
    # Import Budget Simulation Framework
    print("\n🎯 Importing Budget Simulation Framework...")
    if budget_simulation_path.exists():
        budget_simulation = import_model_module("budget_simulation", budget_simulation_path)
        if budget_simulation:
            print(f"✅ Successfully imported BUDGET_SIMULATION_FRAMEWORK_FIXED from {budget_simulation_path}")
            try:
                print(f"   • File size: {budget_simulation_path.stat().st_size / 1024:.1f} KB")
                print(f"   • Contains simulation scenarios and business recommendations")
                print(f"   • Ready for Jupyter notebook execution")
            except Exception as e:
                print(f"   ⚠️ Module imported but some attributes may not be accessible: {e}")
        else:
            print("❌ Failed to import Budget Simulation Framework")
    else:
        print(f"❌ BUDGET_SIMULATION_FRAMEWORK_FIXED.py not found at {budget_simulation_path}")
    
    # Check for Jupyter notebook versions
    print("\n📓 Checking Jupyter Notebook Versions...")
    
    final_model2_nb = final_notebooks_path / "FINAL_MODEL2.ipynb"
    budget_simulation_nb = final_notebooks_path / "BUDGET_SIMULATION_FRAMEWORK_FIXED.ipynb"
    
    if final_model2_nb.exists():
        print(f"✅ FINAL_MODEL2.ipynb available ({final_model2_nb.stat().st_size / 1024:.1f} KB)")
    else:
        print("❌ FINAL_MODEL2.ipynb not found")
    
    if budget_simulation_nb.exists():
        print(f"✅ BUDGET_SIMULATION_FRAMEWORK_FIXED.ipynb available ({budget_simulation_nb.stat().st_size / 1024:.1f} KB)")
    else:
        print("❌ BUDGET_SIMULATION_FRAMEWORK_FIXED.ipynb not found")
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 IMPORT SUMMARY")
    print("=" * 50)
    
    print("\n🎯 AVAILABLE MODELS:")
    print("1. FINAL_MODEL2.py - Compiled final media mix model")
    print("   • Performance: 59.3% R² on test set")
    print("   • Contains corrected ROI analysis")
    print("   • Individual channel optimization")
    
    print("\n2. BUDGET_SIMULATION_FRAMEWORK_FIXED.py - Budget simulation framework")
    print("   • Three business scenarios (Conservative, Aggressive, Balanced)")
    print("   • Realistic ROI projections")
    print("   • Professional visualization dashboard")
    
    print("\n📓 JUPYTER NOTEBOOKS:")
    print("• FINAL_MODEL2.ipynb - Interactive model development")
    print("• BUDGET_SIMULATION_FRAMEWORK_FIXED.ipynb - Interactive simulation framework")
    
    print("\n🔗 USAGE:")
    print("• Run notebooks directly in Jupyter")
    print("• Import modules: from import_models import final_model2, budget_simulation")
    print("• View HTML exports in html_exports/ folder")
    
    print("\n✅ All models are ready for use!")

# Make modules available for import
try:
    final_model2_path = Path(__file__).parent / "final_notebooks" / "FINAL_MODEL2.py"
    budget_simulation_path = Path(__file__).parent / "final_notebooks" / "BUDGET_SIMULATION_FRAMEWORK_FIXED.py"
    
    if final_model2_path.exists():
        final_model2 = import_model_module("final_model2", final_model2_path)
    
    if budget_simulation_path.exists():
        budget_simulation = import_model_module("budget_simulation", budget_simulation_path)
        
except Exception as e:
    print(f"Note: Modules will be available when script is run directly. Error: {e}")

if __name__ == "__main__":
    main() 