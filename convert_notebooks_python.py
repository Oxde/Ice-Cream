#!/usr/bin/env python3
"""
📚 Jupyter Notebook to PDF Converter (Python Version)
Advanced script with multiple conversion methods for better reliability
"""

import os
import sys
import subprocess
import glob
from pathlib import Path
import nbformat
from nbconvert import PDFExporter, HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class NotebookConverter:
    def __init__(self, notebooks_dir="notebooks", output_dir="pdfbooks"):
        self.notebooks_dir = Path(notebooks_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.successful_conversions = 0
        self.failed_conversions = 0
        self.conversion_methods = []
    
    def check_dependencies(self):
        """Check and install required dependencies"""
        logger.info("🔍 Checking dependencies...")
        
        # Check if in virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            logger.info("✅ Running in virtual environment")
        else:
            logger.warning("⚠️  Not in virtual environment - consider activating .venv")
        
        # Required packages
        required_packages = ['nbconvert', 'nbformat']
        optional_packages = ['weasyprint', 'pyppeteer']
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} is available")
            except ImportError:
                logger.error(f"❌ {package} is required but not installed")
                self.install_package(package)
        
        for package in optional_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} is available (optional)")
            except ImportError:
                logger.info(f"ℹ️  {package} not available (optional)")
    
    def install_package(self, package):
        """Install a Python package"""
        logger.info(f"📦 Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info(f"✅ Successfully installed {package}")
        except subprocess.CalledProcessError:
            logger.error(f"❌ Failed to install {package}")
            sys.exit(1)
    
    def check_latex(self):
        """Check if LaTeX is available"""
        latex_commands = ['xelatex', 'pdflatex', 'lualatex']
        available_latex = []
        
        for cmd in latex_commands:
            try:
                subprocess.run([cmd, '--version'], 
                             capture_output=True, check=True, timeout=10)
                available_latex.append(cmd)
                logger.info(f"✅ {cmd} is available")
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                logger.info(f"❌ {cmd} not available")
        
        return available_latex
    
    def find_notebooks(self):
        """Find all .ipynb files in the notebooks directory"""
        pattern = str(self.notebooks_dir / "*.ipynb")
        notebooks = glob.glob(pattern)
        
        if not notebooks:
            logger.warning(f"No .ipynb files found in {self.notebooks_dir}")
            return []
        
        logger.info(f"📚 Found {len(notebooks)} notebook files:")
        for nb in notebooks:
            logger.info(f"  - {Path(nb).name}")
        
        return notebooks
    
    def convert_with_nbconvert_pdf(self, notebook_path, latex_engine='xelatex'):
        """Convert using nbconvert PDF exporter"""
        try:
            logger.info(f"  🔄 Trying PDF conversion with {latex_engine}...")
            
            # Configure PDF exporter
            pdf_exporter = PDFExporter()
            pdf_exporter.latex_command = [latex_engine, '-interaction=nonstopmode', '{filename}']
            
            # Remove problematic cells if any
            pdf_exporter.register_preprocessor(TagRemovePreprocessor(remove_cell_tags={"remove"}), enabled=True)
            
            # Read notebook
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            
            # Convert
            (body, resources) = pdf_exporter.from_notebook_node(notebook)
            
            # Save PDF
            output_path = self.output_dir / f"{Path(notebook_path).stem}.pdf"
            with open(output_path, 'wb') as f:
                f.write(body)
            
            logger.info(f"  ✅ PDF created: {output_path.name}")
            return True, str(output_path)
            
        except Exception as e:
            logger.info(f"  ❌ PDF conversion failed with {latex_engine}: {str(e)}")
            return False, str(e)
    
    def convert_with_html_intermediate(self, notebook_path):
        """Convert to HTML first, then try to convert to PDF"""
        try:
            logger.info(f"  🔄 Converting via HTML intermediate...")
            
            # Configure HTML exporter
            html_exporter = HTMLExporter()
            html_exporter.template_name = 'classic'
            
            # Read notebook
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            
            # Convert to HTML
            (body, resources) = html_exporter.from_notebook_node(notebook)
            
            # Save HTML
            html_path = self.output_dir / f"{Path(notebook_path).stem}.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(body)
            
            # Try to convert HTML to PDF using weasyprint
            try:
                import weasyprint
                logger.info(f"  🔄 Converting HTML to PDF with weasyprint...")
                
                pdf_path = self.output_dir / f"{Path(notebook_path).stem}.pdf"
                weasyprint.HTML(filename=str(html_path)).write_pdf(str(pdf_path))
                
                # Remove intermediate HTML file
                html_path.unlink()
                
                logger.info(f"  ✅ PDF created via HTML: {pdf_path.name}")
                return True, str(pdf_path)
                
            except ImportError:
                logger.info(f"  ✅ HTML created: {html_path.name} (weasyprint not available for PDF)")
                return True, str(html_path)
            except Exception as e:
                logger.info(f"  ⚠️  HTML created: {html_path.name} (PDF conversion failed: {str(e)})")
                return True, str(html_path)
                
        except Exception as e:
            logger.info(f"  ❌ HTML conversion failed: {str(e)}")
            return False, str(e)
    
    def convert_with_pandoc(self, notebook_path):
        """Try conversion using pandoc if available"""
        try:
            # Check if pandoc is available
            subprocess.run(['pandoc', '--version'], 
                         capture_output=True, check=True, timeout=5)
            
            logger.info(f"  🔄 Trying conversion with pandoc...")
            
            output_path = self.output_dir / f"{Path(notebook_path).stem}.pdf"
            
            # Convert using pandoc
            cmd = [
                'pandoc',
                str(notebook_path),
                '-o', str(output_path),
                '--pdf-engine=xelatex',
                '--variable', 'geometry:margin=1in'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and output_path.exists():
                logger.info(f"  ✅ PDF created with pandoc: {output_path.name}")
                return True, str(output_path)
            else:
                logger.info(f"  ❌ Pandoc conversion failed: {result.stderr}")
                return False, result.stderr
                
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            logger.info(f"  ❌ Pandoc not available")
            return False, "Pandoc not available"
    
    def convert_notebook(self, notebook_path):
        """Convert a single notebook using multiple methods"""
        notebook_name = Path(notebook_path).name
        logger.info(f"\n📖 Converting: {notebook_name}")
        
        conversion_methods = [
            ("nbconvert with xelatex", lambda: self.convert_with_nbconvert_pdf(notebook_path, 'xelatex')),
            ("nbconvert with pdflatex", lambda: self.convert_with_nbconvert_pdf(notebook_path, 'pdflatex')),
            ("HTML intermediate", lambda: self.convert_with_html_intermediate(notebook_path)),
            ("Pandoc", lambda: self.convert_with_pandoc(notebook_path))
        ]
        
        for method_name, method_func in conversion_methods:
            try:
                success, result = method_func()
                if success:
                    self.successful_conversions += 1
                    self.conversion_methods.append((notebook_name, method_name))
                    return True
            except Exception as e:
                logger.info(f"  ❌ {method_name} failed: {str(e)}")
                continue
        
        # If all methods failed
        logger.error(f"  ❌ All conversion methods failed for {notebook_name}")
        self.failed_conversions += 1
        return False
    
    def convert_all(self):
        """Convert all notebooks in the directory"""
        logger.info("🚀 Starting Jupyter Notebook to PDF Conversion Process")
        logger.info("=" * 60)
        
        # Check dependencies
        self.check_dependencies()
        
        # Check LaTeX availability
        available_latex = self.check_latex()
        if not available_latex:
            logger.warning("⚠️  No LaTeX engines found. Only HTML output will be available.")
            logger.info("💡 To install LaTeX on macOS: brew install --cask basictex")
        
        # Find notebooks
        notebooks = self.find_notebooks()
        if not notebooks:
            return
        
        logger.info(f"\n🔄 Starting conversion of {len(notebooks)} notebooks...")
        
        # Convert each notebook
        for notebook in notebooks:
            self.convert_notebook(notebook)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print conversion summary"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 CONVERSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Successfully converted: {self.successful_conversions} files")
        logger.info(f"❌ Failed conversions: {self.failed_conversions} files")
        
        if self.conversion_methods:
            logger.info(f"\n📋 Conversion methods used:")
            for notebook, method in self.conversion_methods:
                logger.info(f"  {notebook}: {method}")
        
        logger.info(f"\n📁 Output location: {self.output_dir}/")
        
        # List generated files
        output_files = list(self.output_dir.glob("*"))
        if output_files:
            logger.info(f"\n📄 Generated files:")
            for file in sorted(output_files):
                if file.is_file():
                    size = file.stat().st_size / 1024  # KB
                    logger.info(f"  {file.name} ({size:.1f} KB)")
        
        logger.info(f"\n🎉 Conversion process completed!")


def main():
    """Main function"""
    # Change to IceCream directory if not already there
    if not Path("notebooks").exists():
        icecream_path = Path("IceCream")
        if icecream_path.exists():
            os.chdir(icecream_path)
            logger.info(f"📁 Changed to directory: {icecream_path.absolute()}")
        else:
            logger.error("❌ Cannot find notebooks directory. Please run from project root.")
            sys.exit(1)
    
    # Create converter and run
    converter = NotebookConverter()
    converter.convert_all()
    
    # Optional: Open output directory
    try:
        import webbrowser
        choice = input("\n🔍 Would you like to open the pdfbooks folder? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            output_path = Path("pdfbooks").absolute()
            webbrowser.open(f"file://{output_path}")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main() 