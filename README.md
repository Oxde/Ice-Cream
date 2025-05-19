# Marketing Mix Modelling Data Preparation Workflow

This project provides a two-step workflow to convert raw Excel (.xlsx) files into a cleaned and unified CSV dataset suitable for Marketing Mix Modelling (MMM).

## Prerequisites

- Python 3.x (it is recommended to use a virtual environment)
- Required Python packages (see `requirements.txt`)

## Setup

1.  **Clone the repository (if applicable) or download the project files.**

2.  **Navigate to the project root directory in your terminal.**

3.  **Create a Python virtual environment:**
    ```bash
    python3 -m venv venv
    ```
    (On some systems, you might need to use `python` instead of `python3`)

4.  **Activate the virtual environment:**
    -   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
    -   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    You should see `(venv)` at the beginning of your terminal prompt, indicating the virtual environment is active.

5.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Directory Structure

Ensure your project has the following structure:

```
. (project_root)/
├── cleaning/
│   ├── 01_convert_excel_to_csv.py
│   ├── 02_clean_unified_data.py
│   └── 03_export_to_excel.py
├── raw_excel_files/      <-- Place your raw .xlsx files here
├── data_csv/             <-- Intermediate CSVs will be generated here (gitignored)
├── cleaning/processed/   <-- Cleaned CSVs will be generated here (gitignored)
└── cleaning/processed_xlsx/ <-- Excel exports will be generated here (gitignored)
└── README.md
```

- `cleaning/`: Contains the Python scripts for the workflow.
- `raw_excel_files/`: **You need to create this directory.** Place all your source Excel files (e.g., `sales.xlsx`, `tv_promo.xlsx`) in this folder.
- `data_csv/`: This directory will be automatically created by the first script and will store the converted CSV files. It is gitignored.
- `cleaning/processed/`: This directory will be automatically created by the second script and will store the final `unified_dataset_base.csv` and other intermediate cleaned CSVs. It is gitignored.
- `cleaning/processed_xlsx/`: This directory will be automatically created by the third script and will store Excel versions of the files from `cleaning/processed/`. It is gitignored.

## Workflow Steps

Follow these steps to process your data:

### Step 1: Convert Excel to CSV

This step converts all `.xlsx` files from the `raw_excel_files/` directory into `.csv` files and saves them in the `data_csv/` directory.

1.  **Navigate to the `cleaning` directory** in your terminal:
    ```bash
    cd cleaning
    ```
2.  **Run the first script**:
    ```bash
    python 01_convert_excel_to_csv.py
    ```
    The script will inform you if the `raw_excel_files/` directory doesn't exist or if no Excel files are found. It will create the `data_csv/` directory if it doesn't exist.

### Step 2: Clean and Unify CSV Data

This step takes the `.csv` files from `data_csv/`, cleans them according to predefined rules (including skipping `promo_wide.csv`), and merges them into a single `unified_dataset_base.csv` file. This final file will be saved in the `cleaning/processed/` directory.

1.  **Ensure you are still in the `cleaning` directory** in your terminal.
2.  **Run the second script**:
    ```bash
    python 02_clean_unified_data.py
    ```
    This script will output progress messages and confirm the creation of `cleaning/processed/unified_dataset_base.csv` and other cleaned CSV files in `cleaning/processed/`.

### Step 3: (Optional) Export Processed CSVs to Excel

This step converts all `.csv` files from the `cleaning/processed/` directory into `.xlsx` files and saves them in the `cleaning/processed_xlsx/` directory.

1.  **Ensure you are still in the `cleaning` directory** in your terminal.
2.  **Run the third script**:
    ```bash
    python 03_export_to_excel.py
    ```
    This script will create the `cleaning/processed_xlsx/` directory if it doesn't exist and convert all CSVs found in `cleaning/processed/`.

## Output

- Intermediate CSV files: `data_csv/<filename>.csv`
- Cleaned CSV files: `cleaning/processed/<filename>_cleaned.csv`
- Final cleaned and unified dataset (CSV): `cleaning/processed/unified_dataset_base.csv`
- Excel versions of processed files: `cleaning/processed_xlsx/<filename>.xlsx`

## Notes

- The `promo_wide`