import pandas as pd
import os
import glob

def export_processed_csv_to_excel():
    """
    Converts all .csv files from the 'processed/' directory to .xlsx format
    and saves them in the 'processed_xlsx/' directory.
    Both directories are expected to be relative to this script's location (i.e., within 'cleaning/').
    """
    input_csv_dir = 'processed/'
    output_excel_dir = 'processed_xlsx/'

    # Ensure input directory exists
    if not os.path.exists(input_csv_dir):
        print(f"Error: Input directory '{input_csv_dir}' not found.")
        print(f"Please ensure '02_clean_unified_data.py' has been run successfully to generate CSV files in '{input_csv_dir}'.")
        return

    # Create output Excel directory if it doesn't exist
    os.makedirs(output_excel_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(input_csv_dir, '*.csv'))

    if not csv_files:
        print(f"No .csv files found in '{input_csv_dir}'.")
        return

    print(f"Found {len(csv_files)} CSV files in '{input_csv_dir}' to convert to Excel.")
    success_count = 0

    for csv_file_path in csv_files:
        try:
            base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
            df = pd.read_csv(csv_file_path)
            
            excel_file_path = os.path.join(output_excel_dir, f"{base_name}.xlsx")
            df.to_excel(excel_file_path, index=False)
            
            print(f"Converted: {csv_file_path} → {excel_file_path}")
            success_count += 1
        except Exception as e:
            print(f"Error processing {csv_file_path}: {e}")

    if success_count > 0:
        print(f"\nSuccessfully converted {success_count} CSV file(s) to Excel format in '{os.path.abspath(output_excel_dir)}'.")
    else:
        print("\nNo files were converted.")

if __name__ == "__main__":
    print("Starting conversion of processed CSV files to Excel...")
    export_processed_csv_to_excel()
    print("\nExcel export process complete.") 