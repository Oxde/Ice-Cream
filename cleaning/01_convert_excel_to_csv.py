import pandas as pd
import os
import glob

def convert_excel_to_csv():
    """
    Converts all .xlsx files from a specified input directory to .csv format
    in a specified output directory.
    """
    raw_data_dir = '../raw_excel_files'
    csv_output_dir = '../data_csv'

    # Create input directory if it doesn't exist, and prompt user
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)
        print(f"Created directory: {raw_data_dir}")
        print(f"Please place your raw .xlsx files in the '{raw_data_dir}' directory.")
        return

    # Create output CSV directory if it doesn't exist
    os.makedirs(csv_output_dir, exist_ok=True)

    excel_files = glob.glob(os.path.join(raw_data_dir, '*.xlsx'))

    if not excel_files:
        print(f"No .xlsx files found in {raw_data_dir}. Please add your Excel files there.")
        return

    print(f"Found {len(excel_files)} Excel files to convert.")

    for file_path in excel_files:
        try:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            df = pd.read_excel(file_path)
            
            csv_path = os.path.join(csv_output_dir, f"{base_name}.csv")
            df.to_csv(csv_path, index=False)
            
            print(f"Converted: {file_path} → {csv_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    convert_excel_to_csv()
    print("\nExcel to CSV conversion complete.")
    print(f"CSV files are saved in '{os.path.abspath('../data_csv')}'") 