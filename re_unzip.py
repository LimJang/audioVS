import zipfile
import os

zip_path = 'compiler.zip'
extract_path = 'compiler'

if os.path.exists(zip_path):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")
else:
    print("compiler.zip not found. Please make sure it exists in the project root.")
