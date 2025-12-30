# Copy this code into a NEW cell in your Notebook BEFORE the cell that loads the data
# This will automatically find or set the correct data path

import os

# Try to find the CSV file in current directory or common locations
possible_paths = [
    "PS_20174392719_1491204439457_log.csv",  # Same directory
    "./PS_20174392719_1491204439457_log.csv",  # Current directory
    "../PS_20174392719_1491204439457_log.csv",  # Parent directory
    "data/PS_20174392719_1491204439457_log.csv",  # data subfolder
    "/kaggle/input/paysim1/PS_20174392719_1491204439457_log.csv"  # Kaggle path
]

DATA_PATH = None
for path in possible_paths:
    if os.path.exists(path):
        DATA_PATH = path
        print(f"[OK] Found data file at: {DATA_PATH}")
        break

if DATA_PATH is None:
    print("[ERROR] Data file not found!")
    print("\nPlease do one of the following:")
    print("1. Download the dataset from Kaggle:")
    print("   https://www.kaggle.com/datasets/ealaxi/paysim1")
    print("2. Place the file 'PS_20174392719_1491204439457_log.csv' in the current directory")
    print("3. Or update DATA_PATH manually with the correct path")
    print("\nCurrent directory:", os.getcwd())
    raise FileNotFoundError("Data file not found. Please download and place the CSV file.")

# Now you can use DATA_PATH in the next cell
print(f"\n[SUCCESS] Using data path: {DATA_PATH}")

