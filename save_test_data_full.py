# Copy this code into a NEW cell in your Notebook AFTER training models
# This script saves the full test dataset with all columns including nameOrig and nameDest

import joblib
import numpy as np
import pandas as pd
import os

print("=" * 60)
print("Saving Full Test Dataset with Account Names")
print("=" * 60)

# Check for required variables
required_vars = ['X_test', 'y_test', 'df']

missing_vars = []
for var_name in required_vars:
    if var_name not in globals():
        missing_vars.append(var_name)

if missing_vars:
    print("\n[ERROR] The following variables are missing:")
    for var in missing_vars:
        print(f"   - {var}")
    print("\n[WARNING] Please make sure:")
    print("   1. All Notebook cells are executed")
    print("   2. Data split is completed (X_test, y_test exist)")
    print("   3. Original dataframe 'df' is available")
    raise NameError("Required variables are missing")

print("\n[OK] All required variables found\n")

try:
    # Get test sample size
    np.random.seed(42)  # For reproducibility
    test_sample_size = min(10000, len(X_test))
    test_indices = np.random.choice(len(X_test), test_sample_size, replace=False)
    
    print(f"[INFO] Sampling {test_sample_size} transactions from test set...")
    
    # Method 1: Try to get full data from original dataframe
    # We need to map test indices back to original dataframe indices
    if 'df' in globals():
        print("[INFO] Attempting to extract full data from original dataframe...")
        
        # Check if we can map indices
        # If X_test is a DataFrame with index, we can use it
        if hasattr(X_test, 'index'):
            try:
                # Get the original indices from X_test
                original_indices = X_test.index[test_indices] if hasattr(X_test.index, '__getitem__') else test_indices
                
                # Get full data from original dataframe
                if hasattr(df, 'iloc'):
                    test_data_full = df.iloc[original_indices].copy()
                else:
                    test_data_full = df.loc[original_indices].copy()
                
                # Add target column
                if hasattr(y_test, 'iloc'):
                    test_data_full['isFraud'] = y_test.iloc[test_indices].values
                else:
                    test_data_full['isFraud'] = y_test[test_indices]
                
                # Save as CSV
                csv_filename = 'test_data_full.csv'
                test_data_full.to_csv(csv_filename, index=False)
                print(f"[OK] Full test data saved as CSV: {csv_filename}")
                print(f"     Rows: {len(test_data_full)}")
                print(f"     Columns: {list(test_data_full.columns)}")
                
                # Also save as pickle for faster loading
                pickle_filename = 'test_data_full.pkl'
                joblib.dump({
                    'X_test_full': test_data_full,
                    'X_test': X_test.iloc[test_indices].copy() if hasattr(X_test, 'iloc') else X_test[test_indices].copy(),
                    'y_test': y_test.iloc[test_indices].copy() if hasattr(y_test, 'iloc') else y_test[test_indices]
                }, pickle_filename)
                print(f"[OK] Full test data saved as pickle: {pickle_filename}")
                
            except Exception as e:
                print(f"[WARNING] Could not map indices directly: {str(e)}")
                print("[INFO] Trying alternative method...")
                
                # Alternative: Sample directly from original dataframe
                if len(df) >= test_sample_size:
                    # Sample random rows from original dataframe
                    sample_indices = np.random.choice(len(df), test_sample_size, replace=False)
                    test_data_full = df.iloc[sample_indices].copy() if hasattr(df, 'iloc') else df.loc[sample_indices].copy()
                    
                    # Save as CSV
                    csv_filename = 'test_data_full.csv'
                    test_data_full.to_csv(csv_filename, index=False)
                    print(f"[OK] Full test data saved as CSV (random sample): {csv_filename}")
                    print(f"     Rows: {len(test_data_full)}")
                    print(f"     Columns: {list(test_data_full.columns)}")
                    
                    # Save as pickle
                    pickle_filename = 'test_data_full.pkl'
                    # We need to create X_test equivalent for this sample
                    # Drop nameOrig and nameDest to match X_test structure
                    X_test_sample = test_data_full.drop(columns=['nameOrig', 'nameDest', 'isFraud']).copy()
                    y_test_sample = test_data_full['isFraud'].copy()
                    
                    joblib.dump({
                        'X_test_full': test_data_full,
                        'X_test': X_test_sample,
                        'y_test': y_test_sample
                    }, pickle_filename)
                    print(f"[OK] Full test data saved as pickle: {pickle_filename}")
                else:
                    print("[ERROR] Original dataframe is smaller than required sample size")
                    raise ValueError("Cannot create test sample")
        else:
            print("[WARNING] X_test does not have index attribute")
            raise ValueError("Cannot map test indices")
    else:
        print("[ERROR] Original dataframe 'df' not found")
        print("[INFO] Cannot save full data with nameOrig and nameDest")
        raise NameError("Original dataframe 'df' is required")
    
    # Show file sizes
    print("\n" + "=" * 60)
    print("[SUCCESS] Test data files saved successfully!")
    print("=" * 60)
    
    print("\n[INFO] Saved file sizes:")
    files = ['test_data_full.csv', 'test_data_full.pkl']
    
    total_size = 0
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)  # in MB
            total_size += size
            print(f"   {file}: {size:.2f} MB")
    
    print(f"\n   Total size: {total_size:.2f} MB")
    print("\n[SUCCESS] Test data is ready for use in Streamlit app!")
    
except Exception as e:
    print(f"\n[ERROR] An error occurred: {str(e)}")
    import traceback
    print("\n[DEBUG] Full traceback:")
    traceback.print_exc()
    raise

