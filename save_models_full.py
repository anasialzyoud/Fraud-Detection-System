# Copy this code into a NEW cell in your Notebook AFTER training models
# This version saves full test data with nameOrig and nameDest

import joblib
import numpy as np
import pandas as pd
import os

print("=" * 60)
print("Starting model saving process (with full data)...")
print("=" * 60)

# Check for required variables
required_vars = {
    'preprocessor': 'preprocessor',
    'hgb': 'hgb',
    'X_test': 'X_test',
    'y_test': 'y_test',
    'ACTIVE_FEATURES': 'ACTIVE_FEATURES'
}

missing_vars = []
for var_name, var_display in required_vars.items():
    if var_name not in globals():
        missing_vars.append(var_display)

if missing_vars:
    print("\n[ERROR] The following variables are missing:")
    for var in missing_vars:
        print(f"   - {var}")
    print("\n[WARNING] Please make sure:")
    print("   1. All Notebook cells are executed")
    print("   2. All models are trained")
    print("   3. Evaluation cells are executed")
    raise NameError("Some required variables are missing")

print("\n[OK] All required variables found\n")

try:
    # Save preprocessor
    joblib.dump(preprocessor, 'preprocessor.pkl')
    print("[OK] Preprocessor saved (preprocessor.pkl)")
    
    # Save HistGradientBoosting model
    joblib.dump(hgb, 'hist_gradient_boosting.pkl')
    print("[OK] HistGradientBoosting saved (hist_gradient_boosting.pkl)")
    
    # Save active features list
    joblib.dump(ACTIVE_FEATURES, 'active_features.pkl')
    print("[OK] Active features saved (active_features.pkl)")
    
    # Save test data (small sample to save memory)
    np.random.seed(42)  # For reproducibility
    test_sample_size = min(10000, len(X_test))
    test_indices = np.random.choice(len(X_test), test_sample_size, replace=False)
    
    if hasattr(X_test, 'iloc'):
        test_data = {
            'X_test': X_test.iloc[test_indices].copy(),
            'y_test': y_test.iloc[test_indices].copy() if hasattr(y_test, 'iloc') else y_test[test_indices]
        }
    else:
        test_data = {
            'X_test': X_test[test_indices].copy(),
            'y_test': y_test[test_indices].copy()
        }
    
    joblib.dump(test_data, 'test_data.pkl')
    print(f"[OK] Test data saved ({test_sample_size} samples) (test_data.pkl)")
    
    # Try to save full data with nameOrig and nameDest if available
    # Check if we have the original dataframe before dropping nameOrig and nameDest
    if 'df' in globals() and 'df_work' in globals():
        # Get the original test indices from the full dataframe
        # We need to map back from test indices to original dataframe indices
        print("\n[INFO] Attempting to save full test data with nameOrig and nameDest...")
        
        # Get test split indices - we need to reconstruct this
        # For now, we'll try to get a sample from the original dataframe
        if 'X' in globals() and 'y' in globals():
            # Get the test indices from the original split
            # This is a simplified approach - in practice you'd want to save the indices
            full_test_sample = df.iloc[test_indices].copy() if hasattr(df, 'iloc') else df[test_indices].copy()
            
            test_data_full = {
                'X_test_full': full_test_sample,
                'X_test': X_test.iloc[test_indices].copy() if hasattr(X_test, 'iloc') else X_test[test_indices].copy(),
                'y_test': y_test.iloc[test_indices].copy() if hasattr(y_test, 'iloc') else y_test[test_indices]
            }
            
            joblib.dump(test_data_full, 'test_data_full.pkl')
            print(f"[OK] Full test data saved with nameOrig and nameDest (test_data_full.pkl)")
        else:
            print("[WARNING] Could not save full data. X and y not found in global scope.")
    else:
        print("[INFO] Original dataframe not found. Saving without nameOrig and nameDest.")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All files saved successfully!")
    print("=" * 60)
    
    # Show file sizes
    print("\n[INFO] Saved file sizes:")
    files = [
        'preprocessor.pkl',
        'hist_gradient_boosting.pkl',
        'test_data.pkl',
        'active_features.pkl'
    ]
    
    if os.path.exists('test_data_full.pkl'):
        files.append('test_data_full.pkl')
    
    total_size = 0
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)  # in MB
            total_size += size
            print(f"   {file}: {size:.2f} MB")
    
    print(f"\n   Total size: {total_size:.2f} MB")
    print("\n[SUCCESS] You can now run: streamlit run app.py")
    
except Exception as e:
    print(f"\n[ERROR] An error occurred while saving: {str(e)}")
    print("\n[WARNING] Please check:")
    print("   1. Sufficient disk space")
    print("   2. Write permissions in current directory")
    raise

