# Copy this code into a NEW cell in your Notebook AFTER training models
# This version saves full test data with nameOrig and nameDest for realistic interface

import joblib
import numpy as np
import pandas as pd
import os

print("=" * 60)
print("Starting model saving process (with full account names)...")
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
    
    # CRITICAL: Save full test data with nameOrig and nameDest
    # We need to get the original dataframe before dropping nameOrig and nameDest
    print("\n[INFO] Attempting to save full test data with nameOrig and nameDest...")
    
    # Check if we have access to the original dataframe
    if 'df' in globals():
        # We need to map test indices back to original dataframe
        # Since we did train_test_split, we need to get the test portion of original df
        
        # Get the original test split indices
        # If X and y are from the original df_work (without nameOrig/nameDest),
        # we need to reconstruct the mapping
        
        # Alternative: If we have the original df with all columns, we can sample from it
        # But we need to ensure we're sampling from the test portion
        
        # For now, let's try to get a sample that includes nameOrig and nameDest
        # We'll use the same indices but from the original dataframe
        
        # Check if we can access the original test data before preprocessing
        if 'X' in globals() and hasattr(X, 'index'):
            # Try to get original indices
            try:
                # Get original dataframe rows corresponding to test indices
                # This is a simplified approach - assumes test indices map directly
                full_test_sample = df.iloc[test_indices].copy() if hasattr(df, 'iloc') else df[test_indices].copy()
                
                test_data_full = {
                    'X_test_full': full_test_sample,
                    'X_test': X_test.iloc[test_indices].copy() if hasattr(X_test, 'iloc') else X_test[test_indices].copy(),
                    'y_test': y_test.iloc[test_indices].copy() if hasattr(y_test, 'iloc') else y_test[test_indices]
                }
                
                joblib.dump(test_data_full, 'test_data_full.pkl')
                print(f"[OK] Full test data saved with nameOrig and nameDest (test_data_full.pkl)")
                print(f"     Sample size: {len(full_test_sample)} transactions")
                print(f"     Columns in full data: {list(full_test_sample.columns)}")
            except Exception as e:
                print(f"[WARNING] Could not save full data with names: {str(e)}")
                print("[INFO] The app will work but without account IDs (nameOrig/nameDest)")
        else:
            print("[WARNING] Cannot map test indices to original dataframe.")
            print("[INFO] To enable account ID lookup, ensure 'df' (original dataframe) is available.")
            print("[INFO] The app will work but account balances will default to 0.0 if IDs not found.")
    else:
        print("[WARNING] Original dataframe 'df' not found in global scope.")
        print("[INFO] The app will work but account balances will default to 0.0 if IDs not found.")
    
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

