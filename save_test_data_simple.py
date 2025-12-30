# Copy this ENTIRE code into a NEW cell in your Notebook
# This script saves the full test dataset with nameOrig and nameDest

import joblib
import numpy as np
import pandas as pd
import os

print("=" * 60)
print("Saving Full Test Dataset with Account Names")
print("=" * 60)

# Check required variables
if 'df' not in globals():
    print("\n[ERROR] Original dataframe 'df' not found!")
    print("[INFO] Please make sure you have run the data loading cell.")
    raise NameError("Dataframe 'df' is required")

print("\n[OK] Original dataframe found\n")

try:
    # Sample size (same as in save_models.py)
    np.random.seed(42)  # For reproducibility
    test_sample_size = min(10000, len(df))
    
    print(f"[INFO] Sampling {test_sample_size} transactions from original dataframe...")
    
    # Get fraud rate to maintain distribution
    if 'isFraud' in df.columns:
        fraud_rate = df['isFraud'].mean()
        print(f"[INFO] Fraud rate in dataset: {fraud_rate:.4f}")
        
        # Sample maintaining fraud distribution
        fraud_indices = df[df['isFraud'] == 1].index.values
        normal_indices = df[df['isFraud'] == 0].index.values
        
        n_fraud = int(test_sample_size * fraud_rate)
        n_normal = test_sample_size - n_fraud
        
        # Ensure we don't exceed available samples
        n_fraud = min(n_fraud, len(fraud_indices))
        n_normal = min(n_normal, len(normal_indices))
        
        fraud_sample = np.random.choice(fraud_indices, n_fraud, replace=False)
        normal_sample = np.random.choice(normal_indices, n_normal, replace=False)
        
        combined_indices = np.concatenate([fraud_sample, normal_sample])
        np.random.shuffle(combined_indices)
        
        test_data_full = df.loc[combined_indices].copy()
    else:
        # If no isFraud column, just random sample
        sample_indices = np.random.choice(len(df), test_sample_size, replace=False)
        test_data_full = df.iloc[sample_indices].copy()
    
    # Save as CSV
    csv_filename = 'test_data_full.csv'
    test_data_full.to_csv(csv_filename, index=False)
    print(f"\n[OK] Full test data saved as CSV: {csv_filename}")
    print(f"     Rows: {len(test_data_full)}")
    print(f"     Columns: {list(test_data_full.columns)}")
    
    # Show sample of data
    print(f"\n[INFO] Sample of saved data:")
    print(test_data_full[['nameOrig', 'nameDest', 'type', 'amount', 'isFraud']].head())
    
    # Also save as pickle for faster loading
    pickle_filename = 'test_data_full.pkl'
    
    # Create X_test equivalent (drop nameOrig, nameDest, isFraud for model input)
    X_test_sample = test_data_full.drop(columns=['nameOrig', 'nameDest', 'isFraud'], errors='ignore').copy()
    y_test_sample = test_data_full['isFraud'].copy()
    
    # Apply same feature engineering as in preprocessing
    if 'hour' not in X_test_sample.columns and 'step' in X_test_sample.columns:
        X_test_sample['hour'] = (X_test_sample['step'] % 24).astype(np.int16)
        X_test_sample['day'] = (X_test_sample['step'] // 24).astype(np.int16)
    
    if 'orig_delta' not in X_test_sample.columns:
        X_test_sample['orig_delta'] = X_test_sample['oldbalanceOrg'] - X_test_sample['newbalanceOrig']
        X_test_sample['dest_delta'] = X_test_sample['newbalanceDest'] - X_test_sample['oldbalanceDest']
        X_test_sample['balance_error_orig'] = X_test_sample['orig_delta'] - X_test_sample['amount']
        X_test_sample['balance_error_dest'] = X_test_sample['dest_delta'] - X_test_sample['amount']
        
        # Clean floating-point noise
        epsilon = 1e-6
        X_test_sample['balance_error_orig'] = X_test_sample['balance_error_orig'].where(
            X_test_sample['balance_error_orig'].abs() > epsilon, 0.0
        )
        X_test_sample['balance_error_dest'] = X_test_sample['balance_error_dest'].where(
            X_test_sample['balance_error_dest'].abs() > epsilon, 0.0
        )
    
    if 'isFlaggedFraud' not in X_test_sample.columns:
        X_test_sample['isFlaggedFraud'] = 0  # Default value
    
    joblib.dump({
        'X_test_full': test_data_full,
        'X_test': X_test_sample,
        'y_test': y_test_sample
    }, pickle_filename)
    
    print(f"\n[OK] Full test data saved as pickle: {pickle_filename}")
    
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
    print("\n[INFO] Files created:")
    print("   ✅ test_data_full.csv (can be opened in Excel)")
    print("   ✅ test_data_full.pkl (for fast loading in Streamlit)")
    
except Exception as e:
    print(f"\n[ERROR] An error occurred: {str(e)}")
    import traceback
    print("\n[DEBUG] Full traceback:")
    traceback.print_exc()
    raise
