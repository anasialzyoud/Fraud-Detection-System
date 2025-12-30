# Copy this entire cell and paste it in a new Notebook cell
# This version works without encoding issues

import joblib
import numpy as np
import pandas as pd
import os

print("=" * 60)
print("Starting model saving process...")
print("=" * 60)

# Check for required variables
required_vars = {
    'preprocessor': 'preprocessor',
    'lr': 'lr',
    'rf': 'rf',
    'hgb': 'hgb',
    'w_lr': 'w_lr',
    'w_rf': 'w_rf',
    'w_hgb': 'w_hgb',
    't_ens': 't_ens',
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
    
    # Save models
    joblib.dump(lr, 'logistic_regression.pkl')
    print("[OK] Logistic Regression saved (logistic_regression.pkl)")
    
    joblib.dump(rf, 'random_forest.pkl')
    print("[OK] Random Forest saved (random_forest.pkl)")
    
    joblib.dump(hgb, 'hist_gradient_boosting.pkl')
    print("[OK] HistGradientBoosting saved (hist_gradient_boosting.pkl)")
    
    # Save Ensemble configuration
    ensemble_config = {
        'w_lr': w_lr,
        'w_rf': w_rf,
        'w_hgb': w_hgb,
        'threshold': t_ens
    }
    joblib.dump(ensemble_config, 'ensemble_config.pkl')
    print("[OK] Ensemble config saved (ensemble_config.pkl)")
    
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
    
    # Save active features list
    joblib.dump(ACTIVE_FEATURES, 'active_features.pkl')
    print("[OK] Active features saved (active_features.pkl)")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All files saved successfully!")
    print("=" * 60)
    
    # Show file sizes
    print("\n[INFO] Saved file sizes:")
    files = [
        'preprocessor.pkl',
        'logistic_regression.pkl',
        'random_forest.pkl',
        'hist_gradient_boosting.pkl',
        'ensemble_config.pkl',
        'test_data.pkl',
        'active_features.pkl'
    ]
    
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

