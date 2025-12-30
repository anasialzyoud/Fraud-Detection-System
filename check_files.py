"""
سكريبت للتحقق من وجود جميع الملفات المطلوبة
"""
import os
import sys

# إعداد الترميز للنصوص العربية
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

required_files = [
    'preprocessor.pkl',
    'logistic_regression.pkl',
    'random_forest.pkl',
    'hist_gradient_boosting.pkl',
    'ensemble_config.pkl',
    'test_data.pkl',
    'active_features.pkl'
]

print("=" * 50)
print("Checking required files...")
print("=" * 50)

missing_files = []
existing_files = []

for file in required_files:
    if os.path.exists(file):
        print(f"[OK] {file} - Found")
        existing_files.append(file)
    else:
        print(f"[X]  {file} - Missing")
        missing_files.append(file)

print("\n" + "=" * 50)
if missing_files:
    print(f"[ERROR] Missing files: {len(missing_files)}")
    print("\nMissing files:")
    for file in missing_files:
        print(f"  - {file}")
    print("\n[WARNING] Please run save_models.py from Notebook first")
else:
    print("[SUCCESS] All files found! You can run Streamlit now")
    print("\nNext step: streamlit run app.py")
print("=" * 50)

