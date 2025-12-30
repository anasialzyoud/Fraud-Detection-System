"""
سكريبت لحفظ النماذج والمعالج المسبق وبيانات الاختبار
يجب تشغيل هذا السكريبت من داخل notebook بعد تدريب النماذج
"""

import joblib
import numpy as np
import pandas as pd
import os

print("=" * 60)
print("بدء عملية حفظ النماذج والبيانات...")
print("=" * 60)

# التحقق من وجود المتغيرات المطلوبة
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
    print("\n❌ خطأ: المتغيرات التالية غير موجودة:")
    for var in missing_vars:
        print(f"   - {var}")
    print("\n⚠️ يرجى التأكد من:")
    print("   1. تشغيل جميع خلايا Notebook")
    print("   2. تدريب جميع النماذج")
    print("   3. تشغيل خلايا التقييم")
    raise NameError("بعض المتغيرات المطلوبة غير موجودة")

print("\n✓ تم التحقق من وجود جميع المتغيرات المطلوبة\n")

try:
    # حفظ المعالج المسبق
    joblib.dump(preprocessor, 'preprocessor.pkl')
    print("✓ تم حفظ المعالج المسبق (preprocessor.pkl)")
    
    # حفظ النماذج
    joblib.dump(lr, 'logistic_regression.pkl')
    print("✓ تم حفظ Logistic Regression (logistic_regression.pkl)")
    
    joblib.dump(rf, 'random_forest.pkl')
    print("✓ تم حفظ Random Forest (random_forest.pkl)")
    
    joblib.dump(hgb, 'hist_gradient_boosting.pkl')
    print("✓ تم حفظ HistGradientBoosting (hist_gradient_boosting.pkl)")
    
    # حفظ الأوزان والعتبة للـ Ensemble
    ensemble_config = {
        'w_lr': w_lr,
        'w_rf': w_rf,
        'w_hgb': w_hgb,
        'threshold': t_ens
    }
    joblib.dump(ensemble_config, 'ensemble_config.pkl')
    print("✓ تم حفظ إعدادات Ensemble (ensemble_config.pkl)")
    
    # حفظ بيانات الاختبار (عينة صغيرة لتوفير الذاكرة)
    # حفظ 10000 عينة عشوائية من بيانات الاختبار
    np.random.seed(42)  # للتكرار
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
    print(f"✓ تم حفظ {test_sample_size} عينة من بيانات الاختبار (test_data.pkl)")
    
    # حفظ قائمة الميزات النشطة
    joblib.dump(ACTIVE_FEATURES, 'active_features.pkl')
    print("✓ تم حفظ قائمة الميزات النشطة (active_features.pkl)")
    
    print("\n" + "=" * 60)
    print("✅ تم حفظ جميع الملفات بنجاح!")
    print("=" * 60)
    
    # عرض حجم الملفات
    print("\n📊 حجم الملفات المحفوظة:")
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
            size = os.path.getsize(file) / (1024 * 1024)  # بالميجابايت
            total_size += size
            print(f"   {file}: {size:.2f} MB")
    
    print(f"\n   إجمالي الحجم: {total_size:.2f} MB")
    print("\n✅ يمكنك الآن تشغيل: streamlit run app.py")
    
except Exception as e:
    print(f"\n❌ حدث خطأ أثناء الحفظ: {str(e)}")
    print("\n⚠️ يرجى التحقق من:")
    print("   1. وجود مساحة كافية على القرص")
    print("   2. صلاحيات الكتابة في المجلد الحالي")
    raise

