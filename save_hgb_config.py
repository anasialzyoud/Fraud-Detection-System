"""
سكريبت لحفظ threshold لنموذج HistGradientBoosting
يجب تشغيل هذا السكريبت من داخل notebook بعد حساب threshold الأمثل
"""

import joblib

# قيمة threshold المحسوبة من validation set (من Notebook)
# تم حسابه من: best_hgb_f1 = thr_hgb.loc[thr_hgb["f1"].idxmax()]
# القيمة: 0.220886
hgb_threshold = 0.22  # يمكن تحديثها بالقيمة الدقيقة إذا كانت مختلفة

# حفظ config HGB
hgb_config = {
    'threshold': hgb_threshold,
    'model_type': 'HistGradientBoosting'
}

joblib.dump(hgb_config, 'hgb_config.pkl')
print("[OK] Saved HistGradientBoosting config (hgb_config.pkl)")
print(f"  Threshold: {hgb_threshold}")

