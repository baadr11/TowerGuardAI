"""
validation.py — TowerGuard Comprehensive Validation
======================================================
Produces:
  1. Per-class Precision, Recall, F1
  2. Confusion Matrix
  3. Feature Importance + data-leakage guard
  4. Temporal-overlap check for geographic split
  5. rssi_prior ablation study
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
)

from .config import (
    MODEL_CONFIG, SEVERITY_CONFIG, ARABIC_LABELS,
    TOWERGUARD_FEATURES, TARGET_COL,
)

log = logging.getLogger("towerguard.validation")


# ══════════════════════════════════════════════════════════════════════════════
#  التقرير الشامل بالعربية
# ══════════════════════════════════════════════════════════════════════════════

def generate_validation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    feature_importances: Optional[Dict[str, float]] = None,
    split_type: str = "جغرافي (تبوك + عسير)",
) -> Dict[str, Any]:
    """
    يُنتج تقرير تحقق شامل بالعربية.

    Parameters
    ----------
    y_true : القيم الحقيقية
    y_pred : القيم المتوقعة
    y_proba : احتمالات التنبؤ (اختياري)
    feature_importances : أهمية المؤشرات (اختياري)
    split_type : نوع التقسيم

    Returns
    -------
    تقرير شامل (dict)
    """
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    is_multiclass = len(labels) > 2
    avg = "macro" if is_multiclass else "binary"

    report = {
        "title": "TowerGuard Comprehensive Validation Report",
        "نوع_التقسيم": split_type,
        "عدد_العينات": int(len(y_true)),
        "عدد_الفئات": int(len(labels)),
    }

    # ══════════════════════════════════════════════════════════════════════
    #  1. المقاييس الإجمالية
    # ══════════════════════════════════════════════════════════════════════

    report["المقاييس_الإجمالية"] = {
        ARABIC_LABELS["accuracy"]: round(accuracy_score(y_true, y_pred), 4),
        ARABIC_LABELS["precision"]: round(precision_score(y_true, y_pred, average=avg, zero_division=0), 4),
        ARABIC_LABELS["recall"]: round(recall_score(y_true, y_pred, average=avg, zero_division=0), 4),
        ARABIC_LABELS["f1_score"]: round(f1_score(y_true, y_pred, average=avg, zero_division=0), 4),
    }

    # ══════════════════════════════════════════════════════════════════════
    #  2. المقاييس لكل فئة
    # ══════════════════════════════════════════════════════════════════════

    per_class = {}
    prec_per = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    rec_per = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    for i, label in enumerate(labels):
        label_name = SEVERITY_CONFIG.SEVERITY_LABELS_AR.get(int(label), str(label))
        n_actual = int((y_true == label).sum())
        n_predicted = int((y_pred == label).sum())

        per_class[f"المستوى {int(label)} ({label_name})"] = {
            "العدد_الفعلي": n_actual,
            "العدد_المتوقع": n_predicted,
            ARABIC_LABELS["precision"]: round(float(prec_per[i]), 4),
            ARABIC_LABELS["recall"]: round(float(rec_per[i]), 4),
            ARABIC_LABELS["f1_score"]: round(float(f1_per[i]), 4),
        }

    report["المقاييس_لكل_فئة"] = per_class

    # ══════════════════════════════════════════════════════════════════════
    #  3. مصفوفة الارتباك
    # ══════════════════════════════════════════════════════════════════════

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_dict = {}
    for i, actual_label in enumerate(labels):
        actual_name = SEVERITY_CONFIG.SEVERITY_LABELS_AR.get(int(actual_label), str(actual_label))
        row = {}
        for j, pred_label in enumerate(labels):
            pred_name = SEVERITY_CONFIG.SEVERITY_LABELS_AR.get(int(pred_label), str(pred_label))
            row[f"متوقع: {pred_name}"] = int(cm[i][j])
        cm_dict[f"فعلي: {actual_name}"] = row

    report[ARABIC_LABELS["confusion_matrix"]] = cm_dict
    report["مصفوفة_الارتباك_خام"] = cm.tolist()

    # ══════════════════════════════════════════════════════════════════════
    #  4. أهمية المؤشرات + فحص Data Leakage
    # ══════════════════════════════════════════════════════════════════════

    if feature_importances:
        sorted_features = sorted(feature_importances.items(), key=lambda x: -x[1])
        fi_report = {}
        for feat, imp in sorted_features:
            fi_report[feat] = f"{imp * 100:.1f}%"
        report[ARABIC_LABELS["feature_importance"]] = fi_report

        top_feat, top_imp = sorted_features[0]
        if top_imp > MODEL_CONFIG.max_top_feature_importance:
            report["تحذير_data_leakage"] = (
                f"⚠️ المؤشر '{top_feat}' يسيطر على {top_imp * 100:.1f}% من الأهمية — "
                f"يتجاوز الحد الأقصى ({MODEL_CONFIG.max_top_feature_importance * 100:.0f}%). "
                f"يُرجى فحص data leakage!"
            )

    # ══════════════════════════════════════════════════════════════════════
    #  5. فحوصات القبول
    # ══════════════════════════════════════════════════════════════════════

    checks = {}
    overall_recall = recall_score(y_true, y_pred, average=avg, zero_division=0)

    checks["الاستدعاء_≥_0.70"] = {
        "القيمة": round(overall_recall, 4),
        "الحد": MODEL_CONFIG.min_recall,
        "النتيجة": "✅ اجتاز" if overall_recall >= MODEL_CONFIG.min_recall else "❌ فشل",
    }

    if is_multiclass and len(labels) >= 3:
        critical_idx = list(labels).index(2) if 2 in labels else -1
        if critical_idx >= 0:
            critical_recall = float(rec_per[critical_idx])
            checks["استدعاء_الحرج_≥_0.70"] = {
                "القيمة": round(critical_recall, 4),
                "الحد": MODEL_CONFIG.min_recall,
                "النتيجة": "✅ اجتاز" if critical_recall >= MODEL_CONFIG.min_recall else "❌ فشل — النموذج خطير!",
            }

    report["فحوصات_القبول"] = checks

    # ══════════════════════════════════════════════════════════════════════
    #  rssi_prior dominance check
    # ══════════════════════════════════════════════════════════════════════

    if feature_importances and "rssi_prior" in feature_importances:
        rssi_prior_imp = feature_importances["rssi_prior"]
        if rssi_prior_imp > 0.35:
            report["تحذير_rssi_prior"] = (
                f"⚠️ المؤشر 'rssi_prior' يسيطر على {rssi_prior_imp * 100:.1f}% من الأهمية — "
                f"يتجاوز 35%. هذا ليس تسرب بيانات، لكنه استغلال للارتباط الذاتي الزمني (AR(1)). "
                f"يُنصح بتنفيذ دراسة الاستئصال (ablation study) — أعد التدريب بدون rssi_prior."
            )

    return report


def print_validation_report(report: Dict[str, Any]):
    """يطبع التقرير بتنسيق مقروء."""
    print("\n" + "=" * 70)
    print(f"  📋 {report.get('title', report.get('عنوان', 'Validation Report'))}")
    print("=" * 70)
    print(f"  نوع التقسيم: {report.get('نوع_التقسيم', '—')}")
    print(f"  عدد العينات: {report.get('عدد_العينات', '—')}")
    print(f"  عدد الفئات: {report.get('عدد_الفئات', '—')}")

    # المقاييس الإجمالية
    print("\n  ── المقاييس الإجمالية ──")
    for k, v in report.get("المقاييس_الإجمالية", {}).items():
        print(f"    {k}: {v}")

    # لكل فئة
    print("\n  ── المقاييس لكل فئة ──")
    for cls, metrics in report.get("المقاييس_لكل_فئة", {}).items():
        print(f"    {cls}:")
        for k, v in metrics.items():
            print(f"      {k}: {v}")

    # مصفوفة الارتباك
    print(f"\n  ── {ARABIC_LABELS['confusion_matrix']} ──")
    cm_raw = report.get("مصفوفة_الارتباك_خام", [])
    if cm_raw:
        cm_arr = np.array(cm_raw)
        labels_ar = [SEVERITY_CONFIG.SEVERITY_LABELS_AR.get(i, str(i)) for i in range(len(cm_arr))]

        # Header
        header = "        " + "  ".join(f"{l:>8}" for l in labels_ar)
        print(f"    {header}")
        for i, row in enumerate(cm_arr):
            row_str = "  ".join(f"{int(v):>8}" for v in row)
            print(f"    {labels_ar[i]:>8}  {row_str}")

    # أهمية المؤشرات
    fi = report.get(ARABIC_LABELS["feature_importance"])
    if fi:
        print(f"\n  ── {ARABIC_LABELS['feature_importance']} ──")
        for feat, imp in fi.items():
            print(f"    {feat}: {imp}")

    # تحذيرات
    if "تحذير_data_leakage" in report:
        print(f"\n  {report['تحذير_data_leakage']}")

    # فحوصات القبول
    print("\n  ── فحوصات القبول ──")
    for check_name, check_data in report.get("فحوصات_القبول", {}).items():
        print(f"    {check_name}: {check_data['القيمة']} (الحد: {check_data['الحد']}) → {check_data['النتيجة']}")

    print("\n" + "=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
#  rssi_prior Ablation Study
# ══════════════════════════════════════════════════════════════════════════════

def rssi_prior_ablation_study(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_class=None,
    model_params: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    يُقارن أداء النموذج مع وبدون rssi_prior.

    إذا انخفضت الدقة >15% بدون rssi_prior:
      → النموذج يعتمد على الارتباط الذاتي، لا على فيزياء الأعطال.

    Parameters
    ----------
    X_train, y_train : بيانات التدريب
    X_test, y_test : بيانات الاختبار
    model_class : فئة النموذج (افتراضي: RandomForestClassifier)
    model_params : معاملات النموذج

    Returns
    -------
    تقرير المقارنة (dict)
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    if model_class is None:
        model_class = RandomForestClassifier
    if model_params is None:
        model_params = {
            "n_estimators": MODEL_CONFIG.n_estimators,
            "max_depth": MODEL_CONFIG.max_depth,
            "random_state": MODEL_CONFIG.random_state,
        }

    results = {}

    for variant_name, drop_col in [("مع rssi_prior", None), ("بدون rssi_prior", "rssi_prior")]:
        X_tr = X_train.drop(columns=[drop_col]) if drop_col and drop_col in X_train.columns else X_train.copy()
        X_te = X_test.drop(columns=[drop_col]) if drop_col and drop_col in X_test.columns else X_test.copy()

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = model_class(**model_params)
        model.fit(X_tr_s, y_train)
        y_pred = model.predict(X_te_s)

        avg = "macro" if len(np.unique(y_train)) > 2 else "binary"
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg, zero_division=0)

        results[variant_name] = {
            "الدقة_العامة": round(acc, 4),
            "مقياس_F1": round(f1, 4),
            "الاستدعاء": round(rec, 4),
            "عدد_المؤشرات": X_tr.shape[1],
        }

    # ── حساب الفرق ──
    acc_with = results["مع rssi_prior"]["الدقة_العامة"]
    acc_without = results["بدون rssi_prior"]["الدقة_العامة"]
    drop_pct = (acc_with - acc_without) / acc_with * 100

    ablation_report = {
        "title": "rssi_prior Ablation Study — TowerGuard",
        "النتائج": results,
        "الانخفاض_بالنسبة_المئوية": round(drop_pct, 2),
        "الحكم": (
            f"❌ النموذج يعتمد بشكل مفرط على rssi_prior (انخفاض {drop_pct:.1f}%>15%) — "
            "يركب موجة AR(1) بدلاً من تعلّم أنماط الأعطال"
            if drop_pct > 15
            else f"✅ rssi_prior مفيد لكن ليس مهيمناً (انخفاض {drop_pct:.1f}%≤15%)"
        ),
    }

    log.info("  📊 دراسة استئصال rssi_prior:")
    log.info("    مع rssi_prior:    %.4f F1", results["مع rssi_prior"]["مقياس_F1"])
    log.info("    بدون rssi_prior:  %.4f F1", results["بدون rssi_prior"]["مقياس_F1"])
    log.info("    الانخفاض: %.1f%%  → %s",
             drop_pct, "⚠️ مفرط" if drop_pct > 15 else "✅ مقبول")

    return ablation_report


# ══════════════════════════════════════════════════════════════════════════════
#  Temporal Overlap Check (train vs test)
# ══════════════════════════════════════════════════════════════════════════════

def check_temporal_overlap(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    tower_id_col: str = "tower_id",
) -> Dict[str, Any]:
    """
    يتحقق من عدم وجود تداخل زمني بين بيانات التدريب والاختبار.

    التداخل الزمني يعني أن نفس الفترة الزمنية لنفس البرج موجودة في
    مجموعتي التدريب والاختبار — مما يؤدي إلى تضخم الدقة.

    Parameters
    ----------
    train_df : DataFrame بيانات التدريب
    test_df : DataFrame بيانات الاختبار
    timestamp_col : اسم عمود الوقت
    tower_id_col : اسم عمود معرّف البرج

    Returns
    -------
    تقرير التداخل (dict)
    """
    overlap_report = {
        "title": "Temporal Overlap Check — TowerGuard",
    }

    # ── فحص تقاطع معرّفات الأبراج ──
    if tower_id_col in train_df.columns and tower_id_col in test_df.columns:
        train_towers = set(train_df[tower_id_col].unique())
        test_towers = set(test_df[tower_id_col].unique())
        shared_towers = train_towers & test_towers

        overlap_report["shared_towers"] = len(shared_towers)
        overlap_report["أبراج_التدريب"] = len(train_towers)
        overlap_report["أبراج_الاختبار"] = len(test_towers)

        if len(shared_towers) > 0:
            overlap_report["تحذير_أبراج"] = (
                f"⚠️ يوجد {len(shared_towers)} برج مشترك بين التدريب والاختبار — "
                "هذا يُضعف التقسيم الجغرافي"
            )
        else:
            overlap_report["حالة_الأبراج"] = "✅ لا أبراج مشتركة — التقسيم الجغرافي سليم"

    # ── فحص تقاطع النوافذ الزمنية ──
    if timestamp_col in train_df.columns and timestamp_col in test_df.columns:
        train_times = pd.to_datetime(train_df[timestamp_col], errors="coerce")
        test_times = pd.to_datetime(test_df[timestamp_col], errors="coerce")

        train_range = (train_times.min(), train_times.max())
        test_range = (test_times.min(), test_times.max())

        temporal_overlap = (train_range[1] >= test_range[0]) and (test_range[1] >= train_range[0])

        overlap_report["نطاق_التدريب"] = f"{train_range[0]} → {train_range[1]}"
        overlap_report["نطاق_الاختبار"] = f"{test_range[0]} → {test_range[1]}"
        overlap_report["temporal_overlap"] = temporal_overlap

        if temporal_overlap:
            overlap_report["تحذير_زمني"] = (
                "⚠️ يوجد تداخل زمني بين التدريب والاختبار — "
                "قد يستغل النموذج الارتباط الذاتي الزمني بدلاً من التعميم الجغرافي"
            )
        else:
            overlap_report["حالة_زمنية"] = "✅ لا تداخل زمني — التقسيم سليم"
    else:
        overlap_report["ملاحظة"] = "لا يوجد عمود زمني — التقسيم الجغرافي فقط"

    return overlap_report


# ══════════════════════════════════════════════════════════════════════════════
#  Quick Validate — تنفيذ سريع
# ══════════════════════════════════════════════════════════════════════════════

def quick_validate(bundle, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """تحقق سريع من نموذج محمّل."""
    X_scaled = bundle.scaler.transform(X_test)
    y_pred = bundle.model.predict(X_scaled)

    report = generate_validation_report(
        y_true=y_test.values,
        y_pred=y_pred,
        feature_importances=dict(zip(bundle.features, bundle.model.feature_importances_)),
    )

    print_validation_report(report)
    return report


if __name__ == "__main__":
    # تشغيل كاختبار مستقل
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # توليد بيانات تجريبية
    rng = np.random.default_rng(42)
    n = 1000
    y_true = rng.choice([0, 1, 2], size=n, p=[0.60, 0.25, 0.15])
    # محاكاة تنبؤات بدقة ~80%
    y_pred = y_true.copy()
    flip = rng.choice(n, size=int(n * 0.2), replace=False)
    y_pred[flip] = rng.choice([0, 1, 2], size=len(flip))

    report = generate_validation_report(
        y_true=y_true,
        y_pred=y_pred,
        feature_importances={f: rng.random() * 0.15 for f in TOWERGUARD_FEATURES},
    )
    print_validation_report(report)
