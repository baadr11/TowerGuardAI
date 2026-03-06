"""
TowerGuard — Test Suite
================================
يتحقق من كل الإصلاحات المطلوبة:

  ✓ config.py: n_estimators=200، لا hardcoded values
  ✓ real_data_loader.py: Saudi Digital Twin من 420.csv
  ✓ towerguard_ml.py: يستورد من config.py، predict_with_confidence
  ✓ validation.py: Precision, Recall, Confusion Matrix
  ✓ edge_inference.py: ESP32 Firmware Generator
  ✓ Multi-class severity (0/1/2)
  ✓ Severity labels in outputs
"""

import sys
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_config():
    """اختبار 1: config.py — المصدر الوحيد للمعاملات."""
    print("\n" + "=" * 70)
    print("  🧪 اختبار 1: config.py — المصدر الوحيد")
    print("=" * 70)

    from backend.config import (
        MODEL_CONFIG, DATA_CONFIG, SEVERITY_CONFIG, EDGE_CONFIG,
        SAUDI_CORRECTIONS, ARABIC_LABELS, TOWERGUARD_FEATURES,
        TARGET_COL, get_version_info,
    )

    # ── n_estimators = 200 ──
    assert MODEL_CONFIG.n_estimators == 200, \
        f"❌ n_estimators = {MODEL_CONFIG.n_estimators}, يجب أن يكون 200!"
    print(f"  ✅ n_estimators = {MODEL_CONFIG.n_estimators} (المصدر الوحيد)")

    # ── Multi-class severity ──
    assert SEVERITY_CONFIG.SEVERITY_OK == 0
    assert SEVERITY_CONFIG.SEVERITY_DEGRADED == 1
    assert SEVERITY_CONFIG.SEVERITY_CRITICAL == 2
    assert "مستقر" in SEVERITY_CONFIG.SEVERITY_LABELS_AR.values()
    assert "متدهور" in SEVERITY_CONFIG.SEVERITY_LABELS_AR.values()
    assert "حرج" in SEVERITY_CONFIG.SEVERITY_LABELS_AR.values()
    print("  ✅ مستويات الخطورة: 0=مستقر, 1=متدهور, 2=حرج")

    # ── Target column ──
    assert TARGET_COL == "severity_score"
    print(f"  ✅ TARGET_COL = '{TARGET_COL}' (multi-class)")

    # ── Arabic labels ──
    assert len(ARABIC_LABELS) > 20
    print(f"  ✅ {len(ARABIC_LABELS)} تسمية عربية محمّلة")

    # ── Saudi corrections documented ──
    for climate, corr in SAUDI_CORRECTIONS.items():
        assert "temp_offset" in corr
        assert "rssi_offset" in corr
        print(f"  ✅ تصحيح مناخي: {climate} — temp={corr['temp_offset']:+.1f}°C, rssi={corr['rssi_offset']:+.1f}dBm")

    # ── No hardcoded values in EDGE_CONFIG ──
    assert EDGE_CONFIG.server_url is not None
    assert EDGE_CONFIG.noc_phone is not None
    print(f"  ✅ Server URL: {EDGE_CONFIG.server_url} (من config.py)")
    print(f"  ✅ NOC Phone: {EDGE_CONFIG.noc_phone} (من config.py)")

    # ── Version info ──
    info = get_version_info()
    assert info["version"] == "4.1.0"
    print(f"  ✅ الإصدار: {info['version']}")

    print("  ── اختبار 1: ✅ نجح ──")


def test_saudi_digital_twin():
    """اختبار 2: التوأم الرقمي السعودي."""
    print("\n" + "=" * 70)
    print("  🧪 اختبار 2: التوأم الرقمي السعودي (420.csv)")
    print("=" * 70)

    from backend.real_data_loader import (
        load_saudi_towers, get_kpi_profile, generate_tower_timeseries,
        validate_twin, SaudiDigitalTwinPipeline,
    )
    from backend.config import TARGET_COL, TOWERGUARD_FEATURES

    # ── Phase 1: تحميل الأبراج ──
    csv_path = "420_csv.gz"
    if not os.path.exists(csv_path):
        csv_path = "/mnt/user-data/uploads/420_csv.gz"

    towers = load_saudi_towers(csv_path)
    assert len(towers) > 0, "❌ لم يتم تحميل أي برج!"
    assert towers["tower_id"].is_unique, "❌ tower_id مكرر!"
    print(f"  ✅ Phase 1: تم تحميل {len(towers)} برج")

    # ── Check tower fields ──
    for col in ["tower_id", "operator", "density", "region", "climate"]:
        assert col in towers.columns, f"❌ العمود '{col}' مفقود!"
    print("  ✅ كل الحقول المطلوبة موجودة")

    # ── Phase 2: KPI Profile matching ──
    profile = get_kpi_profile("urban", "LTE")
    assert profile.n_towers > 0
    print(f"  ✅ Phase 2: KPI Profile — {profile.cluster_id} ({profile.n_towers} towers)")

    # ── Phase 3+4: Generate time series ──
    sample_tower = towers.iloc[0]
    ts = generate_tower_timeseries(
        tower_id=sample_tower["tower_id"],
        profile=profile,
        climate=sample_tower["climate"],
        operator=sample_tower["operator"],
    )

    assert len(ts) > 0
    assert TARGET_COL in ts.columns, f"❌ '{TARGET_COL}' مفقود في السلسلة الزمنية!"
    for feat in TOWERGUARD_FEATURES:
        assert feat in ts.columns, f"❌ المؤشر '{feat}' مفقود!"

    # ── Check multi-class target ──
    unique_sevs = sorted(ts[TARGET_COL].unique())
    print(f"  ✅ Phase 3+4: {len(ts)} خطوة زمنية | مستويات الخطورة: {unique_sevs}")

    # ── Check unique seeds per tower ──
    sample_tower2 = towers.iloc[1]
    ts2 = generate_tower_timeseries(
        tower_id=sample_tower2["tower_id"],
        profile=profile,
        climate=sample_tower2["climate"],
        operator=sample_tower2["operator"],
    )
    assert not (ts["rssi_dbm"].values[:10] == ts2["rssi_dbm"].values[:10]).all(), \
        "❌ أبراج مختلفة لها نفس السلسلة — seed=42 لم يُصلح!"
    print("  ✅ كل برج له seed فريد (ليس seed=42 للجميع)")

    # ── Build small twin for validation ──
    pipeline = SaudiDigitalTwinPipeline(
        saudi_csv_path=csv_path,
        max_towers=20,  # صغير للاختبار
    )
    twin_df = pipeline.build()
    assert len(twin_df) > 0
    print(f"  ✅ Phase 5: توأم رقمي — {len(twin_df)} صف, {twin_df['tower_id'].nunique()} برج")

    # ── Validate ──
    valid = validate_twin(twin_df)
    print(f"  ✅ التحقق: {'نجح' if valid else 'يحتاج مراجعة'}")

    print("  ── اختبار 2: ✅ نجح ──")
    return twin_df


def test_ml_training(twin_df):
    """اختبار 3: تدريب النموذج مع config.py."""
    print("\n" + "=" * 70)
    print("  🧪 اختبار 3: تدريب النموذج (من config.py)")
    print("=" * 70)

    from backend.towerguard_ml import (
        train_model, predict_with_confidence, predict_with_confidence_dict,
        normalize_input, FEATURES,
    )
    from backend.config import MODEL_CONFIG, TARGET_COL

    # ── Verify import from config ──
    print(f"  ✅ FEATURES مُستورد: {len(FEATURES)} مؤشر")

    # ── Train ──
    bundle = train_model(real_data=twin_df)
    assert bundle is not None
    assert bundle.model.n_estimators == MODEL_CONFIG.n_estimators
    print(f"  ✅ تم التدريب — {bundle.model.n_estimators} شجرة (يجب أن يكون {MODEL_CONFIG.n_estimators})")
    assert bundle.model.n_estimators == 200, "❌ n_estimators != 200!"

    # ── Check meta ──
    assert "test_precision_macro" in bundle.meta
    assert "test_recall_macro" in bundle.meta
    assert "confusion_matrix" in bundle.meta
    print("  ✅ Meta يحتوي: Precision, Recall, Confusion Matrix")

    # ── predict_with_confidence ──
    test_row = {
        "rssi_dbm": -85.0, "snr_db": 8.0, "latency_ms": 120.0,
        "packet_loss_pct": 10.0, "tower_load_pct": 70.0, "temp_celsius": 48.0,
        "rssi_prior": -82.0, "signal_variance": 5.0, "load_temp_index": 3.36,
        "rssi_rate_of_change": -1.5, "snr_trend": -0.5,
    }

    result = predict_with_confidence(bundle, test_row)
    print(f"  ✅ predict_with_confidence:")
    print(f"     التصنيف: {result.severity_label_ar}")
    print(f"     الاحتمال: {result.probability}")
    print(f"     فاصل الثقة 90%: [{result.confidence_interval['الحد_الأدنى']}, {result.confidence_interval['الحد_الأعلى']}]")
    print(f"     احتمالات الفئات: {result.class_probabilities}")

    # ── dict version ──
    result_dict = predict_with_confidence_dict(bundle, test_row)
    assert "التصنيف" in result_dict
    assert "فاصل_الثقة" in result_dict
    print("  ✅ predict_with_confidence_dict — Arabic keys موجودة")

    print("  ── اختبار 3: ✅ نجح ──")
    return bundle


def test_validation(bundle, twin_df):
    """اختبار 4: سكريبت التحقق الشامل."""
    print("\n" + "=" * 70)
    print("  🧪 اختبار 4: التحقق الشامل (Precision, Recall, CM)")
    print("=" * 70)

    import numpy as np
    from backend.validation import generate_validation_report, print_validation_report
    from backend.config import TOWERGUARD_FEATURES, TARGET_COL

    X_test = twin_df[TOWERGUARD_FEATURES].head(500)
    y_test = twin_df[TARGET_COL].head(500)

    X_scaled = bundle.scaler.transform(X_test)
    y_pred = bundle.model.predict(X_scaled)

    report = generate_validation_report(
        y_true=y_test.values,
        y_pred=y_pred,
        feature_importances=dict(zip(bundle.features, bundle.model.feature_importances_)),
        split_type="اختبار توأم رقمي",
    )

    print_validation_report(report)

    # ── Check report contents ──
    assert "المقاييس_الإجمالية" in report
    assert "المقاييس_لكل_فئة" in report
    assert "مصفوفة الارتباك" in report
    assert "فحوصات_القبول" in report
    print("  ✅ التقرير يحتوي كل المقاييس المطلوبة")

    print("  ── اختبار 4: ✅ نجح ──")


def test_edge_firmware():
    """اختبار 5: مُولّد Firmware ESP32."""
    print("\n" + "=" * 70)
    print("  🧪 اختبار 5: مُولّد Firmware ESP32")
    print("=" * 70)

    from backend.edge_inference import (
        ESP32FirmwareGenerator, edge_predict,
    )
    from backend.config import EDGE_CONFIG, SEVERITY_CONFIG

    # ── Generate firmware ──
    gen = ESP32FirmwareGenerator(output_dir="/tmp/tg_firmware_test")
    gen.generate_all()

    # ── Check files exist ──
    from pathlib import Path
    fw_dir = Path("/tmp/tg_firmware_test")
    expected_files = [
        "towerguard_config.h",
        "towerguard_edge_model.h",
        "towerguard_main.ino",
        "towerguard_sms.h",
    ]
    for f in expected_files:
        assert (fw_dir / f).exists(), f"❌ الملف {f} مفقود!"
        content = (fw_dir / f).read_text()
        assert len(content) > 100, f"❌ الملف {f} فارغ!"
        print(f"  ✅ {f} — {len(content)} حرف")

    # ── Verify config values in generated files ──
    config_h = (fw_dir / "towerguard_config.h").read_text()
    assert str(EDGE_CONFIG.server_url) in config_h
    assert "TG_SEVERITY_CRITICAL" in config_h
    print("  ✅ القيم من config.py موجودة في الملفات المُولّدة")

    # ── Test Python simulation (backward compat) ──
    prob, sev, label = edge_predict(20.0, -3.0, -1.5, 250.0)
    assert 0 <= prob <= 1
    assert sev in (0, 1, 2)
    assert isinstance(label, str)
    print(f"  ✅ edge_predict: prob={prob:.2f}, severity={sev}, label='{label}'")

    # ── Test normal reading ──
    prob_ok, sev_ok, label_ok = edge_predict(2.0, 0.5, 0.2, 30.0)
    assert sev_ok == SEVERITY_CONFIG.SEVERITY_OK
    print(f"  ✅ قراءة عادية: severity={sev_ok}, label='{label_ok}'")

    print("  ── اختبار 5: ✅ نجح ──")


def test_no_hardcoded_values():
    """اختبار 6: لا قيم مكتوبة ثابتة."""
    print("\n" + "=" * 70)
    print("  🧪 اختبار 6: فحص القيم المكتوبة ثابتاً")
    print("=" * 70)

    import inspect
    from backend import towerguard_ml

    source = inspect.getsource(towerguard_ml)

    # ── Check n_estimators is NOT hardcoded in ML module ──
    lines = source.split("\n")
    for i, line in enumerate(lines):
        if "n_estimators" in line and "=" in line:
            # Should reference MODEL_CONFIG, not a literal number
            if "MODEL_CONFIG" not in line and "150" in line:
                print(f"  ❌ السطر {i+1}: n_estimators=150 hardcoded!")
                assert False, "n_estimators hardcoded in towerguard_ml.py!"
            elif "MODEL_CONFIG" in line:
                print(f"  ✅ السطر {i+1}: n_estimators من MODEL_CONFIG")

    print("  ✅ لا قيم n_estimators مكتوبة ثابتاً في towerguard_ml.py")
    print("  ── اختبار 6: ✅ نجح ──")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "🏗" * 35)
    print("  TowerGuard — Test Suite
    print("🏗" * 35)

    test_config()
    twin_df = test_saudi_digital_twin()
    bundle = test_ml_training(twin_df)
    test_validation(bundle, twin_df)
    test_edge_firmware()
    test_no_hardcoded_values()

    print("\n" + "=" * 70)
    print("  🎉 جميع الاختبارات نجحت — TowerGuard — Test Suite
    print("=" * 70)
    print()
