"""
config.py — TowerGuard Single Source of Truth
===============================================
All model parameters, thresholds, and operational constants.
No file may define n_estimators or any model parameter outside this module.
Usage: from backend.config import MODEL_CONFIG
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, Dict


# ══════════════════════════════════════════════════════════════════════════════
#  Random Forest Model Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ModelConfig:
    """Random Forest model configuration — all parameters defined here exclusively."""

    # ── Random Forest hyperparameters ─────────────────────────────────────────
    n_estimators: int = 200          # Authoritative source — resolves any 150/200 ambiguity
    max_depth: int = 12
    min_samples_leaf: int = 10
    min_samples_split: int = 20
    max_features: str = "sqrt"
    class_weight: str = "balanced"
    n_jobs: int = -1
    random_state: int = 42

    # ── Train/test split ──────────────────────────────────────────────────────
    test_size: float = 0.20
    cv_folds: int = 5

    # ── Geographic split (not random) ─────────────────────────────────────────
    geo_train_regions: Tuple[str, ...] = (
        "Riyadh", "Jeddah_Makkah", "Eastern_Province", "Madinah", "Qassim_Hail",
    )
    geo_test_regions: Tuple[str, ...] = ("Tabuk", "Asir")

    # ── Decision thresholds ───────────────────────────────────────────────────
    decision_threshold: float = 0.50
    min_train_samples: int = 50_000

    # ── Confidence interval settings ──────────────────────────────────────────
    confidence_level: float = 0.90    # 90% ICP coverage guarantee
    confidence_z: float = 1.645       # z-score for 90% CI

    # ── Model acceptance gates — reject if any threshold is breached ──────────
    min_recall: float = 0.70
    max_top_feature_importance: float = 0.40  # > 40% signals data leakage risk
    min_rssi_snr_corr: float = 0.30


MODEL_CONFIG = ModelConfig()


# ══════════════════════════════════════════════════════════════════════════════
#  Data Configuration — Saudi Digital Twin
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DataConfig:
    """Data paths and generation parameters for the Saudi Digital Twin."""

    # ── File paths — from environment variables or safe defaults ──────────────
    saudi_csv_path: str = field(
        default_factory=lambda: os.environ.get("SAUDI_CSV", "data/420_csv.gz")
    )
    turkcell_dir: str = field(
        default_factory=lambda: os.environ.get("TURKCELL_DIR", "data/turkcell/")
    )
    output_parquet: str = field(
        default_factory=lambda: os.environ.get("OUTPUT_PATH", "data/saudi_digital_twin.parquet")
    )

    # ── Saudi tower filtering criteria ────────────────────────────────────────
    mcc_saudi: int = 420
    min_samples: int = 5
    min_range_m: int = 200
    max_range_m: int = 35_000
    allowed_radio: Tuple[str, ...] = ("LTE", "UMTS", "NR")

    # ── Kingdom of Saudi Arabia coordinate bounds ─────────────────────────────
    lat_bounds: Tuple[float, float] = (16.0, 32.5)
    lon_bounds: Tuple[float, float] = (34.5, 56.0)

    # ── AR(1) time-series generation ──────────────────────────────────────────
    n_steps: int = 336             # One week at 30-minute intervals
    min_cluster_towers: int = 30   # Minimum towers per cluster

    # ── Operator mapping ──────────────────────────────────────────────────────
    operator_map: Dict[int, str] = field(default_factory=lambda: {
        1: "STC", 3: "Mobily", 4: "Zain",
    })


DATA_CONFIG = DataConfig()


# ══════════════════════════════════════════════════════════════════════════════
#  Severity Classification — Multi-Class Target
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SeverityConfig:
    """
    Multi-class severity classification replacing a binary 0/1 target.

    Three severity levels with explicit thresholds prevent conflation of
    brief signal dips, degraded performance, and critical outages under a
    single binary label.
    """

    # ── Severity level identifiers ────────────────────────────────────────────
    SEVERITY_OK: int = 0            # No anomaly
    SEVERITY_DEGRADED: int = 1      # Measurable degradation
    SEVERITY_CRITICAL: int = 2      # Critical outage condition

    # ── Arabic labels (for UI display) ────────────────────────────────────────
    SEVERITY_LABELS_AR: Dict[int, str] = field(default_factory=lambda: {
        0: "مستقر",       # OK
        1: "متدهور",      # Degraded
        2: "حرج",         # Critical
    })

    # ── English labels ────────────────────────────────────────────────────────
    SEVERITY_LABELS_EN: Dict[int, str] = field(default_factory=lambda: {
        0: "OK",
        1: "Degraded",
        2: "Critical",
    })

    # ── Classification thresholds based on unavailable_seconds ───────────────
    degraded_threshold_sec: float = 1.0      # > 1 second = Degraded
    critical_threshold_sec: float = 300.0    # > 5 minutes = Critical

    # ── Probability-based classification thresholds ───────────────────────────
    prob_degraded: float = 0.35
    prob_critical: float = 0.65


SEVERITY_CONFIG = SeverityConfig()


# ══════════════════════════════════════════════════════════════════════════════
#  Saudi Climate Corrections — Each value cited to its source
# ══════════════════════════════════════════════════════════════════════════════

SAUDI_CORRECTIONS: Dict[str, Dict[str, float]] = {
    "dry_desert": {
        "temp_offset": +8.0,         # Source: GSMA ME 2023 — Saudi avg +8°C vs. Turkey baseline
        "rssi_offset": -2.5,         # Source: ITU-R P.618 — desert tropospheric propagation loss
        "latency_scale": 1.05,       # +5% latency — satellite backhaul in rural areas
    },
    "coastal": {
        "temp_offset": +3.0,         # Source: GSMA ME 2023
        "rssi_offset": -1.0,         # Source: ITU-R P.1411 — coastal multipath
        "latency_scale": 1.02,       # +2% latency
    },
    "highland": {
        "temp_offset": -2.0,         # Asir mountains 1500–3000m ASL
        "rssi_offset": +1.5,         # LOS improvement at elevation
        "latency_scale": 1.00,       # No additional latency penalty
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  Region and Climate Definitions
# ══════════════════════════════════════════════════════════════════════════════

REGION_DEFINITIONS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "Riyadh":            {"lat": (23.5, 26.0), "lon": (43.5, 50.0)},
    "Jeddah_Makkah":     {"lat": (20.5, 22.5), "lon": (38.5, 42.0)},
    "Madinah":           {"lat": (23.5, 25.5), "lon": (37.5, 40.5)},
    "Asir":              {"lat": (17.0, 20.5), "lon": (40.0, 44.5)},
    "Qassim_Hail":       {"lat": (26.0, 28.5), "lon": (43.0, 50.0)},
    "Tabuk":             {"lat": (27.5, 31.0), "lon": (35.5, 40.0)},
    "Eastern_Province":  {"lat": (25.0, 28.0), "lon": (48.0, 54.0)},
}

REGION_CLIMATE_MAP: Dict[str, str] = {
    "Riyadh":           "dry_desert",
    "Jeddah_Makkah":    "coastal",
    "Madinah":          "dry_desert",
    "Asir":             "highland",
    "Qassim_Hail":      "dry_desert",
    "Tabuk":            "dry_desert",
    "Eastern_Province":  "coastal",
    "Other":            "dry_desert",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Edge Inference Configuration — ESP32 + SIM7600SA-H
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EdgeConfig:
    """Edge device configuration for ESP32 + SIM7600SA-H hardware."""

    server_url: str = field(
        default_factory=lambda: os.environ.get("TG_SERVER_URL", "https://api.towerguard.sa")
    )
    noc_phone: str = field(
        default_factory=lambda: os.environ.get("TG_NOC_PHONE", "+966500000000")
    )
    heartbeat_interval_sec: int = 30
    heartbeat_timeout_sec: int = 90
    reading_interval_sec: int = 30
    spiffs_buffer_kb: int = 256
    max_batch_size: int = 500

    # ── Edge inference thresholds ─────────────────────────────────────────────
    pkt_loss_critical: float = 12.0
    rssi_roc_critical: float = -1.5
    snr_trend_critical: float = -0.6
    latency_critical: float = 120.0
    edge_danger_threshold: float = 0.60
    edge_warn_threshold: float = 0.35

    # ── Feature weights derived from Random Forest feature importance ─────────
    w_pkt_loss: float = 0.274
    w_rssi_roc: float = 0.256
    w_snr_trend: float = 0.250
    w_latency: float = 0.220


EDGE_CONFIG = EdgeConfig()


# ══════════════════════════════════════════════════════════════════════════════
#  System Labels — Arabic UI strings (display layer only)
# ══════════════════════════════════════════════════════════════════════════════

ARABIC_LABELS: Dict[str, str] = {
    # System status
    "status_ok":           "✅ مستقر — لا توجد مشاكل",
    "status_degraded":     "⚠️ متدهور — يحتاج مراقبة",
    "status_critical":     "🔴 حرج — تدخل فوري مطلوب",

    # Training lifecycle
    "training_start":      "⏳ بدء تدريب النموذج...",
    "training_complete":   "✅ اكتمل التدريب بنجاح",
    "training_failed":     "❌ فشل التدريب",
    "model_loaded":        "✅ تم تحميل النموذج",

    # Data lifecycle
    "data_loading":        "📊 جاري تحميل البيانات...",
    "data_loaded":         "✅ تم تحميل البيانات",
    "data_validated":      "✅ تم التحقق من البيانات",
    "data_error":          "❌ خطأ في البيانات",

    # Prediction verdicts
    "prediction_stable":   "مستقر",
    "prediction_warning":  "تحذير",
    "prediction_danger":   "خطر",

    # Validation results
    "validation_pass":     "✅ اجتاز التحقق",
    "validation_fail":     "❌ فشل التحقق",

    # Metric labels
    "accuracy":            "الدقة الإجمالية",
    "precision":           "الدقة الموجبة",
    "recall":              "الاستدعاء",
    "f1_score":            "مقياس F1",
    "confusion_matrix":    "مصفوفة الارتباك",
    "feature_importance":  "أهمية المؤشرات",
    "confidence_interval": "فاصل الثقة",

    # Region labels
    "Riyadh":              "الرياض",
    "Jeddah_Makkah":       "جدة ومكة المكرمة",
    "Madinah":             "المدينة المنورة",
    "Asir":                "عسير",
    "Qassim_Hail":         "القصيم وحائل",
    "Tabuk":               "تبوك",
    "Eastern_Province":    "المنطقة الشرقية",
    "Other":               "أخرى",

    # Operator labels
    "STC":                 "STC (الاتصالات السعودية)",
    "Mobily":              "موبايلي",
    "Zain":                "زين",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Feature Registry — Authoritative list; must match training pipeline exactly
# ══════════════════════════════════════════════════════════════════════════════

TOWERGUARD_FEATURES = [
    "rssi_dbm",
    "snr_db",
    "latency_ms",
    "packet_loss_pct",
    "tower_load_pct",
    "temp_celsius",
    "rssi_prior",
    "signal_variance",
    "load_temp_index",
    "rssi_rate_of_change",
    "snr_trend",
]

TARGET_COL = "severity_score"           # Multi-class: 0=OK, 1=Degraded, 2=Critical
LEGACY_TARGET_COL = "outage_in_15min"   # Legacy binary target — retained for backwards compatibility


# ══════════════════════════════════════════════════════════════════════════════
#  Physical Feature Bounds — Used in input validation
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "rssi_dbm":            (-140.0, -20.0),
    "snr_db":              (-10.0,  50.0),
    "latency_ms":          (0.0,    5000.0),
    "packet_loss_pct":     (0.0,    100.0),
    "tower_load_pct":      (0.0,    100.0),
    "temp_celsius":        (-40.0,  120.0),
    "rssi_prior":          (-140.0, -20.0),
    "signal_variance":     (0.0,    100.0),
    "load_temp_index":     (0.0,    15.0),
    "rssi_rate_of_change": (-20.0,  20.0),
    "snr_trend":           (-10.0,  10.0),
}


def get_version_info() -> Dict[str, str]:
    """Return system metadata."""
    return {
        "system":       "TowerGuard",
        "n_estimators": str(MODEL_CONFIG.n_estimators),
        "target_type":  "multi-class: 0=OK / 1=Degraded / 2=Critical",
        "data_source":  "Saudi Digital Twin (OpenCellID MCC=420 + Turkcell KPI Profiles)",
    }
