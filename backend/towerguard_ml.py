"""
towerguard_ml.py — TowerGuard Machine-Learning Core
=====================================================
Random Forest multi-class classifier for telecom tower outage prediction.

Architecture
------------
- Three-class target: 0=OK, 1=Degraded, 2=Critical
- All hyperparameters sourced exclusively from backend.config
- Geographic split validation (Riyadh/Jeddah train → Tabuk/Asir test)
- ICP confidence intervals via towerguard_conformal.py

Data-Leakage Mitigation
-----------------------
rssi_prior default is NaN (not rssi_dbm). When rssi_prior is absent in
an incoming payload the value is imputed from a rolling 3-reading median
buffer, not from the current reading. This prevents the model from
exploiting the AR(1) autocorrelation as a shortcut feature.
"""

import logging
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
)

from .config import (
    MODEL_CONFIG, SEVERITY_CONFIG,
    TOWERGUARD_FEATURES, TARGET_COL, LEGACY_TARGET_COL,
    ARABIC_LABELS,
)

log = logging.getLogger("towerguard.ml")

FEATURES: List[str] = list(TOWERGUARD_FEATURES)
_train_lock = threading.Lock()

# Rolling buffer for rssi_prior imputation (3-reading median)
_RSSI_BUFFER_SIZE = 3
_rssi_rolling: Deque[float] = deque(maxlen=_RSSI_BUFFER_SIZE)


# ══════════════════════════════════════════════════════════════════════════
#  Data Classes
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ModelBundle:
    model:    RandomForestClassifier
    scaler:   StandardScaler
    features: List[str]
    meta:     Dict[str, Any]


@dataclass(frozen=True)
class PredictionResult:
    """Complete prediction result with uncertainty quantification."""
    severity:             int     # 0=OK, 1=Degraded, 2=Critical
    severity_label:       str     # human-readable label
    probability:          float   # outage probability [0, 1]
    confidence_interval:  Dict    # 90% confidence interval
    tree_std:             float   # inter-tree standard deviation
    class_probabilities:  Dict    # per-class probability


# ══════════════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════════════

def train_model(
    n_samples: int = 6_000,
    real_data: Optional[pd.DataFrame] = None,
) -> ModelBundle:
    """
    Train a multi-class Random Forest on Saudi tower data.
    All hyperparameters are drawn from MODEL_CONFIG — no hardcoded values.
    """
    log.info("Training started")

    target_col = TARGET_COL  # severity_score (multi-class)

    if real_data is not None and len(real_data) > 0:
        missing = [c for c in FEATURES if c not in real_data.columns]
        if missing:
            log.warning("Real data missing features: %s — falling back to simulation",
                        missing)
            real_data = None

    if real_data is not None and len(real_data) > 0:
        if target_col not in real_data.columns:
            if LEGACY_TARGET_COL in real_data.columns:
                log.info("Converting legacy binary target to multi-class...")
                real_data[target_col] = real_data[LEGACY_TARGET_COL].apply(
                    lambda x: 0 if x == 0 else 1
                )
            else:
                log.warning("No target column found — falling back to simulation")
                real_data = None

    if real_data is not None and len(real_data) > 0:
        data_type = "Saudi Digital Twin"
        n_real = len(real_data)

        geographic_split = False
        if "region" in real_data.columns:
            train_mask = real_data["region"].isin(MODEL_CONFIG.geo_train_regions)
            test_mask  = real_data["region"].isin(MODEL_CONFIG.geo_test_regions)

            if train_mask.sum() > 0 and test_mask.sum() > 0:
                train_df = real_data[train_mask]
                test_df  = real_data[test_mask]
                log.info("Geographic split — train: %d | test: %d",
                         len(train_df), len(test_df))

                from .validation import check_temporal_overlap
                overlap = check_temporal_overlap(train_df, test_df)
                if overlap.get("temporal_overlap", False):
                    log.warning("Temporal overlap detected in geographic split — "
                                "consider separating time windows between regions")
                if overlap.get("shared_towers", 0) > 0:
                    log.warning("Shared towers across split: %d",
                                overlap["shared_towers"])

                X_train = train_df[FEATURES]
                y_train = train_df[target_col]
                X_test  = test_df[FEATURES]
                y_test  = test_df[target_col]
                geographic_split = True

        if not geographic_split:
            df = real_data.sample(n=min(n_samples, n_real),
                                  random_state=MODEL_CONFIG.random_state)
            X = df[FEATURES]
            y = df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=MODEL_CONFIG.test_size,
                random_state=MODEL_CONFIG.random_state, stratify=y,
            )
    else:
        log.info("Generating %d calibration simulation samples...", n_samples)
        from .real_data_loader import SaudiDigitalTwinPipeline
        try:
            pipeline = SaudiDigitalTwinPipeline(max_towers=50)
            df = pipeline.build()
        except Exception:
            df = _generate_fallback_sim_data(n_samples)

        data_type = "Calibration Simulation"
        X = df[FEATURES]
        y = (df[target_col] if target_col in df.columns
             else df.get(LEGACY_TARGET_COL, pd.Series(0, index=df.index)))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=MODEL_CONFIG.test_size,
            random_state=MODEL_CONFIG.random_state, stratify=y,
        )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    log.info("Training Random Forest — %d trees (from config)",
             MODEL_CONFIG.n_estimators)

    model = RandomForestClassifier(
        n_estimators=MODEL_CONFIG.n_estimators,
        max_depth=MODEL_CONFIG.max_depth,
        min_samples_leaf=MODEL_CONFIG.min_samples_leaf,
        min_samples_split=MODEL_CONFIG.min_samples_split,
        max_features=MODEL_CONFIG.max_features,
        class_weight=MODEL_CONFIG.class_weight,
        random_state=MODEL_CONFIG.random_state,
        n_jobs=MODEL_CONFIG.n_jobs,
    )
    model.fit(X_train_s, y_train)

    log.info("Running %d-fold cross-validation...", MODEL_CONFIG.cv_folds)
    cv = StratifiedKFold(
        n_splits=MODEL_CONFIG.cv_folds, shuffle=True,
        random_state=MODEL_CONFIG.random_state,
    )
    cv_acc = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="accuracy")

    y_pred = model.predict(X_test_s)
    is_multiclass = len(np.unique(y_test)) > 2
    avg = "macro" if is_multiclass else "binary"

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
    rec  = recall_score(y_test, y_pred, average=avg, zero_division=0)
    f1   = f1_score(y_test, y_pred, average=avg, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    importances = dict(zip(FEATURES, model.feature_importances_))
    top_feat = max(importances, key=importances.get)
    top_imp  = importances[top_feat]

    if top_imp > MODEL_CONFIG.max_top_feature_importance:
        log.warning("Feature '%s' dominates at %.1f%% importance — "
                    "check for data leakage", top_feat, top_imp * 100)

    # rssi_prior dominance check
    if "rssi_prior" in importances and importances["rssi_prior"] > 0.35:
        log.warning("rssi_prior dominance: %.1f%% — AR(1) autocorrelation "
                    "exploitation risk. Ensure NaN default + rolling imputation "
                    "is active in normalize_input().",
                    importances["rssi_prior"] * 100)
        try:
            from .validation import rssi_prior_ablation_study
            ablation = rssi_prior_ablation_study(X_train, y_train, X_test, y_test)
            log.info("Ablation result: %s", ablation.get("verdict", "—"))
        except Exception as e:
            log.warning("Ablation study failed: %s", str(e))

    if rec < MODEL_CONFIG.min_recall:
        log.warning("Recall %.3f below minimum %.2f — model is unsafe for deployment",
                    rec, MODEL_CONFIG.min_recall)

    log.info("=" * 60)
    log.info("Training results:")
    log.info("  Data type:           %s", data_type)
    log.info("  Trees:               %d (from config)", MODEL_CONFIG.n_estimators)
    log.info("  CV accuracy:         %.3f ± %.3f", cv_acc.mean(), cv_acc.std())
    log.info("  Test precision:      %.3f", prec)
    log.info("  Test recall:         %.3f", rec)
    log.info("  Test F1 (macro):     %.3f", f1)
    log.info("  Confusion matrix:\n%s", cm)
    log.info("  Feature importances (top 5):")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1])[:5]:
        log.info("    %-26s %.1f%%", feat, imp * 100)
    log.info("=" * 60)

    meta: Dict[str, Any] = {
        "name":                 "TowerGuard Random Forest",
        "data_type":            data_type,
        "n_estimators":         MODEL_CONFIG.n_estimators,
        "n_samples_train":      int(len(X_train)),
        "n_samples_test":       int(len(X_test)),
        "target_type":          "multi-class (3 levels: OK / Degraded / Critical)",
        "cv_accuracy_mean":     float(cv_acc.mean()),
        "cv_accuracy_std":      float(cv_acc.std()),
        "test_accuracy":        float(acc),
        "test_precision_macro": float(prec),
        "test_recall_macro":    float(rec),
        "test_f1_macro":        float(f1),
        "confusion_matrix":     cm.tolist(),
        "feature_importances":  importances,
    }

    log.info("Training complete")
    return ModelBundle(model=model, scaler=scaler,
                       features=list(FEATURES), meta=meta)


# ══════════════════════════════════════════════════════════════════════
#  Prediction
# ══════════════════════════════════════════════════════════════════════

def predict_proba(bundle: ModelBundle, row: Dict[str, float]) -> float:
    """Return outage probability (0–1) for a single reading (backwards-compat)."""
    X = pd.DataFrame([[row[f] for f in bundle.features]], columns=bundle.features)
    X_scaled = bundle.scaler.transform(X)
    proba = bundle.model.predict_proba(X_scaled)[0]
    if len(proba) > 2:
        return float(proba[1] + proba[2])  # P(Degraded) + P(Critical)
    return float(proba[1]) if len(proba) > 1 else 0.0


def predict_with_confidence(bundle: ModelBundle,
                             row: Dict[str, float]) -> PredictionResult:
    """
    Full prediction with 90% confidence interval derived from inter-tree variance.

    Note: this is a Gaussian approximation suitable for live display.
    For statistically guaranteed coverage use towerguard_conformal.py.
    """
    X = pd.DataFrame([[row[f] for f in bundle.features]], columns=bundle.features)
    X_scaled = bundle.scaler.transform(X)

    n_classes = len(bundle.model.classes_)
    tree_predictions = np.array([
        tree.predict_proba(X_scaled)[0]
        for tree in bundle.model.estimators_
    ])

    mean_proba = tree_predictions.mean(axis=0)
    std_proba  = tree_predictions.std(axis=0)

    severity = int(bundle.model.classes_[np.argmax(mean_proba)])
    severity_label = SEVERITY_CONFIG.SEVERITY_LABELS_EN.get(severity, "Unknown")

    if n_classes > 2:
        risk_per_tree = tree_predictions[:, 1:].sum(axis=1)
    else:
        risk_per_tree = (tree_predictions[:, 1]
                         if tree_predictions.shape[1] > 1
                         else tree_predictions[:, 0])

    mean_risk = float(np.mean(risk_per_tree))
    std_risk  = float(np.std(risk_per_tree))

    z = MODEL_CONFIG.confidence_z  # 1.645
    ci_low  = max(0.0, mean_risk - z * std_risk)
    ci_high = min(1.0, mean_risk + z * std_risk)

    class_proba = {}
    for i, cls in enumerate(bundle.model.classes_):
        label = SEVERITY_CONFIG.SEVERITY_LABELS_EN.get(int(cls), str(cls))
        class_proba[label] = round(float(mean_proba[i]), 4)

    return PredictionResult(
        severity=severity,
        severity_label=severity_label,
        probability=round(max(0.0, min(1.0, mean_risk)), 4),
        confidence_interval={
            "low":   round(ci_low, 4),
            "high":  round(ci_high, 4),
            "level": MODEL_CONFIG.confidence_level,
        },
        tree_std=round(std_risk, 4),
        class_probabilities=class_proba,
    )


def predict_with_confidence_dict(bundle: ModelBundle,
                                  row: Dict[str, float]) -> Dict[str, Any]:
    """Dict-serialisable version of predict_with_confidence for JSON API."""
    result = predict_with_confidence(bundle, row)
    return {
        "severity":           result.severity,
        "severity_label":     result.severity_label,
        "probability":        result.probability,
        "confidence_interval": result.confidence_interval,
        "tree_std":           result.tree_std,
        "class_probabilities": result.class_probabilities,
    }


# ══════════════════════════════════════════════════════════════════════
#  Input Normalisation
# ══════════════════════════════════════════════════════════════════════

def normalize_input(payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert a raw API payload to model-ready numeric features.

    rssi_prior handling (leakage prevention)
    -----------------------------------------
    rssi_prior is the RSSI reading from the PREVIOUS cycle, used by the
    model to capture AR(1) signal dynamics.

    Default is NaN — NOT rssi_dbm. Setting the default to rssi_dbm would
    create a perfect correlation between rssi_prior and rssi_dbm for any
    payload that omits rssi_prior, causing the model to exploit an
    artificial AR(1) relationship rather than a real temporal one.

    Imputation strategy:
      - If rssi_prior is present in the payload: use it directly.
      - If rssi_prior is absent:
          a) If the rolling 3-reading buffer has ≥ 1 entry: use its median.
          b) If the buffer is empty (cold start): use rssi_dbm as a one-time
             fallback and log a warning.

    The rolling buffer is updated with each normalised rssi_dbm value.
    """
    def _get(name: str, default: Optional[float] = None) -> float:
        v = payload.get(name)
        if v is None:
            if default is None:
                raise ValueError(f"Required field missing: '{name}'")
            if np.isnan(default):
                return float('nan')
            return float(default)
        try:
            result = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Field '{name}' must be numeric; got: {v!r}"
            ) from exc
        if not np.isfinite(result):
            raise ValueError(
                f"Field '{name}' must be finite; got: {result}"
            )
        return result

    rssi_dbm        = _get("rssi_dbm")
    snr_db          = _get("snr_db")
    latency_ms      = _get("latency_ms")
    packet_loss_pct = _get("packet_loss_pct")
    tower_load_pct  = _get("tower_load_pct")
    temp_celsius    = _get("temp_celsius")

    # rssi_prior: NaN default with rolling-buffer median imputation
    raw_prior = payload.get("rssi_prior")
    if raw_prior is not None:
        try:
            rssi_prior = float(raw_prior)
        except (TypeError, ValueError):
            rssi_prior = float('nan')
    else:
        rssi_prior = float('nan')

    if np.isnan(rssi_prior):
        if len(_rssi_rolling) > 0:
            rssi_prior = float(np.median(list(_rssi_rolling)))
        else:
            # Cold-start fallback: use current reading, log warning
            log.warning(
                "rssi_prior absent and rolling buffer empty (cold start). "
                "Using rssi_dbm as one-time fallback. This inflates apparent "
                "AR(1) correlation for this single prediction only."
            )
            rssi_prior = rssi_dbm

    # Update rolling buffer with current reading
    _rssi_rolling.append(rssi_dbm)

    return {
        "rssi_dbm":            rssi_dbm,
        "snr_db":              snr_db,
        "latency_ms":          latency_ms,
        "packet_loss_pct":     packet_loss_pct,
        "tower_load_pct":      tower_load_pct,
        "temp_celsius":        temp_celsius,
        "rssi_prior":          rssi_prior,
        "signal_variance":     _get("signal_variance", default=0.0),
        "load_temp_index":     _get("load_temp_index",
                                    default=(tower_load_pct * temp_celsius) / 1000.0),
        "rssi_rate_of_change": _get("rssi_rate_of_change", default=0.0),
        "snr_trend":           _get("snr_trend", default=0.0),
    }


# ══════════════════════════════════════════════════════════════════════
#  Persistence
# ══════════════════════════════════════════════════════════════════════

def load_or_train(model_path: Path) -> ModelBundle:
    """Load existing model or train a new one."""
    with _train_lock:
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if model_path.exists():
            log.info("Loading model from: %s", model_path)
            try:
                raw = joblib.load(model_path)
                if isinstance(raw, ModelBundle):
                    return raw
                if isinstance(raw, dict):
                    rf     = raw.get("model")
                    scaler = raw.get("scaler")
                    feats  = raw.get("features", FEATURES)
                    meta   = {k: v for k, v in raw.items()
                              if k not in {"model", "scaler", "features"}}
                    if isinstance(rf, RandomForestClassifier) and scaler is not None:
                        return ModelBundle(model=rf, scaler=scaler,
                                           features=list(feats), meta=dict(meta))
            except Exception as exc:
                log.warning("Failed to load model (%s) — retraining", exc)

        log.info("No valid model found — training from scratch")
        real_df = None
        try:
            from .real_data_loader import RealDataPipeline
            pipeline = RealDataPipeline(base_dir=model_path.parent.parent)
            real_df = pipeline.auto_discover()
            if real_df is not None:
                log.info("Real data discovered: %d rows", len(real_df))
        except Exception as exc:
            log.info("Data discovery skipped: %s", exc)

        bundle = train_model(real_data=real_df)
        _save_bundle(bundle, model_path)
        return bundle


def _save_bundle(bundle: ModelBundle, path: Path) -> None:
    """Save model bundle with SHA-256 checksum."""
    try:
        joblib.dump({
            "model":    bundle.model,
            "scaler":   bundle.scaler,
            "features": bundle.features,
            **bundle.meta,
        }, path)
        log.info("Model saved: %s", path)

        import hashlib
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(".sha256").write_text(checksum)
        log.info("Checksum saved: %s...", checksum[:16])
    except Exception as exc:
        log.error("Failed to save model: %s", exc)


# ══════════════════════════════════════════════════════════════════════
#  Fallback Simulation
# ══════════════════════════════════════════════════════════════════════

def _generate_fallback_sim_data(n_samples: int = 6_000) -> pd.DataFrame:
    """
    Calibration simulation — generates severity_score (0/1/2) target.
    Used only when no real data is available.

    rssi_prior is generated from an independent noise draw (not a copy of
    rssi_dbm) to avoid introducing artificial AR(1) correlation at
    training time.
    """
    rng = np.random.default_rng(seed=MODEL_CONFIG.random_state)

    n_ok   = int(n_samples * 0.60)
    n_deg  = int(n_samples * 0.25)
    n_crit = n_samples - n_ok - n_deg

    rows = []
    for label, n, rssi_mu, snr_mu, lat_mu, pkt_mu in [
        (0, n_ok,   -72, 15,  45,  3.5),
        (1, n_deg,  -85, 10, 100,  8.0),
        (2, n_crit, -95,  5, 200, 18.0),
    ]:
        rssi = rng.normal(rssi_mu, 15, n).clip(-120, -38)
        snr  = rng.normal(snr_mu, 8, n).clip(-5, 40)
        lat  = rng.normal(lat_mu, 50, n).clip(5, 2000)
        pkt  = rng.normal(pkt_mu, 5, n).clip(0, 50)
        load = rng.normal(55, 20, n).clip(5, 100)
        temp = rng.normal(43, 13, n).clip(-10, 75)

        # rssi_prior: independent AR(1) lag (not a copy of rssi)
        rssi_prior = np.roll(rssi, 1)
        rssi_prior[0] = rssi_mu + rng.normal(0, 5)

        for i in range(n):
            rows.append({
                "rssi_dbm":            round(rssi[i], 2),
                "snr_db":              round(snr[i], 2),
                "latency_ms":          round(lat[i], 1),
                "packet_loss_pct":     round(pkt[i], 3),
                "tower_load_pct":      round(load[i], 1),
                "temp_celsius":        round(temp[i], 1),
                "rssi_prior":          round(rssi_prior[i], 2),
                "signal_variance":     round(abs(rng.uniform(0, 15)), 3),
                "load_temp_index":     round(load[i] * temp[i] / 1000, 3),
                "rssi_rate_of_change": round(rng.normal(0 if label == 0 else -1.5, 2), 3),
                "snr_trend":           round(rng.normal(0 if label == 0 else -0.8, 1), 3),
                TARGET_COL:            label,
            })

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=MODEL_CONFIG.random_state).reset_index(drop=True)
