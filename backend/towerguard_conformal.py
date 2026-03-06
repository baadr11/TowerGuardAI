"""
towerguard_conformal.py — Inductive Conformal Prediction (ICP) Engine
======================================================================
TowerGuard — Split-Conformal Prediction with Finite-Sample Coverage Guarantee

Mathematical Foundation
-----------------------
Classical Gaussian confidence intervals (z = 1.645 × σ) provide no
marginal coverage guarantee for Random Forest outputs because:
  1. RF residuals are not Gaussian.
  2. Adjacent towers exhibit spatial autocorrelation that violates the
     i.i.d. assumption required by the Gaussian CI derivation.
  3. No distribution-free guarantee exists under the Gaussian framing.

Split-Conformal / ICP guarantees:
  P(y_true ∈ [ci_lo, ci_hi]) ≥ 1 − α   (marginal, finite-sample)
without any distributional assumptions, provided the calibration set
is exchangeable with the test distribution.

Score function: Absolute Residual  s_i = |y_i − p̂_i|
Finite-sample quantile correction: ceil((n+1) × q) / n  (Venn-ABERS /
Tibshirani et al. 2019) ensures the inequality holds for all n ≥ 1.

References
----------
- Papadopoulos et al. (2002) "Inductive Confidence Machines for Regression"
- Angelopoulos & Bates (2023) "A Gentle Introduction to Conformal Prediction"
- Tibshirani et al. (2019) "Conformal Prediction Under Covariate Shift"
- MAPIE library: https://mapie.readthedocs.io

Usage
-----
    from towerguard_conformal import calibrate_conformal, predict_conformal_set

    # Split data: train / calibration / test
    # Model must be trained on train set ONLY — calibration data must be
    # held out from training.
    model = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)

    icp_params = calibrate_conformal(model, X_cal, y_cal, alpha=0.10)
    results    = predict_conformal_set(model, X_test, icp_params)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, asdict
from typing import Optional
import json


# ═══════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ICPParams:
    """ICP calibration parameters computed from the calibration set."""
    q_lo:       float   # lower quantile (α/2) of nonconformity scores
    q_hi:       float   # upper quantile (1 − α/2) of nonconformity scores
    alpha:      float   # error level (0.10 → 90% coverage)
    coverage:   float   # empirical coverage on calibration set
    n_cal:      int     # calibration set size
    method:     str = "ICP-Split-Conformal"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ICPParams":
        return cls(**d)


@dataclass
class ICPResult:
    """Single prediction result with ICP confidence interval."""
    tower_id:   Optional[str]  # tower identifier
    prob:       float           # outage probability
    ci_lo:      float           # lower bound (guaranteed)
    ci_hi:      float           # upper bound (guaranteed)
    sev:        int             # 0=OK, 1=Degraded, 2=Critical
    coverage:   float = 0.90    # coverage guarantee level

    def to_api_response(self) -> dict:
        """Format result for FastAPI endpoint response."""
        SEV_LABELS = ['OK', 'Degraded', 'Critical']
        return {
            'tower_id':    self.tower_id,
            'probability': round(self.prob, 4),
            'ci_lo':       round(self.ci_lo, 4),
            'ci_hi':       round(self.ci_hi, 4),
            'ci_width':    round(self.ci_hi - self.ci_lo, 4),
            'coverage':    self.coverage,
            'severity':    self.sev,
            'severity_label': SEV_LABELS[self.sev],
            'method':      'ICP-Split-Conformal',
            'ci_label':    f"ICP {self.coverage:.0%} → "
                           f"[{round(self.ci_lo, 3)}, {round(self.ci_hi, 3)}]",
        }


# ═══════════════════════════════════════════════════════════════════
#  calibrate_conformal() — calibration phase
# ═══════════════════════════════════════════════════════════════════

def calibrate_conformal(
    model:       RandomForestClassifier,
    X_cal:       np.ndarray,
    y_cal:       np.ndarray,
    alpha:       float = 0.10,
    spatial_ids: Optional[np.ndarray] = None,
    verbose:     bool = True,
) -> ICPParams:
    """
    Calibrate the ICP engine on a held-out calibration set.

    Mathematical guarantee:
        For any new sample (x*, y*) drawn from the same distribution:
        P(y* ∈ [p̂* − q_hi, p̂* + q_hi]) ≥ 1 − alpha
        This holds regardless of the distribution shape.

    Nonconformity score:
        s_i = |y_i − p̂_i|   (Absolute Residual)

    Finite-sample quantile correction (Tibshirani et al. 2019):
        corrected_level = ceil((n + 1) × q_level) / n
        This ensures coverage holds for all finite n, not just asymptotically.

    Args:
        model:       RandomForest trained on the training set ONLY.
                     X_cal must NOT have been seen during model.fit().
        X_cal:       Calibration features [n_cal, n_features]
        y_cal:       True labels {0, 1} or continuous in [0, 1] [n_cal]
        alpha:       Error level (0.10 → 90% coverage, 0.05 → 95%)
        spatial_ids: Region identifiers for spatial-dependency diagnostics
        verbose:     Print calibration statistics

    Returns:
        ICPParams: parameters for use in predict_conformal_set()
    """
    if len(X_cal) < 50:
        raise ValueError(
            f"[ICP] Calibration set too small ({len(X_cal)} samples). "
            "Minimum: 50. Recommended: 200+."
        )

    # Predicted probabilities from the model
    probs = model.predict_proba(X_cal)[:, 1]  # P(outage=1)

    # Nonconformity scores — Absolute Residual: s_i = |y_i − p̂_i|
    # Properties:
    #   1. No symmetry assumption required
    #   2. Interpretable: score = prediction error magnitude
    #   3. Theoretically grounded in conformal prediction literature
    scores = np.abs(y_cal.astype(float) - probs)

    n = len(scores)

    # Finite-sample correction: ceil((n+1) × q_level) / n
    # Ensures the marginal coverage inequality holds for all n ≥ 1.
    corrected_lo = np.ceil((n + 1) * (alpha / 2)) / n
    corrected_hi = np.ceil((n + 1) * (1 - alpha / 2)) / n

    corrected_lo = float(np.clip(corrected_lo, 0.0, 1.0))
    corrected_hi = float(np.clip(corrected_hi, 0.0, 1.0))

    q_lo = float(np.quantile(scores, corrected_lo))
    q_hi = float(np.quantile(scores, corrected_hi))

    # Empirical coverage check on the calibration set
    ci_lo_vals = np.clip(probs - q_hi, 0, 1)
    ci_hi_vals = np.clip(probs + q_hi, 0, 1)
    empirical_coverage = float(np.mean(
        (y_cal.astype(float) >= ci_lo_vals) &
        (y_cal.astype(float) <= ci_hi_vals)
    ))

    # Spatial dependency diagnostic (warning only — does not invalidate guarantee)
    if spatial_ids is not None:
        unique_regions = len(np.unique(spatial_ids))
        if unique_regions < 7:  # 7 administrative regions in Saudi Arabia
            print(
                f"[ICP] WARNING: {unique_regions} region(s) in calibration set. "
                "Recommended: represent all 7 regions to ensure per-region coverage."
            )

    params = ICPParams(
        q_lo=q_lo,
        q_hi=q_hi,
        alpha=alpha,
        coverage=empirical_coverage,
        n_cal=n,
    )

    if verbose:
        meets = empirical_coverage >= (1 - alpha)
        print(f"\n{'=' * 60}")
        print(f"[ICP] Split-Conformal Prediction — Calibration Complete")
        print(f"[ICP]   Guarantee: P(y ∈ CI) ≥ {1 - alpha:.0%}")
        print(f"[ICP]   Empirical coverage: {empirical_coverage:.3f} "
              f"({'PASS' if meets else 'FAIL — investigate distribution shift'})")
        print(f"[ICP]   q_lo = {q_lo:.4f}  |  q_hi = {q_hi:.4f}")
        print(f"[ICP]   n_cal = {n}  |  mean CI width = {2 * q_hi:.4f}")
        print(f"[ICP]   Note: Gaussian z=1.645 × σ ≠ coverage guarantee; "
              f"ICP = distribution-free guarantee")
        print(f"{'=' * 60}\n")

    return params


# ═══════════════════════════════════════════════════════════════════
#  predict_conformal_set() — inference with guaranteed coverage
# ═══════════════════════════════════════════════════════════════════

def predict_conformal_set(
    model:      RandomForestClassifier,
    X:          np.ndarray,
    icp_params: ICPParams,
    tower_ids:  Optional[list] = None,
) -> list[ICPResult]:
    """
    Predict with ICP confidence intervals guaranteed at (1 − alpha) level.

    Replaces the legacy Gaussian CI:
        ci_lo = prob − 1.645 × σ   ← no coverage guarantee
        ci_hi = prob + 1.645 × σ   ← no coverage guarantee

    With the conformal CI:
        ci_lo = clip(prob − q_hi, 0, 1)   ← marginal guarantee ≥ 1 − alpha
        ci_hi = clip(prob + q_hi, 0, 1)   ← marginal guarantee ≥ 1 − alpha

    Args:
        model:      RandomForest (same instance used in calibration)
        X:          Features [n_samples, n_features]
        icp_params: Output of calibrate_conformal()
        tower_ids:  Optional tower identifiers

    Returns:
        list[ICPResult]: results with guaranteed confidence intervals
    """
    probs = model.predict_proba(X)[:, 1]

    results = []
    for i, p in enumerate(probs):
        ci_lo = float(np.clip(p - icp_params.q_hi, 0, 1))
        ci_hi = float(np.clip(p + icp_params.q_hi, 0, 1))
        sev = 0 if p < 0.35 else (1 if p < 0.65 else 2)
        results.append(ICPResult(
            tower_id=tower_ids[i] if tower_ids else None,
            prob=float(p),
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            sev=sev,
            coverage=1 - icp_params.alpha,
        ))

    return results


# ═══════════════════════════════════════════════════════════════════
#  Persistence helpers
# ═══════════════════════════════════════════════════════════════════

def save_icp_params(params: ICPParams, path: str = "icp_params.json") -> None:
    """Persist ICP parameters for FastAPI server startup."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(params.to_dict(), f, indent=2)
    print(f"[ICP] Parameters saved to {path}")


def load_icp_params(path: str = "icp_params.json") -> ICPParams:
    """Load ICP parameters at server startup."""
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return ICPParams.from_dict(d)


def validate_coverage(
    model:      RandomForestClassifier,
    X_test:     np.ndarray,
    y_test:     np.ndarray,
    icp_params: ICPParams,
) -> dict:
    """
    Validate that ICP achieves the guaranteed coverage on a held-out test set.
    Coverage must be >= 1 − alpha for the guarantee to hold.
    """
    results = predict_conformal_set(model, X_test, icp_params)
    covered = sum(
        r.ci_lo <= float(y_true) <= r.ci_hi
        for r, y_true in zip(results, y_test)
    )
    coverage = covered / len(y_test)
    target   = 1 - icp_params.alpha
    meets    = coverage >= target

    print(f"[ICP-Validation] Empirical coverage: {coverage:.3f} | "
          f"Target: {target:.2f} | "
          f"{'GUARANTEE MET' if meets else 'GUARANTEE VIOLATED — check for distribution shift'}")

    return {
        'coverage':       coverage,
        'target':         target,
        'guarantee_met':  meets,
        'n_test':         len(y_test),
        'method':         'ICP-Split-Conformal',
        'gaussian_note':  f"Gaussian z=1.645 does not guarantee {target:.0%}; "
                          f"ICP provides a distribution-free guarantee.",
    }


# ═══════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("TowerGuard — ICP Engine Self-Test")
    print("=" * 60)

    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=3000, n_features=11, n_informative=8,
        random_state=42
    )

    # 60% train | 20% calibration | 20% test
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.40, random_state=42)
    X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_tr, y_tr)

    params  = calibrate_conformal(model, X_cal, y_cal, alpha=0.10)
    results = predict_conformal_set(model, X_te, params)

    validate_coverage(model, X_te, y_te, params)

    print("\nSample results:")
    for r in results[:5]:
        sev_label = ['OK', 'Degraded', 'Critical'][r.sev]
        print(f"  prob={r.prob:.3f} | CI=[{r.ci_lo:.3f}, {r.ci_hi:.3f}] | sev={sev_label}")
