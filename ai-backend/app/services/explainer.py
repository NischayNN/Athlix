"""
explainer.py
------------
Adds human-readable explainability to the Athlix injury risk predictions.

Strategy: Hybrid (SHAP values + plain-English sentence templates)
    - SHAP TreeExplainer computes the exact per-feature contribution to each
      prediction, giving us *numerically grounded* explanations.
    - A sentence-template layer translates each SHAP contribution into a
      plain English bullet point, so non-technical users understand WHY
      the risk score is what it is.

Falls back to a fast rule-based mode if SHAP is unavailable.

Lives in app/services/ alongside risk_engine.py.

Public API
~~~~~~~~~~
    explain_prediction(input_features: dict) -> Explanation
    get_risk_score_with_explanation(input_features: dict) -> dict
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SERVICES_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR  = os.path.abspath(os.path.join(_SERVICES_DIR, "..", ".."))
_DATA_DIR     = os.path.join(_BACKEND_DIR, "data")

if _SERVICES_DIR not in sys.path:
    sys.path.insert(0, _SERVICES_DIR)

from generate_dataset import engineer_features, generate_raw_dataset
from risk_engine import (
    FUSION_FATIGUE_FORM_BOOST,
    FUSION_RECOVERY_BOOST,
    FUSION_TRIPLE_THREAT_BONUS,
    FATIGUE_HIGH_THRESHOLD,
    FORM_DECAY_HIGH_THRESHOLD,
    RECOVERY_LOW_THRESHOLD,
    _load_model,
    _validate_input,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature metadata — human labels + direction awareness
# ---------------------------------------------------------------------------

# Each entry: (human_label, unit, high_is_bad)
#   high_is_bad=True  → increasing value raises risk
#   high_is_bad=False → decreasing value raises risk (e.g. recovery)
_FEATURE_META: dict[str, tuple[str, str, bool]] = {
    "training_load":    ("Training load",     "/10",   True),
    "recovery_score":   ("Recovery score",    "/100",  False),
    "fatigue_index":    ("Fatigue index",     "/10",   True),
    "form_decay":       ("Form decay",        "%",     True),
    "previous_injury":  ("Prior injury",      "",      True),
    "ACWR":             ("Workload ratio",    "",      True),
    "recovery_deficit": ("Recovery deficit",  "/100",  True),
    "fatigue_trend":    ("Fatigue trend",     "/10",   True),
}

# Magnitude thresholds for sentence phrasing
_MAJOR_CONTRIBUTION = 5.0   # SHAP |value| >= this → "significantly"
_MINOR_CONTRIBUTION = 1.5   # SHAP |value| >= this → included in reasons


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class Explanation:
    """
    Full explainability result for a single prediction.

    Attributes
    ----------
    risk_score     : Final risk score from the engine [0-100].
    risk_level     : "Low" | "Medium" | "High".
    headline       : One-line summary e.g. "Risk is High because:".
    reasons        : Ordered list of plain-English bullet points.
    shap_values    : Dict of {feature: shap_contribution} (raw numbers).
    fusion_flags   : Rule-based flags triggered (from RiskOutput).
    mode           : "shap" or "rule-based" — which explainer was used.
    """
    risk_score:  float
    risk_level:  str
    headline:    str
    reasons:     list[str]
    shap_values: dict[str, float] = field(default_factory=dict)
    fusion_flags: list[str]       = field(default_factory=list)
    mode:        str               = "shap"

    def __str__(self) -> str:
        lines = [self.headline]
        for reason in self.reasons:
            lines.append(f"  - {reason}")
        if self.fusion_flags:
            lines.append("")
            lines.append("  Additional risk flags:")
            for flag in self.fusion_flags:
                lines.append(f"    * {flag}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "risk_score":   round(self.risk_score, 2),
            "risk_level":   self.risk_level,
            "headline":     self.headline,
            "reasons":      self.reasons,
            "shap_values":  {k: round(v, 3) for k, v in self.shap_values.items()},
            "fusion_flags": self.fusion_flags,
            "mode":         self.mode,
        }


# ---------------------------------------------------------------------------
# SHAP-based explainer
# ---------------------------------------------------------------------------

def _build_reference_data() -> tuple:
    """
    Build the scaled reference dataset used to fit the SHAP explainer.

    Returns (X_ref DataFrame, feature_cols list, fitted scaler).
    """
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    ref_raw  = generate_raw_dataset(n_samples=500)
    ref_eng  = engineer_features(ref_raw)
    feat_cols = [c for c in ref_eng.columns if c != "injury_risk"]

    scaler = MinMaxScaler()
    scaler.fit(ref_eng[feat_cols])

    X_ref = pd.DataFrame(
        scaler.transform(ref_eng[feat_cols]),
        columns=feat_cols,
    )
    return X_ref, feat_cols, scaler


def _shap_contributions(
    model,
    X_single: "np.ndarray",
    feature_names: list[str],
) -> dict[str, float]:
    """
    Compute SHAP values for a single prediction row.

    Uses TreeExplainer (exact, fast for XGBoost / RandomForest).
    Returns a dict mapping feature name -> SHAP contribution to the prediction.
    """
    import shap

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_single)

    # shap_vals shape: (1, n_features) for regressors
    contributions = shap_vals[0] if shap_vals.ndim == 2 else shap_vals
    return dict(zip(feature_names, contributions.tolist()))


# ---------------------------------------------------------------------------
# Sentence builder
# ---------------------------------------------------------------------------

def _contribution_to_sentence(
    feature: str,
    shap_val: float,
    raw_value: float,
) -> str | None:
    """
    Convert a single (feature, shap_contribution, raw_value) triplet into a
    plain English sentence, or None if the contribution is negligible.

    Parameters
    ----------
    feature   : column name
    shap_val  : SHAP contribution (positive = raises risk, negative = lowers)
    raw_value : original un-normalised feature value

    Returns
    -------
    str | None
    """
    if abs(shap_val) < _MINOR_CONTRIBUTION:
        return None

    label, unit, high_is_bad = _FEATURE_META.get(feature, (feature, "", True))
    magnitude = "significantly " if abs(shap_val) >= _MAJOR_CONTRIBUTION else ""

    # Risk-raising contributions
    if shap_val > 0:
        if feature == "training_load":
            return f"Training load is {magnitude}high ({raw_value:.1f}{unit}), increasing strain on the body"
        if feature == "fatigue_index":
            return f"Fatigue index is {magnitude}elevated ({raw_value:.1f}{unit}), indicating accumulated exhaustion"
        if feature == "form_decay":
            pct = raw_value * 100
            return f"Form decay is {magnitude}high ({pct:.0f}%), biomechanical technique has degraded"
        if feature == "previous_injury":
            return "Prior injury history raises baseline vulnerability"
        if feature == "recovery_deficit":
            return f"Recovery deficit is {magnitude}high ({raw_value:.1f}{unit}), athlete is under-recovered"
        if feature == "ACWR":
            return f"Workload ratio (ACWR) is {raw_value:.2f} — above 1.3 signals overtraining risk"
        if feature == "fatigue_trend":
            return f"Fatigue is trending upward ({raw_value:.1f}{unit}) over the past week"
        if feature == "recovery_score":
            return f"Recovery score is low ({raw_value:.1f}{unit}), athlete has not recovered sufficiently"
        return f"{label} is contributing to elevated risk ({raw_value:.2f}{unit})"

    # Risk-lowering contributions
    else:
        if feature == "recovery_score":
            return f"Recovery score is good ({raw_value:.1f}{unit}), partially reducing risk"
        if feature == "training_load":
            return f"Training load is moderate ({raw_value:.1f}{unit}), not adding excessive strain"
        if feature == "fatigue_index":
            return f"Fatigue level is low ({raw_value:.1f}{unit}), athlete is fresh"
        return f"{label} is helping lower the risk ({raw_value:.2f}{unit})"


def _build_reasons(
    shap_dict: dict[str, float],
    raw_features: dict,
    top_n: int = 5,
) -> list[str]:
    """
    Build an ordered, filtered list of plain-English reason sentences.

    Only includes contributions above the minor threshold, sorted by
    absolute SHAP value (most impactful first).
    """
    # Sort by |shap| descending
    sorted_items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

    reasons = []
    for feat, shap_val in sorted_items[:top_n]:
        raw_val = raw_features.get(feat, 0.0)
        sentence = _contribution_to_sentence(feat, shap_val, raw_val)
        if sentence:
            reasons.append(sentence)

    return reasons if reasons else ["No dominant individual risk factors identified."]


# ---------------------------------------------------------------------------
# Rule-based fallback explainer
# ---------------------------------------------------------------------------

def _rule_based_reasons(raw_features: dict) -> list[str]:
    """
    Simple threshold-driven explanations — used if SHAP is unavailable.
    Always produces at least one reason.
    """
    reasons = []

    tl  = raw_features.get("training_load",   5.0)
    rs  = raw_features.get("recovery_score",  50.0)
    fi  = raw_features.get("fatigue_index",   5.0)
    fd  = raw_features.get("form_decay",      0.5)
    pi  = raw_features.get("previous_injury", 0)

    if tl >= 8.0:
        reasons.append(f"Training load is very high ({tl:.1f}/10), increasing strain on the body")
    elif tl >= 6.0:
        reasons.append(f"Training load is elevated ({tl:.1f}/10)")

    if fi > FATIGUE_HIGH_THRESHOLD:
        reasons.append(f"Fatigue index is high ({fi:.1f}/10), indicating significant accumulated exhaustion")
    elif fi > 5.0:
        reasons.append(f"Fatigue index is moderate ({fi:.1f}/10)")

    if fd > FORM_DECAY_HIGH_THRESHOLD:
        reasons.append(f"Form decay is high ({fd * 100:.0f}%), biomechanical technique has significantly degraded")
    elif fd > 0.4:
        reasons.append(f"Form decay is noticeable ({fd * 100:.0f}%)")

    if rs < RECOVERY_LOW_THRESHOLD:
        reasons.append(f"Recovery score is critically low ({rs:.0f}/100), athlete has not recovered sufficiently")
    elif rs < 50:
        reasons.append(f"Recovery score is below average ({rs:.0f}/100)")

    if pi:
        reasons.append("Prior injury history raises baseline vulnerability to re-injury")

    return reasons if reasons else ["Risk is elevated due to a combination of mild factors."]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_prediction(
    input_features: dict,
    model_name: str = "xgboost",
    top_n: int = 5,
) -> Explanation:
    """
    Explain WHY the injury risk score is what it is.

    Runs the full risk scoring pipeline then uses SHAP TreeExplainer to
    decompose the model's prediction into per-feature contributions, which
    are translated into plain English bullet points ordered by impact.

    Parameters
    ----------
    input_features : dict
        Same schema as get_risk_score():
        training_load, recovery_score, fatigue_index, form_decay,
        previous_injury.  Missing keys get sensible defaults.
    model_name : str
        "xgboost" (default) or "random_forest".
    top_n : int
        Maximum number of reasons to include in the output.

    Returns
    -------
    Explanation
        Dataclass with headline, ordered reasons list, and raw SHAP values.

    Example
    -------
    >>> exp = explain_prediction({
    ...     "training_load": 8.5, "recovery_score": 25.0,
    ...     "fatigue_index": 8.1, "form_decay": 0.82,
    ...     "previous_injury": 1,
    ... })
    >>> print(exp)
    Risk is High (100.0/100) because:
      - Training load is significantly high (8.5/10), increasing strain on the body
      - Fatigue index is significantly elevated (8.1/10), indicating accumulated exhaustion
      ...
    """
    import pandas as pd
    from risk_engine import get_risk_score

    # 1. Validate + get the risk score (with fusion)
    raw_features = _validate_input(input_features)
    risk_result  = get_risk_score(input_features, model_name=model_name)

    # 2. Build scaled input for SHAP
    X_ref, feat_cols, scaler = _build_reference_data()
    eng_df = engineer_features(pd.DataFrame([raw_features]))
    X_row  = scaler.transform(eng_df[feat_cols])

    # 3. Attempt SHAP explanation
    shap_dict: dict[str, float] = {}
    mode = "shap"

    try:
        model     = _load_model(model_name)
        shap_dict = _shap_contributions(model, X_row, feat_cols)
        reasons   = _build_reasons(shap_dict, raw_features, top_n=top_n)
        logger.info("SHAP explanation computed successfully.")
    except Exception as exc:
        logger.warning("SHAP failed (%s), falling back to rule-based.", exc)
        reasons = _rule_based_reasons(raw_features)
        mode = "rule-based"

    # 4. Build headline
    level    = risk_result.risk_level
    score    = risk_result.risk_score
    headline = f"Risk is {level} ({score:.1f}/100) because:"

    return Explanation(
        risk_score   = score,
        risk_level   = level,
        headline     = headline,
        reasons      = reasons,
        shap_values  = shap_dict,
        fusion_flags = risk_result.flags,
        mode         = mode,
    )


def get_risk_score_with_explanation(input_features: dict) -> dict:
    """
    Convenience wrapper — returns risk score + full explanation as a dict.

    Suitable for direct use in FastAPI route handlers.

    Returns
    -------
    dict with keys: risk_score, risk_level, headline, reasons,
                    shap_values, fusion_flags, mode.
    """
    exp = explain_prediction(input_features)
    return exp.to_dict()


# ---------------------------------------------------------------------------
# Entry point — demo
# ---------------------------------------------------------------------------

def _demo(label: str, features: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Scenario: {label}")
    print(f"{'=' * 60}")
    exp = explain_prediction(features)
    print(exp)
    if exp.shap_values:
        print(f"\n  SHAP breakdown (top contributors):")
        sorted_shap = sorted(exp.shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        for feat, val in sorted_shap[:5]:
            direction = "raises" if val > 0 else "lowers"
            bar_len   = min(int(abs(val) / 15 * 30), 30)
            bar       = ("+" if val > 0 else "-") * bar_len
            print(f"    {feat:<22} {val:>+7.3f}  [{bar:<30}] {direction} risk")
    print(f"  (explainer mode: {exp.mode})")


if __name__ == "__main__":
    scenarios = [
        (
            "Elite athlete - low risk",
            {"training_load": 3.0, "recovery_score": 88.0,
             "fatigue_index": 1.5, "form_decay": 0.12, "previous_injury": 0},
        ),
        (
            "Overloaded athlete - maximum risk",
            {"training_load": 8.5, "recovery_score": 25.0,
             "fatigue_index": 8.1, "form_decay": 0.82, "previous_injury": 1},
        ),
        (
            "Under-recovered - moderate risk",
            {"training_load": 4.0, "recovery_score": 18.0,
             "fatigue_index": 6.5, "form_decay": 0.55, "previous_injury": 0},
        ),
    ]

    for label, feats in scenarios:
        _demo(label, feats)

    print(f"\n{'=' * 60}")
