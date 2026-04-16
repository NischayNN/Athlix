"""
risk_engine.py
--------------
Converts raw ML model output into a structured, interpretable risk score
with level classification and rule-based fusion adjustments.

Lives in app/services/ alongside generate_dataset.py, train_models.py,
feature_engineering.py, and pose_service.py.

Public API
~~~~~~~~~~
    get_risk_score(input_features: dict) -> RiskOutput

Usage example
~~~~~~~~~~~~~
    from app.services.risk_engine import get_risk_score

    result = get_risk_score({
        "training_load":   8.5,
        "recovery_score":  25.0,
        "fatigue_index":   8.1,
        "form_decay":      0.82,
        "previous_injury": 1,
    })
    print(result)
    # Risk = 78.4 (High)
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow importing sibling services
# ---------------------------------------------------------------------------
_SERVICES_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR  = os.path.abspath(os.path.join(_SERVICES_DIR, "..", ".."))
_DATA_DIR     = os.path.join(_BACKEND_DIR, "data")

if _SERVICES_DIR not in sys.path:
    sys.path.insert(0, _SERVICES_DIR)

from generate_dataset import engineer_features, generate_raw_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risk level thresholds
# ---------------------------------------------------------------------------
LEVEL_LOW_MAX    = 30
LEVEL_MEDIUM_MAX = 70
# Anything above LEVEL_MEDIUM_MAX is HIGH

# ---------------------------------------------------------------------------
# Fusion rule thresholds (raw, un-normalised feature space)
# ---------------------------------------------------------------------------
FATIGUE_HIGH_THRESHOLD  = 7.0    # fatigue_index  >  this → "high fatigue"
FORM_DECAY_HIGH_THRESHOLD = 0.70  # form_decay     >  this → "poor form"
RECOVERY_LOW_THRESHOLD  = 35.0   # recovery_score <  this → "low recovery"

# Penalty/boost magnitudes (applied to the 0-100 score)
FUSION_FATIGUE_FORM_BOOST  = 8.0   # both fatigue AND form are bad
FUSION_RECOVERY_BOOST      = 5.0   # recovery is poor
FUSION_TRIPLE_THREAT_BONUS = 4.0   # all three conditions at once


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class RiskOutput:
    """
    Structured result returned by get_risk_score().

    Attributes
    ----------
    risk_score       : float  - Final risk value in [0, 100].
    risk_level       : str    - "Low" | "Medium" | "High".
    model_score      : float  - Raw XGBoost model output before fusion.
    fusion_delta     : float  - Total adjustment applied by the rule engine.
    flags            : list   - Human-readable list of triggered fusion rules.
    """
    risk_score:   float
    risk_level:   str
    model_score:  float
    fusion_delta: float
    flags:        list[str] = field(default_factory=list)

    def __str__(self) -> str:
        base = f"Risk = {self.risk_score:.1f}% ({self.risk_level})"
        if self.flags:
            base += f"  |  Flags: {', '.join(self.flags)}"
        return base

    def to_dict(self) -> dict:
        return {
            "risk_score":   round(self.risk_score, 2),
            "risk_level":   self.risk_level,
            "model_score":  round(self.model_score, 2),
            "fusion_delta": round(self.fusion_delta, 2),
            "flags":        self.flags,
        }


# ---------------------------------------------------------------------------
# Model loader (lazy singleton)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict = {}

def _load_model(name: str = "xgboost"):
    """
    Load a serialised model from data/ and cache it in memory.

    Falls back to random_forest.pkl if xgboost.pkl is not found.
    """
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]

    pkl_path = os.path.join(_DATA_DIR, f"{name}.pkl")

    if not os.path.exists(pkl_path):
        fallback = "random_forest" if name == "xgboost" else "xgboost"
        fallback_path = os.path.join(_DATA_DIR, f"{fallback}.pkl")
        if os.path.exists(fallback_path):
            logger.warning("Model '%s' not found. Falling back to '%s'.", name, fallback)
            pkl_path = fallback_path
            name = fallback
        else:
            raise FileNotFoundError(
                f"No trained model found in {_DATA_DIR}. "
                "Run app/services/train_models.py first."
            )

    model = joblib.load(pkl_path)
    _MODEL_CACHE[name] = model
    logger.info("Model '%s' loaded from %s.", name, pkl_path)
    return model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_level(score: float) -> str:
    """Map a numeric score to a human-readable risk level."""
    if score <= LEVEL_LOW_MAX:
        return "Low"
    elif score <= LEVEL_MEDIUM_MAX:
        return "Medium"
    return "High"


def _validate_input(features: dict) -> dict:
    """
    Validate and fill defaults for required input keys.

    Required keys
    -------------
    training_load, recovery_score, fatigue_index, form_decay, previous_injury

    Raises ValueError on out-of-range values.
    """
    defaults = {
        "training_load":   5.0,
        "recovery_score":  50.0,
        "fatigue_index":   5.0,
        "form_decay":      0.5,
        "previous_injury": 0,
    }

    validated = {k: features.get(k, v) for k, v in defaults.items()}

    # Range checks
    if not (1 <= validated["training_load"] <= 10):
        raise ValueError(f"training_load must be 1-10, got {validated['training_load']}")
    if not (0 <= validated["recovery_score"] <= 100):
        raise ValueError(f"recovery_score must be 0-100, got {validated['recovery_score']}")
    if not (0 <= validated["fatigue_index"] <= 10):
        raise ValueError(f"fatigue_index must be 0-10, got {validated['fatigue_index']}")
    if not (0 <= validated["form_decay"] <= 1):
        raise ValueError(f"form_decay must be 0-1, got {validated['form_decay']}")
    if validated["previous_injury"] not in (0, 1):
        raise ValueError(f"previous_injury must be 0 or 1, got {validated['previous_injury']}")

    return validated


def _apply_fusion(
    model_score: float,
    features: dict,
) -> tuple[float, float, list[str]]:
    """
    Apply rule-based fusion adjustments on top of the ML model score.

    Rules
    -----
    1. High fatigue AND high form decay  → +FUSION_FATIGUE_FORM_BOOST
    2. Low recovery score                → +FUSION_RECOVERY_BOOST
    3. All three conditions simultaneously → additional +FUSION_TRIPLE_THREAT_BONUS

    Parameters
    ----------
    model_score : float  Raw model prediction (0-100).
    features    : dict   Validated, raw (un-normalised) feature dict.

    Returns
    -------
    (adjusted_score, total_delta, flags_list)
    """
    delta = 0.0
    flags: list[str] = []

    fatigue_high  = features["fatigue_index"]  > FATIGUE_HIGH_THRESHOLD
    form_bad      = features["form_decay"]     > FORM_DECAY_HIGH_THRESHOLD
    recovery_low  = features["recovery_score"] < RECOVERY_LOW_THRESHOLD

    # Rule 1: High fatigue + bad form
    if fatigue_high and form_bad:
        delta += FUSION_FATIGUE_FORM_BOOST
        flags.append(
            f"High fatigue ({features['fatigue_index']:.1f}) + "
            f"poor form ({features['form_decay']:.2f}) +{FUSION_FATIGUE_FORM_BOOST:.0f}"
        )

    # Rule 2: Low recovery
    if recovery_low:
        delta += FUSION_RECOVERY_BOOST
        flags.append(
            f"Low recovery ({features['recovery_score']:.1f}) +{FUSION_RECOVERY_BOOST:.0f}"
        )

    # Rule 3: Triple-threat bonus
    if fatigue_high and form_bad and recovery_low:
        delta += FUSION_TRIPLE_THREAT_BONUS
        flags.append(f"Triple-threat (all risk factors active) +{FUSION_TRIPLE_THREAT_BONUS:.0f}")

    adjusted = float(np.clip(model_score + delta, 0.0, 100.0))
    return adjusted, delta, flags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_risk_score(
    input_features: dict,
    model_name: str = "xgboost",
) -> RiskOutput:
    """
    Convert raw athlete features into a structured injury risk score.

    Pipeline
    --------
    1. Validate & fill missing input fields.
    2. Build a single-row DataFrame and apply feature engineering
       (ACWR, recovery_deficit, fatigue_trend).
    3. Scale features with Min-Max (fitted on the training distribution).
    4. Run the trained ML model to get a base prediction.
    5. Apply rule-based fusion logic to adjust the score.
    6. Classify the final score into Low / Medium / High.

    Parameters
    ----------
    input_features : dict
        Must contain: training_load, recovery_score, fatigue_index,
                      form_decay, previous_injury.
        Missing keys fall back to sensible defaults.
    model_name : str
        Which saved model to use ("xgboost" or "random_forest").

    Returns
    -------
    RiskOutput
        Dataclass with risk_score, risk_level, model_score,
        fusion_delta, and triggered flags.

    Example
    -------
    >>> result = get_risk_score({
    ...     "training_load": 8.5, "recovery_score": 25.0,
    ...     "fatigue_index": 8.1, "form_decay": 0.82,
    ...     "previous_injury": 1,
    ... })
    >>> print(result)
    Risk = 82.3% (High)  |  Flags: High fatigue...
    """
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    # 1. Validate
    features = _validate_input(input_features)

    # 2. Build DataFrame & engineer features
    raw_df = pd.DataFrame([features])
    eng_df = engineer_features(raw_df)

    # 3. Scale — at inference time there is no injury_risk target column,
    #    so we refit a MinMaxScaler on a reference training distribution and
    #    apply it to the single inference row.
    #    The scaler is intentionally refitted from the same synthetic generator
    #    that produced the training data (same seed = same distribution).
    ref_raw    = generate_raw_dataset(n_samples=500)
    ref_eng    = engineer_features(ref_raw)
    feat_cols  = [c for c in ref_eng.columns if c != "injury_risk"]

    scaler = MinMaxScaler()
    scaler.fit(ref_eng[feat_cols])

    X = scaler.transform(eng_df[feat_cols])

    # 4. Model prediction
    model        = _load_model(model_name)
    model_score  = float(np.clip(model.predict(X)[0], 0.0, 100.0))

    # 5. Fusion adjustments
    final_score, delta, flags = _apply_fusion(model_score, features)

    # 6. Classify
    level = _classify_level(final_score)

    return RiskOutput(
        risk_score=round(final_score, 2),
        risk_level=level,
        model_score=round(model_score, 2),
        fusion_delta=round(delta, 2),
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Entry point — demo
# ---------------------------------------------------------------------------

def _sep(w: int = 60) -> str:
    return "-" * w


def _demo_case(label: str, features: dict) -> None:
    """Run one demo scenario and print a formatted result."""
    result = get_risk_score(features)
    pct_bar_len = int(result.risk_score / 100 * 40)
    pct_bar     = "#" * pct_bar_len + "-" * (40 - pct_bar_len)

    print(f"\n  Scenario : {label}")
    print(f"  Input    : {features}")
    print(f"  {_sep(52)}")
    print(f"  Model score  (pre-fusion) : {result.model_score:.1f}")
    print(f"  Fusion delta              : +{result.fusion_delta:.1f}")
    print(f"  Final risk score          : {result.risk_score:.1f} / 100")
    print(f"  Risk level                : {result.risk_level}")
    print(f"  [{pct_bar}] {result.risk_score:.0f}%")
    if result.flags:
        print("  Triggered rules:")
        for flag in result.flags:
            print(f"      * {flag}")
    print(f"  --> {result}")


if __name__ == "__main__":
    print(_sep(60))
    print("  Athlix - Risk Engine Demo")
    print(_sep(60))

    scenarios = [
        (
            "Elite athlete — fresh and recovered",
            {"training_load": 3.0, "recovery_score": 88.0,
             "fatigue_index": 1.5, "form_decay": 0.12, "previous_injury": 0},
        ),
        (
            "Moderate load — average condition",
            {"training_load": 5.5, "recovery_score": 55.0,
             "fatigue_index": 5.0, "form_decay": 0.45, "previous_injury": 0},
        ),
        (
            "Overloaded athlete — poor recovery",
            {"training_load": 8.5, "recovery_score": 25.0,
             "fatigue_index": 8.1, "form_decay": 0.82, "previous_injury": 1},
        ),
        (
            "Heavy training — prior injury history",
            {"training_load": 9.2, "recovery_score": 40.0,
             "fatigue_index": 7.5, "form_decay": 0.75, "previous_injury": 1},
        ),
        (
            "Low load but severely under-recovered",
            {"training_load": 2.0, "recovery_score": 10.0,
             "fatigue_index": 8.5, "form_decay": 0.60, "previous_injury": 0},
        ),
    ]

    for label, feats in scenarios:
        _demo_case(label, feats)

    print(f"\n{_sep(60)}")
    print("  Risk level thresholds:")
    print(f"    0  - {LEVEL_LOW_MAX}   -> Low")
    print(f"    {LEVEL_LOW_MAX + 1}  - {LEVEL_MEDIUM_MAX}  -> Medium")
    print(f"    {LEVEL_MEDIUM_MAX + 1} - 100 -> High")
    print(_sep(60))
