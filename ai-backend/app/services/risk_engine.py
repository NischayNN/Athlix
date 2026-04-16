from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field

import joblib
import numpy as np

_SERVICES_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR  = os.path.abspath(os.path.join(_SERVICES_DIR, "..", ".."))
_DATA_DIR     = os.path.join(_BACKEND_DIR, "data")

if _SERVICES_DIR not in sys.path:
    sys.path.insert(0, _SERVICES_DIR)

from generate_dataset import engineer_features, generate_raw_dataset

logger = logging.getLogger(__name__)

LEVEL_LOW_MAX    = 30
LEVEL_MEDIUM_MAX = 70

FATIGUE_HIGH_THRESHOLD    = 7.0
FORM_DECAY_HIGH_THRESHOLD = 0.50   # lowered: triggers on moderately bad form
RECOVERY_LOW_THRESHOLD    = 35.0

FUSION_FATIGUE_FORM_BOOST  = 15.0  # raised: bad form + fatigue is high risk
FUSION_RECOVERY_BOOST      = 8.0   # raised: poor recovery amplifies risk
FUSION_TRIPLE_THREAT_BONUS = 6.0   # raised: all three bad → serious penalty


@dataclass
class RiskOutput:
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


_MODEL_CACHE: dict = {}

def _load_model(name: str = "xgboost"):
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]

    pkl_path = os.path.join(_DATA_DIR, f"{name}.pkl")

    if not os.path.exists(pkl_path):
        fallback = "random_forest" if name == "xgboost" else "xgboost"
        fallback_path = os.path.join(_DATA_DIR, f"{fallback}.pkl")
        if os.path.exists(fallback_path):
            pkl_path = fallback_path
            name = fallback
        else:
            raise FileNotFoundError(
                f"No trained model found in {_DATA_DIR}."
            )

    model = joblib.load(pkl_path)
    _MODEL_CACHE[name] = model
    return model


def _classify_level(score: float) -> str:
    if score <= LEVEL_LOW_MAX:
        return "Low"
    elif score <= LEVEL_MEDIUM_MAX:
        return "Medium"
    return "High"


def _validate_input(features: dict) -> dict:
    defaults = {
        "training_load":   5.0,
        "recovery_score":  50.0,
        "fatigue_index":   5.0,
        "form_decay":      0.5,
        "previous_injury": 0,
    }

    validated = {k: features.get(k, v) for k, v in defaults.items()}

    if not (1 <= validated["training_load"] <= 10):
        raise ValueError(f"training_load must be 1-10")
    if not (0 <= validated["recovery_score"] <= 100):
        raise ValueError(f"recovery_score must be 0-100")
    if not (0 <= validated["fatigue_index"] <= 10):
        raise ValueError(f"fatigue_index must be 0-10")
    if not (0 <= validated["form_decay"] <= 1):
        raise ValueError(f"form_decay must be 0-1")
    if validated["previous_injury"] not in (0, 1):
        raise ValueError(f"previous_injury must be 0 or 1")

    return validated


def _apply_fusion(
    model_score: float,
    features: dict,
) -> tuple[float, float, list[str]]:
    delta = 0.0
    flags: list[str] = []

    fatigue_high  = features["fatigue_index"]  > FATIGUE_HIGH_THRESHOLD
    form_bad      = features["form_decay"]     > FORM_DECAY_HIGH_THRESHOLD
    recovery_low  = features["recovery_score"] < RECOVERY_LOW_THRESHOLD

    if fatigue_high and form_bad:
        delta += FUSION_FATIGUE_FORM_BOOST
        flags.append(
            f"High fatigue ({features['fatigue_index']:.1f}) + "
            f"poor form ({features['form_decay']:.2f}) +{FUSION_FATIGUE_FORM_BOOST:.0f}"
        )

    if recovery_low:
        delta += FUSION_RECOVERY_BOOST
        flags.append(
            f"Low recovery ({features['recovery_score']:.1f}) +{FUSION_RECOVERY_BOOST:.0f}"
        )

    if fatigue_high and form_bad and recovery_low:
        delta += FUSION_TRIPLE_THREAT_BONUS
        flags.append(f"Triple-threat +{FUSION_TRIPLE_THREAT_BONUS:.0f}")

    adjusted = float(np.clip(model_score + delta, 0.0, 100.0))
    return adjusted, delta, flags


def get_risk_score(
    input_features: dict,
    model_name: str = "xgboost",
) -> RiskOutput:
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    features = _validate_input(input_features)

    raw_df = pd.DataFrame([features])
    eng_df = engineer_features(raw_df)

    ref_raw    = generate_raw_dataset(n_samples=500)
    ref_eng    = engineer_features(ref_raw)
    feat_cols  = [c for c in ref_eng.columns if c != "injury_risk"]

    scaler = MinMaxScaler()
    scaler.fit(ref_eng[feat_cols])

    X = scaler.transform(eng_df[feat_cols])

    model       = _load_model(model_name)
    model_score = float(np.clip(model.predict(X)[0], 0.0, 100.0))

    # ------------------------------------------------------------------
    # Direct weighted formula — form_decay gets 50% of the weight so
    # that real video quality strongly drives the final risk score.
    #
    # Scaling inputs to a 0-100 space:
    #   form_decay   : already 0-1  → × 100
    #   fatigue_index: 0-10         → × 10
    #   training_load: 1-10         → × 10
    #   recovery_score: 0-100       → already in range
    # ------------------------------------------------------------------
    fd = features["form_decay"]    * 100.0   # 0-100
    fi = features["fatigue_index"] * 10.0    # 0-100
    ld = features["training_load"] * 10.0    # 0-100
    rs = features["recovery_score"]           # 0-100

    direct_score = (
        0.50 * fd +
        0.25 * fi +
        0.15 * ld +
        0.10 * (100.0 - rs)
    )
    direct_score = float(np.clip(direct_score, 0.0, 100.0))

    # Blend: 35% ML model (generalisation) + 65% direct formula (form sensitivity)
    blended_score = float(np.clip(0.35 * model_score + 0.65 * direct_score, 0.0, 100.0))

    print(f"[DEBUG] form_decay      : {features['form_decay']:.4f}  ({fd:.1f}/100)")
    print(f"[DEBUG] direct_score    : {direct_score:.2f}")
    print(f"[DEBUG] model_score     : {model_score:.2f}")
    print(f"[DEBUG] blended_score   : {blended_score:.2f}")

    final_score, delta, flags = _apply_fusion(blended_score, features)

    print(f"[DEBUG] final risk      : {final_score:.2f}")

    level = _classify_level(final_score)

    return RiskOutput(
        risk_score=round(final_score, 2),
        risk_level=level,
        model_score=round(model_score, 2),
        fusion_delta=round(delta, 2),
        flags=flags,
    )
