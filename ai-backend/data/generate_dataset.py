"""
generate_dataset.py
-------------------
Synthetic dataset generator and feature engineering pipeline for the
Athlix AI Injury Predictor.

Run directly:
    python ai-backend/data/generate_dataset.py

Or import the two public functions in other modules:
    from data.generate_dataset import generate_raw_dataset, engineer_features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# STEP 1 - Generate raw synthetic dataset
# ---------------------------------------------------------------------------

def generate_raw_dataset(n_samples: int = 500) -> pd.DataFrame:
    """
    Create a synthetic athlete-monitoring dataset.

    Columns
    -------
    training_load    : Daily training stress (1-10). Higher = more load on body.
    recovery_score   : How well the athlete recovered (0-100). Higher = better.
    fatigue_index    : Accumulated fatigue level (0-10). Higher = more tired.
    form_decay       : Technique degradation due to fatigue (0-1). Higher = worse form.
    previous_injury  : Binary flag - has the athlete been injured before? (0 or 1)
    injury_risk      : TARGET - estimated injury probability (0-100).

    The injury risk target is computed from a weighted combination of the raw
    features plus Gaussian noise, then clipped to [0, 100].
    """
    training_load   = np.random.uniform(1, 10, n_samples)          # daily load 1-10
    recovery_score  = np.random.uniform(0, 100, n_samples)         # sleep + HRV proxy
    fatigue_index   = np.random.uniform(0, 10, n_samples)          # cumulative fatigue
    form_decay      = np.random.uniform(0, 1, n_samples)           # technique quality loss
    previous_injury = np.random.randint(0, 2, n_samples)           # binary history flag

    # --- Composite risk score with domain-inspired weighting ---
    # High load, high fatigue, poor recovery, and prior injury all push risk up.
    noise = np.random.normal(0, 5, n_samples)
    injury_risk = (
        training_load  * 4.5   # load is the strongest driver
        + fatigue_index * 3.5  # fatigue amplifies risk
        + (100 - recovery_score) * 0.15  # poor recovery -> higher risk
        + form_decay   * 10    # bad form correlates strongly with injury
        + previous_injury * 10  # history of injury raises baseline risk
        + noise
    )
    injury_risk = np.clip(injury_risk, 0, 100)

    return pd.DataFrame({
        "training_load":   training_load.round(2),
        "recovery_score":  recovery_score.round(2),
        "fatigue_index":   fatigue_index.round(2),
        "form_decay":      form_decay.round(4),
        "previous_injury": previous_injury,
        "injury_risk":     injury_risk.round(2),
    })


# ---------------------------------------------------------------------------
# STEP 2 - Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features on top of the raw dataset.

    Derived Features
    ----------------
    ACWR (Acute:Chronic Workload Ratio)
        Industry-standard metric used in sports science.
        Acute load  = average training_load over last 7 rows (1-week window).
        Chronic load = average training_load over last 28 rows (4-week window).
        A ratio > 1.3 is generally considered a 'danger zone' for injury.

    recovery_deficit
        = 100 - recovery_score
        Flips the recovery scale so that higher values mean MORE deficit,
        making it a positive risk indicator like the other features.

    fatigue_trend
        Rolling 7-row mean of fatigue_index.
        Captures whether fatigue is building up over time rather than just
        looking at a single snapshot. A rising trend is a warning sign.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``generate_raw_dataset``.

    Returns
    -------
    pd.DataFrame
        Original columns + derived features, index preserved.
    """
    df = df.copy()  # never mutate the caller's DataFrame

    # -- ACWR ----------------------------------------------------------------
    acute_window   = 7
    chronic_window = 28

    acute_load   = df["training_load"].rolling(window=acute_window,  min_periods=1).mean()
    chronic_load = df["training_load"].rolling(window=chronic_window, min_periods=1).mean()

    # Avoid division-by-zero: if chronic_load is 0 treat ACWR as 1.0 (neutral)
    df["ACWR"] = np.where(
        chronic_load != 0,
        (acute_load / chronic_load).round(4),
        1.0,
    )

    # -- Recovery Deficit ----------------------------------------------------
    df["recovery_deficit"] = (100 - df["recovery_score"]).round(2)

    # -- Fatigue Trend (7-day rolling mean) ----------------------------------
    df["fatigue_trend"] = (
        df["fatigue_index"]
        .rolling(window=acute_window, min_periods=1)
        .mean()
        .round(4)
    )

    return df


# ---------------------------------------------------------------------------
# STEP 3 - Normalise / scale features
# ---------------------------------------------------------------------------

def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Apply Min-Max scaling to all feature columns (everything except the target).

    Min-Max scales each column to the range [0, 1], preserving the shape of the
    distribution while making all features comparable in magnitude - important
    for distance-based models and gradient-boosted trees alike.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered feature DataFrame (output of ``engineer_features``).

    Returns
    -------
    scaled_df : pd.DataFrame
        The scaled DataFrame (target column ``injury_risk`` is NOT scaled).
    scaler : MinMaxScaler
        Fitted scaler instance - save this for inference-time transforms.
    """
    feature_cols = [c for c in df.columns if c != "injury_risk"]
    target_col   = df[["injury_risk"]].copy()

    scaler     = MinMaxScaler()
    scaled_arr = scaler.fit_transform(df[feature_cols])

    scaled_df = pd.DataFrame(scaled_arr, columns=feature_cols, index=df.index)
    scaled_df["injury_risk"] = target_col.values  # re-attach unscaled target

    return scaled_df, scaler


# ---------------------------------------------------------------------------
# STEP 4 - Pretty-print feature explanations
# ---------------------------------------------------------------------------

FEATURE_EXPLANATIONS: dict[str, str] = {
    "training_load":    "Daily training intensity scored 1-10. Core driver of injury risk.",
    "recovery_score":   "Composite recovery metric 0-100 (sleep quality, HRV, soreness). Higher = better.",
    "fatigue_index":    "Accumulated fatigue level 0-10. Rises with consecutive hard sessions.",
    "form_decay":       "Technique degradation 0-1. Quantifies how much biomechanical form has degraded.",
    "previous_injury":  "Binary flag (0/1). Prior injury history raises baseline vulnerability.",
    "injury_risk":      "TARGET - estimated injury probability 0-100.",
    "ACWR":             "Acute:Chronic Workload Ratio. Values > 1.3 signal elevated overtraining risk.",
    "recovery_deficit": "= 100 - recovery_score. Positive risk indicator; higher = less recovered.",
    "fatigue_trend":    "7-day rolling mean of fatigue_index. Detects sustained fatigue build-up.",
}


def print_feature_guide(df: pd.DataFrame) -> None:
    """Print a human-readable guide for every column present in *df*."""
    sep = "-" * 65
    print("\n" + sep)
    print("  FEATURE GUIDE")
    print(sep)
    for col in df.columns:
        explanation = FEATURE_EXPLANATIONS.get(col, "No description available.")
        print(f"  {col:<22} | {explanation}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Entry point - run as a script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nAthlix - AI Injury Predictor | Dataset Pipeline\n")

    # 1. Generate raw data
    print("[1] Generating raw synthetic dataset (500 athletes)...")
    raw_df = generate_raw_dataset(n_samples=500)
    print(f"    Shape: {raw_df.shape}  (rows x columns)\n")

    # 2. Feature engineering
    print("[2] Engineering derived features...")
    engineered_df = engineer_features(raw_df)
    print(f"    Shape after engineering: {engineered_df.shape}\n")

    # 3. Scale features
    print("[3] Scaling features with Min-Max Scaler...")
    scaled_df, scaler = scale_features(engineered_df)
    print(f"    Shape after scaling:     {scaled_df.shape}\n")

    # 4. Show first 5 rows
    print("[4] First 5 rows of the ENGINEERED (pre-scale) dataset:")
    print(engineered_df.head().to_string(index=True))

    print("\n[5] First 5 rows of the SCALED dataset (features in [0, 1]):")
    print(scaled_df.head().to_string(index=True))

    # 5. Feature guide
    print_feature_guide(scaled_df)

    # 6. Basic statistics
    print("[6] Descriptive statistics (engineered dataset):")
    print(engineered_df.describe().round(2).to_string())

    # 7. Save to CSV for downstream use
    output_path = "ai-backend/data/athlete_injury_dataset.csv"
    engineered_df.to_csv(output_path, index=False)
    print(f"\n[OK] Dataset saved -> {output_path}")
