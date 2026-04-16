"""
feature_engineering.py
-----------------------
Transforms raw MediaPipe pose landmarks into a structured biomechanical
feature vector that can be consumed by the downstream ML model.

Design decisions
~~~~~~~~~~~~~~~~
* Thin service layer — no MediaPipe imports here; all pose processing is
  absorbed by *PoseService*.  This module only knows about *Landmark* and
  *BiomechanicalFeatures* schemas.
* Angle calculations are delegated to *angle_utils* for testability.
* Future ML-model inference will be plugged in at the ``predict_risk`` hook
  at the bottom of this module.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from app.models.schemas import BiomechanicalFeatures, JointAngles, Landmark
from app.utils.angle_utils import calculate_angle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe landmark name → index mapping (BlazePose 33-point model)
# We only reference the subset relevant to injury risk.
# ---------------------------------------------------------------------------
_LM_INDEX: Dict[str, int] = {
    # Upper body
    "LEFT_SHOULDER":  11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW":     13,
    "RIGHT_ELBOW":    14,
    "LEFT_WRIST":     15,
    "RIGHT_WRIST":    16,
    # Lower body
    "LEFT_HIP":       23,
    "RIGHT_HIP":      24,
    "LEFT_KNEE":      25,
    "RIGHT_KNEE":     26,
    "LEFT_ANKLE":     27,
    "RIGHT_ANKLE":    28,
    "LEFT_HEEL":      29,
    "RIGHT_HEEL":     30,
    "LEFT_FOOT_INDEX":  31,
    "RIGHT_FOOT_INDEX": 32,
}

# Minimum visibility threshold — landmarks below this are treated as missing
_MIN_VISIBILITY: float = 0.4


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_point(
    landmarks: List[Landmark], name: str
) -> Optional[Tuple[float, float, float]]:
    """
    Return the (x, y, z) tuple for a named landmark, or ``None`` if the
    landmark is missing / below the visibility threshold.
    """
    target_idx = _LM_INDEX.get(name)
    if target_idx is None:
        return None

    for lm in landmarks:
        # landmarks list is ordered 0-32; name lookup is O(n) but n=33
        if lm.name == name:
            if lm.visibility < _MIN_VISIBILITY:
                logger.debug("Landmark %s skipped — visibility %.2f below threshold.", name, lm.visibility)
                return None
            return (lm.x, lm.y, lm.z)

    return None


def _safe_angle(
    landmarks: List[Landmark], a_name: str, b_name: str, c_name: str
) -> Optional[float]:
    """
    Calculate the angle at joint *b* given three landmark names.
    Returns ``None`` if any landmark is missing or below visibility threshold.
    """
    a = _get_point(landmarks, a_name)
    b = _get_point(landmarks, b_name)
    c = _get_point(landmarks, c_name)

    if None in (a, b, c):
        return None

    return calculate_angle(a, b, c)


def _symmetry_score(left_angle: Optional[float], right_angle: Optional[float]) -> Optional[float]:
    """
    Compute left-right symmetry for a given joint pair.

    Returns a score in [0, 1] where 1 means perfect symmetry.
    Returns ``None`` if either angle is unavailable.
    """
    if left_angle is None or right_angle is None:
        return None

    # Maximum expected angular difference for complete asymmetry is 180°
    diff = abs(left_angle - right_angle)
    return round(1.0 - min(diff / 180.0, 1.0), 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_joint_angles(landmarks: List[Landmark]) -> JointAngles:
    """
    Compute all joint angles from the landmark list.

    The three-point angle convention used throughout is:
        (proximal_joint, vertex_joint, distal_joint)

    E.g. knee angle: hip → knee → ankle
    """
    return JointAngles(
        # ── Knees ──────────────────────────────────────────────────────
        left_knee=_safe_angle(landmarks, "LEFT_HIP",  "LEFT_KNEE",  "LEFT_ANKLE"),
        right_knee=_safe_angle(landmarks, "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),

        # ── Hips ────────────────────────────────────────────────────────
        left_hip=_safe_angle(landmarks, "LEFT_SHOULDER",  "LEFT_HIP",  "LEFT_KNEE"),
        right_hip=_safe_angle(landmarks, "RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),

        # ── Ankles (dorsiflexion proxy) ─────────────────────────────────
        left_ankle=_safe_angle(landmarks, "LEFT_KNEE",  "LEFT_ANKLE",  "LEFT_FOOT_INDEX"),
        right_ankle=_safe_angle(landmarks, "RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),

        # ── Elbows ───────────────────────────────────────────────────────
        left_elbow=_safe_angle(landmarks, "LEFT_SHOULDER",  "LEFT_ELBOW",  "LEFT_WRIST"),
        right_elbow=_safe_angle(landmarks, "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),

        # ── Shoulders ────────────────────────────────────────────────────
        left_shoulder=_safe_angle(landmarks, "LEFT_ELBOW",  "LEFT_SHOULDER",  "LEFT_HIP"),
        right_shoulder=_safe_angle(landmarks, "RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),

        # ── Back (trunk inclination proxy) ───────────────────────────────
        # Uses mid-shoulder → mid-hip vector relative to vertical
        back=_compute_back_angle(landmarks),
    )


def _compute_back_angle(landmarks: List[Landmark]) -> Optional[float]:
    """
    Estimate trunk / back angle using the midpoint between shoulders and hips.
    The angle is measured relative to the gravitational vertical (downward = 0°).
    """
    ls = _get_point(landmarks, "LEFT_SHOULDER")
    rs = _get_point(landmarks, "RIGHT_SHOULDER")
    lh = _get_point(landmarks, "LEFT_HIP")
    rh = _get_point(landmarks, "RIGHT_HIP")

    if None in (ls, rs, lh, rh):
        return None

    # Midpoints
    mid_shoulder = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2, (ls[2] + rs[2]) / 2)
    mid_hip      = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2, (lh[2] + rh[2]) / 2)

    # Reference point directly above the hip (vertical axis)
    vertical_ref = (mid_hip[0], mid_hip[1] - 1.0, mid_hip[2])

    return calculate_angle(mid_shoulder, mid_hip, vertical_ref)


def build_feature_vector(
    frame_index: int,
    landmarks: List[Landmark],
) -> BiomechanicalFeatures:
    """
    Build the full ``BiomechanicalFeatures`` object for a single frame.

    This is the main entry point called by the route handler.  The risk score
    field is intentionally left ``None`` until the ML model is integrated.

    Parameters
    ----------
    frame_index : int
        Zero-based frame index.
    landmarks : list of Landmark
        Parsed output from ``PoseService.process_frame``.

    Returns
    -------
    BiomechanicalFeatures
    """
    angles = extract_joint_angles(landmarks)

    # --- Symmetry: average across available joint pairs ---
    symmetry_values = [
        _symmetry_score(angles.left_knee,     angles.right_knee),
        _symmetry_score(angles.left_hip,      angles.right_hip),
        _symmetry_score(angles.left_ankle,    angles.right_ankle),
        _symmetry_score(angles.left_elbow,    angles.right_elbow),
        _symmetry_score(angles.left_shoulder, angles.right_shoulder),
    ]
    valid_symmetry = [v for v in symmetry_values if v is not None]
    symmetry_score = round(sum(valid_symmetry) / len(valid_symmetry), 4) if valid_symmetry else None

    return BiomechanicalFeatures(
        frame_index=frame_index,
        joint_angles=angles,
        symmetry_score=symmetry_score,
        form_deviation_score=None,   # ← Placeholder: populate with rule-based or ML logic
        predicted_risk_score=None,   # ← Placeholder: ML model hook
    )


# ---------------------------------------------------------------------------
# Future ML model integration hook
# ---------------------------------------------------------------------------

def predict_risk(features: BiomechanicalFeatures) -> float:
    """
    Placeholder for ML model inference.

    Replace the body of this function once the XGBoost / neural-net model is
    serialised and loaded via ``model/model.pkl``.

    Parameters
    ----------
    features : BiomechanicalFeatures

    Returns
    -------
    float
        Predicted risk score in the range [0, 100].
    """
    # TODO: Load model from disk (joblib.load) and call model.predict()
    logger.debug("predict_risk called — ML model not yet integrated, returning placeholder.")
    return 0.0
