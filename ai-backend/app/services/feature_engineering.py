"""
feature_engineering.py
-----------------------
Two responsibilities:

1. **Form analysis** — rule-based biomechanical fault detection from
   ``PoseLandmarkItem`` objects (the compact, id-indexed type produced by
   ``pose_service.detect_pose``).  This is the primary new capability added in
   this iteration.

2. **Feature vector building** — transforms the richer ``Landmark`` objects
   (used by the upload / video pipeline) into a ``BiomechanicalFeatures``
   struct ready for downstream ML model inference.

Design decisions
~~~~~~~~~~~~~~~~
* Zero MediaPipe imports here — pose processing is absorbed by ``pose_service``.
* All three form checks are implemented in dedicated private functions so
  each rule can be unit-tested independently.
* Thresholds are injected as a ``FormThresholds`` dataclass (with field
  defaults) so they can be overridden per-request without touching code.
* The ML-model inference hook (``predict_risk``) is intentionally left as a
  stub until the XGBoost model is serialised and loaded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.models.schemas import (
    BiomechanicalFeatures,
    FormFlags,
    FormThresholds,
    JointAngles,
    Landmark,
    PoseLandmarkItem,
)
from app.utils.angle_utils import calculate_angle

logger = logging.getLogger(__name__)


# ===========================================================================
# ─── Landmark index map ────────────────────────────────────────────────────
# ===========================================================================

_LM_INDEX: Dict[str, int] = {
    # Upper body
    "LEFT_SHOULDER":    11,
    "RIGHT_SHOULDER":   12,
    "LEFT_ELBOW":       13,
    "RIGHT_ELBOW":      14,
    "LEFT_WRIST":       15,
    "RIGHT_WRIST":      16,
    # Lower body
    "LEFT_HIP":         23,
    "RIGHT_HIP":        24,
    "LEFT_KNEE":        25,
    "RIGHT_KNEE":       26,
    "LEFT_ANKLE":       27,
    "RIGHT_ANKLE":      28,
    "LEFT_HEEL":        29,
    "RIGHT_HEEL":       30,
    "LEFT_FOOT_INDEX":  31,
    "RIGHT_FOOT_INDEX": 32,
}

_MIN_VISIBILITY: float = 0.4   # Landmarks below this are treated as missing


# ===========================================================================
# ─── Internal helpers (shared by both pipelines) ───────────────────────────
# ===========================================================================

def _get_lm_item(
    landmarks: List[PoseLandmarkItem], name: str
) -> Optional[PoseLandmarkItem]:
    """
    Fetch a ``PoseLandmarkItem`` by name via its index.

    Returns ``None`` if:
    * The name is not in ``_LM_INDEX``.
    * The landmark list is too short (BlazePose always returns 33).
    * Visibility is below ``_MIN_VISIBILITY``.
    """
    idx = _LM_INDEX.get(name)
    if idx is None or idx >= len(landmarks):
        return None
    lm = landmarks[idx]
    if lm.visibility < _MIN_VISIBILITY:
        logger.debug("'%s' visibility %.2f below threshold — skipped.", name, lm.visibility)
        return None
    return lm


def _get_point(
    landmarks: List[Landmark], name: str
) -> Optional[Tuple[float, float, float]]:
    """
    Return ``(x, y, z)`` for a named ``Landmark`` (upload/video pipeline).

    Returns ``None`` if the landmark is missing or below visibility threshold.
    """
    for lm in landmarks:
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
    Calculate the angle at joint *b* from the ``Landmark`` list.
    Returns ``None`` if any landmark is missing / below threshold.
    """
    a = _get_point(landmarks, a_name)
    b = _get_point(landmarks, b_name)
    c = _get_point(landmarks, c_name)
    if None in (a, b, c):
        return None
    return calculate_angle(a, b, c)


def _symmetry_score(
    left_angle: Optional[float], right_angle: Optional[float]
) -> Optional[float]:
    """
    Left-right angular symmetry in [0, 1]  (1.0 = perfect symmetry).
    Returns ``None`` if either angle is unavailable.
    """
    if left_angle is None or right_angle is None:
        return None
    diff = abs(left_angle - right_angle)
    return round(1.0 - min(diff / 180.0, 1.0), 4)


# ===========================================================================
# ─── Form Analysis ─────────────────────────────────────────────────────────
# ===========================================================================

def _check_knee_valgus(
    landmarks: List[PoseLandmarkItem],
    tolerance: float,
) -> Optional[bool]:
    """
    Detect **knee valgus** — knee collapsing inward relative to the ankle.

    Method
    ------
    In a frontal or near-frontal view, a valgus knee moves *medially* past the
    ankle.  In normalised image coordinates:

    * For the **left** leg: left knee x should be *right of* (greater than)
      left ankle x.  Valgus → knee x < ankle x by more than ``tolerance``.
    * For the **right** leg: right knee x should be *left of* (less than)
      right ankle x.  Valgus → knee x > ankle x by more than ``tolerance``.

    The flag is ``True`` if *either* side shows valgus.

    Parameters
    ----------
    landmarks : list of PoseLandmarkItem
    tolerance : float
        Maximum allowed inward deviation (normalised width units).

    Returns
    -------
    bool | None
        ``True``  → valgus detected on at least one side.
        ``False`` → no valgus on either detectable side.
        ``None``  → both knees / ankles occluded; cannot determine.
    """
    left_knee  = _get_lm_item(landmarks, "LEFT_KNEE")
    left_ankle = _get_lm_item(landmarks, "LEFT_ANKLE")
    right_knee  = _get_lm_item(landmarks, "RIGHT_KNEE")
    right_ankle = _get_lm_item(landmarks, "RIGHT_ANKLE")

    results: List[bool] = []

    # Left leg: knee should sit to the right of the ankle (higher x)
    if left_knee is not None and left_ankle is not None:
        # valgus when knee is inward (to the left = smaller x) past tolerance
        deviation = left_ankle.x - left_knee.x   # positive = knee is medial
        results.append(deviation > tolerance)
        logger.debug(
            "Left knee valgus check: knee_x=%.4f ankle_x=%.4f deviation=%.4f threshold=%.4f → %s",
            left_knee.x, left_ankle.x, deviation, tolerance, results[-1],
        )

    # Right leg: knee should sit to the left of the ankle (smaller x)
    if right_knee is not None and right_ankle is not None:
        # valgus when knee is inward (to the right = larger x) past tolerance
        deviation = right_knee.x - right_ankle.x  # positive = knee is medial
        results.append(deviation > tolerance)
        logger.debug(
            "Right knee valgus check: knee_x=%.4f ankle_x=%.4f deviation=%.4f threshold=%.4f → %s",
            right_knee.x, right_ankle.x, deviation, tolerance, results[-1],
        )

    if not results:
        return None   # Cannot determine — all landmarks occluded

    return any(results)


def _check_bad_back_posture(
    back_angle: Optional[float],
    threshold: float,
) -> Optional[bool]:
    """
    Detect **excessive forward bending** using the trunk inclination angle.

    Parameters
    ----------
    back_angle : float | None
        Trunk inclination in degrees (0° = upright, 90° = fully horizontal).
    threshold : float
        Angle above which forward lean is flagged as excessive.

    Returns
    -------
    bool | None
        ``True``  → excessive forward lean detected.
        ``False`` → posture is within acceptable range.
        ``None``  → back angle could not be computed.
    """
    if back_angle is None:
        return None
    result = back_angle > threshold
    logger.debug(
        "Back posture check: back_angle=%.2f threshold=%.2f → %s",
        back_angle, threshold, result,
    )
    return result


def _check_insufficient_depth(
    landmarks: List[PoseLandmarkItem],
    margin: float,
) -> Optional[bool]:
    """
    Detect **insufficient squat depth** — hips not reaching knee level.

    Method
    ------
    In normalised image coordinates, **y increases downward**.  For the hips to
    be "at or below" knee level, hip.y must be >= knee.y.  We require hip.y to
    exceed knee.y by at least ``margin`` to allow some measurement noise.

    Both sides are checked; the flag is ``True`` if *either* side is
    insufficient.

    Parameters
    ----------
    landmarks : list of PoseLandmarkItem
    margin : float
        Required y-difference (normalised). hip.y must be > knee.y + margin.

    Returns
    -------
    bool | None
        ``True``  → squat depth is insufficient on at least one side.
        ``False`` → depth is adequate on all detectable sides.
        ``None``  → required landmarks not visible.
    """
    left_hip   = _get_lm_item(landmarks, "LEFT_HIP")
    left_knee  = _get_lm_item(landmarks, "LEFT_KNEE")
    right_hip  = _get_lm_item(landmarks, "RIGHT_HIP")
    right_knee = _get_lm_item(landmarks, "RIGHT_KNEE")

    results: List[bool] = []

    if left_hip is not None and left_knee is not None:
        depth_ok = left_hip.y > (left_knee.y + margin)
        results.append(not depth_ok)
        logger.debug(
            "Left squat depth: hip_y=%.4f knee_y=%.4f margin=%.4f → depth_ok=%s",
            left_hip.y, left_knee.y, margin, depth_ok,
        )

    if right_hip is not None and right_knee is not None:
        depth_ok = right_hip.y > (right_knee.y + margin)
        results.append(not depth_ok)
        logger.debug(
            "Right squat depth: hip_y=%.4f knee_y=%.4f margin=%.4f → depth_ok=%s",
            right_hip.y, right_knee.y, margin, depth_ok,
        )

    if not results:
        return None

    return any(results)


# ---------------------------------------------------------------------------
# Public form analysis entry-point
# ---------------------------------------------------------------------------

def analyze_form(
    landmarks: List[PoseLandmarkItem],
    back_angle: Optional[float],
    thresholds: Optional[FormThresholds] = None,
) -> FormFlags:
    """
    Run all three form-quality checks and return a ``FormFlags`` object.

    This is the primary entry-point called by ``pose_service.detect_pose``.

    Parameters
    ----------
    landmarks : list of PoseLandmarkItem
        All 33 BlazePose landmarks for one frame.
    back_angle : float | None
        Pre-computed trunk inclination angle from ``compute_all_angles()``.
        Passed in to avoid recomputing it here.
    thresholds : FormThresholds | None
        Custom thresholds.  Defaults are used when ``None`` is passed.

    Returns
    -------
    FormFlags
        ``{"knee_valgus": ..., "bad_back_posture": ..., "insufficient_depth": ...}``
    """
    cfg = thresholds or FormThresholds()

    knee_valgus       = _check_knee_valgus(landmarks, cfg.knee_valgus_tolerance)
    bad_back_posture  = _check_bad_back_posture(back_angle, cfg.bad_back_angle_threshold)
    insufficient_depth = _check_insufficient_depth(landmarks, cfg.squat_depth_margin)

    flags = FormFlags(
        knee_valgus=knee_valgus,
        bad_back_posture=bad_back_posture,
        insufficient_depth=insufficient_depth,
    )
    logger.info(
        "Form analysis — knee_valgus=%s bad_back=%s insufficient_depth=%s",
        knee_valgus, bad_back_posture, insufficient_depth,
    )
    return flags


# ===========================================================================
# ─── Feature vector building (upload/video pipeline) ──────────────────────
# ===========================================================================

def _compute_back_angle_from_landmark_schema(landmarks: List[Landmark]) -> Optional[float]:
    """
    Estimate trunk inclination from the richer ``Landmark`` schema.
    Used only by the upload/video pipeline; the pose-detection pipeline
    uses ``angle_utils.compute_back_angle`` which works on ``PoseLandmarkItem``.
    """
    ls = _get_point(landmarks, "LEFT_SHOULDER")
    rs = _get_point(landmarks, "RIGHT_SHOULDER")
    lh = _get_point(landmarks, "LEFT_HIP")
    rh = _get_point(landmarks, "RIGHT_HIP")

    if None in (ls, rs, lh, rh):
        return None

    mid_shoulder = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2, (ls[2] + rs[2]) / 2)
    mid_hip      = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2, (lh[2] + rh[2]) / 2)
    vertical_ref = (mid_hip[0], mid_hip[1] - 1.0, mid_hip[2])

    return calculate_angle(mid_shoulder, mid_hip, vertical_ref)


def extract_joint_angles(landmarks: List[Landmark]) -> JointAngles:
    """
    Compute the full ``JointAngles`` struct from the ``Landmark`` list
    (used by the upload/video endpoint pipeline).
    """
    return JointAngles(
        left_knee  = _safe_angle(landmarks, "LEFT_HIP",  "LEFT_KNEE",  "LEFT_ANKLE"),
        right_knee = _safe_angle(landmarks, "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
        left_hip   = _safe_angle(landmarks, "LEFT_SHOULDER",  "LEFT_HIP",  "LEFT_KNEE"),
        right_hip  = _safe_angle(landmarks, "RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
        left_ankle  = _safe_angle(landmarks, "LEFT_KNEE",  "LEFT_ANKLE",  "LEFT_FOOT_INDEX"),
        right_ankle = _safe_angle(landmarks, "RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
        left_elbow  = _safe_angle(landmarks, "LEFT_SHOULDER",  "LEFT_ELBOW",  "LEFT_WRIST"),
        right_elbow = _safe_angle(landmarks, "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
        left_shoulder  = _safe_angle(landmarks, "LEFT_ELBOW",  "LEFT_SHOULDER",  "LEFT_HIP"),
        right_shoulder = _safe_angle(landmarks, "RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),
        back = _compute_back_angle_from_landmark_schema(landmarks),
    )


def build_feature_vector(
    frame_index: int,
    landmarks: List[Landmark],
) -> BiomechanicalFeatures:
    """
    Build the full ``BiomechanicalFeatures`` object for a single frame
    (used by the upload/video endpoint pipeline).

    The risk score is intentionally left ``None`` until the ML model is
    integrated — see ``predict_risk()`` below.

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
        form_deviation_score=None,   # ← Placeholder: rule-based or ML logic
        predicted_risk_score=None,   # ← Placeholder: ML model hook
    )


# ===========================================================================
# ─── ML model integration hook ─────────────────────────────────────────────
# ===========================================================================

def predict_risk(features: BiomechanicalFeatures) -> float:
    """
    Placeholder for ML model inference.

    Replace the body of this function once the XGBoost model is
    serialised and loaded via ``model/model.pkl``.

    Parameters
    ----------
    features : BiomechanicalFeatures

    Returns
    -------
    float
        Predicted risk score in [0, 100].
    """
    # TODO: joblib.load("model/model.pkl") → model.predict(feature_vector)
    logger.debug("predict_risk called — ML model not yet integrated, returning placeholder.")
    return 0.0
