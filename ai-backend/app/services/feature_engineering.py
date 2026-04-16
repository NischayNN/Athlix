from __future__ import annotations

import logging
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

_LM_INDEX: Dict[str, int] = {
    "LEFT_SHOULDER":    11,
    "RIGHT_SHOULDER":   12,
    "LEFT_ELBOW":       13,
    "RIGHT_ELBOW":      14,
    "LEFT_WRIST":       15,
    "RIGHT_WRIST":      16,
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

_MIN_VISIBILITY: float = 0.4


def _get_lm_item(landmarks: List[PoseLandmarkItem], name: str) -> Optional[PoseLandmarkItem]:
    idx = _LM_INDEX.get(name)
    if idx is None or idx >= len(landmarks):
        return None
    lm = landmarks[idx]
    return lm if lm.visibility >= _MIN_VISIBILITY else None


def _get_point(landmarks: List[Landmark], name: str) -> Optional[Tuple[float, float, float]]:
    for lm in landmarks:
        if lm.name == name:
            return (lm.x, lm.y, lm.z) if lm.visibility >= _MIN_VISIBILITY else None
    return None


def _safe_angle(landmarks: List[Landmark], a_name: str, b_name: str, c_name: str) -> Optional[float]:
    a = _get_point(landmarks, a_name)
    b = _get_point(landmarks, b_name)
    c = _get_point(landmarks, c_name)
    if None in (a, b, c):
        return None
    return calculate_angle(a, b, c)


def _symmetry_score(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return round(1.0 - min(abs(left - right) / 180.0, 1.0), 4)


def _check_knee_valgus(landmarks: List[PoseLandmarkItem], tolerance: float) -> Optional[bool]:
    left_knee   = _get_lm_item(landmarks, "LEFT_KNEE")
    left_ankle  = _get_lm_item(landmarks, "LEFT_ANKLE")
    right_knee  = _get_lm_item(landmarks, "RIGHT_KNEE")
    right_ankle = _get_lm_item(landmarks, "RIGHT_ANKLE")

    results: List[bool] = []

    if left_knee and left_ankle:
        # Left valgus: knee moves inward (lower x) past ankle
        results.append((left_ankle.x - left_knee.x) > tolerance)

    if right_knee and right_ankle:
        # Right valgus: knee moves inward (higher x) past ankle
        results.append((right_knee.x - right_ankle.x) > tolerance)

    return any(results) if results else None


def _check_bad_back_posture(back_angle: Optional[float], threshold: float) -> Optional[bool]:
    if back_angle is None:
        return None
    return back_angle > threshold


def _check_insufficient_depth(landmarks: List[PoseLandmarkItem], margin: float) -> Optional[bool]:
    left_hip   = _get_lm_item(landmarks, "LEFT_HIP")
    left_knee  = _get_lm_item(landmarks, "LEFT_KNEE")
    right_hip  = _get_lm_item(landmarks, "RIGHT_HIP")
    right_knee = _get_lm_item(landmarks, "RIGHT_KNEE")

    results: List[bool] = []

    # y increases downward in normalised image coords; hip must be below knee (higher y)
    if left_hip and left_knee:
        results.append(not (left_hip.y > left_knee.y + margin))
    if right_hip and right_knee:
        results.append(not (right_hip.y > right_knee.y + margin))

    return any(results) if results else None


def analyze_form(
    landmarks: List[PoseLandmarkItem],
    back_angle: Optional[float],
    thresholds: Optional[FormThresholds] = None,
) -> FormFlags:
    cfg = thresholds or FormThresholds()
    return FormFlags(
        knee_valgus=_check_knee_valgus(landmarks, cfg.knee_valgus_tolerance),
        bad_back_posture=_check_bad_back_posture(back_angle, cfg.bad_back_angle_threshold),
        insufficient_depth=_check_insufficient_depth(landmarks, cfg.squat_depth_margin),
    )


def _compute_back_angle_from_landmark_schema(landmarks: List[Landmark]) -> Optional[float]:
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
    return JointAngles(
        left_knee      = _safe_angle(landmarks, "LEFT_HIP",        "LEFT_KNEE",      "LEFT_ANKLE"),
        right_knee     = _safe_angle(landmarks, "RIGHT_HIP",       "RIGHT_KNEE",     "RIGHT_ANKLE"),
        left_hip       = _safe_angle(landmarks, "LEFT_SHOULDER",   "LEFT_HIP",       "LEFT_KNEE"),
        right_hip      = _safe_angle(landmarks, "RIGHT_SHOULDER",  "RIGHT_HIP",      "RIGHT_KNEE"),
        left_ankle     = _safe_angle(landmarks, "LEFT_KNEE",       "LEFT_ANKLE",     "LEFT_FOOT_INDEX"),
        right_ankle    = _safe_angle(landmarks, "RIGHT_KNEE",      "RIGHT_ANKLE",    "RIGHT_FOOT_INDEX"),
        left_elbow     = _safe_angle(landmarks, "LEFT_SHOULDER",   "LEFT_ELBOW",     "LEFT_WRIST"),
        right_elbow    = _safe_angle(landmarks, "RIGHT_SHOULDER",  "RIGHT_ELBOW",    "RIGHT_WRIST"),
        left_shoulder  = _safe_angle(landmarks, "LEFT_ELBOW",      "LEFT_SHOULDER",  "LEFT_HIP"),
        right_shoulder = _safe_angle(landmarks, "RIGHT_ELBOW",     "RIGHT_SHOULDER", "RIGHT_HIP"),
        back           = _compute_back_angle_from_landmark_schema(landmarks),
    )


def build_feature_vector(frame_index: int, landmarks: List[Landmark]) -> BiomechanicalFeatures:
    angles = extract_joint_angles(landmarks)

    symmetry_values = [
        _symmetry_score(angles.left_knee,     angles.right_knee),
        _symmetry_score(angles.left_hip,      angles.right_hip),
        _symmetry_score(angles.left_ankle,    angles.right_ankle),
        _symmetry_score(angles.left_elbow,    angles.right_elbow),
        _symmetry_score(angles.left_shoulder, angles.right_shoulder),
    ]
    valid = [v for v in symmetry_values if v is not None]
    symmetry_score = round(sum(valid) / len(valid), 4) if valid else None

    return BiomechanicalFeatures(
        frame_index=frame_index,
        joint_angles=angles,
        symmetry_score=symmetry_score,
        form_deviation_score=None,
        predicted_risk_score=None,
    )


def predict_risk(features: BiomechanicalFeatures) -> float:
    """
    Derive an injury risk score from a BiomechanicalFeatures object.

    Delegates to risk_engine.get_risk_score() which loads the trained XGBoost
    model and applies rule-based fusion on top of the raw prediction.

    Parameters
    ----------
    features : BiomechanicalFeatures

    Returns
    -------
    float
        Final risk score in [0, 100] after fusion adjustments.
    """
    from app.services.risk_engine import get_risk_score  # lazy import avoids circular deps

    # Map the available biomechanical features onto the risk engine's input contract.
    # Fields not captured by the pose pipeline (training_load, fatigue_index, etc.)
    # fall back to the risk_engine defaults until the full athlete session context
    # is passed through the route layer.
    input_features = {
        "form_decay":      features.form_deviation_score or 0.0,
        "previous_injury": 0,  # placeholder — inject from athlete profile when available
    }

    result = get_risk_score(input_features)
    logger.info(
        "predict_risk -> score=%.1f level=%s delta=%.1f",
        result.risk_score, result.risk_level, result.fusion_delta,
    )
    return result.risk_score
