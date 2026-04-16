"""
schemas.py
----------
Pydantic models that define the shape of every API request and response within
the Athlix Injury Prediction backend.  Using explicit schemas keeps the API
contract clear, enables automatic documentation generation, and provides
runtime validation of all input / output data.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ProcessingStatus(str, Enum):
    """High-level status of a processing job."""
    SUCCESS = "success"
    PARTIAL = "partial"   # Some frames failed but the job completed
    FAILED  = "failed"


class RiskLevel(str, Enum):
    """Categorical injury-risk classification."""
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class Landmark(BaseModel):
    """Normalised 3-D position of a single pose landmark."""

    name: str = Field(..., description="Human-readable landmark name (e.g. 'LEFT_KNEE').")
    x: float  = Field(..., ge=0.0, le=1.0, description="Horizontal position relative to image width.")
    y: float  = Field(..., ge=0.0, le=1.0, description="Vertical position relative to image height.")
    z: float  = Field(..., description="Depth relative to hip centre (MediaPipe convention).")
    visibility: float = Field(0.0, ge=0.0, le=1.0, description="Landmark detection confidence [0, 1].")


class PoseLandmarkItem(BaseModel):
    """
    Compact landmark representation for the ``/detect-pose`` endpoint.
    Matches the specified output contract exactly::

        {"id": 0, "name": "NOSE", "x": 0.5, "y": 0.3, "z": -0.02,
         "visibility": 0.99}
    """

    id: int   = Field(..., ge=0, le=32, description="BlazePose landmark index (0–32).")
    name: str = Field(..., description="Human-readable landmark name (e.g. 'LEFT_KNEE').")
    x: float  = Field(..., description="Horizontal position, normalised to [0, 1] relative to image width.")
    y: float  = Field(..., description="Vertical position, normalised to [0, 1] relative to image height.")
    z: float  = Field(..., description="Depth relative to hip centre (MediaPipe convention). Can be negative.")
    visibility: float = Field(..., ge=0.0, le=1.0, description="Detection confidence for this landmark [0, 1].")


class AngleResult(BaseModel):
    """
    The three primary biomechanical angles computed for a frame.

    All values are in **degrees** and may be ``None`` if the required
    landmarks were not detected with sufficient confidence.

    Example::

        {
          "knee_angle": 162.3,
          "hip_angle":  174.7,
          "back_angle":   8.1
        }
    """

    knee_angle: Optional[float] = Field(
        None,
        description=(
            "Knee flexion angle (hip→knee→ankle), averaged left & right. "
            "~180° = fully extended, ~40° = deep squat."
        ),
    )
    hip_angle: Optional[float] = Field(
        None,
        description=(
            "Hip flexion angle (shoulder→hip→knee), averaged left & right. "
            "~180° = standing upright, ~60° = deep squat."
        ),
    )
    back_angle: Optional[float] = Field(
        None,
        description=(
            "Trunk inclination angle relative to vertical "
            "(0° = upright, 90° = horizontal lean)."
        ),
    )


class FormFlags(BaseModel):
    """
    Boolean form-quality flags derived from biomechanical rule analysis.

    Each flag is ``True`` when the corresponding fault is **detected**.
    A flag is ``None`` when the required landmarks were not visible enough
    to make a determination.

    Example::

        {
          "knee_valgus":          true,
          "bad_back_posture":     false,
          "insufficient_depth":   false
        }
    """

    knee_valgus: Optional[bool] = Field(
        None,
        description=(
            "True when one or both knees collapse inward (valgus) relative to "
            "the corresponding ankle's horizontal position."
        ),
    )
    bad_back_posture: Optional[bool] = Field(
        None,
        description=(
            "True when the trunk inclination angle exceeds the configured "
            "forward-lean threshold, indicating excessive forward bending."
        ),
    )
    insufficient_depth: Optional[bool] = Field(
        None,
        description=(
            "True when hip y-coordinate does not reach knee y-coordinate level, "
            "indicating the squat depth is shallower than required. "
            "(In normalised image coordinates y increases downward.)"
        ),
    )


class FormThresholds(BaseModel):
    """
    Configurable thresholds for form-analysis rules.

    All thresholds have sensible defaults but can be overridden per-request
    to support different exercises or athlete profiles.
    """

    knee_valgus_tolerance: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description=(
            "Maximum allowed inward x-offset of the knee relative to the ankle "
            "(normalised image width). Knee is flagged as valgus when its x "
            "differs from the ankle's x by more than this amount in the inward "
            "direction.  Default: 0.05 (~5% of frame width)."
        ),
    )
    bad_back_angle_threshold: float = Field(
        35.0,
        ge=0.0,
        le=90.0,
        description=(
            "Back inclination angle (degrees) above which the posture is flagged "
            "as excessive forward lean.  Default: 35°."
        ),
    )
    squat_depth_margin: float = Field(
        0.02,
        ge=0.0,
        le=0.2,
        description=(
            "How much lower (in normalised y) the hip must be than the knee to "
            "count as sufficient depth.  In image coords, y increases downward so "
            "hip.y > knee.y means the hip is lower.  Default: 0.02."
        ),
    )


class PoseDetectionResponse(BaseModel):
    """
    Response shape for ``POST /detect-pose``.

    Example::

        {
          "pose_detected": true,
          "landmark_count": 33,
          "processing_time_ms": 52.4,
          "landmarks": [
            {"id": 0, "name": "NOSE", "x": 0.51, "y": 0.12, "z": -0.01, "visibility": 0.98},
            ...
          ],
          "angles": {
            "knee_angle": 162.3,
            "hip_angle": 174.7,
            "back_angle": 8.1
          }
        }
    """

    pose_detected: bool
    landmark_count: int               = Field(..., description="Number of landmarks returned (0 if no pose detected).")
    processing_time_ms: float         = Field(..., description="Server-side inference time in milliseconds.")
    landmarks: List[PoseLandmarkItem] = Field(default_factory=list)
    angles: Optional[AngleResult]  = Field(
        None,
        description="Computed joint angles. Null when no pose is detected.",
    )
    form_flags: Optional[FormFlags] = Field(
        None,
        description="Form-quality flags from rule-based analysis. Null when no pose is detected.",
    )


class JointAngles(BaseModel):
    """Calculated biomechanical angles (degrees) for a single frame."""

    left_knee:   Optional[float] = Field(None, description="Knee flexion angle on the left side.")
    right_knee:  Optional[float] = Field(None, description="Knee flexion angle on the right side.")
    left_hip:    Optional[float] = Field(None, description="Hip flexion angle on the left side.")
    right_hip:   Optional[float] = Field(None, description="Hip flexion angle on the right side.")
    left_ankle:  Optional[float] = Field(None, description="Ankle dorsiflexion angle on the left side.")
    right_ankle: Optional[float] = Field(None, description="Ankle dorsiflexion angle on the right side.")
    left_elbow:  Optional[float] = Field(None, description="Elbow flexion angle on the left side.")
    right_elbow: Optional[float] = Field(None, description="Elbow flexion angle on the right side.")
    left_shoulder:  Optional[float] = Field(None, description="Shoulder angle on the left side.")
    right_shoulder: Optional[float] = Field(None, description="Shoulder angle on the right side.")
    back: Optional[float] = Field(None, description="Trunk / lumbar spine angle estimate.")


class BiomechanicalFeatures(BaseModel):
    """
    Engineered feature vector for a single frame, ready for ML model inference.
    All values are placeholders until the ML model is integrated.
    """

    frame_index: int = Field(..., description="Zero-based frame index within the video.")
    joint_angles: JointAngles
    symmetry_score: Optional[float] = Field(
        None,
        ge=0.0, le=1.0,
        description="Left-right movement symmetry (1.0 = perfect symmetry).",
    )
    form_deviation_score: Optional[float] = Field(
        None,
        ge=0.0, le=1.0,
        description="How much the posture deviates from ideal form (0 = perfect form).",
    )
    # Placeholder — populated after ML model integration
    predicted_risk_score: Optional[float] = Field(
        None,
        ge=0.0, le=100.0,
        description="Predicted injury risk score [0–100]. Null until ML model is wired.",
    )


class FrameResult(BaseModel):
    """Processing result for one video frame or uploaded image."""

    frame_index: int
    pose_detected: bool
    landmarks: Optional[List[Landmark]] = None
    features: Optional[BiomechanicalFeatures]   = None
    error_message: Optional[str]                = None


# ---------------------------------------------------------------------------
# Top-level response bodies
# ---------------------------------------------------------------------------

class VideoProcessingResponse(BaseModel):
    """
    Full response returned after processing an uploaded video file.
    Contains per-frame results and aggregate statistics.
    """

    status: ProcessingStatus
    total_frames: int     = Field(..., description="Total frames extracted from the video.")
    processed_frames: int = Field(..., description="Frames on which pose detection was attempted.")
    detected_frames: int  = Field(..., description="Frames where at least one pose was detected.")
    processing_time_ms: float = Field(..., description="Total server-side processing time in ms.")
    frame_results: List[FrameResult]
    # Aggregate risk summary (placeholder until ML model is integrated)
    aggregate_risk_score: Optional[float]   = Field(None, ge=0.0, le=100.0)
    aggregate_risk_level: Optional[RiskLevel] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class FrameProcessingResponse(BaseModel):
    """
    Response returned after processing a single uploaded image/frame.
    """

    status: ProcessingStatus
    processing_time_ms: float
    result: FrameResult
    metadata: Dict[str, str] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """API health-check payload."""

    status: str   = "ok"
    version: str
    ml_model_loaded: bool = Field(
        False,
        description="Whether an ML model is currently in memory (False until integration).",
    )
