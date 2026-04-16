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
