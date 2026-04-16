from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED  = "failed"


class RiskLevel(str, Enum):
    """Categorical injury-risk classification."""
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


# ---------------------------------------------------------------------------
# Risk Engine output schema
# ---------------------------------------------------------------------------

class RiskOutput(BaseModel):
    """
    Structured injury risk result returned by the Risk Engine.

    Mirrors the RiskOutput dataclass in app/services/risk_engine.py so that
    FastAPI routes can serialise the result directly.

    Example response::

        {
          "risk_score": 78.4,
          "risk_level": "High",
          "model_score": 66.4,
          "fusion_delta": 12.0,
          "flags": [
            "High fatigue (8.1) + poor form (0.82) +8",
            "Low recovery (25.0) +5"
          ]
        }
    """

    risk_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Final injury risk score after fusion adjustments [0-100].",
    )
    risk_level: str = Field(
        ...,
        description="Categorical level: 'Low' (0-30), 'Medium' (31-70), or 'High' (71-100).",
    )
    model_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Raw ML model prediction before rule-based fusion.",
    )
    fusion_delta: float = Field(
        ...,
        description="Total score adjustment applied by the risk fusion rules.",
    )
    flags: List[str] = Field(
        default_factory=list,
        description="Human-readable list of fusion rules that were triggered.",
    )


class Landmark(BaseModel):
    name: str
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    z: float
    visibility: float = Field(0.0, ge=0.0, le=1.0)


class PoseLandmarkItem(BaseModel):
    id: int   = Field(..., ge=0, le=32)
    name: str
    x: float
    y: float
    z: float
    visibility: float = Field(..., ge=0.0, le=1.0)


class AngleResult(BaseModel):
    knee_angle: Optional[float] = None
    hip_angle:  Optional[float] = None
    back_angle: Optional[float] = None


class FormFlags(BaseModel):
    knee_valgus:        Optional[bool] = None
    bad_back_posture:   Optional[bool] = None
    insufficient_depth: Optional[bool] = None


class FormThresholds(BaseModel):
    knee_valgus_tolerance:    float = Field(0.05, ge=0.0, le=1.0)
    bad_back_angle_threshold: float = Field(35.0, ge=0.0, le=90.0)
    squat_depth_margin:       float = Field(0.02, ge=0.0, le=0.2)


class PoseDetectionResponse(BaseModel):
    pose_detected:      bool
    landmark_count:     int
    processing_time_ms: float
    landmarks:  List[PoseLandmarkItem] = Field(default_factory=list)
    angles:     Optional[AngleResult]  = None
    form_flags: Optional[FormFlags]    = None


class JointAngles(BaseModel):
    left_knee:      Optional[float] = None
    right_knee:     Optional[float] = None
    left_hip:       Optional[float] = None
    right_hip:      Optional[float] = None
    left_ankle:     Optional[float] = None
    right_ankle:    Optional[float] = None
    left_elbow:     Optional[float] = None
    right_elbow:    Optional[float] = None
    left_shoulder:  Optional[float] = None
    right_shoulder: Optional[float] = None
    back:           Optional[float] = None


class BiomechanicalFeatures(BaseModel):
    frame_index:          int
    joint_angles:         JointAngles
    symmetry_score:       Optional[float] = Field(None, ge=0.0, le=1.0)
    form_deviation_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    predicted_risk_score: Optional[float] = Field(None, ge=0.0, le=100.0)


class FrameResult(BaseModel):
    frame_index:   int
    pose_detected: bool
    landmarks:     Optional[List[Landmark]]          = None
    features:      Optional[BiomechanicalFeatures]   = None
    error_message: Optional[str]                     = None


class VideoProcessingResponse(BaseModel):
    status:            ProcessingStatus
    total_frames:      int
    processed_frames:  int
    detected_frames:   int
    processing_time_ms: float
    frame_results:     List[FrameResult]
    aggregate_risk_score: Optional[float]    = Field(None, ge=0.0, le=100.0)
    aggregate_risk_level: Optional[RiskLevel] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class FrameProcessingResponse(BaseModel):
    status:            ProcessingStatus
    processing_time_ms: float
    result:            FrameResult
    metadata: Dict[str, str] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status:          str  = "ok"
    version:         str
    ml_model_loaded: bool = False


# ---------------------------------------------------------------------------
# Risk Assessment API schemas
# ---------------------------------------------------------------------------

class RiskAssessmentRequest(BaseModel):
    """
    Input body for POST /assess-risk.

    All fields have sensible defaults so partial payloads are accepted.
    """
    training_load:   float = Field(5.0,  ge=1.0,  le=10.0,  description="Daily training intensity (1-10).")
    recovery_score:  float = Field(50.0, ge=0.0,  le=100.0, description="Recovery quality score (0-100).")
    fatigue_index:   float = Field(5.0,  ge=0.0,  le=10.0,  description="Accumulated fatigue level (0-10).")
    form_decay:      float = Field(0.5,  ge=0.0,  le=1.0,   description="Biomechanical form degradation (0-1).")
    previous_injury: int   = Field(0,    ge=0,    le=1,     description="Prior injury history flag (0 or 1).")


class RiskAssessmentResponse(BaseModel):
    """Full response from POST /assess-risk."""
    status:     ProcessingStatus
    assessment: RiskOutput
    metadata:   Dict[str, str] = Field(default_factory=dict)
