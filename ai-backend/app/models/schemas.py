from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED  = "failed"


class RiskLevel(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


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
    landmarks:     Optional[List[Landmark]]        = None
    features:      Optional[BiomechanicalFeatures] = None
    error_message: Optional[str]                   = None


class VideoProcessingResponse(BaseModel):
    status:               ProcessingStatus
    total_frames:         int
    processed_frames:     int
    detected_frames:      int
    processing_time_ms:   float
    frame_results:        List[FrameResult]
    aggregate_risk_score: Optional[float]     = Field(None, ge=0.0, le=100.0)
    aggregate_risk_level: Optional[RiskLevel] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class FrameProcessingResponse(BaseModel):
    status:             ProcessingStatus
    processing_time_ms: float
    result:             FrameResult
    metadata: Dict[str, str] = Field(default_factory=dict)


class SetSnapshot(BaseModel):
    """Angle snapshot from one completed set, used to track form over a session."""
    set_index:  int
    timestamp:  Optional[float] = None
    knee_angle: Optional[float] = None
    hip_angle:  Optional[float] = None
    back_angle: Optional[float] = None


class FatigueInput(BaseModel):
    training_load:      float = Field(..., gt=0)
    sleep_hours:        float = Field(..., gt=0, le=24)
    previous_sets_data: Optional[List[SetSnapshot]] = None


class FatigueResult(BaseModel):
    fatigue_score:   float
    recovery_score:  float
    form_decay_rate: Optional[float] = None


class FeatureVector(BaseModel):
    """Flat, ML-ready feature vector combining pose angles and fatigue metrics."""
    knee_angle:      Optional[float] = None
    hip_angle:       Optional[float] = None
    back_angle:      Optional[float] = None
    fatigue_score:   float
    recovery_score:  float
    load:            float
    form_decay_rate: Optional[float] = None


class ProcessFrameResponse(BaseModel):
    pose_detected:   bool
    feature_vector:  Optional[FeatureVector] = None
    form_flags:      Optional[FormFlags]     = None
    angles:          Optional[AngleResult]   = None
    processing_time_ms: float
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status:          str  = "ok"
    version:         str
    ml_model_loaded: bool = False
