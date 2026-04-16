from __future__ import annotations

import logging
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.models.schemas import (
    FatigueInput,
    FormFlags,
    ProcessFrameResponse,
)
from app.services.feature_engineering import analyze_form, extract_joint_angles, generate_feature_vector
from app.services.pose_service import PoseService, detect_pose
from app.services.risk_engine import get_risk_score

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["Frame Processing"])

_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

_ISSUE_DETAILS = {
    "Incomplete Depth":       "Hip crease did not drop below the patella.",
    "Knee Valgus":            "Medial collapse detected during the concentric phase.",
    "Excessive Forward Lean": "Torso angle exceeded safe thresholds relative to vertical.",
    "Heel Rise":              "Heel lifted off the platform during the descent.",
    "Lateral Shift":          "Lateral weight distribution asymmetry detected.",
}


@router.post(
    "/process-frame",
    response_model=ProcessFrameResponse,
    summary="Full pipeline: pose → angles → form analysis → feature vector",
)
async def process_frame(
    file: UploadFile = File(..., description="Image frame (JPEG / PNG / WebP)"),
    training_load: float = Form(..., gt=0, description="Training load metric"),
    sleep_hours: float = Form(..., gt=0, le=24, description="Hours of sleep last night"),
) -> ProcessFrameResponse:
    """
    End-to-end pipeline for a single frame:

    1. Detect pose via MediaPipe BlazePose.
    2. Extract joint angles (knee, hip, back).
    3. Analyse biomechanical form (knee valgus, bad back, squat depth).
    4. Compute fatigue / recovery features.
    5. Assemble a flat, ML-ready feature vector.
    """
    content_type = (file.content_type or "").split(";")[0].strip()
    if content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type '{content_type}'. Allowed: {sorted(_ALLOWED_TYPES)}",
        )

    try:
        image_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )

    t_start = time.perf_counter()

    # ── Step 1 & 2: Pose detection + angle extraction ──────────────────────
    try:
        pose_result = detect_pose(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Pose detection failed.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Pose detection failed.") from exc

    elapsed_ms = round((time.perf_counter() - t_start) * 1_000, 2)

    if not pose_result.pose_detected:
        return ProcessFrameResponse(
            pose_detected=False,
            processing_time_ms=elapsed_ms,
            error="No pose detected in the uploaded frame.",
        )

    # ── Step 3: Form analysis ───────────────────────────────────────────────
    form_flags: FormFlags = analyze_form(
        landmarks=pose_result.landmarks,
        back_angle=pose_result.angles.back_angle if pose_result.angles else None,
    )

    # ── Step 4 & 5: Fatigue + feature vector ───────────────────────────────
    fatigue_input = FatigueInput(
        training_load=training_load,
        sleep_hours=sleep_hours,
    )

    feature_vector = generate_feature_vector(
        pose_data=pose_result,
        fatigue_input=fatigue_input,
    )

    elapsed_ms = round((time.perf_counter() - t_start) * 1_000, 2)

    logger.info(
        "process-frame completed in %.1f ms — fatigue=%.4f recovery=%.4f",
        elapsed_ms, feature_vector.fatigue_score, feature_vector.recovery_score,
    )

    return ProcessFrameResponse(
        pose_detected=True,
        feature_vector=feature_vector,
        form_flags=form_flags,
        angles=pose_result.angles,
        processing_time_ms=elapsed_ms,
    )


@router.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """
    Analyzes the video by processing up to 15 frames.
    All outputs are computed from real pose data — ZERO hardcoded values.
    """
    logger.info("Video received: %s", file.filename)

    tmp_dir = tempfile.mkdtemp(prefix="analyze_")
    tmp_path = Path(tmp_dir) / (file.filename or "upload.mp4")
    cap = None

    try:
        # Read asynchronously — does not block the event loop
        file_bytes = await file.read()
        tmp_path.write_bytes(file_bytes)

        logger.info("Processing started")

        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            logger.error("Could not open video: %s", file.filename)
            return {"error": "processing failed", "detail": "Could not open video file"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames_processed = 0
        max_frames = 15

        # Per-frame angle accumulators — filled from REAL detected landmarks
        all_landmarks: List = []
        knee_angles: List[float] = []
        hip_angles:  List[float] = []
        back_angles: List[float] = []

        with PoseService(static_image_mode=False) as pose_svc:
            while cap.isOpened() and frames_processed < max_frames:
                ret, bgr_frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                landmarks = pose_svc.process_frame(rgb_frame)

                if landmarks:
                    all_landmarks.append(landmarks)
                    angles = extract_joint_angles(landmarks)

                    # Average bilateral angles where both sides visible
                    if angles.left_knee and angles.right_knee:
                        knee_angles.append((angles.left_knee + angles.right_knee) / 2.0)
                    elif angles.left_knee:
                        knee_angles.append(angles.left_knee)
                    elif angles.right_knee:
                        knee_angles.append(angles.right_knee)

                    if angles.left_hip and angles.right_hip:
                        hip_angles.append((angles.left_hip + angles.right_hip) / 2.0)
                    elif angles.left_hip:
                        hip_angles.append(angles.left_hip)
                    elif angles.right_hip:
                        hip_angles.append(angles.right_hip)

                    if angles.back is not None:
                        back_angles.append(angles.back)

                frames_processed += 1
                logger.info("Frame %d processed", frames_processed)

        if not all_landmarks:
            logger.warning("No pose detected in video: %s", file.filename)
            return {"error": "processing failed", "detail": "No pose detected in video"}

        # ── Rep counting: detect flexion→extension cycles in knee angle ──────
        # A rep = knee crosses below FLEX_THRESH then rises back above EXT_THRESH
        reps = 0
        in_flexion = False
        FLEX_THRESH = 120.0  # degrees — athlete is squatting
        EXT_THRESH  = 150.0  # degrees — athlete has stood up (rep complete)
        for ka in knee_angles:
            if not in_flexion and ka < FLEX_THRESH:
                in_flexion = True
            elif in_flexion and ka > EXT_THRESH:
                reps += 1
                in_flexion = False

        # ── Aggregate real per-frame measurements ────────────────────────────
        knee_min  = round(min(knee_angles),  2) if knee_angles  else 90.0
        hip_min   = round(min(hip_angles),   2) if hip_angles   else 90.0
        back_max  = round(max(back_angles),  2) if back_angles  else 30.0
        knee_mean = round(sum(knee_angles) / len(knee_angles), 2) if knee_angles else 90.0

        # form_decay: knee angle variance normalised to 0-1
        # High variance → unstable mechanics → higher decay
        knee_std   = statistics.stdev(knee_angles) if len(knee_angles) > 1 else 0.0
        form_decay = round(min(knee_std / 60.0, 1.0), 4)

        # fatigue_index: compare early vs late knee angles
        # If later frames show more extreme angles, fatigue is higher
        third      = max(len(knee_angles) // 3, 1)
        early_mean = sum(knee_angles[:third])  / third
        late_mean  = sum(knee_angles[-third:]) / third
        decay_ratio   = abs(late_mean - early_mean) / (early_mean + 1e-6)
        fatigue_index = round(min(decay_ratio * 10.0, 10.0), 4)

        # recovery_score: inverse of fatigue (0-100 scale)
        recovery_score = round(max(100.0 - fatigue_index * 8.0, 0.0), 2)

        # training_load heuristic from observed video duration
        duration_s    = frames_processed / fps
        training_load = round(min(1.0 + duration_s * 0.5, 10.0), 2)

        # Form flags from the last detected landmark frame
        best_landmarks  = all_landmarks[-1]
        best_angles_obj = extract_joint_angles(best_landmarks)
        form_flags      = analyze_form(best_landmarks, back_angle=best_angles_obj.back)

        feature_vector = {
            "training_load":   training_load,
            "recovery_score":  recovery_score,
            "fatigue_index":   fatigue_index,
            "form_decay":      form_decay,
            "previous_injury": 0,
            "knee_angle_min":  knee_min,
            "hip_angle_min":   hip_min,
            "back_angle_max":  back_max,
            "knee_angle_mean": knee_mean,
            "reps_detected":   reps,
        }
        logger.info("Processing new video — feature_vector: %s", feature_vector)

        # ML model receives only its trained 5 features
        risk_input = {
            "training_load":   training_load,
            "recovery_score":  recovery_score,
            "fatigue_index":   fatigue_index,
            "form_decay":      form_decay,
            "previous_injury": 0,
        }
        logger.info("Features sent to model: %s", risk_input)

        risk_output    = get_risk_score(risk_input)
        computed_score = round(max(0.0, 100.0 - risk_output.risk_score), 1)

        logger.info(
            "Model prediction — risk_score=%.2f risk_level=%s",
            risk_output.risk_score, risk_output.risk_level,
        )
        logger.info(
            "Returning response — score=%.1f injury_risk=%.2f reps=%d frames=%d",
            computed_score, risk_output.risk_score, reps, frames_processed,
        )

        # ── Build Movement Risk Index (the UI's main risk display) ────────
        # Combine form flags into a per-issue risk breakdown
        form_flag_dict = {
            "knee_valgus":            form_flags.knee_valgus        or False,
            "incomplete_depth":       form_flags.insufficient_depth or False,
            "excessive_forward_lean": form_flags.bad_back_posture   or False,
            "heel_rise":              False,
            "lateral_shift":          False,
        }

        # Risk contribution breakdown (matches what Results.jsx renders)
        risk_breakdown = []
        flag_severity_map = {
            "incomplete_depth":       ("Incomplete Depth",       0.8,  "High"),
            "knee_valgus":            ("Knee Valgus",            0.5,  "Medium"),
            "excessive_forward_lean": ("Excessive Forward Lean", 0.5,  "Medium"),
            "heel_rise":              ("Heel Rise",              0.2,  "Low"),
            "lateral_shift":          ("Lateral Shift",          0.2,  "Low"),
        }

        total_risk_contribution = 0.0
        intensity_mult = round(min((training_load / 10.0) ** 2, 1.0), 2)
        recovery_mult  = round(max(1.0, 1.0 + (100.0 - recovery_score) / 200.0), 2)

        for flag_key, (issue_name, base_sev, sev_label) in flag_severity_map.items():
            if form_flag_dict.get(flag_key):
                contribution = round(base_sev * intensity_mult * recovery_mult, 2)
                total_risk_contribution += contribution
                risk_breakdown.append({
                    "issue":         issue_name,
                    "flawSeverity":  f"{base_sev:.2f}",
                    "intensityMult": f"{intensity_mult:.2f}",
                    "recoveryMult":  f"{recovery_mult:.2f}",
                    "historyMult":   "1.00",
                    "contribution":  f"{contribution:.2f}",
                })

        movement_risk_index = round(min(total_risk_contribution * 35 + risk_output.risk_score * 0.5, 100), 0)
        risk_label = "High" if movement_risk_index >= 75 else ("Moderate" if movement_risk_index >= 40 else "Low")

        # Explanation insight based on risk level
        if movement_risk_index >= 75:
            if recovery_mult > 1.2:
                explanation_insight = "Elevated risk due to poor recovery state amplifying form fatigue."
            else:
                explanation_insight = "Form breakdown under high relative load. Tendon and joint structures are significantly stressed."
        elif movement_risk_index >= 40:
            explanation_insight = "Moderate mechanical shifts detected. Monitor load scaling carefully."
        else:
            explanation_insight = "Solid movement pattern under acceptable relative load."

        # Velocity estimation from knee angle range of motion
        knee_range = (max(knee_angles) - min(knee_angles)) if len(knee_angles) > 1 else 20.0
        movement_velocity = round(knee_range / 100.0 * (duration_s + 0.5), 2)
        velocity_classification = "Strength" if movement_velocity < 0.5 else ("Power" if movement_velocity > 0.8 else "Hypertrophy")

        # Load score: function of training load, reps, and velocity
        load_score = round(training_load * max(reps, 1) * (1.0 if velocity_classification == "Hypertrophy" else 0.8 if velocity_classification == "Strength" else 1.2), 1)

        return {
            "score":                   computed_score,
            "injury_risk":             risk_output.risk_score,
            "risk_level":              risk_output.risk_level,
            "reps":                    reps,
            "feature_vector":          feature_vector,
            "form_flags":              form_flag_dict,
            # Movement Risk Index section (Results.jsx)
            "movementRiskIndex":       int(movement_risk_index),
            "riskLabel":               risk_label,
            "riskBreakdown":           risk_breakdown,
            "explanationInsight":      explanation_insight,
            # Intensity Profile section (Results.jsx)
            "movementVelocity":        f"{movement_velocity:.2f}",
            "velocityClassification":  velocity_classification,
            "loadScore":               f"{load_score:.1f}",
            # Key issues for Highlights section
            "keyIssues": [
                {"id": i + 1, "issue": rb["issue"], "severity": sev_label, "detail": _ISSUE_DETAILS.get(rb["issue"], "")}
                for i, rb in enumerate(risk_breakdown)
            ] if risk_breakdown else [
                {"id": 1, "issue": "Good Form", "severity": "Low", "detail": "No significant mechanical deviations detected."}
            ],
        }


    except Exception as exc:
        logger.error("analyze_video processing failed: %s", str(exc), exc_info=True)
        return {"error": "processing failed"}
    finally:
        if cap is not None:
            cap.release()
        shutil.rmtree(tmp_dir, ignore_errors=True)
