from __future__ import annotations

import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.models.schemas import (
    FatigueInput,
    FormFlags,
    ProcessFrameResponse,
)
from app.services.feature_engineering import analyze_form, generate_feature_vector
from app.services.pose_service import detect_pose

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["Frame Processing"])

_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


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
