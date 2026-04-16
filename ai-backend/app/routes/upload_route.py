"""
upload_route.py
---------------
FastAPI router that exposes endpoints for:

  * POST /upload/frame  — Process a single image (JPEG / PNG).
  * POST /upload/video  — Process an MP4 / MOV video file frame-by-frame.

Both endpoints return structured JSON (see ``app/models/schemas.py``).

Notes
~~~~~
* Large videos are streamed to a temporary file and cleaned up after
  processing to keep memory usage bounded.
* Frame-by-frame processing runs synchronously for now.  For production,
  offload heavy video jobs to a background task queue (Celery / RQ).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.models.schemas import (
    BiomechanicalFeatures,
    FrameProcessingResponse,
    FrameResult,
    ProcessingStatus,
    VideoProcessingResponse,
)
from app.services.feature_engineering import build_feature_vector
from app.services.pose_service import PoseService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["Upload & Processing"])

# ---------------------------------------------------------------------------
# Permitted MIME types
# ---------------------------------------------------------------------------
_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
_ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/avi"}

# Maximum frames sampled from a single video (avoids runaway processing time)
_MAX_VIDEO_FRAMES = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_content_type(file: UploadFile, allowed: set[str]) -> None:
    """Raise 415 if the uploaded file's content type is not in *allowed*."""
    content_type = file.content_type or ""
    # Strip encoding qualifiers e.g. "image/jpeg; charset=utf-8"
    base_type = content_type.split(";")[0].strip()
    if base_type not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported media type '{base_type}'. "
                f"Allowed: {sorted(allowed)}"
            ),
        )


def _bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR frame to RGB for MediaPipe."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes to an OpenCV BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not decode the uploaded image.  Ensure it is a valid JPEG or PNG.",
        )
    return img


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/frame",
    response_model=FrameProcessingResponse,
    summary="Process a single image frame",
    description=(
        "Upload a JPEG or PNG image.  The service runs BlazePose pose detection, "
        "computes joint angles, and returns a structured feature vector."
    ),
)
async def upload_frame(
    file: UploadFile = File(..., description="Image file (JPEG / PNG / WebP)."),
) -> FrameProcessingResponse:
    _validate_content_type(file, _ALLOWED_IMAGE_TYPES)

    t_start = time.perf_counter()

    try:
        raw_bytes = await file.read()
    except Exception as exc:
        logger.exception("Failed to read uploaded frame.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    # Decode image
    bgr_frame = _decode_image_bytes(raw_bytes)
    rgb_frame = _bgr_to_rgb(bgr_frame)

    # Pose detection — use static_image_mode=True for a single frame
    with PoseService(static_image_mode=True) as pose_svc:
        landmarks = pose_svc.process_frame(rgb_frame)

    pose_detected = landmarks is not None
    features: BiomechanicalFeatures | None = None

    if pose_detected:
        features = build_feature_vector(frame_index=0, landmarks=landmarks)

    elapsed_ms = (time.perf_counter() - t_start) * 1_000

    result = FrameResult(
        frame_index=0,
        pose_detected=pose_detected,
        landmarks=landmarks,
        features=features,
        error_message=None if pose_detected else "No pose detected in the uploaded frame.",
    )

    return FrameProcessingResponse(
        status=ProcessingStatus.SUCCESS if pose_detected else ProcessingStatus.PARTIAL,
        processing_time_ms=round(elapsed_ms, 2),
        result=result,
        metadata={"filename": file.filename or "unknown", "source": "frame"},
    )


@router.post(
    "/video",
    response_model=VideoProcessingResponse,
    summary="Process a video file",
    description=(
        "Upload an MP4 / MOV video file.  The service extracts frames at the "
        f"native frame rate (up to {_MAX_VIDEO_FRAMES} frames), runs BlazePose "
        "on each frame, and returns per-frame feature vectors."
    ),
)
async def upload_video(
    file: UploadFile = File(..., description="Video file (MP4 / MOV / AVI)."),
) -> VideoProcessingResponse:
    _validate_content_type(file, _ALLOWED_VIDEO_TYPES)

    t_start = time.perf_counter()

    # --- Save to a temporary file so OpenCV can seek ----------------------
    tmp_dir  = tempfile.mkdtemp(prefix="athlix_")
    tmp_path = Path(tmp_dir) / (file.filename or "upload.mp4")

    try:
        with tmp_path.open("wb") as buf:
            shutil.copyfileobj(file.file, buf)

        # --- Extract and process frames ------------------------------------
        frame_results: List[FrameResult] = []
        cap = cv2.VideoCapture(str(tmp_path))

        if not cap.isOpened():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="OpenCV could not open the uploaded video file.",
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("Video opened: %d total frames — %s", total_frames, file.filename)

        with PoseService(static_image_mode=False) as pose_svc:
            frame_idx = 0
            detected  = 0

            while cap.isOpened() and frame_idx < _MAX_VIDEO_FRAMES:
                ret, bgr_frame = cap.read()
                if not ret:
                    break

                rgb_frame = _bgr_to_rgb(bgr_frame)
                landmarks = pose_svc.process_frame(rgb_frame)
                pose_detected = landmarks is not None

                features: BiomechanicalFeatures | None = None
                error_msg: str | None = None

                if pose_detected:
                    detected += 1
                    try:
                        features = build_feature_vector(frame_index=frame_idx, landmarks=landmarks)
                    except Exception as exc:
                        logger.warning("Feature extraction failed at frame %d: %s", frame_idx, exc)
                        error_msg = f"Feature extraction error: {exc}"
                else:
                    error_msg = "No pose detected."

                frame_results.append(
                    FrameResult(
                        frame_index=frame_idx,
                        pose_detected=pose_detected,
                        landmarks=landmarks,
                        features=features,
                        error_message=error_msg,
                    )
                )

                frame_idx += 1

        cap.release()

    finally:
        # Always clean up temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed_ms = (time.perf_counter() - t_start) * 1_000
    processed_count = len(frame_results)

    status_val = (
        ProcessingStatus.SUCCESS
        if detected > 0
        else ProcessingStatus.PARTIAL
        if processed_count > 0
        else ProcessingStatus.FAILED
    )

    return VideoProcessingResponse(
        status=status_val,
        total_frames=total_frames,
        processed_frames=processed_count,
        detected_frames=detected,
        processing_time_ms=round(elapsed_ms, 2),
        frame_results=frame_results,
        aggregate_risk_score=None,   # Placeholder — computed by ML model
        aggregate_risk_level=None,
        metadata={
            "filename": file.filename or "unknown",
            "max_frames_cap": str(_MAX_VIDEO_FRAMES),
        },
    )
