"""
pose_service.py
---------------
Service layer for all MediaPipe BlazePose interactions.

Responsibilities
~~~~~~~~~~~~~~~~
* Initialise and hold a single MediaPipe Pose instance (singleton-per-worker).
* Expose a clean, reusable interface for processing individual frames
  (numpy arrays) so upper layers remain decoupled from MediaPipe internals.
* Provide ``detect_pose()`` — a high-level helper that accepts raw image bytes
  (as received from an HTTP upload), decodes them, runs BlazePose, and returns
  a list of ``PoseLandmarkItem`` objects ready for JSON serialisation.

Notes
~~~~~
* ``static_image_mode = False`` is used for video processing (enables the
  tracker for consecutive frames → faster inference).
* ``static_image_mode = True`` is used when processing a single uploaded
  image so the full detector fires every call.
* The public ``detect_pose()`` function is the primary entry-point for the
  ``POST /detect-pose`` route.  It is intentionally kept stateless so it can
  be called without instantiating a class.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from app.models.schemas import AngleResult, FormFlags, Landmark, PoseLandmarkItem, PoseDetectionResponse
from app.services.feature_engineering import analyze_form
from app.utils.angle_utils import compute_all_angles

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe module handles (module-level so they are imported once)
# ---------------------------------------------------------------------------
_mp_pose        = mp.solutions.pose
_POSE_LANDMARKS = _mp_pose.PoseLandmark   # Enum with all 33 landmark names


# ---------------------------------------------------------------------------
# PoseService — reusable class for video / sequential frame processing
# ---------------------------------------------------------------------------

class PoseService:
    """
    Wraps MediaPipe BlazePose for single-frame and video-stream processing.

    Usage — single frame / image
    ----------------------------
    >>> service = PoseService(static_image_mode=True)
    >>> landmarks = service.process_frame(rgb_frame)

    Usage — video / sequential frames
    ----------------------------------
    >>> service = PoseService(static_image_mode=False)
    >>> for frame in frames:
    ...     landmarks = service.process_frame(frame)
    >>> service.close()

    Recommended — use as a context manager to guarantee resource cleanup:

    >>> with PoseService(static_image_mode=False) as svc:
    ...     for frame in frames:
    ...         landmarks = svc.process_frame(frame)
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        static_image_mode : bool
            ``True``  → full detector on every frame (images).
            ``False`` → use tracker between frames (video, ~3× faster).
        model_complexity : {0, 1, 2}
            BlazePose model size.  0 = lite, 1 = full (default), 2 = heavy.
        smooth_landmarks : bool
            Apply temporal smoothing across frames (video only).
        enable_segmentation : bool
            Compute a segmentation mask.  Disabled by default for speed.
        min_detection_confidence : float
            Minimum confidence for the initial person-detection stage.
        min_tracking_confidence : float
            Minimum confidence to accept landmark tracking vs. re-detecting.
        """
        self._pose = _mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        logger.info(
            "PoseService initialised — static_image_mode=%s, model_complexity=%s",
            static_image_mode,
            model_complexity,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, rgb_frame: np.ndarray) -> Optional[List[Landmark]]:
        """
        Run BlazePose on a single RGB frame.

        Parameters
        ----------
        rgb_frame : np.ndarray
            A ``(H, W, 3)`` uint8 array in **RGB** colour order.

        Returns
        -------
        list[Landmark] | None
            Ordered list of all 33 body landmarks, or ``None`` if no pose
            was detected in the frame.
        """
        if rgb_frame is None or rgb_frame.size == 0:
            logger.warning("process_frame received an empty or None frame; skipping.")
            return None

        # MediaPipe requires a non-writable view to avoid accidental mutation
        rgb_frame.flags.writeable = False
        results = self._pose.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if not results.pose_landmarks:
            return None

        return _parse_landmarks_to_schema(results.pose_landmarks)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._pose.close()
        logger.info("PoseService closed and MediaPipe resources released.")

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "PoseService":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Standalone high-level helper — primary entry-point for /detect-pose
# ---------------------------------------------------------------------------

def detect_pose(image_bytes: bytes) -> PoseDetectionResponse:
    """
    Decode raw image bytes, run BlazePose, and return a ``PoseDetectionResponse``
    ready for direct JSON serialisation.

    This function is intentionally stateless — it creates a short-lived
    ``PoseService`` for a single image, which is the correct mode for the
    ``POST /detect-pose`` endpoint.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of a JPEG, PNG, or WebP image as received from an HTTP
        ``UploadFile`` read.

    Returns
    -------
    PoseDetectionResponse
        Contains ``pose_detected``, ``landmark_count``, ``processing_time_ms``,
        and ``landmarks`` (list of 33 items when a pose is found, empty when not).

    Raises
    ------
    ValueError
        If the bytes cannot be decoded as a valid image.
    """
    t_start = time.perf_counter()

    # --- Decode image bytes → numpy BGR array ---------------------------------
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if bgr_frame is None:
        raise ValueError(
            "Image decoding failed. Ensure the uploaded file is a valid "
            "JPEG, PNG, or WebP image."
        )

    # Convert BGR (OpenCV default) → RGB (MediaPipe expectation)
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    # --- Run BlazePose --------------------------------------------------------
    # static_image_mode=True → full detector fires (no tracker state to corrupt)
    with _mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=False,      # meaningless for single frames
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose:
        rgb_frame.flags.writeable = False
        results = pose.process(rgb_frame)
        rgb_frame.flags.writeable = True

    elapsed_ms = (time.perf_counter() - t_start) * 1_000

    # --- Build response -------------------------------------------------------
    if not results.pose_landmarks:
        logger.info("detect_pose: no pose detected (%.1f ms).", elapsed_ms)
        return PoseDetectionResponse(
            pose_detected=False,
            landmark_count=0,
            processing_time_ms=round(elapsed_ms, 2),
            landmarks=[],
        )

    landmark_items = _parse_landmarks_to_items(results.pose_landmarks)

    # --- Compute joint angles from the detected landmarks ----------------------
    raw_angles = compute_all_angles(landmark_items)
    angles = AngleResult(
        knee_angle=raw_angles["knee_angle"],
        hip_angle=raw_angles["hip_angle"],
        back_angle=raw_angles["back_angle"],
    )

    logger.info(
        "detect_pose: %d landmarks detected, angles=knee:%.1f hip:%.1f back:%.1f (%.1f ms).",
        len(landmark_items),
        raw_angles["knee_angle"] or 0.0,
        raw_angles["hip_angle"]  or 0.0,
        raw_angles["back_angle"] or 0.0,
        elapsed_ms,
    )

    return PoseDetectionResponse(
        pose_detected=True,
        landmark_count=len(landmark_items),
        processing_time_ms=round(elapsed_ms, 2),
        landmarks=landmark_items,
        angles=angles,
        form_flags=analyze_form(
            landmarks=landmark_items,
            back_angle=raw_angles["back_angle"],
        ),
    )


# ---------------------------------------------------------------------------
# Private parsing helpers
# ---------------------------------------------------------------------------

def _parse_landmarks_to_schema(pose_landmarks) -> List[Landmark]:
    """
    Convert a MediaPipe ``NormalizedLandmarkList`` to the richer internal
    ``Landmark`` schema (used by upload_route / feature engineering).
    """
    landmarks: List[Landmark] = []
    for idx, lm in enumerate(pose_landmarks.landmark):
        try:
            name = _POSE_LANDMARKS(idx).name
        except ValueError:
            name = f"LANDMARK_{idx}"

        landmarks.append(
            Landmark(
                name=name,
                x=float(np.clip(lm.x, 0.0, 1.0)),
                y=float(np.clip(lm.y, 0.0, 1.0)),
                z=float(lm.z),
                visibility=float(np.clip(lm.visibility, 0.0, 1.0)),
            )
        )
    return landmarks


def _parse_landmarks_to_items(pose_landmarks) -> List[PoseLandmarkItem]:
    """
    Convert a MediaPipe ``NormalizedLandmarkList`` to the compact
    ``PoseLandmarkItem`` schema used by ``POST /detect-pose``.

    Output matches the required contract::

        {"id": 0, "name": "NOSE", "x": 0.51, "y": 0.12, "z": -0.01,
         "visibility": 0.98}
    """
    items: List[PoseLandmarkItem] = []
    for idx, lm in enumerate(pose_landmarks.landmark):
        try:
            name = _POSE_LANDMARKS(idx).name
        except ValueError:
            name = f"LANDMARK_{idx}"

        items.append(
            PoseLandmarkItem(
                id=idx,
                name=name,
                x=round(float(lm.x), 6),
                y=round(float(lm.y), 6),
                z=round(float(lm.z), 6),
                visibility=round(float(np.clip(lm.visibility, 0.0, 1.0)), 6),
            )
        )
    return items
