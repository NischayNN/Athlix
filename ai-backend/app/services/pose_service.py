"""
pose_service.py
---------------
Service layer for all MediaPipe BlazePose interactions.

Responsibilities
~~~~~~~~~~~~~~~~
* Initialise and hold a single MediaPipe Pose instance (singleton-per-worker).
* Expose a clean interface for processing individual frames (numpy arrays).
* Convert raw MediaPipe landmark data into the project's internal *Landmark*
  schema so that upper layers remain decoupled from MediaPipe internals.

Notes
~~~~~
* `static_image_mode = False` is used for video processing (enables the
  tracker for consecutive frames → faster inference).
* `static_image_mode = True` will be set when processing a single uploaded
  image to force the full detector to run on each call.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import List, Optional

import mediapipe as mp
import numpy as np

from app.models.schemas import Landmark

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe constants
# ---------------------------------------------------------------------------
_mp_pose     = mp.solutions.pose
_POSE_LANDMARKS = _mp_pose.PoseLandmark   # Enum with all 33 landmark names


class PoseService:
    """
    Wraps MediaPipe BlazePose for single-frame and video-stream processing.

    Usage (single frame / image)
    ----------------------------
    >>> service = PoseService(static_image_mode=True)
    >>> landmarks = service.process_frame(rgb_frame)

    Usage (video / sequential frames)
    ----------------------------------
    >>> service = PoseService(static_image_mode=False)
    >>> for frame in frames:
    ...     landmarks = service.process_frame(frame)
    >>> service.close()

    Or use as a context manager:
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
            ``True`` → full detector on every frame (images).
            ``False`` → use tracker between frames (video, faster).
        model_complexity : {0, 1, 2}
            BlazePose model size. 0 = lite, 1 = full, 2 = heavy.
        smooth_landmarks : bool
            Apply landmark smoothing across frames (only relevant for video).
        enable_segmentation : bool
            Whether to compute a segmentation mask.  Disabled by default for
            speed.
        min_detection_confidence : float
            Minimum confidence for initial detection.
        min_tracking_confidence : float
            Minimum confidence to accept tracking result over re-detection.
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
        Run BlazePose on a single RGB frame and return a list of *Landmark*
        objects or ``None`` if no pose was detected.

        Parameters
        ----------
        rgb_frame : np.ndarray
            A (H, W, 3) uint8 array in **RGB** colour order.

        Returns
        -------
        list of Landmark, or None
        """
        if rgb_frame is None or rgb_frame.size == 0:
            logger.warning("process_frame received an empty or None frame; skipping.")
            return None

        # MediaPipe requires the array to be writable
        rgb_frame.flags.writeable = False
        results = self._pose.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if not results.pose_landmarks:
            return None

        return self._parse_landmarks(results.pose_landmarks)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._pose.close()
        logger.info("PoseService closed and resources released.")

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "PoseService":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_landmarks(pose_landmarks) -> List[Landmark]:
        """
        Convert a MediaPipe ``NormalizedLandmarkList`` into a list of project
        *Landmark* schemas.
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
