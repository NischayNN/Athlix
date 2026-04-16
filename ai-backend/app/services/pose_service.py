from __future__ import annotations

import logging
import time
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

from app.models.schemas import AngleResult, FormFlags, Landmark, PoseLandmarkItem, PoseDetectionResponse
from app.services.feature_engineering import analyze_form
from app.utils.angle_utils import compute_all_angles

logger = logging.getLogger(__name__)

try:
    _mp_pose        = mp.solutions.pose
    _POSE_LANDMARKS = _mp_pose.PoseLandmark
    _MP_AVAILABLE   = True
except AttributeError:
    _mp_pose        = None
    _POSE_LANDMARKS = None
    _MP_AVAILABLE   = False


class PoseService:
    """Wraps MediaPipe BlazePose for single-frame and video-stream processing."""

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._pose = None
        if _MP_AVAILABLE:
            self._pose = _mp_pose.Pose(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                smooth_landmarks=smooth_landmarks,
                enable_segmentation=enable_segmentation,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

    def process_frame(self, rgb_frame: np.ndarray) -> Optional[List[Landmark]]:
        if rgb_frame is None or rgb_frame.size == 0:
            return None

        if not _MP_AVAILABLE:
            time.sleep(0.1)
            return _mock_landmarks()

        rgb_frame.flags.writeable = False
        results = self._pose.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if not results.pose_landmarks:
            return None

        return _parse_landmarks_to_schema(results.pose_landmarks)

    def close(self) -> None:
        if _MP_AVAILABLE:
            self._pose.close()

    def __enter__(self) -> "PoseService":
        return self

    def __exit__(self, *_) -> None:
        self.close()

def _mock_landmarks() -> List[Landmark]:
    """Fallback mocks if mediapipe is missing."""
    pts = []
    names = ["LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"]
    for i, nm in enumerate(names):
        pts.append(Landmark(name=nm, x=0.5, y=0.5, z=0.0, visibility=1.0))
    return pts


def detect_pose(image_bytes: bytes) -> PoseDetectionResponse:
    """Decode image bytes, run BlazePose, compute angles and form flags, and return a structured response."""
    t_start = time.perf_counter()

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if bgr_frame is None:
        raise ValueError("Image decoding failed. Ensure the file is a valid JPEG, PNG, or WebP.")

    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    if not _MP_AVAILABLE:
        results = type('MockResults', (), {'pose_landmarks': True})()
        _parse_landmarks_to_items = _mock_landmark_items
    else:
        with _mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            smooth_landmarks=False,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        ) as pose:
            rgb_frame.flags.writeable = False
            results = pose.process(rgb_frame)
            rgb_frame.flags.writeable = True

    elapsed_ms = (time.perf_counter() - t_start) * 1_000

    if not results.pose_landmarks:
        return PoseDetectionResponse(
            pose_detected=False,
            landmark_count=0,
            processing_time_ms=round(elapsed_ms, 2),
            landmarks=[],
        )

    landmark_items = _parse_landmarks_to_items(results.pose_landmarks)
    raw_angles = compute_all_angles(landmark_items)

    return PoseDetectionResponse(
        pose_detected=True,
        landmark_count=len(landmark_items),
        processing_time_ms=round(elapsed_ms, 2),
        landmarks=landmark_items,
        angles=AngleResult(
            knee_angle=raw_angles["knee_angle"],
            hip_angle=raw_angles["hip_angle"],
            back_angle=raw_angles["back_angle"],
        ),
        form_flags=analyze_form(
            landmarks=landmark_items,
            back_angle=raw_angles["back_angle"],
        ),
    )


def _parse_landmarks_to_schema(pose_landmarks) -> List[Landmark]:
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

def _mock_landmark_items(pose_landmarks) -> List[PoseLandmarkItem]:
    items = []
    import random
    from app.services.feature_engineering import _LM_INDEX
    for name, idx in _LM_INDEX.items():
        items.append(
            PoseLandmarkItem(
                id=idx,
                name=name,
                x=round(random.uniform(0.4, 0.6), 6),
                y=round(random.uniform(0.4, 0.6), 6),
                z=round(random.uniform(-0.1, 0.1), 6),
                visibility=1.0,
            )
        )
    return items
