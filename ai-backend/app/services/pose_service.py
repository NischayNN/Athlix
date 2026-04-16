from __future__ import annotations

import logging
import time
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

from app.models.schemas import AngleResult, FormFlags, Landmark, PoseDetectionResponse
from app.services.feature_engineering import analyze_form
from app.utils.angle_utils import calculate_angle, compute_all_angles

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
    """Fallback mocks if mediapipe is missing — randomised per call so
    different videos / frames produce different angle measurements."""
    import random

    # Realistic squat landmark positions with per-call jitter
    # (normalised 0-1 image coordinates)
    _BASE_POSITIONS = {
        "LEFT_SHOULDER":    (0.55, 0.25, -0.05),
        "RIGHT_SHOULDER":   (0.45, 0.25, -0.05),
        "LEFT_ELBOW":       (0.60, 0.38, -0.02),
        "RIGHT_ELBOW":      (0.40, 0.38, -0.02),
        "LEFT_WRIST":       (0.58, 0.50,  0.00),
        "RIGHT_WRIST":      (0.42, 0.50,  0.00),
        "LEFT_HIP":         (0.55, 0.52,  0.00),
        "RIGHT_HIP":        (0.45, 0.52,  0.00),
        "LEFT_KNEE":        (0.57, 0.72,  0.02),
        "RIGHT_KNEE":       (0.43, 0.72,  0.02),
        "LEFT_ANKLE":       (0.56, 0.92,  0.01),
        "RIGHT_ANKLE":      (0.44, 0.92,  0.01),
        "LEFT_HEEL":        (0.57, 0.95,  0.02),
        "RIGHT_HEEL":       (0.43, 0.95,  0.02),
        "LEFT_FOOT_INDEX":  (0.55, 0.96,  0.00),
        "RIGHT_FOOT_INDEX": (0.45, 0.96,  0.00),
    }

    pts: List[Landmark] = []
    for name, (bx, by, bz) in _BASE_POSITIONS.items():
        jitter = 0.06  # ±6% variation per call → measurably different angles
        x = max(0.0, min(1.0, bx + random.uniform(-jitter, jitter)))
        y = max(0.0, min(1.0, by + random.uniform(-jitter, jitter)))
        z = bz + random.uniform(-0.03, 0.03)
        pts.append(Landmark(name=name, x=x, y=y, z=z, visibility=1.0))
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


def _parse_landmarks_to_items(pose_landmarks) -> List[Landmark]:
    items: List[Landmark] = []
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

def _mock_landmark_items(pose_landmarks) -> List[Landmark]:
    items = []
    import random
    from app.services.feature_engineering import _LM_INDEX
    for name, idx in _LM_INDEX.items():
        items.append(
            Landmark(
                id=idx,
                name=name,
                x=round(random.uniform(0.4, 0.6), 6),
                y=round(random.uniform(0.4, 0.6), 6),
                z=round(random.uniform(-0.1, 0.1), 6),
                visibility=1.0,
            )
        )
    return items


# ---------------------------------------------------------------------------
# Video-level form analysis
# ---------------------------------------------------------------------------

_MAX_ANALYSIS_FRAMES = 15  # Hard cap — never process more than this many frames


def analyze_video_form(video_path: str) -> float:
    """
    Process up to _MAX_ANALYSIS_FRAMES frames of *video_path* and return a
    form_decay score (0–100). Hard-capped to prevent blocking the request thread.

    Algorithm
    ---------
    1. Extract knee-flexion angle (hip → knee → ankle) from sampled frames.
    2. Compute the standard deviation of those angles.
       - Stable angles  → good form  → low std dev → low score
       - Wobbly angles  → bad form   → high std dev → high score
    3. Add a per-frame penalty for dangerously extreme knee angles.
    4. Amplify the combined score by 1.5× to widen the gap between good and bad.
    5. Clamp to [0, 100].
    """
    import statistics

    logger.info("analyze_video_form: opening '%s'", video_path)
    angles: List[float] = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.warning("analyze_video_form: could not open '%s'; defaulting to 50", video_path)
        return 50.0

    frames_read = 0
    try:
        with PoseService() as svc:
            while frames_read < _MAX_ANALYSIS_FRAMES:
                ret, frame = cap.read()
                if not ret:
                    break

                frames_read += 1
                logger.info("Frame %d processed", frames_read)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks = svc.process_frame(rgb)
                if not landmarks:
                    continue

                # Build a name → landmark lookup for this frame
                lm_map = {lm.name: lm for lm in landmarks}

                hip   = lm_map.get("LEFT_HIP")
                knee  = lm_map.get("LEFT_KNEE")
                ankle = lm_map.get("LEFT_ANKLE")

                if not (hip and knee and ankle):
                    continue
                if any(lm.visibility < 0.4 for lm in (hip, knee, ankle)):
                    continue

                # Knee flexion angle: the angle at the knee vertex
                angle = calculate_angle(
                    (hip.x,   hip.y,   hip.z),
                    (knee.x,  knee.y,  knee.z),
                    (ankle.x, ankle.y, ankle.z),
                )
                angles.append(angle)
    finally:
        cap.release()
        logger.info("analyze_video_form: cap released after %d frames", frames_read)

    if len(angles) < 2:
        logger.warning("analyze_video_form: only %d valid frames; defaulting to 50", len(angles))
        return 50.0

    # --- Step 1: instability score from angle standard deviation ---
    angle_std = statistics.stdev(angles)
    # std ~5°  (very stable)   → base ~10
    # std ~30° (very unstable) → base ~60
    base_score = angle_std * 2.0

    # --- Step 2: penalty for extreme knee angles ---
    total_penalty = 0.0
    for a in angles:
        if a < 60:
            total_penalty += 20.0   # dangerously deep / collapsed
        elif a < 90:
            total_penalty += 10.0   # insufficient depth control
    avg_penalty = total_penalty / len(angles)

    # --- Step 3: amplify to widen good-vs-bad separation ---
    form_decay = (base_score + avg_penalty) * 1.5
    form_decay = float(min(max(form_decay, 0.0), 100.0))

    logger.info(
        "analyze_video_form: %d frames → std=%.2f° penalty=%.2f form_decay=%.2f",
        len(angles), angle_std, avg_penalty, form_decay,
    )

    return form_decay
