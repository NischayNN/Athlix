"""
angle_utils.py
--------------
Geometric utility functions for computing joint angles from 3D pose landmarks.
All angles are calculated using vectors derived from three joint positions and
returned in degrees.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Point3D = Tuple[float, float, float]  # (x, y, z) in normalized image space


# ---------------------------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------------------------

def _to_array(point: Point3D) -> np.ndarray:
    """Convert a (x, y, z) tuple to a NumPy array."""
    return np.array(point, dtype=np.float64)


def calculate_angle(a: Point3D, b: Point3D, c: Point3D) -> float:
    """
    Calculate the angle (in degrees) at joint *b*, formed by the vectors b→a
    and b→c.

    Parameters
    ----------
    a : Point3D
        First endpoint joint (e.g. hip).
    b : Point3D
        Vertex joint where the angle is measured (e.g. knee).
    c : Point3D
        Second endpoint joint (e.g. ankle).

    Returns
    -------
    float
        Angle in degrees in the range [0, 180].
    """
    vec_ba = _to_array(a) - _to_array(b)
    vec_bc = _to_array(c) - _to_array(b)

    norm_ba = np.linalg.norm(vec_ba)
    norm_bc = np.linalg.norm(vec_bc)

    # Guard against zero-length vectors (landmark not detected / coincident)
    if norm_ba < 1e-9 or norm_bc < 1e-9:
        return 0.0

    cos_angle = np.dot(vec_ba, vec_bc) / (norm_ba * norm_bc)
    # Clamp to [-1, 1] to prevent NaN from floating-point drift
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))

    return math.degrees(math.acos(cos_angle))


def calculate_angle_2d(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    2-D variant of *calculate_angle* that ignores the z axis entirely.

    Useful when depth data is unreliable (e.g. single monocular camera and
    MediaPipe depth estimation is noisy).

    Parameters
    ----------
    a, b, c : (x, y) tuples
        Same semantics as *calculate_angle*.

    Returns
    -------
    float
        Angle in degrees in the range [0, 180].
    """
    a2 = (a[0], a[1], 0.0)
    b2 = (b[0], b[1], 0.0)
    c2 = (c[0], c[1], 0.0)
    return calculate_angle(a2, b2, c2)


def euclidean_distance(p1: Point3D, p2: Point3D) -> float:
    """
    Euclidean distance between two 3-D points.

    Parameters
    ----------
    p1, p2 : Point3D

    Returns
    -------
    float
    """
    return float(np.linalg.norm(_to_array(p1) - _to_array(p2)))


def normalize_landmark(x: float, y: float, z: float) -> Point3D:
    """
    Return a validated landmark tuple, clamping coordinates to the expected
    MediaPipe normalized range [0, 1] for x and y axes.

    Parameters
    ----------
    x, y : float
        Normalized coordinates (0–1).
    z : float
        Depth value (already relative to hip depth in MediaPipe output).

    Returns
    -------
    Point3D
    """
    return (
        float(np.clip(x, 0.0, 1.0)),
        float(np.clip(y, 0.0, 1.0)),
        float(z),
    )
