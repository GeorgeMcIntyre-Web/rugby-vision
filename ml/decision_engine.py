"""Decision engine utilities for forward pass analysis.

This module will host the heuristics-driven components that make up the
forward-pass decision engine. Task 6.5 focuses on confidence scoring, so this
file currently exposes the confidence computation helper that combines signal
quality metrics collected throughout the pipeline.
"""

from __future__ import annotations

from statistics import fmean
from typing import List, Sequence

__all__ = ["compute_confidence"]

# Tunable heuristics used by the confidence scoring function. The values balance
# realism (based on rugby broadcast setups) with stability for synthetic tests.
_MAX_REPROJECTION_ERROR = 0.75  # meters; larger errors severely reduce trust
_DEFAULT_ERROR_FACTOR = 0.6  # fallback when reprojection metrics are missing
_MIN_ERROR_FACTOR = 0.1  # never drop to exactly zero to avoid divide-by-zero
_TARGET_CAMERA_COUNT = 4  # ideal number of cameras contributing to 3D solve
_SINGLE_CAMERA_FACTOR = 0.35  # heavy penalty when geometry relies on 1 camera
_DEFAULT_CAMERA_FACTOR = 0.5  # baseline for minimal stereo coverage
_MAX_TRAJECTORY_VARIANCE = 1.0  # meters^2; higher variance implies noisy track
_MIN_SMOOTHNESS_FACTOR = 0.2  # prevent total collapse due to noise outliers


def compute_confidence(
    detection_confidences: List[float],
    reprojection_errors: List[float],
    trajectory_variance: float,
    num_cameras: int,
) -> float:
    """Compute the overall confidence score for a pass decision.

    The score is derived by combining three components:
    1. **Base detection confidence** – aggregate confidence from raw detections.
    2. **3D quality factor** – penalizes poor reprojection error and low camera
       counts.
    3. **Trajectory smoothness factor** – rewards stable ball trajectories.

    Args:
        detection_confidences: Detector confidence scores (0.0 - 1.0).
        reprojection_errors: Per-frame reprojection errors in meters.
        trajectory_variance: Variance of the smoothed trajectory in meters^2.
        num_cameras: Number of cameras used in the 3D reconstruction.

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    base_confidence = _aggregate_detection_confidence(detection_confidences)
    if base_confidence == 0.0:
        return 0.0

    geometry_quality = _compute_3d_quality_factor(
        reprojection_errors,
        num_cameras,
    )
    smoothness_factor = _compute_smoothness_factor(trajectory_variance)

    combined = base_confidence * geometry_quality * smoothness_factor
    return _clamp(combined)


def _aggregate_detection_confidence(confidences: Sequence[float]) -> float:
    if not confidences:
        return 0.0

    sanitized = [_clamp(value) for value in confidences]
    return _clamp(fmean(sanitized))


def _compute_3d_quality_factor(
    reprojection_errors: Sequence[float],
    num_cameras: int,
) -> float:
    error_factor = _compute_error_factor(reprojection_errors)
    camera_factor = _compute_camera_factor(num_cameras)

    # Weight reprojection accuracy slightly higher than camera coverage because
    # large errors often indicate calibration issues even with many cameras.
    weighted = (0.6 * error_factor) + (0.4 * camera_factor)
    return _clamp(weighted)


def _compute_error_factor(errors: Sequence[float]) -> float:
    if not errors:
        return _DEFAULT_ERROR_FACTOR

    mean_error = fmean(abs(value) for value in errors)
    if mean_error <= 0.0:
        return 1.0

    normalized = mean_error / _MAX_REPROJECTION_ERROR
    if normalized >= 1.0:
        return _MIN_ERROR_FACTOR

    factor = 1.0 - normalized
    if factor < _MIN_ERROR_FACTOR:
        return _MIN_ERROR_FACTOR

    return factor


def _compute_camera_factor(num_cameras: int) -> float:
    if num_cameras <= 0:
        return 0.0

    if num_cameras == 1:
        return _SINGLE_CAMERA_FACTOR

    normalized = num_cameras / _TARGET_CAMERA_COUNT
    if normalized >= 1.0:
        return 1.0

    if normalized < _DEFAULT_CAMERA_FACTOR:
        return _DEFAULT_CAMERA_FACTOR

    return normalized


def _compute_smoothness_factor(trajectory_variance: float) -> float:
    if trajectory_variance <= 0.0:
        return 1.0

    normalized = trajectory_variance / _MAX_TRAJECTORY_VARIANCE
    if normalized >= 1.0:
        return _MIN_SMOOTHNESS_FACTOR

    penalty = 0.6 * normalized
    factor = 1.0 - penalty
    if factor < _MIN_SMOOTHNESS_FACTOR:
        return _MIN_SMOOTHNESS_FACTOR

    return factor


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    if value < lower:
        return lower

    if value > upper:
        return upper

    return value
