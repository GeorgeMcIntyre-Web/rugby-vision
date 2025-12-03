"""Decision engine utilities for forward pass analysis.

This module hosts the heuristics-driven components that make up the
forward-pass decision engine. It includes pass detection, trajectory analysis,
confidence scoring, and the high-level decision routine used in Phase 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, pvariance
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Point3D",
    "Vector3D",
    "DecisionResult",
    "ConfidenceInputs",
    "smooth_trajectory",
    "compute_ball_velocity",
    "detect_pass_events",
    "analyze_forward_pass",
    "compute_confidence",
]

# Tunable heuristics used by multiple helpers. The values balance realism (based
# on rugby broadcast setups) with stability for synthetic tests.
_MAX_REPROJECTION_ERROR = 0.75  # meters; larger errors severely reduce trust
_DEFAULT_ERROR_FACTOR = 0.6  # fallback when reprojection metrics are missing
_MIN_ERROR_FACTOR = 0.1  # never drop to exactly zero to avoid divide-by-zero
_TARGET_CAMERA_COUNT = 4  # ideal number of cameras contributing to 3D solve
_SINGLE_CAMERA_FACTOR = 0.35  # heavy penalty when geometry relies on 1 camera
_DEFAULT_CAMERA_FACTOR = 0.5  # baseline for minimal stereo coverage
_MAX_TRAJECTORY_VARIANCE = 1.0  # meters^2; higher variance implies noisy track
_MIN_SMOOTHNESS_FACTOR = 0.2  # prevent collapse due to noise outliers
_PASS_SPEED_THRESHOLD = 5.0  # m/s; heuristic threshold for pass start
_PASS_END_SPEED_FACTOR = 0.4  # relative speed threshold for pass end
_PASS_STABLE_DISPLACEMENT = 0.35  # meters; stabilization window target
_DISPLACEMENT_TOLERANCE = 0.05  # meters; treat tiny displacement as neutral


@dataclass(frozen=True)
class Point3D:
    """Simple 3D point."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Vector3D:
    """Simple 3D vector with convenience helpers."""

    x: float
    y: float
    z: float

    def magnitude(self) -> float:
        return math.sqrt((self.x ** 2) + (self.y ** 2) + (self.z ** 2))

    def normalized(self) -> "Vector3D":
        norm = self.magnitude()
        if norm == 0.0:
            raise ValueError("Cannot normalize zero-length vector")
        return Vector3D(self.x / norm, self.y / norm, self.z / norm)


@dataclass(frozen=True)
class DecisionResult:
    """Represents the outcome of a forward-pass analysis."""

    is_forward: bool
    confidence: float
    explanation: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ConfidenceInputs:
    """Container for raw signals needed by the confidence heuristic."""

    detection_confidences: Sequence[float]
    reprojection_errors: Sequence[float]
    num_cameras: int


def smooth_trajectory(
    trajectory: Sequence[Optional[Point3D]],
    window_size: int = 5,
) -> List[Point3D]:
    """Smooth a 3D trajectory using interpolation plus moving average.

    Args:
        trajectory: Raw 3D points (missing samples can be represented as None).
        window_size: Window for the moving average filter (must be odd >= 3).

    Returns:
        Smoothed list of points. Missing samples are interpolated linearly.
    """
    if not trajectory:
        return []

    if window_size < 1:
        raise ValueError("window_size must be positive")

    filled = _interpolate_missing_points(trajectory)
    if len(filled) < 2 or window_size == 1:
        return filled

    half_window = window_size // 2
    smoothed: List[Point3D] = []
    for idx in range(len(filled)):
        start = max(0, idx - half_window)
        end = min(len(filled), idx + half_window + 1)
        window = filled[start:end]
        if not window:
            smoothed.append(filled[idx])
            continue
        smoothed.append(
            Point3D(
                x=fmean(point.x for point in window),
                y=fmean(point.y for point in window),
                z=fmean(point.z for point in window),
            )
        )
    return smoothed


def compute_ball_velocity(
    trajectory: Sequence[Point3D],
    timestamps: Sequence[float],
) -> List[Vector3D]:
    """Compute velocity vectors from a 3D trajectory.

    Returns one velocity vector per frame, using forward differences.
    """
    if len(trajectory) != len(timestamps):
        raise ValueError("trajectory and timestamps must align")

    if not trajectory:
        return []

    if len(trajectory) == 1:
        return [Vector3D(0.0, 0.0, 0.0)]

    velocities: List[Vector3D] = [Vector3D(0.0, 0.0, 0.0)]
    for idx in range(1, len(trajectory)):
        dt = timestamps[idx] - timestamps[idx - 1]
        if dt <= 0.0:
            velocities.append(Vector3D(0.0, 0.0, 0.0))
            continue
        delta = _vector_between(trajectory[idx - 1], trajectory[idx])
        velocities.append(
            Vector3D(
                x=delta.x / dt,
                y=delta.y / dt,
                z=delta.z / dt,
            )
        )

    velocities[0] = velocities[1]
    return velocities


def detect_pass_events(
    ball_trajectory_3d: Sequence[Optional[Point3D]],
    timestamps: Sequence[float],
    speed_threshold: float = _PASS_SPEED_THRESHOLD,
) -> Tuple[int, int]:
    """Detect pass start/end frame indices based on velocity cues."""
    processed = _interpolate_missing_points(ball_trajectory_3d)
    return _detect_pass_from_smoothed(processed, timestamps, speed_threshold)


def analyze_forward_pass(
    ball_trajectory_3d: Sequence[Optional[Point3D]],
    passer_trajectory_3d: Optional[Sequence[Point3D]],
    timestamps: Sequence[float],
    field_axis_forward: Vector3D,
    confidence_inputs: Optional[ConfidenceInputs] = None,
) -> DecisionResult:
    """Analyze whether a pass is forward per simplified rugby law heuristics."""
    if len(ball_trajectory_3d) != len(timestamps):
        raise ValueError("Ball trajectory and timestamps must have equal length")

    if len(ball_trajectory_3d) < 2:
        return DecisionResult(
            is_forward=False,
            confidence=0.0,
            explanation="Insufficient ball trajectory data",
            metadata=None,
        )

    axis = field_axis_forward.normalized()
    detection_trajectory = _interpolate_missing_points(ball_trajectory_3d)
    smoothed = smooth_trajectory(ball_trajectory_3d)
    start_idx, end_idx = _detect_pass_from_smoothed(
        detection_trajectory,
        timestamps,
        _PASS_SPEED_THRESHOLD,
    )

    metadata: Dict[str, Any] = {
        "pass_window": (start_idx, end_idx),
        "axis": axis,
    }

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        confidence = 0.0
        if confidence_inputs is not None:
            confidence = compute_confidence(
                list(confidence_inputs.detection_confidences),
                list(confidence_inputs.reprojection_errors),
                trajectory_variance=0.0,
                num_cameras=confidence_inputs.num_cameras,
            )
        return DecisionResult(
            is_forward=False,
            confidence=confidence,
            explanation="Unable to detect a complete pass event",
            metadata=metadata,
        )

    ball_start = smoothed[start_idx]
    ball_end = smoothed[end_idx]
    ball_displacement = _project_displacement(ball_start, ball_end, axis)

    passer_displacement = 0.0
    if passer_trajectory_3d and len(passer_trajectory_3d) >= 2:
        passer_displacement = _project_displacement(
            passer_trajectory_3d[0],
            passer_trajectory_3d[-1],
            axis,
        )

    relative_displacement = ball_displacement - passer_displacement
    is_forward = relative_displacement > _DISPLACEMENT_TOLERANCE

    metadata.update(
        {
            "ball_displacement_m": ball_displacement,
            "passer_displacement_m": passer_displacement,
            "relative_displacement_m": relative_displacement,
        }
    )

    explanation = (
        "Ball advanced %.2fm relative to passer" % relative_displacement
        if is_forward
        else "Ball did not travel beyond passer's forward movement"
    )

    variance = _compute_trajectory_variance(smoothed[start_idx : end_idx + 1])
    confidence = _fallback_confidence(relative_displacement, ball_displacement)
    if confidence_inputs is not None:
        confidence = compute_confidence(
            list(confidence_inputs.detection_confidences),
            list(confidence_inputs.reprojection_errors),
            trajectory_variance=variance,
            num_cameras=confidence_inputs.num_cameras,
        )

    return DecisionResult(
        is_forward=is_forward,
        confidence=confidence,
        explanation=explanation,
        metadata=metadata,
    )


def compute_confidence(
    detection_confidences: Sequence[float],
    reprojection_errors: Sequence[float],
    trajectory_variance: float,
    num_cameras: int,
) -> float:
    """Compute the overall confidence score for a pass decision."""
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


def _detect_pass_from_smoothed(
    smoothed_trajectory: Sequence[Point3D],
    timestamps: Sequence[float],
    speed_threshold: float,
) -> Tuple[int, int]:
    if len(smoothed_trajectory) != len(timestamps):
        raise ValueError("Smoothed trajectory and timestamps must align")

    if len(smoothed_trajectory) < 2:
        return (-1, -1)

    velocities = compute_ball_velocity(smoothed_trajectory, timestamps)
    speeds = [vector.magnitude() for vector in velocities]

    start_idx = _find_pass_start(speeds, speed_threshold)
    if start_idx == -1:
        return (-1, -1)

    end_idx = _find_pass_end(
        smoothed_trajectory,
        speeds,
        start_idx,
        speed_threshold,
    )
    return (start_idx, end_idx)


def _find_pass_start(speeds: Sequence[float], threshold: float) -> int:
    if not speeds:
        return -1

    for idx, speed in enumerate(speeds):
        if speed >= threshold:
            return idx
    return -1


def _find_pass_end(
    trajectory: Sequence[Point3D],
    speeds: Sequence[float],
    start_idx: int,
    threshold: float,
) -> int:
    if start_idx >= len(trajectory) - 1:
        return len(trajectory) - 1

    for idx in range(start_idx + 1, len(trajectory)):
        window_start = max(start_idx, idx - 3)
        window_points = trajectory[window_start : idx + 1]
        displacement = _window_displacement(window_points)
        speed = speeds[idx]
        if displacement <= _PASS_STABLE_DISPLACEMENT and speed <= threshold * _PASS_END_SPEED_FACTOR:
            return idx
    return len(trajectory) - 1


def _project_displacement(start: Point3D, end: Point3D, axis: Vector3D) -> float:
    delta = _vector_between(start, end)
    return _dot(delta, axis)


def _compute_trajectory_variance(points: Sequence[Point3D]) -> float:
    if len(points) < 2:
        return 0.0

    xs = [point.x for point in points]
    ys = [point.y for point in points]
    zs = [point.z for point in points]

    var_x = pvariance(xs)
    var_y = pvariance(ys)
    var_z = pvariance(zs)
    return (var_x + var_y + var_z) / 3.0


def _fallback_confidence(relative_displacement: float, ball_displacement: float) -> float:
    displacement_score = min(abs(relative_displacement) / 2.5, 1.0)
    motion_score = min(abs(ball_displacement) / 4.0, 1.0)
    base = 0.3 + (0.5 * displacement_score) + (0.2 * motion_score)
    return _clamp(base)


def _interpolate_missing_points(
    trajectory: Sequence[Optional[Point3D]],
) -> List[Point3D]:
    points = list(trajectory)
    if not points:
        return []

    valid_indices = [idx for idx, point in enumerate(points) if point is not None]
    if not valid_indices:
        return [Point3D(0.0, 0.0, 0.0) for _ in points]

    filled: List[Point3D] = [Point3D(0.0, 0.0, 0.0)] * len(points)
    first_valid = valid_indices[0]
    first_value = points[first_valid]
    assert first_value is not None

    for idx in range(first_valid + 1):
        filled[idx] = first_value

    last_valid_index = first_valid
    last_valid_point = first_value

    for idx in range(first_valid + 1, len(points)):
        current = points[idx]
        if current is not None:
            filled[idx] = current
            last_valid_index = idx
            last_valid_point = current
            continue

        next_index = _find_next_valid_index(points, idx)
        if next_index is None:
            filled[idx] = last_valid_point
            continue

        next_point = points[next_index]
        assert next_point is not None
        fraction = (idx - last_valid_index) / (next_index - last_valid_index)
        interpolated = _lerp_point(last_valid_point, next_point, fraction)
        filled[idx] = interpolated

    return filled


def _find_next_valid_index(
    points: Sequence[Optional[Point3D]],
    start_idx: int,
) -> Optional[int]:
    for idx in range(start_idx + 1, len(points)):
        if points[idx] is not None:
            return idx
    return None


def _lerp_point(start: Point3D, end: Point3D, fraction: float) -> Point3D:
    return Point3D(
        x=start.x + (end.x - start.x) * fraction,
        y=start.y + (end.y - start.y) * fraction,
        z=start.z + (end.z - start.z) * fraction,
    )


def _vector_between(start: Point3D, end: Point3D) -> Vector3D:
    return Vector3D(end.x - start.x, end.y - start.y, end.z - start.z)


def _dot(a: Vector3D, b: Vector3D) -> float:
    return (a.x * b.x) + (a.y * b.y) + (a.z * b.z)


def _window_displacement(points: Sequence[Point3D]) -> float:
    if len(points) < 2:
        return 0.0
    return _vector_between(points[0], points[-1]).magnitude()


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
