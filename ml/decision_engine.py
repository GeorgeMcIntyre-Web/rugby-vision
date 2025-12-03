"""Forward pass decision engine utilities."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pvariance
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math


Point3D = Tuple[float, float, float]
Vector3D = Tuple[float, float, float]


@dataclass
class DecisionResult:
    """Decision result for a forward-pass analysis."""

    is_forward: bool
    confidence: float
    explanation: str
    metadata: Optional[Dict[str, Any]] = None


def smooth_trajectory(trajectory: Sequence[Point3D], window_size: int = 5) -> List[Point3D]:
    """Smooth a trajectory using a moving average window."""
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if not trajectory:
        return []
    if window_size == 1 or len(trajectory) < 3:
        return list(trajectory)
    half_window = window_size // 2
    smoothed: List[Point3D] = []
    for idx, _ in enumerate(trajectory):
        start = max(0, idx - half_window)
        end = min(len(trajectory), idx + half_window + 1)
        window = trajectory[start:end]
        smoothed.append(_average_points(window))
    return smoothed


def compute_ball_velocity(
    trajectory: Sequence[Point3D],
    timestamps: Sequence[float],
) -> List[Vector3D]:
    """Compute velocity vectors from positions and timestamps."""
    if len(trajectory) != len(timestamps):
        raise ValueError("trajectory and timestamps must be the same length")
    if len(trajectory) < 2:
        raise ValueError("trajectory must contain at least two points")
    velocities: List[Vector3D] = []
    for idx in range(len(trajectory)):
        if idx == 0:
            velocities.append(_velocity_between(trajectory[0], trajectory[1], timestamps[0], timestamps[1]))
            continue
        if idx == len(trajectory) - 1:
            velocities.append(_velocity_between(trajectory[-2], trajectory[-1], timestamps[-2], timestamps[-1]))
            continue
        velocities.append(
            _velocity_between(
                trajectory[idx - 1],
                trajectory[idx + 1],
                timestamps[idx - 1],
                timestamps[idx + 1],
            )
        )
    return velocities


def detect_pass_events(
    ball_trajectory_3d: Sequence[Point3D],
    timestamps: Sequence[float],
    velocity_threshold: float = 5.0,
    stabilization_window: int = 3,
) -> Tuple[int, int]:
    """Detect pass start/end frame indices using velocity heuristics."""
    if len(ball_trajectory_3d) != len(timestamps):
        raise ValueError("trajectory and timestamps must be the same length")
    if len(ball_trajectory_3d) < 2:
        raise ValueError("ball trajectory must contain at least two positions")
    velocities = compute_ball_velocity(ball_trajectory_3d, timestamps)
    speeds = [_vector_length(vec) for vec in velocities]
    start_idx = _first_index_above_threshold(speeds, velocity_threshold)
    if start_idx is None:
        return 0, len(ball_trajectory_3d) - 1
    end_idx = _detect_stabilization(ball_trajectory_3d, speeds, start_idx, stabilization_window)
    return start_idx, end_idx


def compute_confidence(
    detection_confidences: Optional[Sequence[float]],
    reprojection_errors: Optional[Sequence[float]],
    trajectory_variance: float,
    num_cameras: int,
) -> float:
    """Compute aggregated confidence score."""
    if num_cameras < 1:
        raise ValueError("num_cameras must be at least 1")
    base_conf = _clamp(mean(detection_confidences) if detection_confidences else 0.65, 0.0, 1.0)
    avg_error = mean(reprojection_errors) if reprojection_errors else 0.4
    error_factor = _clamp(1.0 - min(avg_error / 3.0, 0.9), 0.1, 1.0)
    camera_factor = _clamp(min(num_cameras, 4) / 4.0, 0.25, 1.0)
    smoothness_factor = _clamp(1.0 - min(trajectory_variance / 5.0, 0.9), 0.1, 1.0)
    confidence = base_conf * error_factor * camera_factor * smoothness_factor
    return round(_clamp(confidence, 0.0, 1.0), 3)


def analyze_forward_pass(
    ball_trajectory_3d: Sequence[Optional[Point3D]],
    passer_trajectory_3d: Optional[Sequence[Optional[Point3D]]],
    timestamps: Sequence[float],
    field_axis_forward: Vector3D,
    *,
    detection_confidences: Optional[Sequence[float]] = None,
    reprojection_errors: Optional[Sequence[float]] = None,
    num_cameras: int = 2,
    smoothing_window: int = 5,
) -> DecisionResult:
    """Analyse a pass using ball and passer trajectories."""
    if len(ball_trajectory_3d) != len(timestamps):
        raise ValueError("ball trajectory and timestamps must be the same length")
    if len(ball_trajectory_3d) < 2:
        raise ValueError("ball trajectory needs at least two samples")
    normalized_axis = _normalize(field_axis_forward)
    filled_ball, missing_ball = _fill_missing_points(ball_trajectory_3d, timestamps)
    raw_velocities = compute_ball_velocity(filled_ball, timestamps)
    raw_speeds = [_vector_length(vec) for vec in raw_velocities]
    raw_variance = pvariance(raw_speeds) if len(raw_speeds) > 1 else 0.0
    smoothed_ball = smooth_trajectory(filled_ball, smoothing_window)
    velocities = compute_ball_velocity(smoothed_ball, timestamps)
    speeds = [_vector_length(vec) for vec in velocities]
    variance = pvariance(speeds) if len(speeds) > 1 else 0.0
    start_idx, end_idx = detect_pass_events(smoothed_ball, timestamps)
    ball_forward_disp = _project_displacement(smoothed_ball[start_idx], smoothed_ball[end_idx], normalized_axis)
    passer_forward_disp = 0.0
    has_passer_data = bool(passer_trajectory_3d)
    if has_passer_data:
        filled_passer, _ = _fill_missing_points(passer_trajectory_3d or [], timestamps)
        smoothed_passer = smooth_trajectory(filled_passer, max(3, smoothing_window // 2))
        passer_start = smoothed_passer[min(start_idx, len(smoothed_passer) - 1)]
        passer_end = smoothed_passer[min(end_idx, len(smoothed_passer) - 1)]
        passer_forward_disp = _project_displacement(passer_start, passer_end, normalized_axis)
    margin = ball_forward_disp - passer_forward_disp
    margin_tolerance = 0.15
    is_forward = margin > margin_tolerance
    base_confidence = compute_confidence(detection_confidences, reprojection_errors, variance, num_cameras)
    margin_factor = _clamp(min(abs(margin) / 5.0, 1.0), 0.3, 1.0)
    passer_factor = 1.0 if has_passer_data else 0.75
    final_confidence = round(_clamp(base_confidence * margin_factor * passer_factor, 0.0, 1.0), 3)
    explanation = _build_explanation(is_forward, final_confidence, ball_forward_disp, passer_forward_disp)
    metadata = {
        "ball_forward_displacement": round(ball_forward_disp, 3),
        "passer_forward_displacement": round(passer_forward_disp, 3),
        "margin": round(margin, 3),
        "pass_frames": (start_idx, end_idx),
        "trajectory_variance": round(variance, 4),
        "raw_variance": round(raw_variance, 4),
        "smoothing_window": smoothing_window,
        "interpolated_frames": missing_ball,
        "has_passer_data": has_passer_data,
    }
    return DecisionResult(
        is_forward=is_forward,
        confidence=final_confidence,
        explanation=explanation,
        metadata=metadata,
    )


def _fill_missing_points(
    trajectory: Sequence[Optional[Point3D]],
    timestamps: Sequence[float],
) -> Tuple[List[Point3D], List[int]]:
    """Interpolate missing points in a trajectory."""
    if len(trajectory) != len(timestamps):
        raise ValueError("trajectory and timestamps must be the same length")
    if not trajectory:
        return [], []
    filled: List[Point3D] = []
    missing_indices: List[int] = []
    last_known_idx: Optional[int] = None
    last_known_point: Optional[Point3D] = None
    for idx, point in enumerate(trajectory):
        if point is not None:
            filled.append(point)
            last_known_idx = idx
            last_known_point = point
            continue
        missing_indices.append(idx)
        next_idx, next_point = _find_next_known(trajectory, idx)
        if last_known_point is None and next_point is None:
            raise ValueError("trajectory must contain at least one known point")
        if last_known_point is None:
            filled.append(next_point)  # type: ignore[arg-type]
            continue
        if next_point is None:
            filled.append(last_known_point)
            continue
        ratio = _interpolation_ratio(idx, last_known_idx, next_idx, timestamps)
        filled.append(_interpolate_points(last_known_point, next_point, ratio))
    return filled, missing_indices


def _average_points(points: Sequence[Point3D]) -> Point3D:
    """Average a list of points component-wise."""
    count = len(points)
    if count == 0:
        raise ValueError("points must not be empty")
    sum_x = sum(point[0] for point in points)
    sum_y = sum(point[1] for point in points)
    sum_z = sum(point[2] for point in points)
    return (sum_x / count, sum_y / count, sum_z / count)


def _velocity_between(
    start: Point3D,
    end: Point3D,
    start_time: float,
    end_time: float,
) -> Vector3D:
    """Compute velocity between two points."""
    delta_t = end_time - start_time
    if delta_t <= 0:
        raise ValueError("timestamps must be strictly increasing")
    delta = _subtract_points(end, start)
    return (delta[0] / delta_t, delta[1] / delta_t, delta[2] / delta_t)


def _subtract_points(a: Point3D, b: Point3D) -> Vector3D:
    """Subtract point b from point a."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vector_length(vec: Vector3D) -> float:
    """Compute vector magnitude."""
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


def _normalize(vec: Vector3D) -> Vector3D:
    """Normalize a vector."""
    length = _vector_length(vec)
    if length == 0:
        raise ValueError("cannot normalize zero-length vector")
    return (vec[0] / length, vec[1] / length, vec[2] / length)


def _project_displacement(start: Point3D, end: Point3D, axis: Vector3D) -> float:
    """Project displacement onto field axis."""
    displacement = _subtract_points(end, start)
    return displacement[0] * axis[0] + displacement[1] * axis[1] + displacement[2] * axis[2]


def _first_index_above_threshold(values: Sequence[float], threshold: float) -> Optional[int]:
    """Return first index where value >= threshold."""
    for idx, value in enumerate(values):
        if value >= threshold:
            return idx
    return None


def _detect_stabilization(
    trajectory: Sequence[Point3D],
    speeds: Sequence[float],
    start_idx: int,
    stabilization_window: int,
) -> int:
    """Detect end frame when speed stabilizes."""
    if stabilization_window < 1:
        stabilization_window = 1
    stationary_threshold = 1.0
    displacement_threshold = 0.5
    for idx in range(start_idx + 1, len(trajectory)):
        window_start = max(start_idx, idx - stabilization_window + 1)
        window = speeds[window_start : idx + 1]
        if not window:
            continue
        if max(window) > stationary_threshold:
            continue
        step_disp = _vector_length(_subtract_points(trajectory[idx], trajectory[idx - 1]))
        if step_disp < displacement_threshold:
            return idx
    return len(trajectory) - 1


def _find_next_known(
    trajectory: Sequence[Optional[Point3D]],
    current_idx: int,
) -> Tuple[Optional[int], Optional[Point3D]]:
    """Find the next known point after current_idx."""
    for idx in range(current_idx + 1, len(trajectory)):
        if trajectory[idx] is not None:
            return idx, trajectory[idx]
    return None, None


def _interpolation_ratio(
    idx: int,
    last_known_idx: Optional[int],
    next_known_idx: Optional[int],
    timestamps: Sequence[float],
) -> float:
    """Compute interpolation ratio between known samples."""
    if last_known_idx is None or next_known_idx is None:
        return 0.5
    last_time = timestamps[last_known_idx]
    next_time = timestamps[next_known_idx]
    span = next_time - last_time
    if span <= 0:
        return 0.5
    current_time = timestamps[idx]
    return _clamp((current_time - last_time) / span, 0.0, 1.0)


def _interpolate_points(start: Point3D, end: Point3D, ratio: float) -> Point3D:
    """Linearly interpolate between two points."""
    return (
        start[0] + (end[0] - start[0]) * ratio,
        start[1] + (end[1] - start[1]) * ratio,
        start[2] + (end[2] - start[2]) * ratio,
    )


def _build_explanation(
    is_forward: bool,
    confidence: float,
    ball_displacement: float,
    passer_displacement: float,
) -> str:
    """Create human-readable explanation."""
    direction = "forward" if is_forward else "not forward"
    return (
        f"Ball moved {ball_displacement:.2f}m along field axis while passer moved "
        f"{passer_displacement:.2f}m. Decision: {direction} "
        f"({confidence * 100:.1f}% confidence)."
    )


def _clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp value to [min_value, max_value]."""
    return max(min_value, min(value, max_value))

