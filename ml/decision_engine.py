"""Decision engine utilities for forward pass analysis.

Currently focuses on pass event detection heuristics (Task 6.2).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple


Point3D = Tuple[float, float, float]
Vector3D = Tuple[float, float, float]

DEFAULT_FRAME_INTERVAL_S = 1.0 / 30.0


def detect_pass_events(
    ball_trajectory_3d: Sequence[Optional[Point3D]],
    *,
    frame_interval_s: float = DEFAULT_FRAME_INTERVAL_S,
    min_speed_mps: float = 5.0,
    min_travel_distance_m: float = 1.0,
    detection_window: int = 4,
    min_fast_frames: int = 2,
    stabilization_speed_mps: float = 2.0,
    stabilization_window: int = 3,
    max_stationary_displacement_m: float = 0.3,
    min_flight_frames: int = 4,
) -> Tuple[int, int]:
    """Detect pass start/end frames from a 3D ball trajectory.

    Args:
        ball_trajectory_3d: Sequence of ball positions per frame. Missing frames
            can be expressed as None.
        frame_interval_s: Seconds between frames (defaults to 30 FPS).
        min_speed_mps: Speed threshold that indicates a pass impulse.
        min_travel_distance_m: Required displacement within detection_window.
        detection_window: Frames inspected to validate the impulse.
        min_fast_frames: Minimum high-speed frames to qualify as a pass.
        stabilization_speed_mps: Speed considered "controlled" catch speed.
        stabilization_window: Consecutive low-speed frames that mark pass end.
        max_stationary_displacement_m: Max displacement between frames to be
            deemed stationary.
        min_flight_frames: Minimum frames between start and end detection to
            avoid premature end markers.

    Returns:
        Tuple of (start_frame, end_frame). Returns (-1, -1) if no pass is found.
        If the ball is never clearly caught, end_frame falls back to the last
        observed frame.
    """
    if frame_interval_s <= 0:
        raise ValueError("frame_interval_s must be positive")

    if detection_window < 1:
        raise ValueError("detection_window must be >= 1")

    if stabilization_window < 1:
        raise ValueError("stabilization_window must be >= 1")

    if len(ball_trajectory_3d) < 2:
        return (-1, -1)

    positions = [_normalize_point(point) for point in ball_trajectory_3d]
    speeds = _compute_segment_speeds(positions, frame_interval_s)

    start_frame = _find_pass_start(
        positions,
        speeds,
        min_speed_mps=min_speed_mps,
        min_travel_distance_m=min_travel_distance_m,
        detection_window=detection_window,
        min_fast_frames=min_fast_frames,
    )

    if start_frame == -1:
        return (-1, -1)

    end_frame = _find_pass_end(
        positions,
        speeds,
        start_idx=start_frame,
        stabilization_speed_mps=stabilization_speed_mps,
        stabilization_window=stabilization_window,
        max_stationary_displacement_m=max_stationary_displacement_m,
        min_flight_frames=min_flight_frames,
    )

    return start_frame, end_frame


def _normalize_point(point: Optional[Sequence[float]]) -> Optional[Point3D]:
    """Convert arbitrary coordinate representation to Point3D."""
    if point is None:
        return None

    if len(point) != 3:
        raise ValueError("Trajectory points must be 3D coordinates")

    x, y, z = (float(coord) for coord in point)

    if not all(math.isfinite(coord) for coord in (x, y, z)):
        raise ValueError("Trajectory coordinates must be finite values")

    return (x, y, z)


def _compute_segment_speeds(
    positions: Sequence[Optional[Point3D]],
    frame_interval_s: float,
) -> List[Optional[float]]:
    """Compute instantaneous speeds between consecutive frames."""
    speeds: List[Optional[float]] = []
    if len(positions) < 2:
        return speeds

    inv_dt = 1.0 / frame_interval_s

    for idx in range(len(positions) - 1):
        start = positions[idx]
        end = positions[idx + 1]

        if start is None or end is None:
            speeds.append(None)
            continue

        distance = _point_distance(start, end)
        speeds.append(distance * inv_dt)

    return speeds


def _find_pass_start(
    positions: Sequence[Optional[Point3D]],
    speeds: Sequence[Optional[float]],
    *,
    min_speed_mps: float,
    min_travel_distance_m: float,
    detection_window: int,
    min_fast_frames: int,
) -> int:
    """Identify the first frame where the ball leaves the passer's hands."""
    last_index = len(positions) - 1
    if last_index < 1:
        return -1

    for idx, speed in enumerate(speeds):
        if speed is None:
            continue

        if speed < min_speed_mps:
            continue

        window_end = min(idx + detection_window, last_index)
        if window_end <= idx:
            continue

        if not _has_valid_points(positions[idx], positions[window_end]):
            continue

        displacement = _point_distance(
            positions[idx], positions[window_end]  # type: ignore[arg-type]
        )

        if displacement < min_travel_distance_m:
            continue

        fast_frames = _count_high_speed_frames(
            speeds,
            start_idx=idx,
            stop_frame_idx=window_end,
            threshold=min_speed_mps * 0.8,
        )

        if fast_frames < min_fast_frames:
            continue

        return min(idx + 1, last_index)

    return -1


def _find_pass_end(
    positions: Sequence[Optional[Point3D]],
    speeds: Sequence[Optional[float]],
    *,
    start_idx: int,
    stabilization_speed_mps: float,
    stabilization_window: int,
    max_stationary_displacement_m: float,
    min_flight_frames: int,
) -> int:
    """Identify when the ball is caught or grounded."""
    if start_idx < 0:
        return len(positions) - 1

    stable_run = 0
    min_idx_for_end = start_idx + min_flight_frames

    for idx in range(start_idx + 1, len(positions) - 1):
        if idx < min_idx_for_end:
            stable_run = 0
            continue

        speed = speeds[idx] if idx < len(speeds) else None
        if speed is None:
            stable_run = 0
            continue

        if speed > stabilization_speed_mps:
            stable_run = 0
            continue

        current_point = positions[idx]
        next_point = positions[idx + 1]

        if not _has_valid_points(current_point, next_point):
            stable_run = 0
            continue

        displacement = _point_distance(
            current_point, next_point  # type: ignore[arg-type]
        )

        if displacement > max_stationary_displacement_m:
            stable_run = 0
            continue

        stable_run += 1

        if stable_run >= stabilization_window:
            return idx + 1

    return len(positions) - 1


def _count_high_speed_frames(
    speeds: Sequence[Optional[float]],
    *,
    start_idx: int,
    stop_frame_idx: int,
    threshold: float,
) -> int:
    """Count how many segments stay above a given threshold."""
    if stop_frame_idx <= start_idx:
        return 0

    stop_speed_idx = min(stop_frame_idx, len(speeds))
    count = 0

    for speed in speeds[start_idx:stop_speed_idx]:
        if speed is None:
            continue

        if speed >= threshold:
            count += 1

    return count


def _has_valid_points(
    p1: Optional[Point3D],
    p2: Optional[Point3D],
) -> bool:
    """Quick helper to verify two points exist."""
    return p1 is not None and p2 is not None


def _point_distance(p1: Point3D, p2: Point3D) -> float:
    """Euclidean distance between two 3D points."""
    return math.dist(p1, p2)
