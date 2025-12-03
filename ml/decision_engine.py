"""Phase 6 decision engine utilities for trajectory analysis.

This module currently focuses on Task 6.3 deliverables:
    - Trajectory smoothing with polynomial fitting
    - Velocity vector computation from smoothed 3D positions
    - Handling of missing measurements via interpolation
    - Trajectory variance calculation for confidence estimation inputs

Future tasks (6.4+ and 6.5) will extend this module with full decision
logic and confidence scoring that build on these primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Point3D:
    """Represents a 3D point in meters."""

    x: float
    y: float
    z: float

    def as_array(self) -> np.ndarray:
        """Convert point to numpy array."""
        return np.array([self.x, self.y, self.z], dtype=float)

    @staticmethod
    def from_array(values: Sequence[float]) -> "Point3D":
        """Create a point from iterable values."""
        if len(values) != 3:
            raise ValueError("Point3D requires exactly 3 values")
        return Point3D(float(values[0]), float(values[1]), float(values[2]))


@dataclass(frozen=True)
class Vector3D:
    """Represents a 3D vector (e.g., velocity)."""

    x: float
    y: float
    z: float

    def magnitude(self) -> float:
        """Return vector magnitude."""
        return float(np.sqrt(self.x**2 + self.y**2 + self.z**2))

    def as_array(self) -> np.ndarray:
        """Convert vector to numpy array."""
        return np.array([self.x, self.y, self.z], dtype=float)

    @staticmethod
    def from_array(values: Sequence[float]) -> "Vector3D":
        """Create a vector from iterable values."""
        if len(values) != 3:
            raise ValueError("Vector3D requires exactly 3 values")
        return Vector3D(float(values[0]), float(values[1]), float(values[2]))


def smooth_trajectory(
    trajectory: Sequence[Optional[Point3D]],
    window_size: int = 5,
    poly_degree: int = 2
) -> List[Point3D]:
    """Smooth a 3D trajectory using sliding-window polynomial fitting.

    Args:
        trajectory: Sequence of ball positions (None indicates missing data).
        window_size: Sliding window size (must be odd).
        poly_degree: Polynomial degree for the fit (default quadratic).

    Returns:
        Smoothed trajectory with same length as input.

    Raises:
        ValueError: If parameters are invalid or no valid points exist.
    """
    if len(trajectory) == 0:
        return []

    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd for polynomial smoothing")

    points = _interpolate_missing_points(trajectory)
    if len(points) < 3:
        return points

    adjusted_window = min(window_size, len(points))
    if adjusted_window % 2 == 0:
        adjusted_window -= 1

    if adjusted_window < 3:
        return points

    poly_degree = max(1, min(poly_degree, adjusted_window - 1))
    indices = np.arange(len(points))
    coords = np.array([p.as_array() for p in points], dtype=float)
    smoothed = np.empty_like(coords)
    half_window = adjusted_window // 2

    for idx in range(len(points)):
        start = max(0, idx - half_window)
        end = min(len(points), idx + half_window + 1)
        window_indices = indices[start:end]

        for axis in range(3):
            values = coords[start:end, axis]
            degree = min(poly_degree, len(window_indices) - 1)

            if degree <= 0:
                smoothed[idx, axis] = coords[idx, axis]
                continue

            coeffs = np.polyfit(window_indices, values, deg=degree)
            smoothed[idx, axis] = np.polyval(coeffs, indices[idx])

    return [Point3D.from_array(row) for row in smoothed]


def compute_ball_velocity(
    trajectory: Sequence[Optional[Point3D]],
    timestamps: Sequence[float],
    smooth_first: bool = True,
    window_size: int = 5
) -> List[Vector3D]:
    """Compute velocity vectors from a 3D trajectory.

    Args:
        trajectory: Sequence of 3D ball positions (None allowed).
        timestamps: Sequence of timestamps (seconds) corresponding to positions.
        smooth_first: Whether to smooth the trajectory prior to differentiation.
        window_size: Window size for smoothing if enabled.

    Returns:
        List of velocity vectors with same length as trajectory.

    Raises:
        ValueError: If inputs are invalid or timestamps are not increasing.
    """
    if len(trajectory) != len(timestamps):
        raise ValueError("trajectory and timestamps must be the same length")

    if len(trajectory) < 2:
        return []

    if smooth_first:
        positions = smooth_trajectory(trajectory, window_size=window_size)
    else:
        positions = _interpolate_missing_points(trajectory)

    times = np.array(timestamps, dtype=float)
    if not np.all(np.isfinite(times)):
        raise ValueError("timestamps must be finite numbers")

    deltas = np.diff(times)
    if np.any(deltas <= 0):
        raise ValueError("timestamps must be strictly increasing")

    arrays = np.array([p.as_array() for p in positions], dtype=float)
    velocities: List[Vector3D] = []

    for idx in range(len(arrays)):
        if idx == 0:
            delta_pos = arrays[1] - arrays[0]
            delta_time = times[1] - times[0]
        elif idx == len(arrays) - 1:
            delta_pos = arrays[-1] - arrays[-2]
            delta_time = times[-1] - times[-2]
        else:
            delta_pos = arrays[idx + 1] - arrays[idx - 1]
            delta_time = times[idx + 1] - times[idx - 1]

        if delta_time <= 0:
            raise ValueError("timestamps must be strictly increasing")

        velocity_vector = delta_pos / delta_time
        velocities.append(Vector3D.from_array(velocity_vector))

    return velocities


def compute_trajectory_variance(trajectory: Sequence[Point3D]) -> float:
    """Compute variance of displacement magnitudes between consecutive points.

    Lower variance indicates a smoother trajectory and can be inverted into
    a smoothness-based confidence score by downstream components.

    Args:
        trajectory: Sequence of 3D points (must contain at least 2 entries).

    Returns:
        Variance of step magnitudes (0.0 if insufficient data).
    """
    if len(trajectory) < 2:
        return 0.0

    coords = np.array([p.as_array() for p in trajectory], dtype=float)
    displacements = np.diff(coords, axis=0)
    if displacements.size == 0:
        return 0.0

    magnitudes = np.linalg.norm(displacements, axis=1)
    if magnitudes.size == 0:
        return 0.0

    return float(np.var(magnitudes))


def _interpolate_missing_points(
    trajectory: Sequence[Optional[Point3D]]
) -> List[Point3D]:
    """Fill missing trajectory points with linear interpolation.

    Args:
        trajectory: Sequence possibly containing None entries.

    Returns:
        List of Point3D with missing values interpolated.

    Raises:
        ValueError: If no valid points exist.
    """
    if len(trajectory) == 0:
        return []

    arrays = np.array([
        point.as_array() if point is not None else [np.nan, np.nan, np.nan]
        for point in trajectory
    ], dtype=float)

    valid_mask = ~np.isnan(arrays)
    if not valid_mask.any():
        raise ValueError("trajectory contains no valid points to interpolate")

    indices = np.arange(len(arrays))

    for axis in range(3):
        axis_values = arrays[:, axis]
        axis_valid = ~np.isnan(axis_values)
        if not axis_valid.any():
            raise ValueError("trajectory axis contains only missing values")

        axis_values[~axis_valid] = np.interp(
            indices[~axis_valid],
            indices[axis_valid],
            axis_values[axis_valid]
        )
        arrays[:, axis] = axis_values

    return [Point3D.from_array(row) for row in arrays]
