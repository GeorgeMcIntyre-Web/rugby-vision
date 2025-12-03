"""Synthetic trajectory helpers for decision engine tests."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

Point3D = Tuple[float, float, float]
Vector3D = Tuple[float, float, float]


def generate_timestamps(frame_count: int, frame_interval: float = 0.1) -> List[float]:
    """Generate monotonically increasing timestamps."""
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    if frame_interval <= 0:
        raise ValueError("frame_interval must be positive")
    return [idx * frame_interval for idx in range(frame_count)]


def generate_linear_trajectory(
    start: Point3D,
    delta_per_frame: Vector3D,
    frame_count: int,
) -> List[Point3D]:
    """Create a linear trajectory with constant displacement per frame."""
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    trajectory: List[Point3D] = []
    for idx in range(frame_count):
        trajectory.append(
            (
                start[0] + delta_per_frame[0] * idx,
                start[1] + delta_per_frame[1] * idx,
                start[2] + delta_per_frame[2] * idx,
            )
        )
    return trajectory


def generate_noise_pattern(frame_count: int, magnitude: float) -> List[Vector3D]:
    """Create deterministic alternating noise pattern for reproducible tests."""
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    return [
        (
            magnitude if idx % 2 == 0 else -magnitude,
            0.0,
            0.0,
        )
        for idx in range(frame_count)
    ]


def apply_noise(
    trajectory: Sequence[Point3D],
    noise_pattern: Sequence[Vector3D],
) -> List[Point3D]:
    """Apply additive noise pattern to a base trajectory."""
    if len(trajectory) != len(noise_pattern):
        raise ValueError("trajectory and noise_pattern must be the same length")
    noisy: List[Point3D] = []
    for point, noise in zip(trajectory, noise_pattern):
        noisy.append(
            (
                point[0] + noise[0],
                point[1] + noise[1],
                point[2] + noise[2],
            )
        )
    return noisy


def inject_missing_points(
    trajectory: Sequence[Point3D],
    missing_indices: Iterable[int],
) -> List[Optional[Point3D]]:
    """Inject None entries at specified indices."""
    missing_set = set(missing_indices)
    with_missing: List[Optional[Point3D]] = []
    for idx, point in enumerate(trajectory):
        if idx in missing_set:
            with_missing.append(None)
            continue
        with_missing.append(point)
    return with_missing

