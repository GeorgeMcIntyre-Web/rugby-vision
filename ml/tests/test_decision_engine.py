"""Tests for ML decision engine trajectory utilities."""

import pytest

from ml.decision_engine import (
    Point3D,
    compute_ball_velocity,
    compute_trajectory_variance,
    smooth_trajectory,
)


class TestSmoothTrajectory:
    """Tests for trajectory smoothing and interpolation."""

    def test_interpolates_missing_points(self) -> None:
        """Ensure None entries are interpolated smoothly."""
        trajectory = [
            Point3D(0.0, 0.0, 0.0),
            None,
            Point3D(2.0, 0.0, 0.0),
            Point3D(3.0, 0.0, 0.0),
        ]

        smoothed = smooth_trajectory(trajectory, window_size=3)

        assert len(smoothed) == len(trajectory)
        assert smoothed[1].x == pytest.approx(1.0, abs=0.15)
        assert smoothed[1].y == pytest.approx(0.0, abs=1e-6)
        assert smoothed[1].z == pytest.approx(0.0, abs=1e-6)

    def test_reduces_spikes_from_noise(self) -> None:
        """Noisy spike should be pulled toward neighborhood average."""
        base = [Point3D(float(i), 0.0, 0.0) for i in range(6)]
        noisy = base.copy()
        noisy[3] = Point3D(9.0, 0.0, 0.0)  # Introduce spike

        smoothed = smooth_trajectory(noisy, window_size=5)

        spike_distance = abs(noisy[3].x - 3.0)
        smoothed_distance = abs(smoothed[3].x - 3.0)
        assert smoothed_distance < spike_distance


class TestComputeBallVelocity:
    """Tests for velocity vector estimation."""

    def test_constant_speed_motion(self) -> None:
        """Linear motion should yield constant velocity magnitude."""
        trajectory = [Point3D(float(i), 0.0, 0.0) for i in range(6)]
        timestamps = [i * 0.1 for i in range(6)]  # 10 m/s along X

        velocities = compute_ball_velocity(
            trajectory,
            timestamps,
            smooth_first=False,
        )

        assert len(velocities) == len(trajectory)
        for vector in velocities:
            assert vector.x == pytest.approx(10.0, rel=0.01)
            assert vector.y == pytest.approx(0.0, abs=1e-6)
            assert vector.z == pytest.approx(0.0, abs=1e-6)

    def test_raises_on_length_mismatch(self) -> None:
        """Trajectory and timestamps must align."""
        trajectory = [Point3D(0.0, 0.0, 0.0), Point3D(1.0, 0.0, 0.0)]
        timestamps = [0.0]

        with pytest.raises(ValueError, match="same length"):
            compute_ball_velocity(trajectory, timestamps)

    def test_raises_on_non_monotonic_timestamps(self) -> None:
        """Timestamps must be strictly increasing."""
        trajectory = [Point3D(0.0, 0.0, 0.0), Point3D(1.0, 0.0, 0.0)]
        timestamps = [0.0, 0.0]

        with pytest.raises(ValueError, match="strictly increasing"):
            compute_ball_velocity(trajectory, timestamps)


class TestTrajectoryVariance:
    """Tests for smoothness-related metrics."""

    def test_zero_variance_for_uniform_motion(self) -> None:
        """Uniform step sizes should yield zero variance."""
        trajectory = [Point3D(float(i), 0.0, 0.0) for i in range(5)]

        variance = compute_trajectory_variance(trajectory)
        assert variance == pytest.approx(0.0, abs=1e-9)

    def test_higher_variance_for_irregular_motion(self) -> None:
        """Irregular motion should increase variance."""
        base = [Point3D(float(i), 0.0, 0.0) for i in range(5)]
        irregular = base.copy()
        irregular[3] = Point3D(6.0, 0.0, 0.0)

        base_variance = compute_trajectory_variance(base)
        irregular_variance = compute_trajectory_variance(irregular)

        assert irregular_variance > base_variance
