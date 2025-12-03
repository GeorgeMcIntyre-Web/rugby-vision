"""Unit tests for the decision engine helpers."""

import pytest

from ml.decision_engine import (
    ConfidenceInputs,
    Point3D,
    Vector3D,
    analyze_forward_pass,
    compute_ball_velocity,
    compute_confidence,
    detect_pass_events,
    smooth_trajectory,
)


class TestTrajectoryProcessing:
    """Tests for trajectory smoothing and velocity helpers."""

    def test_smooth_trajectory_interpolates_missing_points(self) -> None:
        """Missing samples should be filled by interpolation before smoothing."""
        raw = [
            Point3D(0.0, 0.0, 0.0),
            None,
            Point3D(2.0, 0.0, 0.0),
            Point3D(2.5, 0.0, 0.0),
        ]

        smoothed = smooth_trajectory(raw, window_size=3)

        assert len(smoothed) == 4
        assert smoothed[1].x == pytest.approx(1.0, rel=1e-3)
        assert smoothed[2].x == pytest.approx(1.833333, rel=1e-3)

    def test_compute_ball_velocity_linear_motion(self) -> None:
        """Uniform motion should yield constant velocity vectors."""
        trajectory = [
            Point3D(0.0, 0.0, 0.0),
            Point3D(1.0, 0.0, 0.0),
            Point3D(2.0, 0.0, 0.0),
        ]
        timestamps = [0.0, 0.5, 1.0]

        velocities = compute_ball_velocity(trajectory, timestamps)

        assert len(velocities) == 3
        assert velocities[0].x == pytest.approx(2.0, rel=1e-3)
        assert velocities[1].x == pytest.approx(2.0, rel=1e-3)
        assert velocities[2].x == pytest.approx(2.0, rel=1e-3)


class TestPassDetection:
    """Tests for pass start/end detection."""

    def test_detect_pass_events_identifies_basic_pass(self) -> None:
        """Velocity spike followed by stabilization should mark a pass."""
        timestamps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
        trajectory = [
            Point3D(0.0, 0.0, 0.0),
            Point3D(0.0, 0.0, 0.0),
            Point3D(0.1, 0.0, 0.0),
            Point3D(1.6, 0.0, 0.0),
            Point3D(3.1, 0.0, 0.0),
            Point3D(3.2, 0.0, 0.0),
            Point3D(3.25, 0.0, 0.0),
            Point3D(3.26, 0.0, 0.0),
        ]

        start_idx, end_idx = detect_pass_events(trajectory, timestamps)

        assert start_idx == 3
        assert end_idx >= 5

    def test_detect_pass_events_handles_insufficient_motion(self) -> None:
        """Without a significant velocity change, no pass should be flagged."""
        timestamps = [0.0, 0.2, 0.4]
        trajectory = [
            Point3D(0.0, 0.0, 0.0),
            Point3D(0.1, 0.0, 0.0),
            Point3D(0.2, 0.0, 0.0),
        ]

        assert detect_pass_events(trajectory, timestamps) == (-1, -1)


class TestAnalyzeForwardPass:
    """Tests for the high-level forward pass decision logic."""

    def test_analyze_forward_pass_flags_forward_pass(self) -> None:
        """Ball that clearly outruns the passer should be marked forward."""
        timestamps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
        ball_trajectory = [
            Point3D(0.0, 0.0, 0.0),
            Point3D(0.1, 0.0, 0.0),
            Point3D(1.5, 0.0, 0.0),
            Point3D(3.0, 0.0, 0.0),
            Point3D(4.2, 0.0, 0.0),
            Point3D(4.4, 0.0, 0.0),
            Point3D(4.4, 0.0, 0.0),
            Point3D(4.4, 0.0, 0.0),
        ]
        passer_trajectory = [
            Point3D(0.0, 0.0, 0.0),
            Point3D(0.4, 0.0, 0.0),
            Point3D(0.8, 0.0, 0.0),
            Point3D(1.0, 0.0, 0.0),
        ]
        confidence_inputs = ConfidenceInputs(
            detection_confidences=[0.92] * len(ball_trajectory),
            reprojection_errors=[0.05] * len(ball_trajectory),
            num_cameras=4,
        )

        result = analyze_forward_pass(
            ball_trajectory_3d=ball_trajectory,
            passer_trajectory_3d=passer_trajectory,
            timestamps=timestamps,
            field_axis_forward=Vector3D(1.0, 0.0, 0.0),
            confidence_inputs=confidence_inputs,
        )

        assert result.is_forward is True
        assert "advanced" in result.explanation
        assert 0.6 <= result.confidence <= 1.0
        assert result.metadata["relative_displacement_m"] > 0

    def test_analyze_forward_pass_handles_backward_pass(self) -> None:
        """If the ball stays behind the passer, it should not be forward."""
        timestamps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ball_trajectory = [
            Point3D(0.0, 0.0, 0.0),
            Point3D(0.1, 0.0, 0.0),
            Point3D(2.2, 0.0, 0.0),
            Point3D(2.35, 0.0, 0.0),
            Point3D(2.4, 0.0, 0.0),
            Point3D(2.4, 0.0, 0.0),
        ]
        passer_trajectory = [
            Point3D(0.0, 0.0, 0.0),
            Point3D(1.4, 0.0, 0.0),
            Point3D(2.8, 0.0, 0.0),
        ]

        result = analyze_forward_pass(
            ball_trajectory_3d=ball_trajectory,
            passer_trajectory_3d=passer_trajectory,
            timestamps=timestamps,
            field_axis_forward=Vector3D(1.0, 0.0, 0.0),
        )

        assert result.is_forward is False
        assert "did not" in result.explanation.lower()
        assert 0.0 <= result.confidence <= 1.0


class TestComputeConfidence:
    """Tests for the compute_confidence helper."""

    def test_compute_confidence_high_quality_inputs(self) -> None:
        """High-quality signals should yield a strong confidence score."""
        confidence = compute_confidence(
            detection_confidences=[0.92, 0.95, 0.9],
            reprojection_errors=[0.05, 0.08, 0.04],
            trajectory_variance=0.03,
            num_cameras=4,
        )

        assert 0.8 < confidence <= 1.0

    def test_compute_confidence_low_quality_inputs(self) -> None:
        """Poor detections, geometry, and smoothness should reduce confidence."""
        confidence = compute_confidence(
            detection_confidences=[0.4, 0.35],
            reprojection_errors=[0.8, 1.1, 0.9],
            trajectory_variance=1.5,
            num_cameras=1,
        )

        assert 0.0 <= confidence < 0.05

    def test_compute_confidence_no_detections_returns_zero(self) -> None:
        """Missing detections should produce zero confidence."""
        confidence = compute_confidence(
            detection_confidences=[],
            reprojection_errors=[0.1, 0.2],
            trajectory_variance=0.2,
            num_cameras=3,
        )

        assert confidence == 0.0

    def test_compute_confidence_missing_geometry_metrics_uses_defaults(self) -> None:
        """Confidence falls back to defaults when reprojection errors are absent."""
        confidence = compute_confidence(
            detection_confidences=[0.9],
            reprojection_errors=[],
            trajectory_variance=0.2,
            num_cameras=2,
        )

        assert 0.4 <= confidence <= 0.5
