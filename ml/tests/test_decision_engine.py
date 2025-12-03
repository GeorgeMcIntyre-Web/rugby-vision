"""Unit tests for the decision engine helpers."""

from ml.decision_engine import compute_confidence


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
