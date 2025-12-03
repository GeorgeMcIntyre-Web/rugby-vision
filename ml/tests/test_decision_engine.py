"""Tests for the forward pass decision engine."""

from __future__ import annotations

from typing import List, Optional, Sequence

from ml.decision_engine import (
    DecisionResult,
    analyze_forward_pass,
    compute_confidence,
    detect_pass_events,
    smooth_trajectory,
)
from ml.tests.fixtures.trajectories import (
    apply_noise,
    generate_linear_trajectory,
    generate_noise_pattern,
    generate_timestamps,
    inject_missing_points,
)

FORWARD_AXIS = (1.0, 0.0, 0.0)


class TestAnalyzeForwardPass:
    """End-to-end decision scenarios."""

    def _run_analysis(
        self,
        ball: Sequence[Optional[tuple[float, float, float]]],
        passer: Optional[Sequence[Optional[tuple[float, float, float]]]],
        timestamps: Sequence[float],
        *,
        detection_conf: Optional[List[float]] = None,
        reprojection_err: Optional[List[float]] = None,
        num_cameras: int = 3,
    ) -> DecisionResult:
        return analyze_forward_pass(
            ball_trajectory_3d=ball,
            passer_trajectory_3d=passer,
            timestamps=timestamps,
            field_axis_forward=FORWARD_AXIS,
            detection_confidences=detection_conf,
            reprojection_errors=reprojection_err,
            num_cameras=num_cameras,
        )

    def test_clearly_forward_pass(self) -> None:
        frame_count = 20
        timestamps = generate_timestamps(frame_count, 0.08)
        ball = generate_linear_trajectory((0.0, 0.0, 1.0), (0.55, 0.0, 0.0), frame_count)
        passer = generate_linear_trajectory((-1.0, 0.0, 1.0), (0.1, 0.0, 0.0), frame_count)
        detection_conf = [0.92] * frame_count
        reprojection_err = [0.12] * frame_count

        result = self._run_analysis(
            ball,
            passer,
            timestamps,
            detection_conf=detection_conf,
            reprojection_err=reprojection_err,
            num_cameras=4,
        )

        assert result.is_forward is True
        assert result.confidence >= 0.5
        assert result.metadata is not None
        assert result.metadata["margin"] > 3.0

    def test_clearly_backward_pass(self) -> None:
        frame_count = 18
        timestamps = generate_timestamps(frame_count, 0.08)
        ball = generate_linear_trajectory((5.0, 0.0, 1.0), (-0.25, 0.0, 0.0), frame_count)
        passer = generate_linear_trajectory((4.0, 0.0, 1.0), (0.08, 0.0, 0.0), frame_count)
        detection_conf = [0.88] * frame_count

        result = self._run_analysis(ball, passer, timestamps, detection_conf=detection_conf, num_cameras=3)

        assert result.is_forward is False
        assert result.metadata is not None
        assert result.metadata["margin"] < -1.0

    def test_borderline_forward_low_confidence(self) -> None:
        frame_count = 15
        timestamps = generate_timestamps(frame_count, 0.1)
        ball = generate_linear_trajectory((0.0, 0.0, 1.0), (0.2, 0.0, 0.0), frame_count)
        passer = generate_linear_trajectory((0.0, 0.0, 1.0), (0.14, 0.0, 0.0), frame_count)

        result = self._run_analysis(ball, passer, timestamps, num_cameras=2)

        assert result.is_forward is True
        assert result.confidence < 0.5
        assert 0.3 < result.metadata["margin"] < 1.0  # type: ignore[index]

    def test_borderline_backward_pass(self) -> None:
        frame_count = 15
        timestamps = generate_timestamps(frame_count, 0.1)
        ball = generate_linear_trajectory((1.0, 0.0, 1.0), (0.08, 0.0, 0.0), frame_count)
        passer = generate_linear_trajectory((1.0, 0.0, 1.0), (0.1, 0.0, 0.0), frame_count)

        result = self._run_analysis(ball, passer, timestamps)

        assert result.is_forward is False
        assert result.confidence < 0.5
        assert result.metadata is not None
        assert result.metadata["margin"] < 0

    def test_no_passer_data_reduces_confidence(self) -> None:
        frame_count = 16
        timestamps = generate_timestamps(frame_count, 0.08)
        ball = generate_linear_trajectory((0.0, 0.0, 1.0), (0.5, 0.0, 0.0), frame_count)
        passer = generate_linear_trajectory((-0.5, 0.0, 1.0), (0.12, 0.0, 0.0), frame_count)

        result_with_passer = self._run_analysis(ball, passer, timestamps, num_cameras=3)
        result_without_passer = self._run_analysis(ball, None, timestamps, num_cameras=3)

        assert result_with_passer.is_forward is True
        assert result_without_passer.is_forward is True
        assert result_without_passer.confidence < result_with_passer.confidence
        assert result_without_passer.metadata is not None
        assert result_without_passer.metadata["has_passer_data"] is False

    def test_noisy_trajectory_gets_smoothed(self) -> None:
        frame_count = 22
        timestamps = generate_timestamps(frame_count, 0.08)
        base_ball = generate_linear_trajectory((0.0, 0.0, 1.0), (0.5, 0.0, 0.0), frame_count)
        noise = generate_noise_pattern(frame_count, magnitude=0.3)
        noisy_ball = apply_noise(base_ball, noise)
        passer = generate_linear_trajectory((-0.2, 0.0, 1.0), (0.12, 0.0, 0.0), frame_count)

        baseline = self._run_analysis(base_ball, passer, timestamps, num_cameras=3)
        noisy_result = self._run_analysis(noisy_ball, passer, timestamps, num_cameras=3)

        assert baseline.is_forward is True
        assert noisy_result.is_forward is True
        assert noisy_result.metadata is not None
        assert baseline.metadata is not None
        assert abs(noisy_result.metadata["margin"] - baseline.metadata["margin"]) < 1.5
        assert noisy_result.confidence >= baseline.confidence * 0.6

    def test_missing_frames_are_interpolated(self) -> None:
        frame_count = 18
        timestamps = generate_timestamps(frame_count, 0.08)
        ball_base = generate_linear_trajectory((0.0, 0.0, 1.0), (0.45, 0.0, 0.0), frame_count)
        passer_base = generate_linear_trajectory((-0.5, 0.0, 1.0), (0.1, 0.0, 0.0), frame_count)
        missing = [3, 4, 7]
        ball = inject_missing_points(ball_base, missing)
        passer = inject_missing_points(passer_base, [7])

        result = self._run_analysis(ball, passer, timestamps, num_cameras=3)

        assert result.is_forward is True
        assert result.metadata is not None
        assert result.metadata["interpolated_frames"] == missing
        assert result.metadata["has_passer_data"] is True


class TestDecisionEngineUtilities:
    """Unit tests for helper utilities."""

    def test_smooth_trajectory_reduces_noise(self) -> None:
        frame_count = 10
        base = generate_linear_trajectory((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), frame_count)
        noise = generate_noise_pattern(frame_count, magnitude=0.4)
        noisy = apply_noise(base, noise)

        smoothed = smooth_trajectory(noisy, window_size=5)
        velocities_raw = [abs(noisy[idx + 1][0] - noisy[idx][0]) for idx in range(frame_count - 1)]
        velocities_smooth = [abs(smoothed[idx + 1][0] - smoothed[idx][0]) for idx in range(frame_count - 1)]

        assert max(velocities_smooth) <= max(velocities_raw)

    def test_compute_confidence_combines_factors(self) -> None:
        detection_conf = [0.95, 0.9, 0.92]
        reprojection_err = [0.15, 0.2, 0.25]
        confidence = compute_confidence(
            detection_confidences=detection_conf,
            reprojection_errors=reprojection_err,
            trajectory_variance=0.5,
            num_cameras=4,
        )
        assert 0.5 < confidence <= 1.0

    def test_detect_pass_events_uses_velocity_threshold(self) -> None:
        timestamps = generate_timestamps(6, 0.08)
        # Ball stationary first two frames, then rapid movement
        ball = [
            (0.0, 0.0, 0.0),
            (0.02, 0.0, 0.0),
            (0.05, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.6, 0.0, 0.0),
            (2.0, 0.0, 0.0),
        ]

        start_idx, end_idx = detect_pass_events(ball, timestamps, velocity_threshold=5.0)

        assert start_idx >= 2
        assert end_idx == len(ball) - 1

