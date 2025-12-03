"""Tests for the forward pass decision engine."""

from typing import List, Optional

import numpy as np
import pytest

from ml.decision_engine import analyze_forward_pass


def _linear_trajectory(
    start: float,
    delta: float,
    steps: int,
    y: float = 0.0,
    z: float = 1.5
) -> List[np.ndarray]:
    """Create linear trajectory along X axis."""
    if steps < 2:
        raise ValueError("steps must be >= 2")

    trajectory: List[np.ndarray] = []
    for idx in range(steps):
        fraction = idx / (steps - 1)
        position = np.array(
            [
                start + delta * fraction,
                y,
                z,
            ],
            dtype=float,
        )
        trajectory.append(position)
    return trajectory


def _timestamps(steps: int, dt: float = 0.1) -> List[float]:
    """Generate timestamps."""
    return [i * dt for i in range(steps)]


class TestAnalyzeForwardPass:
    """Decision logic behaviour tests."""

    field_axis = np.array([1.0, 0.0, 0.0], dtype=float)

    def _run_decision(
        self,
        ball_delta: float,
        passer_delta: Optional[float],
        steps: int = 6
    ):
        """Helper to run analysis with synthetic trajectories."""
        ball_traj = _linear_trajectory(10.0, ball_delta, steps)
        passer_traj = (
            _linear_trajectory(12.0, passer_delta, steps) if passer_delta is not None else None
        )
        return analyze_forward_pass(ball_traj, passer_traj, _timestamps(steps), self.field_axis)

    def test_clearly_forward_pass(self):
        """Ball far outpaces passer -> forward decision."""
        result = self._run_decision(ball_delta=5.0, passer_delta=1.0)

        assert result.is_forward
        assert result.confidence > 0.7
        assert "FORWARD" in result.explanation

    def test_clearly_backward_pass(self):
        """Ball travels backward relative to passer."""
        result = self._run_decision(ball_delta=-2.0, passer_delta=1.0)

        assert not result.is_forward
        assert result.confidence > 0.7
        assert "NOT FORWARD" in result.explanation

    def test_borderline_forward_low_confidence(self):
        """Small margin over passer yields forward with modest confidence."""
        result = self._run_decision(ball_delta=1.5, passer_delta=1.0)

        assert result.is_forward
        assert 0.3 < result.confidence < 0.7

    def test_borderline_backward(self):
        """Ball moves slightly forward but passer keeps pace."""
        result = self._run_decision(ball_delta=0.8, passer_delta=1.0)

        assert not result.is_forward
        assert result.confidence < 0.6

    def test_no_passer_data(self):
        """Falls back to ball-only logic with reduced confidence."""
        result = self._run_decision(ball_delta=3.0, passer_delta=None)

        assert result.is_forward
        assert result.confidence < 0.7
        assert "Passer data unavailable" in result.explanation

    def test_invalid_axis_raises(self):
        """Zero axis vector should raise ValueError."""
        ball_traj = _linear_trajectory(10.0, 2.0, 4)
        timestamps = _timestamps(4)

        with pytest.raises(ValueError, match="Field axis vector is zero"):
            analyze_forward_pass(ball_traj, ball_traj, timestamps, np.array([0.0, 0.0, 0.0]))

    def test_mismatched_lengths_raise(self):
        """Trajectory/timestamp mismatch should error."""
        ball_traj = _linear_trajectory(10.0, 2.0, 4)
        passer_traj = _linear_trajectory(12.0, 1.0, 4)
        timestamps = _timestamps(3)

        with pytest.raises(ValueError, match="same length"):
            analyze_forward_pass(ball_traj, passer_traj, timestamps, self.field_axis)
