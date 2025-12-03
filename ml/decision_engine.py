"""Forward pass decision logic built on 3D trajectories."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

Point3D = np.ndarray
Vector3D = np.ndarray
TrajectorySample = Tuple[Point3D, float]

# Thresholds based on practical rugby officiating heuristics.
FORWARD_DISPLACEMENT_TOLERANCE_METERS = 0.3
MINIMUM_BALL_SAMPLES = 2
MINIMUM_BALL_DISPLACEMENT_METERS = 0.1


@dataclass
class DecisionResult:
    """Human-readable forward pass decision."""

    is_forward: bool
    confidence: float
    explanation: str
    metadata: Optional[Dict[str, Any]] = None


def analyze_forward_pass(
    ball_trajectory_3d: Sequence[Optional[Point3D]],
    passer_trajectory_3d: Optional[Sequence[Optional[Point3D]]],
    timestamps: Sequence[float],
    field_axis_forward: Vector3D
) -> DecisionResult:
    """Analyze whether the pass is forward using 3D trajectories.

    Args:
        ball_trajectory_3d: Ball positions aligned with timestamps (None when missing).
        passer_trajectory_3d: Passer positions aligned with timestamps or None.
        timestamps: Timebase shared by trajectories (seconds).
        field_axis_forward: Vector pointing in the forward direction of play.

    Returns:
        DecisionResult describing the call, confidence, and supporting metadata.

    Raises:
        ValueError: If sequence lengths mismatch or axis is invalid.
    """
    _validate_inputs(ball_trajectory_3d, passer_trajectory_3d, timestamps, field_axis_forward)

    axis_unit = _normalize_vector(field_axis_forward)
    ball_samples = _extract_valid_samples(ball_trajectory_3d, timestamps)

    if len(ball_samples) < MINIMUM_BALL_SAMPLES:
        explanation = "Insufficient ball trajectory data for decision."
        metadata = {
            "ball_sample_count": len(ball_samples),
            "required_ball_samples": MINIMUM_BALL_SAMPLES,
        }
        return DecisionResult(False, 0.0, explanation, metadata)

    passer_samples = _extract_valid_samples(passer_trajectory_3d, timestamps)
    passer_available = len(passer_samples) >= MINIMUM_BALL_SAMPLES

    ball_forward = _compute_forward_displacement(ball_samples, axis_unit)
    passer_forward = _compute_forward_displacement(passer_samples, axis_unit) if passer_available else 0.0
    displacement_margin = ball_forward - passer_forward

    if abs(ball_forward) < MINIMUM_BALL_DISPLACEMENT_METERS:
        confidence = _estimate_confidence(
            ball_forward,
            displacement_margin,
            len(ball_samples),
            len(passer_samples),
            passer_available,
        )
        explanation = (
            "Ball displacement below minimum threshold; "
            "unable to classify pass direction confidently."
        )
        metadata = _build_metadata(
            ball_samples,
            passer_samples,
            axis_unit,
            ball_forward,
            passer_forward,
            displacement_margin,
            passer_available,
        )
        return DecisionResult(False, confidence, explanation, metadata)

    is_forward = displacement_margin > FORWARD_DISPLACEMENT_TOLERANCE_METERS
    confidence = _estimate_confidence(
        ball_forward,
        displacement_margin,
        len(ball_samples),
        len(passer_samples),
        passer_available,
    )

    explanation = _build_explanation(
        is_forward,
        ball_forward,
        passer_forward,
        displacement_margin,
        passer_available,
    )

    metadata = _build_metadata(
        ball_samples,
        passer_samples,
        axis_unit,
        ball_forward,
        passer_forward,
        displacement_margin,
        passer_available,
    )

    return DecisionResult(is_forward, confidence, explanation, metadata)


def _validate_inputs(
    ball_trajectory: Sequence[Optional[Point3D]],
    passer_trajectory: Optional[Sequence[Optional[Point3D]]],
    timestamps: Sequence[float],
    axis: Vector3D
) -> None:
    """Validate lengths and axis integrity."""
    if len(ball_trajectory) != len(timestamps):
        raise ValueError("Ball trajectory and timestamps must be the same length.")

    if passer_trajectory is not None and len(passer_trajectory) != len(timestamps):
        raise ValueError("Passer trajectory must align with timestamps when provided.")

    if np.linalg.norm(axis) < 1e-6:
        raise ValueError("Field axis vector is zero; cannot determine forward direction.")


def _extract_valid_samples(
    trajectory: Optional[Sequence[Optional[Point3D]]],
    timestamps: Sequence[float]
) -> List[TrajectorySample]:
    """Collect trajectory samples with valid 3D positions."""
    if trajectory is None:
        return []

    samples: List[TrajectorySample] = []
    for point, timestamp in zip(trajectory, timestamps):
        if point is None:
            continue

        samples.append((np.asarray(point, dtype=float), float(timestamp)))

    return samples


def _normalize_vector(axis: Vector3D) -> Vector3D:
    """Normalize axis vector."""
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        raise ValueError("Cannot normalize near-zero axis vector.")
    return axis / norm


def _compute_forward_displacement(
    samples: Sequence[TrajectorySample],
    axis_unit: Vector3D
) -> float:
    """Project net displacement onto field axis."""
    if len(samples) < MINIMUM_BALL_SAMPLES:
        return 0.0

    start = samples[0][0]
    end = samples[-1][0]
    displacement_vector = end - start
    return float(np.dot(displacement_vector, axis_unit))


def _compute_duration(samples: Sequence[TrajectorySample]) -> float:
    """Compute duration covered by trajectory samples."""
    if len(samples) < MINIMUM_BALL_SAMPLES:
        return 0.0
    return float(samples[-1][1] - samples[0][1])


def _compute_velocity(displacement: float, duration: float) -> float:
    """Compute velocity magnitude along field axis."""
    if duration <= 0.0:
        return 0.0
    return displacement / duration


def _estimate_confidence(
    ball_forward: float,
    displacement_margin: float,
    ball_sample_count: int,
    passer_sample_count: int,
    passer_available: bool
) -> float:
    """Estimate confidence using displacement margin and data quality."""
    if ball_sample_count == 0:
        return 0.0

    displacement_factor = min(abs(ball_forward) / 5.0, 1.0) * 0.25
    margin_ratio = min(abs(displacement_margin) / 3.0, 1.0)
    margin_weight = 0.45 if passer_available else 0.2
    margin_factor = margin_ratio * margin_weight
    sample_factor = min(ball_sample_count, 10) / 10 * 0.15
    passer_factor = (
        min(passer_sample_count, 10) / 10 * 0.1 if passer_available else 0.0
    )
    availability_bonus = 0.1 if passer_available else -0.1

    confidence = 0.1 + displacement_factor + margin_factor + sample_factor + passer_factor + availability_bonus
    return float(np.clip(confidence, 0.0, 1.0))


def _build_metadata(
    ball_samples: Sequence[TrajectorySample],
    passer_samples: Sequence[TrajectorySample],
    axis_unit: Vector3D,
    ball_forward: float,
    passer_forward: float,
    displacement_margin: float,
    passer_available: bool
) -> Dict[str, Any]:
    """Assemble debug metadata for downstream consumers."""
    ball_duration = _compute_duration(ball_samples)
    passer_duration = _compute_duration(passer_samples) if passer_available else 0.0

    return {
        "ball_forward_displacement_m": ball_forward,
        "passer_forward_displacement_m": passer_forward if passer_available else None,
        "displacement_margin_m": displacement_margin,
        "tolerance_m": FORWARD_DISPLACEMENT_TOLERANCE_METERS,
        "ball_velocity_m_per_s": _compute_velocity(ball_forward, ball_duration),
        "passer_velocity_m_per_s": (
            _compute_velocity(passer_forward, passer_duration) if passer_available else None
        ),
        "ball_sample_count": len(ball_samples),
        "passer_sample_count": len(passer_samples),
        "passer_data_available": passer_available,
        "axis_unit": axis_unit.tolist(),
    }


def _build_explanation(
    is_forward: bool,
    ball_forward: float,
    passer_forward: float,
    displacement_margin: float,
    passer_available: bool
) -> str:
    """Create human-readable explanation string."""
    decision_text = "FORWARD" if is_forward else "NOT FORWARD"
    base = (
        f"{decision_text}: Ball moved {ball_forward:.2f} m "
        f"vs passer {passer_forward:.2f} m along field axis."
    )

    margin_text = (
        f" Margin {displacement_margin:.2f} m compared to tolerance "
        f"{FORWARD_DISPLACEMENT_TOLERANCE_METERS:.2f} m."
    )

    if not passer_available:
        margin_text += " Passer data unavailable; used ball displacement only."

    return base + margin_text
