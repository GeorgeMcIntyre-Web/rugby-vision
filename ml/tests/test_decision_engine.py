"""Tests for pass event detection heuristics."""

from __future__ import annotations

from ml.decision_engine import detect_pass_events


FRAME_INTERVAL = 1.0 / 30.0


def test_detect_pass_events_finds_start_and_end():
    """Ball accelerates, then stabilizes in receiver's hands."""
    trajectory = [
        (0.0, 0.0, 1.0),
        (0.01, 0.0, 1.0),
        (0.02, 0.0, 1.0),
        (0.04, 0.0, 1.0),
        (0.5, 0.0, 1.0),
        (1.1, 0.0, 1.0),
        (2.1, 0.0, 1.0),
        (3.1, 0.0, 1.0),
        (4.0, 0.0, 1.0),
        (4.05, 0.0, 1.0),
        (4.07, 0.0, 1.0),
        (4.08, 0.0, 1.0),
        (4.081, 0.0, 1.0),
    ]

    start, end = detect_pass_events(trajectory, frame_interval_s=FRAME_INTERVAL)

    assert start == 4
    assert end == 11


def test_detect_pass_events_returns_last_frame_if_not_caught():
    """When the ball never slows down, end should be the last observation."""
    trajectory = [
        (0.0, 0.0, 1.0),
        (0.02, 0.0, 1.0),
        (0.04, 0.0, 1.0),
        (0.5, 0.0, 1.0),
        (1.3, 0.0, 1.0),
        (2.4, 0.0, 1.0),
        (3.8, 0.0, 1.0),
        (5.3, 0.0, 1.0),
    ]

    start, end = detect_pass_events(trajectory, frame_interval_s=FRAME_INTERVAL)

    assert start == 4
    assert end == len(trajectory) - 1


def test_detect_pass_events_requires_significant_motion():
    """Slow ball movement should not be considered a pass."""
    trajectory = [
        (0.0, 0.0, 1.0),
        (0.02, 0.0, 1.0),
        (0.04, 0.0, 1.0),
        (0.05, 0.0, 1.0),
        (0.07, 0.0, 1.0),
        (0.08, 0.0, 1.0),
    ]

    assert detect_pass_events(trajectory, frame_interval_s=FRAME_INTERVAL) == (-1, -1)


def test_detect_pass_events_ignores_short_stabilization_before_second_touch():
    """Short bobbles should not prematurely mark the pass end."""
    trajectory = [
        (0.0, 0.0, 1.0),
        (0.01, 0.0, 1.0),
        (0.02, 0.0, 1.0),
        (0.03, 0.0, 1.0),
        (0.6, 0.0, 1.0),
        (1.6, 0.0, 1.0),
        (2.6, 0.0, 1.0),
        (3.6, 0.0, 1.0),
        (3.65, 0.0, 1.0),
        (3.67, 0.0, 1.0),
        (4.8, 0.0, 1.0),
        (5.8, 0.0, 1.0),
        (5.85, 0.0, 1.0),
        (5.87, 0.0, 1.0),
        (5.88, 0.0, 1.0),
        (5.881, 0.0, 1.0),
    ]

    start, end = detect_pass_events(trajectory, frame_interval_s=FRAME_INTERVAL)

    assert start == 4
    assert end == 15


def test_detect_pass_events_handles_missing_frames():
    """None values should be skipped but still yield valid detection."""
    trajectory = [
        (0.0, 0.0, 1.0),
        None,
        (0.02, 0.0, 1.0),
        (0.03, 0.0, 1.0),
        (0.8, 0.0, 1.0),
        (1.6, 0.0, 1.0),
        (2.6, 0.0, 1.0),
        (3.6, 0.0, 1.0),
        (4.0, 0.0, 1.0),
        (4.05, 0.0, 1.0),
        (4.07, 0.0, 1.0),
        (4.08, 0.0, 1.0),
    ]

    start, end = detect_pass_events(trajectory, frame_interval_s=FRAME_INTERVAL)

    assert start == 4
    assert end == 10
