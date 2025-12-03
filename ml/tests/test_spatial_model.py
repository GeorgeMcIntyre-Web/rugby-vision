"""Tests for spatial modeling and frame state generation."""

import pytest
import numpy as np
from typing import Dict, List

from ml.calibration import CameraCalibration
from ml.detection_tracking_api import (
    DetectionTrackingResult,
    ClipDefinition,
)
from ml.detector import Detection
from ml.tracker import Track
from ml.field_coords import FieldModel, get_standard_rugby_field
from ml.spatial_model import (
    FrameState,
    build_frame_states,
    get_frame_state_summary,
)


def create_test_calibration(camera_id: str, translation: np.ndarray) -> CameraCalibration:
    """Create a test camera calibration."""
    intrinsic = np.array([
        [1500.0, 0.0, 960.0],
        [0.0, 1500.0, 540.0],
        [0.0, 0.0, 1.0]
    ])
    
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = translation
    
    return CameraCalibration(
        camera_id=camera_id,
        intrinsic=intrinsic,
        extrinsic=extrinsic
    )


class TestFrameState:
    """Tests for FrameState dataclass."""
    
    def test_empty_frame_state(self):
        """Test creating empty frame state."""
        frame_state = FrameState(
            timestamp=0.0,
            frame_number=0
        )
        
        assert frame_state.timestamp == 0.0
        assert frame_state.frame_number == 0
        assert frame_state.ball_pos_3d is None
        assert len(frame_state.players_pos_3d) == 0
        assert not frame_state.ball_detected
        assert frame_state.n_players_tracked == 0
    
    def test_frame_state_with_ball(self):
        """Test frame state with ball position."""
        ball_pos = np.array([35.0, 50.0, 1.5])
        
        frame_state = FrameState(
            timestamp=1.0,
            frame_number=30,
            ball_pos_3d=ball_pos
        )
        
        assert frame_state.ball_detected
        assert np.array_equal(frame_state.ball_pos_3d, ball_pos)
    
    def test_frame_state_with_players(self):
        """Test frame state with player positions."""
        players = {
            1: np.array([10.0, 20.0, 0.0]),
            2: np.array([15.0, 25.0, 0.0]),
            3: np.array([20.0, 30.0, 0.0]),
        }
        
        frame_state = FrameState(
            timestamp=2.0,
            frame_number=60,
            players_pos_3d=players
        )
        
        assert frame_state.n_players_tracked == 3
        assert 1 in frame_state.players_pos_3d
        assert 2 in frame_state.players_pos_3d
        assert 3 in frame_state.players_pos_3d


class TestBuildFrameStates:
    """Tests for building frame states from tracking results."""
    
    def test_empty_tracking_results(self):
        """Test that empty tracking results raises error."""
        calibrations = {
            "cam_0": create_test_calibration("cam_0", np.array([0, 0, 0])),
            "cam_1": create_test_calibration("cam_1", np.array([2, 0, 0])),
        }
        field = get_standard_rugby_field()
        
        with pytest.raises(ValueError, match="No tracking results provided"):
            build_frame_states({}, calibrations, field)
    
    def test_insufficient_calibrations(self):
        """Test that insufficient calibrations raises error."""
        # Create mock tracking result
        result = DetectionTrackingResult(
            clip_id="test",
            frame_count=1,
            detections_per_camera={"cam_0": []},
            tracks_per_camera={"cam_0": []},
            detections_by_frame={},
            tracks_by_frame={}
        )
        
        tracking_results = {"cam_0": result}
        
        # Only 1 calibration
        calibrations = {
            "cam_0": create_test_calibration("cam_0", np.array([0, 0, 0])),
        }
        
        field = get_standard_rugby_field()
        
        with pytest.raises(ValueError, match="Need at least 2 calibrated cameras"):
            build_frame_states(tracking_results, calibrations, field)
    
    def test_build_frame_states_no_detections(self):
        """Test building frame states with no detections."""
        # Create mock tracking results with empty detections
        result = DetectionTrackingResult(
            clip_id="test",
            frame_count=5,
            detections_per_camera={"cam_0": [], "cam_1": []},
            tracks_per_camera={"cam_0": [], "cam_1": []},
            detections_by_frame={i: [] for i in range(5)},
            tracks_by_frame={i: [] for i in range(5)}
        )
        
        tracking_results = {"cam_0": result, "cam_1": result}
        
        calibrations = {
            "cam_0": create_test_calibration("cam_0", np.array([0, 0, 0])),
            "cam_1": create_test_calibration("cam_1", np.array([2, 0, 0])),
        }
        
        field = get_standard_rugby_field()
        
        frame_states = build_frame_states(tracking_results, calibrations, field, fps=30.0)
        
        assert len(frame_states) == 5
        
        # All frames should have no detections
        for fs in frame_states:
            assert not fs.ball_detected
            assert fs.n_players_tracked == 0
    
    def test_frame_state_timestamps(self):
        """Test that frame state timestamps are correct."""
        result = DetectionTrackingResult(
            clip_id="test",
            frame_count=10,
            detections_per_camera={"cam_0": [], "cam_1": []},
            tracks_per_camera={"cam_0": [], "cam_1": []},
            detections_by_frame={i: [] for i in range(10)},
            tracks_by_frame={i: [] for i in range(10)}
        )
        
        tracking_results = {"cam_0": result, "cam_1": result}
        
        calibrations = {
            "cam_0": create_test_calibration("cam_0", np.array([0, 0, 0])),
            "cam_1": create_test_calibration("cam_1", np.array([2, 0, 0])),
        }
        
        field = get_standard_rugby_field()
        fps = 25.0
        
        frame_states = build_frame_states(tracking_results, calibrations, field, fps=fps)
        
        assert len(frame_states) == 10
        
        for i, fs in enumerate(frame_states):
            expected_timestamp = i / fps
            assert np.isclose(fs.timestamp, expected_timestamp)
            assert fs.frame_number == i


class TestGetFrameStateSummary:
    """Tests for frame state summary statistics."""
    
    def test_empty_frame_states(self):
        """Test summary for empty list."""
        summary = get_frame_state_summary([])
        
        assert summary['n_frames'] == 0
        assert summary['ball_detection_rate'] == 0.0
        assert summary['avg_players_tracked'] == 0.0
    
    def test_summary_no_detections(self):
        """Test summary with no ball detections."""
        frame_states = [
            FrameState(timestamp=i*0.033, frame_number=i)
            for i in range(10)
        ]
        
        summary = get_frame_state_summary(frame_states)
        
        assert summary['n_frames'] == 10
        assert summary['ball_detection_rate'] == 0.0
        assert summary['avg_players_tracked'] == 0.0
        assert summary['min_players_tracked'] == 0
        assert summary['max_players_tracked'] == 0
    
    def test_summary_with_ball(self):
        """Test summary with some ball detections."""
        frame_states = []
        for i in range(10):
            ball_pos = np.array([35.0, 50.0, 1.0]) if i % 2 == 0 else None
            frame_states.append(FrameState(
                timestamp=i*0.033,
                frame_number=i,
                ball_pos_3d=ball_pos
            ))
        
        summary = get_frame_state_summary(frame_states)
        
        assert summary['n_frames'] == 10
        assert summary['ball_detection_rate'] == 0.5  # 50% detected
    
    def test_summary_with_players(self):
        """Test summary with varying player counts."""
        frame_states = []
        for i in range(10):
            n_players = i  # 0 to 9 players
            players = {
                j: np.array([10.0 + j, 20.0, 0.0])
                for j in range(n_players)
            }
            frame_states.append(FrameState(
                timestamp=i*0.033,
                frame_number=i,
                players_pos_3d=players
            ))
        
        summary = get_frame_state_summary(frame_states)
        
        assert summary['n_frames'] == 10
        assert summary['min_players_tracked'] == 0
        assert summary['max_players_tracked'] == 9
        assert summary['avg_players_tracked'] == 4.5  # Average of 0-9
