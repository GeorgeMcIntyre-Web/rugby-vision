"""Spatial modeling for rugby tracking: frame states with 3D positions.

Orchestrates the full pipeline from 2D detections/tracks to 3D field coordinates.
Builds time series of FrameState objects containing ball and player positions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ml.calibration import CameraCalibration
from ml.detection_tracking_api import DetectionTrackingResult
from ml.field_coords import FieldModel, transform_to_field_coords
from ml.triangulation import triangulate_point, Observation


@dataclass
class FrameState:
    """3D spatial state for a single frame.
    
    Attributes:
        timestamp: Frame timestamp in seconds
        frame_number: Frame index in video sequence
        ball_pos_3d: Ball position in field coordinates [X, Y, Z] or None
        players_pos_3d: Dictionary mapping track_id to 3D position in field coords
        ball_detected: Whether ball was detected in this frame
        n_players_tracked: Number of players successfully tracked in 3D
    """
    timestamp: float
    frame_number: int
    ball_pos_3d: Optional[np.ndarray] = None
    players_pos_3d: Dict[int, np.ndarray] = field(default_factory=dict)
    ball_detected: bool = False
    n_players_tracked: int = 0
    
    def __post_init__(self) -> None:
        """Update derived fields."""
        self.n_players_tracked = len(self.players_pos_3d)
        self.ball_detected = self.ball_pos_3d is not None


def build_frame_states(
    tracking_results: Dict[str, DetectionTrackingResult],
    calibrations: Dict[str, CameraCalibration],
    field_model: FieldModel,
    fps: float = 30.0
) -> List[FrameState]:
    """Build 3D frame states from multi-camera tracking results.
    
    Args:
        tracking_results: Dictionary mapping camera_id to DetectionTrackingResult
        calibrations: Dictionary mapping camera_id to CameraCalibration
        field_model: Rugby field model for coordinate transformation
        fps: Frames per second for timestamp calculation
    
    Returns:
        List of FrameState objects, one per frame
    
    Raises:
        ValueError: If inputs are invalid
    """
    if len(tracking_results) == 0:
        raise ValueError("No tracking results provided")
    
    if len(calibrations) < 2:
        raise ValueError(f"Need at least 2 calibrated cameras, got {len(calibrations)}")
    
    # Determine frame range (assume all cameras cover same frames)
    first_camera = list(tracking_results.keys())[0]
    n_frames = len(tracking_results[first_camera].detections_by_frame)
    
    frame_states = []
    
    for frame_idx in range(n_frames):
        timestamp = frame_idx / fps
        
        # Reconstruct ball position
        ball_pos_3d = _reconstruct_ball_position(
            frame_idx, tracking_results, calibrations, field_model
        )
        
        # Reconstruct player positions
        players_pos_3d = _reconstruct_player_positions(
            frame_idx, tracking_results, calibrations, field_model
        )
        
        frame_state = FrameState(
            timestamp=timestamp,
            frame_number=frame_idx,
            ball_pos_3d=ball_pos_3d,
            players_pos_3d=players_pos_3d
        )
        
        frame_states.append(frame_state)
    
    return frame_states


def _reconstruct_ball_position(
    frame_idx: int,
    tracking_results: Dict[str, DetectionTrackingResult],
    calibrations: Dict[str, CameraCalibration],
    field_model: FieldModel
) -> Optional[np.ndarray]:
    """Reconstruct 3D ball position for a single frame.
    
    Args:
        frame_idx: Frame index
        tracking_results: Multi-camera tracking results
        calibrations: Camera calibrations
        field_model: Field coordinate system
    
    Returns:
        3D ball position in field coordinates or None
    """
    observations = []
    
    for camera_id, result in tracking_results.items():
        if camera_id not in calibrations:
            continue
        
        if frame_idx not in result.detections_by_frame:
            continue
        
        detections = result.detections_by_frame[frame_idx]
        ball_detections = [d for d in detections if d.class_name == 'ball']
        
        if len(ball_detections) == 0:
            continue
        
        # Use highest confidence ball detection
        ball = max(ball_detections, key=lambda d: d.confidence)
        
        # Get ball center in pixel coordinates
        x_center = (ball.x1 + ball.x2) / 2
        y_center = (ball.y1 + ball.y2) / 2
        
        observations.append((calibrations[camera_id], (x_center, y_center)))
    
    if len(observations) < 2:
        return None
    
    # Triangulate to get world coordinates
    point_3d_world = triangulate_point(observations)
    
    if point_3d_world is None:
        return None
    
    # Transform to field coordinates
    ball_pos_field = transform_to_field_coords(point_3d_world, field_model)
    
    return ball_pos_field


def _reconstruct_player_positions(
    frame_idx: int,
    tracking_results: Dict[str, DetectionTrackingResult],
    calibrations: Dict[str, CameraCalibration],
    field_model: FieldModel
) -> Dict[int, np.ndarray]:
    """Reconstruct 3D positions for all tracked players in a frame.
    
    Args:
        frame_idx: Frame index
        tracking_results: Multi-camera tracking results
        calibrations: Camera calibrations
        field_model: Field coordinate system
    
    Returns:
        Dictionary mapping track_id to 3D position in field coordinates
    """
    # Collect all track IDs across cameras for this frame
    all_track_ids = set()
    for result in tracking_results.values():
        if frame_idx in result.tracks_by_frame:
            track_ids = [t.track_id for t in result.tracks_by_frame[frame_idx]
                        if t.class_name == 'player']
            all_track_ids.update(track_ids)
    
    players_pos_3d = {}
    
    for track_id in all_track_ids:
        observations = []
        
        for camera_id, result in tracking_results.items():
            if camera_id not in calibrations:
                continue
            
            if frame_idx not in result.tracks_by_frame:
                continue
            
            # Find this track in this camera
            tracks = [t for t in result.tracks_by_frame[frame_idx]
                     if t.track_id == track_id and t.class_name == 'player']
            
            if len(tracks) == 0:
                continue
            
            track = tracks[0]
            last_detection = track.detections[-1]
            
            # Use bottom-center of bounding box (feet position)
            x_center = (last_detection.x1 + last_detection.x2) / 2
            y_bottom = last_detection.y2
            
            observations.append((calibrations[camera_id], (x_center, y_bottom)))
        
        if len(observations) < 2:
            continue
        
        # Triangulate
        point_3d_world = triangulate_point(observations)
        
        if point_3d_world is None:
            continue
        
        # Transform to field coordinates
        player_pos_field = transform_to_field_coords(point_3d_world, field_model)
        
        if player_pos_field is not None:
            players_pos_3d[track_id] = player_pos_field
    
    return players_pos_3d


def get_frame_state_summary(frame_states: List[FrameState]) -> Dict:
    """Get summary statistics for a sequence of frame states.
    
    Args:
        frame_states: List of FrameState objects
    
    Returns:
        Dictionary with summary statistics
    """
    if len(frame_states) == 0:
        return {
            'n_frames': 0,
            'ball_detection_rate': 0.0,
            'avg_players_tracked': 0.0,
            'min_players_tracked': 0,
            'max_players_tracked': 0
        }
    
    n_frames_with_ball = sum(1 for fs in frame_states if fs.ball_detected)
    player_counts = [fs.n_players_tracked for fs in frame_states]
    
    return {
        'n_frames': len(frame_states),
        'ball_detection_rate': n_frames_with_ball / len(frame_states),
        'avg_players_tracked': np.mean(player_counts),
        'min_players_tracked': min(player_counts),
        'max_players_tracked': max(player_counts)
    }
