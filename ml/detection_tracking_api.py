"""Detection and tracking orchestration API.

This module provides a high-level API for running detection and tracking
on video clips. It coordinates the detector and tracker to process all frames
in a clip and return structured output.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from ml.detector import Detector, Detection
from ml.tracker import Tracker, Track

logger = logging.getLogger(__name__)


@dataclass
class ClipDefinition:
    """Definition of a video clip to process.
    
    Attributes:
        clip_id: Unique identifier for the clip
        camera_ids: List of camera identifiers
        frames_per_camera: Dict mapping camera_id to list of frames
        start_frame: Starting frame number
        end_frame: Ending frame number
    """
    
    clip_id: str
    camera_ids: List[str]
    frames_per_camera: Dict[str, List[np.ndarray]]
    start_frame: int = 0
    end_frame: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate clip definition after initialization."""
        # Guard clause: validate camera_ids
        if not self.camera_ids:
            raise ValueError("camera_ids cannot be empty")
        
        # Guard clause: validate frames_per_camera
        for camera_id in self.camera_ids:
            if camera_id not in self.frames_per_camera:
                raise ValueError(
                    f"Camera {camera_id} not found in frames_per_camera"
                )
        
        # Set end_frame if not specified
        if self.end_frame is None:
            # Use the minimum number of frames across all cameras
            frame_counts = [
                len(frames) for frames in self.frames_per_camera.values()
            ]
            self.end_frame = min(frame_counts) if frame_counts else 0


@dataclass
class DetectionTrackingResult:
    """Result of detection and tracking for a video clip.
    
    Attributes:
        clip_id: Unique identifier for the clip
        detections_per_camera: Dict mapping camera_id to list of detections
        tracks_per_camera: Dict mapping camera_id to list of tracks
        detections_by_frame: Dict mapping frame_idx to list of detections
        tracks_by_frame: Dict mapping frame_idx to list of tracks
        frame_count: Total number of frames processed
        detection_count: Total number of detections
        track_count: Total number of active tracks
        metadata: Additional metadata
    """
    
    clip_id: str
    detections_per_camera: Dict[str, List[Detection]] = field(
        default_factory=dict
    )
    tracks_per_camera: Dict[str, List[Track]] = field(default_factory=dict)
    detections_by_frame: Dict[int, List[Detection]] = field(
        default_factory=dict
    )
    tracks_by_frame: Dict[int, List[Track]] = field(
        default_factory=dict
    )
    frame_count: int = 0
    detection_count: int = 0
    track_count: int = 0
    metadata: Dict[str, any] = field(default_factory=dict)


def run_detection_and_tracking(
    clip_definition: ClipDefinition,
    detector: Optional[Detector] = None,
    tracker_per_camera: Optional[Dict[str, Tracker]] = None,
) -> DetectionTrackingResult:
    """Run detection and tracking on a video clip.
    
    Orchestrates detector and tracker for all frames in a clip across
    multiple cameras. Each camera gets its own tracker instance to maintain
    separate track IDs.
    
    Args:
        clip_definition: ClipDefinition with frames and camera information
        detector: Detector instance (creates default if None)
        tracker_per_camera: Dict of Tracker instances per camera (creates if None)
        
    Returns:
        DetectionTrackingResult with all detections and tracks
        
    Example:
        >>> clip = ClipDefinition(
        ...     clip_id="test_clip",
        ...     camera_ids=["cam1", "cam2"],
        ...     frames_per_camera={"cam1": [frame1, frame2], "cam2": [frame3, frame4]}
        ... )
        >>> result = run_detection_and_tracking(clip)
        >>> print(f"Processed {result.frame_count} frames")
    """
    logger.info(
        f"Starting detection and tracking for clip {clip_definition.clip_id}"
    )
    
    # Guard clause: validate clip definition
    if not clip_definition.camera_ids:
        logger.error("No cameras specified in clip definition")
        return DetectionTrackingResult(clip_id=clip_definition.clip_id)
    
    # Initialize detector if not provided
    if detector is None:
        detector = Detector(use_mock=True)
    
    # Initialize trackers per camera if not provided
    if tracker_per_camera is None:
        tracker_per_camera = {
            camera_id: Tracker()
            for camera_id in clip_definition.camera_ids
        }
    
    # Storage for results
    all_detections_per_camera: Dict[str, List[Detection]] = {
        camera_id: [] for camera_id in clip_definition.camera_ids
    }
    all_tracks_per_camera: Dict[str, List[Track]] = {}
    detections_by_frame: Dict[int, List[Detection]] = {}
    tracks_by_frame: Dict[int, List[Track]] = {}
    
    total_frames_processed = 0
    
    # Process each camera
    for camera_id in clip_definition.camera_ids:
        logger.info(f"Processing camera {camera_id}")
        
        # Guard clause: check if camera has frames
        if camera_id not in clip_definition.frames_per_camera:
            logger.warning(f"No frames for camera {camera_id}")
            continue
        
        frames = clip_definition.frames_per_camera[camera_id]
        
        # Guard clause: check frame range
        start_idx = max(0, clip_definition.start_frame)
        end_idx = min(
            len(frames),
            clip_definition.end_frame if clip_definition.end_frame else len(frames)
        )
        
        if start_idx >= end_idx:
            logger.warning(
                f"Invalid frame range for camera {camera_id}: "
                f"[{start_idx}, {end_idx})"
            )
            continue
        
        # Process frames for this camera
        tracker = tracker_per_camera[camera_id]
        
        for frame_idx in range(start_idx, end_idx):
            frame = frames[frame_idx]
            
            # Guard clause: skip invalid frames
            if frame is None or frame.size == 0:
                logger.warning(
                    f"Invalid frame at index {frame_idx} for camera {camera_id}"
                )
                continue
            
            # Run detection
            detections = detector.detect(frame, camera_id, frame_idx)
            all_detections_per_camera[camera_id].extend(detections)
            
            # Add to frame-indexed storage
            if frame_idx not in detections_by_frame:
                detections_by_frame[frame_idx] = []
            detections_by_frame[frame_idx].extend(detections)
            
            # Update tracker
            tracker.update(detections)
            
            # Add tracks to frame-indexed storage
            if frame_idx not in tracks_by_frame:
                tracks_by_frame[frame_idx] = []
            tracks_by_frame[frame_idx].extend(tracker.get_active_tracks())
            
            total_frames_processed += 1
        
        # Get final tracks for this camera
        final_tracks = tracker.get_active_tracks()
        all_tracks_per_camera[camera_id] = final_tracks
        
        logger.info(
            f"Camera {camera_id}: {len(all_detections_per_camera[camera_id])} "
            f"detections, {len(final_tracks)} tracks"
        )
    
    # Compute summary statistics
    total_detections = sum(
        len(dets) for dets in all_detections_per_camera.values()
    )
    total_tracks = sum(
        len(tracks) for tracks in all_tracks_per_camera.values()
    )
    
    # Build result
    result = DetectionTrackingResult(
        clip_id=clip_definition.clip_id,
        detections_per_camera=all_detections_per_camera,
        tracks_per_camera=all_tracks_per_camera,
        detections_by_frame=detections_by_frame,
        tracks_by_frame=tracks_by_frame,
        frame_count=total_frames_processed,
        detection_count=total_detections,
        track_count=total_tracks,
        metadata={
            "cameras": clip_definition.camera_ids,
            "frame_range": [clip_definition.start_frame, clip_definition.end_frame],
        },
    )
    
    logger.info(
        f"Detection and tracking complete for clip {clip_definition.clip_id}: "
        f"{total_frames_processed} frames, {total_detections} detections, "
        f"{total_tracks} tracks"
    )
    
    return result


def get_detections_summary(result: DetectionTrackingResult) -> Dict[str, any]:
    """Get a summary of detections from the result.
    
    Args:
        result: DetectionTrackingResult from run_detection_and_tracking
        
    Returns:
        Dict with summary statistics
    """
    # Count detections by class
    player_count = 0
    ball_count = 0
    
    for detections in result.detections_per_camera.values():
        for detection in detections:
            if detection.class_name == 'player':
                player_count += 1
            if detection.class_name == 'ball':
                ball_count += 1
    
    # Count tracks by class
    player_tracks = 0
    ball_tracks = 0
    
    for tracks in result.tracks_per_camera.values():
        for track in tracks:
            if track.class_name == 'player':
                player_tracks += 1
            if track.class_name == 'ball':
                ball_tracks += 1
    
    return {
        "clip_id": result.clip_id,
        "total_frames": result.frame_count,
        "total_detections": result.detection_count,
        "player_detections": player_count,
        "ball_detections": ball_count,
        "total_tracks": result.track_count,
        "player_tracks": player_tracks,
        "ball_tracks": ball_tracks,
        "cameras": len(result.detections_per_camera),
    }
