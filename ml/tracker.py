"""Object tracking module for maintaining track IDs across frames.

This module provides tracking capabilities to maintain consistent track IDs
for detected objects (players and ball) across video frames. Uses simplified
IOU-based tracking similar to ByteTrack.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from ml.detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Represents a tracked object across multiple frames.
    
    Attributes:
        track_id: Unique identifier for this track
        class_name: Object class ('player' or 'ball')
        detections: List of Detection objects in chronological order
        last_update_frame: Frame ID of most recent detection
        is_active: Whether this track is still actively tracked
    """
    
    track_id: int
    class_name: str
    detections: List[Detection] = field(default_factory=list)
    last_update_frame: int = 0
    is_active: bool = True
    
    def __post_init__(self) -> None:
        """Validate track data after initialization."""
        # Guard clause: validate class name
        if self.class_name not in ['player', 'ball']:
            raise ValueError(
                f"Invalid class_name '{self.class_name}'. "
                "Must be 'player' or 'ball'."
            )
    
    def add_detection(self, detection: Detection) -> None:
        """Add a detection to this track.
        
        Args:
            detection: Detection object to add
        """
        # Guard clause: validate detection class matches track class
        if detection.class_name != self.class_name:
            logger.warning(
                f"Detection class '{detection.class_name}' does not match "
                f"track class '{self.class_name}'"
            )
            return
        
        self.detections.append(detection)
        self.last_update_frame = detection.frame_id
        self.is_active = True
    
    @property
    def length(self) -> int:
        """Get the number of detections in this track.
        
        Returns:
            Number of detections
        """
        return len(self.detections)
    
    @property
    def latest_detection(self) -> Optional[Detection]:
        """Get the most recent detection in this track.
        
        Returns:
            Latest Detection or None if track is empty
        """
        if not self.detections:
            return None
        return self.detections[-1]
    
    @property
    def latest_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """Get the bounding box of the latest detection.
        
        Returns:
            Bounding box (x, y, w, h) or None if no detections
        """
        latest = self.latest_detection
        if latest is None:
            return None
        return latest.bbox


class Tracker:
    """Object tracker using IOU-based association.
    
    Maintains consistent track IDs for objects across frames using
    Intersection over Union (IOU) for matching detections to existing tracks.
    
    This is a simplified tracking approach suitable for Phase 4. For production,
    consider more sophisticated methods like DeepSORT or ByteTrack.
    
    Usage:
        >>> tracker = Tracker()
        >>> detections = detector.detect(frame, "cam1", 0)
        >>> tracks = tracker.update(detections)
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
    ) -> None:
        """Initialize the tracker.
        
        Args:
            iou_threshold: Minimum IOU for matching detection to track
            max_age: Maximum frames a track can go without updates before deletion
            min_hits: Minimum detections before a track is considered confirmed
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        
        self.tracks: Dict[int, Track] = {}
        self.next_track_id: int = 1
        self.frame_count: int = 0
        
        logger.info(
            f"Tracker initialized: iou_threshold={iou_threshold}, "
            f"max_age={max_age}, min_hits={min_hits}"
        )
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracks with new detections.
        
        Args:
            detections: List of Detection objects from current frame
            
        Returns:
            List of active Track objects (including new and existing)
        """
        self.frame_count += 1
        
        # Guard clause: no detections
        if not detections:
            self._age_tracks()
            return self.get_active_tracks()
        
        # Match detections to existing tracks
        matched_tracks, unmatched_detections = self._match_detections(
            detections
        )
        
        # Update matched tracks
        for track_id, detection in matched_tracks:
            self.tracks[track_id].add_detection(detection)
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            self._create_track(detection)
        
        # Age out old tracks
        self._age_tracks()
        
        return self.get_active_tracks()
    
    def _match_detections(
        self,
        detections: List[Detection],
    ) -> Tuple[List[Tuple[int, Detection]], List[Detection]]:
        """Match detections to existing tracks using IOU.
        
        Args:
            detections: List of Detection objects to match
            
        Returns:
            Tuple of (matched_pairs, unmatched_detections)
            - matched_pairs: List of (track_id, detection) tuples
            - unmatched_detections: Detections that didn't match any track
        """
        matched_pairs: List[Tuple[int, Detection]] = []
        unmatched_detections: List[Detection] = []
        
        # Guard clause: no active tracks
        active_tracks = self.get_active_tracks()
        if not active_tracks:
            return matched_pairs, detections
        
        # Create a copy of detections to track which are unmatched
        remaining_detections = list(detections)
        
        # For each detection, find best matching track
        for detection in detections:
            best_track_id: Optional[int] = None
            best_iou: float = self.iou_threshold
            
            # Find best matching track
            for track in active_tracks:
                # Only match same class
                if track.class_name != detection.class_name:
                    continue
                
                # Skip if track already matched in this frame
                if any(tid == track.track_id for tid, _ in matched_pairs):
                    continue
                
                # Compute IOU
                track_bbox = track.latest_bbox
                if track_bbox is None:
                    continue
                
                iou = self._compute_iou(detection.bbox, track_bbox)
                
                # Update best match if better IOU
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track.track_id
            
            # Add to matched pairs or unmatched detections
            if best_track_id is not None:
                matched_pairs.append((best_track_id, detection))
                remaining_detections.remove(detection)
        
        unmatched_detections = remaining_detections
        return matched_pairs, unmatched_detections
    
    def _compute_iou(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float],
    ) -> float:
        """Compute Intersection over Union between two bounding boxes.
        
        Args:
            bbox1: First bounding box (x, y, w, h)
            bbox2: Second bounding box (x, y, w, h)
            
        Returns:
            IOU score between 0 and 1
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to (x1, y1, x2, y2) format
        box1_x2 = x1 + w1
        box1_y2 = y1 + h1
        box2_x2 = x2 + w2
        box2_y2 = y2 + h2
        
        # Compute intersection area
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)
        
        # Guard clause: no intersection
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # Compute union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        # Guard clause: avoid division by zero
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
    
    def _create_track(self, detection: Detection) -> None:
        """Create a new track from a detection.
        
        Args:
            detection: Detection object to start tracking
        """
        track = Track(
            track_id=self.next_track_id,
            class_name=detection.class_name,
            detections=[detection],
            last_update_frame=detection.frame_id,
            is_active=True,
        )
        
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
        
        logger.debug(
            f"Created new track {track.track_id} for {detection.class_name}"
        )
    
    def _age_tracks(self) -> None:
        """Age out tracks that haven't been updated recently."""
        for track_id, track in list(self.tracks.items()):
            # Skip if track was updated this frame
            if track.last_update_frame == self.frame_count:
                continue
            
            # Calculate age (frames since last update)
            age = self.frame_count - track.last_update_frame
            
            # Deactivate old tracks
            if age > self.max_age:
                track.is_active = False
                logger.debug(
                    f"Deactivated track {track_id} after {age} frames"
                )
    
    def get_active_tracks(self) -> List[Track]:
        """Get all currently active tracks.
        
        Returns:
            List of active Track objects
        """
        return [
            track
            for track in self.tracks.values()
            if track.is_active
        ]
    
    def get_confirmed_tracks(self) -> List[Track]:
        """Get confirmed tracks (with minimum number of detections).
        
        Returns:
            List of confirmed active Track objects
        """
        return [
            track
            for track in self.get_active_tracks()
            if track.length >= self.min_hits
        ]
    
    def reset(self) -> None:
        """Reset tracker state (clear all tracks)."""
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_count = 0
        logger.info("Tracker reset")
