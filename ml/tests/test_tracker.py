"""Unit tests for tracker module."""

import pytest
from ml.detector import Detection
from ml.tracker import Track, Tracker


class TestTrack:
    """Tests for Track dataclass."""
    
    def test_track_creation_valid(self):
        """Test creating a valid Track."""
        track = Track(
            track_id=1,
            class_name="player",
            detections=[],
            last_update_frame=0,
            is_active=True,
        )
        
        assert track.track_id == 1
        assert track.class_name == "player"
        assert track.detections == []
        assert track.is_active is True
    
    def test_track_invalid_class(self):
        """Test Track with invalid class_name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid class_name"):
            Track(
                track_id=1,
                class_name="invalid",
            )
    
    def test_track_add_detection(self):
        """Test adding detection to track."""
        track = Track(track_id=1, class_name="player")
        detection = Detection(
            camera_id="cam1",
            frame_id=5,
            bbox=(10.0, 20.0, 50.0, 100.0),
            class_name="player",
            confidence=0.8,
        )
        
        track.add_detection(detection)
        
        assert len(track.detections) == 1
        assert track.detections[0] == detection
        assert track.last_update_frame == 5
    
    def test_track_add_detection_wrong_class(self):
        """Test adding detection with wrong class doesn't add."""
        track = Track(track_id=1, class_name="player")
        detection = Detection(
            camera_id="cam1",
            frame_id=5,
            bbox=(10.0, 20.0, 50.0, 100.0),
            class_name="ball",  # Different class
            confidence=0.8,
        )
        
        track.add_detection(detection)
        
        # Should not add detection
        assert len(track.detections) == 0
    
    def test_track_length_property(self):
        """Test Track length property."""
        track = Track(track_id=1, class_name="player")
        
        assert track.length == 0
        
        detection1 = Detection(
            camera_id="cam1", frame_id=0,
            bbox=(0.0, 0.0, 10.0, 10.0),
            class_name="player", confidence=0.8
        )
        detection2 = Detection(
            camera_id="cam1", frame_id=1,
            bbox=(10.0, 10.0, 10.0, 10.0),
            class_name="player", confidence=0.8
        )
        
        track.add_detection(detection1)
        assert track.length == 1
        
        track.add_detection(detection2)
        assert track.length == 2
    
    def test_track_latest_detection_property(self):
        """Test Track latest_detection property."""
        track = Track(track_id=1, class_name="player")
        
        assert track.latest_detection is None
        
        detection1 = Detection(
            camera_id="cam1", frame_id=0,
            bbox=(0.0, 0.0, 10.0, 10.0),
            class_name="player", confidence=0.8
        )
        detection2 = Detection(
            camera_id="cam1", frame_id=1,
            bbox=(10.0, 10.0, 10.0, 10.0),
            class_name="player", confidence=0.8
        )
        
        track.add_detection(detection1)
        assert track.latest_detection == detection1
        
        track.add_detection(detection2)
        assert track.latest_detection == detection2
    
    def test_track_latest_bbox_property(self):
        """Test Track latest_bbox property."""
        track = Track(track_id=1, class_name="player")
        
        assert track.latest_bbox is None
        
        detection = Detection(
            camera_id="cam1", frame_id=0,
            bbox=(5.0, 10.0, 20.0, 30.0),
            class_name="player", confidence=0.8
        )
        
        track.add_detection(detection)
        assert track.latest_bbox == (5.0, 10.0, 20.0, 30.0)


class TestTracker:
    """Tests for Tracker class."""
    
    def test_tracker_initialization(self):
        """Test Tracker initialization."""
        tracker = Tracker(
            iou_threshold=0.3,
            max_age=30,
            min_hits=3,
        )
        
        assert tracker.iou_threshold == 0.3
        assert tracker.max_age == 30
        assert tracker.min_hits == 3
        assert len(tracker.tracks) == 0
        assert tracker.next_track_id == 1
    
    def test_tracker_update_creates_new_tracks(self):
        """Test tracker creates new tracks for first detections."""
        tracker = Tracker()
        detections = [
            Detection(
                camera_id="cam1", frame_id=0,
                bbox=(10.0, 10.0, 20.0, 30.0),
                class_name="player", confidence=0.8
            ),
            Detection(
                camera_id="cam1", frame_id=0,
                bbox=(50.0, 50.0, 20.0, 30.0),
                class_name="player", confidence=0.8
            ),
        ]
        
        tracks = tracker.update(detections)
        
        assert len(tracks) == 2
        assert all(track.length == 1 for track in tracks)
    
    def test_tracker_update_empty_detections(self):
        """Test tracker update with empty detections list."""
        tracker = Tracker()
        
        tracks = tracker.update([])
        
        assert tracks == []
    
    def test_tracker_matches_detections_to_tracks(self):
        """Test tracker matches similar detections to existing tracks."""
        tracker = Tracker(iou_threshold=0.3)
        
        # Frame 0: Create initial track
        detections_frame0 = [
            Detection(
                camera_id="cam1", frame_id=0,
                bbox=(10.0, 10.0, 20.0, 30.0),
                class_name="player", confidence=0.8
            ),
        ]
        tracks = tracker.update(detections_frame0)
        assert len(tracks) == 1
        initial_track_id = tracks[0].track_id
        
        # Frame 1: Similar detection (should match)
        detections_frame1 = [
            Detection(
                camera_id="cam1", frame_id=1,
                bbox=(15.0, 15.0, 20.0, 30.0),  # Slightly moved
                class_name="player", confidence=0.8
            ),
        ]
        tracks = tracker.update(detections_frame1)
        
        assert len(tracks) == 1
        assert tracks[0].track_id == initial_track_id
        assert tracks[0].length == 2
    
    def test_tracker_creates_new_track_for_distant_detection(self):
        """Test tracker creates new track for distant detection."""
        tracker = Tracker(iou_threshold=0.3)
        
        # Frame 0
        detections_frame0 = [
            Detection(
                camera_id="cam1", frame_id=0,
                bbox=(10.0, 10.0, 20.0, 30.0),
                class_name="player", confidence=0.8
            ),
        ]
        tracker.update(detections_frame0)
        
        # Frame 1: Far away detection (should create new track)
        detections_frame1 = [
            Detection(
                camera_id="cam1", frame_id=1,
                bbox=(200.0, 200.0, 20.0, 30.0),  # Far away
                class_name="player", confidence=0.8
            ),
        ]
        tracks = tracker.update(detections_frame1)
        
        # Should have 2 tracks now (1 old, 1 new)
        assert len(tracker.tracks) == 2
    
    def test_tracker_ages_out_old_tracks(self):
        """Test tracker deactivates tracks not updated for max_age frames."""
        tracker = Tracker(max_age=2)
        
        # Frame 0: Create track
        detections = [
            Detection(
                camera_id="cam1", frame_id=0,
                bbox=(10.0, 10.0, 20.0, 30.0),
                class_name="player", confidence=0.8
            ),
        ]
        tracker.update(detections)
        assert len(tracker.get_active_tracks()) == 1
        
        # Frame 1-3: No detections (track ages)
        for _ in range(3):
            tracker.update([])
        
        # Track should be deactivated after max_age frames
        active_tracks = tracker.get_active_tracks()
        assert len(active_tracks) == 0
    
    def test_tracker_separates_player_and_ball_tracks(self):
        """Test tracker maintains separate tracks for players and ball."""
        tracker = Tracker()
        
        detections = [
            Detection(
                camera_id="cam1", frame_id=0,
                bbox=(10.0, 10.0, 20.0, 30.0),
                class_name="player", confidence=0.8
            ),
            Detection(
                camera_id="cam1", frame_id=0,
                bbox=(15.0, 15.0, 10.0, 10.0),
                class_name="ball", confidence=0.7
            ),
        ]
        
        tracks = tracker.update(detections)
        
        player_tracks = [t for t in tracks if t.class_name == "player"]
        ball_tracks = [t for t in tracks if t.class_name == "ball"]
        
        assert len(player_tracks) == 1
        assert len(ball_tracks) == 1
    
    def test_tracker_compute_iou_perfect_overlap(self):
        """Test IOU computation with perfect overlap."""
        tracker = Tracker()
        
        bbox1 = (10.0, 10.0, 20.0, 30.0)
        bbox2 = (10.0, 10.0, 20.0, 30.0)
        
        iou = tracker._compute_iou(bbox1, bbox2)
        
        assert iou == 1.0
    
    def test_tracker_compute_iou_no_overlap(self):
        """Test IOU computation with no overlap."""
        tracker = Tracker()
        
        bbox1 = (10.0, 10.0, 20.0, 30.0)
        bbox2 = (100.0, 100.0, 20.0, 30.0)
        
        iou = tracker._compute_iou(bbox1, bbox2)
        
        assert iou == 0.0
    
    def test_tracker_compute_iou_partial_overlap(self):
        """Test IOU computation with partial overlap."""
        tracker = Tracker()
        
        bbox1 = (0.0, 0.0, 20.0, 20.0)  # Area = 400
        bbox2 = (10.0, 10.0, 20.0, 20.0)  # Area = 400
        # Intersection = 10x10 = 100
        # Union = 400 + 400 - 100 = 700
        # IOU = 100/700 ≈ 0.143
        
        iou = tracker._compute_iou(bbox1, bbox2)
        
        assert 0.14 < iou < 0.15
    
    def test_tracker_get_confirmed_tracks(self):
        """Test getting confirmed tracks (min_hits threshold)."""
        tracker = Tracker(min_hits=3)
        
        detection = Detection(
            camera_id="cam1", frame_id=0,
            bbox=(10.0, 10.0, 20.0, 30.0),
            class_name="player", confidence=0.8
        )
        
        # Frame 0-1: Not confirmed yet
        tracker.update([detection])
        assert len(tracker.get_confirmed_tracks()) == 0
        
        detection.frame_id = 1
        tracker.update([detection])
        assert len(tracker.get_confirmed_tracks()) == 0
        
        # Frame 2: Now confirmed
        detection.frame_id = 2
        tracker.update([detection])
        assert len(tracker.get_confirmed_tracks()) == 1
    
    def test_tracker_reset(self):
        """Test tracker reset clears all state."""
        tracker = Tracker()
        
        # Add some detections
        detections = [
            Detection(
                camera_id="cam1", frame_id=0,
                bbox=(10.0, 10.0, 20.0, 30.0),
                class_name="player", confidence=0.8
            ),
        ]
        tracker.update(detections)
        
        assert len(tracker.tracks) > 0
        assert tracker.next_track_id > 1
        
        # Reset
        tracker.reset()
        
        assert len(tracker.tracks) == 0
        assert tracker.next_track_id == 1
        assert tracker.frame_count == 0
