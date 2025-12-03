"""Unit tests for detector module."""

import pytest
import numpy as np
from ml.detector import Detection, Detector


class TestDetection:
    """Tests for Detection dataclass."""
    
    def test_detection_creation_valid(self):
        """Test creating a valid Detection."""
        detection = Detection(
            camera_id="cam1",
            frame_id=42,
            bbox=(10.0, 20.0, 50.0, 100.0),
            class_name="player",
            confidence=0.85,
        )
        
        assert detection.camera_id == "cam1"
        assert detection.frame_id == 42
        assert detection.bbox == (10.0, 20.0, 50.0, 100.0)
        assert detection.class_name == "player"
        assert detection.confidence == 0.85
    
    def test_detection_invalid_class(self):
        """Test Detection with invalid class_name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid class_name"):
            Detection(
                camera_id="cam1",
                frame_id=0,
                bbox=(0.0, 0.0, 10.0, 10.0),
                class_name="invalid",
                confidence=0.5,
            )
    
    def test_detection_invalid_confidence_too_low(self):
        """Test Detection with confidence < 0 raises ValueError."""
        with pytest.raises(ValueError, match="Confidence"):
            Detection(
                camera_id="cam1",
                frame_id=0,
                bbox=(0.0, 0.0, 10.0, 10.0),
                class_name="player",
                confidence=-0.1,
            )
    
    def test_detection_invalid_confidence_too_high(self):
        """Test Detection with confidence > 1 raises ValueError."""
        with pytest.raises(ValueError, match="Confidence"):
            Detection(
                camera_id="cam1",
                frame_id=0,
                bbox=(0.0, 0.0, 10.0, 10.0),
                class_name="player",
                confidence=1.5,
            )
    
    def test_detection_invalid_bbox_negative_width(self):
        """Test Detection with negative width raises ValueError."""
        with pytest.raises(ValueError, match="Bbox dimensions"):
            Detection(
                camera_id="cam1",
                frame_id=0,
                bbox=(0.0, 0.0, -10.0, 10.0),
                class_name="player",
                confidence=0.5,
            )
    
    def test_detection_invalid_bbox_negative_height(self):
        """Test Detection with negative height raises ValueError."""
        with pytest.raises(ValueError, match="Bbox dimensions"):
            Detection(
                camera_id="cam1",
                frame_id=0,
                bbox=(0.0, 0.0, 10.0, -10.0),
                class_name="player",
                confidence=0.5,
            )
    
    def test_detection_center_property(self):
        """Test Detection center property computation."""
        detection = Detection(
            camera_id="cam1",
            frame_id=0,
            bbox=(10.0, 20.0, 50.0, 100.0),
            class_name="player",
            confidence=0.8,
        )
        
        center_x, center_y = detection.center
        assert center_x == 35.0  # 10 + 50/2
        assert center_y == 70.0  # 20 + 100/2
    
    def test_detection_area_property(self):
        """Test Detection area property computation."""
        detection = Detection(
            camera_id="cam1",
            frame_id=0,
            bbox=(0.0, 0.0, 50.0, 100.0),
            class_name="player",
            confidence=0.8,
        )
        
        assert detection.area == 5000.0  # 50 * 100


class TestDetector:
    """Tests for Detector class."""
    
    def test_detector_initialization_mock(self):
        """Test Detector initialization with mock mode."""
        detector = Detector(use_mock=True)
        
        assert detector.use_mock is True
        assert detector.model is None
        assert detector.confidence_threshold == 0.5
    
    def test_detector_initialization_custom_threshold(self):
        """Test Detector initialization with custom threshold."""
        detector = Detector(use_mock=True, confidence_threshold=0.7)
        
        assert detector.confidence_threshold == 0.7
    
    def test_detector_detect_returns_list(self):
        """Test detect() returns a list of Detection objects."""
        detector = Detector(use_mock=True)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        detections = detector.detect(frame, "cam1", 0)
        
        assert isinstance(detections, list)
        assert all(isinstance(d, Detection) for d in detections)
    
    def test_detector_detect_empty_frame(self):
        """Test detect() with empty frame returns empty list."""
        detector = Detector(use_mock=True)
        frame = np.array([])
        
        detections = detector.detect(frame, "cam1", 0)
        
        assert detections == []
    
    def test_detector_detect_invalid_frame_shape(self):
        """Test detect() with invalid frame shape returns empty list."""
        detector = Detector(use_mock=True)
        frame = np.zeros((720, 1280), dtype=np.uint8)  # 2D instead of 3D
        
        detections = detector.detect(frame, "cam1", 0)
        
        assert detections == []
    
    def test_detector_detect_none_frame(self):
        """Test detect() with None frame returns empty list."""
        detector = Detector(use_mock=True)
        
        detections = detector.detect(None, "cam1", 0)
        
        assert detections == []
    
    def test_detector_mock_generates_players(self):
        """Test mock detector generates player detections."""
        detector = Detector(use_mock=True)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        detections = detector.detect(frame, "cam1", 0)
        
        player_detections = [d for d in detections if d.class_name == "player"]
        assert len(player_detections) > 0
    
    def test_detector_mock_generates_ball(self):
        """Test mock detector can generate ball detection."""
        detector = Detector(use_mock=True, confidence_threshold=0.3)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Run multiple times to account for randomness
        ball_found = False
        for frame_id in range(10):
            detections = detector.detect(frame, "cam1", frame_id)
            ball_detections = [d for d in detections if d.class_name == "ball"]
            if len(ball_detections) > 0:
                ball_found = True
                break
        
        assert ball_found, "Mock detector should generate ball detections"
    
    def test_detector_mock_detections_valid(self):
        """Test all mock detections are valid Detection objects."""
        detector = Detector(use_mock=True)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        detections = detector.detect(frame, "cam1", 42)
        
        for detection in detections:
            # Check all required fields exist
            assert detection.camera_id == "cam1"
            assert detection.frame_id == 42
            assert len(detection.bbox) == 4
            assert detection.class_name in ["player", "ball"]
            assert 0.0 <= detection.confidence <= 1.0
    
    def test_detector_mock_bbox_within_frame(self):
        """Test mock detections have bboxes within frame bounds."""
        detector = Detector(use_mock=True)
        height, width = 720, 1280
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        detections = detector.detect(frame, "cam1", 0)
        
        for detection in detections:
            x, y, w, h = detection.bbox
            assert x >= 0
            assert y >= 0
            assert x + w <= width + 1  # Allow small rounding error
            assert y + h <= height + 1
    
    def test_detector_mock_reproducible(self):
        """Test mock detector produces same results for same inputs."""
        detector = Detector(use_mock=True)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        detections1 = detector.detect(frame, "cam1", 42)
        detections2 = detector.detect(frame, "cam1", 42)
        
        # Should produce same number of detections
        assert len(detections1) == len(detections2)
        
        # Should have same bboxes (approximately)
        for d1, d2 in zip(detections1, detections2):
            assert d1.camera_id == d2.camera_id
            assert d1.frame_id == d2.frame_id
            assert d1.class_name == d2.class_name
            # Bboxes should be identical (deterministic random seed)
            assert d1.bbox == d2.bbox
    
    def test_detector_respects_confidence_threshold(self):
        """Test detector respects confidence threshold for ball."""
        # High threshold should sometimes exclude ball
        detector = Detector(use_mock=True, confidence_threshold=0.9)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Run multiple times
        ball_counts = []
        for frame_id in range(20):
            detections = detector.detect(frame, "cam1", frame_id)
            ball_detections = [d for d in detections if d.class_name == "ball"]
            ball_counts.append(len(ball_detections))
        
        # With high threshold, some frames should have no ball
        assert any(count == 0 for count in ball_counts), \
            "High threshold should exclude some ball detections"
