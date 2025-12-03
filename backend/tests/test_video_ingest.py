"""Unit tests for video ingestion module."""

import pytest
import numpy as np
from typing import List
import tempfile
import os

from backend.video_ingest import (
    VideoSource,
    VideoSourceConfig,
    VideoIngestor,
    Frame,
    VideoMetadata,
)


class TestVideoSourceConfig:
    """Tests for VideoSourceConfig dataclass."""

    def test_config_creation_with_defaults(self) -> None:
        """Test creating config with default values."""
        config = VideoSourceConfig(
            path="/path/to/video.mp4",
            camera_id="cam1",
        )

        assert config.path == "/path/to/video.mp4"
        assert config.camera_id == "cam1"
        assert config.source_type == "file"
        assert config.offset_seconds == 0.0
        assert config.calibration_path is None

    def test_config_creation_with_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = VideoSourceConfig(
            path="rtsp://stream",
            camera_id="stream_cam",
            source_type="stream",
            offset_seconds=2.5,
            calibration_path="/path/to/calibration.json",
        )

        assert config.source_type == "stream"
        assert config.offset_seconds == 2.5
        assert config.calibration_path == "/path/to/calibration.json"


class TestFrame:
    """Tests for Frame dataclass."""

    def test_frame_creation_valid(self) -> None:
        """Test creating a valid frame."""
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame = Frame(
            camera_id="cam1",
            timestamp=1.5,
            frame_number=45,
            image=image,
            is_valid=True,
        )

        assert frame.camera_id == "cam1"
        assert frame.timestamp == 1.5
        assert frame.frame_number == 45
        assert frame.is_valid is True
        assert frame.image.shape == (720, 1280, 3)

    def test_frame_creation_invalid(self) -> None:
        """Test creating an invalid frame (missing image)."""
        frame = Frame(
            camera_id="cam2",
            timestamp=2.0,
            frame_number=-1,
            image=None,
            is_valid=False,
        )

        assert frame.is_valid is False
        assert frame.image is None


class TestVideoSource:
    """Tests for VideoSource class."""

    def test_source_initialization(self) -> None:
        """Test VideoSource initialization."""
        config = VideoSourceConfig(path="test.mp4", camera_id="test_cam")
        source = VideoSource(config)

        assert source.config == config
        assert source.capture is None
        assert source.metadata is None

    def test_open_nonexistent_file_returns_false(self) -> None:
        """Test opening non-existent file returns False."""
        config = VideoSourceConfig(
            path="/nonexistent/video.mp4",
            camera_id="test_cam",
        )
        source = VideoSource(config)

        result = source.open()

        assert result is False
        assert source.capture is None

    @pytest.mark.skip(reason="Requires actual video file for integration test")
    def test_open_valid_file_returns_true(self) -> None:
        """Test opening valid video file."""
        # This would require a real video file
        # Skip for unit tests, implement in integration tests
        pass

    @pytest.mark.skip(reason="Requires actual video file for integration test")
    def test_read_frame_returns_valid_frame(self) -> None:
        """Test reading frame from opened video."""
        # This would require a real video file
        pass

    def test_seek_negative_timestamp_returns_false(self) -> None:
        """Test seeking to negative timestamp returns False."""
        config = VideoSourceConfig(path="test.mp4", camera_id="test_cam")
        source = VideoSource(config)
        source._is_opened = True  # Mock opened state
        source.capture = object()  # Mock capture object

        result = source.seek(-1.0)

        assert result is False


class TestVideoIngestor:
    """Tests for VideoIngestor class."""

    def test_ingestor_initialization_valid_sources(self) -> None:
        """Test VideoIngestor initialization with valid sources."""
        sources = [
            VideoSourceConfig(path="video1.mp4", camera_id="cam1"),
            VideoSourceConfig(path="video2.mp4", camera_id="cam2"),
        ]
        ingestor = VideoIngestor(sources)

        assert len(ingestor.source_configs) == 2
        assert ingestor.buffer_window == 5.0
        assert not ingestor._loaded

    def test_ingestor_initialization_empty_sources_raises_error(self) -> None:
        """Test initialization with empty sources raises ValueError."""
        with pytest.raises(ValueError, match="Sources list cannot be empty"):
            VideoIngestor([])

    def test_ingestor_initialization_single_source_raises_error(self) -> None:
        """Test initialization with single source raises ValueError."""
        sources = [VideoSourceConfig(path="video1.mp4", camera_id="cam1")]

        with pytest.raises(
            ValueError, match="At least 2 cameras required"
        ):
            VideoIngestor(sources)

    def test_get_frame_at_time_not_loaded_raises_error(self) -> None:
        """Test getting frame when sources not loaded raises RuntimeError."""
        sources = [
            VideoSourceConfig(path="video1.mp4", camera_id="cam1"),
            VideoSourceConfig(path="video2.mp4", camera_id="cam2"),
        ]
        ingestor = VideoIngestor(sources)

        with pytest.raises(RuntimeError, match="Sources not loaded"):
            ingestor.get_frame_at_time(1.0)

    def test_get_frame_at_time_negative_timestamp_raises_error(self) -> None:
        """Test getting frame with negative timestamp raises ValueError."""
        sources = [
            VideoSourceConfig(path="video1.mp4", camera_id="cam1"),
            VideoSourceConfig(path="video2.mp4", camera_id="cam2"),
        ]
        ingestor = VideoIngestor(sources)
        ingestor._loaded = True  # Mock loaded state

        with pytest.raises(ValueError, match="Timestamp must be non-negative"):
            ingestor.get_frame_at_time(-1.0)

    def test_get_frame_batch_invalid_time_range_raises_error(self) -> None:
        """Test getting frame batch with invalid time range raises ValueError."""
        sources = [
            VideoSourceConfig(path="video1.mp4", camera_id="cam1"),
            VideoSourceConfig(path="video2.mp4", camera_id="cam2"),
        ]
        ingestor = VideoIngestor(sources)
        ingestor._loaded = True  # Mock loaded state

        with pytest.raises(
            ValueError, match="end_time must be greater than start_time"
        ):
            ingestor.get_frame_batch(start_time=10.0, end_time=5.0)


# Integration tests would require actual video files
# These should be implemented separately with test fixtures
