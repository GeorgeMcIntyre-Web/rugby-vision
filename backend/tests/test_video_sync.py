"""Unit tests for video synchronization module."""

import pytest
import numpy as np
from typing import Dict, List

from backend.video_ingest import Frame
from backend.video_sync import VideoSynchronizer, SyncBuffer


class TestVideoSynchronizer:
    """Tests for VideoSynchronizer class."""

    def test_synchronizer_initialization_default_tolerance(self) -> None:
        """Test VideoSynchronizer initialization with default tolerance."""
        sync = VideoSynchronizer()

        assert sync.sync_tolerance == 0.033

    def test_synchronizer_initialization_custom_tolerance(self) -> None:
        """Test VideoSynchronizer initialization with custom tolerance."""
        sync = VideoSynchronizer(sync_tolerance=0.050)

        assert sync.sync_tolerance == 0.050

    def test_synchronizer_initialization_negative_tolerance_raises_error(
        self,
    ) -> None:
        """Test initialization with negative tolerance raises ValueError."""
        with pytest.raises(ValueError, match="sync_tolerance must be positive"):
            VideoSynchronizer(sync_tolerance=-0.01)

    def test_synchronizer_initialization_zero_tolerance_raises_error(
        self,
    ) -> None:
        """Test initialization with zero tolerance raises ValueError."""
        with pytest.raises(ValueError, match="sync_tolerance must be positive"):
            VideoSynchronizer(sync_tolerance=0.0)

    def test_synchronize_frames_empty_dict_raises_error(self) -> None:
        """Test synchronizing with empty frames dict raises ValueError."""
        sync = VideoSynchronizer()

        with pytest.raises(ValueError, match="Frames dictionary cannot be empty"):
            sync.synchronize_frames({}, target_time=1.0)

    def test_synchronize_frames_aligns_within_tolerance(self) -> None:
        """Test synchronizing frames within tolerance."""
        sync = VideoSynchronizer(sync_tolerance=0.040)

        # Create mock frames
        frames = {
            "cam1": [
                Frame(
                    camera_id="cam1",
                    timestamp=1.000,
                    frame_number=30,
                    image=np.zeros((720, 1280, 3)),
                    is_valid=True,
                )
            ],
            "cam2": [
                Frame(
                    camera_id="cam2",
                    timestamp=1.033,
                    frame_number=31,
                    image=np.zeros((720, 1280, 3)),
                    is_valid=True,
                )
            ],
        }

        aligned = sync.synchronize_frames(frames, target_time=1.0)

        assert "cam1" in aligned
        assert "cam2" in aligned
        assert aligned["cam1"].timestamp == 1.000
        assert aligned["cam2"].timestamp == 1.033
        assert aligned["cam1"].is_valid is True
        assert aligned["cam2"].is_valid is True

    def test_synchronize_frames_creates_invalid_frame_outside_tolerance(
        self,
    ) -> None:
        """Test synchronizing creates invalid frame when outside tolerance."""
        sync = VideoSynchronizer(sync_tolerance=0.020)

        # Create mock frames - cam2 is outside tolerance
        frames = {
            "cam1": [
                Frame(
                    camera_id="cam1",
                    timestamp=1.000,
                    frame_number=30,
                    image=np.zeros((720, 1280, 3)),
                    is_valid=True,
                )
            ],
            "cam2": [
                Frame(
                    camera_id="cam2",
                    timestamp=1.050,  # 50ms difference, outside 20ms tolerance
                    frame_number=31,
                    image=np.zeros((720, 1280, 3)),
                    is_valid=True,
                )
            ],
        }

        aligned = sync.synchronize_frames(frames, target_time=1.0)

        assert aligned["cam1"].is_valid is True
        assert aligned["cam2"].is_valid is False

    def test_build_sync_buffer_valid_frames(self) -> None:
        """Test building sync buffer with valid frames."""
        sync = VideoSynchronizer()

        frames = {
            "cam1": [
                Frame(
                    camera_id="cam1",
                    timestamp=1.0,
                    frame_number=30,
                    image=None,
                    is_valid=True,
                ),
                Frame(
                    camera_id="cam1",
                    timestamp=2.0,
                    frame_number=60,
                    image=None,
                    is_valid=True,
                ),
            ],
            "cam2": [
                Frame(
                    camera_id="cam2",
                    timestamp=1.0,
                    frame_number=30,
                    image=None,
                    is_valid=True,
                ),
                Frame(
                    camera_id="cam2",
                    timestamp=2.0,
                    frame_number=60,
                    image=None,
                    is_valid=True,
                ),
            ],
        }

        buffer = sync.build_sync_buffer(frames, target_fps=30.0)

        assert buffer.start_time == 1.0
        assert buffer.end_time == 2.0
        assert buffer.target_fps == 30.0
        assert "cam1" in buffer.frames
        assert "cam2" in buffer.frames

    def test_build_sync_buffer_empty_frames_raises_error(self) -> None:
        """Test building buffer with empty frames raises ValueError."""
        sync = VideoSynchronizer()

        with pytest.raises(ValueError, match="Frames dictionary cannot be empty"):
            sync.build_sync_buffer({}, target_fps=30.0)

    def test_build_sync_buffer_invalid_fps_raises_error(self) -> None:
        """Test building buffer with invalid fps raises ValueError."""
        sync = VideoSynchronizer()

        frames = {
            "cam1": [
                Frame(
                    camera_id="cam1",
                    timestamp=1.0,
                    frame_number=30,
                    image=None,
                    is_valid=True,
                )
            ]
        }

        with pytest.raises(ValueError, match="target_fps must be positive"):
            sync.build_sync_buffer(frames, target_fps=0.0)

    def test_compute_time_offset_returns_correct_offset(self) -> None:
        """Test computing time offset between cameras."""
        sync = VideoSynchronizer()

        reference_frames = [
            Frame(
                camera_id="ref",
                timestamp=0.0,
                frame_number=0,
                image=None,
                is_valid=True,
            )
        ]

        target_frames = [
            Frame(
                camera_id="target",
                timestamp=2.5,
                frame_number=75,
                image=None,
                is_valid=True,
            )
        ]

        offset = sync.compute_time_offset(reference_frames, target_frames)

        assert offset == 2.5

    def test_compute_time_offset_empty_reference_raises_error(self) -> None:
        """Test computing offset with empty reference raises ValueError."""
        sync = VideoSynchronizer()

        target_frames = [
            Frame(
                camera_id="target",
                timestamp=1.0,
                frame_number=30,
                image=None,
                is_valid=True,
            )
        ]

        with pytest.raises(ValueError, match="Reference frames list is empty"):
            sync.compute_time_offset([], target_frames)

    def test_compute_time_offset_empty_target_raises_error(self) -> None:
        """Test computing offset with empty target raises ValueError."""
        sync = VideoSynchronizer()

        reference_frames = [
            Frame(
                camera_id="ref",
                timestamp=0.0,
                frame_number=0,
                image=None,
                is_valid=True,
            )
        ]

        with pytest.raises(ValueError, match="Target frames list is empty"):
            sync.compute_time_offset(reference_frames, [])


class TestSyncBuffer:
    """Tests for SyncBuffer dataclass."""

    def test_sync_buffer_creation(self) -> None:
        """Test creating SyncBuffer."""
        frames: Dict[str, List[Frame]] = {
            "cam1": [
                Frame(
                    camera_id="cam1",
                    timestamp=1.0,
                    frame_number=30,
                    image=None,
                    is_valid=True,
                )
            ],
            "cam2": [
                Frame(
                    camera_id="cam2",
                    timestamp=1.0,
                    frame_number=30,
                    image=None,
                    is_valid=True,
                )
            ],
        }

        buffer = SyncBuffer(
            frames=frames,
            start_time=1.0,
            end_time=5.0,
            target_fps=30.0,
        )

        assert buffer.start_time == 1.0
        assert buffer.end_time == 5.0
        assert buffer.target_fps == 30.0
        assert len(buffer.frames) == 2
