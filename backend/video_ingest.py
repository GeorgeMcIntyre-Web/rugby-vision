"""Video ingestion module for multi-camera video sources.

This module provides classes for loading and managing video from multiple
camera sources (files, streams, synthetic). It coordinates frame retrieval
and passes data to the synchronization layer.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Metadata for a video source."""

    camera_id: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    codec: str


@dataclass
class Frame:
    """Represents a single video frame."""

    camera_id: str
    timestamp: float
    frame_number: int
    image: Optional[np.ndarray]
    is_valid: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoSourceConfig:
    """Configuration for a video source."""

    path: str
    camera_id: str
    source_type: str = "file"  # "file", "stream", "synthetic"
    offset_seconds: float = 0.0
    calibration_path: Optional[str] = None


class VideoSource:
    """Represents a single camera video source.

    Handles opening, reading, and seeking video files or streams.
    Provides frame extraction with metadata.
    """

    def __init__(self, config: VideoSourceConfig) -> None:
        """Initialize video source.

        Args:
            config: Configuration for this video source
        """
        self.config = config
        self.capture: Optional[cv2.VideoCapture] = None
        self.metadata: Optional[VideoMetadata] = None
        self._is_opened = False

    def open(self) -> bool:
        """Open the video source.

        Returns:
            True if successful, False otherwise
        """
        # Guard: Check if already opened
        if self._is_opened:
            logger.warning(f"Video source {self.config.camera_id} already opened")
            return True

        # Guard: Check file exists for file sources
        if self.config.source_type == "file":
            if not os.path.exists(self.config.path):
                logger.error(f"Video file not found: {self.config.path}")
                return False

        try:
            self.capture = cv2.VideoCapture(self.config.path)

            if not self.capture.isOpened():
                logger.error(f"Failed to open video: {self.config.path}")
                return False

            # Extract metadata
            self.metadata = self._extract_metadata()
            self._is_opened = True

            logger.info(
                f"Opened video source {self.config.camera_id}: "
                f"{self.metadata.width}x{self.metadata.height} @ {self.metadata.fps}fps"
            )
            return True

        except Exception as e:
            logger.error(f"Error opening video source {self.config.camera_id}: {e}")
            return False

    def read_frame(self) -> Optional[Frame]:
        """Read the next frame from the video source.

        Returns:
            Frame object if successful, None if no more frames or error
        """
        # Guard: Check if opened
        if not self._is_opened or self.capture is None:
            logger.error(f"Video source {self.config.camera_id} not opened")
            return None

        try:
            ret, image = self.capture.read()

            if not ret:
                return None

            frame_number = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            timestamp_ms = self.capture.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = timestamp_ms / 1000.0 + self.config.offset_seconds

            return Frame(
                camera_id=self.config.camera_id,
                timestamp=timestamp,
                frame_number=frame_number,
                image=image,
                is_valid=True,
                metadata={"original_timestamp_ms": timestamp_ms},
            )

        except Exception as e:
            logger.error(
                f"Error reading frame from {self.config.camera_id}: {e}"
            )
            return None

    def seek(self, timestamp: float) -> bool:
        """Seek to a specific timestamp.

        Args:
            timestamp: Target timestamp in seconds

        Returns:
            True if successful, False otherwise
        """
        # Guard: Check if opened
        if not self._is_opened or self.capture is None:
            logger.error(f"Video source {self.config.camera_id} not opened")
            return False

        # Guard: Validate timestamp
        if timestamp < 0:
            logger.error("Timestamp must be non-negative")
            return False

        try:
            # Adjust for offset
            adjusted_timestamp = timestamp - self.config.offset_seconds
            timestamp_ms = adjusted_timestamp * 1000.0

            self.capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
            return True

        except Exception as e:
            logger.error(f"Error seeking to {timestamp}s: {e}")
            return False

    def get_metadata(self) -> Optional[VideoMetadata]:
        """Get video metadata.

        Returns:
            VideoMetadata if available, None otherwise
        """
        return self.metadata

    def release(self) -> None:
        """Release video resources."""
        if self.capture is not None:
            self.capture.release()
            self._is_opened = False
            logger.info(f"Released video source {self.config.camera_id}")

    def _extract_metadata(self) -> VideoMetadata:
        """Extract metadata from opened video capture.

        Returns:
            VideoMetadata object

        Raises:
            ValueError: If capture is not opened
        """
        if self.capture is None:
            raise ValueError("Cannot extract metadata: capture not initialized")

        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        # Get codec (fourcc code)
        fourcc = int(self.capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        return VideoMetadata(
            camera_id=self.config.camera_id,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            codec=codec,
        )


class VideoIngestor:
    """Main facade for managing multiple video sources.

    Coordinates frame reading across multiple cameras and provides
    a high-level API for clip analysis.
    """

    def __init__(
        self, sources: List[VideoSourceConfig], buffer_window: float = 5.0
    ) -> None:
        """Initialize video ingestor.

        Args:
            sources: List of video source configurations
            buffer_window: Time window for frame buffering (seconds)

        Raises:
            ValueError: If sources list is invalid
        """
        # Guard: Check sources list
        if not sources:
            raise ValueError("Sources list cannot be empty")

        # Guard: Minimum cameras
        if len(sources) < 2:
            raise ValueError("At least 2 cameras required for 3D reconstruction")

        self.source_configs = sources
        self.buffer_window = buffer_window
        self.sources: Dict[str, VideoSource] = {}
        self._loaded = False

    def load_sources(self) -> None:
        """Load and open all video sources.

        Raises:
            RuntimeError: If any source fails to load
        """
        # Guard: Check if already loaded
        if self._loaded:
            logger.warning("Sources already loaded")
            return

        logger.info(f"Loading {len(self.source_configs)} video sources...")

        for config in self.source_configs:
            source = VideoSource(config)

            if not source.open():
                raise RuntimeError(f"Failed to open source: {config.camera_id}")

            self.sources[config.camera_id] = source

        self._loaded = True
        logger.info(f"Successfully loaded {len(self.sources)} video sources")

    def get_frame_at_time(self, timestamp: float) -> Dict[str, Frame]:
        """Get synchronized frames at a specific timestamp.

        Args:
            timestamp: Target timestamp in seconds

        Returns:
            Dictionary mapping camera_id to Frame

        Raises:
            RuntimeError: If sources not loaded
        """
        # Guard: Check if loaded
        if not self._loaded:
            raise RuntimeError("Sources not loaded. Call load_sources() first.")

        # Guard: Validate timestamp
        if timestamp < 0:
            raise ValueError("Timestamp must be non-negative")

        frames: Dict[str, Frame] = {}

        for camera_id, source in self.sources.items():
            # Seek to timestamp
            if not source.seek(timestamp):
                logger.warning(
                    f"Failed to seek to {timestamp}s for {camera_id}"
                )
                # Create invalid frame placeholder
                frames[camera_id] = Frame(
                    camera_id=camera_id,
                    timestamp=timestamp,
                    frame_number=-1,
                    image=None,
                    is_valid=False,
                    metadata={},
                )
                continue

            # Read frame
            frame = source.read_frame()

            if frame is None:
                logger.warning(
                    f"Failed to read frame at {timestamp}s for {camera_id}"
                )
                frames[camera_id] = Frame(
                    camera_id=camera_id,
                    timestamp=timestamp,
                    frame_number=-1,
                    image=None,
                    is_valid=False,
                    metadata={},
                )
                continue

            frames[camera_id] = frame

        return frames

    def get_frame_batch(
        self, start_time: float, end_time: float, target_fps: Optional[float] = None
    ) -> List[Dict[str, Frame]]:
        """Get batch of synchronized frames over a time window.

        Args:
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            target_fps: Target frame rate (uses minimum source fps if None)

        Returns:
            List of frame dictionaries (camera_id -> Frame)

        Raises:
            ValueError: If time range is invalid
            RuntimeError: If sources not loaded
        """
        # Guard: Check if loaded
        if not self._loaded:
            raise RuntimeError("Sources not loaded. Call load_sources() first.")

        # Guard: Validate time range
        if end_time <= start_time:
            raise ValueError("end_time must be greater than start_time")

        # Determine target fps
        if target_fps is None:
            target_fps = self._compute_target_fps()

        # Compute timestamps to sample
        frame_interval = 1.0 / target_fps
        timestamps = np.arange(start_time, end_time, frame_interval)

        logger.info(
            f"Fetching {len(timestamps)} frame sets from {start_time}s to {end_time}s"
        )

        batch: List[Dict[str, Frame]] = []

        for timestamp in timestamps:
            frames = self.get_frame_at_time(float(timestamp))
            batch.append(frames)

        return batch

    def release(self) -> None:
        """Release all video sources."""
        for source in self.sources.values():
            source.release()

        self.sources.clear()
        self._loaded = False
        logger.info("Released all video sources")

    def _compute_target_fps(self) -> float:
        """Compute target frame rate (minimum of all sources).

        Returns:
            Target FPS value

        Raises:
            RuntimeError: If no sources loaded or metadata unavailable
        """
        if not self.sources:
            raise RuntimeError("No sources loaded")

        fps_values: List[float] = []

        for source in self.sources.values():
            metadata = source.get_metadata()

            if metadata is None:
                raise RuntimeError(
                    f"Metadata not available for {source.config.camera_id}"
                )

            fps_values.append(metadata.fps)

        return min(fps_values)
