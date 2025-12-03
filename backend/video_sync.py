"""Video synchronization module for multi-camera frame alignment.

This module provides classes for synchronizing frames from multiple cameras
to a common timeline, handling frame rate differences, jitter, and missing frames.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from backend.video_ingest import Frame

logger = logging.getLogger(__name__)


@dataclass
class SyncBuffer:
    """Buffer for synchronized frames."""

    frames: Dict[str, List[Frame]]  # camera_id -> sorted list of frames
    start_time: float
    end_time: float
    target_fps: float


class VideoSynchronizer:
    """Synchronizes frames from multiple cameras to common timeline.

    Handles timestamp alignment, frame rate differences, and missing frames.
    """

    def __init__(self, sync_tolerance: float = 0.033) -> None:
        """Initialize video synchronizer.

        Args:
            sync_tolerance: Maximum time difference for frame matching (seconds)
                           Default: 33ms (1 frame at 30fps)
        """
        # Guard: Validate tolerance
        if sync_tolerance <= 0:
            raise ValueError("sync_tolerance must be positive")

        self.sync_tolerance = sync_tolerance

    def synchronize_frames(
        self, frames: Dict[str, List[Frame]], target_time: float
    ) -> Dict[str, Frame]:
        """Synchronize frames from multiple cameras to target timestamp.

        Args:
            frames: Dictionary mapping camera_id to list of frames
            target_time: Target timestamp to synchronize to

        Returns:
            Dictionary mapping camera_id to closest synchronized frame

        Raises:
            ValueError: If frames dict is empty
        """
        # Guard: Check frames dict
        if not frames:
            raise ValueError("Frames dictionary cannot be empty")

        synchronized: Dict[str, Frame] = {}

        for camera_id, camera_frames in frames.items():
            # Guard: Check if camera has frames
            if not camera_frames:
                logger.warning(f"No frames available for camera {camera_id}")
                synchronized[camera_id] = self._create_invalid_frame(
                    camera_id, target_time
                )
                continue

            # Find closest frame
            closest_frame = self._find_closest_frame(
                camera_frames, target_time
            )

            # Check if within tolerance
            time_diff = abs(closest_frame.timestamp - target_time)

            if time_diff > self.sync_tolerance:
                logger.warning(
                    f"Frame for {camera_id} at {closest_frame.timestamp:.3f}s "
                    f"exceeds tolerance ({time_diff:.3f}s > {self.sync_tolerance:.3f}s)"
                )
                synchronized[camera_id] = self._create_invalid_frame(
                    camera_id, target_time
                )
                continue

            synchronized[camera_id] = closest_frame

        return synchronized

    def build_sync_buffer(
        self, frames: Dict[str, List[Frame]], target_fps: float
    ) -> SyncBuffer:
        """Build synchronization buffer from frame lists.

        Args:
            frames: Dictionary mapping camera_id to list of frames
            target_fps: Target frame rate for synchronization

        Returns:
            SyncBuffer object

        Raises:
            ValueError: If frames dict is empty or target_fps invalid
        """
        # Guard: Check frames
        if not frames:
            raise ValueError("Frames dictionary cannot be empty")

        # Guard: Validate target_fps
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")

        # Determine time range
        start_times: List[float] = []
        end_times: List[float] = []

        for camera_frames in frames.values():
            if not camera_frames:
                continue

            start_times.append(camera_frames[0].timestamp)
            end_times.append(camera_frames[-1].timestamp)

        # Guard: Check if we have valid time ranges
        if not start_times or not end_times:
            raise ValueError("No valid frames to build sync buffer")

        # Use latest start and earliest end for overlapping region
        start_time = max(start_times)
        end_time = min(end_times)

        # Guard: Check valid time range
        if end_time <= start_time:
            raise ValueError(
                f"No overlapping time range: start={start_time:.3f}s, end={end_time:.3f}s"
            )

        logger.info(
            f"Built sync buffer: [{start_time:.3f}s, {end_time:.3f}s] @ {target_fps}fps"
        )

        return SyncBuffer(
            frames=frames,
            start_time=start_time,
            end_time=end_time,
            target_fps=target_fps,
        )

    def get_synchronized_batch(
        self, buffer: SyncBuffer, min_cameras: int = 2
    ) -> List[Dict[str, Frame]]:
        """Get batch of synchronized frames from buffer.

        Args:
            buffer: Sync buffer containing frames
            min_cameras: Minimum number of cameras required per timestamp

        Returns:
            List of synchronized frame dictionaries

        Raises:
            ValueError: If min_cameras is invalid
        """
        # Guard: Validate min_cameras
        if min_cameras < 1:
            raise ValueError("min_cameras must be at least 1")

        # Compute timestamps to sample
        frame_interval = 1.0 / buffer.target_fps
        timestamps = np.arange(
            buffer.start_time, buffer.end_time, frame_interval
        )

        synchronized_batch: List[Dict[str, Frame]] = []

        for timestamp in timestamps:
            synced_frames = self.synchronize_frames(
                buffer.frames, float(timestamp)
            )

            # Check if we have enough valid cameras
            if self._is_frame_batch_valid(synced_frames, min_cameras):
                synchronized_batch.append(synced_frames)
            else:
                logger.debug(
                    f"Skipping timestamp {timestamp:.3f}s: insufficient valid cameras"
                )

        logger.info(
            f"Synchronized {len(synchronized_batch)} frame sets from {len(timestamps)} timestamps"
        )

        return synchronized_batch

    def compute_time_offset(
        self, reference_frames: List[Frame], target_frames: List[Frame]
    ) -> float:
        """Compute time offset between target and reference camera.

        Uses first frame timestamps.

        Args:
            reference_frames: Frames from reference camera
            target_frames: Frames from target camera

        Returns:
            Time offset in seconds (target - reference)

        Raises:
            ValueError: If frame lists are empty
        """
        # Guard: Check reference frames
        if not reference_frames:
            raise ValueError("Reference frames list is empty")

        # Guard: Check target frames
        if not target_frames:
            raise ValueError("Target frames list is empty")

        ref_timestamp = reference_frames[0].timestamp
        target_timestamp = target_frames[0].timestamp

        offset = target_timestamp - ref_timestamp

        logger.info(
            f"Computed time offset: {offset:.3f}s "
            f"(reference={ref_timestamp:.3f}s, target={target_timestamp:.3f}s)"
        )

        return offset

    def _find_closest_frame(
        self, frames: List[Frame], target_time: float
    ) -> Frame:
        """Find frame closest to target timestamp.

        Args:
            frames: List of frames (assumed sorted by timestamp)
            target_time: Target timestamp

        Returns:
            Closest frame

        Raises:
            ValueError: If frames list is empty
        """
        if not frames:
            raise ValueError("Frames list is empty")

        # Binary search for closest frame
        timestamps = np.array([f.timestamp for f in frames])
        idx = np.argmin(np.abs(timestamps - target_time))

        return frames[int(idx)]

    def _create_invalid_frame(
        self, camera_id: str, timestamp: float
    ) -> Frame:
        """Create placeholder frame for missing data.

        Args:
            camera_id: Camera identifier
            timestamp: Timestamp for the frame

        Returns:
            Invalid Frame object
        """
        return Frame(
            camera_id=camera_id,
            timestamp=timestamp,
            frame_number=-1,
            image=None,
            is_valid=False,
            metadata={"reason": "sync_failed"},
        )

    def _is_frame_batch_valid(
        self, frames: Dict[str, Frame], min_cameras: int
    ) -> bool:
        """Check if frame batch has sufficient valid cameras.

        Args:
            frames: Dictionary of frames
            min_cameras: Minimum required valid cameras

        Returns:
            True if batch is valid, False otherwise
        """
        valid_count = sum(1 for f in frames.values() if f.is_valid)
        return valid_count >= min_cameras
