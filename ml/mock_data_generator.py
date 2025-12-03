"""Mock/synthetic video data generator for testing.

Generates synthetic rugby footage with simple ball and player movements
for testing the video ingestion, synchronization, and analysis pipeline
without requiring real match footage.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SceneConfig:
    """Configuration for synthetic scene generation."""

    field_length: float = 100.0  # meters
    field_width: float = 70.0  # meters
    ball_trajectory: List[Tuple[float, float, float]] = None  # (x, y, z) positions
    player_positions: List[Tuple[float, float]] = None  # (x, y) positions
    background_color: Tuple[int, int, int] = (34, 139, 34)  # Green field

    def __post_init__(self) -> None:
        """Initialize default values if not provided."""
        if self.ball_trajectory is None:
            # Default: ball moves forward along field
            self.ball_trajectory = [
                (35.0, 40.0, 1.0),
                (35.0, 50.0, 1.5),
                (35.0, 60.0, 1.0),
            ]

        if self.player_positions is None:
            # Default: two players (passer and receiver)
            self.player_positions = [(35.0, 45.0), (35.0, 55.0)]


@dataclass
class CameraConfig:
    """Camera configuration for synthetic video generation."""

    camera_id: str
    position: Tuple[float, float, float]  # (x, y, z) in meters
    look_at: Tuple[float, float, float]  # Point camera looks at
    fov: float = 60.0  # Field of view in degrees
    resolution: Tuple[int, int] = (1280, 720)  # Width x Height


class SyntheticVideoGenerator:
    """Generates synthetic rugby video for testing."""

    def __init__(self) -> None:
        """Initialize synthetic video generator."""
        self.scene: Optional[SceneConfig] = None

    def create_video(
        self,
        output_path: str,
        camera_config: CameraConfig,
        scene: SceneConfig,
        duration: float,
        fps: float = 30.0,
    ) -> bool:
        """Create a synthetic video file.

        Args:
            output_path: Path to output video file
            camera_config: Camera configuration
            scene: Scene configuration
            duration: Video duration in seconds
            fps: Frames per second

        Returns:
            True if successful, False otherwise
        """
        # Guard: Validate duration
        if duration <= 0:
            logger.error("Duration must be positive")
            return False

        # Guard: Validate fps
        if fps <= 0:
            logger.error("FPS must be positive")
            return False

        try:
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                camera_config.resolution,
            )

            if not writer.isOpened():
                logger.error(f"Failed to create video writer for {output_path}")
                return False

            total_frames = int(duration * fps)
            logger.info(
                f"Generating {total_frames} frames for {camera_config.camera_id}"
            )

            for frame_idx in range(total_frames):
                timestamp = frame_idx / fps
                frame = self._render_frame(
                    timestamp, camera_config, scene
                )
                writer.write(frame)

            writer.release()
            logger.info(f"Created synthetic video: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error creating synthetic video: {e}")
            return False

    def create_test_videos(
        self,
        output_dir: str,
        scene: Optional[SceneConfig] = None,
        duration: float = 5.0,
        fps: float = 30.0,
    ) -> List[str]:
        """Create a set of test videos from multiple camera angles.

        Args:
            output_dir: Directory to save videos
            scene: Scene configuration (uses default if None)
            duration: Video duration in seconds
            fps: Frames per second

        Returns:
            List of created video file paths
        """
        if scene is None:
            scene = SceneConfig()

        # Define camera configurations
        cameras = [
            CameraConfig(
                camera_id="cam1",
                position=(50.0, 50.0, 10.0),  # Side view
                look_at=(35.0, 50.0, 1.0),
            ),
            CameraConfig(
                camera_id="cam2",
                position=(35.0, 30.0, 8.0),  # End view
                look_at=(35.0, 50.0, 1.0),
            ),
            CameraConfig(
                camera_id="cam3",
                position=(20.0, 50.0, 12.0),  # Elevated side view
                look_at=(35.0, 50.0, 1.0),
            ),
        ]

        created_videos: List[str] = []

        for cam_config in cameras:
            output_path = f"{output_dir}/{cam_config.camera_id}.mp4"

            if self.create_video(output_path, cam_config, scene, duration, fps):
                created_videos.append(output_path)

        return created_videos

    def _render_frame(
        self,
        timestamp: float,
        camera_config: CameraConfig,
        scene: SceneConfig,
    ) -> np.ndarray:
        """Render a single frame.

        Args:
            timestamp: Current timestamp in seconds
            camera_config: Camera configuration
            scene: Scene configuration

        Returns:
            Rendered frame as numpy array (BGR format)
        """
        width, height = camera_config.resolution
        frame = np.full((height, width, 3), scene.background_color, dtype=np.uint8)

        # Draw field markings
        self._draw_field_markings(frame, camera_config, scene)

        # Draw players
        self._draw_players(frame, camera_config, scene)

        # Draw ball
        self._draw_ball(frame, timestamp, camera_config, scene)

        # Add camera ID text
        cv2.putText(
            frame,
            camera_config.camera_id,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        # Add timestamp
        cv2.putText(
            frame,
            f"t={timestamp:.2f}s",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return frame

    def _draw_field_markings(
        self,
        frame: np.ndarray,
        camera_config: CameraConfig,
        scene: SceneConfig,
    ) -> None:
        """Draw field markings on frame.

        Args:
            frame: Frame to draw on
            camera_config: Camera configuration
            scene: Scene configuration
        """
        # Simple field lines (simplified projection)
        height, width = frame.shape[:2]

        # Draw center line
        center_y = height // 2
        cv2.line(frame, (0, center_y), (width, center_y), (255, 255, 255), 2)

        # Draw 22m lines
        line_22m_1 = int(center_y - height * 0.22)
        line_22m_2 = int(center_y + height * 0.22)
        cv2.line(frame, (0, line_22m_1), (width, line_22m_1), (255, 255, 255), 1)
        cv2.line(frame, (0, line_22m_2), (width, line_22m_2), (255, 255, 255), 1)

    def _draw_players(
        self,
        frame: np.ndarray,
        camera_config: CameraConfig,
        scene: SceneConfig,
    ) -> None:
        """Draw players on frame.

        Args:
            frame: Frame to draw on
            camera_config: Camera configuration
            scene: Scene configuration
        """
        for player_pos in scene.player_positions:
            # Project 3D position to 2D
            pixel_pos = self._project_3d_to_2d(
                (player_pos[0], player_pos[1], 1.8),  # Player height ~1.8m
                camera_config,
                scene,
            )

            # Draw player as circle
            if self._is_point_in_frame(pixel_pos, frame.shape):
                cv2.circle(
                    frame,
                    pixel_pos,
                    15,  # Radius
                    (0, 0, 255),  # Red color
                    -1,  # Filled
                )

    def _draw_ball(
        self,
        frame: np.ndarray,
        timestamp: float,
        camera_config: CameraConfig,
        scene: SceneConfig,
    ) -> None:
        """Draw ball on frame with animated trajectory.

        Args:
            frame: Frame to draw on
            timestamp: Current timestamp
            camera_config: Camera configuration
            scene: Scene configuration
        """
        # Interpolate ball position based on timestamp
        ball_pos = self._interpolate_ball_position(timestamp, scene)

        if ball_pos is None:
            return

        # Project to 2D
        pixel_pos = self._project_3d_to_2d(ball_pos, camera_config, scene)

        # Draw ball
        if self._is_point_in_frame(pixel_pos, frame.shape):
            cv2.circle(
                frame,
                pixel_pos,
                8,  # Radius
                (255, 255, 255),  # White color
                -1,  # Filled
            )

    def _interpolate_ball_position(
        self, timestamp: float, scene: SceneConfig
    ) -> Optional[Tuple[float, float, float]]:
        """Interpolate ball position at given timestamp.

        Args:
            timestamp: Current timestamp in seconds
            scene: Scene configuration

        Returns:
            Ball 3D position (x, y, z) or None if out of range
        """
        if not scene.ball_trajectory:
            return None

        # Simple linear interpolation between trajectory points
        # Assume trajectory points are evenly spaced over 5 seconds
        trajectory_duration = 5.0
        num_points = len(scene.ball_trajectory)

        if timestamp < 0 or timestamp > trajectory_duration:
            return scene.ball_trajectory[-1]  # Return last position

        # Find interpolation segment
        segment_duration = trajectory_duration / (num_points - 1)
        segment_idx = int(timestamp / segment_duration)
        segment_idx = min(segment_idx, num_points - 2)

        # Interpolate
        t = (timestamp - segment_idx * segment_duration) / segment_duration
        p1 = scene.ball_trajectory[segment_idx]
        p2 = scene.ball_trajectory[segment_idx + 1]

        return (
            p1[0] + t * (p2[0] - p1[0]),
            p1[1] + t * (p2[1] - p1[1]),
            p1[2] + t * (p2[2] - p1[2]),
        )

    def _project_3d_to_2d(
        self,
        point_3d: Tuple[float, float, float],
        camera_config: CameraConfig,
        scene: SceneConfig,
    ) -> Tuple[int, int]:
        """Project 3D point to 2D pixel coordinates (simplified).

        Args:
            point_3d: 3D point (x, y, z)
            camera_config: Camera configuration
            scene: Scene configuration

        Returns:
            Pixel coordinates (x, y)
        """
        # Simplified projection (not physically accurate, just for visualization)
        width, height = camera_config.resolution

        # Normalize field coordinates to [0, 1]
        norm_x = point_3d[0] / scene.field_width
        norm_y = point_3d[1] / scene.field_length

        # Simple perspective effect based on camera position
        cam_x, cam_y, cam_z = camera_config.position

        # Apply perspective scaling (closer = larger)
        scale = 1.0 / (1.0 + abs(cam_y - point_3d[1]) / 50.0)

        # Map to pixel coordinates
        pixel_x = int(norm_x * width * scale + width * (1 - scale) / 2)
        pixel_y = int((1 - norm_y) * height * scale + height * (1 - scale) / 2)

        return (pixel_x, pixel_y)

    def _is_point_in_frame(
        self, pixel_pos: Tuple[int, int], frame_shape: Tuple[int, ...]
    ) -> bool:
        """Check if pixel position is within frame bounds.

        Args:
            pixel_pos: Pixel coordinates (x, y)
            frame_shape: Frame shape (height, width, channels)

        Returns:
            True if point is in frame, False otherwise
        """
        height, width = frame_shape[:2]
        x, y = pixel_pos

        return 0 <= x < width and 0 <= y < height
