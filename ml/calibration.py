"""Camera calibration module for multi-view 3D reconstruction.

Provides datastructures and utilities for loading and validating camera calibration
parameters including intrinsic and extrinsic matrices.

Coordinate System:
    - Camera coordinates: X-right, Y-down, Z-forward (OpenCV convention)
    - Intrinsic matrix: 3x3 matrix with [fx, 0, cx], [0, fy, cy], [0, 0, 1]
    - Extrinsic matrix: 4x4 transformation from world to camera coordinates
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import json
import numpy as np
import yaml


@dataclass
class CameraCalibration:
    """Camera calibration parameters for 3D reconstruction.
    
    Attributes:
        camera_id: Unique identifier for the camera (e.g., "cam_0", "main_camera")
        intrinsic: 3x3 intrinsic matrix containing focal lengths and principal point
                   [[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]]
        extrinsic: 4x4 extrinsic transformation matrix from world to camera coords
                   [[r11, r12, r13, tx],
                    [r21, r22, r23, ty],
                    [r31, r32, r33, tz],
                    [0, 0, 0, 1]]
    """
    camera_id: str
    intrinsic: np.ndarray  # 3x3
    extrinsic: np.ndarray  # 4x4
    
    def __post_init__(self) -> None:
        """Validate calibration parameters after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate matrix shapes and values.
        
        Raises:
            ValueError: If matrices have invalid shapes or values
        """
        if not isinstance(self.camera_id, str) or len(self.camera_id) == 0:
            raise ValueError("camera_id must be a non-empty string")
        
        # Validate intrinsic matrix
        if self.intrinsic.shape != (3, 3):
            raise ValueError(f"Intrinsic matrix must be 3x3, got {self.intrinsic.shape}")
        
        if not np.allclose(self.intrinsic[2, :], [0, 0, 1]):
            raise ValueError("Intrinsic matrix bottom row must be [0, 0, 1]")
        
        fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
        if fx <= 0 or fy <= 0:
            raise ValueError(f"Focal lengths must be positive, got fx={fx}, fy={fy}")
        
        # Validate extrinsic matrix
        if self.extrinsic.shape != (4, 4):
            raise ValueError(f"Extrinsic matrix must be 4x4, got {self.extrinsic.shape}")
        
        if not np.allclose(self.extrinsic[3, :], [0, 0, 0, 1]):
            raise ValueError("Extrinsic matrix bottom row must be [0, 0, 0, 1]")
        
        # Validate rotation part (top-left 3x3) is orthogonal
        rotation = self.extrinsic[:3, :3]
        identity_check = rotation @ rotation.T
        if not np.allclose(identity_check, np.eye(3), atol=1e-3):
            raise ValueError("Extrinsic rotation matrix must be orthogonal")
    
    def get_projection_matrix(self) -> np.ndarray:
        """Compute 3x4 projection matrix P = K[R|t].
        
        Returns:
            3x4 projection matrix mapping 3D world points to 2D image points
        """
        # Extract R|t (3x4) from extrinsic
        rt_matrix = self.extrinsic[:3, :]
        return self.intrinsic @ rt_matrix


def load_calibration_from_file(path: Path | str) -> Dict[str, CameraCalibration]:
    """Load camera calibrations from JSON or YAML file.
    
    Expected file format:
    {
        "cameras": {
            "cam_0": {
                "intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                "extrinsic": [[r11, r12, r13, tx], ...]
            },
            ...
        }
    }
    
    Args:
        path: Path to calibration file (JSON or YAML)
    
    Returns:
        Dictionary mapping camera_id to CameraCalibration objects
    
    Raises:
        FileNotFoundError: If calibration file doesn't exist
        ValueError: If file format is invalid
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    
    # Load file based on extension
    if path.suffix.lower() == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
    elif path.suffix.lower() in ['.yaml', '.yml']:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .json or .yaml")
    
    if 'cameras' not in data:
        raise ValueError("Calibration file must contain 'cameras' key")
    
    calibrations = {}
    for camera_id, cam_data in data['cameras'].items():
        if 'intrinsic' not in cam_data:
            raise ValueError(f"Camera {camera_id} missing 'intrinsic' matrix")
        if 'extrinsic' not in cam_data:
            raise ValueError(f"Camera {camera_id} missing 'extrinsic' matrix")
        
        intrinsic = np.array(cam_data['intrinsic'], dtype=np.float64)
        extrinsic = np.array(cam_data['extrinsic'], dtype=np.float64)
        
        calibrations[camera_id] = CameraCalibration(
            camera_id=camera_id,
            intrinsic=intrinsic,
            extrinsic=extrinsic
        )
    
    return calibrations


def validate_calibration_set(calibrations: Dict[str, CameraCalibration]) -> None:
    """Validate a set of camera calibrations.
    
    Args:
        calibrations: Dictionary of camera calibrations
    
    Raises:
        ValueError: If calibration set is invalid
    """
    if len(calibrations) < 2:
        raise ValueError(f"Need at least 2 cameras for 3D reconstruction, got {len(calibrations)}")
    
    # Check for duplicate camera IDs (redundant but explicit)
    camera_ids = list(calibrations.keys())
    if len(camera_ids) != len(set(camera_ids)):
        raise ValueError("Duplicate camera IDs detected")
